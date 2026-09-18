"""Train on train queries, stop on validation NDCG, evaluate test once."""
from pathlib import Path
import copy
import hashlib
import json
import random
import platform
import statistics
import torch
from .evaluation import metrics, summarize
from .features import edge_tensors
from .model import EvidenceRanker, save_model
from .retrieval import retrieve, tensors, relations


def fit(graph, queries, seed=0, graph_enabled=True, epochs=100, patience=15):
    torch.manual_seed(seed); random.seed(seed); torch.set_num_threads(1)
    model = EvidenceRanker(relations(graph), graph_enabled=graph_enabled)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.003, weight_decay=.01)
    train = [q for q in queries if q['split']=='train']
    valid = [q for q in queries if q['split']=='validation']
    prepared = []
    for q in train:
        g = graph.snapshot(q['as_of']); x, lexical, edge = tensors(g, q['text'], model)
        grades = torch.tensor([q['relevant'].get(n.id, 0) for n in g.nodes], dtype=torch.float)
        prepared.append((x, edge, grades/grades.sum()))
    best, best_state, stale, best_epoch = -1, None, 0, 0
    history = []
    for epoch in range(epochs):
        model.train(); order = list(range(len(prepared))); random.shuffle(order); loss_total = 0
        for index in order:
            x, edge, target = prepared[index]
            optimizer.zero_grad(set_to_none=True)
            scores = model(x, *edge)
            loss = -(target*torch.log_softmax(scores, dim=0)).sum()
            if not torch.isfinite(loss): raise RuntimeError('Non-finite training loss')
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
            loss_total += loss.item()
        model.eval(); values = []
        for q in valid:
            r = retrieve(graph, q['text'], q['as_of'], 'gnn' if graph_enabled else 'mlp', model)
            values.append(metrics([n['id'] for n in r['results']], q['relevant'])['ndcg_at_k'])
        score = statistics.mean(values)
        history.append({'epoch': epoch+1, 'loss': loss_total/len(train), 'validation_ndcg_at_5': score})
        if score > best + 1e-8:
            best, best_state, stale, best_epoch = score, copy.deepcopy(model.state_dict()), 0, epoch+1
        else: stale += 1
        if stale >= patience: break
    model.load_state_dict(best_state); model.eval()
    return model, {'seed': seed, 'best_epoch': best_epoch, 'validation_ndcg_at_5': best,
                   'parameters': sum(p.numel() for p in model.parameters()), 'history': history}


def benchmark(graph, queries, output, seeds=(0, 1, 2), epochs=100):
    directory = Path(output); directory.mkdir(parents=True, exist_ok=True)
    report = {'format': 'graph-evidence-benchmark-v1', 'dataset_scope': 'Original authored engineering fixture; a pipeline diagnostic, not an external benchmark.',
              'protocol': 'Fixed corpus, held-out query intent groups. Test judgments never construct graph edges or select checkpoints. Unjudged documents are treated as nonrelevant only in this fully authored fixture.',
              'graph_fingerprint': graph.fingerprint,
              'query_fingerprint': hashlib.sha256(json.dumps(queries, sort_keys=True).encode()).hexdigest(),
              'environment': {'python': platform.python_version(), 'torch': torch.__version__, 'platform': platform.platform(), 'threads': 1},
              'counts': {s: sum(q['split']==s for q in queries) for s in ['train','validation','test']}, 'k': 5, 'runs': []}
    test = [q for q in queries if q['split']=='test']
    for method in ['bm25', 'diffusion', 'mlp', 'gnn']:
        for seed in seeds if method in {'mlp','gnn'} else [0]:
            model, training = (fit(graph, queries, seed, method=='gnn', epochs) if method in {'mlp','gnn'} else (None, None))
            rows = []
            for q in test:
                r = retrieve(graph, q['text'], q['as_of'], method, model)
                rows.append({'id': q['id'], **metrics([n['id'] for n in r['results']], q['relevant'])})
            item = {'method': method, 'seed': seed, 'training': training, 'test': summarize(rows), 'per_query': rows}
            report['runs'].append(item)
            if model:
                save_model(model, directory/f'{method}-seed-{seed}.json', graph.fingerprint)
            print(method, seed, json.dumps(item['test']), flush=True)
    # Mechanistic ablation: hold seed-0 GNN weights fixed; do not select by test.
    from .model import load_model
    from .data import Graph
    model = load_model(directory/'gnn-seed-0.json', graph.fingerprint)
    graph_without = Graph({'nodes': [vars(n) for n in graph.nodes], 'edges': []})
    rows = []
    for q in test:
        r = retrieve(graph_without, q['text'], q['as_of'], 'gnn', model)
        rows.append({'id':q['id'], **metrics([n['id'] for n in r['results']], q['relevant'])})
    report['fixed_weight_no_edges'] = {'seed':0, 'test':summarize(rows), 'note':'Sensitivity intervention at inference, not a separately tuned model or causal explanation.'}
    report['limitations'] = ['Small authored corpus and test set', 'One fixed known graph; not unseen-graph generalization',
                             'Hash text features, not a pretrained language encoder', 'Intervals resample test queries for each fixed model, not training seeds',
                             'Full-graph computation is a prototype for bounded graphs', 'No LLM answer-quality benchmark performed']
    (directory/'benchmark.json').write_text(json.dumps(report, indent=2), encoding='utf8')
    # A replay keeps measured results separate from illustrative browser math.
    replay = {'scope': report['dataset_scope'], 'graph': graph.to_dict(), 'seed':0, 'queries':[]}
    for q in test:
        replay['queries'].append({'id':q['id'], 'text':q['text'], 'relevant':q['relevant'],
                                  'lexical':retrieve(graph,q['text'],q['as_of'],'bm25'),
                                  'gnn':retrieve(graph,q['text'],q['as_of'],'gnn',model,trace_vectors=True)})
    (directory/'replay.json').write_text(json.dumps(replay), encoding='utf8')
    return report
