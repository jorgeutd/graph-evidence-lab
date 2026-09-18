"""Deterministic text features, a lexical baseline and graph diffusion."""
from collections import Counter
import hashlib
import math
import re
import torch

STOP = frozenset('a an the is are to of in for and or with how what which does can do from on by be it as this'.split())


def tokens(text):
    return [t for t in re.findall(r'[a-z0-9]+', text.lower()) if t not in STOP]


def encode(text, dimensions=128):
    # No fitted vocabulary or future-corpus statistics. Hash collisions remain a
    # limitation; production experiments can replace this with a pinned encoder.
    out = torch.zeros(dimensions)
    for term, count in Counter(tokens(text)).items():
        index = int.from_bytes(hashlib.sha256(term.encode()).digest()[:8], 'little') % dimensions
        out[index] += 1 + math.log(count)
    return out / out.norm().clamp_min(1e-12)


def bm25(query, documents, k1=1.2, b=0.75):
    bags = [Counter(tokens(d)) for d in documents]
    lengths = [sum(d.values()) for d in bags]
    avg = sum(lengths) / max(len(lengths), 1) or 1
    terms = set(tokens(query)); scores = []
    for bag, length in zip(bags, lengths):
        score = 0.0
        for term in terms:
            df = sum(term in d for d in bags)
            idf = math.log(1 + (len(bags) - df + .5) / (df + .5))
            tf = bag[term]
            score += idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * length / avg))
        scores.append(score)
    return torch.tensor(scores)


def edge_tensors(graph, relation_names):
    index = {n.id: i for i, n in enumerate(graph.nodes)}
    relations = {name: i for i, name in enumerate(relation_names)}
    edges, kinds, weights = [], [], []
    for e in graph.edges:
        for source, target, relation in [(e.source, e.target, e.relation), (e.target, e.source, 'reverse:'+e.relation)]:
            if relation not in relations:
                raise ValueError('Unknown relation: '+relation+'. Retrain with the new schema.')
            edges.append([index[source], index[target]])
            kinds.append(relations[relation]); weights.append(e.confidence)
    return (torch.tensor(edges, dtype=torch.long).reshape(-1, 2).T.contiguous(),
            torch.tensor(kinds, dtype=torch.long), torch.tensor(weights))


def query_features(graph, query, dimensions=128):
    texts = [n.title+' '+n.text for n in graph.nodes]
    lexical = bm25(query, texts)
    doc = torch.stack([encode(t, dimensions) for t in texts])
    q = encode(query, dimensions).expand_as(doc)
    x = torch.cat([doc, q, doc*q, (doc*q).sum(-1, keepdim=True),
                   (lexical / lexical.max().clamp_min(1e-12)).unsqueeze(-1)], dim=-1)
    return x, lexical


def diffuse(scores, edge_index, edge_weights, alpha=.2, steps=30):
    restart = scores.clamp_min(0)
    restart = restart / restart.sum().clamp_min(1e-12)
    if restart.sum() == 0:
        return restart
    source, target = edge_index
    degree = torch.zeros_like(scores).index_add_(0, source, edge_weights)
    rank = restart.clone()
    for _ in range(steps):
        spread = torch.zeros_like(rank)
        spread.index_add_(0, target, rank[source] * edge_weights / degree[source].clamp_min(1e-12))
        dangling = rank[degree == 0].sum()
        rank = alpha*restart + (1-alpha)*(spread+dangling*restart)
    return rank
