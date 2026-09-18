"""Query-conditioned relational message passing with explicit contributions."""
import torch
from torch import nn


class RelationalLayer(nn.Module):
    def __init__(self, hidden, relations):
        super().__init__()
        self.self_projection = nn.Linear(hidden, hidden)
        self.message_projection = nn.Linear(hidden, hidden, bias=False)
        self.relation = nn.Embedding(relations, hidden)
        self.norm = nn.LayerNorm(hidden)

    def forward(self, h, edge_index, edge_kind, confidence):
        source, target = edge_index
        # The query is already encoded in h. A typed gate controls message
        # channels; confidence scales both numerator and neighborhood normalizer.
        gate = self.relation(edge_kind).sigmoid()
        messages = self.message_projection(h[source]) * gate * confidence[:, None]
        total = torch.zeros_like(h).index_add_(0, target, messages)
        denominator = torch.zeros(h.shape[0], device=h.device).index_add_(0, target, confidence)
        aggregate = total / denominator.clamp_min(1e-12)[:, None]
        pre = self.self_projection(h) + aggregate
        result = self.norm(h + torch.nn.functional.gelu(pre))
        return result, {'messages': messages, 'aggregate': aggregate, 'denominator': denominator}


class EvidenceRanker(nn.Module):
    def __init__(self, relation_names, dimensions=128, hidden=32, depth=2, graph_enabled=True):
        super().__init__()
        self.config = dict(relation_names=relation_names, dimensions=dimensions, hidden=hidden,
                           depth=depth, graph_enabled=graph_enabled)
        self.input = nn.Linear(dimensions*3+2, hidden)
        self.layers = nn.ModuleList([RelationalLayer(hidden, len(relation_names)) for _ in range(depth)])
        self.output = nn.Sequential(nn.Linear(hidden, hidden//2), nn.GELU(), nn.Linear(hidden//2, 1))

    def forward(self, x, edges, kinds, confidence, trace=False):
        if not self.config['graph_enabled']:
            edges, kinds, confidence = edges[:, :0], kinds[:0], confidence[:0]
        h = torch.nn.functional.gelu(self.input(x)); states = [h]
        contributions = []
        for layer in self.layers:
            h, detail = layer(h, edges, kinds, confidence)
            states.append(h); contributions.append(detail)
        scores = self.output(h).squeeze(-1)
        return (scores, states, contributions) if trace else scores


def save_model(model, path, graph_fingerprint):
    # JSON avoids executable pickle deserialization for user-supplied checkpoints.
    import json
    from pathlib import Path
    Path(path).write_text(json.dumps({'format': 'graph-evidence-v1', 'config': model.config,
                                    'graph_fingerprint': graph_fingerprint,
                                    'state': {k: v.detach().cpu().tolist() for k, v in model.state_dict().items()}}), encoding='utf8')


def load_model(path, expected_graph=None):
    import json
    from pathlib import Path
    file = Path(path)
    if file.stat().st_size > 20_000_000:
        raise ValueError('Checkpoint exceeds 20 MB prototype limit')
    data = json.loads(file.read_text(encoding='utf8'))
    if data['format'] != 'graph-evidence-v1':
        raise ValueError('Unsupported model format')
    if expected_graph and data['graph_fingerprint'] != expected_graph:
        raise ValueError('Checkpoint and graph fingerprint differ; explicitly train for this graph')
    c = data['config']
    if not 1 <= c['depth'] <= 8 or not 4 <= c['hidden'] <= 256 or not 8 <= c['dimensions'] <= 1024 or len(c['relation_names']) > 128:
        raise ValueError('Invalid checkpoint architecture')
    model = EvidenceRanker(**c)
    weights = {k: torch.tensor(v) for k, v in data['state'].items()}
    if any(not torch.isfinite(v).all() for v in weights.values()):
        raise ValueError('Non-finite checkpoint weights')
    model.load_state_dict(weights); model.eval()
    return model
