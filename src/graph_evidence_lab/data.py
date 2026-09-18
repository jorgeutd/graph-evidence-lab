"""Strict, time-aware graph inputs. Relevance labels never construct edges."""
from __future__ import annotations
from dataclasses import dataclass
from datetime import date
from pathlib import Path
import hashlib
import json
import math


def day(value: str) -> date:
    return date.fromisoformat(value)


@dataclass(frozen=True)
class Node:
    id: str
    title: str
    text: str
    kind: str
    source: str
    available_at: str


@dataclass(frozen=True)
class Edge:
    source: str
    target: str
    relation: str
    available_at: str
    confidence: float = 1.0


class Graph:
    def __init__(self, payload: dict):
        self.nodes = [Node(**n) for n in payload['nodes']]
        self.edges = [Edge(**e) for e in payload['edges']]
        if not self.nodes or len(self.nodes) > 5000 or len(self.edges) > 50000:
            raise ValueError('Use 1–5,000 nodes and at most 50,000 edges for this full-graph prototype.')
        ids = {n.id for n in self.nodes}
        if len(ids) != len(self.nodes):
            raise ValueError('Duplicate node ID')
        for n in self.nodes:
            day(n.available_at)
            if not n.id or not n.text.strip() or len(n.text) > 20000:
                raise ValueError('Each node needs an ID and 1–20,000 characters of text.')
        seen = set()
        for e in self.edges:
            day(e.available_at)
            key = (e.source, e.target, e.relation)
            if e.source not in ids or e.target not in ids or not e.relation:
                raise ValueError('Edge endpoints and relation must exist')
            if key in seen:
                raise ValueError('Duplicate edge')
            if not math.isfinite(e.confidence) or not 0 <= e.confidence <= 1:
                raise ValueError('Edge confidence must be finite and in [0,1]')
            seen.add(key)
        # Canonical ordering makes tensor construction and fingerprints stable.
        self.nodes.sort(key=lambda n: n.id)
        self.edges.sort(key=lambda e: (e.source, e.target, e.relation))
        self.fingerprint = hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True).encode()).hexdigest()

    @classmethod
    def load(cls, path: str | Path) -> Graph:
        return cls(json.loads(Path(path).read_text(encoding='utf8')))

    def to_dict(self) -> dict:
        # Never expose a dataclass __dict__: callers could mutate a frozen
        # snapshot and silently invalidate its fingerprint.
        return {'nodes': [dict(vars(n)) for n in self.nodes], 'edges': [dict(vars(e)) for e in self.edges]}

    def snapshot(self, as_of: str) -> Graph:
        cutoff = day(as_of)
        nodes = [n for n in self.nodes if day(n.available_at) <= cutoff]
        ids = {n.id for n in nodes}
        edges = [e for e in self.edges if day(e.available_at) <= cutoff and e.source in ids and e.target in ids]
        if not nodes:
            raise ValueError('No graph evidence was available at this date')
        return Graph({'nodes': [vars(n) for n in nodes], 'edges': [vars(e) for e in edges]})


def load_queries(path: str | Path, graph: Graph) -> list[dict]:
    queries = json.loads(Path(path).read_text(encoding='utf8'))
    seen_ids, group_split = set(), {}
    for q in queries:
        if q['id'] in seen_ids or q['split'] not in {'train', 'validation', 'test'}:
            raise ValueError('Duplicate query ID or invalid split')
        if q['group'] in group_split and group_split[q['group']] != q['split']:
            raise ValueError('A query intent group crosses splits')
        if not q['text'].strip() or not q['relevant']:
            raise ValueError('Each evaluation query needs text and relevance judgments')
        available = {n.id for n in graph.snapshot(q['as_of']).nodes}
        if any(k not in available or not math.isfinite(v) or v <= 0 for k, v in q['relevant'].items()):
            raise ValueError('Relevance refers to missing/future evidence or invalid grades')
        seen_ids.add(q['id']); group_split[q['group']] = q['split']
    if set(q['split'] for q in queries) != {'train', 'validation', 'test'}:
        raise ValueError('Train, validation and test queries are required')
    return queries
