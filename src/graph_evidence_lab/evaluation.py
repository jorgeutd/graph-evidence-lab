"""Ranking metrics and query-level uncertainty; no test-driven model selection."""
import math
import random
import statistics


def metrics(ranked_ids, relevant, k=5):
    if k < 1 or not relevant or len(set(ranked_ids)) != len(ranked_ids):
        raise ValueError('Positive k, relevance labels, and unique ranked IDs required')
    chosen = ranked_ids[:k]
    gains = [float(relevant.get(i, 0)) for i in chosen]
    ideal = sorted(relevant.values(), reverse=True)[:k]
    dcg = lambda values: sum((2**g-1)/math.log2(i+2) for i, g in enumerate(values))
    return {'recall_at_k': sum(i in relevant for i in chosen)/len(relevant),
            'ndcg_at_k': dcg(gains)/dcg(ideal),
            'mrr_at_k': next((1/(i+1) for i, n in enumerate(chosen) if n in relevant), 0.0)}


def summarize(rows, seed=1729):
    rng = random.Random(seed); result = {}
    for key in ('recall_at_k', 'ndcg_at_k', 'mrr_at_k'):
        values = [r[key] for r in rows]
        means = sorted(statistics.mean(rng.choices(values, k=len(values))) for _ in range(1000))
        result[key] = {'mean': statistics.mean(values), 'bootstrap_95': [means[24], means[974]], 'queries': len(values)}
    return result
