# Evaluate the retrieval decision

## Locked protocol

The fixture was authored before the first benchmark run. It has 24 training queries in 12 paraphrase groups, 8 validation queries and 12 held-out test queries in distinct groups. A group split reduces direct paraphrase leakage; it does not remove overlap in terminology, document targets or author assumptions. This is known-corpus, held-out-intent evaluation, not temporal forecasting or unseen-graph evaluation.

Every neural model uses the same query/document features, a 32-channel representation, two update layers, AdamW (learning rate 0.003, weight decay 0.01), gradient clipping at 1.0, at most 100 epochs and patience 15. Soft-target cross entropy uses normalized relevance grades. The checkpoint with the highest validation NDCG@5 is restored. Seeds 0, 1 and 2 are fixed in advance.

Compare BM25, lexical-seeded graph diffusion, the retrained feature-only neural model and the relational GNN. Diffusion uses restart probability 0.2 and 30 iterations. Neural models do not get extra document fields or test labels. The separate seed-0 edge-removal intervention holds trained weights fixed; it is sensitivity analysis, not a causal attribution or a tuned competitor.

## Read the report

`benchmark.json` includes corpus and judgment fingerprints, environment, protocol, per-query metrics, selected epochs and complete validation history. Recall@5 counts retrieved relevant documents, MRR@5 uses the first relevant rank, and NDCG@5 uses gain `2^grade - 1` with logarithmic discount. The bootstrap draws 1,000 resamples of test query rows with seed 1729. Its percentile interval is conditional on one trained model and excludes seed variability.

`replay.json` includes seed-0 BM25 and GNN rankings, eligible node IDs, per-layer states and confidence-weighted messages. Timings are diagnostic CPU wall times, including Python work, without a controlled latency benchmark. Do not market these as serving performance.

## What would support a stronger claim

Use a versioned external corpus and independently judged questions; hold out organizations, time periods or graphs as the intended deployment requires. Add a tuned dense retriever and cross-encoder. Measure candidate coverage, retrieval accuracy, grounded answer quality, abstention, citation entailment, cost and latency separately. Use paired query comparisons, multiple seeds, failure slices and a predeclared decision rule. Report graph construction cost and errors.

A GNN is justified only when it improves a task enough to offset its operational cost. If simpler methods win, keep them.

## First measured result · 18 September 2026

The [reference run](https://github.com/jorgeutd/graph-evidence-lab/actions/runs/35352452200) passed **29 tests** and trained three fixed seeds. Test NDCG@5 was **0.811 for BM25**, **0.815 for graph diffusion**, **0.729 for the feature-only neural baseline** (three-seed mean), and **0.776 for the relational GNN** (three-seed mean). GNN seed results were 0.801, 0.776 and 0.749. These are point estimates on twelve authored questions; the intervals overlap, and no superiority claim is warranted.

The current GNN does not beat the simpler baselines on mean NDCG. With seed-0 weights held fixed, removing all graph edges reduces NDCG from 0.801 to 0.701. This demonstrates sensitivity to graph computation, not that the trained graph approach is the best retrieval system.

Inspect [the complete report](../reports/benchmark.json), [all twelve recorded queries and messages](../reports/replay.json), and [provenance](../reports/provenance.json). The fixture and hyperparameters were not changed to improve these test results. A next model revision needs a fresh held-out evaluation set if these outcomes guide its development.

Try the recorded checkpoint without retraining:

```sh
graph-evidence query 'How should an LLM handle instructions inside retrieved documents?' --checkpoint artifacts/gnn-seed-0.json --context
```
