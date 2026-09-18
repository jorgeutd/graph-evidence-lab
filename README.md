# Graph Evidence Lab

**Can relationships improve retrieval enough to justify a graph model?**

A small, inspectable research system by Jorge Grisman: a query-conditioned relational GNN ranks technical evidence, produces a bounded context packet for an LLM, and records the messages behind each score. Compare it with BM25, seeded graph diffusion and a feature-only neural baseline before deciding to use a graph.

This is independently authored public work. The engineering handbook and relevance judgments are fictional teaching fixtures, with no employer documents or production data. It is a working research prototype, not a published research result or a production service.

## Start here

Python 3.11+; the automated reference run uses Python 3.12 on a Linux CPU.

```sh
git clone https://github.com/jorgeutd/graph-evidence-lab.git
cd graph-evidence-lab
python -m venv .venv
# Activate .venv using the command for your shell.
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[dev]'
pytest
graph-evidence benchmark --seeds 0 1 2 --epochs 100
graph-evidence query 'How do graph edges improve evidence retrieval?'
graph-evidence query 'How do I bound retry side effects?' --method bm25 --context
graph-evidence serve
```

Open `http://127.0.0.1:8000/docs` for the local API. Training writes JSON checkpoints, a benchmark report and a per-query replay to `runs/baseline/`. JSON checkpoints contain numerical weights, not executable pickle objects.

## What is implemented

```text
Query + as-of date
       │
       ▼
Eligible graph snapshot ──► BM25 / seeded diffusion
       │
       ▼
Hashed text + query interaction + lexical score
       │
       ▼
2 relational message-passing layers
  typed channel gates · confidence-weighted mean · residual update
       │
       ▼
Node scores ──► ranked documents ──► bounded evidence packet
       │                                  │
       ▼                                  ▼
Message / timing trace              Optional external LLM
                                         (not invoked here)
```

The neural model is **trained**, with validation-based checkpoint selection. It is a custom relational MPNN, not an implementation of GCN, GraphSAGE, GATv2 or G-Retriever. Node features are deterministic hashed text features, not pretrained language embeddings. Ranking scores are not probabilities.

The feature-only baseline disables every edge while retaining the same self/update/readout network. Its nominal parameter count includes unused message parameters; its effective capacity is therefore smaller. The fixed-weight edge-removal intervention is reported separately from this retrained baseline.

## Evaluation that can disagree with the idea

The original fixture has **37 nodes, 48 directed typed edges and 44 queries**: 24 train, 8 validation, 12 test. One future node and its edge are deliberately excluded at the September 2026 cutoff. Reverse edges are created only after time filtering. Query intent groups never cross splits; the corpus is known across splits.

The benchmark reports Recall@5, graded NDCG@5 and MRR@5, per-query outcomes, three training seeds, validation history and query-bootstrap intervals. Test judgments neither construct edges nor select a checkpoint. Twelve authored test questions cannot establish generalization, statistical superiority or production value. There is no LLM answer-quality evaluation.

The first measured run passed 29 tests. BM25 scored 0.811 NDCG@5, graph diffusion 0.815, and the GNN averaged 0.776 across three seeds on twelve authored test questions. The GNN does not beat the simpler baselines on this metric.

CI trains the models and uploads the exact report and replay. See [Actions](https://github.com/jorgeutd/graph-evidence-lab/actions) for measured runs. The checked-in report, replay and seed-0 checkpoint are described in [the evaluation protocol](docs/evaluation.md). A lexical baseline winning is a useful result.

## Use your own graph

See [the schema and temporal contract](docs/data-contract.md). Provide graph JSON and query judgments, then run:

```sh
graph-evidence benchmark --graph my-graph.json --queries my-queries.json --output runs/my-graph
graph-evidence query 'Your question' --graph my-graph.json --checkpoint runs/my-graph/gnn-seed-0.json
```

The checkpoint must match the graph fingerprint. Changing the corpus requires explicit retraining in this prototype; an inductive deployment policy is a separate extension. Full-graph execution is bounded to 5,000 nodes and 50,000 input edges and may still be slow on large text corpora. Use sampled neighborhoods, cached encodings and measured resource budgets before scaling it.

## Connect an LLM without hiding the retrieval

`POST /context` returns evidence with source identifiers, a snapshot fingerprint and a character budget. An application can place that evidence in a data field for its chosen model and validate bracketed source IDs with `citation_check`. This check verifies **membership only**, not factual support. Measure retrieval and answer quality separately. The library does not call a hosted model, execute tools or fetch evidence URLs.

The FastAPI server binds to localhost, has no authentication and is intended for local experiments. A prompt instruction is not a security boundary. See [research and limitations](docs/research.md) for the next experiments worth running.

## Explore and read

- [Interactive 3D graph lab](https://jorgeutd.github.io/labs/graphs/)
- [Research explainer and primary papers](https://jorgeutd.github.io/notes/graph-neural-networks/)
- [Research context](docs/research.md)
- [Evaluation protocol](docs/evaluation.md)
- MIT license. Dependencies retain their respective licenses.
