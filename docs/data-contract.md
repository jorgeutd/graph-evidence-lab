# Graph and evaluation contract

A graph is JSON with `nodes` and `edges` lists. Every node needs `id`, `title`, `text`, `kind`, `source` and ISO date `available_at`. Every edge needs `source` and `target` node IDs, a nonempty `relation`, an availability date, and a finite `confidence` between 0 and 1. Repeated endpoint/relation triples are rejected. Availability means the date evidence could be used, not necessarily the date an event occurred.

```json
{
  "nodes": [
    {"id":"cache","title":"Cache budget","text":"Retained attention state consumes memory.","kind":"inference","source":"example://notes/cache","available_at":"2026-08-01"},
    {"id":"batch","title":"Batch scheduling","text":"Concurrent sequences share a memory budget.","kind":"serving","source":"example://notes/batch","available_at":"2026-08-01"}
  ],
  "edges": [
    {"source":"batch","target":"cache","relation":"requires","confidence":1.0,"available_at":"2026-08-02"}
  ]
}
```

A query is an object with `id`, `group`, `split` (`train`, `validation` or `test`), `text`, `as_of` and `relevant`: a map of node IDs to positive finite relevance grades. All three splits are required. Keep related paraphrases in the same group and split. Judged documents must exist at the query cutoff.

The authored fixture uses grades 1 (supporting), 2 (directly useful) and 3 (central). All unlisted nodes are treated as nonrelevant for this deliberately small fixture. Real incomplete judgments need pooling, adjudication and sensitivity analysis; do not silently apply that assumption to a large unjudged corpus.

Snapshots filter nodes and edges, including endpoint availability, before reverse edges and text features are built. BM25 document frequencies use only the eligible snapshot. A fixed graph vocabulary of relation names is architecture metadata; no label, target grade or future document text enters the features.

Source strings are provenance identifiers. The service never downloads them. Examples use `example://` to make fictional provenance explicit. Supply stable, authorized source identifiers for your own corpus; do not publish private documents or credentials.
