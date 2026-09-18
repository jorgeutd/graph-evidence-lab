# Graph learning and LLM evidence

Reviewed 18 September 2026. This project explores a practical question from graph retrieval research; it does not reproduce the cited systems or claim their results.

- [Message Passing Neural Networks, 2017](https://arxiv.org/abs/1704.01212): a shared formulation for learned messages, aggregation and updates.
- [GraphSAGE, 2017](https://arxiv.org/abs/1706.02216): inductive aggregation and neighborhood sampling.
- [GATv2, 2021/ICLR 2022](https://arxiv.org/abs/2105.14491): query-dependent neighbor ranking and limits of original GAT attention.
- [G-Retriever, NeurIPS 2024](https://arxiv.org/abs/2402.07630): textual graph QA with subgraph retrieval and graph-conditioned language generation. Its [public implementation](https://github.com/XiaoxinHe/G-Retriever) is a useful starting point for a full model reproduction. This project instead ranks evidence and exports readable context; it has no PCST solver or graph soft-prompt training.
- [NGM-RAG, July 2026 preprint](https://arxiv.org/abs/2607.11159): combines neural graph matching and retrieval signals. Author-reported results are scoped to its evaluated datasets; no public implementation was verified in this review.
- [Chimaera, September 2026](https://arxiv.org/abs/2609.08709): combines multiple GNN experts over language-model representations with learned selection. The full paper is accessible. Its linked code repository returned 404 during this review; reproducibility remains unverified.
- [Knowledge-Graph Based Augmentation versus RAG for Cultural QA, September 2026 preprint](https://arxiv.org/abs/2609.18317): a useful counterweight to graph hype. Graph context can be compact, but conventional RAG remains competitive; benchmark adaptation and a narrow multiple-choice setting limit general conclusions.
- [ALIGNN 2.0, September 2026 preprint](https://arxiv.org/abs/2609.19487): an application-specific direction for materials, incorporating atom/bond and line-graph structure. It is not evidence that the same architecture is best for technical retrieval.

## Next experiments

Replace hashed features with a pinned pretrained encoder, preserving the lexical baseline. Learn on a larger independently judged corpus. Compare relation-aware sampling and full-graph inference under the same budget. Test corrupted and missing links, new documents and semantically misleading edges. Add an explicit language-model adapter only after establishing retrieval value; then evaluate answer grounding and unsupported claims with held-out evidence.

The 3D portfolio teaching model uses fixed small matrices so visitors can inspect every operation. The repository replay comes from a separately trained 32-channel model. The visual analogy does not turn fixed teaching weights into measured model behavior.
