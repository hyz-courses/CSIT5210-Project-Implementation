# CSIT5210 Project Implementation (Group 1)

## Plan

### Week 1-2: (Oct. 1 - Oct. 4) Preprocess

- [v] Environment Setup (See the other repo)
- [x] Download MovieLens and Amazon Dataset
- [x] Pre-process Dataset with SR protocol
    - Keep each user's interaction sequence (with clipping)
    - Divide train/validation/test sets
    - Map item ID to title
    - Finally construct a history-next item pair set.

---

### Week 3-6: (Oct. 5 - Nov. 1) Model Realization and Training

- [x] **Stage 1:** Collaborative Supervised Fine-Tuning (CSFT)
    - Construct instruction format as inputs.
    - Self-regression loss
    - Fine-tune LLM (LoRA?)
- [x] **Stage 2:** Item-level Embedding Modeling (IEM)
    - Change casual mask to bidirectional attention
    - Masked Next Token Prediction (MNTP)
        - Mask 20% tokens for single item's title
        - Train model to rebuild masked token
    - Item-level Contrasive Learning (IC)
        - For each item, do two dropout on its title, obtaining two views.
        - Use InfoNCE loss to cluster positive smaple pairs and push out negative pairs. 
        - $\tau = 0.2$
- [x] Build embedding extraction methods
    - Input item title to tuned LLM.
    - Average pool for output token's hidden stage, obtaining item embedding.

---

### Week 5-7 (Oct. 26 - Nov. 10) Evaluation & Benchmark

- [x] Realize Transformer-based SASRec
- [x] Configure LLM2Rec embeddings to downstream recommender.
    - Use LLM2Rec's output embedding as SASRec's initial embedding.
    - Add a lightweight adapter.
- [x] 
    - Recall@10/20 & NDCG@10/20
    - Full ranking (全物品排序)
    - Average 3 runs of three random seeds.

---

