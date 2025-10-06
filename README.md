# CSIT5210 Project Implementation (Group 1)

## Plan

### Week 1-2: (Oct. 1 - Oct. 4) Preprocess

- [x] Environment Setup (See the other repo)
- [x] Download Amazon Dataset
- [x] Pre-process Dataset with SR protocol
    - Keep each user's interaction sequence (with clipping)
    - Divide train/validation/test sets
    - Map item ID to title
    - Finally construct a history-next item pair set.

---

### Week 3-6: (Oct. 5 - Nov. 1) Model Realization and Training

- [ ] **Stage 1:** Collaborative Supervised Fine-Tuning (CSFT)
    - Construct instruction format as inputs.
    - Self-regression loss
    - Fine-tune LLM (LoRA?)
- [ ] **Stage 2:** Item-level Embedding Modeling (IEM)
    - Change casual mask to bidirectional attention
    - Masked Next Token Prediction (MNTP)
        - Mask 20% tokens for single item's title
        - Train model to rebuild masked token
    - Item-level Contrasive Learning (IC)
        - For each item, do two dropout on its title, obtaining two views.
        - Use InfoNCE loss to cluster positive smaple pairs and push out negative pairs. 
        - $\tau = 0.2$
- [ ] Build embedding extraction methods
    - Input item title to tuned LLM.
    - Average pool for output token's hidden stage, obtaining item embedding.

---

### Week 5-7 (Oct. 26 - Nov. 10) Evaluation & Benchmark

- [ ] Realize Transformer-based SASRec
- [ ] Configure LLM2Rec embeddings to downstream recommender.
    - Use LLM2Rec's output embedding as SASRec's initial embedding.
    - Add a lightweight adapter.
- [ ] 
    - Recall@10/20 & NDCG@10/20
    - Full ranking (全物品排序)
    - Average 3 runs of three random seeds.

---

## Citations

LLM2Rec: LLM2Rec: Large Language Models Are Powerful Embedding Models for Sequential Recommendation

```bibtex
@inproceedings{he2025llm2rec,
  title={LLM2Rec: Large Language Models Are Powerful Embedding Models for Sequential Recommendation},
  author={He, Yingzhi and Liu, Xiaohao and Zhang, An and Ma, Yunshan and Chua, Tat-Seng},
  booktitle={Proceedings of the 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining V. 2},
  pages={896--907},
  year={2025}
}
```

Amazon Reviews 2023

```bibtex
@article{hou2024bridging,
  title={Bridging Language and Items for Retrieval and Recommendation},
  author={Hou, Yupeng and Li, Jiacheng and He, Zhankui and Yan, An and Chen, Xiusi and McAuley, Julian},
  journal={arXiv preprint arXiv:2403.03952},
  year={2024}
}
```

