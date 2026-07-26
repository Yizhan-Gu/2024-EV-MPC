# Advanced forecasting related work and implementation map

Reviewed: 2026-07-26

This is a focused primary-source review, not a systematic literature review.
The 2025--2026 EV papers below are arXiv preprints and should be checked again
for peer-reviewed versions before manuscript submission.

## General time-series models

| Work | Primary idea | Decision for this project |
|---|---|---|
| [DLinear / LTSF-Linear (Zeng et al., 2022/AAAI 2023)](https://arxiv.org/abs/2205.13504) | Very simple linear models can outperform heavier Transformers on standard long-horizon tasks. | Implemented as a required strong simple baseline with moving-average trend/seasonal decomposition. |
| [PatchTST (Nie et al., ICLR 2023)](https://arxiv.org/abs/2211.14730) | Patch time segments and share a channel-independent Transformer encoder. | Next targeted Transformer baseline; useful for longer histories without quadratic attention over every timestamp. |
| [iTransformer (Liu et al., ICLR 2024)](https://arxiv.org/abs/2310.06625) | Treat each variable's full history as one token and attend across variables. | Implemented. This maps cleanly to count, energy, arrival, departure, and dwell variables. |
| [SAMformer (Ilbert et al., ICML 2024)](https://arxiv.org/abs/2402.10198) | Shallow channel-wise Transformer plus RevIN and sharpness-aware optimization addresses small-data instability. | Candidate ablation if iTransformer is unstable or overfits; SAM training cost must be justified. |
| [TimeMixer (Wang et al., ICLR 2024)](https://arxiv.org/abs/2405.14616) | Decomposable multiscale MLP mixing of seasonal and trend patterns. | Next efficient multiscale baseline. |
| [TimeMixer++ (Wang et al., 2024)](https://arxiv.org/abs/2410.16032) | Multi-resolution temporal/frequency pattern machine across several time-series tasks. | Literature context only for now; full implementation is too broad before TimeMixer and simpler baselines are exhausted. |
| [Graph WaveNet (Wu et al., IJCAI 2019)](https://arxiv.org/abs/1906.00121) | Adaptive learned graph plus dilated temporal convolutions. | Motivates a future learned-adjacency ablation. The current compact graph model deliberately uses a transparent training-only graph first. |

## Recent EV charging-demand work

| Work | Relevance | Design implication |
|---|---|---|
| [EV-STLLM (Fan et al., 2025)](https://arxiv.org/abs/2507.09527) | Combines station-network adjacency, multi-frequency/multi-scale features, and a partially frozen graph-attention large model for charging/occupancy forecasting. | Confirms the value of station-level spatio-temporal structure, but its complexity is not justified until small baselines and leakage-safe splits are beaten. |
| [TW-GCN (Tupayachi et al., 2025)](https://arxiv.org/abs/2510.09048) | Combines GCNs with temporal models and multimodal traffic/weather inputs; reports strong 1D-CNN performance. | Supports keeping both a TCN and a graph-temporal model. Exogenous data should be added only when available at both training and forecast time. |
| [Spatio-temporal modelling of EV charging demand (Teng et al., 2026)](https://arxiv.org/abs/2604.19841) | Station-level latent-Gaussian modeling with interpretable spatial/temporal effects and uncertainty on a large Scotland dataset. | Reinforces uncertainty quantification and interpretable station effects as paper-level requirements, not optional extras. |

## Resulting model ladder

The intended comparison should increase complexity only when each preceding
level is established:

1. seasonal naive and ridge;
2. DLinear;
3. LSTM and TCN;
4. PatchTST, iTransformer, and TimeMixer;
5. charger graph-temporal model with no graph, correlation graph, physical
   graph, and learned graph ablations;
6. calibrated probabilistic versions connected to rolling MPC.

A large/foundation model is not automatically a stronger scientific baseline.
For this dataset, the key hypothesis concerns the **choice of prediction
entity**, so the benchmark must prevent architecture complexity from obscuring
the EV-versus-charger comparison.
