# Showing new listings for Thursday, 9 July 2026
Auto update papers at about 2:30am UTC (10:30am Beijing time) every weekday.


阅读 `Usage.md`了解如何使用此repo实现个性化的Arxiv论文推送

See `Usage.md` for instructions on how to personalize the repo. 


Keyword list: ['text-to-speech', 'text to speech', 'tts', 'LLM-based', 'speech', 'voice']


Excluded: []


### Today: 3papers 
#### Compress the Cache, Not the Speech Embedding: KV Compression for Efficient Speech LLMs
 - **Authors:** Ke-Han Lu, Keqi Deng, Ruchao Fan, Rui Zhao, Jinyu Li
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2607.06827

 - **Pdf link:** https://arxiv.org/pdf/2607.06827

 - **Abstract**
 Speech large language models (Speech LLMs) typically encode speech into sequences far longer than text, creating a major efficiency bottleneck during autoregressive decoding. A common remedy is to compress the speech sequence at the adapter level to remove temporal redundancy before it enters the LLM; however, such early downsampling risks discarding fine-grained information that cannot be recovered. We propose SpeechKV, which applies a learned pooling to the KV cache of speech tokens inside the LLM. This design allows the LLM to fuse speech and text internally while directly accelerating decoding. Trained on 71K hours of speech data, SpeechKV compresses the speech to approximately text-level granularity yet maintains performance on par with or even slightly better than the uncompressed baseline, with relative gains of 6.6% on out-of-domain entity recognition and 2.3% on OpenASR, while delivering at least 1.49 times decoding speedup that scales with audio length.
#### UBG-Net: An Uncertainty-aware Bayesian Gating Network for Robust Audio-Visual Speech Recognition
 - **Authors:** Jinjie Fu, Hang Chen, Wu Guo, Zhijun Zhang, Kuiliang Li, Peng Gao
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2607.06892

 - **Pdf link:** https://arxiv.org/pdf/2607.06892

 - **Abstract**
 Audio-Visual speech recognition systems often degrade in real-world scenarios due to signal corruption and distribution shifts. To address this, we propose a unified uncertainty-modeling framework, namely the uncertainty-aware Bayesian gating network (UBG-Net). UBG-Net features a Modality Uncertainty-aware Bayesian Fusion (MUBF) mechanism that injects signal-level aleatoric uncertainty into a Bayesian network to model epistemic uncertainty, thereby ensuring robust fusion of pre-trained backbone features. For inference, we introduce Distribution Uncertainty-aware Hierarchical Voting (DUHV) to select transcripts from Monte Carlo samples, prioritizing frequency and using inference scores in case of a tie. Experiments on the AVCocktail and LRS2 datasets demonstrate the overall superiority of UBG-Net compared to SOTA baselines. Ablation studies confirm that MUBF and DUHV effectively filter noise, enhancing fusion and decoding robustness.
#### Text-Independent Speaker Verification Using Discrete Audio Tokens
 - **Authors:** Zheng Liang, Junjie Li, Kong Aik Lee
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2607.07579

 - **Pdf link:** https://arxiv.org/pdf/2607.07579

 - **Abstract**
 Neural audio codecs (NACs) enable efficient audio compression and have achieved success in downstream tasks such as speech synthesis. However, their discrete representations consistently underperform traditional spectral features in automatic speaker verification (ASV). We empirically demonstrate that speaker cues are implicitly preserved in discrete tokens but remain underutilized by conventional ASV training paradigms. To address this, we propose a Cross-Feature Knowledge Distillation (CFKD) framework. By guiding the codec-based student to mimic the embedding space of a strong Fbank-based teacher, CFKD provides structured supervision for effective utilization of speaker information in tokens. Experiments on the VoxCeleb benchmarks show that CFKD substantially improves the ASV performance of codec-based systems, allowing them to approach the accuracy of Fbank-based teacher models and highlighting the potential of discrete audio tokens for diverse speech tasks.


by Zyzzyva0381 (Windy). 


2026-07-09
