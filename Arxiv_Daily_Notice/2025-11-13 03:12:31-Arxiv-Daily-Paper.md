# Showing new listings for Thursday, 13 November 2025
Auto update papers at about 2:30am UTC (10:30am Beijing time) every weekday.


阅读 `Usage.md`了解如何使用此repo实现个性化的Arxiv论文推送

See `Usage.md` for instructions on how to personalize the repo. 


Keyword list: ['text-to-speech', 'text to speech', 'tts', 'LLM-based', 'speech', 'voice']


Excluded: []


### Today: 2papers 
#### ParaS2S: Benchmarking and Aligning Spoken Language Models for Paralinguistic-aware Speech-to-Speech Interaction
 - **Authors:** Shu-wen Yang, Ming Tu, Andy T. Liu, Xinghua Qu, Hung-yi Lee, Lu Lu, Yuxuan Wang, Yonghui Wu
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Signal Processing (eess.SP)
 - **Arxiv link:** https://arxiv.org/abs/2511.08723

 - **Pdf link:** https://arxiv.org/pdf/2511.08723

 - **Abstract**
 Speech-to-Speech (S2S) models have shown promising dialogue capabilities, but their ability to handle paralinguistic cues--such as emotion, tone, and speaker attributes--and to respond appropriately in both content and style remains underexplored. Progress is further hindered by the scarcity of high-quality and expressive demonstrations. To address this, we introduce a novel reinforcement learning (RL) framework for paralinguistic-aware S2S, ParaS2S, which evaluates and optimizes both content and speaking style directly at the waveform level. We first construct ParaS2SBench, a benchmark comprehensively evaluates S2S models' output for content and style appropriateness from diverse and challenging input queries. It scores the fitness of input-output pairs and aligns well with human judgments, serving as an automatic judge for model outputs. With this scalable scoring feedback, we enable the model to explore and learn from diverse unlabeled speech via Group Relative Policy Optimization (GRPO). Experiments show that existing S2S models fail to respond appropriately to paralinguistic attributes, performing no better than pipeline-based baselines. Our RL approach achieves a 11% relative improvement in response content and style's appropriateness on ParaS2SBench over supervised fine-tuning (SFT), surpassing all prior models while requiring substantially fewer warm-up annotations than pure SFT.
#### Towards Effective and Efficient Non-autoregressive decoders for Conformer and LLM-based ASR using Block-based Attention Mask
 - **Authors:** Tianzi Wang, Xurong Xie, Zengrui Jin, Mengzhe Geng, Jiajun Deng, Zhaoqing Li, Shoukang Hu, Shujie Hu, Guinan Li, Mingyu Cui, Helen Meng, Xunying Liu
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2511.09084

 - **Pdf link:** https://arxiv.org/pdf/2511.09084

 - **Abstract**
 Automatic speech recognition (ASR) systems often rely on autoregressive (AR) Transformer decoder architectures, which limit efficient inference parallelization due to their sequential nature. To this end, non-autoregressive (NAR) approaches aim primarily to achieve significant decoding speedup while the maintaining recognition accuracy that is comparable to AR baselines. This paper proposes a novel NAR block-based attention mask decoder (AMD) that effectively improves decoding efficiency while maintaining ASR accuracy, and also offering flexibility in balancing the performance-efficiency trade-off on both Conformer and large language model (LLM)-based ASR systems. The proposed AMD performs parallel inference within contiguous blocks of output labels while maintaining monotonic left-to-right prediction between blocks. A one-pass beam search algorithm is designed to dynamically fuse Connectionist Temporal Classification (CTC), AR decoder, and AMD probabilities. Experiments are conducted on normal speech LS960 and DBank elderly speech across: a) The Conformer encoder-decoder ASR system with filterbank input features; b) its integration with WavLM features; and c) further advancement by integrating an LLM-based decoder. On the LS960 task, the proposed AMD empowered tripartite decoder achieves decoding speedup ratios of up to 1.44x, 1.55x, and 2.31x under the three model configurations over the CTC + AR baselines, without statistically significant WER increases. When operating with real-time factors (RTFs) comparable to the baselines, the tripartite decoder produces statistically significant WER reductions of 0.19%, 0.62% and 0.13% absolute (4.3%, 16.3%, and 3.8% relative). Similar improvements are also obtained on the DBank task.


by Zyzzyva0381 (Windy). 


2025-11-13
