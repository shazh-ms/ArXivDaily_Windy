# Showing new listings for Thursday, 12 March 2026
Auto update papers at about 2:30am UTC (10:30am Beijing time) every weekday.


阅读 `Usage.md`了解如何使用此repo实现个性化的Arxiv论文推送

See `Usage.md` for instructions on how to personalize the repo. 


Keyword list: ['text-to-speech', 'text to speech', 'tts', 'LLM-based', 'speech', 'voice']


Excluded: []


### Today: 5papers 
#### Calibration-Reasoning Framework for Descriptive Speech Quality Assessment
 - **Authors:** Elizaveta Kostenok, Mathieu Salzmann, Milos Cernak
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Computation and Language (cs.CL)
 - **Arxiv link:** https://arxiv.org/abs/2603.10175

 - **Pdf link:** https://arxiv.org/pdf/2603.10175

 - **Abstract**
 Explainable speech quality assessment requires moving beyond Mean Opinion Scores (MOS) to analyze underlying perceptual dimensions. To address this, we introduce a novel post-training method that tailors the foundational Audio Large Language Model for multidimensional reasoning, detection and classification of audio artifacts. First, a calibration stage aligns the model to predict predefined perceptual dimensions. Second, a reinforcement learning stage leverages Group Relative Policy Optimization (GRPO) with dimension-specific rewards to heavily enhance accuracy of descriptions and temporal localization of quality issues. With this approach we reach state-of-the-art results of 0.71 mean PCC score on the multidimensional QualiSpeech benchmark and 13% improvement in MOS prediction driven by RL-based reasoning. Furthermore, our fine-grained GRPO rewards substantially advance the model's ability to pinpoint and classify audio artifacts in time.
#### Speech Codec Probing from Semantic and Phonetic Perspectives
 - **Authors:** Xuan Shi, Chang Zeng, Tiantian Feng, Shih-Heng Wang, Jianbo Ma, Shrikanth Narayanan
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Computation and Language (cs.CL)
 - **Arxiv link:** https://arxiv.org/abs/2603.10371

 - **Pdf link:** https://arxiv.org/pdf/2603.10371

 - **Abstract**
 Speech tokenizers are essential for connecting speech to large language models (LLMs) in multimodal systems. These tokenizers are expected to preserve both semantic and acoustic information for downstream understanding and generation. However, emerging evidence suggests that what is termed "semantic" in speech representations does not align with text-derived semantics: a mismatch that can degrade multimodal LLM performance. In this paper, we systematically analyze the information encoded by several widely used speech tokenizers, disentangling their semantic and phonetic content through word-level probing tasks, layerwise representation analysis, and cross-modal alignment metrics such as CKA. Our results show that current tokenizers primarily capture phonetic rather than lexical-semantic structure, and we derive practical implications for the design of next-generation speech tokenization methods.
#### FireRedASR2S: A State-of-the-Art Industrial-Grade All-in-One Automatic Speech Recognition System
 - **Authors:** Kaituo Xu, Yan Jia, Kai Huang, Junjie Chen, Wenpeng Li, Kun Liu, Feng-Long Xie, Xu Tang, Yao Hu
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2603.10420

 - **Pdf link:** https://arxiv.org/pdf/2603.10420

 - **Abstract**
 We present FireRedASR2S, a state-of-the-art industrial-grade all-in-one automatic speech recognition (ASR) system. It integrates four modules in a unified pipeline: ASR, Voice Activity Detection (VAD), Spoken Language Identification (LID), and Punctuation Prediction (Punc). All modules achieve SOTA performance on the evaluated benchmarks: FireRedASR2: An ASR module with two variants, FireRedASR2-LLM (8B+ parameters) and FireRedASR2-AED (1B+ parameters), supporting speech and singing transcription for Mandarin, Chinese dialects and accents, English, and code-switching. Compared to FireRedASR, FireRedASR2 delivers improved recognition accuracy and broader dialect and accent coverage. FireRedASR2-LLM achieves 2.89% average CER on 4 public Mandarin benchmarks and 11.55% on 19 public Chinese dialects and accents benchmarks, outperforming competitive baselines including Doubao-ASR, Qwen3-ASR, and Fun-ASR. FireRedVAD: An ultra-lightweight module (0.6M parameters) based on the Deep Feedforward Sequential Memory Network (DFSMN), supporting streaming VAD, non-streaming VAD, and multi-label VAD (mVAD). On the FLEURS-VAD-102 benchmark, it achieves 97.57% frame-level F1 and 99.60% AUC-ROC, outperforming Silero-VAD, TEN-VAD, FunASR-VAD, and WebRTC-VAD. FireRedLID: An Encoder-Decoder LID module supporting 100+ languages and 20+ Chinese dialects and accents. On FLEURS (82 languages), it achieves 97.18% utterance-level accuracy, outperforming Whisper and SpeechBrain. FireRedPunc: A BERT-style punctuation prediction module for Chinese and English. On multi-domain benchmarks, it achieves 78.90% average F1, outperforming FunASR-Punc (62.77%). To advance research in speech processing, we release model weights and code at this https URL.
#### G-STAR: End-to-End Global Speaker-Tracking Attributed Recognition
 - **Authors:** Jing Peng, Ziyi Chen, Haoyu Li, Yucheng Wang, Duo Ma, Mengtian Li, Yunfan Du, Dezhu Xu, Kai Yu, Shuai Wang
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Artificial Intelligence (cs.AI); Human-Computer Interaction (cs.HC); Multimedia (cs.MM); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2603.10468

 - **Pdf link:** https://arxiv.org/pdf/2603.10468

 - **Abstract**
 We study timestamped speaker-attributed ASR for long-form, multi-party speech with overlap, where chunk-wise inference must preserve meeting-level speaker identity consistency while producing time-stamped, speaker-labeled transcripts. Previous Speech-LLM systems tend to prioritize either local diarization or global labeling, but often lack the ability to capture fine-grained temporal boundaries or robust cross-chunk identity linking. We propose G-STAR, an end-to-end system that couples a time-aware speaker-tracking module with a Speech-LLM transcription backbone. The tracker provides structured speaker cues with temporal grounding, and the LLM generates attributed text conditioned on these cues. G-STAR supports both component-wise optimization and joint end-to-end training, enabling flexible learning under heterogeneous supervision and domain shift. Experiments analyze cue fusion, local versus long-context trade-offs and hierarchical objectives.
#### MOS-Bias: From Hidden Gender Bias to Gender-Aware Speech Quality Assessment
 - **Authors:** Wenze Ren, Yi-Cheng Lin, Wen-Chin Huang, Erica Cooper, Ryandhimas E. Zezario, Hsin-Min Wang, Hung-yi Lee, Yu Tsao
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2603.10723

 - **Pdf link:** https://arxiv.org/pdf/2603.10723

 - **Abstract**
 The Mean Opinion Score (MOS) serves as the standard metric for speech quality assessment, yet biases in human annotations remain underexplored. We conduct the first systematic analysis of gender bias in MOS, revealing that male listeners consistently assign higher scores than female listeners--a gap that is most pronounced in low-quality speech and gradually diminishes as quality improves. This quality-dependent structure proves difficult to eliminate through simple calibration. We further demonstrate that automated MOS models trained on aggregated labels exhibit predictions skewed toward male standards of perception. To address this, we propose a gender-aware model that learns gender-specific scoring patterns through abstracting binary group embeddings, thereby improving overall and gender-specific prediction accuracy. This study establishes that gender bias in MOS constitutes a systematic, learnable pattern demanding attention in equitable speech evaluation.


by Zyzzyva0381 (Windy). 


2026-03-12
