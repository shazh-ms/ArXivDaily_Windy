# Showing new listings for Monday, 16 February 2026
Auto update papers at about 2:30am UTC (10:30am Beijing time) every weekday.


阅读 `Usage.md`了解如何使用此repo实现个性化的Arxiv论文推送

See `Usage.md` for instructions on how to personalize the repo. 


Keyword list: ['text-to-speech', 'text to speech', 'tts', 'LLM-based', 'speech', 'voice']


Excluded: []


### Today: 6papers 
#### Acoustivision Pro: An Open-Source Interactive Platform for Room Impulse Response Analysis and Acoustic Characterization
 - **Authors:** Mandip Goswami
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD); Signal Processing (eess.SP)
 - **Arxiv link:** https://arxiv.org/abs/2602.12299

 - **Pdf link:** https://arxiv.org/pdf/2602.12299

 - **Abstract**
 Room acoustics analysis plays a central role in architectural design, audio engineering, speech intelligibility assessment, and hearing research. Despite the availability of standardized metrics such as reverberation time, clarity, and speech transmission index, accessible tools that combine rigorous signal processing with intuitive visualization remain scarce. This paper presents AcoustiVision Pro, an open-source web-based platform for comprehensive room impulse response (RIR) analysis. The system computes twelve distinct acoustic parameters from uploaded or dataset-sourced RIRs, provides interactive 3D visualizations of early reflections, generates frequency-dependent decay characteristics through waterfall plots, and checks compliance against international standards including ANSI S12.60 and ISO 3382. We introduce the accompanying RIRMega and RIRMega Speech datasets hosted on Hugging Face, containing thousands of simulated room impulse responses with full metadata. The platform supports real-time auralization through FFT-based convolution, exports detailed PDF reports suitable for engineering documentation, and provides CSV data export for further analysis. We describe the mathematical foundations underlying each acoustic metric, detail the system architecture, and present preliminary case studies demonstrating the platform's utility across diverse application domains including classroom acoustics, healthcare facility design, and recording studio evaluation.
#### Decoder-only Conformer with Modality-aware Sparse Mixtures of Experts for ASR
 - **Authors:** Jaeyoung Lee, Masato Mimura
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Artificial Intelligence (cs.AI); Computation and Language (cs.CL); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2602.12546

 - **Pdf link:** https://arxiv.org/pdf/2602.12546

 - **Abstract**
 We present a decoder-only Conformer for automatic speech recognition (ASR) that processes speech and text in a single stack without external speech encoders or pretrained large language models (LLM). The model uses a modality-aware sparse mixture of experts (MoE): disjoint expert pools for speech and text with hard routing and top-1 selection, embedded in hybrid-causality Conformer blocks (bidirectional for speech, causal for text). Training combines CTC on speech positions with label-smoothed cross-entropy for text generation. Our 113M-parameter model consistently improves WER over a 139M AED baseline on Librispeech (2.8% vs. 3.2% test-clean; 5.6% vs. 6.0% test-other). On Common Voice 16.1 with a single multilingual model across five languages, our approach reduces average WER from 12.2% to 10.6%. To our knowledge, this is the first randomly initialized decoder-only ASR that surpasses strong AED baselines via modality-aware routing and sparse MoE, achieving better accuracy with fewer active parameters and without alignment/adaptation modules.
#### A two-step approach for speech enhancement in low-SNR scenarios using cyclostationary beamforming and DNNs
 - **Authors:** Giovanni Bologni, Nicolás Arrieta Larraza, Richard Heusdens, Richard C. Hendriks
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2602.12986

 - **Pdf link:** https://arxiv.org/pdf/2602.12986

 - **Abstract**
 Deep Neural Networks (DNNs) often struggle to suppress noise at low signal-to-noise ratios (SNRs). This paper addresses speech enhancement in scenarios dominated by harmonic noise and proposes a framework that integrates cyclostationarity-aware preprocessing with lightweight DNN-based denoising. A cyclic minimum power distortionless response (cMPDR) spectral beamformer is used as a preprocessing block. It exploits the spectral correlations of cyclostationary noise to suppress harmonic components prior to learning-based enhancement and does not require modifications to the DNN architecture. The proposed pipeline is evaluated in a single-channel setting using two DNN architectures: a simple and lightweight convolutional recurrent neural network (CRNN), and a state-of-the-art model, namely ultra-low complexity network (ULCNet). Experiments on synthetic data and real-world recordings dominated by rotating machinery noise demonstrate consistent improvements over end-to-end DNN baselines, particularly at low SNRs. Remarkably, a parameter-efficient CRNN with cMPDR preprocessing surpasses the performance of the larger ULCNet operating on raw or Wiener-filtered inputs. These results indicate that explicitly incorporating cyclostationarity as a signal prior is more effective than increasing model capacity alone for suppressing harmonic interference.
#### Retrieval-Augmented Self-Taught Reasoning Model with Adaptive Chain-of-Thought for ASR Named Entity Correction
 - **Authors:** Junjie An, Jingguang Tian, Tianyi Wang, Yu Gao, Xiaofeng Mou, Yi Xu
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Artificial Intelligence (cs.AI); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2602.12287

 - **Pdf link:** https://arxiv.org/pdf/2602.12287

 - **Abstract**
 End-to-end automatic speech recognition (ASR) systems frequently misrecognize domain-specific phrases like named entities, which can cause catastrophic failures in downstream tasks. A new family of named entity correction methods based on large language models (LLMs) has recently emerged. However, these approaches have yet to fully exploit the sophisticated reasoning capabilities inherent to LLMs. To bridge this gap, we propose a novel retrieval-augmented generation framework for correcting named entity errors in ASR. Our approach consists of two key components: (1) a rephrasing language model (RLM) for named entity recognition, followed by candidate retrieval using a phonetic-level edit distance; and (2) a novel self-taught reasoning model with adaptive chain-of-thought (A-STAR) that dynamically adjusts the depth of its reasoning based on task difficulty. Experiments on the AISHELL-1 and Homophone datasets demonstrate the effectiveness of our method, which achieves relative reductions in the named entity character error rate of 17.96\% and 34.42\%, respectively, compared to a strong baseline.
#### Beyond Musical Descriptors: Extracting Preference-Bearing Intent in Music Queries
 - **Authors:** Marion Baranes, Romain Hennequin, Elena V. Epure
 - **Subjects:** Subjects:
Sound (cs.SD); Computation and Language (cs.CL); Information Retrieval (cs.IR); Machine Learning (cs.LG); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2602.12301

 - **Pdf link:** https://arxiv.org/pdf/2602.12301

 - **Abstract**
 Although annotated music descriptor datasets for user queries are increasingly common, few consider the user's intent behind these descriptors, which is essential for effectively meeting their needs. We introduce MusicRecoIntent, a manually annotated corpus of 2,291 Reddit music requests, labeling musical descriptors across seven categories with positive, negative, or referential preference-bearing roles. We then investigate how reliably large language models (LLMs) can extract these music descriptors, finding that they do capture explicit descriptors but struggle with context-dependent ones. This work can further serve as a benchmark for fine-grained modeling of user intent and for gaining insights into improving LLM-based music understanding systems.
#### Lamer-SSL: Layer-aware Mixture of LoRA Experts for Continual Multilingual Expansion of Self-supervised Models without Forgetting
 - **Authors:** Jing Xu, Minglin Wu, Xueyuan Chen, Xixin Wu, Helen Meng
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2602.12746

 - **Pdf link:** https://arxiv.org/pdf/2602.12746

 - **Abstract**
 Despite their impressive performance, self-supervised speech models often struggle to generalize to new languages and tend to forget previously acquired knowledge during continual training. To address this, we propose Lamer-SSL, a parameter-efficient framework that integrates a Layer-Aware MixturE of LoRA Experts (Lamer) module with a replay strategy. The Lamer module enables flexible balancing between shared and language-specific representations, while layer-aware expert allocation assigns more experts to deeper layers where semantic information is richer. Meanwhile, the replay strategy retains prior knowledge using minimal data, mitigating forgetting during continual training. Experiments on automatic speech recognition (ASR) and language identification (LID) demonstrate that Lamer-SSL extends self-supervised models to new languages effectively while maintaining strong performance on previously learned languages with only 2.14% parameters being trainable.


by Zyzzyva0381 (Windy). 


2026-02-16
