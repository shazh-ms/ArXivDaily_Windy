# Showing new listings for Tuesday, 7 April 2026
Auto update papers at about 2:30am UTC (10:30am Beijing time) every weekday.


阅读 `Usage.md`了解如何使用此repo实现个性化的Arxiv论文推送

See `Usage.md` for instructions on how to personalize the repo. 


Keyword list: ['text-to-speech', 'text to speech', 'tts', 'LLM-based', 'speech', 'voice']


Excluded: []


### Today: 6papers 
#### Rewriting TTS Inference Economics: Lightning V2 on Tenstorrent Achieves 4x Lower Cost Than NVIDIA L40S
 - **Authors:** Ranjith M. S., Akshat Mandloi, Sudarshan Kamath
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Distributed, Parallel, and Cluster Computing (cs.DC); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2604.03279

 - **Pdf link:** https://arxiv.org/pdf/2604.03279

 - **Abstract**
 Text-to-Speech (TTS) models are significantly more numerically fragile than Large Language Models (LLMs) due to their continuous waveform generation and perceptual sensitivity to small numerical perturbations. While aggressive precision reduction techniques such as BlockFloat8 (BFP8) and low-fidelity (LoFi) compute have been widely adopted in language models, applying similar strategies to TTS systems often results in audible artifacts, phase instability, and spectral distortion. In this work, we present Lightning V2, a production-grade TTS model co-optimized for Tenstorrent hardware. Through precision-aware architectural design and hardware-software co-optimization, we achieve over 95% LoFi computational fidelity and more than 80% BlockFloat8 deployment without measurable degradation in audio quality. Leveraging Tenstorrent's Network-on-Chip (NoC), distributed SRAM, and deterministic execution model, we reduce memory movement and redundant weight fetches, enabling efficient low-precision inference. Compared to an NVIDIA L40S baseline, Lightning V2 achieves approximately 4x lower on-prem accelerator cost at equivalent throughput, while maintaining production audio fidelity. Our results demonstrate that precision co-design, combined with hardware-aware optimization, can fundamentally reshape the economics of real-time speech inference.
#### MALEFA: Multi-grAnularity Learning and Effective False Alarm Suppression for Zero-shot Keyword Spotting
 - **Authors:** Lo-Ya Li, Tien-Hong Lo, Jeih-Weih Hung, Shih-Chieh Huang, Berlin Chen
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2604.03689

 - **Pdf link:** https://arxiv.org/pdf/2604.03689

 - **Abstract**
 User-defined keyword spotting (KWS) without resorting to domain-specific pre-labeled training data is of fundamental importance in building adaptable and personalized voice interfaces. However, such systems are still faced with arduous challenges, including constrained computational resources and limited annotated training data. Existing methods also struggle to distinguish acoustically similar keywords, often leading to a pesky false alarm rate (FAR) in real-world deployments. To mitigate these limitations, we put forward MALEFA, a novel lightweight zero-shot KWS framework that jointly learns utterance- and phoneme-level alignments via cross-attention and a multi-granularity contrastive learning objective. Evaluations on four public benchmark datasets show that MALEFA achieves a high accuracy of 90%, significantly reducing FAR to 0.007% on the AMI dataset. Beyond its strong performance, MALEFA demonstrates high computational efficiency and can readily support real-time deployment on resource-constrained devices.
#### AffectSpeech: A Large-Scale Emotional Speech Dataset with Fine-Grained Textual Descriptions for Speech Emotion Captioning and Synthesis
 - **Authors:** Tianhua Qi, Wenming Zheng, BjÃ¶rn W. Schuller, Zhaojie Luo, Haizhou Li
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD); Signal Processing (eess.SP)
 - **Arxiv link:** https://arxiv.org/abs/2604.04160

 - **Pdf link:** https://arxiv.org/pdf/2604.04160

 - **Abstract**
 Emotion is essential in spoken communication, yet most existing frameworks in speech emotion modeling rely on predefined categories or low-dimensional continuous attributes, which offer limited expressive capacity. Recent advances in speech emotion captioning and synthesis have shown that textual descriptions provide a more flexible and interpretable alternative for representing affective characteristics in speech. However, progress in this direction is hindered by the lack of an emotional speech dataset aligned with reliable and fine-grained natural language annotations. To tackle this, we introduce AffectSpeech, a large-scale corpus of human-recorded speech enriched with structured descriptions for fine-grained emotion analysis and generation. Each utterance is characterized across six complementary dimensions, including sentiment polarity, open-vocabulary emotion captions, intensity level, prosodic attributes, prominent segments, and semantic content, enabling multi-granular modeling of vocal expression. To balance annotation quality and scalability, we adopt a human-LLM collaborative annotation pipeline that integrates algorithmic pre-labeling, multi-LLM description generation, and human-in-the-loop verification. Furthermore, these annotations are reformulated into diverse descriptive styles to enhance linguistic diversity and reduce stylistic bias in downstream modeling. Experimental results on speech emotion captioning and synthesis demonstrate that models trained on AffectSpeech consistently achieve superior performance across multiple evaluation settings.
#### Full-Duplex-Bench-v3: Benchmarking Tool Use for Full-Duplex Voice Agents Under Real-World Disfluency
 - **Authors:** Guan-Ting Lin, Chen Chen, Zhehuai Chen, Hung-yi Lee
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Computation and Language (cs.CL)
 - **Arxiv link:** https://arxiv.org/abs/2604.04847

 - **Pdf link:** https://arxiv.org/pdf/2604.04847

 - **Abstract**
 We introduce Full-Duplex-Bench-v3 (FDB-v3), a benchmark for evaluating spoken language models under naturalistic speech conditions and multi-step tool use. Unlike prior work, our dataset consists entirely of real human audio annotated for five disfluency categories, paired with scenarios requiring chained API calls across four task domains. We evaluate six model configurations -- GPT-Realtime, Gemini Live 2.5, Gemini Live 3.1, Grok, Ultravox v0.7, and a traditional Cascaded pipeline (Whisper$\rightarrow$GPT-4o$\rightarrow$TTS) -- across accuracy, latency, and turn-taking dimensions. GPT-Realtime leads on Pass@1 (0.600) and interruption avoidance (13.5\%); Gemini Live 3.1 achieves the fastest latency (4.25~s) but the lowest turn-take rate (78.0\%); and the Cascaded baseline, despite a perfect turn-take rate, incurs the highest latency (10.12~s). Across all systems, self-correction handling and multi-step reasoning under hard scenarios remain the most consistent failure modes.
#### FastTurn: Unifying Acoustic and Streaming Semantic Cues for Low-Latency and Robust Turn Detection
 - **Authors:** Chengyou Wang, Hongfei Xue, Chunjiang He, Jingbin Hu, Shuiyuan Wang, Bo Wu, Yuyu Ji, Jimeng Zheng, Ruofei Chen, Zhou Zhu, Lei Xie
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2604.01897

 - **Pdf link:** https://arxiv.org/pdf/2604.01897

 - **Abstract**
 Recent advances in AudioLLMs have enabled spoken dialogue systems to move beyond turn-based interaction toward real-time full-duplex communication, where the agent must decide when to speak, yield, or interrupt while the user is still talking. Existing full-duplex approaches either rely on voice activity cues, which lack semantic understanding, or on ASR-based modules, which introduce latency and degrade under overlapping speech and noise. Moreover, available datasets rarely capture realistic interaction dynamics, limiting evaluation and deployment. To mitigate the problem, we propose \textbf{FastTurn}, a unified framework for low-latency and robust turn detection. To advance latency while maintaining performance, FastTurn combines streaming CTC decoding with acoustic features, enabling early decisions from partial observations while preserving semantic cues. We also release a test set based on real human dialogue, capturing authentic turn transitions, overlapping speech, backchannels, pauses, pitch variation, and environmental noise. Experiments show FastTurn achieves higher decision accuracy with lower interruption latency than representative baselines and remains robust under challenging acoustic conditions, demonstrating its effectiveness for practical full-duplex dialogue systems.
#### Joint Fullband-Subband Modeling for High-Resolution SingFake Detection
 - **Authors:** Xuanjun Chen, Chia-Yu Hu, Sung-Feng Huang, Haibin Wu, Hung-yi Lee, Jyh-Shing Roger Jang
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS); Signal Processing (eess.SP)
 - **Arxiv link:** https://arxiv.org/abs/2604.04841

 - **Pdf link:** https://arxiv.org/pdf/2604.04841

 - **Abstract**
 Rapid advances in singing voice synthesis have increased unauthorized imitation risks, creating an urgent need for better Singing Voice Deepfake (SingFake) Detection, also known as SVDD. Unlike speech, singing contains complex pitch, wide dynamic range, and timbral variations. Conventional 16 kHz-sampled detectors prove inadequate, as they discard vital high-frequency information. This study presents the first systematic analysis of high-resolution (44.1 kHz sampling rate) audio for SVDD. We propose a joint fullband-subband modeling framework: the fullband captures global context, while subband-specific experts isolate fine-grained synthesis artifacts unevenly distributed across the spectrum. Experiments on the WildSVDD dataset demonstrate that high-frequency subbands provide essential complementary cues. Our framework significantly outperforms 16 kHz-sampled models, proving that high-resolution audio and strategic subband integration are critical for robust in-the-wild detection.


by Zyzzyva0381 (Windy). 


2026-04-07
