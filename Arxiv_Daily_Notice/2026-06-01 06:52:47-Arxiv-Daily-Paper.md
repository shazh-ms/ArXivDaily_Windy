# Showing new listings for Monday, 1 June 2026
Auto update papers at about 2:30am UTC (10:30am Beijing time) every weekday.


阅读 `Usage.md`了解如何使用此repo实现个性化的Arxiv论文推送

See `Usage.md` for instructions on how to personalize the repo. 


Keyword list: ['text-to-speech', 'text to speech', 'tts', 'LLM-based', 'speech', 'voice']


Excluded: []


### Today: 9papers 
#### Extracting accent features in spoken Brazilian Portuguese without sociolinguistic labels
 - **Authors:** Pedro H. L. Leite, Pedro Benevenuto Valadares, Luiz W. P. Biscainho
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Computation and Language (cs.CL)
 - **Arxiv link:** https://arxiv.org/abs/2605.30457

 - **Pdf link:** https://arxiv.org/pdf/2605.30457

 - **Abstract**
 Regional accent classification in Brazilian Portuguese (pt-BR) suffers from the need for reliable labeling. While large self-supervised learning (SSL) speech models are powerful, their training pipelines dilute sociophonetic information, since accent labels are generally not reliable or are not used in training objectives. This work introduces a novel workflow for feature extraction using only acoustic labels. By isolating explicit regional accent landmarks and using a phoneme-based forced aligner (ZIPA), our targeted feature set captures dialectal variance more effectively than utterance embeddings, demonstrating that localized features can outperform general-purpose architectures on accent-related tasks using minimal and objective data labels.
#### OpenSTBench: Beyond Semantic Evaluation for Speech Translation
 - **Authors:** Yanjie An, Yuxiang Zhao, Yichi Zhang, Qixi Zheng, Yujie Tu, Keqi Deng, Kai Yu, Xie Chen
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Artificial Intelligence (cs.AI)
 - **Arxiv link:** https://arxiv.org/abs/2605.30792

 - **Pdf link:** https://arxiv.org/pdf/2605.30792

 - **Abstract**
 Speech translation systems increasingly span speech-to-text translation (S2TT), speech-to-speech translation (S2ST), offline translation, and streaming generation, producing outputs that differ in modality, speech realization, and timing behavior. Existing evaluation practices assess important aspects such as translation quality, speech quality, and temporal quality, but these aspects are often evaluated under separate protocols, making it difficult to compare heterogeneous systems comprehensively. To address this gap, we present OpenSTBench, a unified multidimensional evaluation framework that organizes heterogeneous speech translation outputs into a shared evaluation format. OpenSTBench supports both S2TT and S2ST systems in offline and streaming settings, and jointly evaluates translation quality, speech quality, speaker preservation, emotion and paralinguistic fidelity, temporal consistency, and latency. Through experiments on representative speech translation systems, we show that systems with strong translation quality can still differ substantially in speech quality, as well as in temporal quality. OpenSTBench provides a reproducible protocol for analyzing these cross-dimensional differences and supporting application-oriented comparison of speech translation systems. The code and datasets are available at this https URL.
#### A Unified and Reproducible Experimentation Framework for Speech Understanding
 - **Authors:** Jing Peng, Junhao Du, Chenghao Wang, Hanqi Li, Yi Yang, Yixuan Wang, Xiaoyu Gu, Guanyu Chen, Yucheng Wang, Jiang Li, Zhangjie Zhao, Haoran Wang, Wenming Tu, Haoyu Li, Duo Ma, Lirong Qian, Yu Xi, Wen Wen, Jiaqi Guo, Hui Zhang, Shuai Fan, Wenbin Jiang, Shuai Wang, Kai Yu
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Artificial Intelligence (cs.AI); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2605.30899

 - **Pdf link:** https://arxiv.org/pdf/2605.30899

 - **Abstract**
 Speech foundation models and Speech LLMs have advanced speech understanding, yet deployment-oriented model selection is hindered by non-comparable evaluations caused by mismatched post-processing, and by training results that are hard to reproduce across data scales and pipelines. We present SURE, a unified experimentation framework that standardizes prediction formats, normalization, and scoring. SURE evaluates strong systems across paradigms, from conventional pipelines to Speech LLMs, on representative tasks under realistic acoustic and linguistic stressors. Beyond evaluation, SURE introduces an agent-assisted training conversion flow that maps paper and code into versioned, runnable training pipelines under a unified protocol on matched open-data subsets. Overall, SURE improves comparability and reproducibility for deployment-oriented evaluation.
#### ImmersiveTTS: Environment-Aware Text-to-Speech with Multimodal Diffusion Transformer and Domain-Specific Representation Alignment
 - **Authors:** Jun-Hak Yun, Seung-Bin Kim, Seong-Whan Lee
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Artificial Intelligence (cs.AI); Computation and Language (cs.CL)
 - **Arxiv link:** https://arxiv.org/abs/2605.30965

 - **Pdf link:** https://arxiv.org/pdf/2605.30965

 - **Abstract**
 Recent advancements in text-guided audio generation have yielded promising results in diverse domains, including sound effects, speech, and music. However, jointly generating speech with environmental audio remains challenging due to the inherent disparities in their acoustic patterns and temporal dynamics. We propose ImmersiveTTS, an environment-aware text-to-speech (TTS) model that generates natural speech seamlessly integrated within environmental contexts by explicitly modeling cross-modal interactions. Our model builds on a multimodal diffusion transformer and fuses transcript-aligned speech latent with text-conditioned environmental context via joint attention. To enhance semantic consistency, we introduce a domain-specific representation alignment objective tailored to environment-aware TTS, leveraging complementary self-supervised representations from speech and audio encoders. Experimental results show that ImmersiveTTS achieves higher naturalness, intelligibility, and audio fidelity than existing approaches across objective metrics and human listening tests.
#### SwanVoice: Expressive Long-Form Zero-Shot Speech Synthesis for Both Monologue and Dialogue
 - **Authors:** Ruiqi Li, Yu Zhang, Changhao Pan, Ke Lei, Xiang Yin, Cheng Yang
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2605.30993

 - **Pdf link:** https://arxiv.org/pdf/2605.30993

 - **Abstract**
 Zero-shot text-to-speech (TTS) has improved substantially for single-speaker synthesis, yet expressive long-form multi-speaker dialogue remains difficult. A common workaround is to synthesize each turn with a monologue TTS model and stitch the outputs together. This adds inference cost and often breaks acoustic consistency, conversational coherence, and affective continuity across turns. Recent dialogue TTS systems have begun to address this setting, but they still struggle to keep expressive coherence, controllable speaker switching, and monologue quality at the same time. We present SwanData-Speech and SwanVoice. SwanData-Speech builds monologue and dialogue corpora from in-the-wild audio, using Swan Forced Aligner for pause-aware word-level alignment and RobustMegaTTS3 for pronunciation-hard cases. Built on these data, SwanVoice is a zero-shot TTS model for 1--4 speakers, combining a 25 Hz VAE, raw-text conditioning with pause-aware symbols and pinyin substitution, and a flow-matching DiT with speaker-turn conditioning. Training starts from monologue speech, moves through mixed and real dialogue data, and then uses DiffusionNFT post-training with phone-level and speaker-similarity rewards. On SwanBench-Speech, SwanVoice obtains higher richness and hierarchy scores than all evaluated open-source baselines in both monologue and dialogue settings, while content accuracy remains the main limitation. Audio demos are available at this https URL.
#### UNISON: A Unified Sound Generation and Editing Framework via Deep LLM Fusion
 - **Authors:** Zhaoqing Li, Haoning Xu, Jingran Su, Yaofang Liu, Zhefan Rao, Huimeng Wang, Jiajun Deng, Tianzi Wang, Zengrui Jin, Rui Liu, Haoxuan Che, Xunying Liu
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2605.31530

 - **Pdf link:** https://arxiv.org/pdf/2605.31530

 - **Abstract**
 We present UNISON, a latent diffusion framework that unifies speech generation, sound generation, and audio editing within a single model. A single model handles text-to-audio, text-to-speech, zero-shot speaker cloning, mixed speech-and-sound generation, scene-level audio editing, speech-in-scene editing, and timed temporal composition, all of which share a single set of weights. Our architecture features two core designs: (1) Layer-wise deep LLM fusion, which injects hidden states from uniformly sampled layers of a frozen MLLM into corresponding MM-DiT blocks via learned projections, providing depth-matched semantic conditioning that improves instruction following over single-layer baselines; and (2) a unified multi-task architecture where task identity is encoded solely by a channel-wise mask and source audio is provided through VAE-encoded channel concatenation. Training is stabilized by an online GPU-side multi-task data synthesis pipeline with task-homogeneous batching and a two-stage curriculum. With 621M--732M trainable parameters, UNISON achieves results competitive with or exceeding task-specialist models across evaluated domains, while being roughly $4\times$ smaller than comparable unified systems.
#### Escaping the Linearity Trap: Manifold Detours for Black-Box Adversarial Attacks on Singing Audio Deepfake Detection
 - **Authors:** Yifan Liao, Yule Liu, Zhen Sun, Zongmin Zhang, Yupeng He, Jiaheng Wei, Xinhu Zheng, Xinlei He
 - **Subjects:** Subjects:
Cryptography and Security (cs.CR); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2605.30366

 - **Pdf link:** https://arxiv.org/pdf/2605.30366

 - **Abstract**
 Recent Singing Voice Synthesis (SVS) advances enable highly realistic but potentially malicious AI covers, making singing voice deepfake detection (SVDD) crucial. Self-Supervised Learning (SSL)-based detectors achieve state-of-the-art performance by fine-tuning speech SSL backbones to capture singing-specific spoof artifacts. Existing adversarial attacks often fail against SSL-SVDD, creating a false impression of inherent robustness. We reveal this stems from two challenges. First, at the objective level, attacks optimize cross-entropy on local surrogates, crossing surrogate-specific boundaries rather than suppressing shared spoof evidence. Second, at the method level, attacks follow the surrogate's dominant gradient direction. In SSL-SVDD, this aligns with fine-tuned artifact-sensitive directions, limiting transferability to unseen detectors - a geometric failure we term the Linearity Trap. To properly evaluate robustness, we propose MARS (Meta-Adversarial Regression of Semantics), a transfer-based black-box framework tailored to SSL-SVDD. Structurally, MARS shifts to hypothesis-evidence manipulation by constructing a natural semantic anchor from the pre-trained SSL space and an artifact anchor from the fine-tuned space. Algorithmically, MARS escapes the Linearity Trap via bi-level optimization: the inner stage induces tangential exploration, while the outer stage guides the audio toward the natural semantic manifold. Experiments on the CtrSVDD benchmark show MARS improves Attack Success Rate (ASR) in in-distribution transfer (13%), out-of-distribution transfer (10%), and cross-task evaluation (36%), highlighting the urgent need for robust SVDD systems.
#### Chatterbox-Flash: Prior-Calibrated Block Diffusion for Streaming Zero-Shot TTS
 - **Authors:** Deokjin Seo, Gangin Park, Kihyun Nam
 - **Subjects:** Subjects:
Sound (cs.SD); Artificial Intelligence (cs.AI); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2605.30748

 - **Pdf link:** https://arxiv.org/pdf/2605.30748

 - **Abstract**
 We present Chatterbox-Flash, a zero-shot text-to-speech model obtained by fine-tuning a pretrained autoregressive TTS decoder into a block-diffusion decoder, enabling parallel token generation within each block while retaining block-by-block streaming. We find that naively transferring mainstream block-diffusion decoding to discrete speech tokens degrades quality, as a long-tail token distribution biases parallel position selection toward a few high-frequency tokens. To mitigate this without architectural modification, we introduce two inference-time techniques: prior-calibrated scoring, which subtracts the block-level marginal token distribution, and an early-decoding schedule, which adaptively terminates iteration based on calibrated confidence. On standard zero-shot TTS benchmarks, Chatterbox-Flash attains high-fidelity synthesis comparable to strong autoregressive and non-autoregressive baselines, while supporting streaming inference with time-to-first-packet on par with streaming AR systems and substantially lower real-time factor. Code and audio samples are available at this https URL.
#### Scaling Conversational Hungarian ASR: The BEA-Dialogue+ Corpus
 - **Authors:** Máté Gedeon, Piroska Zsófia Barta, Péter Mihajlik, Katalin Mády
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Artificial Intelligence (cs.AI); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2605.31469

 - **Pdf link:** https://arxiv.org/pdf/2605.31469

 - **Abstract**
 Conversational automatic speech recognition in Hungarian is constrained by the limited amount of publicly available dialogue-style training data. The BEA-Dialogue corpus addresses this need, but its strictly speaker-disjoint train/dev/eval split reduces the usable material to only 85 hours. In this paper, we introduce BEA-Dialogue+, an expanded version of the corpus that relaxes the split criterion for experimenters and dialogue partners while preserving complete separation of the primary speakers. This results in 200 hours of transcribed natural conversations and enables a controlled study of the trade-off between additional training data and speaker overlap across the splits. We evaluate several Whisper- and FastConformer-based models on both corpus versions, including Serialized Output Training (SOT)-based fine-tuning for dialogue transcription. Our results show that the larger corpus is more challenging for models without fine-tuning, whereas SOT-based adaptation yields consistent improvements in WER, CER, cpWER, and cpCER. Overall, BEA-Dialogue+ provides a substantially larger yet still demanding benchmark for Hungarian dialogue ASR, and a practical resource for training and evaluating dialogue transcription systems.


by Zyzzyva0381 (Windy). 


2026-06-01
