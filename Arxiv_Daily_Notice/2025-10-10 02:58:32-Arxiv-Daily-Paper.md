# Showing new listings for Friday, 10 October 2025
Auto update papers at about 2:30am UTC (10:30am Beijing time) every weekday.


阅读 `Usage.md`了解如何使用此repo实现个性化的Arxiv论文推送

See `Usage.md` for instructions on how to personalize the repo. 


Keyword list: ['text-to-speech', 'text to speech', 'tts', 'LLM-based', 'speech', 'voice']


Excluded: []


### Today: 8papers 
#### Full-Duplex-Bench-v2: A Multi-Turn Evaluation Framework for Duplex Dialogue Systems with an Automated Examiner
 - **Authors:** Guan-Ting Lin, Shih-Yun Shan Kuan, Jiatong Shi, Kai-Wei Chang, Siddhant Arora, Shinji Watanabe, Hung-yi Lee
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2510.07838

 - **Pdf link:** https://arxiv.org/pdf/2510.07838

 - **Abstract**
 While full-duplex speech agents enable natural, low-latency interaction by speaking and listening simultaneously, their consistency and task performance in multi-turn settings remain underexplored. We introduce Full-Duplex-Bench-v2 (FDB-v2), a streaming framework that integrates with an automated examiner that enforces staged goals under two pacing setups (Fast vs. Slow). FDB-v2 covers four task families: daily, correction, entity tracking, and safety. We report turn-taking fluency, multi-turn instruction following, and task-specific competence. The framework is extensible, supporting both commercial APIs and open source models. When we test full-duplex systems with FDB-v2, they often get confused when people talk at the same time, struggle to handle corrections smoothly, and sometimes lose track of who or what is being talked about. Through an open-sourced, standardized streaming protocol and a task set, FDB-v2 makes it easy to extend to new task families, allowing the community to tailor and accelerate evaluation of multi-turn full-duplex systems.
#### Bloodroot: When Watermarking Turns Poisonous For Stealthy Backdoor
 - **Authors:** Kuan-Yu Chen, Yi-Cheng Lin, Jeng-Lin Li, Jian-Jiun Ding
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2510.07909

 - **Pdf link:** https://arxiv.org/pdf/2510.07909

 - **Abstract**
 Backdoor data poisoning is a crucial technique for ownership protection and defending against malicious attacks. Embedding hidden triggers in training data can manipulate model outputs, enabling provenance verification, and deterring unauthorized use. However, current audio backdoor methods are suboptimal, as poisoned audio often exhibits degraded perceptual quality, which is noticeable to human listeners. This work explores the intrinsic stealthiness and effectiveness of audio watermarking in achieving successful poisoning. We propose a novel Watermark-as-Trigger concept, integrated into the Bloodroot backdoor framework via adversarial LoRA fine-tuning, which enhances perceptual quality while achieving a much higher trigger success rate and clean-sample accuracy. Experiments on speech recognition (SR) and speaker identification (SID) datasets show that watermark-based poisoning remains effective under acoustic filtering and model pruning. The proposed Bloodroot backdoor framework not only secures data-to-model ownership, but also well reveals the risk of adversarial misuse.
#### Pseudo2Real: Task Arithmetic for Pseudo-Label Correction in Automatic Speech Recognition
 - **Authors:** Yi-Cheng Lin, Yu-Hsuan Li Liang, Hsuan Su, Tzu-Quan Lin, Shang-Tse Chen, Yun-Nung Chen, Hung-yi Lee
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Computation and Language (cs.CL)
 - **Arxiv link:** https://arxiv.org/abs/2510.08047

 - **Pdf link:** https://arxiv.org/pdf/2510.08047

 - **Abstract**
 Robust ASR under domain shift is crucial because real-world systems encounter unseen accents and domains with limited labeled data. Although pseudo-labeling offers a practical workaround, it often introduces systematic, accent-specific errors that filtering fails to fix. We ask: How can we correct these recurring biases without target ground truth? We propose a simple parameter-space correction: in a source domain containing both real and pseudo-labeled data, two ASR models are fine-tuned from the same initialization, one on ground-truth labels and the other on pseudo-labels, and their weight difference forms a correction vector that captures pseudo-label biases. When applied to a pseudo-labeled target model, this vector enhances recognition, achieving up to a 35% relative Word Error Rate (WER) reduction on AfriSpeech-200 across ten African accents with the Whisper tiny model.
#### DialoSpeech: Dual-Speaker Dialogue Generation with LLM and Flow Matching
 - **Authors:** Hanke Xie, Dake Guo, Chengyou Wang, Yue Li, Wenjie Tian, Xinfa Zhu, Xinsheng Wang, Xiulin Li, Guanqiong Miao, Bo Liu, Lei Xie
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2510.08373

 - **Pdf link:** https://arxiv.org/pdf/2510.08373

 - **Abstract**
 Recent advances in text-to-speech (TTS) synthesis, particularly those leveraging large language models (LLMs), have significantly improved expressiveness and naturalness. However, generating human-like, interactive dialogue speech remains challenging. Current systems face limitations due to the scarcity of dual-track data and difficulties in achieving naturalness, contextual coherence, and interactional dynamics, such as turn-taking, overlapping speech, and speaker consistency, in multi-turn conversations. To address these challenges, we propose DialoSpeech, a dual-track architecture combining a large language model with Chunked Flow Matching for expressive, human-like dialogue speech synthesis. DialoSpeech generates natural multi-turn conversations with coherent speaker turns and natural overlaps, supporting both Chinese and English and cross-lingual speech synthesis. We introduce a data processing pipeline to construct dual-track dialogue datasets, facilitating scalable training and experimental validation. Experiments show that our model outperforms baselines, offering a solution for generating human-like spoken dialogues. Audio samples are available at this https URL
#### MeanVC: Lightweight and Streaming Zero-Shot Voice Conversion via Mean Flows
 - **Authors:** Guobin Ma, Jixun Yao, Ziqian Ning, Yuepeng Jiang, Lingxin Xiong, Lei Xie, Pengcheng Zhu
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2510.08392

 - **Pdf link:** https://arxiv.org/pdf/2510.08392

 - **Abstract**
 Zero-shot voice conversion (VC) aims to transfer timbre from a source speaker to any unseen target speaker while preserving linguistic content. Growing application scenarios demand models with streaming inference capabilities. This has created a pressing need for models that are simultaneously fast, lightweight, and high-fidelity. However, existing streaming methods typically rely on either autoregressive (AR) or non-autoregressive (NAR) frameworks, which either require large parameter sizes to achieve strong performance or struggle to generalize to unseen speakers. In this study, we propose MeanVC, a lightweight and streaming zero-shot VC approach. MeanVC introduces a diffusion transformer with a chunk-wise autoregressive denoising strategy, combining the strengths of both AR and NAR paradigms for efficient streaming processing. By introducing mean flows, MeanVC regresses the average velocity field during training, enabling zero-shot VC with superior speech quality and speaker similarity in a single sampling step by directly mapping from the start to the endpoint of the flow trajectory. Additionally, we incorporate diffusion adversarial post-training to mitigate over-smoothing and further enhance speech quality. Experimental results demonstrate that MeanVC significantly outperforms existing zero-shot streaming VC systems, achieving superior conversion quality with higher efficiency and significantly fewer parameters. Audio demos and code are publicly available at this https URL.
#### LASER: An LLM-based ASR Scoring and Evaluation Rubric
 - **Authors:** Amruta Parulekar, Preethi Jyothi
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Artificial Intelligence (cs.AI); Machine Learning (cs.LG); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2510.07437

 - **Pdf link:** https://arxiv.org/pdf/2510.07437

 - **Abstract**
 Standard ASR evaluation metrics like Word Error Rate (WER) tend to unfairly penalize morphological and syntactic nuances that do not significantly alter sentence semantics. We introduce an LLM-based scoring rubric LASER that leverages state-of-the-art LLMs' in-context learning abilities to learn from prompts with detailed examples. Hindi LASER scores using Gemini 2.5 Pro achieved a very high correlation score of 94% with human annotations. Hindi examples in the prompt were also effective in analyzing errors in other Indian languages such as Marathi, Kannada and Malayalam. We also demonstrate how a smaller LLM like Llama 3 can be finetuned on word-pair examples derived from reference and ASR predictions to predict what kind of penalty should be applied with close to 89% accuracy.
#### Can Speech LLMs Think while Listening?
 - **Authors:** Yi-Jen Shih, Desh Raj, Chunyang Wu, Wei Zhou, SK Bong, Yashesh Gaur, Jay Mahadeokar, Ozlem Kalinli, Mike Seltzer
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Artificial Intelligence (cs.AI); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2510.07497

 - **Pdf link:** https://arxiv.org/pdf/2510.07497

 - **Abstract**
 Recent advances in speech large language models (speech LLMs) have enabled seamless spoken interactions, but these systems still struggle with complex reasoning tasks. Previously, chain-of-thought (CoT) prompting or fine-tuning has been to shown to significantly improve the reasoning abilities of text-based LLMs. In this work, we investigate the effect of CoT fine-tuning for multi-stream speech LLMs, demonstrating that reasoning in text space improves the accuracy of speech LLMs by 2.4x, on average, over a suite of spoken reasoning tasks. Beyond accuracy, the latency of the spoken response is a crucial factor for interacting with voice-based agents. Inspired by the human behavior of "thinking while listening," we propose methods to reduce the additional latency from reasoning by allowing the model to start reasoning before the user query has ended. To achieve this, we introduce an entropy-based metric, "question completeness," which acts as an indicator to guide the model on the optimal time to start reasoning. This method provides greater control over the accuracy-latency trade-off compared with heuristic-based approaches and, under equivalent latency conditions, yields a 4% accuracy gain on ARC-Easy. Finally, we use Direct Preference Optimization (DPO) on preference data created using rejection sampling to push the accuracy-latency pareto frontier further, resulting in a 70% reduction in latency without loss in accuracy.
#### Leveraging Whisper Embeddings for Audio-based Lyrics Matching
 - **Authors:** Eleonora Mancini, Joan Serrà, Paolo Torroni, Yuki Mitsufuji
 - **Subjects:** Subjects:
Sound (cs.SD); Artificial Intelligence (cs.AI); Machine Learning (cs.LG); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2510.08176

 - **Pdf link:** https://arxiv.org/pdf/2510.08176

 - **Abstract**
 Audio-based lyrics matching can be an appealing alternative to other content-based retrieval approaches, but existing methods often suffer from limited reproducibility and inconsistent baselines. In this work, we introduce WEALY, a fully reproducible pipeline that leverages Whisper decoder embeddings for lyrics matching tasks. WEALY establishes robust and transparent baselines, while also exploring multimodal extensions that integrate textual and acoustic features. Through extensive experiments on standard datasets, we demonstrate that WEALY achieves a performance comparable to state-of-the-art methods that lack reproducibility. In addition, we provide ablation studies and analyses on language robustness, loss functions, and embedding strategies. This work contributes a reliable benchmark for future research, and underscores the potential of speech technologies for music information retrieval tasks.


by Zyzzyva0381 (Windy). 


2025-10-10
