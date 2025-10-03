# Showing new listings for Friday, 3 October 2025
Auto update papers at about 2:30am UTC (10:30am Beijing time) every weekday.


阅读 `Usage.md`了解如何使用此repo实现个性化的Arxiv论文推送

See `Usage.md` for instructions on how to personalize the repo. 


Keyword list: ['text-to-speech', 'text to speech', 'tts', 'LLM-based', 'speech', 'voice']


Excluded: []


### Today: 13papers 
#### SLAP: Learning Speaker and Health-Related Representations from Natural Language Supervision
 - **Authors:** Angelika Ando, Auguste Crabeil, Adrien Lesage, Rachid Riad
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2510.01860

 - **Pdf link:** https://arxiv.org/pdf/2510.01860

 - **Abstract**
 Speech encodes paralinguistic information such as demographics, voice quality, and health. Yet no audio foundation model supports zero-shot or out-of-distribution (OOD) generalization to these tasks. We introduce SLAP (Speaker contrastive Language-Audio Pretraining), the first model aligning speech with natural language descriptions of speaker and health metadata through contrastive learning. SLAP combines a Vision Transformer audio encoder with text encoders, trained on more than 3400 hours across 9 datasets with diverse speaker annotations. We evaluated on 38 binary classification tasks spanning demographics, voice characteristics, and clinical assessments across 14 datasets in 7 languages. SLAP achieves 62.9% average F1 in zero-shot evaluation, a 48% relative improvement over CLAP (42.4%), while demonstrating strong OOD generalization to unseen languages and clinical populations. When fine-tuned with linear probing, SLAP reaches 69.3% F1 overall and achieves best-in-class performance on health tasks (57.9% F1), surpassing larger foundation models.
#### Do Bias Benchmarks Generalise? Evidence from Voice-based Evaluation of Gender Bias in SpeechLLMs
 - **Authors:** Shree Harsha Bokkahalli Satish, Gustav Eje Henter, Éva Székely
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Artificial Intelligence (cs.AI); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2510.01254

 - **Pdf link:** https://arxiv.org/pdf/2510.01254

 - **Abstract**
 Recent work in benchmarking bias and fairness in speech large language models (SpeechLLMs) has relied heavily on multiple-choice question answering (MCQA) formats. The model is tasked to choose between stereotypical, anti-stereotypical, or neutral/irrelevant answers given an input speech prompt and an optional text prompt. Such MCQA benchmarks implicitly assume that model performance is consistent across other MCQA tasks, voices, and other task formats such as more realistic, long-form evaluations. In this paper, we probe that assumption. We fine-tune three SpeechLLMs using LoRA adapters to induce specific MCQA behaviours: preference for stereotypical, anti-stereotypical, or neutral/uncertain answers. We then evaluate whether these behaviours generalise to another, distinct MCQA benchmark, and more critically to long-form, creative generation tasks. Our results show that performance on MCQA bias benchmarks fails to reliably predict performances across other MCQA benchmarks, and more importantly across long-form tasks. We conclude that current MCQA bias benchmarks show limited evidence of cross-task generalisation in the speech domain, and also propose an evaluation suite for measuring behaviour transferability in future models and benchmarks.
#### Ovi: Twin Backbone Cross-Modal Fusion for Audio-Video Generation
 - **Authors:** Chetwin Low, Weimin Wang, Calder Katyal
 - **Subjects:** Subjects:
Multimedia (cs.MM); Computer Vision and Pattern Recognition (cs.CV); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2510.01284

 - **Pdf link:** https://arxiv.org/pdf/2510.01284

 - **Abstract**
 Audio-video generation has often relied on complex multi-stage architectures or sequential synthesis of sound and visuals. We introduce Ovi, a unified paradigm for audio-video generation that models the two modalities as a single generative process. By using blockwise cross-modal fusion of twin-DiT modules, Ovi achieves natural synchronization and removes the need for separate pipelines or post hoc alignment. To facilitate fine-grained multimodal fusion modeling, we initialize an audio tower with an architecture identical to that of a strong pretrained video model. Trained from scratch on hundreds of thousands of hours of raw audio, the audio tower learns to generate realistic sound effects, as well as speech that conveys rich speaker identity and emotion. Fusion is obtained by jointly training the identical video and audio towers via blockwise exchange of timing (via scaled-RoPE embeddings) and semantics (through bidirectional cross-attention) on a vast video corpus. Our model enables cinematic storytelling with natural speech and accurate, context-matched sound effects, producing movie-grade video clips. All the demos, code and model weights are published at this https URL
#### RealClass: A Framework for Classroom Speech Simulation with Public Datasets and Game Engines
 - **Authors:** Ahmed Adel Attia, Jing Liu, Carol Espy Wilson
 - **Subjects:** Subjects:
Sound (cs.SD); Artificial Intelligence (cs.AI); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2510.01462

 - **Pdf link:** https://arxiv.org/pdf/2510.01462

 - **Abstract**
 The scarcity of large-scale classroom speech data has hindered the development of AI-driven speech models for education. Classroom datasets remain limited and not publicly available, and the absence of dedicated classroom noise or Room Impulse Response (RIR) corpora prevents the use of standard data augmentation techniques. In this paper, we introduce a scalable methodology for synthesizing classroom noise and RIRs using game engines, a versatile framework that can extend to other domains beyond the classroom. Building on this methodology, we present RealClass, a dataset that combines a synthesized classroom noise corpus with a classroom speech dataset compiled from publicly available corpora. The speech data pairs a children's speech corpus with instructional speech extracted from YouTube videos to approximate real classroom interactions in clean conditions. Experiments on clean and noisy speech show that RealClass closely approximates real classroom speech, making it a valuable asset in the absence of abundant real classroom speech.
#### TalkPlay-Tools: Conversational Music Recommendation with LLM Tool Calling
 - **Authors:** Seungheon Doh, Keunwoo Choi, Juhan Nam
 - **Subjects:** Subjects:
Information Retrieval (cs.IR); Multimedia (cs.MM); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2510.01698

 - **Pdf link:** https://arxiv.org/pdf/2510.01698

 - **Abstract**
 While the recent developments in large language models (LLMs) have successfully enabled generative recommenders with natural language interactions, their recommendation behavior is limited, leaving other simpler yet crucial components such as metadata or attribute filtering underutilized in the system. We propose an LLM-based music recommendation system with tool calling to serve as a unified retrieval-reranking pipeline. Our system positions an LLM as an end-to-end recommendation system that interprets user intent, plans tool invocations, and orchestrates specialized components: boolean filters (SQL), sparse retrieval (BM25), dense retrieval (embedding similarity), and generative retrieval (semantic IDs). Through tool planning, the system predicts which types of tools to use, their execution order, and the arguments needed to find music matching user preferences, supporting diverse modalities while seamlessly integrating multiple database filtering methods. We demonstrate that this unified tool-calling framework achieves competitive performance across diverse recommendation scenarios by selectively employing appropriate retrieval methods based on user queries, envisioning a new paradigm for conversational music recommendation systems.
#### Emotional Text-To-Speech Based on Mutual-Information-Guided Emotion-Timbre Disentanglement
 - **Authors:** Jianing Yang, Sheng Li, Takahiro Shinozaki, Yuki Saito, Hiroshi Saruwatari
 - **Subjects:** Subjects:
Sound (cs.SD); Artificial Intelligence (cs.AI); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2510.01722

 - **Pdf link:** https://arxiv.org/pdf/2510.01722

 - **Abstract**
 Current emotional Text-To-Speech (TTS) and style transfer methods rely on reference encoders to control global style or emotion vectors, but do not capture nuanced acoustic details of the reference speech. To this end, we propose a novel emotional TTS method that enables fine-grained phoneme-level emotion embedding prediction while disentangling intrinsic attributes of the reference speech. The proposed method employs a style disentanglement method to guide two feature extractors, reducing mutual information between timbre and emotion features, and effectively separating distinct style components from the reference speech. Experimental results demonstrate that our method outperforms baseline TTS systems in generating natural and emotionally rich speech. This work highlights the potential of disentangled and fine-grained representations in advancing the quality and flexibility of emotional TTS systems.
#### SingMOS-Pro: An Comprehensive Benchmark for Singing Quality Assessment
 - **Authors:** Yuxun Tang, Lan Liu, Wenhao Feng, Yiwen Zhao, Jionghao Han, Yifeng Yu, Jiatong Shi, Qin Jin
 - **Subjects:** Subjects:
Sound (cs.SD); Artificial Intelligence (cs.AI); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2510.01812

 - **Pdf link:** https://arxiv.org/pdf/2510.01812

 - **Abstract**
 Singing voice generation progresses rapidly, yet evaluating singing quality remains a critical challenge. Human subjective assessment, typically in the form of listening tests, is costly and time consuming, while existing objective metrics capture only limited perceptual aspects. In this work, we introduce SingMOS-Pro, a dataset for automatic singing quality assessment. Building on our preview version SingMOS, which provides only overall ratings, SingMOS-Pro expands annotations of the additional part to include lyrics, melody, and overall quality, offering broader coverage and greater diversity. The dataset contains 7,981 singing clips generated by 41 models across 12 datasets, spanning from early systems to recent advances. Each clip receives at least five ratings from professional annotators, ensuring reliability and consistency. Furthermore, we explore how to effectively utilize MOS data annotated under different standards and benchmark several widely used evaluation methods from related tasks on SingMOS-Pro, establishing strong baselines and practical references for future research. The dataset can be accessed at this https URL.
#### MelCap: A Unified Single-Codebook Neural Codec for High-Fidelity Audio Compression
 - **Authors:** Jingyi Li, Zhiyuan Zhao, Yunfei Liu, Lijian Lin, Ye Zhu, Jiahao Wu, Qiuqiang Kong, Yu Li
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2510.01903

 - **Pdf link:** https://arxiv.org/pdf/2510.01903

 - **Abstract**
 Neural audio codecs have recently emerged as powerful tools for high-quality and low-bitrate audio compression, leveraging deep generative models to learn latent representations of audio signals. However, existing approaches either rely on a single quantizer that only processes speech domain, or on multiple quantizers that are not well suited for downstream tasks. To address this issue, we propose MelCap, a unified "one-codebook-for-all" neural codec that effectively handles speech, music, and general sound. By decomposing audio reconstruction into two stages, our method preserves more acoustic details than previous single-codebook approaches, while achieving performance comparable to mainstream multi-codebook methods. In the first stage, audio is transformed into mel-spectrograms, which are compressed and quantized into compact single tokens using a 2D tokenizer. A perceptual loss is further applied to mitigate the over-smoothing artifacts observed in spectrogram reconstruction. In the second stage, a Vocoder recovers waveforms from the mel discrete tokens in a single forward pass, enabling real-time decoding. Both objective and subjective evaluations demonstrate that MelCap achieves quality on comparable to state-of-the-art multi-codebook codecs, while retaining the computational simplicity of a single-codebook design, thereby providing an effective representation for downstream tasks.
#### Exploring Resolution-Wise Shared Attention in Hybrid Mamba-U-Nets for Improved Cross-Corpus Speech Enhancement
 - **Authors:** Nikolai Lund Kühne, Jesper Jensen, Jan Østergaard, Zheng-Hua Tan
 - **Subjects:** Subjects:
Sound (cs.SD); Artificial Intelligence (cs.AI); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2510.01958

 - **Pdf link:** https://arxiv.org/pdf/2510.01958

 - **Abstract**
 Recent advances in speech enhancement have shown that models combining Mamba and attention mechanisms yield superior cross-corpus generalization performance. At the same time, integrating Mamba in a U-Net structure has yielded state-of-the-art enhancement performance, while reducing both model size and computational complexity. Inspired by these insights, we propose RWSA-MambaUNet, a novel and efficient hybrid model combining Mamba and multi-head attention in a U-Net structure for improved cross-corpus performance. Resolution-wise shared attention (RWSA) refers to layerwise attention-sharing across corresponding time- and frequency resolutions. Our best-performing RWSA-MambaUNet model achieves state-of-the-art generalization performance on two out-of-domain test sets. Notably, our smallest model surpasses all baselines on the out-of-domain DNS 2020 test set in terms of PESQ, SSNR, and ESTOI, and on the out-of-domain EARS-WHAM_v2 test set in terms of SSNR, ESTOI, and SI-SDR, while using less than half the model parameters and a fraction of the FLOPs.
#### Stream RAG: Instant and Accurate Spoken Dialogue Systems with Streaming Tool Usage
 - **Authors:** Siddhant Arora, Haidar Khan, Kai Sun, Xin Luna Dong, Sajal Choudhary, Seungwhan Moon, Xinyuan Zhang, Adithya Sagar, Surya Teja Appini, Kaushik Patnaik, Sanat Sharma, Shinji Watanabe, Anuj Kumar, Ahmed Aly, Yue Liu, Florian Metze, Zhaojiang Lin
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2510.02044

 - **Pdf link:** https://arxiv.org/pdf/2510.02044

 - **Abstract**
 End-to-end speech-in speech-out dialogue systems are emerging as a powerful alternative to traditional ASR-LLM-TTS pipelines, generating more natural, expressive responses with significantly lower latency. However, these systems remain prone to hallucinations due to limited factual grounding. While text-based dialogue systems address this challenge by integrating tools such as web search and knowledge graph APIs, we introduce the first approach to extend tool use directly into speech-in speech-out systems. A key challenge is that tool integration substantially increases response latency, disrupting conversational flow. To mitigate this, we propose Streaming Retrieval-Augmented Generation (Streaming RAG), a novel framework that reduces user-perceived latency by predicting tool queries in parallel with user speech, even before the user finishes speaking. Specifically, we develop a post-training pipeline that teaches the model when to issue tool calls during ongoing speech and how to generate spoken summaries that fuse audio queries with retrieved text results, thereby improving both accuracy and responsiveness. To evaluate our approach, we construct AudioCRAG, a benchmark created by converting queries from the publicly available CRAG dataset into speech form. Experimental results demonstrate that our streaming RAG approach increases QA accuracy by up to 200% relative (from 11.1% to 34.2% absolute) and further enhances user experience by reducing tool use latency by 20%. Importantly, our streaming RAG approach is modality-agnostic and can be applied equally to typed input, paving the way for more agentic, real-time AI assistants.
#### Chain-of-Thought Reasoning in Streaming Full-Duplex End-to-End Spoken Dialogue Systems
 - **Authors:** Siddhant Arora, Jinchuan Tian, Hayato Futami, Jiatong Shi, Yosuke Kashiwagi, Emiru Tsunoo, Shinji Watanabe
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2510.02066

 - **Pdf link:** https://arxiv.org/pdf/2510.02066

 - **Abstract**
 Most end-to-end (E2E) spoken dialogue systems (SDS) rely on voice activity detection (VAD) for turn-taking, but VAD fails to distinguish between pauses and turn completions. Duplex SDS models address this by predicting output continuously, including silence tokens, thus removing the need for explicit VAD. However, they often have complex dual-channel architecture and lag behind cascaded models in semantic reasoning. To overcome these challenges, we propose SCoT: a Streaming Chain-of-Thought (CoT) framework for Duplex SDS, alternating between processing fixed-duration user input and generating responses in a blockwise manner. Using frame-level alignments, we create intermediate targets-aligned user transcripts and system responses for each block. Experiments show that our approach produces more coherent and interpretable responses than existing duplex methods while supporting lower-latency and overlapping interactions compared to turn-by-turn systems.
#### EvolveCaptions: Empowering DHH Users Through Real-Time Collaborative Captioning
 - **Authors:** Liang-Yuan Wu, Dhruv Jain
 - **Subjects:** Subjects:
Human-Computer Interaction (cs.HC); Artificial Intelligence (cs.AI); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2510.02181

 - **Pdf link:** https://arxiv.org/pdf/2510.02181

 - **Abstract**
 Automatic Speech Recognition (ASR) systems often fail to accurately transcribe speech from Deaf and Hard of Hearing (DHH) individuals, especially during real-time conversations. Existing personalization approaches typically require extensive pre-recorded data and place the burden of adaptation on the DHH speaker. We present EvolveCaptions, a real-time, collaborative ASR adaptation system that supports in-situ personalization with minimal effort. Hearing participants correct ASR errors during live conversations. Based on these corrections, the system generates short, phonetically targeted prompts for the DHH speaker to record, which are then used to fine-tune the ASR model. In a study with 12 DHH and six hearing participants, EvolveCaptions reduced Word Error Rate (WER) across all DHH users within one hour of use, using only five minutes of recording time on average. Participants described the system as intuitive, low-effort, and well-integrated into communication. These findings demonstrate the promise of collaborative, real-time ASR adaptation for more equitable communication.
#### High-Fidelity Speech Enhancement via Discrete Audio Tokens
 - **Authors:** Luca A. Lanzendörfer, Frédéric Berdoz, Antonis Asonitis, Roger Wattenhofer
 - **Subjects:** Subjects:
Sound (cs.SD); Machine Learning (cs.LG); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2510.02187

 - **Pdf link:** https://arxiv.org/pdf/2510.02187

 - **Abstract**
 Recent autoregressive transformer-based speech enhancement (SE) methods have shown promising results by leveraging advanced semantic understanding and contextual modeling of speech. However, these approaches often rely on complex multi-stage pipelines and low sampling rate codecs, limiting them to narrow and task-specific speech enhancement. In this work, we introduce DAC-SE1, a simplified language model-based SE framework leveraging discrete high-resolution audio representations; DAC-SE1 preserves fine-grained acoustic details while maintaining semantic coherence. Our experiments show that DAC-SE1 surpasses state-of-the-art autoregressive SE methods on both objective perceptual metrics and in a MUSHRA human evaluation. We release our codebase and model checkpoints to support further research in scalable, unified, and high-quality speech enhancement.


by Zyzzyva0381 (Windy). 


2025-10-03
