# Showing new listings for Tuesday, 10 March 2026
Auto update papers at about 2:30am UTC (10:30am Beijing time) every weekday.


阅读 `Usage.md`了解如何使用此repo实现个性化的Arxiv论文推送

See `Usage.md` for instructions on how to personalize the repo. 


Keyword list: ['text-to-speech', 'text to speech', 'tts', 'LLM-based', 'speech', 'voice']


Excluded: []


### Today: 15papers 
#### Towards Lightweight Adaptation of Speech Enhancement Models in Real-World Environments
 - **Authors:** Longbiao Cheng, Shih-Chii Liu
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Artificial Intelligence (cs.AI); Machine Learning (cs.LG); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2603.07471

 - **Pdf link:** https://arxiv.org/pdf/2603.07471

 - **Abstract**
 Recent studies have shown that post-deployment adaptation can improve the robustness of speech enhancement models in unseen noise conditions. However, existing methods often incur prohibitive computational and memory costs, limiting their suitability for on-device deployment. In this work, we investigate model adaptation in realistic settings with dynamic acoustic scene changes and propose a lightweight framework that augments a frozen backbone with low-rank adapters updated via self-supervised training. Experiments on sequential scene evaluations spanning 111 environments across 37 noise types and three signal-to-noise ratio ranges, including the challenging [-8, 0] dB range, show that our method updates fewer than 1% of the base model's parameters while achieving an average 1.51 dB SI-SDR improvement within only 20 updates per scene. Compared to state-of-the-art approaches, our framework achieves competitive or superior perceptual quality with smoother and more stable convergence, demonstrating its practicality for lightweight on-device adaptation of speech enhancement models under real-world acoustic conditions.
#### Multi-View Based Audio Visual Target Speaker Extraction
 - **Authors:** Peijun Yang, Zhan Jin, Juan Liu, Ming Li
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2603.07696

 - **Pdf link:** https://arxiv.org/pdf/2603.07696

 - **Abstract**
 Audio-Visual Target Speaker Extraction (AVTSE) aims to separate a target speaker's voice from a mixed audio signal using the corresponding visual cues. While most existing AVTSE methods rely exclusively on frontal-view videos, this limitation restricts their robustness in real-world scenarios where non-frontal views are prevalent. Such visual perspectives often contain complementary articulatory information that could enhance speech extraction. In this work, we propose Multi-View Tensor Fusion (MVTF), a novel framework that transforms multi-view learning into single-view performance gains. During the training stage, we leverage synchronized multi-perspective lip videos to learn cross-view correlations through MVTF, where pairwise outer products explicitly model multiplicative interactions between different views of input lip embeddings. At the inference stage, the system supports both single-view and multi-view inputs. Experimental results show that in the single-view inputs, our framework leverages multi-view knowledge to achieve significant performance gains, while in the multi-view mode, it further improves overall performance and enhances the robustness. Our demo, code and data are available at this https URL
#### Language-Invariant Multilingual Speaker Verification for the TidyVoice 2026 Challenge
 - **Authors:** Ze Li, Xiaoxiao Miao, Juan Liu, Ming Li
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2603.08092

 - **Pdf link:** https://arxiv.org/pdf/2603.08092

 - **Abstract**
 Multilingual speaker verification (SV) remains challenging due to limited cross-lingual data and language-dependent information in speaker embeddings. This paper presents a language-invariant multilingual SV system for the TidyVoice 2026 Challenge. We adopt the multilingual self-supervised w2v-BERT 2.0 model as the backbone, enhanced with Layer Adapters and Multi-scale Feature Aggregation to better exploit multi-layer representations. A language-adversarial training strategy with a Gradient Reversal Layer is applied to promote language-invariant speaker embeddings. Moreover, a multilingual zero-shot text-to-speech system is used to synthesize speech in multiple languages, improving language diversity. Experimental results demonstrate that fine-tuning the large-scale pretrained model yields competitive performance, while language-adversarial training further enhances robustness. In addition, synthetic speech augmentation provides additional gains under limited training data conditions. Source code is available at this https URL.
#### Privacy-Preserving End-to-End Full-Duplex Speech Dialogue Models
 - **Authors:** Nikita Kuzmin, Tao Zhong, Jiajun Deng, Yingke Zhu, Tristan Tsoi, Tianxiang Cao, Simon Lui, Kong Aik Lee, Eng Siong Chng
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Artificial Intelligence (cs.AI); Signal Processing (eess.SP)
 - **Arxiv link:** https://arxiv.org/abs/2603.08179

 - **Pdf link:** https://arxiv.org/pdf/2603.08179

 - **Abstract**
 End-to-end full-duplex speech models feed user audio through an always-on LLM backbone, yet the speaker privacy implications of their hidden representations remain unexamined. Following the VoicePrivacy 2024 protocol with a lazy-informed attacker, we show that the hidden states of SALM-Duplex and Moshi leak substantial speaker identity across all transformer layers. Layer-wise and turn-wise analyses reveal that leakage persists across all layers, with SALM-Duplex showing stronger leakage in early layers while Moshi leaks uniformly, and that Linkability rises sharply within the first few turns. We propose two streaming anonymization setups using Stream-Voice-Anon: a waveform-level front-end (Anon-W2W) and a feature-domain replacement (Anon-W2F). Anon-W2F raises EER by over 3.5x relative to the discrete encoder baseline (11.2% to 41.0%), approaching the 50% random-chance ceiling, while Anon-W2W retains 78-93% of baseline sBERT across setups with sub-second response latency (FRL under 0.8 s).
#### DualTurn: Learning Turn-Taking from Dual-Channel Generative Speech Pretraining
 - **Authors:** Shangeth Rajaa
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Computation and Language (cs.CL); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2603.08216

 - **Pdf link:** https://arxiv.org/pdf/2603.08216

 - **Abstract**
 Speech-to-speech models handle turn-taking naturally but offer limited support for tool-calling or complex reasoning, while production ASR-LLM-TTS voice pipelines offer these capabilities but rely on silence timeouts, which lead to unnatural turn-taking. We present DualTurn, which narrows this gap through generative pretraining on dual-channel conversational audio. The model generates both speakers' future audio autoregressively, implicitly learning conversational dynamics without any labels, and is then fine-tuned to predict interpretable turn-taking signals that map directly to agent actions. DualTurn monitors both channels continuously, anticipating turn boundaries and producing five agent actions. On standard benchmarks, DualTurn (0.5B) outperforms both VAP on agent action prediction (wF1 0.633 vs. 0.389) and a 3.1B audio-text model on word-level turn prediction (AUC 0.930 vs. 0.880), while anticipating turn boundaries earlier with fewer interruptions.
#### Quantifying Cross-Lingual Transfer in Paralinguistic Speech Tasks
 - **Authors:** Pol Buitrago, Oriol Pareras, Federico Costa, Javier Hernando
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Computation and Language (cs.CL)
 - **Arxiv link:** https://arxiv.org/abs/2603.08231

 - **Pdf link:** https://arxiv.org/pdf/2603.08231

 - **Abstract**
 Paralinguistic speech tasks are often considered relatively language-agnostic, as they rely on extralinguistic acoustic cues rather than lexical content. However, prior studies report performance degradation under cross-lingual conditions, indicating non-negligible language dependence. Still, these studies typically focus on isolated language pairs or task-specific settings, limiting comparability and preventing a systematic assessment of task-level language dependence. We introduce the Cross-Lingual Transfer Matrix (CLTM), a systematic method to quantify cross-lingual interactions between pairs of languages within a given task. We apply the CLTM to two paralinguistic tasks, gender identification and speaker verification, using a multilingual HuBERT-based encoder, to analyze how donor-language data affects target-language performance during fine-tuning. Our results reveal distinct transfer patterns across tasks and languages, reflecting systematic, language-dependent effects.
#### Bootstrapping Audiovisual Speech Recognition in Zero-AV-Resource Scenarios with Synthetic Visual Data
 - **Authors:** Pol Buitrago, Pol Gàlvez, Oriol Pareras, Javier Hernando
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Computation and Language (cs.CL); Image and Video Processing (eess.IV)
 - **Arxiv link:** https://arxiv.org/abs/2603.08249

 - **Pdf link:** https://arxiv.org/pdf/2603.08249

 - **Abstract**
 Audiovisual speech recognition (AVSR) combines acoustic and visual cues to improve transcription robustness under challenging conditions but remains out of reach for most under-resourced languages due to the lack of labeled video corpora for training. We propose a zero-AV-resource AVSR framework that relies on synthetic visual streams generated by lip-syncing static facial images with real audio. We first evaluate synthetic visual augmentation on Spanish benchmarks, then apply it to Catalan, a language with no annotated audiovisual corpora. We synthesize over 700 hours of talking-head video and fine-tune a pre-trained AV-HuBERT model. On a manually annotated Catalan benchmark, our model achieves near state-of-the-art performance with much fewer parameters and training data, outperforms an identically trained audio-only baseline, and preserves multimodal advantages in noise. Scalable synthetic video thus offers a viable substitute for real recordings in zero-AV-resource AVSR.
#### NLE: Non-autoregressive LLM-based ASR by Transcript Editing
 - **Authors:** Avihu Dekel, Samuel Thomas, Takashi Fukada, George Saon
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2603.08397

 - **Pdf link:** https://arxiv.org/pdf/2603.08397

 - **Abstract**
 While autoregressive (AR) LLM-based ASR systems achieve strong accuracy, their sequential decoding limits parallelism and incurs high latency. We propose NLE, a non-autoregressive (NAR) approach that formulates speech recognition as conditional transcript editing, enabling fully parallel prediction. NLE extracts acoustic embeddings and an initial hypothesis from a pretrained speech encoder, then refines the hypothesis using a bidirectional LLM editor trained with a latent alignment objective. An interleaved padding strategy exploits the identity mapping bias of Transformers, allowing the model to focus on corrections rather than full reconstruction. On the Open ASR leaderboard, NLE++ achieves 5.67% average WER with an RTFx (inverse real-time factor) of 1630. In single-utterance scenarios, NLE achieves 27x speedup over the AR baseline, making it suitable for real-time applications.
#### Scaling Self-Supervised Speech Models Uncovers Deep Linguistic Relationships: Evidence from the Pacific Cluster
 - **Authors:** Minu Kim, Hoirin Kim, David R. Mortensen
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2603.07238

 - **Pdf link:** https://arxiv.org/pdf/2603.07238

 - **Abstract**
 Similarities between language representations derived from Self-Supervised Speech Models (S3Ms) have been observed to primarily reflect geographic proximity or surface typological similarities driven by recent expansion or contact, potentially missing deeper genealogical signals. We investigate how scaling linguistic coverage of an S3M-based language identification system from 126 to 4,017 languages influences this topology. Our results reveal a non-linear effect: while phylogenetic recovery remains stagnant up to the 1K scale, the 4K model displays a dramatic qualitative shift, resolving both clear lineages and complex, long-term linguistic contact. Notably, our analysis reveals the emergence of a robust macro-cluster in the Pacific (comprising Papuan, Oceanic, and Australian languages) and investigates its latent drivers. We find that the 4K model utilizes a more concentrated encoding that captures shared, robust acoustic signatures such as global energy dynamics. These findings suggest that massive S3Ms can internalize multiple layers of language history, providing a promising perspective for computational phylogenetics and the study of language contact.
#### Seeing the Context: Rich Visual Context-Aware Speech Recognition via Multimodal Reasoning
 - **Authors:** Wenjie Tian, Mingchen Shao, Bingshen Mu, Xuelong Geng, Chengyou Wang, Yujie Liao, Zhixian Zhao, Ziyu Zhang, Jingbin Hu, Mengqi Wei, Lei Xie
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2603.07263

 - **Pdf link:** https://arxiv.org/pdf/2603.07263

 - **Abstract**
 Audio-visual speech recognition (AVSR) is an extension of ASR that incorporates visual signals. Current AVSR approaches primarily focus on lip motion, largely overlooking rich context present in the video such as speaking scene and on-screen text. To tackle such CAVSR (AVSR including rich visual Context), we propose VASR designed to "see" and reason the visual context to improve speech recognition. Specifically, we construct an Audio-Visual Chain-of-Thought (AV-CoT) that explicitly enforces intermediate cross-modal grounding between acoustic signals and visual evidence. This evidence-driven reasoning mitigates the "single-modality dominance" problem, where models either over-rely on visual context or fail to utilize it. Besides, to address the data scarcity, we construct and release a corresponding data pipeline and test set. Experiments show that AV-CoT effectively mitigates the single-modality dominance, achieving state-of-the-art performance in CAVSR. The project is open-sourced.
#### Evaluating Parkinson's Disease Detection in Anonymized Speech: A Performance and Acoustic Analysis
 - **Authors:** Carlos Franzreb, Francisco Teixeira, Ben Luks, Sebastian Möller, Alberto Abad
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2603.07544

 - **Pdf link:** https://arxiv.org/pdf/2603.07544

 - **Abstract**
 Automatic detection of Parkinson's disease (PD) from speech is a promising non-invasive diagnostic tool, but it raises significant privacy concerns. Speaker anonymization mitigates these risks, but it may suppress the pathological information necessary for PD detection. We assess the trade-off between privacy and PD detection for two anonymizers (STT-TTS and kNN-VC) using two Spanish datasets. STT-TTS provides better privacy but severely degrades PD detection by eradicating prosodic information. kNN-VC preserves macro-prosodic features such as duration and F0 contours, achieving F1 scores only 3-7\% lower than original baselines, demonstrating that privacy-preserving PD detection is viable when using appropriate anonymization. Finally, an acoustic distortion analysis characterizes specific weaknesses in kNN-VC, offering insights for designing anonymizers that better preserve PD information.
#### WhispEar: A Bi-directional Framework for Scaling Whispered Speech Conversion via Pseudo-Parallel Whisper Generation
 - **Authors:** Zihao Fang, Yingda Shen, Zifan Guan, Tongtong Song, Zhenyi Liu, Zhizheng Wu
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2603.08046

 - **Pdf link:** https://arxiv.org/pdf/2603.08046

 - **Abstract**
 Whispered speech lacks vocal fold vibration and fundamental frequency, resulting in degraded acoustic cues and making whisper-to-normal (W2N) conversion challenging, especially with limited parallel data. We propose WhispEar, a bidirectional framework based on unified semantic representations that capture speaking-mode-invariant information shared by whispered and normal speech. The framework contains both W2N and normal-to-whisper (N2W) models. Notably, the N2W model enables zero-shot pseudo-parallel whisper generation from abundant normal speech, allowing scalable data augmentation for W2N training. Increasing generated data consistently improves performance. We also release the largest bilingual (Chinese-English) whispered-normal parallel corpus to date. Experiments demonstrate that WhispEar outperforms strong baselines and benefits significantly from scalable pseudo-parallel data.
#### Disentangling Reasoning in Large Audio-Language Models for Ambiguous Emotion Prediction
 - **Authors:** Xiaofeng Yu, Jiaheng Dong, Jean Honorio, Abhirup Ghosh, Hong Jia, Ting Dang
 - **Subjects:** Subjects:
Sound (cs.SD); Artificial Intelligence (cs.AI); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2603.08230

 - **Pdf link:** https://arxiv.org/pdf/2603.08230

 - **Abstract**
 Speech emotion recognition plays an important role in various applications. However, most existing approaches predict a single emotion label, oversimplifying the inherently ambiguous nature of human emotional expression. Recent large audio-language models show promise in generating richer outputs, but their reasoning ability for ambiguous emotional understanding remains limited. In this work, we reformulate ambiguous emotion recognition as a distributional reasoning problem and present the first systematic study of ambiguity-aware reasoning in LALMs. Our framework comprises two complementary components: an ambiguity-aware objective that aligns predictions with human perceptual distributions, and a structured ambiguity-aware chain-of-thought supervision that guides reasoning over emotional cues. Experiments on IEMOCAP and CREMA-D demonstrate consistent improvements across SFT, DPO, and GRPO training strategies.
#### Computational modeling of early language learning from acoustic speech and audiovisual input without linguistic priors
 - **Authors:** Okko Räsänen
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Artificial Intelligence (cs.AI); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2603.08359

 - **Pdf link:** https://arxiv.org/pdf/2603.08359

 - **Abstract**
 Learning to understand speech appears almost effortless for typically developing infants, yet from an information-processing perspective, acquiring a language from acoustic speech is an enormous challenge. This chapter reviews recent developments in using computational models to understand early language acquisition from speech and audiovisual input. The focus is on self-supervised and visually grounded models of perceptual learning. We show how these models are becoming increasingly powerful in learning various aspects of speech without strong linguistic priors, and how many features of early language development can be explained through a shared set of learning principles-principles broadly compatible with multiple theories of language acquisition and human cognition. We also discuss how modern learning simulations are gradually becoming more realistic, both in terms of input data and in linking model behavior to empirical findings on infant language development.
#### Benchmarking Language Modeling for Lossless Compression of Full-Fidelity Audio
 - **Authors:** Phillip Long, Zachary Novack, Chris Donahue
 - **Subjects:** Subjects:
Sound (cs.SD); Artificial Intelligence (cs.AI); Machine Learning (cs.LG); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2603.08683

 - **Pdf link:** https://arxiv.org/pdf/2603.08683

 - **Abstract**
 Autoregressive "language" models (LMs) trained on raw waveforms can be repurposed for lossless audio compression, but prior work is limited to 8-bit audio, leaving open whether such approaches work for practical settings (16/24-bit) and can compete with existing codecs. We benchmark LM-based compression on full-fidelity audio across diverse domains (music, speech, bioacoustics), sampling rates (16kHz-48kHz), and bit depths (8, 16, 24-bit). Standard sample-level tokenization becomes intractable at higher bit depths due to vocabulary size (65K for 16-bit; 16.7M for 24-bit). We propose Trilobyte, a byte-level tokenization schema for full resolution audio, improving vocabulary scaling from $O(2^{b})$ to $O(1)$ and enabling the first tractable 24-bit LM-based lossless compression. While LMs consistently outperform FLAC and yield state-of-the-art compression at 8-bit and 16-bit, we observe that compression gains become more modest as bit depth increases beyond 8-bit.


by Zyzzyva0381 (Windy). 


2026-03-10
