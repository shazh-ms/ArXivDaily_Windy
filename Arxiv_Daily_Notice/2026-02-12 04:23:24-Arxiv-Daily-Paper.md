# Showing new listings for Thursday, 12 February 2026
Auto update papers at about 2:30am UTC (10:30am Beijing time) every weekday.


阅读 `Usage.md`了解如何使用此repo实现个性化的Arxiv论文推送

See `Usage.md` for instructions on how to personalize the repo. 


Keyword list: ['text-to-speech', 'text to speech', 'tts', 'LLM-based', 'speech', 'voice']


Excluded: []


### Today: 9papers 
#### AudioRAG: A Challenging Benchmark for Audio Reasoning and Information Retrieval
 - **Authors:** Jingru Lin, Chen Zhang, Tianrui Wang, Haizhou Li
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2602.10656

 - **Pdf link:** https://arxiv.org/pdf/2602.10656

 - **Abstract**
 Due to recent advancements in Large Audio-Language Models (LALMs) that demonstrate remarkable performance across a range of sound-, speech- and music-related tasks, there is a growing interest in proposing benchmarks to assess these models. Existing benchmarks generally focus only on reasoning with internal knowledge, neglecting real-world scenarios that require external information grounding. To bridge this gap, we introduce AudioRAG, a novel benchmark designed to evaluate audio-based reasoning augmented by information retrieval in realistic web environments. This benchmark comprises both LLM-generated and manually curated question-answer pairs. Our evaluations reveal that even the state-of-the-art LALMs struggle to answer these questions. We therefore propose an agentic pipeline that integrates audio reasoning with retrieval-augmented generation, providing a stronger baseline for future research.
#### From Diet to Free Lunch: Estimating Auxiliary Signal Properties using Dynamic Pruning Masks in Speech Enhancement Networks
 - **Authors:** Riccardo Miccini, Clément Laroche, Tobias Piechowiak, Xenofon Fafoutis, Luca Pezzarossa
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Machine Learning (cs.LG); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2602.10666

 - **Pdf link:** https://arxiv.org/pdf/2602.10666

 - **Abstract**
 Speech Enhancement (SE) in audio devices is often supported by auxiliary modules for Voice Activity Detection (VAD), SNR estimation, or Acoustic Scene Classification to ensure robust context-aware behavior and seamless user experience. Just like SE, these tasks often employ deep learning; however, deploying additional models on-device is computationally impractical, whereas cloud-based inference would introduce additional latency and compromise privacy. Prior work on SE employed Dynamic Channel Pruning (DynCP) to reduce computation by adaptively disabling specific channels based on the current input. In this work, we investigate whether useful signal properties can be estimated from these internal pruning masks, thus removing the need for separate models. We show that simple, interpretable predictors achieve up to 93% accuracy on VAD, 84% on noise classification, and an R2 of 0.86 on F0 estimation. With binary masks, predictions reduce to weighted sums, inducing negligible overhead. Our contribution is twofold: on one hand, we examine the emergent behavior of DynCP models through the lens of downstream prediction tasks, to reveal what they are learning; on the other, we repurpose and re-propose DynCP as a holistic solution for efficient SE and simultaneous estimation of signal properties.
#### RE-LLM: Refining Empathetic Speech-LLM Responses by Integrating Emotion Nuance
 - **Authors:** Jing-Han Chen, Bo-Hao Su, Ya-Tse Wu, Chi-Chun Lee
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Computation and Language (cs.CL); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2602.10716

 - **Pdf link:** https://arxiv.org/pdf/2602.10716

 - **Abstract**
 With generative AI advancing, empathy in human-AI interaction is essential. While prior work focuses on emotional reflection, emotional exploration, key to deeper engagement, remains overlooked. Existing LLMs rely on text which captures limited emotion nuances. To address this, we propose RE-LLM, a speech-LLM integrating dimensional emotion embeddings and auxiliary learning. Experiments show statistically significant gains in empathy metrics across three datasets. RE-LLM relatively improves the Emotional Reaction score by 14.79% and 6.76% compared to text-only and speech-LLM baselines on ESD. Notably, it raises the Exploration score by 35.42% and 3.91% on IEMOCAP, 139.28% and 9.83% on ESD, and 60.95% and 22.64% on MSP-PODCAST. It also boosts unweighted accuracy by 5.4% on IEMOCAP, 2.3% on ESD, and 6.9% on MSP-PODCAST in speech emotion recognition. These results highlight the enriched emotional understanding and improved empathetic response generation of RE-LLM.
#### Self-Supervised Learning for Speaker Recognition: A study and review
 - **Authors:** Theo Lepage, Reda Dehak
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Machine Learning (cs.LG); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2602.10829

 - **Pdf link:** https://arxiv.org/pdf/2602.10829

 - **Abstract**
 Deep learning models trained in a supervised setting have revolutionized audio and speech processing. However, their performance inherently depends on the quantity of human-annotated data, making them costly to scale and prone to poor generalization under unseen conditions. To address these challenges, Self-Supervised Learning (SSL) has emerged as a promising paradigm, leveraging vast amounts of unlabeled data to learn relevant representations. The application of SSL for Automatic Speech Recognition (ASR) has been extensively studied, but research on other downstream tasks, notably Speaker Recognition (SR), remains in its early stages. This work describes major SSL instance-invariance frameworks (e.g., SimCLR, MoCo, and DINO), initially developed for computer vision, along with their adaptation to SR. Various SSL methods for SR, proposed in the literature and built upon these frameworks, are also presented. An extensive review of these approaches is then conducted: (1) the effect of the main hyperparameters of SSL frameworks is investigated; (2) the role of SSL components is studied (e.g., data-augmentation, projector, positive sampling); and (3) SSL frameworks are evaluated on SR with in-domain and out-of-domain data, using a consistent experimental setup, and a comprehensive comparison of SSL methods from the literature is provided. Specifically, DINO achieves the best downstream performance and effectively models intra-speaker variability, although it is highly sensitive to hyperparameters and training conditions, while SimCLR and MoCo provide robust alternatives that effectively capture inter-speaker variability and are less prone to collapse. This work aims to highlight recent trends and advancements, identifying current challenges in the field.
#### Emotion-Coherent Speech Data Augmentation and Self-Supervised Contrastive Style Training for Enhancing Kids's Story Speech Synthesis
 - **Authors:** Raymond Chung
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2602.10164

 - **Pdf link:** https://arxiv.org/pdf/2602.10164

 - **Abstract**
 Expressive speech synthesis requires vibrant prosody and well-timed pauses. We propose an effective strategy to augment a small dataset to train an expressive end-to-end Text-to-Speech model. We merge audios of emotionally congruent text using a text emotion recognizer, creating augmented expressive speech data. By training with two-sentence audio, our model learns natural breaks between lines. We further apply self-supervised contrastive training to improve the speaking style embedding extraction from speech. During inference, our model produces multi-sentence speech in one step, guided by the text-predicted speaking style. Evaluations showcase the effectiveness of our proposed approach when compared to a baseline model trained with consecutive two-sentence audio. Our synthesized speeches give a closer inter-sentence pause distribution to the ground truth speech. Subjective evaluations reveal our synthesized speech scored higher in naturalness and style suitability than the baseline.
#### MerkleSpeech: Public-Key Verifiable, Chunk-Localised Speech Provenance via Perceptual Fingerprints and Merkle Commitments
 - **Authors:** Tatsunori Ono
 - **Subjects:** Subjects:
Cryptography and Security (cs.CR); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2602.10166

 - **Pdf link:** https://arxiv.org/pdf/2602.10166

 - **Abstract**
 Speech provenance goes beyond detecting whether a watermark is present. Real workflows involve splicing, quoting, trimming, and platform-level transforms that may preserve some regions while altering others. Neural watermarking systems have made strides in robustness and localised detection, but most deployments produce outputs with no third-party verifiable cryptographic proof tying a time segment to an issuer-signed original. Provenance standards like C2PA adopt signed manifests and Merkle-based fragment validation, yet their bindings target encoded assets and break under re-encoding or routine processing. We propose MerkleSpeech, a system for public-key verifiable, chunk-localised speech provenance offering two tiers of assurance. The first, a robust watermark attribution layer (WM-only), survives common distribution transforms and answers "was this chunk issued by a known party?". The second, a strict cryptographic integrity layer (MSv1), verifies Merkle inclusion of the chunk's fingerprint under an issuer signature. The system computes perceptual fingerprints over short speech chunks, commits them in a Merkle tree whose root is signed with an issuer key, and embeds a compact in-band watermark payload carrying a random content identifier and chunk metadata sufficient to retrieve Merkle inclusion proofs from a repository. Once the payload is extracted, all subsequent verification steps (signature check, fingerprint recomputation, Merkle inclusion) use only public information. The result is a splice-aware timeline indicating which regions pass each tier and why any given region fails. We describe the protocol, provide pseudocode, and present experiments targeting very low false positive rates under resampling, bandpass filtering, and additive noise, informed by recent audits identifying neural codecs as a major stressor for post-hoc audio watermarks.
#### MOSS-Audio-Tokenizer: Scaling Audio Tokenizers for Future Audio Foundation Models
 - **Authors:** Yitian Gong, Kuangwei Chen, Zhaoye Fei, Xiaogui Yang, Ke Chen, Yang Wang, Kexin Huang, Mingshu Chen, Ruixiao Li, Qingyuan Cheng, Shimin Li, Xipeng Qiu
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2602.10934

 - **Pdf link:** https://arxiv.org/pdf/2602.10934

 - **Abstract**
 Discrete audio tokenizers are fundamental to empowering large language models with native audio processing and generation capabilities. Despite recent progress, existing approaches often rely on pretrained encoders, semantic distillation, or heterogeneous CNN-based architectures. These designs introduce fixed inductive biases that limit reconstruction fidelity and hinder effective scaling. In this paper, we argue that discrete audio tokenization should be learned fully end-to-end using a homogeneous and scalable architecture. To this end, we first propose CAT (Causal Audio Tokenizer with Transformer), a purely Transformer-based architecture that jointly optimizes the encoder, quantizer, and decoder from scratch for high-fidelity reconstruction. Building on the CAT architecture, we develop MOSS-Audio-Tokenizer, a large-scale audio tokenizer featuring 1.6 billion parameters, pre-trained on 3 million hours of diverse, general audio data. We show that this simple, fully end-to-end approach built from homogeneous, causal Transformer blocks scales gracefully and supports high-fidelity reconstruction across diverse audio domains. Across speech, sound, and music, MOSS-Audio-Tokenizer consistently outperforms prior codecs over a wide range of bitrates, while exhibiting predictable improvements with increased scale. Notably, leveraging the discrete tokens from our model, we develop the first purely autoregressive TTS model that surpasses prior non-autoregressive and cascaded systems. Furthermore, MOSS-Audio-Tokenizer enables competitive ASR performance without auxiliary encoders. Our findings position the CAT architecture as a unified, scalable interface for the next generation of native audio foundation models.
#### Simultaneous Speech-to-Speech Translation Without Aligned Data
 - **Authors:** Tom Labiausse, Romain Fabre, Yannick Estève, Alexandre Défossez, Neil Zeghidour
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2602.11072

 - **Pdf link:** https://arxiv.org/pdf/2602.11072

 - **Abstract**
 Simultaneous speech translation requires translating source speech into a target language in real-time while handling non-monotonic word dependencies. Traditional approaches rely on supervised training with word-level aligned data, which is difficult to collect at scale and thus depends on synthetic alignments using language-specific heuristics that are suboptimal. We propose Hibiki-Zero, which eliminates the need for word-level alignments entirely. This fundamentally simplifies the training pipeline and enables seamless scaling to diverse languages with varying grammatical structures, removing the bottleneck of designing language-specific alignment heuristics. We first train on sentence-level aligned data to learn speech translation at high latency, then apply a novel reinforcement learning strategy using GRPO to optimize latency while preserving translation quality. Hibiki-Zero achieves state-of-the-art performance in translation accuracy, latency, voice transfer, and naturalness across five X-to-English tasks. Moreover, we demonstrate that our model can be adapted to support a new input language with less than 1000h of speech. We provide examples, model weights, inference code and we release a benchmark containing 45h of multilingual data for speech translation evaluation.
#### SCRAPL: Scattering Transform with Random Paths for Machine Learning
 - **Authors:** Christopher Mitcheltree, Vincent Lostanlen, Emmanouil Benetos, Mathieu Lagrange
 - **Subjects:** Subjects:
Sound (cs.SD); Machine Learning (cs.LG); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2602.11145

 - **Pdf link:** https://arxiv.org/pdf/2602.11145

 - **Abstract**
 The Euclidean distance between wavelet scattering transform coefficients (known as paths) provides informative gradients for perceptual quality assessment of deep inverse problems in computer vision, speech, and audio processing. However, these transforms are computationally expensive when employed as differentiable loss functions for stochastic gradient descent due to their numerous paths, which significantly limits their use in neural network training. Against this problem, we propose "Scattering transform with Random Paths for machine Learning" (SCRAPL): a stochastic optimization scheme for efficient evaluation of multivariable scattering transforms. We implement SCRAPL for the joint time-frequency scattering transform (JTFS) which demodulates spectrotemporal patterns at multiple scales and rates, allowing a fine characterization of intermittent auditory textures. We apply SCRAPL to differentiable digital signal processing (DDSP), specifically, unsupervised sound matching of a granular synthesizer and the Roland TR-808 drum machine. We also propose an initialization heuristic based on importance sampling, which adapts SCRAPL to the perceptual content of the dataset, improving neural network convergence and evaluation performance. We make our code and audio samples available and provide SCRAPL as a Python package.


by Zyzzyva0381 (Windy). 


2026-02-12
