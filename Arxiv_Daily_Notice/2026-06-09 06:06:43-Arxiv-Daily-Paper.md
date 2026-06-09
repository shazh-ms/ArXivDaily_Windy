# Showing new listings for Tuesday, 9 June 2026
Auto update papers at about 2:30am UTC (10:30am Beijing time) every weekday.


阅读 `Usage.md`了解如何使用此repo实现个性化的Arxiv论文推送

See `Usage.md` for instructions on how to personalize the repo. 


Keyword list: ['text-to-speech', 'text to speech', 'tts', 'LLM-based', 'speech', 'voice']


Excluded: []


### Today: 17papers 
#### Paediatric-HGNN: A Hybrid Heterogeneous Graph Neural Network for Detecting Disfluency in Children's Speech via Multiscale Acoustic Fusion
 - **Authors:** Rashini Liyanarachchi, Rachael Mackay, Alison Short, Aditya Joshi, Erik Meijering
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Computation and Language (cs.CL); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2606.08210

 - **Pdf link:** https://arxiv.org/pdf/2606.08210

 - **Abstract**
 Automated stuttering detection (ASD) systems struggle with paediatric speech due to high acoustic variability in developing voices and the subtle distinction between pathological stuttering and typical developmental disfluencies. We introduce Paediatric-HGNN, a framework using a Context-aware Part-whole Interaction Network (CaPIN) tailored for paediatric data. Instead of conventional 1D signal modelling, our approach builds a heterogeneous graph capturing hierarchical relationships between lexical units (word nodes) and fine-grained acoustic segments (frame nodes). Trained on curated paediatric corpora (UCLASS and FluencyBank), Paediatric-HGNN achieves 82.4% weighted accuracy and a Typical Disfluency F1-score of 0.386. Modelling hierarchical lexical-acoustic interactions captures developmental "searching" behaviour, offering a more robust and interpretable tool for early clinical intervention.
#### AeroSpectra Sentinel: An Auditable LLM Prompt-Chaining Decision-Support Workflow for Acute Asthma Risk Assessment from Respiratory Sounds and Clinical Signals
 - **Authors:** Aueaphum Aueawatthanaphisut
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Artificial Intelligence (cs.AI); Machine Learning (cs.LG); Signal Processing (eess.SP)
 - **Arxiv link:** https://arxiv.org/abs/2606.08247

 - **Pdf link:** https://arxiv.org/pdf/2606.08247

 - **Abstract**
 Acute asthma risk assessment requires rapid interpretation of respiratory sounds, oxygenation, airflow limitation, speech ability, work of breathing, mental status, and response to reliever therapy. Conventional audio-only classifiers can detect wheeze-like patterns but often lack transparent clinical reasoning and safe escalation logic. This paper presents AeroSpectra Sentinel, a client-side research prototype and decision-support workflow that combines short-time Fourier transform (STFT) respiratory sound analysis, lightweight machine-learning screening, clinical feature fusion, and a five-stage large language model (LLM) prompt-chaining process. The workflow separates signal acquisition, preprocessing, acoustic feature extraction, ML screening, clinical guardrails, and FHIR-ready reporting. We evaluated the audio screening component on a public respiratory sound dataset containing 1,211 WAV recordings from five labels. Using a stratified subset of 584 recordings, a random forest achieved 91.10% binary accuracy and 78.69% F1-score for asthma-vs-non-asthma screening, while a feature-based multilayer perceptron achieved 89.73% accuracy and 78.26% F1-score. A compact log-spectrogram CNN achieved 73.29% accuracy and 55.17% F1-score. Multiclass classification achieved 77.40% accuracy and 77.23% macro-F1. To evaluate the LLM workflow, we conducted a scenario-based audit on 40 simulated clinical vignettes comparing one-shot prompting, prompt chaining, prompt chaining with guardrails, and prompt chaining with guardrails plus FHIR schema validation. The guardrail-plus-schema variant achieved the strongest simulated safety and documentation consistency. AeroSpectra Sentinel is intended as a research prototype, not as a diagnostic medical device or clinically validated risk-assessment product.
#### Fast and Robust On-Device Speaker Diarization: Relative Minimum Cluster Size for Stride-Accelerated Pipelines
 - **Authors:** Fumiaki Yamaguchi
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2606.08505

 - **Pdf link:** https://arxiv.org/pdf/2606.08505

 - **Abstract**
 Speech applications such as meeting transcription and voice agents would benefit from on-device speaker diarization, but practical adoption is limited by inference cost. We study how far a Pyannote 3.1-based pipeline can be accelerated on consumer hardware (an RTX 5070 Ti GPU and an Apple M4 laptop) while preserving diarization error rate (DER). A simple recipe: coarser segmentation stride and per-chunk embedding, yields multi-fold speedups and is DER-neutral on AMI, but degrades sharply on in-the-wild data: on VoxConverse, DER rises from 0.075 to 0.113. We trace the failure to speaker under-counting in the clustering stage, caused by a fixed minimum cluster size interacting with the reduced number of embeddings per speaker. We propose a relative minimum cluster size, mcs = round(f * n) with f = 0.01, which adapts to the embedding budget per recording. A single value of f recovers VoxConverse DER to 0.079 (about 89% of the lost accuracy) while keeping AMI flat, and the accelerated pipeline reaches up to 12.2x speedup on AMI (MPS) over our CAM++ baseline.
#### G-MaP-SE: Guided Speech Enhancement via GMM-Based Prior Matching
 - **Authors:** Yike Zhu, Ziqian Wang, Zikai Liu, Xingchen Li, Zhuangqi Chen, Xianjun Xia, Chuanzeng Huang, Lei Xie
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2606.08580

 - **Pdf link:** https://arxiv.org/pdf/2606.08580

 - **Abstract**
 Using speaker embeddings as conditioning can strengthen speech enhancement, but most methods either require clean enrollment audio or rely on embeddings extracted from noisy speech, which are fragile under noise and domain shift. We propose G-MaP-SE, a guided enhancement framework that builds a clean-speech embedding prior with a Gaussian Mixture Model (GMM) and refines a noisy conditioning embedding by matching it to this prior. The matched prior embedding is then injected into a time-frequency enhancement backbone via a lightweight gated fusion module. Experiments on VoiceBank+DEMAND and DNS Challenge 2020 datasets show that the proposed prior matching consistently outperforms noisy conditioning and substantially narrows the gap to an oracle clean-conditioning upper bound, while requiring no enrollment audio at inference time. The code, audio samples, and checkpoint are available.
#### BareWave: Waveform-Native Flow-Matching Text-to-Speech
 - **Authors:** Wei Fan, Chao-Hong Tan, Qian Chen, Wen Wang, Xiangang Li, Kejiang Chen, Weiming Zhang, Nenghai Yu
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Artificial Intelligence (cs.AI); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2606.09048

 - **Pdf link:** https://arxiv.org/pdf/2606.09048

 - **Abstract**
 Removing intermediate representations and separately trained decoding stages has become an important direction in generative modeling. In text-to-speech, however, high-quality systems are still commonly built through an intermediate acoustic representation before waveform synthesis. In this work, we present BareWave, a fully waveform-native framework for direct text-to-wave generation in flow-matching TTS. We consider this setting to raise three training challenges: raw-waveform modeling lacks a strong pretrained representational scaffold, different stages of training benefit from different noise schedules, and data-space perceptual objectives do not automatically share the temporal structure of the velocity-space flow objective. As a result, direct waveform training is hard to optimize efficiently, hard to push toward a strong final operating point with a fixed recipe, and hard to integrate effective perceptual refinement. Guided by this view, we develop a direct text-to-wave training framework that combines training-time representation alignment, staged noise scheduling, and velocity-aware perceptual alignment (VAPA), while preserving a single waveform-native inference path without pretrained components at test time. Experiments on zero-shot voice cloning show that strong intelligibility, speaker similarity, and naturalness can be achieved under a fully waveform-native inference path, supporting waveform-native flow-matching TTS as a practical direction. Project page with audio demos is available at this https URL.
#### MeanVC 2: Robust Low-Latency Streaming Zero-Shot Voice Conversion
 - **Authors:** Guobin Ma, Yuxuan Xia, Yuepeng Jiang, Dake Guo, Hanke Xie, Jingbin Hu, Yanbo Wang, Lei Xie, Pengcheng Zhu
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2606.09050

 - **Pdf link:** https://arxiv.org/pdf/2606.09050

 - **Abstract**
 Streaming zero-shot voice conversion (VC) has become increasingly popular due to its potential for real-time applications. The recently proposed MeanVC achieves lightweight streaming zero-shot VC, but it has several limitations: its chunk-wise autoregressive denoising doubles the effective training sequence length, conversion quality degrades under small-chunk settings, and its timbre encoder directly relies on reference mel-spectrograms, making it sensitive to reference audio quality. To address these limitations we propose MeanVC 2. We introduce future-receptive chunking (FRC), which explicitly schedules past and future receptive fields across diffusion transformer decoder layers and removes clean-chunk teacher forcing. By incorporating bounded future context, FRC enables stable conversion with a 40 ms chunk size. We further introduce a universal timbre token encoder, which constructs a timbre representation from a global speaker embedding and retrieves fine-grained timbre cues via cross-attention, improving robustness to low-quality references and enhancing zero-shot speaker similarity. Experimental results show that MeanVC 2 significantly outperforms MeanVC, while reducing latency from 211 ms to 110 ms. Audio samples are publicly available. The source code will be publicly released.
#### HoliDubber: Holistic Video Dubbing for Complex Acoustic Scenes via Text-Guided Audio Synthesis
 - **Authors:** Wenhao Guan, Yifan Duan, Junxi Liu, Yu Gu, Feng Dang, Kaidi Wang, Qingyang Hong, Lin Li, Xie Chen
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2606.09098

 - **Pdf link:** https://arxiv.org/pdf/2606.09098

 - **Abstract**
 Video dubbing is a cornerstone of multimedia content creation, aiming to synthesize synchronized acoustic sequences for visual streams. While Text-to-Speech (TTS) and Text-to-Audio (TTA) generation have each achieved remarkable progress, existing dubbing systems remain confined to isolated speech synthesis without incorporating sound effects and ambient audio, forcing practitioners to rely on fragmented workflows and laborious manual post-mixing. To address this limitation, we present HoliDubber, a holistic video dubbing framework that moves beyond speech-only generation by enabling the joint synthesis of speech and sound effects from a single text prompt. Specifically, HoliDubber adopts a patch-based autoregressive diffusion transformer architecture, where a causal language model autoregressively models aggregated patch embeddings to capture global temporal structure, and a Diffusion Transformer decoder generates high-fidelity continuous tokens within each patch, following a divide-and-conquer strategy. To achieve cross-modal alignment, visual features are encoded into patch-level representations and fused with audio patches via cross-attention, enabling the model to ground speech generation in the speaker's visual articulation dynamics. In addition, we introduce HoliDub-Bench, a benchmark curated from established datasets with synchronized video-text-audio triplets designed for holistic dubbing evaluation. Extensive experiments demonstrate that HoliDubber significantly outperforms existing methods across multiple benchmarks in speech quality, synchronization, and speaker similarity. Furthermore, results on HoliDub-Bench validate the effectiveness of joint speech-and-sound generation, establishing a new paradigm for holistic video dubbing in complex acoustic scenes. \footnote{The demo page of the project is this https URL}
#### FlashTTS: Fast Streaming TTS with MTP Acceleration and X-pred Mean Flow Distillation
 - **Authors:** Hanke Xie, Xiaming Ren, Dake Guo, Ruonan You, Wenhao Li, Jingbin Hu, Guobin Ma, Huakang Chen, Kejie Xu, Rui Huang, Weiguo Tan, Xianrong Wang, Lei Xi
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2606.09141

 - **Pdf link:** https://arxiv.org/pdf/2606.09141

 - **Abstract**
 Recent progress in speech dialogue systems requires Text-to-Speech (TTS) models to be faster and more responsive. Modern speech dialogue systems impose two primary requirements on TTS models: low latency and support for streaming inputs and outputs. However, most existing single-codebook LLM-based TTS methods rely on multi-stage pipelines that lack native streaming capabilities. These systems typically suffer from high end-to-end latency due to slow autoregressive prediction and multi-step flow matching. To address these limitations, we propose FlashTTS, an open-source and low-latency streaming TTS framework. FlashTTS introduces a lagged multi-track architecture that natively processes streaming text and speech inputs, thereby eliminating the need for sentence-level buffering. To accelerate acoustic generation, we integrate parallel Multi-Token Prediction (MTP) with an X-pred mean flow matching decoder. This configuration achieves high-fidelity token-to-mel generation in exactly two function evaluations (2-NFE). By jointly optimizing input processing and decoding efficiency, FlashTTS offers a practical foundation for real-time speech dialogue systems. Experiments show that FlashTTS substantially reduces First-Packet Latency to 325ms compared to robust streaming baselines, all while preserving strong zero-shot voice cloning and cross-lingual intelligibility. Speech samples are available. The model code and checkpoints will be released as open source.
#### A Comparative Study of Pre-trained Speech Encoders and Training Objectives for Large-Scale Indic Spoken Language Identification
 - **Authors:** Agneedh Basu, Pavan Kumar J, Sujith P, Visruth Sanka, Nihar Desai, Prasanta Kumar Ghosh
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2606.09317

 - **Pdf link:** https://arxiv.org/pdf/2606.09317

 - **Abstract**
 Spoken language identification (LID) for Indian languages is a challenging problem due to the large number of languages, significant phonetic overlap among related varieties, and the scarcity of labeled data for many low-resource languages. In this work, we present a systematic comparative study of two pre-trained speech encoders -- Whisper and FastConformer -- combined with a linear classifier for large-scale Indic LID spanning 42 languages across four linguistic families. We evaluate both encoders in frozen (linear probing) and fine-tuned settings, and compare three training objectives: cross-entropy (CE), supervised contrastive loss with cross entropy (CE + supCon), and hierarchical softmax (HSM). Models are trained on the Vaani dataset and evaluated in a cross-corpus setting on Vaani-Test (held-out), FLEURS, and Kathbath, providing insights into domain generalization. The frozen FastConformer encoder achieves over 90\% macro accuracy on FLEURS and Kathbath without any task-specific adaptation, substantially outperforming Whisper on out-of-domain benchmarks, while fine-tuned Whisper yields stronger in-domain performance. HSM consistently outperforms CE and CE+SupCon for both encoders across all benchmarks, with the largest gains on out-of-domain test sets. CE+SupCon degrades FastConformer's cross-corpus generalization, suggesting that the contrastive objective over-specializes representations to in-domain conditions. Per-family analysis shows that Central Indo-Aryan varieties are the hardest to discriminate, with Hindi--Urdu and the Sadri--Chhattisgarhi--Surgujia cluster being the dominant confusion pairs.
#### Factors affecting ASR performance: A study using state of the art ASR models in Indic Languages
 - **Authors:** Agneedh Basu, Pavan Kumar J, Pranav Bhat, Sujith Pulikodan, Visruth Sanka, Nihar Desai, Prasanta Kumar Ghosh
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2606.09335

 - **Pdf link:** https://arxiv.org/pdf/2606.09335

 - **Abstract**
 ASR performance varies across languages, speakers, and recording conditions, yet systematic analysis for Indic languages remain limited. We present a large-scale study of decoded outputs from multiple open-source ASR models evaluated on diverse Indian speech datasets in zero-shot settings. We analyze linguistic, speaker-level, and acoustic factors across Hindi, Bengali, Kannada, Telugu, and Marathi. We examine correlations between WER and speaker traits such as average word length, speaking rate, and utterance duration across multiple model dataset pairs. For Hindi, we further analyze audio factors including telephone codecs, bit depth, resampling, and background noise. Results reveal both cross lingual patterns and language-specific sensitivities, showing how speaker behavior and signal processing choices affect ASR robustness in real world Indic scenarios.
#### Parameter-Efficient Continual Learning for Automatic Speech Recognition
 - **Authors:** Steven Vander Eeckt, Hugo Van hamme
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2606.09342

 - **Pdf link:** https://arxiv.org/pdf/2606.09342

 - **Abstract**
 Speech foundation models enable strong general-purpose ASR and are attractive for downstream adaptation. However, their size and the catastrophic forgetting induced by sequential fine-tuning demand parameter-efficient and regularized training methods, motivating parameter-efficient continual learning (PECL). While PECL has been widely studied in NLP and vision, it has received less attention in ASR. In this paper, we propose a simple yet effective PECL method based on recent advances in parameter-efficient fine-tuning for ASR. We partition pretrained weight matrices into head and tail subspaces according to singular values and restrict adaptation to approximate rotations within the low-energy tail subspace, preserving dominant components and reducing forgetting. For subsequent tasks, rotations are combined via weight averaging to further improve retention. Experiments on two benchmarks demonstrate reduced forgetting and superior overall performance compared to recent PECL baselines.
#### A study on the impact of region specific data on the performance of Indic ASR
 - **Authors:** Agneedh Basu, Pavan Kumar J, Pranav Bhat, Sujith Pulikodan, Visruth Sanka, Nihar Desai, Prasata Kumar Ghosh
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2606.09345

 - **Pdf link:** https://arxiv.org/pdf/2606.09345

 - **Abstract**
 Automatic Speech Recognition (ASR) systems are widely deployed across linguistically diverse regions, yet their ability to generalize across fine-grained geographic variation remains underexplored. We present a systematic study of cross-district ASR generalization for Indian languages, analyzing the impact of regional variation on performance. Using finetuning as a controlled probe, we train models on speech from a single district and evaluate them on other districts within the same language. We examine trends across multiple train test district pairs and quantify performance differences. To assess geographic effects, we analyze the correlation between WER and inter district distance using two distance measures. Our results show consistent correlations between geographic distance and WER, highlighting the challenges of regional generalization and the need for geographically diverse speech data in ASR development and evaluation in India.
#### Rethinking Depth: A study of the Recursive-Transformer for Speech Recognition
 - **Authors:** Thomas Rolland, Carlos Carvalho, Alberto Abad
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2606.09357

 - **Pdf link:** https://arxiv.org/pdf/2606.09357

 - **Abstract**
 Transformer-based architectures have led to significant improvements in Automatic Speech Recognition (ASR), often at the cost of substantially increased model sizes. A promising approach to address this issue is layer sharing through depth recursion, commonly referred to as the Recursive-Transformer, which involves repeatedly applying the same layers within the model. Despite its potential shown in other fields, this technique remains relatively unexplored in ASR. In this paper, we present an experimental study of the Recursive-Transformer applied to ASR encoder architectures. We systematically investigate the impact of recursion depth and layer allocation within the Recursive-based Transformer. Our results demonstrate that the Recursive-Transformer is a viable alternative, especially when recurrence is applied in the latent space with a restricted number of loops, obtaining comparable performance while reducing the parameter count by 66%.
#### Cross-Modal Masking for Robust Silent Speech Synthesis Using sEMG and Lipreading
 - **Authors:** Eder del Blanco, David Gimeno-Gómez, Eva Navas, Carlos-D. Martínez-Hinarejos, Inma Hernáez
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Computation and Language (cs.CL); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2606.09667

 - **Pdf link:** https://arxiv.org/pdf/2606.09667

 - **Abstract**
 Speech restoration through silent speech interfaces (SSIs) has emerged as a promising assistive technology for individuals with impaired or absent laryngeal voice production. Among non-invasive SSI modalities, surface electromyography (sEMG) and video-based lipreading provide complementary articulatory information, yet their integration for continuous speech synthesis remains underexplored. Moreover, existing multimodal approaches rarely address robustness to modality degradation or temporary sensor failure, limiting their applicability in realistic scenarios. In this work, we propose a masked multimodal speech synthesis framework that jointly leverages sEMG and lipreading signals through modality masking during training. Under multispeaker settings, the proposed approach reduces word error rate by up to 14 absolute percentage points compared to the strongest unimodal baseline. Experimental results not only show that masking strategies are critical for these performance gains and robustness under low-bitrate conditions, but also that they generalize better than degradation-specific data augmentations in the presence of modality absence conditions. Phone-level analyses further reveal complementary contributions across modalities, with particularly strong benefits for vowels and for specific consonant groups. Overall, these findings demonstrate the effectiveness and robustness of masked multimodal integration for silent speech synthesis, although adaptation to laryngectomized speakers remains an open research challenge.
#### MeCo: One-Step MeanFlow-based Corrector for Multi-Channel Speech Separation
 - **Authors:** Dohwan Kim, Jung-Woo Choi
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Artificial Intelligence (cs.AI)
 - **Arxiv link:** https://arxiv.org/abs/2606.09677

 - **Pdf link:** https://arxiv.org/pdf/2606.09677

 - **Abstract**
 While discriminative models for multi-channel speech separation excel in reference-based metrics, they often exhibit suboptimal human listening quality. To address this, we propose a novel MeanFlow-based one-step generative corrector (MeCo). MeCo learns a conditional average velocity field to map discriminative estimates directly onto the clean speech manifold in a single step. To maximize one-step generation performance, we introduce Data-Space Optimization (DSO). DSO integrates an $\mathbf{x}_r$-loss, which penalizes prediction errors on longer displacement intervals to serve as a generative objective for human listening quality, with an Endpoint SI-SDR loss that directly optimizes terminal signal fidelity. Experiments demonstrate that MeCo achieves state-of-the-art (SOTA) performance with minimal computational overhead, simultaneously achieving superior signal fidelity and human listening quality in both in-domain and out-of-domain scenarios.
#### Is Text All You Need? Text as a Universal Information Bottleneck for Speech LLMs
 - **Authors:** Ming-Hao Hsu, Yuxuan Hu, Shujie Liu, Jinyu Li, Yan Lu, Zhizheng Wu
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2606.09366

 - **Pdf link:** https://arxiv.org/pdf/2606.09366

 - **Abstract**
 Large language models (LLMs) provide a powerful reasoning backbone for speech understanding, but integrating continuous acoustic signals into a frozen LLM remains challenging. Existing speech-to-LLM interfaces typically operate at two extremes: either enforcing near-discrete token alignment, which benefits transcription but loses paralinguistic information, or learning unconstrained continuous representations, which can drift away from the LLM's input space and degrade autoregressive decoding. In this work, we propose Convex Gate (C-Gate), a speech-to-LLM bridge that constrains all speech representations to lie within the LLM's input embedding manifold with an architectural convex-hull constraint. Concretely, each frame is represented as a convex combination of token embeddings, ensuring compatibility with the pretrained LLM while preserving continuous expressivity. Across automatic speech recognition (ASR) and emotion recognition, C-Gate achieves strong joint performance, improving LibriSpeech WER by up to 48.7% relative while matching or exceeding single-task emotion accuracy. Beyond performance, our analysis reveals a key insight: information is not carried by discrete token identities, but by time-resolved trajectories in the embedding space. Causal interventions confirm that both the trajectory structure and alignment to the pretrained embedding manifold are critical for performance. These results suggest that geometry, rather than token discreteness, is the fundamental design factor in speech-to-LLM interfaces, and provide a controlled regime for studying multimodal integration in frozen LLMs. We release the checkpoint, per-sample outputs, mechanism dumps, and intervention suite for replication.
#### What Makes Synthetic Speech Sound Sarcastic? A Prosody-Controlled Perception Study
 - **Authors:** Zhu Li, Shekhar Nayak, Matt Coler
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2606.09717

 - **Pdf link:** https://arxiv.org/pdf/2606.09717

 - **Abstract**
 Prosody plays a central role in sarcasm perception, yet previous studies have relied on naturally produced speech that lacks fine-grained control over individual acoustic dimensions. As prosodic cues co-vary in natural data, isolating their independent contributions remains challenging. We introduce a controlled framework using neural text-to-speech (TTS) with prompt-based prosodic conditioning to manipulate speech rate, pitch variation, and loudness. An orthogonal stimulus set was constructed to enable causal testing of prosodic cue effects. Human listeners rated sarcasm and naturalness, and their judgments were compared with predictions from a foundation model capable of processing audio input. Results show that loudness primarily drives human sarcasm perception, whereas the model assigns greater weight to speech rate, leading to distinct cue-weighting patterns. This study shows how controllable neural TTS enables investigation of prosodic cue weighting in speech perception.


by Zyzzyva0381 (Windy). 


2026-06-09
