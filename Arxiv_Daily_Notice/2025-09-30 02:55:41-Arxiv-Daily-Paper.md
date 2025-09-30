# Showing new listings for Monday, 29 September 2025
Auto update papers at about 2:30am UTC (10:30am Beijing time) every weekday.


阅读 `Usage.md`了解如何使用此repo实现个性化的Arxiv论文推送

See `Usage.md` for instructions on how to personalize the repo. 


Keyword list: ['text-to-speech', 'text to speech', 'tts', 'LLM-based', 'speech', 'voice']


Excluded: []


### Today: 14papers 
#### Toward a Realistic Encoding Model of Auditory Affective Understanding in the Brain
 - **Authors:** Guandong Pan, Yaqian Yang, Shi Chen, Xin Wang, Longzhao Liu, Hongwei Zheng, Shaoting Tang
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Artificial Intelligence (cs.AI); Human-Computer Interaction (cs.HC)
 - **Arxiv link:** https://arxiv.org/abs/2509.21381

 - **Pdf link:** https://arxiv.org/pdf/2509.21381

 - **Abstract**
 In affective neuroscience and emotion-aware AI, understanding how complex auditory stimuli drive emotion arousal dynamics remains unresolved. This study introduces a computational framework to model the brain's encoding of naturalistic auditory inputs into dynamic behavioral/neural responses across three datasets (SEED, LIRIS, self-collected BAVE). Guided by neurobiological principles of parallel auditory hierarchy, we decompose audio into multilevel auditory features (through classical algorithms and wav2vec 2.0/Hubert) from the original and isolated human voice/background soundtrack elements, mapping them to emotion-related responses via cross-dataset analyses. Our analysis reveals that high-level semantic representations (derived from the final layer of wav2vec 2.0/Hubert) exert a dominant role in emotion encoding, outperforming low-level acoustic features with significantly stronger mappings to behavioral annotations and dynamic neural synchrony across most brain regions ($p < 0.05$). Notably, middle layers of wav2vec 2.0/hubert (balancing acoustic-semantic information) surpass the final layers in emotion induction across datasets. Moreover, human voices and soundtracks show dataset-dependent emotion-evoking biases aligned with stimulus energy distribution (e.g., LIRIS favors soundtracks due to higher background energy), with neural analyses indicating voices dominate prefrontal/temporal activity while soundtracks excel in limbic regions. By integrating affective computing and neuroscience, this work uncovers hierarchical mechanisms of auditory-emotion encoding, providing a foundation for adaptive emotion-aware systems and cross-disciplinary explorations of audio-affective interactions.
#### Multi-Speaker DOA Estimation in Binaural Hearing Aids using Deep Learning and Speaker Count Fusion
 - **Authors:** Farnaz Jazaeri, Homayoun Kamkar-Parsi, François Grondin, Martin Bouchard
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2509.21382

 - **Pdf link:** https://arxiv.org/pdf/2509.21382

 - **Abstract**
 For extracting a target speaker voice, direction-of-arrival (DOA) estimation is crucial for binaural hearing aids operating in noisy, multi-speaker environments. Among the solutions developed for this task, a deep learning convolutional recurrent neural network (CRNN) model leveraging spectral phase differences and magnitude ratios between microphone signals is a popular option. In this paper, we explore adding source-count information for multi-sources DOA estimation. The use of dual-task training with joint multi-sources DOA estimation and source counting is first considered. We then consider using the source count as an auxiliary feature in a standalone DOA estimation system, where the number of active sources (0, 1, or 2+) is integrated into the CRNN architecture through early, mid, and late fusion strategies. Experiments using real binaural recordings are performed. Results show that the dual-task training does not improve DOA estimation performance, although it benefits source-count prediction. However, a ground-truth (oracle) source count used as an auxiliary feature significantly enhances standalone DOA estimation performance, with late fusion yielding up to 14% higher average F1-scores over the baseline CRNN. This highlights the potential of using source-count estimation for robust DOA estimation in binaural hearing aids.
#### ARTI-6: Towards Six-dimensional Articulatory Speech Encoding
 - **Authors:** Jihwan Lee, Sean Foley, Thanathai Lertpetchpun, Kevin Huang, Yoonjeong Lee, Tiantian Feng, Louis Goldstein, Dani Byrd, Shrikanth Narayanan
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Artificial Intelligence (cs.AI); Computation and Language (cs.CL)
 - **Arxiv link:** https://arxiv.org/abs/2509.21447

 - **Pdf link:** https://arxiv.org/pdf/2509.21447

 - **Abstract**
 We propose ARTI-6, a compact six-dimensional articulatory speech encoding framework derived from real-time MRI data that captures crucial vocal tract regions including the velum, tongue root, and larynx. ARTI-6 consists of three components: (1) a six-dimensional articulatory feature set representing key regions of the vocal tract; (2) an articulatory inversion model, which predicts articulatory features from speech acoustics leveraging speech foundation models, achieving a prediction correlation of 0.87; and (3) an articulatory synthesis model, which reconstructs intelligible speech directly from articulatory features, showing that even a low-dimensional representation can generate natural-sounding speech. Together, ARTI-6 provides an interpretable, computationally efficient, and physiologically grounded framework for advancing articulatory inversion, synthesis, and broader speech technology applications. The source code and speech samples are publicly available.
#### HuLA: Prosody-Aware Anti-Spoofing with Multi-Task Learning for Expressive and Emotional Synthetic Speech
 - **Authors:** Aurosweta Mahapatra, Ismail Rasim Ulgen, Berrak Sisman
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Machine Learning (cs.LG); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2509.21676

 - **Pdf link:** https://arxiv.org/pdf/2509.21676

 - **Abstract**
 Current anti-spoofing systems remain vulnerable to expressive and emotional synthetic speech, since they rarely leverage prosody as a discriminative cue. Prosody is central to human expressiveness and emotion, and humans instinctively use prosodic cues such as F0 patterns and voiced/unvoiced structure to distinguish natural from synthetic speech. In this paper, we propose HuLA, a two-stage prosody-aware multi-task learning framework for spoof detection. In Stage 1, a self-supervised learning (SSL) backbone is trained on real speech with auxiliary tasks of F0 prediction and voiced/unvoiced classification, enhancing its ability to capture natural prosodic variation similar to human perceptual learning. In Stage 2, the model is jointly optimized for spoof detection and prosody tasks on both real and synthetic data, leveraging prosodic awareness to detect mismatches between natural and expressive synthetic speech. Experiments show that HuLA consistently outperforms strong baselines on challenging out-of-domain dataset, including expressive, emotional, and cross-lingual attacks. These results demonstrate that explicit prosodic supervision, combined with SSL embeddings, substantially improves robustness against advanced synthetic speech attacks.
#### FastEnhancer: Speed-Optimized Streaming Neural Speech Enhancement
 - **Authors:** Sunghwan Ahn, Jinmo Han, Beom Jun Woo, Nam Soo Kim
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2509.21867

 - **Pdf link:** https://arxiv.org/pdf/2509.21867

 - **Abstract**
 Streaming speech enhancement is a crucial task for real-time applications such as online meetings, smart home appliances, and hearing aids. Deep neural network-based approaches achieve exceptional performance while demanding substantial computational resources. Although recent neural speech enhancement models have succeeded in reducing the number of parameters and multiply-accumulate operations, their sophisticated architectures often introduce significant processing latency on common hardware. In this work, we propose FastEnhancer, a streaming neural speech enhancement model designed explicitly to minimize real-world latency. It features a simple encoder-decoder structure with efficient RNNFormer blocks. Evaluations on various objective metrics show that FastEnhancer achieves state-of-the-art speech quality and intelligibility while simultaneously demonstrating the fastest processing speed on a single CPU thread. Code and pre-trained weights are publicly available (this https URL).
#### AUV: Teaching Audio Universal Vector Quantization with Single Nested Codebook
 - **Authors:** Yushen Chen, Kai Hu, Long Zhou, Shulin Feng, Xusheng Yang, Hangting Chen, Xie Chen
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2509.21968

 - **Pdf link:** https://arxiv.org/pdf/2509.21968

 - **Abstract**
 We propose AUV, a unified neural audio codec with a single codebook, which enables a favourable reconstruction of speech and further extends to general audio, including vocal, music, and sound. AUV is capable of tackling any 16 kHz mixed-domain audio segment at bit rates around 700 bps. To accomplish this, we guide the matryoshka codebook with nested domain-specific partitions, assigned with corresponding teacher models to perform distillation, all in a single-stage training. A conformer-style encoder-decoder architecture with STFT features as audio representation is employed, yielding better audio quality. Comprehensive evaluations demonstrate that AUV exhibits comparable audio reconstruction ability to state-of-the-art domain-specific single-layer quantizer codecs, showcasing the potential of audio universal vector quantization with a single codebook. The pre-trained model and demo samples are available at this https URL.
#### Speak Your Mind: The Speech Continuation Task as a Probe of Voice-Based Model Bias
 - **Authors:** Shree Harsha Bokkahalli Satish, Harm Lameris, Olivier Perrotin, Gustav Eje Henter, Éva Székely
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Computation and Language (cs.CL); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2509.22061

 - **Pdf link:** https://arxiv.org/pdf/2509.22061

 - **Abstract**
 Speech Continuation (SC) is the task of generating a coherent extension of a spoken prompt while preserving both semantic context and speaker identity. Because SC is constrained to a single audio stream, it offers a more direct setting for probing biases in speech foundation models than dialogue does. In this work we present the first systematic evaluation of bias in SC, investigating how gender and phonation type (breathy, creaky, end-creak) affect continuation behaviour. We evaluate three recent models: SpiritLM (base and expressive), VAE-GSLM, and SpeechGPT across speaker similarity, voice quality preservation, and text-based bias metrics. Results show that while both speaker similarity and coherence remain a challenge, textual evaluations reveal significant model and gender interactions: once coherence is sufficiently high (for VAE-GSLM), gender effects emerge on text-metrics such as agency and sentence polarity. In addition, continuations revert toward modal phonation more strongly for female prompts than for male ones, revealing a systematic voice-quality bias. These findings highlight SC as a controlled probe of socially relevant representational biases in speech foundation models, and suggest that it will become an increasingly informative diagnostic as continuation quality improves.
#### Speaker Anonymisation for Speech-based Suicide Risk Detection
 - **Authors:** Ziyun Cui, Sike Jia, Yang Lin, Yinan Duan, Diyang Qu, Runsen Chen, Chao Zhang, Chang Lei, Wen Wu
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2509.22148

 - **Pdf link:** https://arxiv.org/pdf/2509.22148

 - **Abstract**
 Adolescent suicide is a critical global health issue, and speech provides a cost-effective modality for automatic suicide risk detection. Given the vulnerable population, protecting speaker identity is particularly important, as speech itself can reveal personally identifiable information if the data is leaked or maliciously exploited. This work presents the first systematic study of speaker anonymisation for speech-based suicide risk detection. A broad range of anonymisation methods are investigated, including techniques based on traditional signal processing, neural voice conversion, and speech synthesis. A comprehensive evaluation framework is built to assess the trade-off between protecting speaker identity and preserving information essential for suicide risk detection. Results show that combining anonymisation methods that retain complementary information yields detection performance comparable to that of original speech, while achieving protection of speaker identity for vulnerable populations.
#### Towards Cross-Task Suicide Risk Detection via Speech LLM
 - **Authors:** Jialun Li, Weitao Jiang, Ziyun Cui, Yinan Duan, Diyang Qu, Chao Zhang, Runsen Chen, Chang Lei, Wen Wu
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2509.22153

 - **Pdf link:** https://arxiv.org/pdf/2509.22153

 - **Abstract**
 Suicide risk among adolescents remains a critical public health concern, and speech provides a non-invasive and scalable approach for its detection. Existing approaches, however, typically focus on one single speech assessment task at a time. This paper, for the first time, investigates cross-task approaches that unify diverse speech suicide risk assessment tasks within a single model. Specifically, we leverage a speech large language model as the backbone and incorporate a mixture of DoRA experts (MoDE) approach to capture complementary cues across diverse assessments dynamically. The proposed approach was tested on 1,223 participants across ten spontaneous speech tasks. Results demonstrate that MoDE not only achieves higher detection accuracy than both single-task specialised models and conventional joint-tuning approaches, but also provides better confidence calibration, which is especially important for medical detection tasks.
#### Semantic-VAE: Semantic-Alignment Latent Representation for Better Speech Synthesis
 - **Authors:** Zhikang Niu, Shujie Hu, Jeongsoo Choi, Yushen Chen, Peining Chen, Pengcheng Zhu, Yunting Yang, Bowen Zhang, Jian Zhao, Chunhui Wang, Xie Chen
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2509.22167

 - **Pdf link:** https://arxiv.org/pdf/2509.22167

 - **Abstract**
 While mel-spectrograms have been widely utilized as intermediate representations in zero-shot text-to-speech (TTS), their inherent redundancy leads to inefficiency in learning text-speech alignment. Compact VAE-based latent representations have recently emerged as a stronger alternative, but they also face a fundamental optimization dilemma: higher-dimensional latent spaces improve reconstruction quality and speaker similarity, but degrade intelligibility, while lower-dimensional spaces improve intelligibility at the expense of reconstruction fidelity. To overcome this dilemma, we propose Semantic-VAE, a novel VAE framework that utilizes semantic alignment regularization in the latent space. This design alleviates the reconstruction-generation trade-off by capturing semantic structure in high-dimensional latent representations. Extensive experiments demonstrate that Semantic-VAE significantly improves synthesis quality and training efficiency. When integrated into F5-TTS, our method achieves 2.10% WER and 0.64 speaker similarity on LibriSpeech-PC, outperforming mel-based systems (2.23%, 0.60) and vanilla acoustic VAE baselines (2.65%, 0.59). We also release the code and models to facilitate further research.
#### Align2Speak: Improving TTS for Low Resource Languages via ASR-Guided Online Preference Optimization
 - **Authors:** Shehzeen Hussain, Paarth Neekhara, Xuesong Yang, Edresson Casanova, Subhankar Ghosh, Roy Fejgin, Ryan Langman, Mikyas Desta, Leili Tavabi, Jason Li
 - **Subjects:** Subjects:
Artificial Intelligence (cs.AI); Machine Learning (cs.LG); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2509.21718

 - **Pdf link:** https://arxiv.org/pdf/2509.21718

 - **Abstract**
 Developing high-quality text-to-speech (TTS) systems for low-resource languages is challenging due to the scarcity of paired text and speech data. In contrast, automatic speech recognition (ASR) models for such languages are often more accessible, owing to large-scale multilingual pre-training efforts. We propose a framework based on Group Relative Policy Optimization (GRPO) to adapt an autoregressive, multilingual TTS model to new languages. Our method first establishes a language-agnostic foundation for TTS synthesis by training a multilingual baseline with International Phonetic Alphabet (IPA) tokens. Next, we fine-tune this model on limited paired data of the new languages to capture the target language's prosodic features. Finally, we apply GRPO to optimize the model using only unpaired text and speaker prompts, guided by a multi-objective reward from pretrained ASR, speaker verification, and audio quality estimation models. Experiments demonstrate that this pipeline produces intelligible and speaker-consistent speech in low-resource languages, substantially outperforming fine-tuning alone. Furthermore, our GRPO-based framework also improves TTS performance in high-resource languages, surpassing offline alignment methods such as Direct Preference Optimization (DPO) yielding superior intelligibility, speaker similarity, and audio quality.
#### A Parallel Ultra-Low Power Silent Speech Interface based on a Wearable, Fully-dry EMG Neckband
 - **Authors:** Fiona Meier, Giusy Spacone, Sebastian Frey, Luca Benini, Andrea Cossettini
 - **Subjects:** Subjects:
Systems and Control (eess.SY); Sound (cs.SD); Audio and Speech Processing (eess.AS); Signal Processing (eess.SP)
 - **Arxiv link:** https://arxiv.org/abs/2509.21964

 - **Pdf link:** https://arxiv.org/pdf/2509.21964

 - **Abstract**
 We present a wearable, fully-dry, and ultra-low power EMG system for silent speech recognition, integrated into a textile neckband to enable comfortable, non-intrusive use. The system features 14 fully-differential EMG channels and is based on the BioGAP-Ultra platform for ultra-low power (22 mW) biosignal acquisition and wireless transmission. We evaluate its performance on eight speech commands under both vocalized and silent articulation, achieving average classification accuracies of 87$\pm$3% and 68$\pm$3% respectively, with a 5-fold CV approach. To mimic everyday-life conditions, we introduce session-to-session variability by repositioning the neckband between sessions, achieving leave-one-session-out accuracies of 64$\pm$18% and 54$\pm$7% for the vocalized and silent experiments, respectively. These results highlight the robustness of the proposed approach and the promise of energy-efficient silent-speech decoding.
#### Comprehend and Talk: Text to Speech Synthesis via Dual Language Modeling
 - **Authors:** Junjie Cao, Yichen Han, Ruonan Zhang, Xiaoyang Hao, Hongxiang Li, Shuaijiang Zhao, Yue Liu, Xiao-Ping Zhng
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2509.22062

 - **Pdf link:** https://arxiv.org/pdf/2509.22062

 - **Abstract**
 Existing Large Language Model (LLM) based autoregressive (AR) text-to-speech (TTS) systems, while achieving state-of-the-art quality, still face critical challenges. The foundation of this LLM-based paradigm is the discretization of the continuous speech waveform into a sequence of discrete tokens by neural audio codec. However, single codebook modeling is well suited to text LLMs, but suffers from significant information loss; hierarchical acoustic tokens, typically generated via Residual Vector Quantization (RVQ), often lack explicit semantic structure, placing a heavy learning burden on the model. Furthermore, the autoregressive process is inherently susceptible to error accumulation, which can degrade generation stability. To address these limitations, we propose CaT-TTS, a novel framework for robust and semantically-grounded zero-shot synthesis. First, we introduce S3Codec, a split RVQ codec that injects explicit linguistic features into its primary codebook via semantic distillation from a state-of-the-art ASR model, providing a structured representation that simplifies the learning task. Second, we propose an ``Understand-then-Generate'' dual-Transformer architecture that decouples comprehension from rendering. An initial ``Understanding'' Transformer models the cross-modal relationship between text and the audio's semantic tokens to form a high-level utterance plan. A subsequent ``Generation'' Transformer then executes this plan, autoregressively synthesizing hierarchical acoustic tokens. Finally, to enhance generation stability, we introduce Masked Audio Parallel Inference (MAPI), a nearly parameter-free inference strategy that dynamically guides the decoding process to mitigate local errors.
#### MDAR: A Multi-scene Dynamic Audio Reasoning Benchmark
 - **Authors:** Hui Li, Changhao Jiang, Hongyu Wang, Ming Zhang, Jiajun Sun, Zhixiong Yang, Yifei Cao, Shihan Dou, Xiaoran Fan, Baoyu Fan, Tao Ji, Tao Gui, Qi Zhang, Xuanjing Huang
 - **Subjects:** Subjects:
Sound (cs.SD); Artificial Intelligence (cs.AI); Computation and Language (cs.CL); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2509.22461

 - **Pdf link:** https://arxiv.org/pdf/2509.22461

 - **Abstract**
 The ability to reason from audio, including speech, paralinguistic cues, environmental sounds, and music, is essential for AI agents to interact effectively in real-world scenarios. Existing benchmarks mainly focus on static or single-scene settings and do not fully capture scenarios where multiple speakers, unfolding events, and heterogeneous audio sources interact. To address these challenges, we introduce MDAR, a benchmark for evaluating models on complex, multi-scene, and dynamically evolving audio reasoning tasks. MDAR comprises 3,000 carefully curated question-answer pairs linked to diverse audio clips, covering five categories of complex reasoning and spanning three question types. We benchmark 26 state-of-the-art audio language models on MDAR and observe that they exhibit limitations in complex reasoning tasks. On single-choice questions, Qwen2.5-Omni (open-source) achieves 76.67% accuracy, whereas GPT-4o Audio (closed-source) reaches 68.47%; however, GPT-4o Audio substantially outperforms Qwen2.5-Omni on the more challenging multiple-choice and open-ended tasks. Across all three question types, no model achieves 80% performance. These findings underscore the unique challenges posed by MDAR and its value as a benchmark for advancing audio reasoning this http URL and benchmark can be found at this https URL.


by Zyzzyva0381 (Windy). 


2025-09-30
