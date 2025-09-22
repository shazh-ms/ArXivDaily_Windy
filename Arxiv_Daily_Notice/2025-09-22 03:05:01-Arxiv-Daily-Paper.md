# Showing new listings for Monday, 22 September 2025
Auto update papers at about 2:30am UTC (10:30am Beijing time) every weekday.


阅读 `Usage.md`了解如何使用此repo实现个性化的Arxiv论文推送

See `Usage.md` for instructions on how to personalize the repo. 


Keyword list: ['text-to-speech', 'text to speech', 'tts', 'LLM-based', 'speech', 'voice']


Excluded: []


### Today: 28papers 
#### Breathing and Semantic Pause Detection and Exertion-Level Classification in Post-Exercise Speech
 - **Authors:** Yuyu Wang, Wuyue Xia, Huaxiu Yao, Jingping Nie
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Computation and Language (cs.CL); Machine Learning (cs.LG); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2509.15473

 - **Pdf link:** https://arxiv.org/pdf/2509.15473

 - **Abstract**
 Post-exercise speech contains rich physiological and linguistic cues, often marked by semantic pauses, breathing pauses, and combined breathing-semantic pauses. Detecting these events enables assessment of recovery rate, lung function, and exertion-related abnormalities. However, existing works on identifying and distinguishing different types of pauses in this context are limited. In this work, building on a recently released dataset with synchronized audio and respiration signals, we provide systematic annotations of pause types. Using these annotations, we systematically conduct exploratory breathing and semantic pause detection and exertion-level classification across deep learning models (GRU, 1D CNN-LSTM, AlexNet, VGG16), acoustic features (MFCC, MFB), and layer-stratified Wav2Vec2 representations. We evaluate three setups-single feature, feature fusion, and a two-stage detection-classification cascade-under both classification and regression formulations. Results show per-type detection accuracy up to 89$\%$ for semantic, 55$\%$ for breathing, 86$\%$ for combined pauses, and 73$\%$overall, while exertion-level classification achieves 90.5$\%$ accuracy, outperformin prior work.
#### State-of-the-Art Dysarthric Speech Recognition with MetaICL for on-the-fly Personalization
 - **Authors:** Dhruuv Agarwal, Harry Zhang, Yang Yu, Quan Wang
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2509.15516

 - **Pdf link:** https://arxiv.org/pdf/2509.15516

 - **Abstract**
 Personalizing Automatic Speech Recognition (ASR) for dysarthric speech is crucial but challenging due to training and storing of individual user adapters. We propose a hybrid meta-training method for a single model, excelling in zero-shot and few-shot on-the-fly personalization via in-context learning (ICL). Measuring Word Error Rate (WER) on state-of-the-art subsets, the model achieves 13.9% WER on Euphonia which surpasses speaker-independent baselines (17.5% WER) and rivals user-specific personalized models. On SAP Test 1, its 5.3% WER significantly bests the 8% from even personalized adapters. We also demonstrate the importance of example curation, where an oracle text-similarity method shows 5 curated examples can achieve performance similar to 19 randomly selected ones, highlighting a key area for future efficiency gains. Finally, we conduct data ablations to measure the data efficiency of this approach. This work presents a practical, scalable, and personalized solution.
#### Rec-RIR: Monaural Blind Room Impulse Response Identification via DNN-based Reverberant Speech Reconstruction in STFT Domain
 - **Authors:** Pengyu Wang, Xiaofei Li
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Signal Processing (eess.SP)
 - **Arxiv link:** https://arxiv.org/abs/2509.15628

 - **Pdf link:** https://arxiv.org/pdf/2509.15628

 - **Abstract**
 Room impulse response (RIR) characterizes the complete propagation process of sound in an enclosed space. This paper presents Rec-RIR for monaural blind RIR identification. Rec-RIR is developed based on the convolutive transfer function (CTF) approximation, which models reverberation effect within narrow-band filter banks in the short-time Fourier transform (STFT) domain. Specifically, we propose a deep neural network (DNN) with cross-band and narrow-band blocks to estimate the CTF filter. The DNN is trained through reconstructing the noise-free reverberant speech spectra. This objective enables stable and straightforward supervised training. Subsequently, a pseudo intrusive measurement process is employed to convert the CTF filter estimate into time-domain RIR by simulating a common intrusive RIR measurement procedure. Experimental results demonstrate that Rec-RIR achieves state-of-the-art (SOTA) performance in both RIR identification and acoustic parameter estimation. Open-source codes are available online at this https URL.
#### A Steered Response Power Method for Sound Source Localization With Generic Acoustic Models
 - **Authors:** Kaspar Müller, Markus Buck, Simon Doclo, Jan Østergaard, Tobias Wolff
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2509.15702

 - **Pdf link:** https://arxiv.org/pdf/2509.15702

 - **Abstract**
 The steered response power (SRP) method is one of the most popular approaches for acoustic source localization with microphone arrays. It is often based on simplifying acoustic assumptions, such as an omnidirectional sound source in the far field of the microphone array(s), free field propagation, and spatially uncorrelated noise. In reality, however, there are many acoustic scenarios where such assumptions are violated. This paper proposes a generalization of the conventional SRP method that allows to apply generic acoustic models for localization with arbitrary microphone constellations. These models may consider, for instance, level differences in distributed microphones, the directivity of sources and receivers, or acoustic shadowing effects. Moreover, also measured acoustic transfer functions may be applied as acoustic model. We show that the delay-and-sum beamforming of the conventional SRP is not optimal for localization with generic acoustic models. To this end, we propose a generalized SRP beamforming criterion that considers generic acoustic models and spatially correlated noise, and derive an optimal SRP beamformer. Furthermore, we propose and analyze appropriate frequency weightings. Unlike the conventional SRP, the proposed method can jointly exploit observed level and time differences between the microphone signals to infer the source location. Realistic simulations of three different microphone setups with speech under various noise conditions indicate that the proposed method can significantly reduce the mean localization error compared to the conventional SRP and, in particular, a reduction of more than 60% can be archived in noisy conditions.
#### Deep Dubbing: End-to-End Auto-Audiobook System with Text-to-Timbre and Context-Aware Instruct-TTS
 - **Authors:** Ziqi Dai, Yiting Chen, Jiacheng Xu, Liufei Xie, Yuchen Wang, Zhenchuan Yang, Bingsong Bai, Yangsheng Gao, Wenjiang Zhou, Weifeng Zhao, Ruohua Zhou
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2509.15845

 - **Pdf link:** https://arxiv.org/pdf/2509.15845

 - **Abstract**
 The pipeline for multi-participant audiobook production primarily consists of three stages: script analysis, character voice timbre selection, and speech synthesis. Among these, script analysis can be automated with high accuracy using NLP models, whereas character voice timbre selection still relies on manual effort. Speech synthesis uses either manual dubbing or text-to-speech (TTS). While TTS boosts efficiency, it struggles with emotional expression, intonation control, and contextual scene adaptation. To address these challenges, we propose DeepDubbing, an end-to-end automated system for multi-participant audiobook production. The system comprises two main components: a Text-to-Timbre (TTT) model and a Context-Aware Instruct-TTS (CA-Instruct-TTS) model. The TTT model generates role-specific timbre embeddings conditioned on text descriptions. The CA-Instruct-TTS model synthesizes expressive speech by analyzing contextual dialogue and incorporating fine-grained emotional instructions. This system enables the automated generation of multi-participant audiobooks with both timbre-matched character voices and emotionally expressive narration, offering a novel solution for audiobook production.
#### VoXtream: Full-Stream Text-to-Speech with Extremely Low Latency
 - **Authors:** Nikita Torgashov, Gustav Eje Henter, Gabriel Skantze
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Computation and Language (cs.CL); Human-Computer Interaction (cs.HC); Machine Learning (cs.LG); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2509.15969

 - **Pdf link:** https://arxiv.org/pdf/2509.15969

 - **Abstract**
 We present VoXtream, a fully autoregressive, zero-shot streaming text-to-speech (TTS) system for real-time use that begins speaking from the first word. VoXtream directly maps incoming phonemes to audio tokens using a monotonic alignment scheme and a dynamic look-ahead that does not delay onset. Built around an incremental phoneme transformer, a temporal transformer predicting semantic and duration tokens, and a depth transformer producing acoustic tokens, VoXtream achieves, to our knowledge, the lowest initial delay among publicly available streaming TTS: 102 ms on GPU. Despite being trained on a mid-scale 9k-hour corpus, it matches or surpasses larger baselines on several metrics, while delivering competitive quality in both output- and full-streaming settings. Demo and code are available at this https URL.
#### Interpreting the Role of Visemes in Audio-Visual Speech Recognition
 - **Authors:** Aristeidis Papadopoulos, Naomi Harte
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2509.16023

 - **Pdf link:** https://arxiv.org/pdf/2509.16023

 - **Abstract**
 Audio-Visual Speech Recognition (AVSR) models have surpassed their audio-only counterparts in terms of performance. However, the interpretability of AVSR systems, particularly the role of the visual modality, remains under-explored. In this paper, we apply several interpretability techniques to examine how visemes are encoded in AV-HuBERT a state-of-the-art AVSR model. First, we use t-distributed Stochastic Neighbour Embedding (t-SNE) to visualize learned features, revealing natural clustering driven by visual cues, which is further refined by the presence of audio. Then, we employ probing to show how audio contributes to refining feature representations, particularly for visemes that are visually ambiguous or under-represented. Our findings shed light on the interplay between modalities in AVSR and could point to new strategies for leveraging visual information to improve AVSR performance.
#### Rethinking Cross-Corpus Speech Emotion Recognition Benchmarking: Are Paralinguistic Pre-Trained Representations Sufficient?
 - **Authors:** Orchid Chetia Phukan, Mohd Mujtaba Akhtar, Girish, Swarup Ranjan Behera, Parabattina Bhagath, Pailla Balakrishna Reddy, Arun Balaji Buduru
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2509.16182

 - **Pdf link:** https://arxiv.org/pdf/2509.16182

 - **Abstract**
 Recent benchmarks evaluating pre-trained models (PTMs) for cross-corpus speech emotion recognition (SER) have overlooked PTM pre-trained for paralinguistic speech processing (PSP), raising concerns about their reliability, since SER is inherently a paralinguistic task. We hypothesize that PSP-focused PTM will perform better in cross-corpus SER settings. To test this, we analyze state-of-the-art PTMs representations including paralinguistic, monolingual, multilingual, and speaker recognition. Our results confirm that TRILLsson (a paralinguistic PTM) outperforms others, reinforcing the need to consider PSP-focused PTMs in cross-corpus SER benchmarks. This study enhances benchmark trustworthiness and guides PTMs evaluations for reliable cross-corpus SER.
#### Are Multimodal Foundation Models All That Is Needed for Emofake Detection?
 - **Authors:** Mohd Mujtaba Akhtar, Girish, Orchid Chetia Phukan, Swarup Ranjan Behera, Pailla Balakrishna Reddy, Ananda Chandra Nayak, Sanjib Kumar Nayak, Arun Balaji Buduru
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2509.16193

 - **Pdf link:** https://arxiv.org/pdf/2509.16193

 - **Abstract**
 In this work, we investigate multimodal foundation models (MFMs) for EmoFake detection (EFD) and hypothesize that they will outperform audio foundation models (AFMs). MFMs due to their cross-modal pre-training, learns emotional patterns from multiple modalities, while AFMs rely only on audio. As such, MFMs can better recognize unnatural emotional shifts and inconsistencies in manipulated audio, making them more effective at distinguishing real from fake emotional expressions. To validate our hypothesis, we conduct a comprehensive comparative analysis of state-of-the-art (SOTA) MFMs (e.g. LanguageBind) alongside AFMs (e.g. WavLM). Our experiments confirm that MFMs surpass AFMs for EFD. Beyond individual foundation models (FMs) performance, we explore FMs fusion, motivated by findings in related research areas such synthetic speech detection and speech emotion recognition. To this end, we propose SCAR, a novel framework for effective fusion. SCAR introduces a nested cross-attention mechanism, where representations from FMs interact at two stages sequentially to refine information exchange. Additionally, a self-attention refinement module further enhances feature representations by reinforcing important cross-FM cues while suppressing noise. Through SCAR with synergistic fusion of MFMs, we achieve SOTA performance, surpassing both standalone FMs and conventional fusion approaches and previous works on EFD.
#### Emotion-Aware Speech Generation with Character-Specific Voices for Comics
 - **Authors:** Zhiwen Qian, Jinhua Liang, Huan Zhang
 - **Subjects:** Subjects:
Sound (cs.SD); Artificial Intelligence (cs.AI); Multimedia (cs.MM); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2509.15253

 - **Pdf link:** https://arxiv.org/pdf/2509.15253

 - **Abstract**
 This paper presents an end-to-end pipeline for generating character-specific, emotion-aware speech from comics. The proposed system takes full comic volumes as input and produces speech aligned with each character's dialogue and emotional state. An image processing module performs character detection, text recognition, and emotion intensity recognition. A large language model performs dialogue attribution and emotion analysis by integrating visual information with the evolving plot context. Speech is synthesized through a text-to-speech model with distinct voice profiles tailored to each character and emotion. This work enables automated voiceover generation for comics, offering a step toward interactive and immersive comic reading experience.
#### Speech Language Models for Under-Represented Languages: Insights from Wolof
 - **Authors:** Yaya Sy, Dioula Doucouré, Christophe Cerisara, Irina Illina
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2509.15362

 - **Pdf link:** https://arxiv.org/pdf/2509.15362

 - **Abstract**
 We present our journey in training a speech language model for Wolof, an underrepresented language spoken in West Africa, and share key insights. We first emphasize the importance of collecting large-scale, spontaneous, high-quality speech data, and show that continued pretraining HuBERT on this dataset outperforms both the base model and African-centric models on ASR. We then integrate this speech encoder into a Wolof LLM to train the first Speech LLM for this language, extending its capabilities to tasks such as speech translation. Furthermore, we explore training the Speech LLM to perform multi-step Chain-of-Thought before transcribing or translating. Our results show that the Speech LLM not only improves speech recognition but also performs well in speech translation. The models and the code will be openly shared.
#### Exploring Fine-Tuning of Large Audio Language Models for Spoken Language Understanding under Limited Speech data
 - **Authors:** Youngwon Choi, Jaeyoon Jung, Hyeonyu Kim, Huu-Kim Nguyen, Hwayeon Kim
 - **Subjects:** Subjects:
Sound (cs.SD); Computation and Language (cs.CL); Machine Learning (cs.LG); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2509.15389

 - **Pdf link:** https://arxiv.org/pdf/2509.15389

 - **Abstract**
 Large Audio Language Models (LALMs) have emerged as powerful tools for speech-related tasks but remain underexplored for fine-tuning, especially with limited speech data. To bridge this gap, we systematically examine how different fine-tuning schemes including text-only, direct mixing, and curriculum learning affect spoken language understanding (SLU), focusing on scenarios where text-label pairs are abundant while paired speech-label data are limited. Results show that LALMs already achieve competitive performance with text-only fine-tuning, highlighting their strong generalization ability. Adding even small amounts of speech data (2-5%) yields substantial further gains, with curriculum learning particularly effective under scarce data. In cross-lingual SLU, combining source-language speech data with target-language text and minimal target-language speech data enables effective adaptation. Overall, this study provides practical insights into the LALM fine-tuning under realistic data constraints.
#### BiRQ: Bi-Level Self-Labeling Random Quantization for Self-Supervised Speech Recognition
 - **Authors:** Liuyuan Jiang, Xiaodong Cui, Brian Kingsbury, Tianyi Chen, Lisha Chen
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2509.15430

 - **Pdf link:** https://arxiv.org/pdf/2509.15430

 - **Abstract**
 Speech is a rich signal, and labeled audio-text pairs are costly, making self-supervised learning essential for scalable representation learning. A core challenge in speech SSL is generating pseudo-labels that are both informative and efficient: strong labels, such as those used in HuBERT, improve downstream performance but rely on external encoders and multi-stage pipelines, while efficient methods like BEST-RQ achieve simplicity at the cost of weaker labels. We propose BiRQ, a bilevel SSL framework that combines the efficiency of BEST-RQ with the refinement benefits of HuBERT-style label enhancement. The key idea is to reuse part of the model itself as a pseudo-label generator: intermediate representations are discretized by a random-projection quantizer to produce enhanced labels, while anchoring labels derived directly from the raw input stabilize training and prevent collapse. Training is formulated as an efficient first-order bilevel optimization problem, solved end-to-end with differentiable Gumbel-softmax selection. This design eliminates the need for external label encoders, reduces memory cost, and enables iterative label refinement in an end-to-end fashion. BiRQ consistently improves over BEST-RQ while maintaining low complexity and computational efficiency. We validate our method on various datasets, including 960-hour LibriSpeech, 150-hour AMI meetings and 5,000-hour YODAS, demonstrating consistent gains over BEST-RQ.
#### Impact of Phonetics on Speaker Identity in Adversarial Voice Attack
 - **Authors:** Daniyal Kabir Dar, Qiben Yan, Li Xiao, Arun Ross
 - **Subjects:** Subjects:
Sound (cs.SD); Artificial Intelligence (cs.AI); Cryptography and Security (cs.CR); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2509.15437

 - **Pdf link:** https://arxiv.org/pdf/2509.15437

 - **Abstract**
 Adversarial perturbations in speech pose a serious threat to automatic speech recognition (ASR) and speaker verification by introducing subtle waveform modifications that remain imperceptible to humans but can significantly alter system outputs. While targeted attacks on end-to-end ASR models have been widely studied, the phonetic basis of these perturbations and their effect on speaker identity remain underexplored. In this work, we analyze adversarial audio at the phonetic level and show that perturbations exploit systematic confusions such as vowel centralization and consonant substitutions. These distortions not only mislead transcription but also degrade phonetic cues critical for speaker verification, leading to identity drift. Using DeepSpeech as our ASR target, we generate targeted adversarial examples and evaluate their impact on speaker embeddings across genuine and impostor samples. Results across 16 phonetically diverse target phrases demonstrate that adversarial audio induces both transcription errors and identity drift, highlighting the need for phonetic-aware defenses to ensure the robustness of ASR and speaker recognition systems.
#### A Novel Semantic Compression Approach for Ultra-low Bandwidth Voice Communication
 - **Authors:** Ryan Collette, Ross Greenwood, Serena Nicoll
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2509.15462

 - **Pdf link:** https://arxiv.org/pdf/2509.15462

 - **Abstract**
 While existing speech audio codecs designed for compression exploit limited forms of temporal redundancy and allow for multi-scale representations, they tend to represent all features of audio in the same way. In contrast, generative voice models designed for text-to-speech and voice transfer tasks have recently proved effective at factorizing audio signals into high-level semantic representations of fundamentally distinct features. In this paper, we leverage such representations in a novel semantic communications approach to achieve lower bitrates without sacrificing perceptual quality or suitability for specific downstream tasks. Our technique matches or outperforms existing audio codecs on transcription, sentiment analysis, and speaker verification when encoding at 2-4x lower bitrate -- notably surpassing Encodec in perceptual quality and speaker verification while using up to 4x less bitrate.
#### Beyond Video-to-SFX: Video to Audio Synthesis with Environmentally Aware Speech
 - **Authors:** Xinlei Niu, Jianbo Ma, Dylan Harper-Harris, Xiangyu Zhang, Charles Patrick Martin, Jing Zhang
 - **Subjects:** Subjects:
Sound (cs.SD); Multimedia (cs.MM); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2509.15492

 - **Pdf link:** https://arxiv.org/pdf/2509.15492

 - **Abstract**
 The generation of realistic, context-aware audio is important in real-world applications such as video game development. While existing video-to-audio (V2A) methods mainly focus on Foley sound generation, they struggle to produce intelligible speech. Meanwhile, current environmental speech synthesis approaches remain text-driven and fail to temporally align with dynamic video content. In this paper, we propose Beyond Video-to-SFX (BVS), a method to generate synchronized audio with environmentally aware intelligible speech for given videos. We introduce a two-stage modeling method: (1) stage one is a video-guided audio semantic (V2AS) model to predict unified audio semantic tokens conditioned on phonetic cues; (2) stage two is a video-conditioned semantic-to-acoustic (VS2A) model that refines semantic tokens into detailed acoustic tokens. Experiments demonstrate the effectiveness of BVS in scenarios such as video-to-context-aware speech synthesis and immersive audio background conversion, with ablation studies further validating our design. Our demonstration is available at~\href{this https URL}{BVS-Demo}.
#### Chunk Based Speech Pre-training with High Resolution Finite Scalar Quantization
 - **Authors:** Yun Tang, Cindy Tseng
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2509.15579

 - **Pdf link:** https://arxiv.org/pdf/2509.15579

 - **Abstract**
 Low latency speech human-machine communication is becoming increasingly necessary as speech technology advances quickly in the last decade. One of the primary factors behind the advancement of speech technology is self-supervised learning. Most self-supervised learning algorithms are designed with full utterance assumption and compromises have to made if partial utterances are presented, which are common in the streaming applications. In this work, we propose a chunk based self-supervised learning (Chunk SSL) algorithm as an unified solution for both streaming and offline speech pre-training. Chunk SSL is optimized with the masked prediction loss and an acoustic encoder is encouraged to restore indices of those masked speech frames with help from unmasked frames in the same chunk and preceding chunks. A copy and append data augmentation approach is proposed to conduct efficient chunk based pre-training. Chunk SSL utilizes a finite scalar quantization (FSQ) module to discretize input speech features and our study shows a high resolution FSQ codebook, i.e., a codebook with vocabulary size up to a few millions, is beneficial to transfer knowledge from the pre-training task to the downstream tasks. A group masked prediction loss is employed during pre-training to alleviate the high memory and computation cost introduced by the large codebook. The proposed approach is examined in two speech to text tasks, i.e., speech recognition and speech translation. Experimental results on the \textsc{Librispeech} and \textsc{Must-C} datasets show that the proposed method could achieve very competitive results for speech to text tasks at both streaming and offline modes.
#### Thinking in cocktail party: Chain-of-Thought and reinforcement learning for target speaker automatic speech recognition
 - **Authors:** Yiru Zhang, Hang Su, Lichun Fan, Zhenbo Luo, Jian Luan
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2509.15612

 - **Pdf link:** https://arxiv.org/pdf/2509.15612

 - **Abstract**
 Target Speaker Automatic Speech Recognition (TS-ASR) aims to transcribe the speech of a specified target speaker from multi-speaker mixtures in cocktail party scenarios. Recent advancement of Large Audio-Language Models (LALMs) has already brought some new insights to TS-ASR. However, significant room for optimization remains for the TS-ASR task within the LALMs architecture. While Chain of Thoughts (CoT) and Reinforcement Learning (RL) have proven effective in certain speech tasks, TS-ASR, which requires the model to deeply comprehend speech signals, differentiate various speakers, and handle overlapping utterances is particularly well-suited to a reasoning-guided approach. Therefore, we propose a novel framework that incorporates CoT and RL training into TS-ASR for performance improvement. A novel CoT dataset of TS-ASR is constructed, and the TS-ASR model is first trained on regular data and then fine-tuned on CoT data. Finally, the model is further trained with RL using selected data to enhance generalized reasoning capabilities. Experiment results demonstrate a significant improvement of TS-ASR performance with CoT and RL training, establishing a state-of-the-art performance compared with previous works of TS-ASR on comparable datasets.
#### LibriTTS-VI: A Public Corpus and Novel Methods for Efficient Voice Impression Control
 - **Authors:** Junki Ohmura, Yuki Ito, Emiru Tsunoo, Toshiyuki Sekiya, Toshiyuki Kumakura
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2509.15626

 - **Pdf link:** https://arxiv.org/pdf/2509.15626

 - **Abstract**
 Fine-grained control over voice impressions (e.g., making a voice brighter or calmer) is a key frontier for creating more controllable text-to-speech. However, this nascent field faces two key challenges. The first is the problem of impression leakage, where the synthesized voice is undesirably influenced by the speaker's reference audio, rather than the separately specified target impression, and the second is the lack of a public, annotated corpus. To mitigate impression leakage, we propose two methods: 1) a training strategy that separately uses an utterance for speaker identity and another utterance of the same speaker for target impression, and 2) a novel reference-free model that generates a speaker embedding solely from the target impression, achieving the benefits of improved robustness against the leakage and the convenience of reference-free generation. Objective and subjective evaluations demonstrate a significant improvement in controllability. Our best method reduced the mean squared error of 11-dimensional voice impression vectors from 0.61 to 0.41 objectively and from 1.15 to 0.92 subjectively, while maintaining high fidelity. To foster reproducible research, we introduce LibriTTS-VI, the first public voice impression dataset released with clear annotation standards, built upon the LibriTTS-R corpus.
#### The Singing Voice Conversion Challenge 2025: From Singer Identity Conversion To Singing Style Conversion
 - **Authors:** Lester Phillip Violeta, Xueyao Zhang, Jiatong Shi, Yusuke Yasuda, Wen-Chin Huang, Zhizheng Wu, Tomoki Toda
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2509.15629

 - **Pdf link:** https://arxiv.org/pdf/2509.15629

 - **Abstract**
 We present the findings of the latest iteration of the Singing Voice Conversion Challenge, a scientific event aiming to compare and understand different voice conversion systems in a controlled environment. Compared to previous iterations which solely focused on converting the singer identity, this year we also focused on converting the singing style of the singer. To create a controlled environment and thorough evaluations, we developed a new challenge database, introduced two tasks, open-sourced baselines, and conducted large-scale crowd-sourced listening tests and objective evaluations. The challenge was ran for two months and in total we evaluated 26 different systems. The results of the large-scale crowd-sourced listening test showed that top systems had comparable singer identity scores to ground truth samples. However, modeling the singing style and consequently achieving high naturalness still remains a challenge in this task, primarily due to the difficulty in modeling dynamic information in breathy, glissando, and vibrato singing styles.
#### EMO-RL: Emotion-Rule-Based Reinforcement Learning Enhanced Audio-Language Model for Generalized Speech Emotion Recognition
 - **Authors:** Pengcheng Li, Botao Zhao, Zuheng Kang, Junqing Peng, Xiaoyang Qu, Yayun He, Jianzong Wang
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2509.15654

 - **Pdf link:** https://arxiv.org/pdf/2509.15654

 - **Abstract**
 Although Large Audio-Language Models (LALMs) have exhibited outstanding performance in auditory understanding, their performance in affective computing scenarios, particularly in emotion recognition, reasoning, and subtle sentiment differentiation, remains suboptimal. Recent advances in Reinforcement Learning (RL) have shown promise in improving LALMs' reasoning abilities. However, two critical challenges hinder the direct application of RL techniques to Speech Emotion Recognition (SER) tasks: (1) convergence instability caused by ambiguous emotional boundaries and (2) limited reasoning ability when using relatively small models (e.g., 7B-parameter architectures). To overcome these limitations, we introduce EMO-RL, a novel framework incorporating reinforcement learning with two key innovations: Emotion Similarity-Weighted Reward (ESWR) and Explicit Structured Reasoning (ESR). Built upon pretrained LALMs, our method employs group-relative policy optimization with emotion constraints. Comprehensive experiments demonstrate that our EMO-RL training strategies can significantly enhance the emotional reasoning capabilities of LALMs, attaining state-of-the-art results on both the MELD and IEMOCAP datasets, and cross-dataset experiments prove the strong superiority of generalization.
#### Layer-wise Minimal Pair Probing Reveals Contextual Grammatical-Conceptual Hierarchy in Speech Representations
 - **Authors:** Linyang He, Qiaolin Wang, Xilin Jiang, Nima Mesgarani
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2509.15655

 - **Pdf link:** https://arxiv.org/pdf/2509.15655

 - **Abstract**
 Transformer-based speech language models (SLMs) have significantly improved neural speech recognition and understanding. While existing research has examined how well SLMs encode shallow acoustic and phonetic features, the extent to which SLMs encode nuanced syntactic and conceptual features remains unclear. By drawing parallels with linguistic competence assessments for large language models, this study is the first to systematically evaluate the presence of contextual syntactic and semantic features across SLMs for self-supervised learning (S3M), automatic speech recognition (ASR), speech compression (codec), and as the encoder for auditory large language models (AudioLLMs). Through minimal pair designs and diagnostic feature analysis across 71 tasks spanning diverse linguistic levels, our layer-wise and time-resolved analysis uncovers that 1) all speech encode grammatical features more robustly than conceptual ones.
#### TISDiSS: A Training-Time and Inference-Time Scalable Framework for Discriminative Source Separation
 - **Authors:** Yongsheng Feng, Yuetonghui Xu, Jiehui Luo, Hongjia Liu, Xiaobing Li, Feng Yu, Wei Li
 - **Subjects:** Subjects:
Sound (cs.SD); Artificial Intelligence (cs.AI); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2509.15666

 - **Pdf link:** https://arxiv.org/pdf/2509.15666

 - **Abstract**
 Source separation is a fundamental task in speech, music, and audio processing, and it also provides cleaner and larger data for training generative models. However, improving separation performance in practice often depends on increasingly large networks, inflating training and deployment costs. Motivated by recent advances in inference-time scaling for generative modeling, we propose Training-Time and Inference-Time Scalable Discriminative Source Separation (TISDiSS), a unified framework that integrates early-split multi-loss supervision, shared-parameter design, and dynamic inference repetitions. TISDiSS enables flexible speed-performance trade-offs by adjusting inference depth without retraining additional models. We further provide systematic analyses of architectural and training choices and show that training with more inference repetitions improves shallow-inference performance, benefiting low-latency applications. Experiments on standard speech separation benchmarks demonstrate state-of-the-art performance with a reduced parameter count, establishing TISDiSS as a scalable and practical framework for adaptive source separation.
#### VOX-KRIKRI: Unifying Speech and Language through Continuous Fusion
 - **Authors:** Dimitrios Damianos, Leon Voukoutis, Georgios Paraskevopoulos, Vassilis Katsouros
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2509.15667

 - **Pdf link:** https://arxiv.org/pdf/2509.15667

 - **Abstract**
 We present a multimodal fusion framework that bridges pre-trained decoder-based large language models (LLM) and acoustic encoder-decoder architectures such as Whisper, with the aim of building speech-enabled LLMs. Instead of directly using audio embeddings, we explore an intermediate audio-conditioned text space as a more effective mechanism for alignment. Our method operates fully in continuous text representation spaces, fusing Whisper's hidden decoder states with those of an LLM through cross-modal attention, and supports both offline and streaming modes. We introduce \textit{VoxKrikri}, the first Greek speech LLM, and show through analysis that our approach effectively aligns representations across modalities. These results highlight continuous space fusion as a promising path for multilingual and low-resource speech LLMs, while achieving state-of-the-art results for Automatic Speech Recognition in Greek, providing an average $\sim20\%$ relative improvement across benchmarks.
#### Interpretable Modeling of Articulatory Temporal Dynamics from real-time MRI for Phoneme Recognition
 - **Authors:** Jay Park, Hong Nguyen, Sean Foley, Jihwan Lee, Yoonjeong Lee, Dani Byrd, Shrikanth Narayanan
 - **Subjects:** Subjects:
Image and Video Processing (eess.IV); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2509.15689

 - **Pdf link:** https://arxiv.org/pdf/2509.15689

 - **Abstract**
 Real-time Magnetic Resonance Imaging (rtMRI) visualizes vocal tract action, offering a comprehensive window into speech articulation. However, its signals are high dimensional and noisy, hindering interpretation. We investigate compact representations of spatiotemporal articulatory dynamics for phoneme recognition from midsagittal vocal tract rtMRI videos. We compare three feature types: (1) raw video, (2) optical flow, and (3) six linguistically-relevant regions of interest (ROIs) for articulator movements. We evaluate models trained independently on each representation, as well as multi-feature combinations. Results show that multi-feature models consistently outperform single-feature baselines, with the lowest phoneme error rate (PER) of 0.34 obtained by combining ROI and raw video. Temporal fidelity experiments demonstrate a reliance on fine-grained articulatory dynamics, while ROI ablation studies reveal strong contributions from tongue and lips. Our findings highlight how rtMRI-derived features provide accuracy and interpretability, and establish strategies for leveraging articulatory data in speech processing.
#### Direct Simultaneous Translation Activation for Large Audio-Language Models
 - **Authors:** Pei Zhang, Yiming Wang, Jialong Tang, Baosong Yang, Rui Wang, Derek F. Wong, Fei Huang
 - **Subjects:** Subjects:
Sound (cs.SD); Computation and Language (cs.CL); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2509.15692

 - **Pdf link:** https://arxiv.org/pdf/2509.15692

 - **Abstract**
 Simultaneous speech-to-text translation (Simul-S2TT) aims to translate speech into target text in real time, outputting translations while receiving source speech input, rather than waiting for the entire utterance to be spoken. Simul-S2TT research often modifies model architectures to implement read-write strategies. However, with the rise of large audio-language models (LALMs), a key challenge is how to directly activate Simul-S2TT capabilities in base models without additional architectural changes. In this paper, we introduce {\bf Simul}taneous {\bf S}elf-{\bf A}ugmentation ({\bf SimulSA}), a strategy that utilizes LALMs' inherent capabilities to obtain simultaneous data by randomly truncating speech and constructing partially aligned translation. By incorporating them into offline SFT data, SimulSA effectively bridges the distribution gap between offline translation during pretraining and simultaneous translation during inference. Experimental results demonstrate that augmenting only about {\bf 1\%} of the simultaneous data, compared to the full offline SFT data, can significantly activate LALMs' Simul-S2TT capabilities without modifications to model architecture or decoding strategy.
#### Fine-Tuning Large Multimodal Models for Automatic Pronunciation Assessment
 - **Authors:** Ke Wang, Wenning Wei, Yan Deng, Lei He, Sheng Zhao
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2509.15701

 - **Pdf link:** https://arxiv.org/pdf/2509.15701

 - **Abstract**
 Automatic Pronunciation Assessment (APA) is critical for Computer-Assisted Language Learning (CALL), requiring evaluation across multiple granularities and aspects. Large Multimodal Models (LMMs) present new opportunities for APA, but their effectiveness in fine-grained assessment remains uncertain. This work investigates fine-tuning LMMs for APA using the Speechocean762 dataset and a private corpus. Fine-tuning significantly outperforms zero-shot settings and achieves competitive results on single-granularity tasks compared to public and commercial systems. The model performs well at word and sentence levels, while phoneme-level assessment remains challenging. We also observe that the Pearson Correlation Coefficient (PCC) reaches 0.9, whereas Spearman's rank Correlation Coefficient (SCC) remains around 0.6, suggesting that SCC better reflects ordinal consistency. These findings highlight both the promise and limitations of LMMs for APA and point to future work on fine-grained modeling and rank-aware evaluation.
#### Fed-PISA: Federated Voice Cloning via Personalized Identity-Style Adaptation
 - **Authors:** Qi Wang, Shituo Ma, Guoxin Yu, Hanyang Peng, Yue Yu
 - **Subjects:** Subjects:
Sound (cs.SD); Artificial Intelligence (cs.AI); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2509.16010

 - **Pdf link:** https://arxiv.org/pdf/2509.16010

 - **Abstract**
 Voice cloning for Text-to-Speech (TTS) aims to generate expressive and personalized speech from text using limited data from a target speaker. Federated Learning (FL) offers a collaborative and privacy-preserving framework for this task, but existing approaches suffer from high communication costs and tend to suppress stylistic heterogeneity, resulting in insufficient personalization. To address these issues, we propose Fed-PISA, which stands for Federated Personalized Identity-Style Adaptation. To minimize communication costs, Fed-PISA introduces a disentangled Low-Rank Adaptation (LoRA) mechanism: the speaker's timbre is retained locally through a private ID-LoRA, while only a lightweight style-LoRA is transmitted to the server, thereby minimizing parameter exchange. To harness heterogeneity, our aggregation method, inspired by collaborative filtering, is introduced to create custom models for each client by learning from stylistically similar peers. Experiments show that Fed-PISA improves style expressivity, naturalness, and speaker similarity, outperforming standard federated baselines with minimal communication costs.


by Zyzzyva0381 (Windy). 


2025-09-22
