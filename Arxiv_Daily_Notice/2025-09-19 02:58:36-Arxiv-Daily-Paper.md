# Showing new listings for Friday, 19 September 2025
Auto update papers at about 2:30am UTC (10:30am Beijing time) every weekday.


阅读 `Usage.md`了解如何使用此repo实现个性化的Arxiv论文推送

See `Usage.md` for instructions on how to personalize the repo. 


Keyword list: ['text-to-speech', 'text to speech', 'tts', 'LLM-based', 'speech', 'voice']


Excluded: []


### Today: 20papers 
#### SpeechOp: Inference-Time Task Composition for Generative Speech Processing
 - **Authors:** Justin Lovelace, Rithesh Kumar, Jiaqi Su, Ke Chen, Kilian Q Weinberger, Zeyu Jin
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Machine Learning (cs.LG)
 - **Arxiv link:** https://arxiv.org/abs/2509.14298

 - **Pdf link:** https://arxiv.org/pdf/2509.14298

 - **Abstract**
 While generative Text-to-Speech (TTS) systems leverage vast ``in-the-wild" data to achieve remarkable success, speech-to-speech processing tasks like enhancement face data limitations, which lead data-hungry generative approaches to distort speech content and speaker identity. To bridge this gap, we present SpeechOp, a multi-task latent diffusion model that transforms pre-trained TTS models into a universal speech processor capable of performing a wide range of speech tasks and composing them in novel ways at inference time. By adapting a pre-trained TTS model, SpeechOp inherits a rich understanding of natural speech, accelerating training and improving S2S task quality, while simultaneously enhancing core TTS performance. Finally, we introduce Implicit Task Composition (ITC), a novel pipeline where ASR-derived transcripts (e.g., from Whisper) guide SpeechOp's enhancement via our principled inference-time task composition. ITC achieves state-of-the-art content preservation by robustly combining web-scale speech understanding with SpeechOp's generative capabilities. Audio samples are available at this https URL
#### Diffusion-Based Unsupervised Audio-Visual Speech Separation in Noisy Environments with Noise Prior
 - **Authors:** Yochai Yemini, Rami Ben-Ari, Sharon Gannot, Ethan Fetaya
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Machine Learning (cs.LG)
 - **Arxiv link:** https://arxiv.org/abs/2509.14379

 - **Pdf link:** https://arxiv.org/pdf/2509.14379

 - **Abstract**
 In this paper, we address the problem of single-microphone speech separation in the presence of ambient noise. We propose a generative unsupervised technique that directly models both clean speech and structured noise components, training exclusively on these individual signals rather than noisy mixtures. Our approach leverages an audio-visual score model that incorporates visual cues to serve as a strong generative speech prior. By explicitly modelling the noise distribution alongside the speech distribution, we enable effective decomposition through the inverse problem paradigm. We perform speech separation by sampling from the posterior distributions via a reverse diffusion process, which directly estimates and removes the modelled noise component to recover clean constituent signals. Experimental results demonstrate promising performance, highlighting the effectiveness of our direct noise modelling approach in challenging acoustic environments.
#### Multi-Channel Differential ASR for Robust Wearer Speech Recognition on Smart Glasses
 - **Authors:** Yufeng Yang, Yiteng Huang, Yong Xu, Li Wan, Suwon Shon, Yang Liu, Yifeng Fan, Zhaojun Yang, Olivier Siohan, Yue Liu, Ming Sun, Florian Metze
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2509.14430

 - **Pdf link:** https://arxiv.org/pdf/2509.14430

 - **Abstract**
 With the growing adoption of wearable devices such as smart glasses for AI assistants, wearer speech recognition (WSR) is becoming increasingly critical to next-generation human-computer interfaces. However, in real environments, interference from side-talk speech remains a significant challenge to WSR and may cause accumulated errors for downstream tasks such as natural language processing. In this work, we introduce a novel multi-channel differential automatic speech recognition (ASR) method for robust WSR on smart glasses. The proposed system takes differential inputs from different frontends that complement each other to improve the robustness of WSR, including a beamformer, microphone selection, and a lightweight side-talk detection model. Evaluations on both simulated and real datasets demonstrate that the proposed system outperforms the traditional approach, achieving up to an 18.0% relative reduction in word error rate.
#### Mitigating Intra-Speaker Variability in Diarization with Style-Controllable Speech Augmentation
 - **Authors:** Miseul Kim, Soo Jin Park, Kyungguen Byun, Hyeon-Kyeong Shin, Sunkuk Moon, Shuhua Zhang, Erik Visser
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Artificial Intelligence (cs.AI); Signal Processing (eess.SP)
 - **Arxiv link:** https://arxiv.org/abs/2509.14632

 - **Pdf link:** https://arxiv.org/pdf/2509.14632

 - **Abstract**
 Speaker diarization systems often struggle with high intrinsic intra-speaker variability, such as shifts in emotion, health, or content. This can cause segments from the same speaker to be misclassified as different individuals, for example, when one raises their voice or speaks faster during conversation. To address this, we propose a style-controllable speech generation model that augments speech across diverse styles while preserving the target speaker's identity. The proposed system starts with diarized segments from a conventional diarizer. For each diarized segment, it generates augmented speech samples enriched with phonetic and stylistic diversity. And then, speaker embeddings from both the original and generated audio are blended to enhance the system's robustness in grouping segments with high intrinsic intra-speaker variability. We validate our approach on a simulated emotional speech dataset and the truncated AMI dataset, demonstrating significant improvements, with error rate reductions of 49% and 35% on each dataset, respectively.
#### SpeechMLC: Speech Multi-label Classification
 - **Authors:** Miseul Kim, Seyun Um, Hyeonjin Cha, Hong-goo Kang
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Signal Processing (eess.SP)
 - **Arxiv link:** https://arxiv.org/abs/2509.14677

 - **Pdf link:** https://arxiv.org/pdf/2509.14677

 - **Abstract**
 In this paper, we propose a multi-label classification framework to detect multiple speaking styles in a speech sample. Unlike previous studies that have primarily focused on identifying a single target style, our framework effectively captures various speaker characteristics within a unified structure, making it suitable for generalized human-computer interaction applications. The proposed framework integrates cross-attention mechanisms within a transformer decoder to extract salient features associated with each target label from the input speech. To mitigate the data imbalance inherent in multi-label speech datasets, we employ a data augmentation technique based on a speech generation model. We validate our model's effectiveness through multiple objective evaluations on seen and unseen corpora. In addition, we provide an analysis of the influence of human perception on classification accuracy by considering the impact of human labeling agreement on model performance.
#### DAIEN-TTS: Disentangled Audio Infilling for Environment-Aware Text-to-Speech Synthesis
 - **Authors:** Ye-Xin Lu, Yu Gu, Kun Wei, Hui-Peng Du, Yang Ai, Zhen-Hua Ling
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2509.14684

 - **Pdf link:** https://arxiv.org/pdf/2509.14684

 - **Abstract**
 This paper presents DAIEN-TTS, a zero-shot text-to-speech (TTS) framework that enables ENvironment-aware synthesis through Disentangled Audio Infilling. By leveraging separate speaker and environment prompts, DAIEN-TTS allows independent control over the timbre and the background environment of the synthesized speech. Built upon F5-TTS, the proposed DAIEN-TTS first incorporates a pretrained speech-environment separation (SES) module to disentangle the environmental speech into mel-spectrograms of clean speech and environment audio. Two random span masks of varying lengths are then applied to both mel-spectrograms, which, together with the text embedding, serve as conditions for infilling the masked environmental mel-spectrogram, enabling the simultaneous continuation of personalized speech and time-varying environmental audio. To further enhance controllability during inference, we adopt dual class-free guidance (DCFG) for the speech and environment components and introduce a signal-to-noise ratio (SNR) adaptation strategy to align the synthesized speech with the environment prompt. Experimental results demonstrate that DAIEN-TTS generates environmental personalized speech with high naturalness, strong speaker similarity, and high environmental fidelity.
#### MELA-TTS: Joint transformer-diffusion model with representation alignment for speech synthesis
 - **Authors:** Keyu An, Zhiyu Zhang, Changfeng Gao, Yabin Li, Zhendong Peng, Haoxu Wang, Zhihao Du, Han Zhao, Zhifu Gao, Xiangang Li
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2509.14784

 - **Pdf link:** https://arxiv.org/pdf/2509.14784

 - **Abstract**
 This work introduces MELA-TTS, a novel joint transformer-diffusion framework for end-to-end text-to-speech synthesis. By autoregressively generating continuous mel-spectrogram frames from linguistic and speaker conditions, our architecture eliminates the need for speech tokenization and multi-stage processing pipelines. To address the inherent difficulties of modeling continuous features, we propose a representation alignment module that aligns output representations of the transformer decoder with semantic embeddings from a pretrained ASR encoder during training. This mechanism not only speeds up training convergence, but also enhances cross-modal coherence between the textual and acoustic domains. Comprehensive experiments demonstrate that MELA-TTS achieves state-of-the-art performance across multiple evaluation metrics while maintaining robust zero-shot voice cloning capabilities, in both offline and streaming synthesis modes. Our results establish a new benchmark for continuous feature generation approaches in TTS, offering a compelling alternative to discrete-token-based paradigms.
#### Acoustic Simulation Framework for Multi-channel Replay Speech Detection
 - **Authors:** Michael Neri, Tuomas Virtanen
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Cryptography and Security (cs.CR); Sound (cs.SD); Signal Processing (eess.SP)
 - **Arxiv link:** https://arxiv.org/abs/2509.14789

 - **Pdf link:** https://arxiv.org/pdf/2509.14789

 - **Abstract**
 Replay speech attacks pose a significant threat to voice-controlled systems, especially in smart environments where voice assistants are widely deployed. While multi-channel audio offers spatial cues that can enhance replay detection robustness, existing datasets and methods predominantly rely on single-channel recordings. In this work, we introduce an acoustic simulation framework designed to simulate multi-channel replay speech configurations using publicly available resources. Our setup models both genuine and spoofed speech across varied environments, including realistic microphone and loudspeaker impulse responses, room acoustics, and noise conditions. The framework employs measured loudspeaker directionalities during the replay attack to improve the realism of the simulation. We define two spoofing settings, which simulate whether a reverberant or an anechoic speech is used in the replay scenario, and evaluate the impact of omnidirectional and diffuse noise on detection performance. Using the state-of-the-art M-ALRAD model for replay speech detection, we demonstrate that synthetic data can support the generalization capabilities of the detector across unseen enclosures.
#### AmbiDrop: Array-Agnostic Speech Enhancement Using Ambisonics Encoding and Dropout-Based Learning
 - **Authors:** Michael Tatarjitzky, Boaz Rafaely
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2509.14855

 - **Pdf link:** https://arxiv.org/pdf/2509.14855

 - **Abstract**
 Multichannel speech enhancement leverages spatial cues to improve intelligibility and quality, but most learning-based methods rely on specific microphone array geometry, unable to account for geometry changes. To mitigate this limitation, current array-agnostic approaches employ large multi-geometry datasets but may still fail to generalize to unseen layouts. We propose AmbiDrop (Ambisonics with Dropouts), an Ambisonics-based framework that encodes arbitrary array recordings into the spherical harmonics domain using Ambisonics Signal Matching (ASM). A deep neural network is trained on simulated Ambisonics data, combined with channel dropout for robustness against array-dependent encoding errors, therefore omitting the need for a diverse microphone array database. Experiments show that while the baseline and proposed models perform similarly on the training arrays, the baseline degrades on unseen arrays. In contrast, AmbiDrop consistently improves SI-SDR, PESQ, and STOI, demonstrating strong generalization and practical potential for array-agnostic speech enhancement.
#### SynParaSpeech: Automated Synthesis of Paralinguistic Datasets for Speech Generation and Understanding
 - **Authors:** Bingsong Bai, Qihang Lu, Wenbing Yang, Zihan Sun, YueRan Hou, Peilei Jia, Songbai Pu, Ruibo Fu, Yingming Gao, Ya Li, Jun Gao
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Computation and Language (cs.CL)
 - **Arxiv link:** https://arxiv.org/abs/2509.14946

 - **Pdf link:** https://arxiv.org/pdf/2509.14946

 - **Abstract**
 Paralinguistic sounds, like laughter and sighs, are crucial for synthesizing more realistic and engaging speech. However, existing methods typically depend on proprietary datasets, while publicly available resources often suffer from incomplete speech, inaccurate or missing timestamps, and limited real-world relevance. To address these problems, we propose an automated framework for generating large-scale paralinguistic data and apply it to construct the SynParaSpeech dataset. The dataset comprises 6 paralinguistic categories with 118.75 hours of data and precise timestamps, all derived from natural conversational speech. Our contributions lie in introducing the first automated method for constructing large-scale paralinguistic datasets and releasing the SynParaSpeech corpus, which advances speech generation through more natural paralinguistic synthesis and enhances speech understanding by improving paralinguistic event detection. The dataset and audio samples are available at this https URL.
#### Discrete optimal transport is a strong audio adversarial attack
 - **Authors:** Anton Selitskiy, Akib Shahriyar, Jishnuraj Prakasan
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Artificial Intelligence (cs.AI)
 - **Arxiv link:** https://arxiv.org/abs/2509.14959

 - **Pdf link:** https://arxiv.org/pdf/2509.14959

 - **Abstract**
 In this paper, we show that discrete optimal transport (DOT) is an effective black-box adversarial attack against modern audio anti-spoofing countermeasures (CMs). Our attack operates as a post-processing, distribution-alignment step: frame-level WavLM embeddings of generated speech are aligned to an unpaired bona fide pool via entropic OT and a top-$k$ barycentric projection, then decoded with a neural vocoder. Evaluated on ASVspoof2019 and ASVspoof5 with AASIST baselines, DOT yields consistently high equal error rate (EER) across datasets and remains competitive after CM fine-tuning, outperforming several conventional attacks in cross-dataset transfer. Ablation analysis highlights the practical impact of vocoder overlap. Results indicate that distribution-level alignment is a powerful and stable attack surface for deployed CMs.
#### BabyHuBERT: Multilingual Self-Supervised Learning for Segmenting Speakers in Child-Centered Long-Form Recordings
 - **Authors:** ThÃ©o Charlot, Tarek Kunze, Maxime Poli, Alejandrina Cristia, Emmanuel Dupoux, Marvin Lavechin
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Machine Learning (cs.LG); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2509.15001

 - **Pdf link:** https://arxiv.org/pdf/2509.15001

 - **Abstract**
 Child-centered long-form recordings are essential for studying early language development, but existing speech models trained on clean adult data perform poorly due to acoustic and linguistic differences. We introduce BabyHuBERT, the first self-supervised speech representation model trained on 13,000 hours of multilingual child-centered long-form recordings spanning over 40 languages. We evaluate BabyHuBERT on speaker segmentation, identifying when target children speak versus female adults, male adults, or other children -- a fundamental preprocessing step for analyzing naturalistic language experiences. BabyHuBERT achieves F1-scores from 52.1% to 74.4% across six diverse datasets, consistently outperforming W2V2-LL4300 (trained on English long-forms) and standard HuBERT (trained on clean adult speech). Notable improvements include 13.2 absolute F1 points over HuBERT on Vanuatu and 15.9 points on Solomon Islands corpora, demonstrating effectiveness on underrepresented languages. By sharing code and models, BabyHuBERT serves as a foundation model for child speech research, enabling fine-tuning on diverse downstream tasks.
#### From Who Said What to Who They Are: Modular Training-free Identity-Aware LLM Refinement of Speaker Diarization
 - **Authors:** Yu-Wen Chen, William Ho, Maxim Topaz, Julia Hirschberg, Zoran Kostic
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2509.15082

 - **Pdf link:** https://arxiv.org/pdf/2509.15082

 - **Abstract**
 Speaker diarization (SD) struggles in real-world scenarios due to dynamic environments and unknown speaker counts. SD is rarely used alone and is often paired with automatic speech recognition (ASR), but non-modular methods that jointly train on domain-specific data have limited flexibility. Moreover, many applications require true speaker identities rather than SD's pseudo labels. We propose a training-free modular pipeline combining off-the-shelf SD, ASR, and a large language model (LLM) to determine who spoke, what was said, and who they are. Using structured LLM prompting on reconciled SD and ASR outputs, our method leverages semantic continuity in conversational context to refine low-confidence speaker labels and assigns role identities while correcting split speakers. On a real-world patient-clinician dataset, our approach achieves a 29.7% relative error reduction over baseline reconciled SD and ASR. It enhances diarization performance without additional training and delivers a complete pipeline for SD, ASR, and speaker identity detection in practical applications.
#### Real-Time Streaming Mel Vocoding with Generative Flow Matching
 - **Authors:** Simon Welker, Tal Peer, Timo Gerkmann
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Machine Learning (cs.LG); Sound (cs.SD); Signal Processing (eess.SP)
 - **Arxiv link:** https://arxiv.org/abs/2509.15085

 - **Pdf link:** https://arxiv.org/pdf/2509.15085

 - **Abstract**
 The task of Mel vocoding, i.e., the inversion of a Mel magnitude spectrogram to an audio waveform, is still a key component in many text-to-speech (TTS) systems today. Based on generative flow matching, our prior work on generative STFT phase retrieval (DiffPhase), and the pseudoinverse operator of the Mel filterbank, we develop MelFlow, a streaming-capable generative Mel vocoder for speech sampled at 16 kHz with an algorithmic latency of only 32 ms and a total latency of 48 ms. We show real-time streaming capability at this latency not only in theory, but in practice on a consumer laptop GPU. Furthermore, we show that our model achieves substantially better PESQ and SI-SDR values compared to well-established not streaming-capable baselines for Mel vocoding including HiFi-GAN.
#### Listening, Imagining \& Refining: A Heuristic Optimized ASR Correction Framework with LLMs
 - **Authors:** Yutong Liu, Ziyue Zhang, Yongbin Yu, Xiangxiang Wang, Yuqing Cai, Nyima Tashi
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Artificial Intelligence (cs.AI)
 - **Arxiv link:** https://arxiv.org/abs/2509.15095

 - **Pdf link:** https://arxiv.org/pdf/2509.15095

 - **Abstract**
 Automatic Speech Recognition (ASR) systems remain prone to errors that affect downstream applications. In this paper, we propose LIR-ASR, a heuristic optimized iterative correction framework using LLMs, inspired by human auditory perception. LIR-ASR applies a "Listening-Imagining-Refining" strategy, generating phonetic variants and refining them in context. A heuristic optimization with finite state machine (FSM) is introduced to prevent the correction process from being trapped in local optima and rule-based constraints help maintain semantic fidelity. Experiments on both English and Chinese ASR outputs show that LIR-ASR achieves average reductions in CER/WER of up to 1.5 percentage points compared to baselines, demonstrating substantial accuracy gains in transcription.
#### Context-Enhanced Granular Edit Representation for Efficient and Accurate ASR Post-editing
 - **Authors:** Luan Vejsiu, Qianyu Zheng, Haoxuan Chen, Yizhou Han
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2509.14263

 - **Pdf link:** https://arxiv.org/pdf/2509.14263

 - **Abstract**
 Despite ASR technology being full-scale adopted by industry and for large portions of the population, ASR systems often have errors that require editors to post-edit text quality. While LLMs are powerful post-editing tools, baseline full rewrite models have inference inefficiencies because they often generate the same redundant text over and over again. Compact edit representations have existed but often lack the efficacy and context required for optimal accuracy. This paper introduces CEGER (Context-Enhanced Granular Edit Representation), a compact edit representation that was generated for highly accurate, efficient ASR post-editing. CEGER allows LLMs to generate a sequence of structured, fine-grained, contextually rich commands to modify the original ASR output. A separate expansion module deterministically reconstructs the corrected text based on the commands. Extensive experiments on the LibriSpeech dataset that were conducted, CEGER achieves state-of-the-art accuracy, achieving the lowest word error rate (WER) versus full rewrite and prior compact representations.
#### SpeechWeave: Diverse Multilingual Synthetic Text & Audio Data Generation Pipeline for Training Text to Speech Models
 - **Authors:** Karan Dua, Puneet Mittal, Ranjeet Gupta, Hitesh Laxmichand Patel
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Artificial Intelligence (cs.AI); Machine Learning (cs.LG); Multimedia (cs.MM); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2509.14270

 - **Pdf link:** https://arxiv.org/pdf/2509.14270

 - **Abstract**
 High-quality Text-to-Speech (TTS) model training requires extensive and diverse text and speech data. It is challenging to procure such data from real sources due to issues of domain specificity, licensing, and scalability. Large language models (LLMs) can certainly generate textual data, but they create repetitive text with insufficient variation in the prompt during the generation process. Another important aspect in TTS training data is text normalization. Tools for normalization might occasionally introduce anomalies or overlook valuable patterns, and thus impact data quality. Furthermore, it is also impractical to rely on voice artists for large scale speech recording in commercial TTS systems with standardized voices. To address these challenges, we propose SpeechWeave, a synthetic speech data generation pipeline that is capable of automating the generation of multilingual, domain-specific datasets for training TTS models. Our experiments reveal that our pipeline generates data that is 10-48% more diverse than the baseline across various linguistic and phonetic metrics, along with speaker-standardized speech audio while generating approximately 97% correctly normalized text. Our approach enables scalable, high-quality data generation for TTS training, improving diversity, normalization, and voice consistency in the generated datasets.
#### Deploying UDM Series in Real-Life Stuttered Speech Applications: A Clinical Evaluation Framework
 - **Authors:** Eric Zhang, Li Wei, Sarah Chen, Michael Wang (SSHealth Team, AI for Healthcare Laboratory)
 - **Subjects:** Subjects:
Sound (cs.SD); Artificial Intelligence (cs.AI); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2509.14304

 - **Pdf link:** https://arxiv.org/pdf/2509.14304

 - **Abstract**
 Stuttered and dysfluent speech detection systems have traditionally suffered from the trade-off between accuracy and clinical interpretability. While end-to-end deep learning models achieve high performance, their black-box nature limits clinical adoption. This paper looks at the Unconstrained Dysfluency Modeling (UDM) series-the current state-of-the-art framework developed by Berkeley that combines modular architecture, explicit phoneme alignment, and interpretable outputs for real-world clinical deployment. Through extensive experiments involving patients and certified speech-language pathologists (SLPs), we demonstrate that UDM achieves state-of-the-art performance (F1: 0.89+-0.04) while providing clinically meaningful interpretability scores (4.2/5.0). Our deployment study shows 87% clinician acceptance rate and 34% reduction in diagnostic time. The results provide strong evidence that UDM represents a practical pathway toward AI-assisted speech therapy in clinical environments.
#### From Turn-Taking to Synchronous Dialogue: A Survey of Full-Duplex Spoken Language Models
 - **Authors:** Yuxuan Chen, Haoyuan Yu
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2509.14515

 - **Pdf link:** https://arxiv.org/pdf/2509.14515

 - **Abstract**
 True Full-Duplex (TFD) voice communication--enabling simultaneous listening and speaking with natural turn-taking, overlapping speech, and interruptions--represents a critical milestone toward human-like AI interaction. This survey comprehensively reviews Full-Duplex Spoken Language Models (FD-SLMs) in the LLM era. We establish a taxonomy distinguishing Engineered Synchronization (modular architectures) from Learned Synchronization (end-to-end architectures), and unify fragmented evaluation approaches into a framework encompassing Temporal Dynamics, Behavioral Arbitration, Semantic Coherence, and Acoustic Performance. Through comparative analysis of mainstream FD-SLMs, we identify fundamental challenges: synchronous data scarcity, architectural divergence, and evaluation gaps, providing a roadmap for advancing human-AI communication.
#### How Does Instrumental Music Help SingFake Detection?
 - **Authors:** Xuanjun Chen, Chia-Yu Hu, I-Ming Lin, Yi-Cheng Lin, I-Hsiang Chiu, You Zhang, Sung-Feng Huang, Yi-Hsuan Yang, Haibin Wu, Hung-yi Lee, Jyh-Shing Roger Jang
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS); Signal Processing (eess.SP)
 - **Arxiv link:** https://arxiv.org/abs/2509.14675

 - **Pdf link:** https://arxiv.org/pdf/2509.14675

 - **Abstract**
 Although many models exist to detect singing voice deepfakes (SingFake), how these models operate, particularly with instrumental accompaniment, is unclear. We investigate how instrumental music affects SingFake detection from two perspectives. To investigate the behavioral effect, we test different backbones, unpaired instrumental tracks, and frequency subbands. To analyze the representational effect, we probe how fine-tuning alters encoders' speech and music capabilities. Our results show that instrumental accompaniment acts mainly as data augmentation rather than providing intrinsic cues (e.g., rhythm or harmony). Furthermore, fine-tuning increases reliance on shallow speaker features while reducing sensitivity to content, paralinguistic, and semantic information. These insights clarify how models exploit vocal versus instrumental cues and can inform the design of more interpretable and robust SingFake detection systems.


by Zyzzyva0381 (Windy). 


2025-09-19
