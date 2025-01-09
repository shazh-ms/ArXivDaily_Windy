# Showing new listings for Thursday, 9 January 2025
Auto update papers at about 2:30am UTC (10:30am Beijing time) every weekday.


阅读 `Usage.md`了解如何使用此repo实现个性化的Arxiv论文推送

See `Usage.md` for instructions on how to personalize the repo. 


Keyword list: ['acoustic', 'reinforcement learning', 'reverb', 'meta', 'separate', 'reconstruction', 'noise', 'enhance', 'localization', 'speech']


Excluded: []


### Today: 9papers 
#### Artifact-free Sound Quality in DNN-based Closed-loop Systems for Audio Processing
 - **Authors:** chuan Wen, Guy Torfs, Sarah Verhulst
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2501.04116

 - **Pdf link:** https://arxiv.org/pdf/2501.04116

 - **Abstract**
 Recent advances in deep neural networks (DNNs) have significantly improved various audio processing applications, including speech enhancement, synthesis, and hearing aid algorithms. DNN-based closed-loop systems have gained popularity in these applications due to their robust performance and ability to adapt to diverse conditions. Despite their effectiveness, current DNN-based closed-loop systems often suffer from sound quality degradation caused by artifacts introduced by suboptimal sampling methods. To address this challenge, we introduce dCoNNear, a novel DNN architecture designed for seamless integration into closed-loop frameworks. This architecture specifically aims to prevent the generation of spurious artifacts. We demonstrate the effectiveness of dCoNNear through a proof-of-principle example within a closed-loop framework that employs biophysically realistic models of auditory processing for both normal and hearing-impaired profiles to design personalized hearing aid algorithms. Our results show that dCoNNear not only accurately simulates all processing stages of existing non-DNN biophysical models but also eliminates audible artifacts, thereby enhancing the sound quality of the resulting hearing aid algorithms. This study presents a novel, artifact-free closed-loop framework that improves the sound quality of audio processing systems, offering a promising solution for high-fidelity applications in audio and hearing technologies.
#### Decoding EEG Speech Perception with Transformers and VAE-based Data Augmentation
 - **Authors:** Terrance Yu-Hao Chen, Yulin Chen, Pontus Soederhaell, Sadrishya Agrawal, Kateryna Shapovalenko
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Computation and Language (cs.CL); Human-Computer Interaction (cs.HC); Machine Learning (cs.LG); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2501.04359

 - **Pdf link:** https://arxiv.org/pdf/2501.04359

 - **Abstract**
 Decoding speech from non-invasive brain signals, such as electroencephalography (EEG), has the potential to advance brain-computer interfaces (BCIs), with applications in silent communication and assistive technologies for individuals with speech impairments. However, EEG-based speech decoding faces major challenges, such as noisy data, limited datasets, and poor performance on complex tasks like speech perception. This study attempts to address these challenges by employing variational autoencoders (VAEs) for EEG data augmentation to improve data quality and applying a state-of-the-art (SOTA) sequence-to-sequence deep learning architecture, originally successful in electromyography (EMG) tasks, to EEG-based speech decoding. Additionally, we adapt this architecture for word classification tasks. Using the Brennan dataset, which contains EEG recordings of subjects listening to narrated speech, we preprocess the data and evaluate both classification and sequence-to-sequence models for EEG-to-words/sentences tasks. Our experiments show that VAEs have the potential to reconstruct artificial EEG data for augmentation. Meanwhile, our sequence-to-sequence model achieves more promising performance in generating sentences compared to our classification model, though both remain challenging tasks. These findings lay the groundwork for future research on EEG speech perception decoding, with possible extensions to speech production tasks such as silent or imagined speech.
#### ZSVC: Zero-shot Style Voice Conversion with Disentangled Latent Diffusion Models and Adversarial Training
 - **Authors:** Xinfa Zhu, Lei He, Yujia Xiao, Xi Wang, Xu Tan, Sheng Zhao, Lei Xie
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2501.04416

 - **Pdf link:** https://arxiv.org/pdf/2501.04416

 - **Abstract**
 Style voice conversion aims to transform the speaking style of source speech into a desired style while keeping the original speaker's identity. However, previous style voice conversion approaches primarily focus on well-defined domains such as emotional aspects, limiting their practical applications. In this study, we present ZSVC, a novel Zero-shot Style Voice Conversion approach that utilizes a speech codec and a latent diffusion model with speech prompting mechanism to facilitate in-context learning for speaking style conversion. To disentangle speaking style and speaker timbre, we introduce information bottleneck to filter speaking style in the source speech and employ Uncertainty Modeling Adaptive Instance Normalization (UMAdaIN) to perturb the speaker timbre in the style prompt. Moreover, we propose a novel adversarial training strategy to enhance in-context learning and improve style similarity. Experiments conducted on 44,000 hours of speech data demonstrate the superior performance of ZSVC in generating speech with diverse speaking styles in zero-shot scenarios.
#### FleSpeech: Flexibly Controllable Speech Generation with Various Prompts
 - **Authors:** Hanzhao Li, Yuke Li, Xinsheng Wang, Jingbin Hu, Qicong Xie, Shan Yang, Lei Xie
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2501.04644

 - **Pdf link:** https://arxiv.org/pdf/2501.04644

 - **Abstract**
 Controllable speech generation methods typically rely on single or fixed prompts, hindering creativity and flexibility. These limitations make it difficult to meet specific user needs in certain scenarios, such as adjusting the style while preserving a selected speaker's timbre, or choosing a style and generating a voice that matches a character's visual appearance. To overcome these challenges, we propose \textit{FleSpeech}, a novel multi-stage speech generation framework that allows for more flexible manipulation of speech attributes by integrating various forms of control. FleSpeech employs a multimodal prompt encoder that processes and unifies different text, audio, and visual prompts into a cohesive representation. This approach enhances the adaptability of speech synthesis and supports creative and precise control over the generated speech. Additionally, we develop a data collection pipeline for multimodal datasets to facilitate further research and applications in this field. Comprehensive subjective and objective experiments demonstrate the effectiveness of FleSpeech. Audio samples are available at this https URL
#### Listening and Seeing Again: Generative Error Correction for Audio-Visual Speech Recognition
 - **Authors:** Rui Liu, Hongyu Yuan, Haizhou Li
 - **Subjects:** Subjects:
Multimedia (cs.MM); Artificial Intelligence (cs.AI); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2501.04038

 - **Pdf link:** https://arxiv.org/pdf/2501.04038

 - **Abstract**
 Unlike traditional Automatic Speech Recognition (ASR), Audio-Visual Speech Recognition (AVSR) takes audio and visual signals simultaneously to infer the transcription. Recent studies have shown that Large Language Models (LLMs) can be effectively used for Generative Error Correction (GER) in ASR by predicting the best transcription from ASR-generated N-best hypotheses. However, these LLMs lack the ability to simultaneously understand audio and visual, making the GER approach challenging to apply in AVSR. In this work, we propose a novel GER paradigm for AVSR, termed AVGER, that follows the concept of ``listening and seeing again''. Specifically, we first use the powerful AVSR system to read the audio and visual signals to get the N-Best hypotheses, and then use the Q-former-based Multimodal Synchronous Encoder to read the audio and visual information again and convert them into an audio and video compression representation respectively that can be understood by LLM. Afterward, the audio-visual compression representation and the N-Best hypothesis together constitute a Cross-modal Prompt to guide the LLM in producing the best transcription. In addition, we also proposed a Multi-Level Consistency Constraint training criterion, including logits-level, utterance-level and representations-level, to improve the correction accuracy while enhancing the interpretability of audio and visual compression representations. The experimental results on the LRS3 dataset show that our method outperforms current mainstream AVSR systems. The proposed AVGER can reduce the Word Error Rate (WER) by 24% compared to them. Code and models can be found at: this https URL.
#### DrawSpeech: Expressive Speech Synthesis Using Prosodic Sketches as Control Conditions
 - **Authors:** Weidong Chen, Shan Yang, Guangzhi Li, Xixin Wu
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2501.04256

 - **Pdf link:** https://arxiv.org/pdf/2501.04256

 - **Abstract**
 Controlling text-to-speech (TTS) systems to synthesize speech with the prosodic characteristics expected by users has attracted much attention. To achieve controllability, current studies focus on two main directions: (1) using reference speech as prosody prompt to guide speech synthesis, and (2) using natural language descriptions to control the generation process. However, finding reference speech that exactly contains the prosody that users want to synthesize takes a lot of effort. Description-based guidance in TTS systems can only determine the overall prosody, which has difficulty in achieving fine-grained prosody control over the synthesized speech. In this paper, we propose DrawSpeech, a sketch-conditioned diffusion model capable of generating speech based on any prosody sketches drawn by users. Specifically, the prosody sketches are fed to DrawSpeech to provide a rough indication of the expected prosody trends. DrawSpeech then recovers the detailed pitch and energy contours based on the coarse sketches and synthesizes the desired speech. Experimental results show that DrawSpeech can generate speech with a wide variety of prosody and can precisely control the fine-grained prosody in a user-friendly manner. Our implementation and audio samples are publicly available.
#### MAD-UV: The 1st INTERSPEECH Mice Autism Detection via Ultrasound Vocalization Challenge
 - **Authors:** Zijiang Yang, Meishu Song, Xin Jing, Haojie Zhang, Kun Qian, Bin Hu, Kota Tamada, Toru Takumi, Björn W. Schuller, Yoshiharu Yamamoto
 - **Subjects:** Subjects:
Sound (cs.SD); Artificial Intelligence (cs.AI); Machine Learning (cs.LG); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2501.04292

 - **Pdf link:** https://arxiv.org/pdf/2501.04292

 - **Abstract**
 The Mice Autism Detection via Ultrasound Vocalization (MAD-UV) Challenge introduces the first INTERSPEECH challenge focused on detecting autism spectrum disorder (ASD) in mice through their vocalizations. Participants are tasked with developing models to automatically classify mice as either wild-type or ASD models based on recordings with a high sampling rate. Our baseline system employs a simple CNN-based classification using three different spectrogram features. Results demonstrate the feasibility of automated ASD detection, with the considered audible-range features achieving the best performance (UAR of 0.600 for segment-level and 0.625 for subject-level classification). This challenge bridges speech technology and biomedical research, offering opportunities to advance our understanding of ASD models through machine learning approaches. The findings suggest promising directions for vocalization analysis and highlight the potential value of audible and ultrasound vocalizations in ASD detection.
#### Phone-purity Guided Discrete Tokens for Dysarthric Speech Recognition
 - **Authors:** Huimeng Wang, Xurong Xie, Mengzhe Geng, Shujie Hu, Haoning Xu, Youjun Chen, Zhaoqing Li, Jiajun Deng, Xunying Liu
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2501.04379

 - **Pdf link:** https://arxiv.org/pdf/2501.04379

 - **Abstract**
 Discrete tokens extracted provide efficient and domain adaptable speech features. Their application to disordered speech that exhibits articulation imprecision and large mismatch against normal voice remains unexplored. To improve their phonetic discrimination that is weakened during unsupervised K-means or vector quantization of continuous features, this paper proposes novel phone-purity guided (PPG) discrete tokens for dysarthric speech recognition. Phonetic label supervision is used to regularize maximum likelihood and reconstruction error costs used in standard K-means and VAE-VQ based discrete token extraction. Experiments conducted on the UASpeech corpus suggest that the proposed PPG discrete token features extracted from HuBERT consistently outperform hybrid TDNN and End-to-End (E2E) Conformer systems using non-PPG based K-means or VAE-VQ tokens across varying codebook sizes by statistically significant word error rate (WER) reductions up to 0.99\% and 1.77\% absolute (3.21\% and 4.82\% relative) respectively on the UASpeech test set of 16 dysarthric speakers. The lowest WER of 23.25\% was obtained by combining systems using different token features. Consistent improvements on the phone purity metric were also achieved. T-SNE visualization further demonstrates sharper decision boundaries were produced between K-means/VAE-VQ clusters after introducing phone-purity guidance.
#### Right Label Context in End-to-End Training of Time-Synchronous ASR Models
 - **Authors:** Tina Raissi, Ralf Schlter, Hermann Ney
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2501.04521

 - **Pdf link:** https://arxiv.org/pdf/2501.04521

 - **Abstract**
 Current time-synchronous sequence-to-sequence automatic speech recognition (ASR) models are trained by using sequence level cross-entropy that sums over all alignments. Due to the discriminative formulation, incorporating the right label context into the training criterion's gradient causes normalization problems and is not mathematically well-defined. The classic hybrid neural network hidden Markov model (NN-HMM) with its inherent generative formulation enables conditioning on the right label context. However, due to the HMM state-tying the identity of the right label context is never modeled explicitly. In this work, we propose a factored loss with auxiliary left and right label contexts that sums over all alignments. We show that the inclusion of the right label context is particularly beneficial when training data resources are limited. Moreover, we also show that it is possible to build a factored hybrid HMM system by relying exclusively on the full-sum criterion. Experiments were conducted on Switchboard 300h and LibriSpeech 960h.


by Zyzzyva0381 (Windy). 


2025-01-09
