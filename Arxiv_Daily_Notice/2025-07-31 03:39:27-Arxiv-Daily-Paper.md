# Showing new listings for Thursday, 31 July 2025
Auto update papers at about 2:30am UTC (10:30am Beijing time) every weekday.


阅读 `Usage.md`了解如何使用此repo实现个性化的Arxiv论文推送

See `Usage.md` for instructions on how to personalize the repo. 


Keyword list: ['text-to-speech', 'text to speech', 'tts', 'LLM-based', 'speech', 'voice']


Excluded: []


### Today: 6papers 
#### Tiny Noise-Robust Voice Activity Detector for Voice Assistants
 - **Authors:** Hamed Jafarzadeh Asl, Mahsa Ghazvini Nejad, Amin Edraki, Masoud Asgharian, Vahid Partovi Nia
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Artificial Intelligence (cs.AI)
 - **Arxiv link:** https://arxiv.org/abs/2507.22157

 - **Pdf link:** https://arxiv.org/pdf/2507.22157

 - **Abstract**
 Voice Activity Detection (VAD) in the presence of background noise remains a challenging problem in speech processing. Accurate VAD is essential in automatic speech recognition, voice-to-text, conversational agents, etc, where noise can severely degrade the performance. A modern application includes the voice assistant, specially mounted on Artificial Intelligence of Things (AIoT) devices such as cell phones, smart glasses, earbuds, etc, where the voice signal includes background noise. Therefore, VAD modules must remain light-weight due to their practical on-device limitation. The existing models often struggle with low signal-to-noise ratios across diverse acoustic environments. A simple VAD often detects human voice in a clean environment, but struggles to detect the human voice in noisy conditions. We propose a noise-robust VAD that comprises a light-weight VAD, with data pre-processing and post-processing added modules to handle the background noise. This approach significantly enhances the VAD accuracy in noisy environments and requires neither a larger model, nor fine-tuning. Experimental results demonstrate that our approach achieves a notable improvement compared to baselines, particularly in environments with high background noise interference. This modified VAD additionally improving clean speech detection.
#### The Risks and Detection of Overestimated Privacy Protection in Voice Anonymisation
 - **Authors:** Michele Panariello, Sarina Meyer, Pierre Champion, Xiaoxiao Miao, Massimiliano Todisco, Ngoc Thang Vu, Nicholas Evans
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2507.22534

 - **Pdf link:** https://arxiv.org/pdf/2507.22534

 - **Abstract**
 Voice anonymisation aims to conceal the voice identity of speakers in speech recordings. Privacy protection is usually estimated from the difficulty of using a speaker verification system to re-identify the speaker post-anonymisation. Performance assessments are therefore dependent on the verification model as well as the anonymisation system. There is hence potential for privacy protection to be overestimated when the verification system is poorly trained, perhaps with mismatched data. In this paper, we demonstrate the insidious risk of overestimating anonymisation performance and show examples of exaggerated performance reported in the literature. For the worst case we identified, performance is overestimated by 74% relative. We then introduce a means to detect when performance assessment might be untrustworthy and show that it can identify all overestimation scenarios presented in the paper. Our solution is openly available as a fork of the 2024 VoicePrivacy Challenge evaluation toolkit.
#### Modeling Multi-Level Hearing Loss for Speech Intelligibility Prediction
 - **Authors:** Xiajie Zhou, Candy Olivia Mawalim, Masashi Unoki
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2507.22599

 - **Pdf link:** https://arxiv.org/pdf/2507.22599

 - **Abstract**
 The diverse perceptual consequences of hearing loss severely impede speech communication, but standard clinical audiometry, which is focused on threshold-based frequency sensitivity, does not adequately capture deficits in frequency and temporal resolution. To address this limitation, we propose a speech intelligibility prediction method that explicitly simulates auditory degradations according to hearing loss severity by broadening cochlear filters and applying low-pass modulation filtering to temporal envelopes. Speech signals are subsequently analyzed using the spectro-temporal modulation (STM) representations, which reflect how auditory resolution loss alters the underlying modulation structure. In addition, normalized cross-correlation (NCC) matrices quantify the similarity between the STM representations of clean speech and speech in noise. These auditory-informed features are utilized to train a Vision Transformer-based regression model that integrates the STM maps and NCC embeddings to estimate speech intelligibility scores. Evaluations on the Clarity Prediction Challenge corpus show that the proposed method outperforms the Hearing-Aid Speech Perception Index v2 (HASPI v2) in both mild and moderate-to-severe hearing loss groups, with a relative root mean squared error reduction of 16.5% for the mild group and a 6.1% reduction for the moderate-to-severe group. These results highlight the importance of explicitly modeling listener-specific frequency and temporal resolution degradations to improve speech intelligibility prediction and provide interpretability in auditory distortions.
#### Quantum-Inspired Audio Unlearning: Towards Privacy-Preserving Voice Biometrics
 - **Authors:** Shreyansh Pathak, Sonu Shreshtha, Richa Singh, Mayank Vatsa
 - **Subjects:** Subjects:
Sound (cs.SD); Artificial Intelligence (cs.AI); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2507.22208

 - **Pdf link:** https://arxiv.org/pdf/2507.22208

 - **Abstract**
 The widespread adoption of voice-enabled authentication and audio biometric systems have significantly increased privacy vulnerabilities associated with sensitive speech data. Compliance with privacy regulations such as GDPR's right to be forgotten and India's DPDP Act necessitates targeted and efficient erasure of individual-specific voice signatures from already-trained biometric models. Existing unlearning methods designed for visual data inadequately handle the sequential, temporal, and high-dimensional nature of audio signals, leading to ineffective or incomplete speaker and accent erasure. To address this, we introduce QPAudioEraser, a quantum-inspired audio unlearning framework. Our our-phase approach involves: (1) weight initialization using destructive interference to nullify target features, (2) superposition-based label transformations that obscure class identity, (3) an uncertainty-maximizing quantum loss function, and (4) entanglement-inspired mixing of correlated weights to retain model knowledge. Comprehensive evaluations with ResNet18, ViT, and CNN architectures across AudioMNIST, Speech Commands, LibriSpeech, and Speech Accent Archive datasets validate QPAudioEraser's superior performance. The framework achieves complete erasure of target data (0% Forget Accuracy) while incurring minimal impact on model utility, with a performance degradation on retained data as low as 0.05%. QPAudioEraser consistently surpasses conventional baselines across single-class, multi-class, sequential, and accent-level erasure scenarios, establishing the proposed approach as a robust privacy-preserving solution.
#### Adaptive Duration Model for Text Speech Alignment
 - **Authors:** Junjie Cao
 - **Subjects:** Subjects:
Sound (cs.SD); Artificial Intelligence (cs.AI); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2507.22612

 - **Pdf link:** https://arxiv.org/pdf/2507.22612

 - **Abstract**
 Speech-to-text alignment is a critical component of neural text to-speech (TTS) models. Autoregressive TTS models typically use an attention mechanism to learn these alignments on-line. However, these alignments tend to be brittle and often fail to generalize to long utterances and out-of-domain text, leading to missing or repeating words. Most non-autoregressive end to-end TTS models rely on durations extracted from external sources, using additional duration models for alignment. In this paper, we propose a novel duration prediction framework that can give compromising phoneme-level duration distribution with given text. In our experiments, the proposed duration model has more precise prediction and condition adaptation ability compared to previous baseline models. Numerically, it has roughly a 11.3 percents immprovement on alignment accuracy, and makes the performance of zero-shot TTS models more robust to the mismatch between prompt audio and input audio.
#### Next Tokens Denoising for Speech Synthesis
 - **Authors:** Yanqing Liu, Ruiqing Xue, Chong Zhang, Yufei Liu, Gang Wang, Bohan Li, Yao Qian, Lei He, Shujie Liu, Sheng Zhao
 - **Subjects:** Subjects:
Sound (cs.SD); Computation and Language (cs.CL); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2507.22746

 - **Pdf link:** https://arxiv.org/pdf/2507.22746

 - **Abstract**
 While diffusion and autoregressive (AR) models have significantly advanced generative modeling, they each present distinct limitations. AR models, which rely on causal attention, cannot exploit future context and suffer from slow generation speeds. Conversely, diffusion models struggle with key-value (KV) caching. To overcome these challenges, we introduce Dragon-FM, a novel text-to-speech (TTS) design that unifies AR and flow-matching. This model processes 48 kHz audio codec tokens in chunks at a compact 12.5 tokens per second rate. This design enables AR modeling across chunks, ensuring global coherence, while parallel flow-matching within chunks facilitates fast iterative denoising. Consequently, the proposed model can utilize KV-cache across chunks and incorporate future context within each chunk. Furthermore, it bridges continuous and discrete feature modeling, demonstrating that continuous AR flow-matching can predict discrete tokens with finite scalar quantizers. This efficient codec and fast chunk-autoregressive architecture also makes the proposed model particularly effective for generating extended content. Experiment for demos of our work} on podcast datasets demonstrate its capability to efficiently generate high-quality zero-shot podcasts.


by Zyzzyva0381 (Windy). 


2025-07-31
