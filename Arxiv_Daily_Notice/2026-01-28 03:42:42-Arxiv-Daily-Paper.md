# Showing new listings for Wednesday, 28 January 2026
Auto update papers at about 2:30am UTC (10:30am Beijing time) every weekday.


阅读 `Usage.md`了解如何使用此repo实现个性化的Arxiv论文推送

See `Usage.md` for instructions on how to personalize the repo. 


Keyword list: ['text-to-speech', 'text to speech', 'tts', 'LLM-based', 'speech', 'voice']


Excluded: []


### Today: 7papers 
#### Beyond Lips: Integrating Gesture and Lip Cues for Robust Audio-visual Speaker Extraction
 - **Authors:** Zexu Pan, Xinyuan Qian, Shengkui Zhao, Kun Zhou, Bin Ma
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2601.19130

 - **Pdf link:** https://arxiv.org/pdf/2601.19130

 - **Abstract**
 Most audio-visual speaker extraction methods rely on synchronized lip recording to isolate the speech of a target speaker from a multi-talker mixture. However, in natural human communication, co-speech gestures are also temporally aligned with speech, often emphasizing specific words or syllables. These gestures provide complementary visual cues that can be especially valuable when facial or lip regions are occluded or distant. In this work, we move beyond lip-centric approaches and propose SeLG, a model that integrates both lip and upper-body gesture information for robust speaker extraction. SeLG features a cross-attention-based fusion mechanism that enables each visual modality to query and selectively attend to relevant speech features in the mixture. To improve the alignment of gesture representations with speech dynamics, SeLG also employs a contrastive InfoNCE loss that encourages gesture embeddings to align more closely with corresponding lip embeddings, which are more strongly correlated with speech. Experimental results on the YGD dataset, containing TED talks, demonstrate that the proposed contrastive learning strategy significantly improves gesture-based speaker extraction, and that our proposed SeLG model, by effectively fusing lip and gesture cues with an attention mechanism and InfoNCE loss, achieves superior performance compared to baselines, across both complete and partial (i.e., missing-modality) conditions.
#### SE-DiCoW: Self-Enrolled Diarization-Conditioned Whisper
 - **Authors:** Alexander Polok, Dominik Klement, Samuele Cornell, Matthew Wiesner, Jan Černocký, Sanjeev Khudanpur, Lukáš Burget
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Machine Learning (cs.LG)
 - **Arxiv link:** https://arxiv.org/abs/2601.19194

 - **Pdf link:** https://arxiv.org/pdf/2601.19194

 - **Abstract**
 Speaker-attributed automatic speech recognition (ASR) in multi-speaker environments remains a major challenge. While some approaches achieve strong performance when fine-tuned on specific domains, few systems generalize well across out-of-domain datasets. Our prior work, Diarization-Conditioned Whisper (DiCoW), leverages speaker diarization outputs as conditioning information and, with minimal fine-tuning, demonstrated strong multilingual and multi-domain performance. In this paper, we address a key limitation of DiCoW: ambiguity in Silence-Target-Non-target-Overlap (STNO) masks, where two or more fully overlapping speakers may have nearly identical conditioning despite differing transcriptions. We introduce SE-DiCoW (Self-Enrolled Diarization-Conditioned Whisper), which uses diarization output to locate an enrollment segment anywhere in the conversation where the target speaker is most active. This enrollment segment is used as fixed conditioning via cross-attention at each encoder layer. We further refine DiCoW with improved data segmentation, model initialization, and augmentation. Together, these advances yield substantial gains: SE-DiCoW reduces macro-averaged tcpWER by 52.4% relative to the original DiCoW on the EMMA MT-ASR benchmark.
#### Audio Deepfake Detection at the First Greeting: "Hi!"
 - **Authors:** Haohan Shi, Xiyu Shi, Safak Dogan, Tianjin Huang, Yunxiao Zhang
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2601.19573

 - **Pdf link:** https://arxiv.org/pdf/2601.19573

 - **Abstract**
 This paper focuses on audio deepfake detection under real-world communication degradations, with an emphasis on ultra-short inputs (0.5-2.0s), targeting the capability to detect synthetic speech at a conversation opening, e.g., when a scammer says "Hi." We propose Short-MGAA (S-MGAA), a novel lightweight extension of Multi-Granularity Adaptive Time-Frequency Attention, designed to enhance discriminative representation learning for short, degraded inputs subjected to communication processing and perturbations. The S-MGAA integrates two tailored modules: a Pixel-Channel Enhanced Module (PCEM) that amplifies fine-grained time-frequency saliency, and a Frequency Compensation Enhanced Module (FCEM) to supplement limited temporal evidence via multi-scale frequency modeling and adaptive frequency-temporal interaction. Extensive experiments demonstrate that S-MGAA consistently surpasses nine state-of-the-art baselines while achieving strong robustness to degradations and favorable efficiency-accuracy trade-offs, including low RTF, competitive GFLOPs, compact parameters, and reduced training cost, highlighting its strong potential for real-time deployment in communication systems and edge devices.
#### SAM Audio Judge: A Unified Multimodal Framework for Perceptual Evaluation of Audio Separation
 - **Authors:** Helin Wang, Bowen Shi, Andros Tjandra, John Hoffman, Yi-Chiao Wu, Apoorv Vyas, Najim Dehak, Ann Lee, Wei-Ning Hsu
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Artificial Intelligence (cs.AI)
 - **Arxiv link:** https://arxiv.org/abs/2601.19702

 - **Pdf link:** https://arxiv.org/pdf/2601.19702

 - **Abstract**
 The performance evaluation remains a complex challenge in audio separation, and existing evaluation metrics are often misaligned with human perception, course-grained, relying on ground truth signals. On the other hand, subjective listening tests remain the gold standard for real-world evaluation, but they are expensive, time-consuming, and difficult to scale. This paper addresses the growing need for automated systems capable of evaluating audio separation without human intervention. The proposed evaluation metric, SAM Audio Judge (SAJ), is a multimodal fine-grained reference-free objective metric, which shows highly alignment with human perceptions. SAJ supports three audio domains (speech, music and general sound events) and three prompt inputs (text, visual and span), covering four different dimensions of evaluation (recall, percision, faithfulness, and overall). SAM Audio Judge also shows potential applications in data filtering, pseudo-labeling large datasets and reranking in audio separation models. We release our code and pre-trained models at: this https URL.
#### Rethinking Discrete Speech Representation Tokens for Accent Generation
 - **Authors:** Jinzuomu Zhong, Yi Wang, Korin Richmond, Peter Bell
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Computation and Language (cs.CL); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2601.19786

 - **Pdf link:** https://arxiv.org/pdf/2601.19786

 - **Abstract**
 Discrete Speech Representation Tokens (DSRTs) have become a foundational component in speech generation. While prior work has extensively studied phonetic and speaker information in DSRTs, how accent information is encoded in DSRTs remains largely unexplored. In this paper, we present the first systematic investigation of accent information in DSRTs. We propose a unified evaluation framework that measures both accessibility of accent information via a novel Accent ABX task and recoverability via cross-accent Voice Conversion (VC) resynthesis. Using this framework, we analyse DSRTs derived from a variety of speech encoders. Our results reveal that accent information is substantially reduced when ASR supervision is used to fine-tune the encoder, but cannot be effectively disentangled from phonetic and speaker information through naive codebook size reduction. Based on these findings, we propose new content-only and content-accent DSRTs that significantly outperform existing designs in controllable accent generation. Our work highlights the importance of accent-aware evaluation and provides practical guidance for designing DSRTs for accent-controlled speech generation.
#### Enhancing Speech Emotion Recognition using Dynamic Spectral Features and Kalman Smoothing
 - **Authors:** Marouane El Hizabri, Abdelfattah Bezzaz, Ismail Hayoukane, Youssef Taki
 - **Subjects:** Subjects:
Sound (cs.SD); Machine Learning (cs.LG); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2601.18908

 - **Pdf link:** https://arxiv.org/pdf/2601.18908

 - **Abstract**
 Speech Emotion Recognition systems often use static features like Mel-Frequency Cepstral Coefficients (MFCCs), Zero Crossing Rate (ZCR), and Root Mean Square Energy (RMSE). Because of this, they can misclassify emotions when there is acoustic noise in vocal signals. To address this, we added dynamic features using Dynamic Spectral features (Deltas and Delta-Deltas) along with the Kalman Smoothing algorithm. This approach reduces noise and improves emotion classification. Since emotion changes over time, the Kalman Smoothing filter also helped make the classifier outputs more stable. Tests on the RAVDESS dataset showed that this method achieved a state-of-the-art accuracy of 87\% and reduced misclassification between emotions with similar acoustic features
#### A Hybrid Discriminative and Generative System for Universal Speech Enhancement
 - **Authors:** Yinghao Liu, Chengwei Liu, Xiaotao Liang, Haoyin Yan, Shaofei Xue, Zheng Xue
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2601.19113

 - **Pdf link:** https://arxiv.org/pdf/2601.19113

 - **Abstract**
 Universal speech enhancement aims at handling inputs with various speech distortions and recording conditions. In this work, we propose a novel hybrid architecture that synergizes the signal fidelity of discriminative modeling with the reconstruction capabilities of generative modeling. Our system utilizes the discriminative TF-GridNet model with the Sampling-Frequency-Independent strategy to handle variable sampling rates universally. In parallel, an autoregressive model combined with spectral mapping modeling generates detail-rich speech while effectively suppressing generative artifacts. Finally, a fusion network learns adaptive weights of the two outputs under the optimization of signal-level losses and the comprehensive Speech Quality Assessment (SQA) loss. Our proposed system is evaluated in the ICASSP 2026 URGENT Challenge (Track 1) and ranks the third place.


by Zyzzyva0381 (Windy). 


2026-01-28
