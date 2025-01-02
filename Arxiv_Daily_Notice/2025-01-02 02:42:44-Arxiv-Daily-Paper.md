# Showing new listings for Tuesday, 31 December 2024
Auto update papers at about 2:30am UTC (10:30am Beijing time) every weekday.


阅读 `Usage.md`了解如何使用此repo实现个性化的Arxiv论文推送

See `Usage.md` for instructions on how to personalize the repo. 


Keyword list: ['acoustic', 'reinforcement learning', 'reverb', 'meta', 'separate', 'reconstruction', 'noise', 'enhance', 'localization', 'speech']


Excluded: []


### Today: 14papers 
#### CrossSpeech++: Cross-lingual Speech Synthesis with Decoupled Language and Speaker Generation
 - **Authors:** Ji-Hoon Kim, Hong-Sun Yang, Yoon-Cheol Ju, Il-Hwan Kim, Byeong-Yeol Kim, Joon Son Chung
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Artificial Intelligence (cs.AI); Sound (cs.SD); Signal Processing (eess.SP)
 - **Arxiv link:** https://arxiv.org/abs/2412.20048

 - **Pdf link:** https://arxiv.org/pdf/2412.20048

 - **Abstract**
 The goal of this work is to generate natural speech in multiple languages while maintaining the same speaker identity, a task known as cross-lingual speech synthesis. A key challenge of cross-lingual speech synthesis is the language-speaker entanglement problem, which causes the quality of cross-lingual systems to lag behind that of intra-lingual systems. In this paper, we propose CrossSpeech++, which effectively disentangles language and speaker information and significantly improves the quality of cross-lingual speech synthesis. To this end, we break the complex speech generation pipeline into two simple components: language-dependent and speaker-dependent generators. The language-dependent generator produces linguistic variations that are not biased by specific speaker attributes. The speaker-dependent generator models acoustic variations that characterize speaker identity. By handling each type of information in separate modules, our method can effectively disentangle language and speaker representation. We conduct extensive experiments using various metrics, and demonstrate that CrossSpeech++ achieves significant improvements in cross-lingual speech synthesis, outperforming existing methods by a large margin.
#### Distance Based Single-Channel Target Speech Extraction
 - **Authors:** Runwu Shi, Benjamin Yen, Kazuhiro Nakadai
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2412.20144

 - **Pdf link:** https://arxiv.org/pdf/2412.20144

 - **Abstract**
 This paper aims to achieve single-channel target speech extraction (TSE) in enclosures by solely utilizing distance information. This is the first work that utilizes only distance cues without using speaker physiological information for single-channel TSE. Inspired by recent single-channel Distance-based separation and extraction methods, we introduce a novel model that efficiently fuses distance information with time-frequency (TF) bins for TSE. Experimental results in both single-room and multi-room scenarios demonstrate the feasibility and effectiveness of our approach. This method can also be employed to estimate the distances of different speakers in mixed speech. Online demos are available at this https URL.
#### Bird Vocalization Embedding Extraction Using Self-Supervised Disentangled Representation Learning
 - **Authors:** Runwu Shi, Katsutoshi Itoyama, Kazuhiro Nakadai
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD); Signal Processing (eess.SP)
 - **Arxiv link:** https://arxiv.org/abs/2412.20146

 - **Pdf link:** https://arxiv.org/pdf/2412.20146

 - **Abstract**
 This paper addresses the extraction of the bird vocalization embedding from the whole song level using disentangled representation learning (DRL). Bird vocalization embeddings are necessary for large-scale bioacoustic tasks, and self-supervised methods such as Variational Autoencoder (VAE) have shown their performance in extracting such low-dimensional embeddings from vocalization segments on the note or syllable level. To extend the processing level to the entire song instead of cutting into segments, this paper regards each vocalization as the generalized and discriminative part and uses two encoders to learn these two parts. The proposed method is evaluated on the Great Tits dataset according to the clustering performance, and the results outperform the compared pre-trained models and vanilla VAE. Finally, this paper analyzes the informative part of the embedding, further compresses its dimension, and explains the disentangled performance of bird vocalizations.
#### EmoReg: Directional Latent Vector Modeling for Emotional Intensity Regularization in Diffusion-based Voice Conversion
 - **Authors:** Ashishkumar Gudmalwar, Ishan D. Biyani, Nirmesh Shah, Pankaj Wasnik, Rajiv Ratn Shah
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Artificial Intelligence (cs.AI); Multimedia (cs.MM); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2412.20359

 - **Pdf link:** https://arxiv.org/pdf/2412.20359

 - **Abstract**
 The Emotional Voice Conversion (EVC) aims to convert the discrete emotional state from the source emotion to the target for a given speech utterance while preserving linguistic content. In this paper, we propose regularizing emotion intensity in the diffusion-based EVC framework to generate precise speech of the target emotion. Traditional approaches control the intensity of an emotional state in the utterance via emotion class probabilities or intensity labels that often lead to inept style manipulations and degradations in quality. On the contrary, we aim to regulate emotion intensity using self-supervised learning-based feature representations and unsupervised directional latent vector modeling (DVM) in the emotional embedding space within a diffusion-based framework. These emotion embeddings can be modified based on the given target emotion intensity and the corresponding direction vector. Furthermore, the updated embeddings can be fused in the reverse diffusion process to generate the speech with the desired emotion and intensity. In summary, this paper aims to achieve high-quality emotional intensity regularization in the diffusion-based EVC framework, which is the first of its kind work. The effectiveness of the proposed method has been shown across state-of-the-art (SOTA) baselines in terms of subjective and objective evaluations for the English and Hindi languages \footnote{Demo samples are available at the following URL: \url{this https URL}}.
#### Metadata-Enhanced Speech Emotion Recognition: Augmented Residual Integration and Co-Attention in Two-Stage Fine-Tuning
 - **Authors:** Zixiang Wan, Ziyue Qiu, Yiyang Liu, Wei-Qiang Zhang
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2412.20707

 - **Pdf link:** https://arxiv.org/pdf/2412.20707

 - **Abstract**
 Speech Emotion Recognition (SER) involves analyzing vocal expressions to determine the emotional state of speakers, where the comprehensive and thorough utilization of audio information is paramount. Therefore, we propose a novel approach on self-supervised learning (SSL) models that employs all available auxiliary information -- specifically metadata -- to enhance performance. Through a two-stage fine-tuning method in multi-task learning, we introduce the Augmented Residual Integration (ARI) module, which enhances transformer layers in encoder of SSL models. The module efficiently preserves acoustic features across all different levels, thereby significantly improving the performance of metadata-related auxiliary tasks that require various levels of features. Moreover, the Co-attention module is incorporated due to its complementary nature with ARI, enabling the model to effectively utilize multidimensional information and contextual relationships from metadata-related auxiliary tasks. Under pre-trained base models and speaker-independent setup, our approach consistently surpasses state-of-the-art (SOTA) models on multiple SSL encoders for the IEMOCAP dataset.
#### Improving Acoustic Scene Classification in Low-Resource Conditions
 - **Authors:** Zhi Chen, Yun-Fei Shao, Yong Ma, Mingsheng Wei, Le Zhang, Wei-Qiang Zhang
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2412.20722

 - **Pdf link:** https://arxiv.org/pdf/2412.20722

 - **Abstract**
 Acoustic Scene Classification (ASC) identifies an environment based on an audio signal. This paper explores ASC in low-resource conditions and proposes a novel model, DS-FlexiNet, which combines depthwise separable convolutions from MobileNetV2 with ResNet-inspired residual connections for a balance of efficiency and accuracy. To address hardware limitations and device heterogeneity, DS-FlexiNet employs Quantization Aware Training (QAT) for model compression and data augmentation methods like Auto Device Impulse Response (ADIR) and Freq-MixStyle (FMS) to improve cross-device generalization. Knowledge Distillation (KD) from twelve teacher models further enhances performance on unseen devices. The architecture includes a custom Residual Normalization layer to handle domain differences across devices, and depthwise separable convolutions reduce computational overhead without sacrificing feature representation. Experimental results show that DS-FlexiNet excels in both adaptability and performance under resource-constrained conditions.
#### Phoneme-Level Contrastive Learning for User-Defined Keyword Spotting with Flexible Enrollment
 - **Authors:** Li Kewei, Zhou Hengshun, Shen Kai, Dai Yusheng, Du Jun
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2412.20805

 - **Pdf link:** https://arxiv.org/pdf/2412.20805

 - **Abstract**
 User-defined keyword spotting (KWS) enhances the user experience by allowing individuals to customize keywords. However, in open-vocabulary scenarios, most existing methods commonly suffer from high false alarm rates with confusable words and are limited to either audio-only or text-only enrollment. Therefore, in this paper, we first explore the model's robustness against confusable words. Specifically, we propose Phoneme-Level Contrastive Learning (PLCL), which refines and aligns query and source feature representations at the phoneme level. This method enhances the model's disambiguation capability through fine-grained positive and negative comparisons for more accurate alignment, and it is generalizable to jointly optimize both audio-text and audio-audio matching, adapting to various enrollment modes. Furthermore, we maintain a context-agnostic phoneme memory bank to construct confusable negatives for data augmentation. Based on this, a third-category discriminator is specifically designed to distinguish hard negatives. Overall, we develop a robust and flexible KWS system, supporting different modality enrollment methods within a unified framework. Verified on the LibriPhrase dataset, the proposed approach achieves state-of-the-art performance.
#### Enhancing Multimodal Emotion Recognition through Multi-Granularity Cross-Modal Alignment
 - **Authors:** Xuechen Wang, Shiwan Zhao, Haoqin Sun, Hui Wang, Jiaming Zhou, Yong Qin
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Computation and Language (cs.CL); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2412.20821

 - **Pdf link:** https://arxiv.org/pdf/2412.20821

 - **Abstract**
 Multimodal emotion recognition (MER), leveraging speech and text, has emerged as a pivotal domain within human-computer interaction, demanding sophisticated methods for effective multimodal integration. The challenge of aligning features across these modalities is significant, with most existing approaches adopting a singular alignment strategy. Such a narrow focus not only limits model performance but also fails to address the complexity and ambiguity inherent in emotional expressions. In response, this paper introduces a Multi-Granularity Cross-Modal Alignment (MGCMA) framework, distinguished by its comprehensive approach encompassing distribution-based, instance-based, and token-based alignment modules. This framework enables a multi-level perception of emotional information across modalities. Our experiments on IEMOCAP demonstrate that our proposed method outperforms current state-of-the-art techniques.
#### Mouth Articulation-Based Anchoring for Improved Cross-Corpus Speech Emotion Recognition
 - **Authors:** Shreya G. Upadhyay, Ali N. Salman, Carlos Busso, Chi-Chun Lee
 - **Subjects:** Subjects:
Sound (cs.SD); Machine Learning (cs.LG); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2412.19909

 - **Pdf link:** https://arxiv.org/pdf/2412.19909

 - **Abstract**
 Cross-corpus speech emotion recognition (SER) plays a vital role in numerous practical applications. Traditional approaches to cross-corpus emotion transfer often concentrate on adapting acoustic features to align with different corpora, domains, or labels. However, acoustic features are inherently variable and error-prone due to factors like speaker differences, domain shifts, and recording conditions. To address these challenges, this study adopts a novel contrastive approach by focusing on emotion-specific articulatory gestures as the core elements for analysis. By shifting the emphasis on the more stable and consistent articulatory gestures, we aim to enhance emotion transfer learning in SER tasks. Our research leverages the CREMA-D and MSP-IMPROV corpora as benchmarks and it reveals valuable insights into the commonality and reliability of these articulatory gestures. The findings highlight mouth articulatory gesture potential as a better constraint for improving emotion recognition across different settings or domains.
#### ASE: Practical Acoustic Speed Estimation Beyond Doppler via Sound Diffusion Field
 - **Authors:** Sheng Lyu, Chenshu Wu
 - **Subjects:** Subjects:
Human-Computer Interaction (cs.HC); Networking and Internet Architecture (cs.NI); Sound (cs.SD); Audio and Speech Processing (eess.AS); Signal Processing (eess.SP)
 - **Arxiv link:** https://arxiv.org/abs/2412.20142

 - **Pdf link:** https://arxiv.org/pdf/2412.20142

 - **Abstract**
 Passive human speed estimation plays a critical role in acoustic sensing. Despite extensive study, existing systems, however, suffer from various limitations: First, previous acoustic speed estimation exploits Doppler Frequency Shifts (DFS) created by moving targets and relies on microphone arrays, making them only capable of sensing the radial speed within a constrained distance. Second, the channel measurement rate proves inadequate to estimate high moving speeds. To overcome these issues, we present ASE, an accurate and robust Acoustic Speed Estimation system on a single commodity microphone. We model the sound propagation from a unique perspective of the acoustic diffusion field, and infer the speed from the acoustic spatial distribution, a completely different way of thinking about speed estimation beyond prior DFS-based approaches. We then propose a novel Orthogonal Time-Delayed Multiplexing (OTDM) scheme for acoustic channel estimation at a high rate that was previously infeasible, making it possible to estimate high speeds. We further develop novel techniques for motion detection and signal enhancement to deliver a robust and practical system. We implement and evaluate ASE through extensive real-world experiments. Our results show that ASE reliably tracks walking speed, independently of target location and direction, with a mean error of 0.13 m/s, a reduction of 2.5x from DFS, and a detection rate of 97.4% for large coverage, e.g., free walking in a 4m $\times$ 4m room. We believe ASE pushes acoustic speed estimation beyond the conventional DFS-based paradigm and will inspire exciting research in acoustic sensing.
#### Stable-TTS: Stable Speaker-Adaptive Text-to-Speech Synthesis via Prosody Prompting
 - **Authors:** Wooseok Han, Minki Kang, Changhun Kim, Eunho Yang
 - **Subjects:** Subjects:
Sound (cs.SD); Artificial Intelligence (cs.AI); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2412.20155

 - **Pdf link:** https://arxiv.org/pdf/2412.20155

 - **Abstract**
 Speaker-adaptive Text-to-Speech (TTS) synthesis has attracted considerable attention due to its broad range of applications, such as personalized voice assistant services. While several approaches have been proposed, they often exhibit high sensitivity to either the quantity or the quality of target speech samples. To address these limitations, we introduce Stable-TTS, a novel speaker-adaptive TTS framework that leverages a small subset of a high-quality pre-training dataset, referred to as prior samples. Specifically, Stable-TTS achieves prosody consistency by leveraging the high-quality prosody of prior samples, while effectively capturing the timbre of the target speaker. Additionally, it employs a prior-preservation loss during fine-tuning to maintain the synthesis ability for prior samples to prevent overfitting on target samples. Extensive experiments demonstrate the effectiveness of Stable-TTS even under limited amounts of and noisy target speech samples.
#### Language-based Audio Retrieval with Co-Attention Networks
 - **Authors:** Haoran Sun, Zimu Wang, Qiuyi Chen, Jianjun Chen, Jia Wang, Haiyang Zhang
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2412.20914

 - **Pdf link:** https://arxiv.org/pdf/2412.20914

 - **Abstract**
 In recent years, user-generated audio content has proliferated across various media platforms, creating a growing need for efficient retrieval methods that allow users to search for audio clips using natural language queries. This task, known as language-based audio retrieval, presents significant challenges due to the complexity of learning semantic representations from heterogeneous data across both text and audio modalities. In this work, we introduce a novel framework for the language-based audio retrieval task that leverages co-attention mechanismto jointly learn meaningful representations from both modalities. To enhance the model's ability to capture fine-grained cross-modal interactions, we propose a cascaded co-attention architecture, where co-attention modules are stacked or iterated to progressively refine the semantic alignment between text and audio. Experiments conducted on two public datasets show that the proposed method can achieve better performance than the state-of-the-art method. Specifically, our best performed co-attention model achieves a 16.6% improvement in mean Average Precision on Clotho dataset, and a 15.1% improvement on AudioCaps.
#### TangoFlux: Super Fast and Faithful Text to Audio Generation with Flow Matching and Clap-Ranked Preference Optimization
 - **Authors:** Chia-Yu Hung, Navonil Majumder, Zhifeng Kong, Ambuj Mehrish, Rafael Valle, Bryan Catanzaro, Soujanya Poria
 - **Subjects:** Subjects:
Sound (cs.SD); Artificial Intelligence (cs.AI); Computation and Language (cs.CL); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2412.21037

 - **Pdf link:** https://arxiv.org/pdf/2412.21037

 - **Abstract**
 We introduce TangoFlux, an efficient Text-to-Audio (TTA) generative model with 515M parameters, capable of generating up to 30 seconds of 44.1kHz audio in just 3.7 seconds on a single A40 GPU. A key challenge in aligning TTA models lies in the difficulty of creating preference pairs, as TTA lacks structured mechanisms like verifiable rewards or gold-standard answers available for Large Language Models (LLMs). To address this, we propose CLAP-Ranked Preference Optimization (CRPO), a novel framework that iteratively generates and optimizes preference data to enhance TTA alignment. We demonstrate that the audio preference dataset generated using CRPO outperforms existing alternatives. With this framework, TangoFlux achieves state-of-the-art performance across both objective and subjective benchmarks. We open source all code and models to support further research in TTA generation.
#### Two-component spatiotemporal template for activation-inhibition of speech in ECoG
 - **Authors:** Eric Easthope
 - **Subjects:** Subjects:
Neurons and Cognition (q-bio.NC); Computation and Language (cs.CL); Machine Learning (cs.LG); Audio and Speech Processing (eess.AS); Signal Processing (eess.SP)
 - **Arxiv link:** https://arxiv.org/abs/2412.21178

 - **Pdf link:** https://arxiv.org/pdf/2412.21178

 - **Abstract**
 I compute the average trial-by-trial power of band-limited speech activity across epochs of multi-channel high-density electrocorticography (ECoG) recorded from multiple subjects during a consonant-vowel speaking task. I show that previously seen anti-correlations of average beta frequency activity (12-35 Hz) to high-frequency gamma activity (70-140 Hz) during speech movement are observable between individual ECoG channels in the sensorimotor cortex (SMC). With this I fit a variance-based model using principal component analysis to the band-powers of individual channels of session-averaged ECoG data in the SMC and project SMC channels onto their lower-dimensional principal components. Spatiotemporal relationships between speech-related activity and principal components are identified by correlating the principal components of both frequency bands to individual ECoG channels over time using windowed correlation. Correlations of principal component areas to sensorimotor areas reveal a distinct two-component activation-inhibition-like representation for speech that resembles distinct local sensorimotor areas recently shown to have complex interplay in whole-body motor control, inhibition, and posture. Notably the third principal component shows insignificant correlations across all subjects, suggesting two components of ECoG are sufficient to represent SMC activity during speech movement.


by Zyzzyva0381 (Windy). 


2025-01-02
