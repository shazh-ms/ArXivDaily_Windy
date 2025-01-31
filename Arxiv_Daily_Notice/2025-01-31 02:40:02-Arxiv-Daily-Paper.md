# Showing new listings for Friday, 31 January 2025
Auto update papers at about 2:30am UTC (10:30am Beijing time) every weekday.


阅读 `Usage.md`了解如何使用此repo实现个性化的Arxiv论文推送

See `Usage.md` for instructions on how to personalize the repo. 


Keyword list: ['acoustic', 'reinforcement learning', 'reverb', 'meta', 'separate', 'reconstruction', 'noise', 'enhance', 'localization', 'speech']


Excluded: []


### Today: 7papers 
#### Language Modelling for Speaker Diarization in Telephonic Interviews
 - **Authors:** Miquel India, Javier Hernando, José A.R. Fonollosa
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Machine Learning (cs.LG); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2501.17893

 - **Pdf link:** https://arxiv.org/pdf/2501.17893

 - **Abstract**
 The aim of this paper is to investigate the benefit of combining both language and acoustic modelling for speaker diarization. Although conventional systems only use acoustic features, in some scenarios linguistic data contain high discriminative speaker information, even more reliable than the acoustic ones. In this study we analyze how an appropriate fusion of both kind of features is able to obtain good results in these cases. The proposed system is based on an iterative algorithm where a LSTM network is used as a speaker classifier. The network is fed with character-level word embeddings and a GMM based acoustic score created with the output labels from previous iterations. The presented algorithm has been evaluated in a Call-Center database, which is composed of telephone interview audios. The combination of acoustic features and linguistic content shows a 84.29% improvement in terms of a word-level DER as compared to a HMM/VB baseline system. The results of this study confirms that linguistic content can be efficiently used for some speaker recognition tasks.
#### Ambisonics Binaural Rendering via Masked Magnitude Least Squares
 - **Authors:** Or Berebi, Fabian Brinkmann, Stefan Weinzierl, Boaz Rafaely
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2501.18224

 - **Pdf link:** https://arxiv.org/pdf/2501.18224

 - **Abstract**
 Ambisonics rendering has become an integral part of 3D audio for headphones. It works well with existing recording hardware, the processing cost is mostly independent of the number of sound sources, and it elegantly allows for rotating the scene and listener. One challenge in Ambisonics headphone rendering is to find a perceptually well behaved low-order representation of the Head-Related Transfer Functions (HRTFs) that are contained in the rendering pipe-line. Low-order rendering is of interest, when working with microphone arrays containing only a few sensors, or for reducing the bandwidth for signal transmission. Magnitude Least Squares rendering became the de facto standard for this, which discards high-frequency interaural phase information in favor of reducing magnitude errors. Building upon this idea, we suggest Masked Magnitude Least Squares, which optimized the Ambisonics coefficients with a neural network and employs a spatio-spectral weighting mask to control the accuracy of the magnitude reconstruction. In the tested case, the weighting mask helped to maintain high-frequency notches in the low-order HRTFs and improved the modeled median plane localization performance in comparison to MagLS, while only marginally affecting the overall accuracy of the magnitude reconstruction.
#### BSM-iMagLS: ILD Informed Binaural Signal Matching for Reproduction with Head-Mounted Microphone Arrays
 - **Authors:** Or Berebi, Zamir Ben-Hur, David Lou Alon, Boaz Rafaely
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2501.18227

 - **Pdf link:** https://arxiv.org/pdf/2501.18227

 - **Abstract**
 Headphone listening in applications such as augmented and virtual reality (AR and VR) relies on high-quality spatial audio to ensure immersion, making accurate binaural reproduction a critical component. As capture devices, wearable arrays with only a few microphones with irregular arrangement face challenges in achieving a reproduction quality comparable to that of arrays with a large number of microphones. Binaural signal matching (BSM) has recently been presented as a signal-independent approach for generating high-quality binaural signal using only a few microphones, which is further improved using magnitude-least squares (MagLS) optimization at high frequencies. This paper extends BSM with MagLS by introducing interaural level difference (ILD) into the MagLS, integrated into BSM (BSM-iMagLS). Using a deep neural network (DNN)-based solver, BSM-iMagLS achieves joint optimization of magnitude, ILD, and magnitude derivatives, improving spatial fidelity. Performance is validated through theoretical analysis, numerical simulations with diverse HRTFs and head-mounted array geometries, and listening experiments, demonstrating a substantial reduction in ILD errors while maintaining comparable magnitude accuracy to state-of-the-art solutions. The results highlight the potential of BSM-iMagLS to enhance binaural reproduction for wearable and portable devices.
#### Multilayered Intelligent Reflecting Surface for Long-Range Underwater Acoustic Communication
 - **Authors:** Yu Luo, Lina Pu, Aijun Song
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD); Signal Processing (eess.SP); Systems and Control (eess.SY)
 - **Arxiv link:** https://arxiv.org/abs/2501.18355

 - **Pdf link:** https://arxiv.org/pdf/2501.18355

 - **Abstract**
 This article introduces a multilayered acoustic reconfigurable intelligent surface (ML-ARIS) architecture designed for the next generation of underwater communications. ML-ARIS incorporates multiple layers of piezoelectric material in each acoustic reflector, with the load impedance of each layer independently adjustable via a control circuit. This design increases the flexibility in generating reflected signals with desired amplitudes and orthogonal phases, enabling passive in-phase and quadrature (IQ) modulation using a single acoustic reflector. Such a feature enables precise beam steering, enhancing sound levels in targeted directions while minimizing interference in surrounding environments. Extensive simulations and tank experiments were conducted to verify the feasibility of ML-ARIS. The experimental results indicate that implementing IQ modulation with a multilayer structure is indeed practical in real-world scenarios, making it possible to use a single reflection unit to generate reflected waves with high-resolution amplitudes and phases.
#### Task and Perception-aware Distributed Source Coding for Correlated Speech under Bandwidth-constrained Channels
 - **Authors:** Sagnik Bhattacharya, Muhammad Ahmed Mohsin, Ahsan Bilal, John M. Cioffi
 - **Subjects:** Subjects:
Information Theory (cs.IT); Artificial Intelligence (cs.AI); Sound (cs.SD); Audio and Speech Processing (eess.AS); Signal Processing (eess.SP)
 - **Arxiv link:** https://arxiv.org/abs/2501.17879

 - **Pdf link:** https://arxiv.org/pdf/2501.17879

 - **Abstract**
 Emerging wireless AR/VR applications require real-time transmission of correlated high-fidelity speech from multiple resource-constrained devices over unreliable, bandwidth-limited channels. Existing autoencoder-based speech source coding methods fail to address the combination of the following - (1) dynamic bitrate adaptation without retraining the model, (2) leveraging correlations among multiple speech sources, and (3) balancing downstream task loss with realism of reconstructed speech. We propose a neural distributed principal component analysis (NDPCA)-aided distributed source coding algorithm for correlated speech sources transmitting to a central receiver. Our method includes a perception-aware downstream task loss function that balances perceptual realism with task-specific performance. Experiments show significant PSNR improvements under bandwidth constraints over naive autoencoder methods in task-agnostic (19%) and task-aware settings (52%). It also approaches the theoretical upper bound, where all correlated sources are sent to a single encoder, especially in low-bandwidth scenarios. Additionally, we present a rate-distortion-perception trade-off curve, enabling adaptive decisions based on application-specific realism needs.
#### Efficient Audiovisual Speech Processing via MUTUD: Multimodal Training and Unimodal Deployment
 - **Authors:** Joanna Hong, Sanjeel Parekh, Honglie Chen, Jacob Donley, Ke Tan, Buye Xu, Anurag Kumar
 - **Subjects:** Subjects:
Sound (cs.SD); Computer Vision and Pattern Recognition (cs.CV); Multimedia (cs.MM); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2501.18157

 - **Pdf link:** https://arxiv.org/pdf/2501.18157

 - **Abstract**
 Building reliable speech systems often requires combining multiple modalities, like audio and visual cues. While such multimodal solutions frequently lead to improvements in performance and may even be critical in certain cases, they come with several constraints such as increased sensory requirements, computational cost, and modality synchronization, to mention a few. These challenges constrain the direct uses of these multimodal solutions in real-world applications. In this work, we develop approaches where the learning happens with all available modalities but the deployment or inference is done with just one or reduced modalities. To do so, we propose a Multimodal Training and Unimodal Deployment (MUTUD) framework which includes a Temporally Aligned Modality feature Estimation (TAME) module that can estimate information from missing modality using modalities present during inference. This innovative approach facilitates the integration of information across different modalities, enhancing the overall inference process by leveraging the strengths of each modality to compensate for the absence of certain modalities during inference. We apply MUTUD to various audiovisual speech tasks and show that it can reduce the performance gap between the multimodal and corresponding unimodal models to a considerable extent. MUTUD can achieve this while reducing the model size and compute compared to multimodal models, in some cases by almost 80%.
#### AGAV-Rater: Adapting Large Multimodal Model for AI-Generated Audio-Visual Quality Assessment
 - **Authors:** Yuqin Cao, Xiongkuo Min, Yixuan Gao, Wei Sun, Guangtao Zhai
 - **Subjects:** Subjects:
Multimedia (cs.MM); Computer Vision and Pattern Recognition (cs.CV); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2501.18314

 - **Pdf link:** https://arxiv.org/pdf/2501.18314

 - **Abstract**
 Many video-to-audio (VTA) methods have been proposed for dubbing silent AI-generated videos. An efficient quality assessment method for AI-generated audio-visual content (AGAV) is crucial for ensuring audio-visual quality. Existing audio-visual quality assessment methods struggle with unique distortions in AGAVs, such as unrealistic and inconsistent elements. To address this, we introduce AGAVQA, the first large-scale AGAV quality assessment dataset, comprising 3,382 AGAVs from 16 VTA methods. AGAVQA includes two subsets: AGAVQA-MOS, which provides multi-dimensional scores for audio quality, content consistency, and overall quality, and AGAVQA-Pair, designed for optimal AGAV pair selection. We further propose AGAV-Rater, a LMM-based model that can score AGAVs, as well as audio and music generated from text, across multiple dimensions, and selects the best AGAV generated by VTA methods to present to the user. AGAV-Rater achieves state-of-the-art performance on AGAVQA, Text-to-Audio, and Text-to-Music datasets. Subjective tests also confirm that AGAV-Rater enhances VTA performance and user experience. The project page is available at this https URL.


by Zyzzyva0381 (Windy). 


2025-01-31
