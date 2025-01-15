# Showing new listings for Wednesday, 15 January 2025
Auto update papers at about 2:30am UTC (10:30am Beijing time) every weekday.


阅读 `Usage.md`了解如何使用此repo实现个性化的Arxiv论文推送

See `Usage.md` for instructions on how to personalize the repo. 


Keyword list: ['acoustic', 'reinforcement learning', 'reverb', 'meta', 'separate', 'reconstruction', 'noise', 'enhance', 'localization', 'speech']


Excluded: []


### Today: 6papers 
#### Gen-A: Generalizing Ambisonics Neural Encoding to Unseen Microphone Arrays
 - **Authors:** Mikko Heikkinen, Archontis Politis, Konstantinos Drossos, Tuomas Virtanen
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Machine Learning (cs.LG); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2501.08047

 - **Pdf link:** https://arxiv.org/pdf/2501.08047

 - **Abstract**
 Using deep neural networks (DNNs) for encoding of microphone array (MA) signals to the Ambisonics spatial audio format can surpass certain limitations of established conventional methods, but existing DNN-based methods need to be trained separately for each MA. This paper proposes a DNN-based method for Ambisonics encoding that can generalize to arbitrary MA geometries unseen during training. The method takes as inputs the MA geometry and MA signals and uses a multi-level encoder consisting of separate paths for geometry and signal data, where geometry features inform the signal encoder at each level. The method is validated in simulated anechoic and reverberant conditions with one and two sources. The results indicate improvement over conventional encoding across the whole frequency range for dry scenes, while for reverberant scenes the improvement is frequency-dependent.
#### Optimizing Speech Multi-View Feature Fusion through Conditional Computation
 - **Authors:** Weiqiao Shan, Yuhao Zhang, Yuchen Han, Bei Li, Xiaofeng Zhao, Yuang Li, Min Zhang, Hao Yang, Tong Xiao, Jingbo Zhu
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Artificial Intelligence (cs.AI); Computation and Language (cs.CL); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2501.08057

 - **Pdf link:** https://arxiv.org/pdf/2501.08057

 - **Abstract**
 Recent advancements have highlighted the efficacy of self-supervised learning (SSL) features in various speech-related tasks, providing lightweight and versatile multi-view speech representations. However, our study reveals that while SSL features expedite model convergence, they conflict with traditional spectral features like FBanks in terms of update directions. In response, we propose a novel generalized feature fusion framework grounded in conditional computation, featuring a gradient-sensitive gating network and a multi-stage dropout strategy. This framework mitigates feature conflicts and bolsters model robustness to multi-view input features. By integrating SSL and spectral features, our approach accelerates convergence and maintains performance on par with spectral models across multiple speech translation tasks on the MUSTC dataset.
#### Loudspeaker Beamforming to Enhance Speech Recognition Performance of Voice Driven Applications
 - **Authors:** Dimme de Groot, Baturalp Karslioglu, Odette Scharenborg, Jorge Martinez
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2501.08104

 - **Pdf link:** https://arxiv.org/pdf/2501.08104

 - **Abstract**
 In this paper we propose a robust loudspeaker beamforming algorithm which is used to enhance the performance of voice driven applications in scenarios where the loudspeakers introduce the majority of the noise, e.g. when music is playing loudly. The loudspeaker beamformer modifies the loudspeaker playback signals to create a low-acoustic-energy region around the device that implements automatic speech recognition for a voice driven application (VDA). The algorithm utilises a distortion measure based on human auditory perception to limit the distortion perceived by human listeners. Simulations and real-world experiments show that the proposed loudspeaker beamformer improves the speech recognition performance in all tested scenarios. Moreover, the algorithm allows to further reduce the acoustic energy around the VDA device at the expense of reduced objective audio quality at the listener's location.
#### Neural Speech Tracking in a Virtual Acoustic Environment: Audio-Visual Benefit for Unscripted Continuous Speech
 - **Authors:** Mareike Daeglau, Juergen Otten, Giso Grimm, Bojana Mirkovic, Volker Hohmann, Stefan Debener
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2501.08124

 - **Pdf link:** https://arxiv.org/pdf/2501.08124

 - **Abstract**
 The audio visual benefit in speech perception, where congruent visual input enhances auditory processing, is well documented across age groups, particularly in challenging listening conditions and among individuals with varying hearing abilities. However, most studies rely on highly controlled laboratory environments with scripted stimuli. Here, we examine the audio visual benefit using unscripted, natural speech from untrained speakers within a virtual acoustic environment. Using electroencephalography (EEG) and cortical speech tracking, we assessed neural responses across audio visual, audio only, visual only, and masked lip conditions to isolate the role of lip movements. Additionally, we analysed individual differences in acoustic and visual features of the speakers, including pitch, jitter, and lip openness, to explore their influence on the audio visual speech tracking benefit. Results showed a significant audio visual enhancement in speech tracking with background noise, with the masked lip condition performing similarly to the audio-only condition, emphasizing the importance of lip movements in adverse listening situations. Our findings reveal the feasibility of cortical speech tracking with naturalistic stimuli and underscore the impact of individual speaker characteristics on audio-visual integration in real world listening contexts.
#### Bridge-SR: Schr\"odinger Bridge for Efficient SR
 - **Authors:** Chang Li, Zehua Chen, Fan Bao, Jun Zhu
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2501.07897

 - **Pdf link:** https://arxiv.org/pdf/2501.07897

 - **Abstract**
 Speech super-resolution (SR), which generates a waveform at a higher sampling rate from its low-resolution version, is a long-standing critical task in speech restoration. Previous works have explored speech SR in different data spaces, but these methods either require additional compression networks or exhibit limited synthesis quality and inference speed. Motivated by recent advances in probabilistic generative models, we present Bridge-SR, a novel and efficient any-to-48kHz SR system in the speech waveform domain. Using tractable SchrÃ¶dinger Bridge models, we leverage the observed low-resolution waveform as a prior, which is intrinsically informative for the high-resolution target. By optimizing a lightweight network to learn the score functions from the prior to the target, we achieve efficient waveform SR through a data-to-data generation process that fully exploits the instructive content contained in the low-resolution observation. Furthermore, we identify the importance of the noise schedule, data scaling, and auxiliary loss functions, which further improve the SR quality of bridge-based systems. The experiments conducted on the benchmark dataset VCTK demonstrate the efficiency of our system: (1) in terms of sample quality, Bridge-SR outperforms several strong baseline methods under different SR settings, using a lightweight network backbone (1.7M); (2) in terms of inference speed, our 4-step synthesis achieves better performance than the 8-step conditional diffusion counterpart (LSD: 0.911 vs 0.927). Demo at this https URL.
#### CodecFake-Omni: A Large-Scale Codec-based Deepfake Speech Dataset
 - **Authors:** Jiawei Du, Xuanjun Chen, Haibin Wu, Lin Zhang, I-Ming Lin, I-Hsiang Chiu, Wenze Ren, Yuan Tseng, Yu Tsao, Jyh-Shing Roger Jang, Hung-yi Lee
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2501.08238

 - **Pdf link:** https://arxiv.org/pdf/2501.08238

 - **Abstract**
 With the rapid advancement of codec-based speech generation (CoSG) systems, creating fake speech that mimics an individual's identity and spreads misinformation has become remarkably easy. Addressing the risks posed by such deepfake speech has attracted significant attention. However, most existing studies focus on detecting fake data generated by traditional speech generation models. Research on detecting fake speech generated by CoSG systems remains limited and largely unexplored. In this paper, we introduce CodecFake-Omni, a large-scale dataset specifically designed to advance the study of neural codec-based deepfake speech (CodecFake) detection and promote progress within the anti-spoofing community. To the best of our knowledge, CodecFake-Omni is the largest dataset of its kind till writing this paper, encompassing the most diverse range of codec architectures. The training set is generated through re-synthesis using nearly all publicly available open-source 31 neural audio codec models across 21 different codec families (one codec family with different configurations will result in multiple different codec models). The evaluation set includes web-sourced data collected from websites generated by 17 advanced CoSG models with eight codec families. Using this large-scale dataset, we reaffirm our previous findings that anti-spoofing models trained on traditional spoofing datasets generated by vocoders struggle to detect synthesized speech from current CoSG systems. Additionally, we propose a comprehensive neural audio codec taxonomy, categorizing neural audio codecs by their root components: vector quantizer, auxiliary objectives, and decoder types, with detailed explanations and representative examples for each. Using this comprehensive taxonomy, we conduct stratified analysis to provide valuable insights for future CodecFake detection research.


by Zyzzyva0381 (Windy). 


2025-01-15
