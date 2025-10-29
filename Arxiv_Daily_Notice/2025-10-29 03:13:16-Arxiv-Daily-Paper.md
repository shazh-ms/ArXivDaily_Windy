# Showing new listings for Wednesday, 29 October 2025
Auto update papers at about 2:30am UTC (10:30am Beijing time) every weekday.


阅读 `Usage.md`了解如何使用此repo实现个性化的Arxiv论文推送

See `Usage.md` for instructions on how to personalize the repo. 


Keyword list: ['text-to-speech', 'text to speech', 'tts', 'LLM-based', 'speech', 'voice']


Excluded: []


### Today: 8papers 
#### A Neural Model for Contextual Biasing Score Learning and Filtering
 - **Authors:** Wanting Huang, Weiran Wang
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Artificial Intelligence (cs.AI); Computation and Language (cs.CL); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2510.23849

 - **Pdf link:** https://arxiv.org/pdf/2510.23849

 - **Abstract**
 Contextual biasing improves automatic speech recognition (ASR) by integrating external knowledge, such as user-specific phrases or entities, during decoding. In this work, we use an attention-based biasing decoder to produce scores for candidate phrases based on acoustic information extracted by an ASR encoder, which can be used to filter out unlikely phrases and to calculate bonus for shallow-fusion biasing. We introduce a per-token discriminative objective that encourages higher scores for ground-truth phrases while suppressing distractors. Experiments on the Librispeech biasing benchmark show that our method effectively filters out majority of the candidate phrases, and significantly improves recognition accuracy under different biasing conditions when the scores are used in shallow fusion biasing. Our approach is modular and can be used with any ASR system, and the filtering mechanism can potentially boost performance of other biasing methods.
#### Forward Convolutive Prediction for Frame Online Monaural Speech Dereverberation Based on Kronecker Product Decomposition
 - **Authors:** Yujie Zhu, Jilu Jin, Xueqin Luo, Wenxing Yang, Zhong-Qiu Wang, Gongping Huang, Jingdong Chen, Jacob Benesty
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2510.24471

 - **Pdf link:** https://arxiv.org/pdf/2510.24471

 - **Abstract**
 Dereverberation has long been a crucial research topic in speech processing, aiming to alleviate the adverse effects of reverberation in voice communication and speech interaction systems. Among existing approaches, forward convolutional prediction (FCP) has recently attracted attention. It typically employs a deep neural network to predict the direct-path signal and subsequently estimates a linear prediction filter to suppress residual reverberation. However, a major drawback of this approach is that the required linear prediction filter is often excessively long, leading to considerable computational complexity. To address this, our work proposes a novel FCP method based on Kronecker product (KP) decomposition, in which the long prediction filter is modeled as the KP of two much shorter filters. This decomposition significantly reduces the computational cost. An adaptive algorithm is then provided to iteratively update these shorter filters online. Experimental results show that, compared to conventional methods, our approach achieves competitive dereverberation performance while substantially reducing computational cost.
#### emg2speech: synthesizing speech from electromyography using self-supervised speech models
 - **Authors:** Harshavardhana T. Gowda, Lee M. Miller
 - **Subjects:** Subjects:
Sound (cs.SD); Computation and Language (cs.CL); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2510.23969

 - **Pdf link:** https://arxiv.org/pdf/2510.23969

 - **Abstract**
 We present a neuromuscular speech interface that translates electromyographic (EMG) signals collected from orofacial muscles during speech articulation directly into audio. We show that self-supervised speech (SS) representations exhibit a strong linear relationship with the electrical power of muscle action potentials: SS features can be linearly mapped to EMG power with a correlation of $r = 0.85$. Moreover, EMG power vectors corresponding to different articulatory gestures form structured and separable clusters in feature space. This relationship: $\text{SS features}$ $\xrightarrow{\texttt{linear mapping}}$ $\text{EMG power}$ $\xrightarrow{\texttt{gesture-specific clustering}}$ $\text{articulatory movements}$, highlights that SS models implicitly encode articulatory mechanisms. Leveraging this property, we directly map EMG signals to SS feature space and synthesize speech, enabling end-to-end EMG-to-speech generation without explicit articulatory models and vocoder training.
#### TsetlinKWS: A 65nm 16.58uW, 0.63mm2 State-Driven Convolutional Tsetlin Machine-Based Accelerator For Keyword Spotting
 - **Authors:** Baizhou Lin, Yuetong Fang, Renjing Xu, Rishad Shafik, Jagmohan Chauhan
 - **Subjects:** Subjects:
Sound (cs.SD); Hardware Architecture (cs.AR); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2510.24282

 - **Pdf link:** https://arxiv.org/pdf/2510.24282

 - **Abstract**
 The Tsetlin Machine (TM) has recently attracted attention as a low-power alternative to neural networks due to its simple and interpretable inference mechanisms. However, its performance on speech-related tasks remains limited. This paper proposes TsetlinKWS, the first algorithm-hardware co-design framework for the Convolutional Tsetlin Machine (CTM) on the 12-keyword spotting task. Firstly, we introduce a novel Mel-Frequency Spectral Coefficient and Spectral Flux (MFSC-SF) feature extraction scheme together with spectral convolution, enabling the CTM to reach its first-ever competitive accuracy of 87.35% on the 12-keyword spotting task. Secondly, we develop an Optimized Grouped Block-Compressed Sparse Row (OG-BCSR) algorithm that achieves a remarkable 9.84$\times$ reduction in model size, significantly improving the storage efficiency on CTMs. Finally, we propose a state-driven architecture tailored for the CTM, which simultaneously exploits data reuse and sparsity to achieve high energy efficiency. The full system is evaluated in 65 nm process technology, consuming 16.58 $\mu$W at 0.7 V with a compact 0.63 mm$^2$ core area. TsetlinKWS requires only 907k logic operations per inference, representing a 10$\times$ reduction compared to the state-of-the-art KWS accelerators, positioning the CTM as a highly-efficient candidate for ultra-low-power speech applications.
#### Bayesian Speech synthesizers Can Learn from Multiple Teachers
 - **Authors:** Ziyang Zhang, Yifan Gao, Xuenan Xu, Baoxiangli, Wen Wu, Chao Zhang
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2510.24372

 - **Pdf link:** https://arxiv.org/pdf/2510.24372

 - **Abstract**
 Codec-based text-to-speech (TTS) models have recently gained traction for their efficiency and strong performance in voice cloning. However, codec-based TTS faces limitations due to the challenges of pretraining robust speech codecs and the quality degradation introduced by quantization errors. Emerging evidence suggests that continuous-valued generative models can alleviate these issues and serve as a promising alternative. Yet, effectively modelling diverse speech patterns and developing reliable sampling strategies for continuous-valued autoregressive (AR) TTS remains underexplored. In this work, we propose BELLE, Bayesian evidential learning with language modelling for TTS, a novel continuous-valued AR framework that directly predicts mel-spectrograms from textual input. BELLE treats each mel-spectrogram frame as a Gaussian distribution sampled from a learned hyper distribution, enabling principled uncertainty estimation, particularly in scenarios with parallel data (i.e., one text-audio prompt paired with multiple speech samples). To obtain such data, diverse speech samples are synthesized using multiple pre-trained TTS models given the same text-audio prompts, which are distilled into BELLE via Bayesian evidential learning. Experimental results indicate that BELLE demonstrates highly competitive performance compared with the current best open-source TTS models, even though BELLE is trained on a large amount of synthetic data and uses only approximately one-tenth of their training data. Audio samples generated by BELLE are available at this https URL. The code, checkpoints, and synthetic data will be released after the paper is accepted.
#### Your Microphone Array Retains Your Identity: A Robust Voice Liveness Detection System for Smart Speakers
 - **Authors:** Yan Meng, Jiachun Li, Matthew Pillari, Arjun Deopujari, Liam Brennan, Hafsah Shamsie, Haojin Zhu, Yuan Tian
 - **Subjects:** Subjects:
Cryptography and Security (cs.CR); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2510.24393

 - **Pdf link:** https://arxiv.org/pdf/2510.24393

 - **Abstract**
 Though playing an essential role in smart home systems, smart speakers are vulnerable to voice spoofing attacks. Passive liveness detection, which utilizes only the collected audio rather than the deployed sensors to distinguish between live-human and replayed voices, has drawn increasing attention. However, it faces the challenge of performance degradation under the different environmental factors as well as the strict requirement of the fixed user gestures. In this study, we propose a novel liveness feature, array fingerprint, which utilizes the microphone array inherently adopted by the smart speaker to determine the identity of collected audios. Our theoretical analysis demonstrates that by leveraging the circular layout of microphones, compared with existing schemes, array fingerprint achieves a more robust performance under the environmental change and user's movement. Then, to leverage such a fingerprint, we propose ARRAYID, a lightweight passive detection scheme, and elaborate a series of features working together with array fingerprint. Our evaluation on the dataset containing 32,780 audio samples and 14 spoofing devices shows that ARRAYID achieves an accuracy of 99.84%, which is superior to existing passive liveness detection schemes.
#### Online neural fusion of distortionless differential beamformers for robust speech enhancement
 - **Authors:** Yuanhang Qian, Kunlong Zhao, Jilu Jin, Xueqin Luo, Gongping Huang, Jingdong Chen, Jacob Benesty
 - **Subjects:** Subjects:
Sound (cs.SD); Artificial Intelligence (cs.AI); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2510.24497

 - **Pdf link:** https://arxiv.org/pdf/2510.24497

 - **Abstract**
 Fixed beamforming is widely used in practice since it does not depend on the estimation of noise statistics and provides relatively stable performance. However, a single beamformer cannot adapt to varying acoustic conditions, which limits its interference suppression capability. To address this, adaptive convex combination (ACC) algorithms have been introduced, where the outputs of multiple fixed beamformers are linearly combined to improve robustness. Nevertheless, ACC often fails in highly non-stationary scenarios, such as rapidly moving interference, since its adaptive updates cannot reliably track rapid changes. To overcome this limitation, we propose a frame-online neural fusion framework for multiple distortionless differential beamformers, which estimates the combination weights through a neural network. Compared with conventional ACC, the proposed method adapts more effectively to dynamic acoustic environments, achieving stronger interference suppression while maintaining the distortionless constraint.
#### Audio Signal Processing Using Time Domain Mel-Frequency Wavelet Coefficient
 - **Authors:** Rinku Sebastian, Simon O'Keefe, Martin Trefzer
 - **Subjects:** Subjects:
Sound (cs.SD); Artificial Intelligence (cs.AI); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2510.24519

 - **Pdf link:** https://arxiv.org/pdf/2510.24519

 - **Abstract**
 Extracting features from the speech is the most critical process in speech signal processing. Mel Frequency Cepstral Coefficients (MFCC) are the most widely used features in the majority of the speaker and speech recognition applications, as the filtering in this feature is similar to the filtering taking place in the human ear. But the main drawback of this feature is that it provides only the frequency information of the signal but does not provide the information about at what time which frequency is present. The wavelet transform, with its flexible time-frequency window, provides time and frequency information of the signal and is an appropriate tool for the analysis of non-stationary signals like speech. On the other hand, because of its uniform frequency scaling, a typical wavelet transform may be less effective in analysing speech signals, have poorer frequency resolution in low frequencies, and be less in line with human auditory perception. Hence, it is necessary to develop a feature that incorporates the merits of both MFCC and wavelet transform. A great deal of studies are trying to combine both these features. The present Wavelet Transform based Mel-scaled feature extraction methods require more computation when a wavelet transform is applied on top of Mel-scale filtering, since it adds extra processing steps. Here we are proposing a method to extract Mel scale features in time domain combining the concept of wavelet transform, thus reducing the computational burden of time-frequency conversion and the complexity of wavelet extraction. Combining our proposed Time domain Mel frequency Wavelet Coefficient(TMFWC) technique with the reservoir computing methodology has significantly improved the efficiency of audio signal processing.


by Zyzzyva0381 (Windy). 


2025-10-29
