# Showing new listings for Tuesday, 22 April 2025
Auto update papers at about 2:30am UTC (10:30am Beijing time) every weekday.


阅读 `Usage.md`了解如何使用此repo实现个性化的Arxiv论文推送

See `Usage.md` for instructions on how to personalize the repo. 


Keyword list: ['acoustic', 'reinforcement learning', 'reverb', 'meta', 'separate', 'reconstruction', 'noise', 'enhance', 'localization', 'speech']


Excluded: []


### Today: 7papers 
#### Data Augmentation Using Neural Acoustic Fields With Retrieval-Augmented Pre-training
 - **Authors:** Christopher Ick, Gordon Wichern, Yoshiki Masuyama, François G. Germain, Jonathan Le Roux
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Artificial Intelligence (cs.AI); Computer Vision and Pattern Recognition (cs.CV); Machine Learning (cs.LG); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2504.14409

 - **Pdf link:** https://arxiv.org/pdf/2504.14409

 - **Abstract**
 This report details MERL's system for room impulse response (RIR) estimation submitted to the Generative Data Augmentation Workshop at ICASSP 2025 for Augmenting RIR Data (Task 1) and Improving Speaker Distance Estimation (Task 2). We first pre-train a neural acoustic field conditioned by room geometry on an external large-scale dataset in which pairs of RIRs and the geometries are provided. The neural acoustic field is then adapted to each target room by using the enrollment data, where we leverage either the provided room geometries or geometries retrieved from the external dataset, depending on availability. Lastly, we predict the RIRs for each pair of source and receiver locations specified by Task 1, and use these RIRs to train the speaker distance estimation model in Task 2.
#### Predicting speech intelligibility in older adults using the Gammachirp Envelope Similarity Index, GESI
 - **Authors:** Ayako Yamamoto, Fuki Miyazaki, Toshio Irino
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2504.14437

 - **Pdf link:** https://arxiv.org/pdf/2504.14437

 - **Abstract**
 We propose an objective intelligibility measure (OIM), called the Gammachirp Envelope Similarity Index (GESI), that can predict speech intelligibility (SI) in older adults. GESI is a bottom-up model based on psychoacoustic knowledge from the peripheral to the central auditory system and requires no training data. It computes the single SI metric using the gammachirp filterbank (GCFB), the modulation filterbank, and the extended cosine similarity measure. It takes into account not only the hearing level represented in the audiogram, but also the temporal processing characteristics captured by the temporal modulation transfer function (TMTF). To evaluate performance, SI experiments were conducted with older adults of various hearing levels using speech-in-noise with ideal speech enhancement on familiarity-controlled words. The prediction performance was compared with HASPIw2, which was developed for keyword SI prediction. The results showed that GESI predicted the subjective SI scores more accurately than HASPIw2. The effect of introducing TMTF into the GESI algorithm was not significant, indicating that more research is needed to know how to introduce temporal response characteristics into the OIM.
#### DNN based HRIRs Identification with a Continuously Rotating Speaker Array
 - **Authors:** Byeong-Yun Ko, Deokki Min, Hyeonuk Nam, Yong-Hwa Park
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2504.14817

 - **Pdf link:** https://arxiv.org/pdf/2504.14817

 - **Abstract**
 Conventional static measurement of head-related impulse responses (HRIRs) is time-consuming due to the need for repositioning a speaker array for each azimuth angle. Dynamic approaches using analytical models with a continuously rotating speaker array have been proposed, but their accuracy is significantly reduced at high rotational speeds. To address this limitation, we propose a DNN-based HRIRs identification using sequence-to-sequence learning. The proposed DNN model incorporates fully connected (FC) networks to effectively capture HRIR transitions and includes reset and update gates to identify HRIRs over a whole sequence. The model updates the HRIRs vector coefficients based on the gradient of the instantaneous square error (ISE). Additionally, we introduce a learnable normalization process based on the speaker excitation signals to stabilize the gradient scale of ISE across time. A training scheme, referred to as whole-sequence updating and optimization scheme, is also introduced to prevent overfitting. We evaluated the proposed method through simulations and experiments. Simulation results using the FABIAN database show that the proposed method outperforms previous analytic models, achieving over 7 dB improvement in normalized misalignment (NM) and maintaining log spectral distortion (LSD) below 2 dB at a rotational speed of 45°/s. Experimental results with a custom-built speaker array confirm that the proposed method successfully preserved accurate sound localization cues, consistent with those from static measurement. Source code is available at this https URL
#### Quantitative Measures for Passive Sonar Texture Analysis
 - **Authors:** Jarin Ritu, Alexandra Van Dine, Joshua Peeples
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2504.14843

 - **Pdf link:** https://arxiv.org/pdf/2504.14843

 - **Abstract**
 Passive sonar signals contain complex characteristics often arising from environmental noise, vessel machinery, and propagation effects. While convolutional neural networks (CNNs) perform well on passive sonar classification tasks, they can struggle with statistical variations that occur in the data. To investigate this limitation, synthetic underwater acoustic datasets are generated that centered on amplitude and period variations. Two metrics are proposed to quantify and validate these characteristics in the context of statistical and structural texture for passive sonar. These measures are applied to real-world passive sonar datasets to assess texture information in the signals and correlate the performances of the models. Results show that CNNs underperform on statistically textured signals, but incorporating explicit statistical texture modeling yields consistent improvements. These findings highlight the importance of quantifying texture information for passive sonar classification.
#### StableQuant: Layer Adaptive Post-Training Quantization for Speech Foundation Models
 - **Authors:** Yeona Hong, Hyewon Han, Woo-jin Chung, Hong-Goo Kang
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Artificial Intelligence (cs.AI)
 - **Arxiv link:** https://arxiv.org/abs/2504.14915

 - **Pdf link:** https://arxiv.org/pdf/2504.14915

 - **Abstract**
 In this paper, we propose StableQuant, a novel adaptive post-training quantization (PTQ) algorithm for widely used speech foundation models (SFMs). While PTQ has been successfully employed for compressing large language models (LLMs) due to its ability to bypass additional fine-tuning, directly applying these techniques to SFMs may not yield optimal results, as SFMs utilize distinct network architecture for feature extraction. StableQuant demonstrates optimal quantization performance regardless of the network architecture type, as it adaptively determines the quantization range for each layer by analyzing both the scale distributions and overall performance. We evaluate our algorithm on two SFMs, HuBERT and wav2vec2.0, for an automatic speech recognition (ASR) task, and achieve superior performance compared to traditional PTQ methods. StableQuant successfully reduces the sizes of SFM models to a quarter and doubles the inference speed while limiting the word error rate (WER) performance drop to less than 0.3% with 8-bit quantization.
#### On feature representations for marmoset vocal communication analysis
 - **Authors:** Eklavya Sarkar, Kaja Wierucka, Alexandra B. Bosshard, Judith Burkart, Mathew Magimai.-Doss
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2504.14981

 - **Pdf link:** https://arxiv.org/pdf/2504.14981

 - **Abstract**
 The acoustic analysis of marmoset (Callithrix jacchus) vocalizations is often used to understand the evolutionary origins of human language. Currently, the analysis is largely carried out in a manual or semi-manual manner. Thus, there is a need to develop automatic call analysis methods. In that direction, research has been limited to the development of analysis methods with small amounts of data or for specific scenarios. Furthermore, there is lack of prior knowledge about what type of information is relevant for different call analysis tasks. To address these issues, as a first step, this paper explores different feature representation methods, namely, HCTSA-based hand-crafted features Catch22, pre-trained self supervised learning (SSL) based features extracted from neural networks trained on human speech and end-to-end acoustic modeling for call-type classification, caller identification and caller sex identification. Through an investigation on three different marmoset call datasets, we demonstrate that SSL-based feature representations and end-to-end acoustic modeling tend to lead to better systems than Catch22 features for call-type and caller classification. Furthermore, we also highlight the impact of signal bandwidth on the obtained task performances.
#### DiffVox: A Differentiable Model for Capturing and Analysing Professional Effects Distributions
 - **Authors:** Chin-Yun Yu, Marco A. Martínez-Ramírez, Junghyun Koo, Ben Hayes, Wei-Hsiang Liao, György Fazekas, Yuki Mitsufuji
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2504.14735

 - **Pdf link:** https://arxiv.org/pdf/2504.14735

 - **Abstract**
 This study introduces a novel and interpretable model, DiffVox, for matching vocal effects in music production. DiffVox, short for ``Differentiable Vocal Fx", integrates parametric equalisation, dynamic range control, delay, and reverb with efficient differentiable implementations to enable gradient-based optimisation for parameter estimation. Vocal presets are retrieved from two datasets, comprising 70 tracks from MedleyDB and 365 tracks from a private collection. Analysis of parameter correlations highlights strong relationships between effects and parameters, such as the high-pass and low-shelf filters often behaving together to shape the low end, and the delay time correlates with the intensity of the delayed signals. Principal component analysis reveals connections to McAdams' timbre dimensions, where the most crucial component modulates the perceived spaciousness while the secondary components influence spectral brightness. Statistical testing confirms the non-Gaussian nature of the parameter distribution, highlighting the complexity of the vocal effects space. These initial findings on the parameter distributions set the foundation for future research in vocal effects modelling and automatic mixing. Our source code and datasets are accessible at this https URL.


by Zyzzyva0381 (Windy). 


2025-04-22
