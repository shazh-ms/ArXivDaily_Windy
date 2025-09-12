# Showing new listings for Friday, 12 September 2025
Auto update papers at about 2:30am UTC (10:30am Beijing time) every weekday.


阅读 `Usage.md`了解如何使用此repo实现个性化的Arxiv论文推送

See `Usage.md` for instructions on how to personalize the repo. 


Keyword list: ['text-to-speech', 'text to speech', 'tts', 'LLM-based', 'speech', 'voice']


Excluded: []


### Today: 4papers 
#### MAPSS: Manifold-based Assessment of Perceptual Source Separation
 - **Authors:** Amir Ivry, Samuele Cornell, Shinji Watanabe
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2509.09212

 - **Pdf link:** https://arxiv.org/pdf/2509.09212

 - **Abstract**
 Objective assessment of source-separation systems still mismatches subjective human perception, especially when leakage and self-distortion interact. We introduce the Perceptual Separation (PS) and Perceptual Match (PM), the first pair of measures that functionally isolate these two factors. Our intrusive method begins with generating a bank of fundamental distortions for each reference waveform signal in the mixture. Distortions, references, and their respective system outputs from all sources are then independently encoded by a pre-trained self-supervised learning model. These representations are aggregated and projected onto a manifold via diffusion maps, which aligns Euclidean distances on the manifold with dissimilarities of the encoded waveforms. On this manifold, the PM measures the Mahalanobis distance from each output to its attributed cluster that consists of its reference and distortions embeddings, capturing self-distortion. The PS accounts for the Mahalanobis distance of the output to the attributed and to the closest non-attributed clusters, quantifying leakage. Both measures are differentiable and granular, operating at a resolution as low as 50 frames per second. We further derive, for both measures, deterministic error radius and non-asymptotic, high-probability confidence intervals (CIs). Experiments on English, Spanish, and music mixtures show that the PS and PM nearly always achieve the highest linear correlation coefficients with human mean-opinion scores than 14 competitors, reaching as high as 86.36% for speech and 87.21% for music. We observe, at worst, an error radius of 1.39% and a probabilistic 95% CI of 12.21% for these coefficients, which improves reliable and informed evaluation. Using mutual information, the measures complement each other most as their values decrease, suggesting they are jointly more informative as system performance degrades.
#### Over-the-Air Adversarial Attack Detection: from Datasets to Defenses
 - **Authors:** Li Wang, Xiaoyan Lei, Haorui He, Lei Wang, Jie Shi, Zhizheng Wu
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2509.09296

 - **Pdf link:** https://arxiv.org/pdf/2509.09296

 - **Abstract**
 Automatic Speaker Verification (ASV) systems can be used for voice-enabled applications for identity verification. However, recent studies have exposed these systems' vulnerabilities to both over-the-line (OTL) and over-the-air (OTA) adversarial attacks. Although various detection methods have been proposed to counter these threats, they have not been thoroughly tested due to the lack of a comprehensive data set. To address this gap, we developed the AdvSV 2.0 dataset, which contains 628k samples with a total duration of 800 hours. This dataset incorporates classical adversarial attack algorithms, ASV systems, and encompasses both OTL and OTA scenarios. Furthermore, we introduce a novel adversarial attack method based on a Neural Replay Simulator (NRS), which enhances the potency of adversarial OTA attacks, thereby presenting a greater threat to ASV systems. To defend against these attacks, we propose CODA-OCC, a contrastive learning approach within the one-class classification framework. Experimental results show that CODA-OCC achieves an EER of 11.2% and an AUC of 0.95 on the AdvSV 2.0 dataset, outperforming several state-of-the-art detection methods.
#### Listening for "You": Enhancing Speech Image Retrieval via Target Speaker Extraction
 - **Authors:** Wenhao Yang, Jianguo Wei, Wenhuan Lu, Xinyue Song, Xianghu Yue
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Image and Video Processing (eess.IV)
 - **Arxiv link:** https://arxiv.org/abs/2509.09306

 - **Pdf link:** https://arxiv.org/pdf/2509.09306

 - **Abstract**
 Image retrieval using spoken language cues has emerged as a promising direction in multimodal perception, yet leveraging speech in multi-speaker scenarios remains challenging. We propose a novel Target Speaker Speech-Image Retrieval task and a framework that learns the relationship between images and multi-speaker speech signals in the presence of a target speaker. Our method integrates pre-trained self-supervised audio encoders with vision models via target speaker-aware contrastive learning, conditioned on a Target Speaker Extraction and Retrieval module. This enables the system to extract spoken commands from the target speaker and align them with corresponding images. Experiments on SpokenCOCO2Mix and SpokenCOCO3Mix show that TSRE significantly outperforms existing methods, achieving 36.3% and 29.9% Recall@1 in 2 and 3 speaker scenarios, respectively - substantial improvements over single speaker baselines and state-of-the-art models. Our approach demonstrates potential for real-world deployment in assistive robotics and multimodal interaction systems.
#### Acoustic to Articulatory Speech Inversion for Children with Velopharyngeal Insufficiency
 - **Authors:** Saba Tabatabaee, Suzanne Boyce, Liran Oren, Mark Tiede, Carol Espy-Wilson
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2509.09489

 - **Pdf link:** https://arxiv.org/pdf/2509.09489

 - **Abstract**
 Traditional clinical approaches for assessing nasality, such as nasopharyngoscopy and nasometry, involve unpleasant experiences and are problematic for children. Speech Inversion (SI), a noninvasive technique, offers a promising alternative for estimating articulatory movement without the need for physical instrumentation. In this study, an SI system trained on nasalance data from healthy adults is augmented with source information from electroglottography and acoustically derived F0, periodic and aperiodic energy estimates as proxies for glottal control. This model achieves 16.92% relative improvement in Pearson Product-Moment Correlation (PPMC) compared to a previous SI system for nasalance estimation. To adapt the SI system for nasalance estimation in children with Velopharyngeal Insufficiency (VPI), the model initially trained on adult speech was fine-tuned using children with VPI data, yielding an 7.90% relative improvement in PPMC compared to its performance before fine-tuning.


by Zyzzyva0381 (Windy). 


2025-09-12
