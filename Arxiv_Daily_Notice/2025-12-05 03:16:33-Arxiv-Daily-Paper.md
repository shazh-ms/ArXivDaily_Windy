# Showing new listings for Friday, 5 December 2025
Auto update papers at about 2:30am UTC (10:30am Beijing time) every weekday.


阅读 `Usage.md`了解如何使用此repo实现个性化的Arxiv论文推送

See `Usage.md` for instructions on how to personalize the repo. 


Keyword list: ['text-to-speech', 'text to speech', 'tts', 'LLM-based', 'speech', 'voice']


Excluded: []


### Today: 5papers 
#### Towards predicting binaural audio quality in listeners with normal and impaired hearing
 - **Authors:** Thomas Biberger, Stephan D. Ewert
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2512.04792

 - **Pdf link:** https://arxiv.org/pdf/2512.04792

 - **Abstract**
 Eurich et al. (2024) recently introduced the computationally efficient monaural and binaural audio quality model (eMoBi-Q). This model integrates both monaural and binaural auditory features and has been validated across six audio datasets encompassing quality ratings for music and speech, processed via algorithms commonly employed in modern hearing devices (e.g., acoustic transparency, feedback cancellation, and binaural beamforming) or presented via loudspeakers. In the current study, we expand eMoBi-Q to account for perceptual effects of sensorineural hearing loss (HL) on audio quality. For this, the model was extended by a nonlinear auditory filterbank. Given that altered loudness perception is a prevalent issue among listeners with hearing impairment, our goal is to incorporate loudness as a sub-dimension for predicting audio quality in both normal-hearing and hearing-impaired populations. While predicting loudness itself is important in the context of loudness-based hearing aid fitting, loudness as audio quality sub-measure may be helpful for the selection of reliable auditory features in hearing impaired listeners. The parameters of the filterbank and subsequent processing stages were informed by the physiologically-based (binaural) loudness model proposed by Pieper et al. (2018). This study presents and discusses the initial implementation of the extended binaural quality model.
#### TripleC Learning and Lightweight Speech Enhancement for Multi-Condition Target Speech Extraction
 - **Authors:** Ziling Huang (Shanghai Normal University, China)
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2512.04945

 - **Pdf link:** https://arxiv.org/pdf/2512.04945

 - **Abstract**
 In our recent work, we proposed Lightweight Speech Enhancement Guided Target Speech Extraction (LGTSE) and demonstrated its effectiveness in multi-speaker-plus-noise scenarios. However, real-world applications often involve more diverse and complex conditions, such as one-speaker-plus-noise or two-speaker-without-noise. To address this challenge, we extend LGTSE with a Cross-Condition Consistency learning strategy, termed TripleC Learning. This strategy is first validated under multi-speaker-plus-noise condition and then evaluated for its generalization across diverse scenarios. Moreover, building upon the lightweight front-end denoiser in LGTSE, which can flexibly process both noisy and clean mixtures and shows strong generalization to unseen conditions, we integrate TripleC learning with a proposed parallel universal training scheme that organizes batches containing multiple scenarios for the same target speaker. By enforcing consistent extraction across different conditions, easier cases can assist harder ones, thereby fully exploiting diverse training data and fostering a robust universal model. Experimental results on the Libri2Mix three-condition tasks demonstrate that the proposed LGTSE with TripleC learning achieves superior performance over condition-specific models, highlighting its strong potential for universal deployment in real-world speech applications.
#### HiPPO: Exploring A Novel Hierarchical Pronunciation Assessment Approach for Spoken Languages
 - **Authors:** Bi-Cheng Yan, Hsin-Wei Wang, Fu-An Chao, Tien-Hong Lo, Yung-Chang Hsu, Berlin Chen
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2512.04964

 - **Pdf link:** https://arxiv.org/pdf/2512.04964

 - **Abstract**
 Automatic pronunciation assessment (APA) seeks to quantify a second language (L2) learner's pronunciation proficiency in a target language by offering timely and fine-grained diagnostic feedback. Most existing efforts on APA have predominantly concentrated on highly constrained reading-aloud tasks (where learners are prompted to read a reference text aloud); however, assessing pronunciation quality in unscripted speech (or free-speaking scenarios) remains relatively underexplored. In light of this, we first propose HiPPO, a hierarchical pronunciation assessment model tailored for spoken languages, which evaluates an L2 learner's oral proficiency at multiple linguistic levels based solely on the speech uttered by the learner. To improve the overall accuracy of assessment, a contrastive ordinal regularizer and a curriculum learning strategy are introduced for model training. The former aims to generate score-discriminative features by exploiting the ordinal nature of regression targets, while the latter gradually ramps up the training complexity to facilitate the assessment task that takes unscripted speech as input. Experiments conducted on the Speechocean762 benchmark dataset validates the feasibility and superiority of our method in relation to several cutting-edge baselines.
#### Multi-Loss Learning for Speech Emotion Recognition with Energy-Adaptive Mixup and Frame-Level Attention
 - **Authors:** Cong Wang, Yizhong Geng, Yuhua Wen, Qifei Li, Yingming Gao, Ruimin Wang, Chunfeng Wang, Hao Li, Ya Li, Wei Chen
 - **Subjects:** Subjects:
Sound (cs.SD); Artificial Intelligence (cs.AI); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2512.04551

 - **Pdf link:** https://arxiv.org/pdf/2512.04551

 - **Abstract**
 Speech emotion recognition (SER) is an important technology in human-computer interaction. However, achieving high performance is challenging due to emotional complexity and scarce annotated data. To tackle these challenges, we propose a multi-loss learning (MLL) framework integrating an energy-adaptive mixup (EAM) method and a frame-level attention module (FLAM). The EAM method leverages SNR-based augmentation to generate diverse speech samples capturing subtle emotional variations. FLAM enhances frame-level feature extraction for multi-frame emotional cues. Our MLL strategy combines Kullback-Leibler divergence, focal, center, and supervised contrastive loss to optimize learning, address class imbalance, and improve feature separability. We evaluate our method on four widely used SER datasets: IEMOCAP, MSP-IMPROV, RAVDESS, and SAVEE. The results demonstrate our method achieves state-of-the-art performance, suggesting its effectiveness and robustness.
#### RRPO: Robust Reward Policy Optimization for LLM-based Emotional TTS
 - **Authors:** Cong Wang, Changfeng Gao, Yang Xiang, Zhihao Du, Keyu An, Han Zhao, Qian Chen, Xiangang Li, Yingming Gao, Ya Li
 - **Subjects:** Subjects:
Sound (cs.SD); Artificial Intelligence (cs.AI); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2512.04552

 - **Pdf link:** https://arxiv.org/pdf/2512.04552

 - **Abstract**
 Differentiable reinforcement learning (RL) frameworks like DiffRO offer a powerful approach for controllable text-to-speech (TTS), but are vulnerable to reward hacking, particularly for nuanced tasks like emotion control. The policy model can exploit a vanilla Reward Model (RM) by generating acoustic artifacts to achieve spurious rewards, but at the cost of degrading perceptual quality. To address this, we propose Robust Reward Policy Optimization (RRPO), a novel framework that employs a hybrid regularization scheme. This scheme develops a robust RM whose reward signal is more reliably aligned with human perception, compelling the policy to abandon detrimental shortcuts and instead learn the complex features of genuine emotions. Our ablation study confirms the enhanced robustness of our RM, as evidenced by its strong cross-lingual generalization. The subjective evaluation demonstrates that this robust RM effectively mitigates reward hacking, leading to significant improvements in both emotional expressiveness and naturalness over all baselines. Demo page: this https URL.


by Zyzzyva0381 (Windy). 


2025-12-05
