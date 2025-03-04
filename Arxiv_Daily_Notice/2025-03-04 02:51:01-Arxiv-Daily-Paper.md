# Showing new listings for Monday, 3 March 2025
Auto update papers at about 2:30am UTC (10:30am Beijing time) every weekday.


阅读 `Usage.md`了解如何使用此repo实现个性化的Arxiv论文推送

See `Usage.md` for instructions on how to personalize the repo. 


Keyword list: ['acoustic', 'reinforcement learning', 'reverb', 'meta', 'separate', 'reconstruction', 'noise', 'enhance', 'localization', 'speech']


Excluded: []


### Today: 4papers 
#### JiTTER: Jigsaw Temporal Transformer for Event Reconstruction for Self-Supervised Sound Event Detection
 - **Authors:** Hyeonuk Nam, Yong-Hwa Park
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2502.20857

 - **Pdf link:** https://arxiv.org/pdf/2502.20857

 - **Abstract**
 Sound event detection (SED) has significantly benefited from self-supervised learning (SSL) approaches, particularly masked audio transformer for SED (MAT-SED), which leverages masked block prediction to reconstruct missing audio segments. However, while effective in capturing global dependencies, masked block prediction disrupts transient sound events and lacks explicit enforcement of temporal order, making it less suitable for fine-grained event boundary detection. To address these limitations, we propose JiTTER (Jigsaw Temporal Transformer for Event Reconstruction), an SSL framework designed to enhance temporal modeling in transformer-based SED. JiTTER introduces a hierarchical temporal shuffle reconstruction strategy, where audio sequences are randomly shuffled at both the block-level and frame-level, forcing the model to reconstruct the correct temporal order. This pretraining objective encourages the model to learn both global event structures and fine-grained transient details, improving its ability to detect events with sharp onset-offset characteristics. Additionally, we incorporate noise injection during block shuffle, providing a subtle perturbation mechanism that further regularizes feature learning and enhances model robustness. Experimental results on the DESED dataset demonstrate that JiTTER outperforms MAT-SED, achieving a 5.89% improvement in PSDS, highlighting the effectiveness of explicit temporal reasoning in SSL-based SED. Our findings suggest that structured temporal reconstruction tasks, rather than simple masked prediction, offer a more effective pretraining paradigm for sound event representation learning.
#### LiteASR: Efficient Automatic Speech Recognition with Low-Rank Approximation
 - **Authors:** Keisuke Kamahori, Jungo Kasai, Noriyuki Kojima, Baris Kasikci
 - **Subjects:** Subjects:
Machine Learning (cs.LG); Artificial Intelligence (cs.AI); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2502.20583

 - **Pdf link:** https://arxiv.org/pdf/2502.20583

 - **Abstract**
 Modern automatic speech recognition (ASR) models, such as OpenAI's Whisper, rely on deep encoder-decoder architectures, and their encoders are a critical bottleneck for efficient deployment due to high computational intensity. We introduce LiteASR, a low-rank compression scheme for ASR encoders that significantly reduces inference costs while maintaining transcription accuracy. Our approach leverages the strong low-rank properties observed in intermediate activations: by applying principal component analysis (PCA) with a small calibration dataset, we approximate linear transformations with a chain of low-rank matrix multiplications, and further optimize self-attention to work in the reduced dimension. Evaluation results show that our method can compress Whisper large-v3's encoder size by over 50%, matching Whisper medium's size with better transcription accuracy, thereby establishing a new Pareto-optimal frontier of efficiency and performance. The code of LiteASR is available at this https URL.
#### Weakly Supervised Multiple Instance Learning for Whale Call Detection and Localization in Long-Duration Passive Acoustic Monitoring
 - **Authors:** Ragib Amin Nihal, Benjamin Yen, Runwu Shi, Kazuhiro Nakadai
 - **Subjects:** Subjects:
Sound (cs.SD); Artificial Intelligence (cs.AI); Machine Learning (cs.LG); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2502.20838

 - **Pdf link:** https://arxiv.org/pdf/2502.20838

 - **Abstract**
 Marine ecosystem monitoring via Passive Acoustic Monitoring (PAM) generates vast data, but deep learning often requires precise annotations and short segments. We introduce DSMIL-LocNet, a Multiple Instance Learning framework for whale call detection and localization using only bag-level labels. Our dual-stream model processes 2-30 minute audio segments, leveraging spectral and temporal features with attention-based instance selection. Tests on Antarctic whale data show longer contexts improve classification (F1: 0.8-0.9) while medium instances ensure localization precision (0.65-0.70). This suggests MIL can enhance scalable marine monitoring. Code: this https URL
#### Deep learning-based filtering of cross-spectral matrices using generative adversarial networks
 - **Authors:** Christof Puhle
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS); Signal Processing (eess.SP)
 - **Arxiv link:** https://arxiv.org/abs/2502.21097

 - **Pdf link:** https://arxiv.org/pdf/2502.21097

 - **Abstract**
 In this paper, we present a deep-learning method to filter out effects such as ambient noise, reflections, or source directivity from microphone array data represented as cross-spectral matrices. Specifically, we focus on a generative adversarial network (GAN) architecture designed to transform fixed-size cross-spectral matrices. Theses models were trained using sound pressure simulations of varying complexity developed for this purpose. Based on the results from applying these methods in a hyperparameter optimization of an auto-encoding task, we trained the optimized model to perform five distinct transformation tasks derived from different complexities inherent in our sound pressure simulations.


by Zyzzyva0381 (Windy). 


2025-03-04
