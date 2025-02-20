# Showing new listings for Thursday, 20 February 2025
Auto update papers at about 2:30am UTC (10:30am Beijing time) every weekday.


阅读 `Usage.md`了解如何使用此repo实现个性化的Arxiv论文推送

See `Usage.md` for instructions on how to personalize the repo. 


Keyword list: ['acoustic', 'reinforcement learning', 'reverb', 'meta', 'separate', 'reconstruction', 'noise', 'enhance', 'localization', 'speech']


Excluded: []


### Today: 7papers 
#### Adopting Whisper for Confidence Estimation
 - **Authors:** Vaibhav Aggarwal, Shabari S Nair, Yash Verma, Yash Jogi
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Machine Learning (cs.LG)
 - **Arxiv link:** https://arxiv.org/abs/2502.13446

 - **Pdf link:** https://arxiv.org/pdf/2502.13446

 - **Abstract**
 Recent research on word-level confidence estimation for speech recognition systems has primarily focused on lightweight models known as Confidence Estimation Modules (CEMs), which rely on hand-engineered features derived from Automatic Speech Recognition (ASR) outputs. In contrast, we propose a novel end-to-end approach that leverages the ASR model itself (Whisper) to generate word-level confidence scores. Specifically, we introduce a method in which the Whisper model is fine-tuned to produce scalar confidence scores given an audio input and its corresponding hypothesis transcript. Our experiments demonstrate that the fine-tuned Whisper-tiny model, comparable in size to a strong CEM baseline, achieves similar performance on the in-domain dataset and surpasses the CEM baseline on eight out-of-domain datasets, whereas the fine-tuned Whisper-large model consistently outperforms the CEM baseline by a substantial margin across all datasets.
#### Multi-channel Replay Speech Detection using an Adaptive Learnable Beamformer
 - **Authors:** Michael Neri, Tuomas Virtanen
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Signal Processing (eess.SP)
 - **Arxiv link:** https://arxiv.org/abs/2502.13473

 - **Pdf link:** https://arxiv.org/pdf/2502.13473

 - **Abstract**
 Replay attacks belong to the class of severe threats against voice-controlled systems, exploiting the easy accessibility of speech signals by recorded and replayed speech to grant unauthorized access to sensitive data. In this work, we propose a multi-channel neural network architecture called M-ALRAD for the detection of replay attacks based on spatial audio features. This approach integrates a learnable adaptive beamformer with a convolutional recurrent neural network, allowing for joint optimization of spatial filtering and classification. Experiments have been carried out on the ReMASC dataset, which is a state-of-the-art multi-channel replay speech detection dataset encompassing four microphones with diverse array configurations and four environments. Results on the ReMASC dataset show the superiority of the approach compared to the state-of-the-art and yield substantial improvements for challenging acoustic environments. In addition, we demonstrate that our approach is able to better generalize to unseen environments with respect to prior studies.
#### Unsupervised CP-UNet Framework for Denoising DAS Data with Decay Noise
 - **Authors:** Tianye Huang, Aopeng Li, Xiang Li, Jing Zhang, Sijing Xian, Qi Zhang, Mingkong Lu, Guodong Chen, Liangming Xiong, Xiangyun Hu
 - **Subjects:** Subjects:
Sound (cs.SD); Machine Learning (cs.LG); Audio and Speech Processing (eess.AS); Signal Processing (eess.SP); Optics (physics.optics)
 - **Arxiv link:** https://arxiv.org/abs/2502.13395

 - **Pdf link:** https://arxiv.org/pdf/2502.13395

 - **Abstract**
 Distributed acoustic sensor (DAS) technology leverages optical fiber cables to detect acoustic signals, providing cost-effective and dense monitoring capabilities. It offers several advantages including resistance to extreme conditions, immunity to electromagnetic interference, and accurate detection. However, DAS typically exhibits a lower signal-to-noise ratio (S/N) compared to geophones and is susceptible to various noise types, such as random noise, erratic noise, level noise, and long-period noise. This reduced S/N can negatively impact data analyses containing inversion and interpretation. While artificial intelligence has demonstrated excellent denoising capabilities, most existing methods rely on supervised learning with labeled data, which imposes stringent requirements on the quality of the labels. To address this issue, we develop a label-free unsupervised learning (UL) network model based on Context-Pyramid-UNet (CP-UNet) to suppress erratic and random noises in DAS data. The CP-UNet utilizes the Context Pyramid Module in the encoding and decoding process to extract features and reconstruct the DAS data. To enhance the connectivity between shallow and deep features, we add a Connected Module (CM) to both encoding and decoding section. Layer Normalization (LN) is utilized to replace the commonly employed Batch Normalization (BN), accelerating the convergence of the model and preventing gradient explosion during training. Huber-loss is adopted as our loss function whose parameters are experimentally determined. We apply the network to both the 2-D synthetic and filed data. Comparing to traditional denoising methods and the latest UL framework, our proposed method demonstrates superior noise reduction performance.
#### Semi-supervised classification of bird vocalizations
 - **Authors:** Simen Hexeberg, Mandar Chitre, Matthias Hoffmann-Kuhnt, Bing Wen Low
 - **Subjects:** Subjects:
Sound (cs.SD); Artificial Intelligence (cs.AI); Computer Vision and Pattern Recognition (cs.CV); Audio and Speech Processing (eess.AS); Quantitative Methods (q-bio.QM)
 - **Arxiv link:** https://arxiv.org/abs/2502.13440

 - **Pdf link:** https://arxiv.org/pdf/2502.13440

 - **Abstract**
 Changes in bird populations can indicate broader changes in ecosystems, making birds one of the most important animal groups to monitor. Combining machine learning and passive acoustics enables continuous monitoring over extended periods without direct human involvement. However, most existing techniques require extensive expert-labeled datasets for training and cannot easily detect time-overlapping calls in busy soundscapes. We propose a semi-supervised acoustic bird detector designed to allow both the detection of time-overlapping calls (when separated in frequency) and the use of few labeled training samples. The classifier is trained and evaluated on a combination of community-recorded open-source data and long-duration soundscape recordings from Singapore. It achieves a mean F0.5 score of 0.701 across 315 classes from 110 bird species on a hold-out test set, with an average of 11 labeled training samples per class. It outperforms the state-of-the-art BirdNET classifier on a test set of 103 bird species despite significantly fewer labeled training samples. The detector is further tested on 144 microphone-hours of continuous soundscape data. The rich soundscape in Singapore makes suppression of false positives a challenge on raw, continuous data streams. Nevertheless, we demonstrate that achieving high precision in such environments with minimal labeled training data is possible.
#### RestoreGrad: Signal Restoration Using Conditional Denoising Diffusion Models with Jointly Learned Prior
 - **Authors:** Ching-Hua Lee, Chouchang Yang, Jaejin Cho, Yashas Malur Saidutta, Rakshith Sharma Srinivasa, Yilin Shen, Hongxia Jin
 - **Subjects:** Subjects:
Image and Video Processing (eess.IV); Machine Learning (cs.LG); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2502.13574

 - **Pdf link:** https://arxiv.org/pdf/2502.13574

 - **Abstract**
 Denoising diffusion probabilistic models (DDPMs) can be utilized for recovering a clean signal from its degraded observation(s) by conditioning the model on the degraded signal. The degraded signals are themselves contaminated versions of the clean signals; due to this correlation, they may encompass certain useful information about the target clean data distribution. However, existing adoption of the standard Gaussian as the prior distribution in turn discards such information, resulting in sub-optimal performance. In this paper, we propose to improve conditional DDPMs for signal restoration by leveraging a more informative prior that is jointly learned with the diffusion model. The proposed framework, called RestoreGrad, seamlessly integrates DDPMs into the variational autoencoder framework and exploits the correlation between the degraded and clean signals to encode a better diffusion prior. On speech and image restoration tasks, we show that RestoreGrad demonstrates faster convergence (5-10 times fewer training steps) to achieve better quality of restored signals over existing DDPM baselines, and improved robustness to using fewer sampling steps in inference time (2-2.5 times fewer), advocating the advantages of leveraging jointly learned prior for efficiency improvements in the diffusion process.
#### TALKPLAY: Multimodal Music Recommendation with Large Language Models
 - **Authors:** Seungheon Doh, Keunwoo Choi, Juhan Nam
 - **Subjects:** Subjects:
Information Retrieval (cs.IR); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2502.13713

 - **Pdf link:** https://arxiv.org/pdf/2502.13713

 - **Abstract**
 We present TalkPlay, a multimodal music recommendation system that reformulates the recommendation task as large language model token generation. TalkPlay represents music through an expanded token vocabulary that encodes multiple modalities - audio, lyrics, metadata, semantic tags, and playlist co-occurrence. Using these rich representations, the model learns to generate recommendations through next-token prediction on music recommendation conversations, that requires learning the associations natural language query and response, as well as music items. In other words, the formulation transforms music recommendation into a natural language understanding task, where the model's ability to predict conversation tokens directly optimizes query-item relevance. Our approach eliminates traditional recommendation-dialogue pipeline complexity, enabling end-to-end learning of query-aware music recommendations. In the experiment, TalkPlay is successfully trained and outperforms baseline methods in various aspects, demonstrating strong context understanding as a conversational music recommender.
#### Audio-Based Classification of Insect Species Using Machine Learning Models: Cicada, Beetle, Termite, and Cricket
 - **Authors:** Manas V Shetty, Yoga Disha Sendhil Kumar
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2502.13893

 - **Pdf link:** https://arxiv.org/pdf/2502.13893

 - **Abstract**
 This project addresses the challenge of classifying insect species: Cicada, Beetle, Termite, and Cricket using sound recordings. Accurate species identification is crucial for ecological monitoring and pest management. We employ machine learning models such as XGBoost, Random Forest, and K Nearest Neighbors (KNN) to analyze audio features, including Mel Frequency Cepstral Coefficients (MFCC). The potential novelty of this work lies in the combination of diverse audio features and machine learning models to tackle insect classification, specifically focusing on capturing subtle acoustic variations between species that have not been fully leveraged in previous research. The dataset is compiled from various open sources, and we anticipate achieving high classification accuracy, contributing to improved automated insect detection systems.


by Zyzzyva0381 (Windy). 


2025-02-20
