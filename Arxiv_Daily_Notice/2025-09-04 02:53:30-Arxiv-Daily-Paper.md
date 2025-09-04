# Showing new listings for Thursday, 4 September 2025
Auto update papers at about 2:30am UTC (10:30am Beijing time) every weekday.


阅读 `Usage.md`了解如何使用此repo实现个性化的Arxiv论文推送

See `Usage.md` for instructions on how to personalize the repo. 


Keyword list: ['text-to-speech', 'text to speech', 'tts', 'LLM-based', 'speech', 'voice']


Excluded: []


### Today: 11papers 
#### Gaussian Process Regression of Steering Vectors With Physics-Aware Deep Composite Kernels for Augmented Listening
 - **Authors:** Diego Di Carlo (RIKEN AIP), Koyama Shoichi (UTokyo), Nugraha Aditya Arie (RIKEN AIP), Fontaine Mathieu (LTCI, S2A), Bando Yoshiaki (AIST), Yoshii Kazuyoshi (RIKEN AIP)
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Machine Learning (cs.LG); Sound (cs.SD); Signal Processing (eess.SP)
 - **Arxiv link:** https://arxiv.org/abs/2509.02571

 - **Pdf link:** https://arxiv.org/pdf/2509.02571

 - **Abstract**
 This paper investigates continuous representations of steering vectors over frequency and position of microphone and source for augmented listening (e.g., spatial filtering and binaural rendering) with precise control of the sound field perceived by the user. Steering vectors have typically been used for representing the spatial characteristics of the sound field as a function of the listening position. The basic algebraic representation of steering vectors assuming an idealized environment cannot deal with the scattering effect of the sound field. One may thus collect a discrete set of real steering vectors measured in dedicated facilities and super-resolve (i.e., upsample) them. Recently, physics-aware deep learning methods have been effectively used for this purpose. Such deterministic super-resolution, however, suffers from the overfitting problem due to the non-uniform uncertainty over the measurement space. To solve this problem, we integrate an expressive representation based on the neural field (NF) into the principled probabilistic framework based on the Gaussian process (GP). Specifically, we propose a physics-aware composite kernel that model the directional incoming waves and the subsequent scattering effect. Our comprehensive comparative experiment showed the effectiveness of the proposed method under data insufficiency conditions. In downstream tasks such as speech enhancement and binaural rendering using the simulated data of the SPEAR challenge, the oracle performances were attained with less than ten times fewer measurements.
#### IS${}^3$ : Generic Impulsive--Stationary Sound Separation in Acoustic Scenes using Deep Filtering
 - **Authors:** Berger Clémentine (IDS, S2A), Stamadiatis Paraskevas (IDS, S2A), Badeau Roland (IDS, S2A), Essid Slim (IDS, S2A)
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Artificial Intelligence (cs.AI); Sound (cs.SD); Signal Processing (eess.SP)
 - **Arxiv link:** https://arxiv.org/abs/2509.02622

 - **Pdf link:** https://arxiv.org/pdf/2509.02622

 - **Abstract**
 We are interested in audio systems capable of performing a differentiated processing of stationary backgrounds and isolated acoustic events within an acoustic scene, whether for applying specific processing methods to each part or for focusing solely on one while ignoring the other. Such systems have applications in real-world scenarios, including robust adaptive audio rendering systems (e.g., EQ or compression), plosive attenuation in voice mixing, noise suppression or reduction, robust acoustic event classification or even bioacoustics. To this end, we introduce IS${}^3$, a neural network designed for Impulsive--Stationary Sound Separation, that isolates impulsive acoustic events from the stationary background using a deep filtering approach, that can act as a pre-processing stage for the above-mentioned tasks. To ensure optimal training, we propose a sophisticated data generation pipeline that curates and adapts existing datasets for this task. We demonstrate that a learning-based approach, build on a relatively lightweight neural architecture and trained with well-designed and varied data, is successful in this previously unaddressed task, outperforming the Harmonic--Percussive Sound Separation masking method, adapted from music signal processing research, and wavelet filtering on objective separation metrics.
#### Speech Intelligibility Assessment with Uncertainty-Aware Whisper Embeddings and sLSTM
 - **Authors:** Ryandhimas E. Zezario, Dyah A.M.G. Wisnu, Hsin-Min Wang, Yu Tsao
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2509.03013

 - **Pdf link:** https://arxiv.org/pdf/2509.03013

 - **Abstract**
 Non-intrusive speech intelligibility prediction remains challenging due to variability in speakers, noise conditions, and subjective perception. We propose an uncertainty-aware approach that leverages Whisper embeddings in combination with statistical features, specifically the mean, standard deviation, and entropy computed across the embedding dimensions. The entropy, computed via a softmax over the feature dimension, serves as a proxy for uncertainty, complementing global information captured by the mean and standard deviation. To model the sequential structure of speech, we adopt a scalar long short-term memory (sLSTM) network, which efficiently captures long-range dependencies. Building on this foundation, we propose iMTI-Net, an improved multi-target intelligibility prediction network that integrates convolutional neural network (CNN) and sLSTM components within a multitask learning framework. It jointly predicts human intelligibility scores and machine-based word error rates (WER) from Google ASR and Whisper. Experimental results show that iMTI-Net outperforms the original MTI-Net across multiple evaluation metrics, demonstrating the effectiveness of incorporating uncertainty-aware features and the CNN-sLSTM architecture.
#### Non-Intrusive Intelligibility Prediction for Hearing Aids: Recent Advances, Trends, and Challenges
 - **Authors:** Ryandhimas E. Zezario
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2509.03017

 - **Pdf link:** https://arxiv.org/pdf/2509.03017

 - **Abstract**
 This paper provides an overview of recent progress in non-intrusive speech intelligibility prediction for hearing aids (HA). We summarize developments in robust acoustic feature extraction, hearing loss modeling, and the use of emerging architectures for long-sequence processing. Listener-specific adaptation strategies and domain generalization approaches that aim to improve robustness in unseen acoustic environments are also discussed. Remaining challenges, such as the need for large-scale, diverse datasets and reliable cross-profile generalization, are acknowledged. Our goal is to offer a perspective on current trends, ongoing challenges, and possible future directions toward practical and reliable HA-oriented intelligibility prediction systems.
#### A Study on Zero-Shot Non-Intrusive Speech Intelligibility for Hearing Aids Using Large Language Models
 - **Authors:** Ryandhimas E. Zezario, Dyah A.M.G. Wisnu, Hsin-Min Wang, Yu Tsao
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2509.03021

 - **Pdf link:** https://arxiv.org/pdf/2509.03021

 - **Abstract**
 This work focuses on zero-shot non-intrusive speech assessment for hearing aids (HA) using large language models (LLMs). Specifically, we introduce GPT-Whisper-HA, an extension of GPT-Whisper, a zero-shot non-intrusive speech assessment model based on LLMs. GPT-Whisper-HA is designed for speech assessment for HA, incorporating MSBG hearing loss and NAL-R simulations to process audio input based on each individual's audiogram, two automatic speech recognition (ASR) modules for audio-to-text representation, and GPT-4o to predict two corresponding scores, followed by score averaging for the final estimated score. Experimental results indicate that GPT-Whisper-HA achieves a 2.59% relative root mean square error (RMSE) improvement over GPT-Whisper, confirming the potential of LLMs for zero-shot speech assessment in predicting subjective intelligibility for HA users.
#### Improving Perceptual Audio Aesthetic Assessment via Triplet Loss and Self-Supervised Embeddings
 - **Authors:** Dyah A. M. G. Wisnu, Ryandhimas E. Zezario, Stefano Rini, Hsin-Min Wang, Yu Tsao
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Machine Learning (cs.LG); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2509.03292

 - **Pdf link:** https://arxiv.org/pdf/2509.03292

 - **Abstract**
 We present a system for automatic multi-axis perceptual quality prediction of generative audio, developed for Track 2 of the AudioMOS Challenge 2025. The task is to predict four Audio Aesthetic Scores--Production Quality, Production Complexity, Content Enjoyment, and Content Usefulness--for audio generated by text-to-speech (TTS), text-to-audio (TTA), and text-to-music (TTM) systems. A main challenge is the domain shift between natural training data and synthetic evaluation data. To address this, we combine BEATs, a pretrained transformer-based audio representation model, with a multi-branch long short-term memory (LSTM) predictor and use a triplet loss with buffer-based sampling to structure the embedding space by perceptual similarity. Our results show that this improves embedding discriminability and generalization, enabling domain-robust audio quality assessment without synthetic training data.
#### An Effective Strategy for Modeling Score Ordinality and Non-uniform Intervals in Automated Speaking Assessment
 - **Authors:** Tien-Hong Lo, Szu-Yu Chen, Yao-Ting Sung, Berlin Chen
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Machine Learning (cs.LG); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2509.03372

 - **Pdf link:** https://arxiv.org/pdf/2509.03372

 - **Abstract**
 A recent line of research on automated speaking assessment (ASA) has benefited from self-supervised learning (SSL) representations, which capture rich acoustic and linguistic patterns in non-native speech without underlying assumptions of feature curation. However, speech-based SSL models capture acoustic-related traits but overlook linguistic content, while text-based SSL models rely on ASR output and fail to encode prosodic nuances. Moreover, most prior arts treat proficiency levels as nominal classes, ignoring their ordinal structure and non-uniform intervals between proficiency labels. To address these limitations, we propose an effective ASA approach combining SSL with handcrafted indicator features via a novel modeling paradigm. We further introduce a multi-margin ordinal loss that jointly models both the score ordinality and non-uniform intervals of proficiency labels. Extensive experiments on the TEEMI corpus show that our method consistently outperforms strong baselines and generalizes well to unseen prompts.
#### SSVD: Structured SVD for Parameter-Efficient Fine-Tuning and Benchmarking under Domain Shift in ASR
 - **Authors:** Pu Wang, Shinji Watanabe, Hugo Van hamme
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2509.02830

 - **Pdf link:** https://arxiv.org/pdf/2509.02830

 - **Abstract**
 Parameter-efficient fine-tuning (PEFT) has emerged as a scalable solution for adapting large foundation models. While low-rank adaptation (LoRA) is widely used in speech applications, its state-of-the-art variants, e.g., VeRA, DoRA, PiSSA, and SVFT, are developed mainly for language and vision tasks, with limited validation in speech. This work presents the first comprehensive integration and benchmarking of these PEFT methods within ESPnet. We further introduce structured SVD-guided (SSVD) fine-tuning, which selectively rotates input-associated right singular vectors while keeping output-associated vectors fixed to preserve semantic mappings. This design enables robust domain adaptation with minimal trainable parameters and improved efficiency. We evaluate all methods on domain-shifted speech recognition tasks, including child speech and dialectal variation, across model scales from 0.1B to 2B. All implementations are released in ESPnet to support reproducibility and future work.
#### Speech DF Arena: A Leaderboard for Speech DeepFake Detection Models
 - **Authors:** Sandipana Dowerah, Atharva Kulkarni, Ajinkya Kulkarni, Hoan My Tran, Joonas Kalda, Artem Fedorchenko, Benoit Fauve, Damien Lolive, Tanel Alumäe, Matthew Magimai Doss
 - **Subjects:** Subjects:
Sound (cs.SD); Computation and Language (cs.CL); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2509.02859

 - **Pdf link:** https://arxiv.org/pdf/2509.02859

 - **Abstract**
 Parallel to the development of advanced deepfake audio generation, audio deepfake detection has also seen significant progress. However, a standardized and comprehensive benchmark is still missing. To address this, we introduce Speech DeepFake (DF) Arena, the first comprehensive benchmark for audio deepfake detection. Speech DF Arena provides a toolkit to uniformly evaluate detection systems, currently across 14 diverse datasets and attack scenarios, standardized evaluation metrics and protocols for reproducibility and transparency. It also includes a leaderboard to compare and rank the systems to help researchers and developers enhance their reliability and robustness. We include 14 evaluation sets, 12 state-of-the-art open-source and 3 proprietary detection systems. Our study presents many systems exhibiting high EER in out-of-domain scenarios, highlighting the need for extensive cross-domain evaluation. The leaderboard is hosted on Huggingface1 and a toolkit for reproducing results across the listed datasets is available on GitHub.
#### Mitigating Data Imbalance in Automated Speaking Assessment
 - **Authors:** Fong-Chun Tsai, Kuan-Tang Huang, Bi-Cheng Yan, Tien-Hong Lo, Berlin Chen
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Machine Learning (cs.LG); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2509.03010

 - **Pdf link:** https://arxiv.org/pdf/2509.03010

 - **Abstract**
 Automated Speaking Assessment (ASA) plays a crucial role in evaluating second-language (L2) learners proficiency. However, ASA models often suffer from class imbalance, leading to biased predictions. To address this, we introduce a novel objective for training ASA models, dubbed the Balancing Logit Variation (BLV) loss, which perturbs model predictions to improve feature representation for minority classes without modifying the dataset. Evaluations on the ICNALE benchmark dataset show that integrating the BLV loss into a celebrated text-based (BERT) model significantly enhances classification accuracy and fairness, making automated speech evaluation more robust for diverse learners.
#### Comparison of End-to-end Speech Assessment Models for the NOCASA 2025 Challenge
 - **Authors:** Aleksei Žavoronkov, Tanel Alumäe
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2509.03256

 - **Pdf link:** https://arxiv.org/pdf/2509.03256

 - **Abstract**
 This paper presents an analysis of three end-to-end models developed for the NOCASA 2025 Challenge, aimed at automatic word-level pronunciation assessment for children learning Norwegian as a second language. Our models include an encoder-decoder Siamese architecture (E2E-R), a prefix-tuned direct classification model leveraging pretrained wav2vec2.0 representations, and a novel model integrating alignment-free goodness-of-pronunciation (GOP) features computed via CTC. We introduce a weighted ordinal cross-entropy loss tailored for optimizing metrics such as unweighted average recall and mean absolute error. Among the explored methods, our GOP-CTC-based model achieved the highest performance, substantially surpassing challenge baselines and attaining top leaderboard scores.


by Zyzzyva0381 (Windy). 


2025-09-04
