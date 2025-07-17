# Showing new listings for Thursday, 17 July 2025
Auto update papers at about 2:30am UTC (10:30am Beijing time) every weekday.


阅读 `Usage.md`了解如何使用此repo实现个性化的Arxiv论文推送

See `Usage.md` for instructions on how to personalize the repo. 


Keyword list: ['text-to-speech', 'text to speech', 'tts', 'LLM-based', 'speech', 'voice']


Excluded: []


### Today: 7papers 
#### JSQA: Speech Quality Assessment with Perceptually-Inspired Contrastive Pretraining Based on JND Audio Pairs
 - **Authors:** Junyi Fan, Donald Williamson
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Artificial Intelligence (cs.AI); Machine Learning (cs.LG)
 - **Arxiv link:** https://arxiv.org/abs/2507.11636

 - **Pdf link:** https://arxiv.org/pdf/2507.11636

 - **Abstract**
 Speech quality assessment (SQA) is often used to learn a mapping from a high-dimensional input space to a scalar that represents the mean opinion score (MOS) of the perceptual speech quality. Learning such a mapping is challenging for many reasons, but largely because MOS exhibits high levels of inherent variance due to perceptual and experimental-design differences. Many solutions have been proposed, but many approaches do not properly incorporate perceptual factors into their learning algorithms (beyond the MOS label), which could lead to unsatisfactory results. To this end, we propose JSQA, a two-stage framework that pretrains an audio encoder using perceptually-guided contrastive learning on just noticeable difference (JND) pairs, followed by fine-tuning for MOS prediction. We first generate pairs of audio data within JND levels, which are then used to pretrain an encoder to leverage perceptual quality similarity information and map it into an embedding space. The JND pairs come from clean LibriSpeech utterances that are mixed with background noise from CHiME-3, at different signal-to-noise ratios (SNRs). The encoder is later fine-tuned with audio samples from the NISQA dataset for MOS prediction. Experimental results suggest that perceptually-inspired contrastive pretraining significantly improves the model performance evaluated by various metrics when compared against the same network trained from scratch without pretraining. These findings suggest that incorporating perceptual factors into pretraining greatly contributes to the improvement in performance for SQA.
#### VoxATtack: A Multimodal Attack on Voice Anonymization Systems
 - **Authors:** Ahmad Aloradi, Ünal Ege Gaznepoglu, Emanuël A. P. Habets, Daniel Tenbrinck
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2507.12081

 - **Pdf link:** https://arxiv.org/pdf/2507.12081

 - **Abstract**
 Voice anonymization systems aim to protect speaker privacy by obscuring vocal traits while preserving the linguistic content relevant for downstream applications. However, because these linguistic cues remain intact, they can be exploited to identify semantic speech patterns associated with specific speakers. In this work, we present VoxATtack, a novel multimodal de-anonymization model that incorporates both acoustic and textual information to attack anonymization systems. While previous research has focused on refining speaker representations extracted from speech, we show that incorporating textual information with a standard ECAPA-TDNN improves the attacker's performance. Our proposed VoxATtack model employs a dual-branch architecture, with an ECAPA-TDNN processing anonymized speech and a pretrained BERT encoding the transcriptions. Both outputs are projected into embeddings of equal dimensionality and then fused based on confidence weights computed on a per-utterance basis. When evaluating our approach on the VoicePrivacy Attacker Challenge (VPAC) dataset, it outperforms the top-ranking attackers on five out of seven benchmarks, namely B3, B4, B5, T8-5, and T12-5. To further boost performance, we leverage anonymized speech and SpecAugment as augmentation techniques. This enhancement enables VoxATtack to achieve state-of-the-art on all VPAC benchmarks, after scoring 20.6% and 27.2% average equal error rate on T10-2 and T25-1, respectively. Our results demonstrate that incorporating textual information and selective data augmentation reveals critical vulnerabilities in current voice anonymization methods and exposes potential weaknesses in the datasets used to evaluate them.
#### Soft-Constrained Spatially Selective Active Noise Control for Open-fitting Hearables
 - **Authors:** Tong Xiao, Reinhild Roden, Matthias Blau, Simon Doclo
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Signal Processing (eess.SP); Systems and Control (eess.SY)
 - **Arxiv link:** https://arxiv.org/abs/2507.12122

 - **Pdf link:** https://arxiv.org/pdf/2507.12122

 - **Abstract**
 Recent advances in spatially selective active noise control (SSANC) using multiple microphones have enabled hearables to suppress undesired noise while preserving desired speech from a specific direction. Aiming to achieve minimal speech distortion, a hard constraint has been used in previous work in the optimization problem to compute the control filter. In this work, we propose a soft-constrained SSANC system that uses a frequency-independent parameter to trade off between speech distortion and noise reduction. We derive both time- and frequency-domain formulations, and show that conventional active noise control and hard-constrained SSANC represent two limiting cases of the proposed design. We evaluate the system through simulations using a pair of open-fitting hearables in an anechoic environment with one speech source and two noise sources. The simulation results validate the theoretical derivations and demonstrate that for a broad range of the trade-off parameter, the signal-to-noise ratio and the speech quality and intelligibility in terms of PESQ and ESTOI can be substantially improved compared to the hard-constrained design.
#### Towards Scalable AASIST: Refining Graph Attention for Speech Deepfake Detection
 - **Authors:** Ivan Viakhirev, Daniil Sirota, Aleksandr Smirnov, Kirill Borodin
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2507.11777

 - **Pdf link:** https://arxiv.org/pdf/2507.11777

 - **Abstract**
 Advances in voice conversion and text-to-speech synthesis have made automatic speaker verification (ASV) systems more susceptible to spoofing attacks. This work explores modest refinements to the AASIST anti-spoofing architecture. It incorporates a frozen Wav2Vec 2.0 encoder to retain self-supervised speech representations in limited-data settings, substitutes the original graph attention block with a standardized multi-head attention module using heterogeneous query projections, and replaces heuristic frame-segment fusion with a trainable, context-aware integration layer. When evaluated on the ASVspoof 5 corpus, the proposed system reaches a 7.6\% equal error rate (EER), improving on a re-implemented AASIST baseline under the same training conditions. Ablation experiments suggest that each architectural change contributes to the overall performance, indicating that targeted adjustments to established models may help strengthen speech deepfake detection in practical scenarios. The code is publicly available at this https URL.
#### Schrödinger Bridge Consistency Trajectory Models for Speech Enhancement
 - **Authors:** Shuichiro Nishigori, Koichi Saito, Naoki Murata, Masato Hirano, Shusuke Takahashi, Yuki Mitsufuji
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2507.11925

 - **Pdf link:** https://arxiv.org/pdf/2507.11925

 - **Abstract**
 Speech enhancement (SE) utilizing diffusion models is a promising technology that improves speech quality in noisy speech data. Furthermore, the Schrödinger bridge (SB) has recently been used in diffusion-based SE to improve speech quality by resolving a mismatch between the endpoint of the forward process and the starting point of the reverse process. However, the SB still exhibits slow inference owing to the necessity of a large number of function evaluations (NFE) for inference to obtain high-quality results. While Consistency Models (CMs) address this issue by employing consistency training that uses distillation from pretrained models in the field of image generation, it does not improve generation quality when the number of steps increases. As a solution to this problem, Consistency Trajectory Models (CTMs) not only accelerate inference speed but also maintain a favorable trade-off between quality and speed. Furthermore, SoundCTM demonstrates the applicability of CTM techniques to the field of sound generation. In this paper, we present Schrödinger bridge Consistency Trajectory Models (SBCTM) by applying the CTM's technique to the Schrödinger bridge for SE. Additionally, we introduce a novel auxiliary loss, including a perceptual loss, into the original CTM's training framework. As a result, SBCTM achieves an approximately 16x improvement in the real-time factor (RTF) compared to the conventional Schrödinger bridge for SE. Furthermore, the favorable trade-off between quality and speed in SBCTM allows for time-efficient inference by limiting multi-step refinement to cases where 1-step inference is insufficient. Our code, pretrained models, and audio samples are available at this https URL.
#### MambaRate: Speech Quality Assessment Across Different Sampling Rates
 - **Authors:** Panos Kakoulidis, Iakovi Alexiou, Junkwang Oh, Gunu Jho, Inchul Hwang, Pirros Tsiakoulis, Aimilios Chalamandaris
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2507.12090

 - **Pdf link:** https://arxiv.org/pdf/2507.12090

 - **Abstract**
 We propose MambaRate, which predicts Mean Opinion Scores (MOS) with limited bias regarding the sampling rate of the waveform under evaluation. It is designed for Track 3 of the AudioMOS Challenge 2025, which focuses on predicting MOS for speech in high sampling frequencies. Our model leverages self-supervised embeddings and selective state space modeling. The target ratings are encoded in a continuous representation via Gaussian radial basis functions (RBF). The results of the challenge were based on the system-level Spearman's Rank Correllation Coefficient (SRCC) metric. An initial MambaRate version (T16 system) outperformed the pre-trained baseline (B03) by ~14% in a few-shot setting without pre-training. T16 ranked fourth out of five in the challenge, differing by ~6% from the winning system. We present additional results on the BVCC dataset as well as ablations with different representations as input, which outperform the initial T16 version.
#### Towards few-shot isolated word reading assessment
 - **Authors:** Reuben Smit, Retief Louw, Herman Kamper
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2507.12217

 - **Pdf link:** https://arxiv.org/pdf/2507.12217

 - **Abstract**
 We explore an ASR-free method for isolated word reading assessment in low-resource settings. Our few-shot approach compares input child speech to a small set of adult-provided reference templates. Inputs and templates are encoded using intermediate layers from large self-supervised learned (SSL) models. Using an Afrikaans child speech benchmark, we investigate design options such as discretising SSL features and barycentre averaging of the templates. Idealised experiments show reasonable performance for adults, but a substantial drop for child speech input, even with child templates. Despite the success of employing SSL representations in low-resource speech tasks, our work highlights the limitations of SSL representations for processing child data when used in a few-shot classification system.


by Zyzzyva0381 (Windy). 


2025-07-17
