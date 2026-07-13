# Showing new listings for Monday, 13 July 2026
Auto update papers at about 2:30am UTC (10:30am Beijing time) every weekday.


阅读 `Usage.md`了解如何使用此repo实现个性化的Arxiv论文推送

See `Usage.md` for instructions on how to personalize the repo. 


Keyword list: ['text-to-speech', 'text to speech', 'tts', 'LLM-based', 'speech', 'voice']


Excluded: []


### Today: 4papers 
#### Phone Segmentation and Recognition through Phonological Activation Mapping
 - **Authors:** Shikhar Bharadwaj, Kwanghee Choi, Stephen McIntosh, Chin-Jou Li, Eunjung Yeo, Daisuke Saito, Nobuaki Minematsu, Shinji Watanabe, Jian Zhu, David Harwath, David R. Mortensen
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Artificial Intelligence (cs.AI); Computation and Language (cs.CL); Machine Learning (cs.LG); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2607.09020

 - **Pdf link:** https://arxiv.org/pdf/2607.09020

 - **Abstract**
 Phone segmentation and recognition are inherently related tasks, yet modern approaches typically model them separately. We argue that phonetic structure is already latent in the representations of self-supervised speech models (S3Ms), and one only needs to steer them to solve both tasks. We leverage S3M-based Phonological Activation Mapping (SPAM), which maps each S3M representation frame to a vector of phonological feature activations, such as voicing and nasality. On top of SPAM, we introduce two simple but effective lightweight, gradient-descent-free prediction heads: a recognition head and a segmentation head. Our method requires less than a minute of phonetic transcriptions, and generalizes to unseen phones during training. Across a diverse range of datasets, our approach attains strong segmentation and recognition performance.
#### Technical Report for MERL's Real-TSE Challenge Submission
 - **Authors:** Dominik Klement, Yoshiki Masuyama, Christoph Boeddeker, Kohei Saijo, Julius Richter, Gordon Wichern, Jonathan Le Roux
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2607.09043

 - **Pdf link:** https://arxiv.org/pdf/2607.09043

 - **Abstract**
 Target speech extraction (TSE) has largely been dominated by neural network-based approaches trained and evaluated on synthetic fully overlapped data. The Real-TSE Challenge aims to advance performance on real-world far-field noisy and reverberant recordings. This technical report describes MERL's submission to the Real-TSE Challenge. Rather than proposing a novel model architecture, we built upon the baseline model and focused primarily on data preparation and cleaning. Our system was trained in four stages, beginning with pre-training on fully overlapped mixtures and simulated multi-talker conversations with noise and reverberation applied to both the mixture and the enrollment utterances. We then adapted the model to real-world conditions using noisy far-field recordings with pseudo-targets derived from processed close-talk microphone signals. Our submission achieved first place in the second track, demonstrating the critical importance of high-quality data preparation. Furthermore, we observed that DNSMOS and speaker similarity are susceptible to over-optimization, motivating an investigation of their robustness using adversarial attacks. The results show that both metrics can be driven to extreme values without degrading the token error rate or the VAD-based F1 score.
#### Optimal Transport-based Semantic Alignment for LLM-based Audio-Visual Speech Recognition
 - **Authors:** Xugang Lu, Peng Shen, Yu Tsao, Hisashi Kawai
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2607.09001

 - **Pdf link:** https://arxiv.org/pdf/2607.09001

 - **Abstract**
 Large language model (LLM)-based audio-visual speech recognition (LLM-AVSR) has recently demonstrated strong robustness in adverse acoustic environments by leveraging complementary audio and visual information. Existing approaches typically employ independently pretrained acoustic and visual encoders, whose outputs are projected and fused as soft prompts to condition an LLM for speech recognition. However, most methods perform multimodal fusion without explicitly addressing the representational discrepancy between audio, visual and text modalities, potentially limiting the effectiveness of cross-modal integration. In this paper, we propose an optimal transport (OT)-based semantic alignment framework for LLM-AVSR. The proposed method explicitly bridges the modality gap by aligning the acoustic and visual representations with reference to the linguistic embedding space of the LLM before multimodal fusion. Specifically, OT is used to estimate probabilistic coupling matrices that characterize structured correspondences between modality-specific features and linguistic embeddings. The resulting OT couplings are further utilized as soft pseudo-labels to supervise contrastive learning, encouraging the extraction of semantically coherent and cross-modal consistent audio-visual representations. By anchoring multimodal features to the linguistic space of the LLM, the proposed framework facilitates more effective multimodal fusion and decoding. We implement the proposed framework using a Whisper-based acoustic encoder, an AV-HuBERT-based visual encoder, and a LLaMA3.2-3B decoder. Experiments conducted on the LRS3-TED benchmark demonstrate consistent improvements over strong baselines and achieve state-of-the-art performance under both clean and noisy evaluation conditions across a wide range of signal-to-noise ratios (SNRs).
#### ReGen: Hierarchical Multi-Prompt Representation Generation for Efficient Waveform Diffusion Models
 - **Authors:** Sang-Hoon Lee, Ha-Yeong Choi
 - **Subjects:** Subjects:
Sound (cs.SD); Artificial Intelligence (cs.AI); Audio and Speech Processing (eess.AS); Signal Processing (eess.SP)
 - **Arxiv link:** https://arxiv.org/abs/2607.09134

 - **Pdf link:** https://arxiv.org/pdf/2607.09134

 - **Abstract**
 Representation alignment (REPA) has been investigated to accelerate diffusion training, but we observe that regularizing intermediate representations in diffusion Transformers (DiT) may implicitly entangle latents and limit generative capacity. To address this issue, we propose ReGen, a hierarchical multi-prompt representation generation framework that jointly estimates multiple vector fields for both representations and data within a single diffusion model. We further introduce generalized flow matching (GFM) to improve the generalization of conditional flow matching (CFM). We validate ReGen on single-stage waveform diffusion models including neural audio codec and Wave-VAE. ReGen significantly improves waveform generation quality from highly compressed latent representations at 12.5 Hz. We also present ReGenVoice, a latent diffusion model (LDM)-based text-to-speech model that achieves strong speech intelligibility (WER) and speaker similarity (SIM) with a small dataset. Moreover, operating the LDM at 6.25 Hz with rich semantic and acoustic latent representation enables efficient training and sampling, requiring only 1 day of training on 4 GPUs and fast inference with an RTF of 0.08. Audio samples are available at this https URL.


by Zyzzyva0381 (Windy). 


2026-07-13
