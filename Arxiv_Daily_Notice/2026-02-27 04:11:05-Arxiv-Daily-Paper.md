# Showing new listings for Friday, 27 February 2026
Auto update papers at about 2:30am UTC (10:30am Beijing time) every weekday.


阅读 `Usage.md`了解如何使用此repo实现个性化的Arxiv论文推送

See `Usage.md` for instructions on how to personalize the repo. 


Keyword list: ['text-to-speech', 'text to speech', 'tts', 'LLM-based', 'speech', 'voice']


Excluded: []


### Today: 7papers 
#### Moving Speaker Separation via Parallel Spectral-Spatial Processing
 - **Authors:** Yuzhu Wang, Archontis Politis, Konstantinos Drossos, Tuomas Virtanen
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2602.22487

 - **Pdf link:** https://arxiv.org/pdf/2602.22487

 - **Abstract**
 Multi-channel speech separation in dynamic environments is challenging as time-varying spatial and spectral features evolve at different temporal scales. Existing methods typically employ sequential architectures, forcing a single network stream to simultaneously model both feature types, creating an inherent modeling conflict. In this paper, we propose a dual-branch parallel spectral-spatial (PS2) architecture that separately processes spectral and spatial features through parallel streams. The spectral branch uses a bi-directional long short-term memory (BLSTM)-based frequency module, a Mamba-based temporal module, and a self-attention module to model spectral features. The spatial branch employs bi-directional gated recurrent unit (BGRU) networks to process spatial features that encode the evolving geometric relationships between sources and microphones. Features from both branches are integrated through a cross-attention fusion mechanism that adaptively weights their contributions. Experimental results demonstrate that the PS2 outperforms existing state-of-the-art (SOTA) methods by 1.6-2.2 dB in scale-invariant signal-to-distortion ratio (SI-SDR) for moving speaker scenarios, with robust separation quality under different reverberation times (RT60), noise levels, and source movement speeds. Even with fast source movements, the proposed model maintains SI-SDR improvements of over 13 dB. These improvements are consistently observed across multiple datasets, including WHAMR! and our generated WSJ0-Demand-6ch-Move dataset.
#### Deepfake Word Detection by Next-token Prediction using Fine-tuned Whisper
 - **Authors:** Hoan My Tran, Xin Wang, Wanying Ge, Xuechen Liu, Junichi Yamagishi
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Computation and Language (cs.CL)
 - **Arxiv link:** https://arxiv.org/abs/2602.22658

 - **Pdf link:** https://arxiv.org/pdf/2602.22658

 - **Abstract**
 Deepfake speech utterances can be forged by replacing one or more words in a bona fide utterance with semantically different words synthesized by speech generative models. While a dedicated synthetic word detector could be developed, we investigate a cost-effective method that fine-tunes a pre-trained Whisper model to detect synthetic words while transcribing the input utterance via next-token prediction. We further investigate using partially vocoded utterances as the fine-tuning data, thereby reducing the cost of data collection. Our experiments demonstrate that, on in-domain test data, the fine-tuned Whisper yields low synthetic-word detection error rates and transcription error rates. On out-of-domain test data with synthetic words produced by unseen speech generative models, the fine-tuned Whisper remains on par with a dedicated ResNet-based detection model; however, the overall performance degradation calls for strategies to improve its generalization capability.
#### Absorbing Discrete Diffusion for Speech Enhancement
 - **Authors:** Philippe Gonzalez
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2602.22417

 - **Pdf link:** https://arxiv.org/pdf/2602.22417

 - **Abstract**
 Inspired by recent developments in neural speech coding and diffusion-based language modeling, we tackle speech enhancement by modeling the conditional distribution of clean speech codes given noisy speech codes using absorbing discrete diffusion. The proposed approach, which we call ADDSE, leverages both the expressive latent space of neural audio codecs and the non-autoregressive sampling procedure of diffusion models. To efficiently model the hierarchical structure of residual vector quantization codes, we propose RQDiT, which combines techniques from RQ-Transformer and diffusion Transformers for non-autoregressive modeling. Results show competitive performance in terms of non-intrusive objective metrics on two datasets, especially at low signal-to-noise ratios and with few sampling steps. Code and audio examples are available online.
#### Efficient Dialect-Aware Modeling and Conditioning for Low-Resource Taiwanese Hakka Speech Processing
 - **Authors:** An-Ci Peng, Kuan-Tang Huang, Tien-Hong Lo, Hung-Shin Lee, Hsin-Min Wang, Berlin Chen
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Artificial Intelligence (cs.AI); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2602.22522

 - **Pdf link:** https://arxiv.org/pdf/2602.22522

 - **Abstract**
 Taiwanese Hakka is a low-resource, endangered language that poses significant challenges for automatic speech recognition (ASR), including high dialectal variability and the presence of two distinct writing systems (Hanzi and Pinyin). Traditional ASR models often encounter difficulties in this context, as they tend to conflate essential linguistic content with dialect-specific variations across both phonological and lexical dimensions. To address these challenges, we propose a unified framework grounded in the Recurrent Neural Network Transducers (RNN-T). Central to our approach is the introduction of dialect-aware modeling strategies designed to disentangle dialectal "style" from linguistic "content", which enhances the model's capacity to learn robust and generalized representations. Additionally, the framework employs parameter-efficient prediction networks to concurrently model ASR (Hanzi and Pinyin). We demonstrate that these tasks create a powerful synergy, wherein the cross-script objective serves as a mutual regularizer to improve the primary ASR tasks. Experiments conducted on the HAT corpus reveal that our model achieves 57.00% and 40.41% relative error rate reduction on Hanzi and Pinyin ASR, respectively. To our knowledge, this is the first systematic investigation into the impact of Hakka dialectal variations on ASR and the first single model capable of jointly addressing these tasks.
#### Relating the Neural Representations of Vocalized, Mimed, and Imagined Speech
 - **Authors:** Maryam Maghsoudi, Rupesh Chillale, Shihab A. Shamma
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS); Signal Processing (eess.SP)
 - **Arxiv link:** https://arxiv.org/abs/2602.22597

 - **Pdf link:** https://arxiv.org/pdf/2602.22597

 - **Abstract**
 We investigated the relationship among neural representations of vocalized, mimed, and imagined speech recorded using publicly available stereotactic EEG recordings. Most prior studies have focused on decoding speech responses within each condition separately. Here, instead, we explore how responses across conditions relate by training linear spectrogram reconstruction models for each condition and evaluate their generalization across conditions. We demonstrate that linear decoders trained on one condition generally transfer successfully to others, implying shared speech representations. This commonality was assessed with stimulus-level discriminability by performing a rank-based analysis demonstrating preservation of stimulus-specific structure in both within- and across-conditions. Finally, we compared linear reconstructions to those from a nonlinear neural network. While both exhibited cross-condition transfer, linear models achieve superior stimulus-level discriminability.
#### Make It Hard to Hear, Easy to Learn: Long-Form Bengali ASR and Speaker Diarization via Extreme Augmentation and Perfect Alignment
 - **Authors:** Sanjid Hasan, Risalat Labib, A H M Fuad, Bayazid Hasan
 - **Subjects:** Subjects:
Sound (cs.SD); Artificial Intelligence (cs.AI); Computation and Language (cs.CL); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2602.23070

 - **Pdf link:** https://arxiv.org/pdf/2602.23070

 - **Abstract**
 Although Automatic Speech Recognition (ASR) in Bengali has seen significant progress, processing long-duration audio and performing robust speaker diarization remain critical research gaps. To address the severe scarcity of joint ASR and diarization resources for this language, we introduce Lipi-Ghor-882, a comprehensive 882-hour multi-speaker Bengali dataset. In this paper, detailing our submission to the DL Sprint 4.0 competition, we systematically evaluate various architectures and approaches for long-form Bengali speech. For ASR, we demonstrate that raw data scaling is ineffective; instead, targeted fine-tuning utilizing perfectly aligned annotations paired with synthetic acoustic degradation (noise and reverberation) emerges as the singular most effective approach. Conversely, for speaker diarization, we observed that global open-source state-of-the-art models (such as Diarizen) performed surprisingly poorly on this complex dataset. Extensive model retraining yielded negligible improvements; instead, strategic, heuristic post-processing of baseline model outputs proved to be the primary driver for increasing accuracy. Ultimately, this work outlines a highly optimized dual pipeline achieving a $\sim$0.019 Real-Time Factor (RTF), establishing a practical, empirically backed benchmark for low-resource, long-form speech processing.
#### A Mixture-of-Experts Model for Multimodal Emotion Recognition in Conversations
 - **Authors:** Soumya Dutta, Smruthi Balaji, Sriram Ganapathy
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2602.23300

 - **Pdf link:** https://arxiv.org/pdf/2602.23300

 - **Abstract**
 Emotion Recognition in Conversations (ERC) presents unique challenges, requiring models to capture the temporal flow of multi-turn dialogues and to effectively integrate cues from multiple modalities. We propose Mixture of Speech-Text Experts for Recognition of Emotions (MiSTER-E), a modular Mixture-of-Experts (MoE) framework designed to decouple two core challenges in ERC: modality-specific context modeling and multimodal information fusion. MiSTER-E leverages large language models (LLMs) fine-tuned for both speech and text to provide rich utterance-level embeddings, which are then enhanced through a convolutional-recurrent context modeling layer. The system integrates predictions from three experts-speech-only, text-only, and cross-modal-using a learned gating mechanism that dynamically weighs their outputs. To further encourage consistency and alignment across modalities, we introduce a supervised contrastive loss between paired speech-text representations and a KL-divergence-based regulariza-tion across expert predictions. Importantly, MiSTER-E does not rely on speaker identity at any stage. Experiments on three benchmark datasets-IEMOCAP, MELD, and MOSI-show that our proposal achieves 70.9%, 69.5%, and 87.9% weighted F1-scores respectively, outperforming several baseline speech-text ERC systems. We also provide various ablations to highlight the contributions made in the proposed approach.


by Zyzzyva0381 (Windy). 


2026-02-27
