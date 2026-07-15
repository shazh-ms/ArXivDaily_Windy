# Showing new listings for Wednesday, 15 July 2026
Auto update papers at about 2:30am UTC (10:30am Beijing time) every weekday.


阅读 `Usage.md`了解如何使用此repo实现个性化的Arxiv论文推送

See `Usage.md` for instructions on how to personalize the repo. 


Keyword list: ['text-to-speech', 'text to speech', 'tts', 'LLM-based', 'speech', 'voice']


Excluded: []


### Today: 6papers 
#### The Sound of Absence: Audio-Language Embedding Models Struggle with Negation
 - **Authors:** Chun-Yi Kuan, Hung-yi Lee
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Artificial Intelligence (cs.AI); Computation and Language (cs.CL); Machine Learning (cs.LG); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2607.12290

 - **Pdf link:** https://arxiv.org/pdf/2607.12290

 - **Abstract**
 Audio-language embedding models such as CLAP are widely evaluated on matching present sound events, but rarely on negation. We show this affirmation-only evaluation hides a key limitation: these models fail to encode negated sound concepts, mapping affirmative and negated captions to nearly identical representations. To expose this blind spot, we introduce NegEval-Audio, a framework that converts existing datasets into two negation-aware tasks, Retrieval-Neg and Multiple-Choice Negation (MCQ-Neg), to probe whether models distinguish present from absent events. On AudioCaps and Clotho, performance degrades sharply under negation, with negation-type MCQ accuracy falling far below chance, and the failure persists even for a recent multimodal LLM-based embedding model. While a training-free steering method improves MCQ-Neg, it yields marginal gains for Retrieval-Neg. This indicates that affirmation bias is a fundamental flaw in the representation geometry, necessitating explicit negation-aware training objectives.
#### ZipL-Dialog: Memory-Efficient Long-Form Spoken Dialog Synthesis via Latent Flow Matching
 - **Authors:** Jihwan Kim, Nam Soo Kim
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2607.12496

 - **Pdf link:** https://arxiv.org/pdf/2607.12496

 - **Abstract**
 Zero-shot dialog TTS benefits from flow-matching, but minute-scale generation on dense mel-spectrograms causes severe memory bottlenecks, often forcing unnatural chunked synthesis. We propose ZipL-Dialog, which shifts conditional flow-matching into a 4x time-compressed (25 Hz) latent space. To preserve acoustic fidelity under compression, we employ a deterministic mel autoencoder with auxiliary mel-domain supervision and optimize the ZipFormer's hierarchical downsampling schedule. Experiments show that ZipL-Dialog reduces maximum peak GPU memory by 11.22x and accelerates inference by 2.23x over the baseline, substantially lowering the memory footprint of single-pass multi-minute dialog synthesis while maintaining perceptual naturalness.
#### Listen first: Output-based multi-microphone speech enhancement
 - **Authors:** Panos Apostolidis, Svend Feldt, Zheng-Hua Tan, Jan Østergaard, Jesper Jensen
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2607.12529

 - **Pdf link:** https://arxiv.org/pdf/2607.12529

 - **Abstract**
 Traditionally, hearing-aid speech enhancement (SE) algorithms rely on input-based feature estimation, often derived by a voice activity detector (VAD), to configure beamformers. Yet features extracted from noisy microphone signals can become unreliable in challenging acoustic scenes where users most need help. We introduce a novel paradigm in which the settings of a sound processing system are determined by evaluating characteristics of its output. To demonstrate this idea, we employ an output-based system that selects among a set of minimum power distortionless response (MPDR) beamformers. Although MPDR beamformers are typically avoided due to their sensitivity to steering errors, we show that they become effective within an output-based framework. We compare the proposed system to a conventional input-based minimum variance distortionless response (MVDR) baseline. Experimental results show that the proposed system consistently outperforms the MVDR baseline, particularly at low SNRs, in terms of SNR, ESTOI and PESQ.
#### Investigating the Integration of Spatial Information in Foundation-Model-Based Speaker Diarization
 - **Authors:** Marc Deegen, Adrian Meise, Reinhold Haeb-Umbach
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2607.12647

 - **Pdf link:** https://arxiv.org/pdf/2607.12647

 - **Abstract**
 Spatial information gleaned from multi-channel input has been shown to lead to improvements in meeting processing tasks like diarization and source separation. At the same time, diarization based on features extracted by large pretrained single-channel foundation models, such as WavLM, achieved state-of-the-art performance. This work compares three approaches to integrate spatial features into foundation model-based diarization systems: the cascade of a beamformer and a single-channel foundation model, a multi-channel foundation model, and the conditioning of the downstream network on explicitly extracted spatial features. Results show that the beamformer front-end is even detrimental to diarization performance in regions of overlapped speech, while best performance is achieved with the conditioning, demonstrating that the incorporation of explicit spatial features is a competitive approach to foundation-model-supported diarization. This approach is further subjected to a detailed error analysis showing that the conditioning system removes errors to a good extent that would occur when either only spectral or only spatial features were used.
#### Audio Diarization: A New Paradigm for Exploring Audio Recordings with Unknown Event Classes
 - **Authors:** Alexander Werning, Reinhold Haeb-Umbach
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2607.12703

 - **Pdf link:** https://arxiv.org/pdf/2607.12703

 - **Abstract**
 We propose a new task, audio diarization. The motivation is that there are applications, such as audio monitoring in an unknown environment, where initially the sound event classes to be recognized are unknown. For such a scenario, we propose to first localize in time relevant sound events and to classify them, e.g., by comparing with known event classes, in a second step. This contribution is dedicated to the first step, which we call audio diarization, as it is reminiscent of the speaker diarization stage that precedes and simplifies the second stage, speech recognition, in multi-talker conversational speech processing. In this contribution, we define audio diarization as detecting onset and offset times of sound events with overlap for an open set of classes and without user prompts. We show how a speaker diarization system can be adjusted for audio diarization and propose an evaluation setup. Compared to a closed-set sound event detection system, the proposed system achieves similar performance with the additional ability to detect novel sounds.
#### Hybrid Continual Learning for Low-Resource Australian Aboriginal Language Identification
 - **Authors:** Pravina Mylvaganam, Ting Dang, Eliathamby Ambikairajah, Vidhyasaharan Sethu, Jingyao Wu
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2607.11946

 - **Pdf link:** https://arxiv.org/pdf/2607.11946

 - **Abstract**
 Language identification is an important step toward integrating endangered Australian Aboriginal languages (AALs) into speech technologies supporting language revitalisation and digital inclusion. However, extreme data scarcity limits model performance. Transfer learning from high-resource languages shows promise but often suffers from catastrophic forgetting when adapting to new languages. Continual learning (CL) can mitigate this issue, though it remains challenging with very limited data. To address this, we propose two hybrid continual learning methods: Replay Augmented Elastic Weight Consolidation and Constraint Guided Knowledge Distillation to adapt pretrained speech models for AAL identification while preserving previously learned knowledge. Experiments on Warlpiri, Dalabon and Dharawal show that the proposed methods outperform fine-tuning and existing CL baselines, improving adaptation to multiple AALs while maintaining performance on previously learnt high-resource languages.


by Zyzzyva0381 (Windy). 


2026-07-15
