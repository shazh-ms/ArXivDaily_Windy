# Showing new listings for Friday, 3 April 2026
Auto update papers at about 2:30am UTC (10:30am Beijing time) every weekday.


阅读 `Usage.md`了解如何使用此repo实现个性化的Arxiv论文推送

See `Usage.md` for instructions on how to personalize the repo. 


Keyword list: ['text-to-speech', 'text to speech', 'tts', 'LLM-based', 'speech', 'voice']


Excluded: []


### Today: 9papers 
#### Reverberation-Robust Localization of Speakers Using Distinct Speech Onsets and Multi-channel Cross-Correlations
 - **Authors:** Shoufeng Lin
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2604.01524

 - **Pdf link:** https://arxiv.org/pdf/2604.01524

 - **Abstract**
 Many speaker localization methods can be found in the literature. However, speaker localization under strong reverberation still remains a major challenge in the real-world applications. This paper proposes two algorithms for localizing speakers using microphone array recordings of reverberated sounds. To separate concurrent speakers, the first algorithm decomposes microphone signals spectrotemporally into subbands via an auditory filterbank. To suppress reverberation, we propose a novel speech onset detection approach derived from the speech signal and impulse response models, and further propose to formulate the multi-channel cross-correlation coefficient (MCCC) of encoded speech onsets in each subband. The subband results are combined to estimate the directions-of-arrival (DOAs) of speakers. The second algorithm extends the generalized cross-correlation - phase transform (GCC-PHAT) method by using redundant information of multiple microphones to address the reverberation problem. The proposed methods have been evaluated under adverse conditions using not only simulated signals (reverberation time $T_{60}$ of up to $1$s) but also recordings in a real reverberant room ($T_{60} \approx 0.65$s). Comparing with some state-of-the-art localization methods, experimental results confirm that the proposed methods can reliably locate static and moving speakers, in presence of reverberation.
#### Validating Computational Markers of Depressive Behavior: Cross-Linguistic Speech-Based Depression Detection with Neurophysiological Validation
 - **Authors:** Fuxiang Tao, Dongwei Li, Shuning Tang, Xuri Ge, Wei Ma, Anna Esposito, Alessandro Vinciarelli
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2604.01533

 - **Pdf link:** https://arxiv.org/pdf/2604.01533

 - **Abstract**
 Speech-based depression detection has shown promise as an objective diagnostic tool, yet the cross-linguistic robustness of acoustic markers and their neurobiological underpinnings remain underexplored. This study extends Cross-Data Multilevel Attention (CDMA) framework, initially validated on Italian, to investigate these dimensions using a Chinese Mandarin dataset with Electroencephalography (EEG) recordings. We systematically fuse read speech with spontaneous speech across different emotional valences (positive, neutral, negative) to investigate whether emotional arousal is a more critical factor than valence polarity in enhancing detection performance in speech. Additionally, we establish the first neurophysiological validation for a speech-based depression model by correlating its predictions with neural oscillatory patterns during emotional face processing. Our results demonstrate strong cross-linguistic generalizability of the CDMA framework, achieving state-of-the-art performance (F1-score up to 89.6%) on the Chinese dataset, which is comparable to the previous Italian validation. Critically, emotionally valenced speech (both positive and negative) significantly outperformed neutral speech. This comparable performance between positive and negative tasks supports the emotional arousal hypothesis. Most importantly, EEG analysis revealed significant correlations between the model's speech-derived depression estimates and neural oscillatory patterns (theta and alpha bands), demonstrating alignment with established neural markers of emotional dysregulation in depression. This alignment, combined with the model's cross-linguistic robustness, not only supports that the CDMA framework's approach is a universally applicable and neurobiologically validated strategy but also establishes a novel paradigm for the neurophysiological validation of computational mental health models.
#### Robust Pitch Estimation and Tracking for Speakers Based on Subband Encoding and the Generalized Labeled Multi-Bernoulli Filter
 - **Authors:** Shoufeng Lin
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2604.01541

 - **Pdf link:** https://arxiv.org/pdf/2604.01541

 - **Abstract**
 This paper proposes a new pitch estimator and a novel pitch tracker for speakers. We first decompose the sound signal into subbands using an auditory filterbank, assuming time-frequency sparsity of human speech. Instead of directly selecting the number of subbands according to experience, we propose a novel frequency coverage metric to derive the number of subbands and the center frequencies of the filterbank. The subband signals are then encoded inspired by the computational auditory scene analysis (CASA) approach, and the normalized autocorrelations are calculated for pitch estimation. To suppress spurious errors and track the speaker identity, the temporal continuity constraint is exploited and a Generalized Labeled Multi-Bernoulli (GLMB) filter is adapted for pitch tracking, where we use a novel pitch state transition model based on the Ornstein-Uhlenbeck process, and the measurement driven birth model for adaptive new births of pitch targets. Experimental evaluations with various additive noises demonstrate that the proposed methods have achieved better accuracy compared with several state-of-the-art pitch estimation methods in most studied scenarios. Tests using real recordings in a reverberant room also show that the proposed method is robust against reverberation.
#### PhiNet: Speaker Verification with Phonetic Interpretability
 - **Authors:** Yi Ma, Shuai Wang, Tianchi Liu, Haizhou Li
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2604.01590

 - **Pdf link:** https://arxiv.org/pdf/2604.01590

 - **Abstract**
 Despite remarkable progress, automatic speaker verification (ASV) systems typically lack the transparency required for high-accountability applications. Motivated by how human experts perform forensic speaker comparison (FSC), we propose a speaker verification network with phonetic interpretability, PhiNet, designed to enhance both local and global interpretability by leveraging phonetic evidence in decision-making. For users, PhiNet provides detailed phonetic-level comparisons that enable manual inspection of speaker-specific features and facilitate a more critical evaluation of verification outcomes. For developers, it offers explicit reasoning behind verification decisions, simplifying error tracing and informing hyperparameter selection. In our experiments, we demonstrate PhiNet's interpretability with practical examples, including its application in analyzing the impact of different hyperparameters. We conduct both qualitative and quantitative evaluations of the proposed interpretability methods and assess speaker verification performance across multiple benchmark datasets, including VoxCeleb, SITW, and LibriSpeech. Results show that PhiNet achieves performance comparable to traditional black-box ASV models while offering meaningful, interpretable explanations for its decisions, bridging the gap between ASV and forensic analysis.
#### T5Gemma-TTS Technical Report
 - **Authors:** Chihiro Arata, Kiyoshi Kurihara
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2604.01760

 - **Pdf link:** https://arxiv.org/pdf/2604.01760

 - **Abstract**
 Autoregressive neural codec language models have shown strong zero-shot voice cloning ability, but decoder-only architectures treat input text as a prefix that competes with the growing audio sequence for positional capacity, weakening text conditioning over long utterances. We present T5Gemma-TTS, an encoder-decoder codec language model that maintains persistent text conditioning by routing bidirectional text representations through cross-attention at every decoder layer. Built on the T5Gemma pretrained encoder-decoder backbone (2B encoder + 2B decoder; 4B parameters), it inherits rich linguistic knowledge without phoneme conversion and processes text directly at the subword level. To improve duration control, we introduce Progress-Monitoring Rotary Position Embedding (PM-RoPE) in all 26 cross-attention layers, injecting normalized progress signals that help the decoder track target speech length. Trained on 170,000 hours of multilingual speech in English, Chinese, and Japanese, T5Gemma-TTS achieves a statistically significant speaker-similarity gain on Japanese over XTTSv2 (0.677 vs. 0.622; non-overlapping 95% confidence intervals) and the highest numerical Korean speaker similarity (0.747) despite Korean not being included in training, although this margin over XTTSv2 (0.741) is not statistically conclusive. It also attains the lowest numerical Japanese character error rate among five baselines (0.126), though this ranking should be interpreted cautiously because of partial confidence-interval overlap with Kokoro. English results on LibriSpeech should be viewed as an upper-bound estimate because LibriHeavy is a superset of LibriSpeech. Using the same checkpoint, disabling PM-RoPE at inference causes near-complete synthesis failure: CER degrades from 0.129 to 0.982 and duration accuracy drops from 79% to 46%. Code and weights are available at this https URL.
#### GAP-URGENet: A Generative-Predictive Fusion Framework for Universal Speech Enhancement
 - **Authors:** Xiaobin Rong, Yushi Wang, Zheng Wang, Jing Lu
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2604.01832

 - **Pdf link:** https://arxiv.org/pdf/2604.01832

 - **Abstract**
 We introduce GAP-URGENet, a generative-predictive fusion framework developed for Track 1 of the ICASSP 2026 URGENT Challenge. The system integrates a generative branch, which performs full-stack speech restoration in a self-supervised representation domain and reconstructs the waveform via a neural vocoder, along with a predictive branch that performs spectrogram-domain enhancement, providing complementary cues. Outputs from both branches are fused by a post-processing module, which also performs bandwidth extension to generate the enhanced waveform at 48 kHz, later downsampled to the original sampling rate. This generative-predictive fusion improves robustness and perceptual quality, achieving top performance in the blind-test phase and ranking 1st in the objective evaluation. Audio examples are available at this https URL.
#### Combining Masked Language Modeling and Cross-Modal Contrastive Learning for Prosody-Aware TTS
 - **Authors:** Kirill Borodin, Vasiliy Kudryavtsev, Maxim Maslov, Nikita Vasiliev, Mikhail Gorodnichev, Grach Mkrtchian
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2604.01247

 - **Pdf link:** https://arxiv.org/pdf/2604.01247

 - **Abstract**
 We investigate multi-stage pretraining for prosody modeling in diffusion-based TTS. A speaker-conditioned dual-stream encoder is trained with masked language modeling followed by SigLIP-style cross-modal contrastive learning using mixed-phoneme batches, with an additional same-phoneme refinement stage studied separately. We evaluate intrinsic text-audio retrieval and downstream synthesis in Grad-TTS and a latent diffusion TTS system. The two-stage curriculum (MLM + mixed-phoneme contrastive learning) achieves the best overall synthesis quality in terms of intelligibility, speaker similarity, and perceptual measures. Although same-phoneme refinement improves prosodic retrieval, it reduces phoneme discrimination and degrades synthesis. These findings indicate that improvements in embedding-space metrics do not necessarily translate to better generative performance and highlight the need to balance phoneme discrimination and prosodic sensitivity in TTS pretraining.
#### Tracking the emergence of linguistic structure in self-supervised models learning from speech
 - **Authors:** Marianne de Heer Kloots, Martijn Bentum, Hosein Mohebbi, Charlotte Pouw, Gaofei Shen, Willem Zuidema
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Artificial Intelligence (cs.AI); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2604.02043

 - **Pdf link:** https://arxiv.org/pdf/2604.02043

 - **Abstract**
 Self-supervised speech models learn effective representations of spoken language, which have been shown to reflect various aspects of linguistic structure. But when does such structure emerge in model training? We study the encoding of a wide range of linguistic structures, across layers and intermediate checkpoints of six Wav2Vec2 and HuBERT models trained on spoken Dutch. We find that different levels of linguistic structure show notably distinct layerwise patterns as well as learning trajectories, which can partially be explained by differences in their degree of abstraction from the acoustic signal and the timescale at which information from the input is integrated. Moreover, we find that the level at which pre-training objectives are defined strongly affects both the layerwise organization and the learning trajectories of linguistic structures, with greater parallelism induced by higher-order prediction tasks (i.e. iteratively refined pseudo-labels).
#### Prosodic ABX: A Language-Agnostic Method for Measuring Prosodic Contrast in Speech Representations
 - **Authors:** Haitong Sun, Stephen McIntosh, Kwanghee Choi, Eunjung Yeo, Daisuke Saito, Nobuaki Minematsu
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Machine Learning (cs.LG); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2604.02102

 - **Pdf link:** https://arxiv.org/pdf/2604.02102

 - **Abstract**
 Speech representations from self-supervised speech models (S3Ms) are known to be sensitive to phonemic contrasts, but their sensitivity to prosodic contrasts has not been directly measured. The ABX discrimination task has been used to measure phonemic contrast in S3M representations via minimal pairs. We introduce prosodic ABX, an extension of this framework to evaluate prosodic contrast with only a handful of examples and no explicit labels. Also, we build and release a dataset of English and Japanese minimal pairs and use it along with a Mandarin dataset to evaluate contrast in English stress, Japanese pitch accent, and Mandarin tone. Finally, we show that model and layer rankings are often preserved across several experimental conditions, making it practical for low-resource settings.


by Zyzzyva0381 (Windy). 


2026-04-03
