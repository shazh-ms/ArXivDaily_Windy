# Showing new listings for Tuesday, 26 May 2026
Auto update papers at about 2:30am UTC (10:30am Beijing time) every weekday.


阅读 `Usage.md`了解如何使用此repo实现个性化的Arxiv论文推送

See `Usage.md` for instructions on how to personalize the repo. 


Keyword list: ['text-to-speech', 'text to speech', 'tts', 'LLM-based', 'speech', 'voice']


Excluded: []


### Today: 8papers 
#### FC-TTS: Style and Timbre Control in Zero-Shot Text-to-Speech with Disentangled Speech Representations
 - **Authors:** Yoonhyung Lee, Hyunsin Park, Jinhwan Park, Jinkyu Lee
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2605.24618

 - **Pdf link:** https://arxiv.org/pdf/2605.24618

 - **Abstract**
 Recent advances in zero-shot text-to-speech (TTS) have enabled accurate imitation of reference speech in terms of both speaking style and speaker timbre. However, achieving disentangled control over these aspects from separate references remains a challenging task. Several studies have proposed disentangled speech representations that decompose speech into interpretable attributes (e.g., timbre, prosody, and content), providing a promising foundation for TTS with attribute control from separate references. Yet, how to effectively integrate such representations into TTS systems to achieve independent and precise control remains underexplored. In this paper, we present FC-TTS, a zero-shot TTS framework that enables disentangled control of style and timbre by conditioning on two distinct reference utterances. Unlike existing systems that inherit limitations from those pre-trained disentangled representations, FC-TTS introduces key design strategies, including architectural choices, training framework, and auxiliary training objectives, which improve the reliability of attribute separation and dual-reference control. Experiments show that FC-TTS achieves high-fidelity synthesis and competitive zero-shot naturalness, while uniquely supporting consistent and independent manipulation of style and timbre. Audio samples are available at this https URL
#### Rethinking Continual Learning for Speech and Audio: A Representation-Centric Taxonomy and Open Problems
 - **Authors:** Yang Xiao, Siyi Wang, Eun-Jung Holden, Ting Dang
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2605.24863

 - **Pdf link:** https://arxiv.org/pdf/2605.24863

 - **Abstract**
 Speech and audio systems operate in inherently non-stationary environments, yet continual learning (CL) research in this domain, especially in the foundation model era, remains fragmented that fail to account for the coupled, geometry-sensitive nature of acoustic representations. Modern speech foundation models operate over highly entangled, continuous representations that jointly encode linguistic, speaker, and paralinguistic factors within a shared latent space. CL is therefore fundamentally about preserving and evolving shared representation structure rather than retaining isolated task knowledge. In this work, we revisit CL for speech from a representation-centered perspective, and introduce a new taxonomy that organizes CL according to how underlying representation geometry evolves under non-stationary acoustic conditions. We further identify key mismatches between current CL assumptions and speech foundation model behavior, and finally outline a set of open challenges and future research directions.
#### Toward Natural Emotional Text-To-Speech System with Fine-Grained Non-Verbal Expression Control
 - **Authors:** Wangzixi Zhou, Bagus Tris Atmaja, Sakriani Sakti
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2605.25504

 - **Pdf link:** https://arxiv.org/pdf/2605.25504

 - **Abstract**
 While current emotional Text-to-Speech (TTS) models have successfully controlled verbal prosody, they often ignore non-verbal vocalizations (NVs), which are essential for authentic human emotion. Although some non-verbal datasets have recently emerged, they often lack high-quality, fine-grained annotations, which restricts a model's ability to precisely control NV generation. To address this limitation, we propose a novel approach for fine-grained non-verbal expression synthesis. We curate and reprocess female NV utterances from the EARS corpus, develop a new annotation scheme using tags to encode NV types, frequencies, and durations, and build an emotional TTS benchmark to demonstrate its effectiveness. Our evaluation shows that while our NV approach leads to minor trade-offs in perceived naturalness, it significantly improves expressiveness (eMOS 4.20) and emotional recognition accuracy (78.8%). Emotion-specific analysis further reveals that NV cues are highly effective for high-arousal emotions like happy (82.5%) and fear (82.7%), and almost perfectly convey sadness (98.3%).
#### cSTMM: A Unified Complex Spherical Student's $t$ Mixture Model for Directional Statistics in Mask-Based Blind Speech Separation
 - **Authors:** Nobutaka Ito
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2605.25512

 - **Pdf link:** https://arxiv.org/pdf/2605.25512

 - **Abstract**
 Mask-based blind speech separation (BSS) estimates source-wise time-frequency (TF) masks by clustering multichannel observations using spatial information. The directional statistical approach clusters normalized multichannel observations on the complex unit sphere, without explicitly extracting phase and level difference features based on the plane-wave or spherical-wave assumptions. However, prior studies have mostly compared a small number of separately defined directional statistical mixture models, whereas a broader distribution family would enable a more systematic study of how density profiles affect separation performance. We propose the complex spherical Student's t mixture model (cSTMM), a directional mixture model that connects the complex angular central Gaussian mixture model (cACGMM), complex Bingham mixture model (cBMM), and complex Watson mixture model (cWMM) through the degrees-of-freedom parameter $\nu$. We also derive a generalized minorization-maximization (MM) based procedure for parameter estimation. A no-restart evaluation on noise-free LibriSpeech mixtures reverberated with measured room impulse responses shows that a single development-selected value $\nu^\ast=1$ achieved higher test-set mean signal-to-distortion ratio improvements (SDRi) than the cACGMM-equivalent setting $\nu=M$ in all acoustic conditions, with an average condition-wise gain of 0.25dB. The experiments also numerically verify that the proposed formulation numerically recovers the cACGMM, cBMM, and cWMM cases.
#### Ultra-Low-Bitrate Mel-Spectrogram-based Neural Speech Coding with Flow-Matching-based Refinement and Vocoding-driven Reconstruction
 - **Authors:** Hui-Peng Du, Yang Ai, Xiao-Hang Jiang, Yuan Tian, Zhen-Hua Ling
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2605.25669

 - **Pdf link:** https://arxiv.org/pdf/2605.25669

 - **Abstract**
 Ultra-low-bitrate speech coding is pivotal for bandwidth-constrained communication and deep compression, yet maintaining naturalness and speaker identity at such extreme bit budgets remains challenging due to pronounced information loss and quantization instability. To this end, we propose FMelCodec, an ultra-low-bitrate neural speech codec in the mel-spectrogram domain, cast as a three-stage coding-refinement-reconstruction (CRR) framework that can operate at as low as 250 bps. In the CRR framework, the front-end mel-spectrogram coding stage employs a highly aggressive 640x compression/decompression encoder-decoder structure with a single 1024-entry VQ codebook, coupled with an online clustering strategy that reassigns underused codewords to prevent codebook collapse and preserve codebook diversity. The subsequent conditional flow matching (CFM)-based mel-spectrogram refinement stage leverages a lightweight velocity-field estimator and CFM-based solver to refine the codec-degraded mel-spectrogram produced by the preceding decoder, and adopts a self-consistency training scheme that supports fewer iterative inference steps for the purpose of reducing computational overhead. Finally, the vocoding-driven waveform reconstruction stage employs a HiFi-GAN vocoder to faithfully reconstruct waveform from the refined mel-spectrogram. Experiments conducted on two datasets spanning two sampling rates show that, under ultra-low-bitrate constraints of 250 bps for 16 kHz and 750 bps for 48 kHz, both objective and subjective evaluations consistently demonstrate that FMelCodec achieves higher speech reconstruction quality and speaker similarity, while incurring lower computational and model complexity.
#### Zero-Shot Parkinson's Disease Detection from Speech: Comparing Large Audio and Language Models
 - **Authors:** Muhammad Ashad Kabir, Sirajam Munira
 - **Subjects:** Subjects:
Sound (cs.SD); Artificial Intelligence (cs.AI); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2605.24806

 - **Pdf link:** https://arxiv.org/pdf/2605.24806

 - **Abstract**
 Large audio and language models have recently demonstrated zero-shot reasoning capabilities across various domains. However, it remains unclear how the form of audio input, whether handcrafted acoustic features extracted from speech or the raw audio waveform itself, affects performance for Parkinson's disease (PD) detection across different languages. In this study, we systematically compare two input modalities for zero-shot PD detection: (i) handcrafted acoustic features extracted from speech recordings analyzed by a general-purpose LLM, and (ii) direct waveform input analyzed by audio-capable models. Experiments on PD speech datasets in four languages show that performance varies across input modalities, speech tasks, and languages. Handcrafted acoustic features provide more stable performance in a low-resource language (e.g., Bengali), whereas audio input yields dataset-dependent gains. These findings highlight the impact of input modality on zero-shot PD detection from speech.
#### Proactive for Uncertainty: Cause-Aware Error Diagnosis and Interactive Clarification for Spoken Dialogue Systems
 - **Authors:** Yizhou Peng, Ziyang Ma, Changsong Liu, Yi-Wen Chao, Xie Chen, Eng Siong Chng
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2605.25404

 - **Pdf link:** https://arxiv.org/pdf/2605.25404

 - **Abstract**
 Cascaded Automatic Speech Recognition -- Large Language Model (ASR-LLM) pipelines remain popular for industrial Spoken Dialogue Systems (SDS), primarily because their decoupled design ensures perceptual verifiability. However, cascaded systems suffer from error propagation, as transcription failures inevitably cascade to subsequent components, thereby degrading the final interaction quality. Although ASR confidence scores offer a simple filter for unreliable inputs, this approach is fundamentally limited because it typically fails to detect deletion errors or to distinguish between acoustic (inability to hear clearly) and linguistic (inability to understand) mismatches, both of which require targeted recovery strategies. In this paper, we propose a cause-aware error recovery paradigm that fundamentally rethinks robustness in SDS. Unlike traditional confidence filtering, we introduce a suite of small precision-focused detectors that exploit deep ASR latent representations to disentangle token-level errors into perception, comprehension, and deletion failures. This fine-grained diagnostic intelligence empowers the LLM to orchestrate targeted, multi-turn clarification strategies, effectively transforming ambiguous signals into seamless user interactions. Experimental results validate the precision of our approach, which more than doubles the recall on domain-shift errors (57.96% vs. 23.66%) compared to baselines. Crucially, this diagnostic precision yields up to a 30% reduction in WER and a 17% improvement on the downstream task across diverse accents, distortions, and domains.
#### Thaka at KSAA-2026 Task 2: Regularized Fine-Tuning for Arabic Speech Diacritization
 - **Authors:** Meshal Alamr, Hassan Alqaeri, Abdullah Aldahlawi
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2605.25928

 - **Pdf link:** https://arxiv.org/pdf/2605.25928

 - **Abstract**
 We describe the winning system for Task 2 of the KSAA-2026 Shared Task on Arabic Speech Dictation with Automatic Diacritization. The task requires producing fully diacritized Arabic text from speech audio and undiacritized transcripts, with only 2,327 training samples available and no external data permitted. Our system fine-tunes CATT-Whisper, a character-level multimodal model combining a pretrained CATT text encoder with a frozen Whisper speech encoder. The key to our approach is training regularization: R-Drop consistency regularization, Optuna-optimized hyperparameters with high weight decay, and Focal Loss. At inference, we average 200 stochastic forward passes across four model checkpoints using Monte Carlo Dropout at the softmax probability level. The system achieves 23.26% WER on the primary leaderboard metric (with case endings, including no-diacritic positions), placing 1st among all participants.


by Zyzzyva0381 (Windy). 


2026-05-26
