# Showing new listings for Friday, 13 March 2026
Auto update papers at about 2:30am UTC (10:30am Beijing time) every weekday.


阅读 `Usage.md`了解如何使用此repo实现个性化的Arxiv论文推送

See `Usage.md` for instructions on how to personalize the repo. 


Keyword list: ['text-to-speech', 'text to speech', 'tts', 'LLM-based', 'speech', 'voice']


Excluded: []


### Today: 13papers 
#### Can LLMs Help Localize Fake Words in Partially Fake Speech?
 - **Authors:** Lin Zhang, Thomas Thebaud, Zexin Cai, Sanjeev Khudanpur, Daniel Povey, Leibny Paola García-Perera, Matthew Wiesner, Nicholas Andrews
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2603.11205

 - **Pdf link:** https://arxiv.org/pdf/2603.11205

 - **Abstract**
 Large language models (LLMs), trained on large-scale text, have recently attracted significant attention for their strong performance across many tasks. Motivated by this, we investigate whether a text-trained LLM can help localize fake words in partially fake speech, where only specific words within a speech are edited. We build a speech LLM to perform fake word localization via next token prediction. Experiments and analyses on AV-Deepfake1M and PartialEdit indicates that the model frequently leverages editing-style pattern learned from the training data, particularly word-level polarity substitutions for those two databases we discussed, as cues for localizing fake words. Although such particular patterns provide useful information in an in-domain scenario, how to avoid over-reliance on such particular pattern and improve generalization to unseen editing styles remains an open question.
#### Self-Speculative Decoding for LLM-based ASR with CTC Encoder Drafts
 - **Authors:** George Saon, Samuel Thomas, Takashi Fukuda, Tohru Nagano, Avihu Dekel, Luis Lastras
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2603.11243

 - **Pdf link:** https://arxiv.org/pdf/2603.11243

 - **Abstract**
 We propose self-speculative decoding for speech-aware LLMs by using the CTC encoder as a draft model to accelerate auto-regressive (AR) inference and improve ASR accuracy. Our three-step procedure works as follows: (1) if the frame entropies of the CTC output distributions are below a threshold, the greedy CTC hypothesis is accepted as final; (2) otherwise, the CTC hypothesis is verified in a single LLM forward pass using a relaxed acceptance criterion based on token likelihoods; (3) if verification fails, AR decoding resumes from the accepted CTC prefix. Experiments on nine corpora and five languages show that this approach can simultaneously accelerate decoding and reduce WER. On the HuggingFace Open ASR benchmark with a 1B parameter LLM and 440M parameter CTC encoder, we achieve a record 5.58% WER and improve the inverse real time factor by a factor of 4.4 with only a 12% relative WER increase over AR search. Code and model weights are publicly available under a permissive license.
#### SEMamba++: A General Speech Restoration Framework Leveraging Global, Local, and Periodic Spectral Patterns
 - **Authors:** Yongjoon Lee, Jung-Woo Choi
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2603.11669

 - **Pdf link:** https://arxiv.org/pdf/2603.11669

 - **Abstract**
 General speech restoration demands techniques that can interpret complex speech structures under various distortions. While State-Space Models like SEMamba have advanced the state-of-the-art in speech denoising, they are not inherently optimized for critical speech characteristics, such as spectral periodicity or multi-resolution frequency analysis. In this work, we introduce an architecture tailored to incorporate speech-specific features as inductive biases. In particular, we propose Frequency GLP, a frequency feature extraction block that effectively and efficiently leverages the properties of frequency bins. Then, we design a multi-resolution parallel time-frequency dual-processing block to capture diverse spectral patterns, and a learnable mapping to further enhance model performance. With all our ideas combined, the proposed SEMamba++ achieves the best performance among multiple baseline models while remaining computationally efficient.
#### RAF: Relativistic Adversarial Feedback For Universal Speech Synthesis
 - **Authors:** Yongjoon Lee, Jung-Woo Choi
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2603.11678

 - **Pdf link:** https://arxiv.org/pdf/2603.11678

 - **Abstract**
 We propose Relativistic Adversarial Feedback (RAF), a novel training objective for GAN vocoders that improves in-domain fidelity and generalization to unseen scenarios. Although modern GAN vocoders employ advanced architectures, their training objectives often fail to promote generalizable representations. RAF addresses this problem by leveraging speech self-supervised learning models to assist discriminators in evaluating sample quality, encouraging the generator to learn richer representations. Furthermore, we utilize relativistic pairing for real and fake waveforms to improve the modeling of the training data distribution. Experiments across multiple datasets show consistent gains in both objective and subjective metrics on GAN-based vocoders. Importantly, the RAF-trained BigVGAN-base outperforms the LSGAN-trained BigVGAN in perceptual quality using only 12\% of the parameters. Comparative studies further confirm the effectiveness of RAF as a training framework for GAN vocoders.
#### Affect Decoding in Phonated and Silent Speech Production from Surface EMG
 - **Authors:** Simon Pistrosch, Kleanthis Avramidis, Tiantian Feng, Jihwan Lee, Monica Gonzalez-Machorro, Shrikanth Narayanan, Björn W. Schuller
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Artificial Intelligence (cs.AI); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2603.11715

 - **Pdf link:** https://arxiv.org/pdf/2603.11715

 - **Abstract**
 The expression of affect is integral to spoken communication, yet, its link to underlying articulatory execution remains unclear. Measures of articulatory muscle activity such as EMG could reveal how speech production is modulated by emotion alongside acoustic speech analyses. We investigate affect decoding from facial and neck surface electromyography (sEMG) during phonated and silent speech production. For this purpose, we introduce a dataset comprising 2,780 utterances from 12 participants across 3 tasks, on which we evaluate both intra- and inter-subject decoding using a range of features and model embeddings. Our results reveal that EMG representations reliably discriminate frustration with up to 0.845 AUC, and generalize well across articulation modes. Our ablation study further demonstrates that affective signatures are embedded in facial motor activity and persist in the absence of phonation, highlighting the potential of EMG sensing for affect-aware silent speech interfaces.
#### Acoustic-to-Articulatory Inversion of Clean Speech Using an MRI-Trained Model
 - **Authors:** Sofiane Azzouz, Pierre-André Vuissoz, Yves Laprie
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2603.11845

 - **Pdf link:** https://arxiv.org/pdf/2603.11845

 - **Abstract**
 Articulatory acoustic inversion reconstructs vocal tract shapes from speech. Real-time magnetic resonance imaging (rt-MRI) allows simultaneous acquisition of both the acoustic speech signal and articulatory information. Besides the complexity of rt-MRI acquisition, the recorded audio is heavily corrupted by scanner noise and requires denoising to be usable. For practical use, it must be possible to invert speech recorded without MRI noise. In this study, we investigate the use of speech recorded in a clean acoustic environment as an alternative to denoised MRI speech. To this end we compare two signals from the same speaker with identical sentences which are aligned using phonetic segmentation. A model trained on denoised MRI speech is evaluated on both denoised MRI and clean speech. We also assess a model trained and tested only on clean speech. Results show that clean speech supports articulatory inversion effectively, achieving an RMSE of 1.56 mm, close to MRI-based performance.
#### Reconstruction of the Vocal Tract from Speech via Phonetic Representations Using MRI Data
 - **Authors:** Sofiane Azzouz, Pierre-André Vuissoz, Yves Laprie
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2603.11847

 - **Pdf link:** https://arxiv.org/pdf/2603.11847

 - **Abstract**
 Articulatory acoustic inversion aims to reconstruct the complete geometry of the vocal tract from the speech signal. In this paper, we present a comparative study of several levels of phonetic segmentation accuracy, together with a comparison to the baseline introduced in our previous work, which is based on Mel-Frequency Cepstral Coefficients (MFCCs). All the approaches considered are based on a denoised speech signal and aim to investigate the impact of incorporating phonetic information through three successive levels: an uncorrected automatic transcription, a temporally aligned phonetic segmentation, and an expert manual correction following alignment. The models are trained to predict articulatory contours extracted from vocal tract MRI images using an automatic contour tracking method. The results show that, among the models relying on phonetic representations, manual correction after alignment yields the best performance, approaching that of the baseline.
#### Silent Speech Interfaces in the Era of Large Language Models: A Comprehensive Taxonomy and Systematic Review
 - **Authors:** Kele Xu, Yifan Wang, Ming Feng, Qisheng Xu, Wuyang Chen, Yutao Dou, Cheng Yang, Huaimin Wang
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2603.11877

 - **Pdf link:** https://arxiv.org/pdf/2603.11877

 - **Abstract**
 Human-computer interaction has traditionally relied on the acoustic channel, a dependency that introduces systemic vulnerabilities to environmental noise, privacy constraints, and physiological speech impairments. Silent Speech Interfaces (SSIs) emerge as a transformative paradigm that bypasses the acoustic stage by decoding linguistic intent directly from the neuro-muscular-articulatory continuum. This review provides a high-level synthesis of the SSI landscape, transitioning from traditional transducer-centric analysis to a holistic intent-to-execution taxonomy. We systematically evaluate sensing modalities across four critical physiological interception points: neural oscillations, neuromuscular activation, articulatory kinematics (ultrasound/magnetometry), and pervasive active probing via acoustic or radio-frequency sensing. Critically, we analyze the current paradigm shift from heuristic signal processing to Latent Semantic Alignment. In this new era, Large Language Models (LLMs) and deep generative architectures serve as high-level linguistic priors to resolve the ``informational sparsity'' and non-stationarity of biosignals. By mapping fragmented physiological gestures into structured semantic latent spaces, modern SSI frameworks have, for the first time, approached the Word Error Rate usability threshold required for real-world deployment. We further examine the transition of SSIs from bulky laboratory instrumentation to ``invisible interfaces'' integrated into commodity-grade wearables, such as earables and smart glasses. Finally, we outline a strategic roadmap addressing the ``user-dependency paradox'' through self-supervised foundation models and define the ethical boundaries of ``neuro-security'' to protect cognitive liberty in an increasingly interfaced world.
#### Dr. SHAP-AV: Decoding Relative Modality Contributions via Shapley Attribution in Audio-Visual Speech Recognition
 - **Authors:** Umberto Cappellazzo, Stavros Petridis, Maja Pantic
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Computer Vision and Pattern Recognition (cs.CV); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2603.12046

 - **Pdf link:** https://arxiv.org/pdf/2603.12046

 - **Abstract**
 Audio-Visual Speech Recognition (AVSR) leverages both acoustic and visual information for robust recognition under noise. However, how models balance these modalities remains unclear. We present Dr. SHAP-AV, a framework using Shapley values to analyze modality contributions in AVSR. Through experiments on six models across two benchmarks and varying SNR levels, we introduce three analyses: Global SHAP for overall modality balance, Generative SHAP for contribution dynamics during decoding, and Temporal Alignment SHAP for input-output correspondence. Our findings reveal that models shift toward visual reliance under noise yet maintain high audio contributions even under severe degradation. Modality balance evolves during generation, temporal alignment holds under noise, and SNR is the dominant factor driving modality weighting. These findings expose a persistent audio bias, motivating ad-hoc modality-weighting mechanisms and Shapley-based attribution as a standard AVSR diagnostic.
#### Fair-Gate: Fairness-Aware Interpretable Risk Gating for Sex-Fair Voice Biometrics
 - **Authors:** Yangyang Qu, Todisco Massimiliano, Galdi Chiara, Evans Nicholas
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2603.11360

 - **Pdf link:** https://arxiv.org/pdf/2603.11360

 - **Abstract**
 Voice biometric systems can exhibit sex-related performance gaps even when overall verification accuracy is strong. We attribute these gaps to two practical mechanisms: (i) demographic shortcut learning, where speaker classification training exploits spurious correlations between sex and speaker identity, and (ii) feature entanglement, where sex-linked acoustic variation overlaps with identity cues and cannot be removed without degrading speaker discrimination. We propose Fair-Gate, a fairness-aware and interpretable risk-gating framework that addresses both mechanisms in a single pipeline. Fair-Gate applies risk extrapolation to reduce variation in speaker-classification risk across proxy sex groups, and introduces a local complementary gate that routes intermediate features into an identity branch and a sex branch. The gate provides interpretability by producing an explicit routing mask that can be inspected to understand which features are allocated to identity versus sex-related pathways. Experiments on VoxCeleb1 show that Fair-Gate improves the utility--fairness trade-off, yielding more sex-fair ASV performance under challenging evaluation conditions.
#### Continued Pretraining for Low-Resource Swahili ASR: Achieving State-of-the-Art Performance with Minimal Labeled Data
 - **Authors:** Hillary Mutisya, John Mugane
 - **Subjects:** Subjects:
Sound (cs.SD); Machine Learning (cs.LG); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2603.11378

 - **Pdf link:** https://arxiv.org/pdf/2603.11378

 - **Abstract**
 We investigate continued pretraining (CPT) for adapting wav2vec2-bert-2.0 to Swahili automatic speech recognition (ASR). Our approach combines unlabeled audio with limited labeled data through pseudo-labeled CPT followed by supervised finetuning. With 20,000 labeled samples, we achieve 3.24% WER on Common Voice Swahili-an 82% relative improvement over the baseline. This result surpasses the best previously reported academic system (8.3% WER from XLS-R) by 61% relative improvement. We provide concrete data requirements and a replicable methodology applicable to other low-resource languages.
#### AnimeScore: A Preference-Based Dataset and Framework for Evaluating Anime-Like Speech Style
 - **Authors:** Joonyong Park, Jerry Li
 - **Subjects:** Subjects:
Sound (cs.SD); Computation and Language (cs.CL); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2603.11482

 - **Pdf link:** https://arxiv.org/pdf/2603.11482

 - **Abstract**
 Evaluating 'anime-like' voices currently relies on costly subjective judgments, yet no standardized objective metric exists. A key challenge is that anime-likeness, unlike naturalness, lacks a shared absolute scale, making conventional Mean Opinion Score (MOS) protocols unreliable. To address this gap, we propose AnimeScore, a preference-based framework for automatic anime-likeness evaluation via pairwise ranking. We collect 15,000 pairwise judgments from 187 evaluators with free-form descriptions, and acoustic analysis reveals that perceived anime-likeness is driven by controlled resonance shaping, prosodic continuity, and deliberate articulation rather than simple heuristics such as high pitch. We show that handcrafted acoustic features reach a 69.3% AUC ceiling, while SSL-based ranking models achieve up to 90.8% AUC, providing a practical metric that can also serve as a reward signal for preference-based optimization of generative speech models.
#### Resurfacing Paralinguistic Awareness in Large Audio Language Models
 - **Authors:** Hao Yang, Minghan Wang, Tongtong Wu, Lizhen Qu, Ehsan Shareghi, Gholamreza Haffari
 - **Subjects:** Subjects:
Sound (cs.SD); Computation and Language (cs.CL); Multimedia (cs.MM); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2603.11947

 - **Pdf link:** https://arxiv.org/pdf/2603.11947

 - **Abstract**
 Large Audio Language Models (LALMs) have expanded the interaction with human to speech modality, which introduces great interactive potential, due to the paralinguistic cues implicitly indicating the user context. However, building on the current content-centred paradigm, LALMs usually neglect such paralinguistic cues and respond solely based on query content. In this work, to resurface the paralinguistic awareness in LALMs, we introduce five diverse layer-wise analyses to jointly identify paralinguistic layers and semantic understanding layers. Based on these insights, we propose a paralinguistic-enhanced fine-tuning (PE-FT) protocol accordingly to equip LALMs with paralinguistic-aware capabilities, including (1) selective-layer fine-tuning, and (2) an auxiliary dual-level classification head. Our experiments demonstrate that PE-FT protocol efficiently and effectively resurfaces the paralinguistic awareness, even surpassing the performance of the all-layer fine-tuning strategy.


by Zyzzyva0381 (Windy). 


2026-03-13
