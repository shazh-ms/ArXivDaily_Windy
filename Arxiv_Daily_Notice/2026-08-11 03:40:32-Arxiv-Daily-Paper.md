# Showing new listings for Tuesday, 11 August 2026
Auto update papers at about 2:30am UTC (10:30am Beijing time) every weekday.


阅读 `Usage.md`了解如何使用此repo实现个性化的Arxiv论文推送

See `Usage.md` for instructions on how to personalize the repo. 


Keyword list: ['text-to-speech', 'text to speech', 'tts', 'LLM-based', 'speech', 'voice']


Excluded: []


### Today: 10papers 
#### Mitigating Over-Suppression in Speech Enhancement via Inference-Time Rethink-and-Refine Correction Module
 - **Authors:** Mike Qu, Yu-Wen Chen, Julia Hirschberg
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2608.07781

 - **Pdf link:** https://arxiv.org/pdf/2608.07781

 - **Abstract**
 We present a rethink-and-refine correction module that addresses over-suppression, a common failure mode of speech enhancement (SE) models, where speech cues are suppressed alongside noise. Our method operates entirely in the inference stage without additional training, allowing seamless integration with diverse SE models. Given noisy and enhanced signals, we obtain word- or phoneme-level alignments using an automatic speech recognition model and identify intervals where enhancement is unreliable. These intervals are then selectively remixed through convex interpolation, with per-segment weights optimized to maximize a composite objective balancing perceptual quality and speech preservation. Experiments on the URGENT 2024 and 2025, VCTK-DEMAND, and MSP-PODCAST datasets show consistent improvements in perceptual quality, intelligibility, and downstream performance compared to conventional SE alone, demonstrating the benefit of rethink-and-refine framework for robust speech processing.
#### The Voiceprint Fallacy: Why Voices Are Not Unique Biometric Imprints
 - **Authors:** Tianle Yang, Cuiling Zhang, Chengzhe Sun, Siwei Lyu, Phil Rose
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Computation and Language (cs.CL); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2608.07980

 - **Pdf link:** https://arxiv.org/pdf/2608.07980

 - **Abstract**
 In recent years, the term voiceprint has regained attention, particularly in technological applications and policy-making contexts, often carrying the assumption that a person's voice constitutes a stable and unique biometric trace analogous to a fingerprint. Yet this conception has been repeatedly criticized and rejected by forensic voice experts throughout the decades since its introduction. Although voices undoubtedly contain speaker-related information, this simplified conception obscures the highly dynamic and context-dependent nature of speech. This article revisits the voiceprint fallacy and reconsiders what can count as evidence of speaker identity by reviewing the historical development of voiceprint identification, evidence on human voice variability, developments in forensic voice comparison, research on human and automatic speaker recognition, and the recent challenge posed by deepfake speech to speaker identity. We point out that the voiceprint metaphor and its underlying implications are scientifically misleading because they transform a probabilistic source of speaker information into an imagined stable object of identity. To avoid treating voices as imprint-like traces, we recommend that voice evidence be interpreted through validated and calibrated probabilistic frameworks that explicitly account for variability, uncertainty, and alternative explanations.
#### SraVaani 1.0: Scaling Inclusive Speech Recognition for Indic Languages
 - **Authors:** Sujith Pulikodan, Agneedh Basu, Pavan Kumar J, Pranav D Bhat, Suryansh Shukla, Nihar Desai, Prasanta Kumar Ghosh
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2608.08235

 - **Pdf link:** https://arxiv.org/pdf/2608.08235

 - **Abstract**
 India's linguistic landscape spans over 700 languages and thousands of dialects, yet the vast majority of automatic speech recognition (ASR) systems support only a small fraction of this diversity. We present SraVaani-1.0, a multilingual ASR model covering 65 Indian languages and dialects, many of which currently have no publicly available or competing ASR system. SraVaani-1.0 is built on a FastConformer architecture and trained from scratch through a three-stage this http URL the first stage, we perform self-supervised pretraining on 31,255 hours of unlabelled speech from the VAANI corpus using a contrastive learning objective. In the second stage, we introduce an audio-image representation alignment stage that leverages the paired images and speech available in the VAANI corpus. This multimodal alignment encourages the speech encoder to learn semantically richer representations by exploiting the relationship between visual context and spoken content, thereby improving downstream recognition, particularly for low-resource this http URL the final stage, the aligned encoder is fine-tuned end-to-end using a Hybrid Token-and-Duration Transducer (TDT)-CTC decoder on 31,263 hours of labelled multilingual Indian speech compiled from 24 public datasets spanning 65 languages and dialects. We evaluate SraVaani-1.0 against three state-of-the-art multilingual ASR systems across eight benchmarks. SraVaani-1.0 achieves the lowest word error rate (WER) on a large number of language-dataset pairs while remaining competitive with the best-performing systems on high-resource this http URL importantly, it is the only open-source evaluated model that provides transcription capability for multiple low-resource and tribal Indian languages, which are assessed exclusively on the VAANI benchmark.
#### ReLMCodec: Designing Predictable Speech Tokens from Pre-Quantization Phoneme Structure
 - **Authors:** Zixiang Wan, Xusheng Yang, Zheng Wang, Peiji Yang
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2608.08286

 - **Pdf link:** https://arxiv.org/pdf/2608.08286

 - **Abstract**
 Neural speech codecs face a fundamental tension in the language-model era: tokens that support high-fidelity reconstruction are not necessarily easy for autoregressive models to predict. Our controlled analysis of diverse codec and self-supervised speech representations shows that clearer phoneme structure before discrete code assignment is consistently associated with easier autoregressive token prediction. Yet phoneme structure alone is insufficient for high-fidelity reconstruction, which also requires reconstruction-relevant acoustic detail. Guided by this observation, we introduce ReLMCodec, a low-bitrate single-codebook speech codec built upon a preserve--control--refine principle: it preserves the linguistic organization of frozen self-supervised learning (SSL) features at the quantizer input, controls reconstruction-driven drift through Pre-quantization Anchor-Preserving Adaptation (PAPA), and refines the quantized latent space with a training-only WavLM-Large L24 teacher to reduce phoneme-level token fragmentation. Together, these components allow acoustic detail to support waveform reconstruction while keeping the resulting token sequence predictable for autoregressive models. At 650 and 800 bps, ReLMCodec moves the empirical single-stream predictability--reconstruction frontier in our evaluations, with gains that carry over to downstream text-to-speech (TTS) synthesis in both intelligibility and speaker similarity.
#### CtrlSpeech: Coarse-to-Fine Control for Expressive Speech Synthesis
 - **Authors:** Zhisheng Zheng, Xiaohang Sun, Zhu Liu, Caren Chen, Rohith Kumar, Manoj Aggarwal, Gerard Medioni, David Harwath
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Human-Computer Interaction (cs.HC); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2608.08362

 - **Pdf link:** https://arxiv.org/pdf/2608.08362

 - **Abstract**
 Recent Text-To-Speech (TTS) systems have achieved strong naturalness and zero-shot voice cloning performance, but fine-grained control of expressive speech at the word or phoneme level remains challenging. We propose CtrlSpeech, a controllable, expressive TTS framework with coarse-to-fine control. Built on the DiTAR architecture, CtrlSpeech combines global speaker conditioning with phone-aligned pitch, loudness, and duration signals, enabling localized prosodic control while preserving the target speaker's timbre. This design allows users to adjust expressive attributes at a fine temporal granularity, making speech refinement more flexible and controllable. Experimental results show that CtrlSpeech achieves competitive zero-shot TTS performance and improves controllability over expressive attributes, demonstrating its effectiveness for flexible and practical expressive speech synthesis.
#### BAMU: Bitstream-Aware Marginal-Utility Allocation for Frozen Pretrained Neural Speech Codecs
 - **Authors:** Mingyu Zhao, Zijian Lin, Yutang Feng, Jiatao Chen, Fan Wang, Jiehui Luo, Yuhao Ding, Jinchao Zhang, Zhiyong Wu
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2608.08432

 - **Pdf link:** https://arxiv.org/pdf/2608.08432

 - **Abstract**
 Pretrained neural speech codecs typically use a fixed residual vector quantization (RVQ) depth for all frames, ignoring temporal variation in quantization difficulty. We propose BAMU, a bitstream-aware dynamic RVQ allocation framework for frozen pretrained codecs. A lightweight, rate-independent predictor estimates frame- and layer-wise marginal latent-distortion reductions, while a constrained allocator selects prefix-valid depths under an exact serialized-size budget. Experiments on EnCodec and DAC over LibriSpeech, together with VCTK evaluation, show consistent EnCodec gains and DAC improvements mainly at medium and high rates. A 30-listener study confirms a MOS improvement from 3.449 to 3.780 over matched fixed-depth coding.
#### Speaker Role and Language Diarization for Analyzing Multilingual Interviews for Language Proficiency of Older Adults
 - **Authors:** Anfeng Xu, Tiantian Feng, Kevin Huang, Pranali Khobragade, Sudarsana Kadiri, Anushikha Dhankhar, Madeleine Snider, Sarah Gao, Miguel Arce Rentería, Jinkook Lee, Shrikanth Narayanan
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2608.09032

 - **Pdf link:** https://arxiv.org/pdf/2608.09032

 - **Abstract**
 Automatic language proficiency assessment in the context of multilingual interview-based settings remains underexplored. In this work, we develop Whisper-based speaker-role and language diarization systems to automatically extract respondent speech and characterize language usage in multilingual interviews with older adults. We further investigate whether diarization-derived conversational and language-use behaviors can support downstream language proficiency assessment. Results show that language-adapted Whisper models substantially improve language diarization performance for lower-resource and linguistically related Indian languages. Statistical analyses reveal that respondent speech ratio and intended language usage are strong predictors of proficiency ratings. Furthermore, simple diarization-derived behavioral features achieve performance comparable to Whisper-based speech embeddings for proficiency prediction, while combining both yields the best results. Importantly, both the speech and language use statistical analyses and language proficiency prediction performance remain largely preserved when using fully automatic diarization outputs, demonstrating the potential of respondent-centric conversational analysis for scalable language proficiency assessment.
#### Dynamic Clustering for Cross-Segment Permutation Alignment in Long Speech Separation
 - **Authors:** Yuzhu Wang, Archontis Politis, Konstantinos Drossos, Tuomas Virtanen
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2608.09451

 - **Pdf link:** https://arxiv.org/pdf/2608.09451

 - **Abstract**
 Long speech separation typically employs a segment-separation-stitch paradigm where recordings are divided into short segments, processed independently, and stitched together. Its challenge lies in predicting cross-segment permutations. This paper proposes a training-free dynamic clustering approach for cross-segment permutation alignment using speaker embedding reference pools. The method predicts the permutation using the cosine similarity between current segment embeddings and the reference pools. The approach updates reference pools by retaining the most representative speaker embeddings based on their overall cosine similarity with existing references. As a plug-and-play post-processing module compatible with existing separation models, the proposed method demonstrates superior performance compared to existing methods on dense and sparse long speech scenarios, particularly in challenging sparse scenarios with extended utterance gaps, and further shows robustness to speaker count estimation errors in unknown speaker count scenarios.
#### Multilingual Emotion Neurons in Large Audio-Language Models
 - **Authors:** Xiutian Zhao, Philipp Koehn, Björn Schuller, Berrak Sisman
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2608.08772

 - **Pdf link:** https://arxiv.org/pdf/2608.08772

 - **Abstract**
 Emotion is central to human communication, and its expression varies across languages. Large audio-language models (LALMs) achieve strong performance on multilingual speech tasks, yet it remains unclear whether they encode emotion through language-specific correlations or language-agnostic representations. We present the first neuron-level interpretability study of this question. We define Multilingual Emotion Neurons (MLENs) as functional units exhibiting stable emotional selectivity and aligned causal effects across languages, and introduce Consistency-Regularized Fusion (CR-Fusion) to identify them. Across four modern LALMs and 12 typologically diverse languages, emotion-sensitive neurons identified independently per language show minimal overlap, and additional monolingual identification data saturates quickly without isolating more transferable units, motivating identification from pooled cross-lingual evidence. Causal interventions demonstrate that MLENs identified by CR-Fusion provide more precise and transferable affective control than monolingual neuron sets in both zero-shot and low-resource settings. Leave-one-out ablations further reveal asymmetric transfer: individual identification languages, including low-resource ones, contribute non-redundant evidence, while several low-resource languages benefit most from the resulting cross-lingual transfer. Together, our findings provide the first causal, neuron-level account of how LALMs encode emotion across languages, and establish multilingual neuron identification as an effective mechanism for understanding cross-lingual affective behavior.
#### Structured Phonological Representations for Audio-Articulatory rtMRI Speech Classification
 - **Authors:** Abner Hernandez, Tomás Arias Vergara, Daiqi Liu, Andreas Maier, Paula Andrea Pérez-Toro
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2608.09767

 - **Pdf link:** https://arxiv.org/pdf/2608.09767

 - **Abstract**
 Real-time MRI makes it possible to observe vocal-tract articulation during speech, but mapping these articulatory patterns to phonetic and phonological categories remains challenging. We investigate whether PhonoQ, an audio-based model trained to recognize structured phonological features, provides useful information for audio--articulatory modeling. Specifically, we extract representations from PhonoQ's Conformer module, whose training is shaped by supervision for manner, place, voicing, and vowel features. Using articulatory contours with synchronized audio-derived features, we compare WavLM-large and HuBERT-large baselines with models that incorporate PhonoQ-derived representations. Across unseen-speech and unseen-subject settings, these features improve macro-F1 for phonological targets including manner, place, voicing, vowel height, and vowel backness, and also improve fine-grained 39-phoneme classification. In a contour-only inference setting, audio-derived teacher supervision yields modest but consistent gains over contour-only training, indicating that phonological information from synchronized audio can be partially transferred to articulatory models. Finally, posterior analyses show interpretable surface-sensitive patterns consistent with flapping-like /t/ realizations, /t/-/r/ retraction or affrication, and nasal place assimilation.


by Zyzzyva0381 (Windy). 


2026-08-11
