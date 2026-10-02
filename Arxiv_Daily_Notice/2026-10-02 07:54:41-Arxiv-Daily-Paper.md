# Showing new listings for Friday, 2 October 2026
Auto update papers at about 2:30am UTC (10:30am Beijing time) every weekday.


阅读 `Usage.md`了解如何使用此repo实现个性化的Arxiv论文推送

See `Usage.md` for instructions on how to personalize the repo. 


Keyword list: ['text-to-speech', 'text to speech', 'tts', 'LLM-based', 'speech', 'voice']


Excluded: []


### Today: 16papers 
#### When Intent Arrives Late: A Benchmark for Full-Duplex Speech Models under Delayed Intent Revelation
 - **Authors:** Yang Xiao, Tianyi Peng, Hanyu Meng, Ting Dang
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2610.00272

 - **Pdf link:** https://arxiv.org/pdf/2610.00272

 - **Abstract**
 Native full-duplex speech models can respond before a user finishes speaking, making the behavior of the model depend on both the final utterance and when intent-defining information arrives. Existing benchmarks primarily evaluate turn-taking mechanics, whereas safety evaluations typically assume that the complete request is observed prior to response generation. We introduce LateIntent-Bench, a matched-pair benchmark for delayed intent revelation. A shared ambiguous prefix precedes either a benign or a harmful continuation, using a controlled pause to delay when the branches become distinguishable. We define the Premature Response Rate (PRR) to measure whether response onset precedes the revelation of intent. Joint evaluation of harmful and benign engagement distinguishes changes in safety selectivity from general losses in responsiveness. Across 3,136 sessions, four native full-duplex models exhibit distinct response-timing patterns under delayed intent. Three models show increased harmful engagement while maintaining benign responsiveness, whereas one loses engagement with both branches. Inserting a 1.5s silence after intent revelation keeps PRR near the no-pause baseline and substantially reduces changes in harmful engagement. These results demonstrate that evaluating fully specified requests alone fails to capture emerging timing patterns, highlighting the necessity of assessing delayed intent revelation in full-duplex models. The code will be publicly released soon.
#### Multi-agent Auditory Scene Analysis: Improved Localization Speed and Robustness by Multi-beamformed Speech Quality Feedback
 - **Authors:** Caleb Rascon
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Artificial Intelligence (cs.AI); Multiagent Systems (cs.MA)
 - **Arxiv link:** https://arxiv.org/abs/2610.00538

 - **Pdf link:** https://arxiv.org/pdf/2610.00538

 - **Abstract**
 A real-time auditory scene analyzer (ASA) aims to carry out the tasks of locating, separating and classifying the sound sources present in a given acoustic environment. Recently, an effort has been made into modelling an ASA as a multi-agent system, with each one of its agents performing one of the aforementioned tasks and communicating their results to the rest of their peer agents. These communication routes are used as feedback loops to fix local errors at a global level, providing robustness while reducing local complexity. An example of the benefits of this approach is the optimization of speech quality by correcting in real-time the estimated location of the speech source of interest. However, their optimization speed has been shown to be considerably slow. One possible reason is that it solely relies on a series of single quality estimations (provided by a reference-free quality estimator model) that vary considerably from one window to the next, which results in a difficult search space to optimize. In this work, a new optimization mechanism is proposed that instead relies on a series of sets of quality estimations over a range of locations, providing a clearer view of the search space, simplifying its optimization. The proposed ASA now has a considerably smaller optimization time, is more accurate, and is more stable when being evaluated in real-life acoustic scenarios to correct higher levels of localization errors, all while being less complex than previous efforts. The only trade-off is that there is an increase in the response time of the quality estimation agent, but the complete ASA is still able to run in real-time. The performance shown in this work again shows the benefits of modelling an ASA as a multi-agent system.
#### Silence-the-Mimic: Accelerating Imperceptible Perturbation Generation Against Voice Cloning
 - **Authors:** Runqiu Xu
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2610.00662

 - **Pdf link:** https://arxiv.org/pdf/2610.00662

 - **Abstract**
 Deep neural network-based Voice Conversion (VC) and Text-to-Speech (TTS) models have rapidly advanced, enabling realistic voice cloning with minimal input data. Such capabilities raise serious concerns over unauthorized cloning of speaker identities and the associated privacy and security risks. Current imperceptible adversarial protection methods rely on quality control losses that are highly sensitive to hyperparameter tuning and computationally expensive due to lengthy optimization. To address these limitations, we propose a fast protection method that generates perceptually constrained perturbations in the frequency domain under a psychoacoustic masking-based constraint. Our approach strictly enforces perceptibility bounds during adversarial training, eliminating the need for iterative quality balancing and significantly reducing computational cost. Experiments on multiple state-of-the-art VC and TTS models show that STM achieves competitive or superior protection performance with substantially better perceptual quality and up to $45.3\times$ speedup over existing white-box baselines. These results demonstrate the effectiveness of frequency-domain perturbations with perceptual constraints as a practical paradigm for protecting against voice cloning.
#### Frequency-Weighted Soft-Constrained Spatially Selective Active Noise Control for Open-Fitting Hearables
 - **Authors:** Tong Xiao, Reinhild Roden, Matthias Blau, Simon Doclo
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Signal Processing (eess.SP)
 - **Arxiv link:** https://arxiv.org/abs/2610.00721

 - **Pdf link:** https://arxiv.org/pdf/2610.00721

 - **Abstract**
 Recently, soft-constrained spatially selective active noise control (SSANC) for open-fitting hearables has been proposed to enable a flexible trade-off between noise reduction and desired-speech preservation using a scalar trade-off parameter across all frequencies. This paper generalizes soft-constrained SSANC by introducing frequency-dependent weighting of the desired-speech preservation error while maintaining low-delay time-domain active control. Frequency-dependent weightings based on the speech intelligibility index (SII), the modified intermediate reference system (MIRS), the long-term average speech spectrum (LTASS), and a signal-dependent oracle are compared with scalar weighting using measured acoustic impulse responses in a multi-speaker scenario. The results show that LTASS and oracle weightings provide the largest speech-intelligibility improvements and comparable speech-quality improvements at substantially lower distortion levels than those obtained with SII and MIRS weightings. While the oracle weighting provides the best overall perceptual trade-off, LTASS weighting provides a comparable perceptual trade-off without requiring the clean desired-speech signal.
#### Articulatory Source-Filter TTS: Physically Grounded Control through Vocal Tract Kinematics
 - **Authors:** Jesuraj Bandekar, Shinji Watanabe, Prasanta Kumar Ghosh
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2610.00735

 - **Pdf link:** https://arxiv.org/pdf/2610.00735

 - **Abstract**
 Modern neural text-to-speech (TTS) systems achieve remarkable acoustic fidelity but act as black boxes, offering little interpretable control over the vocal tract filter. We propose a controllable source-filter TTS architecture grounded in articulatory kinematics. An Acoustic-to-Articulatory Inversion (AAI) model, enhanced by large-scale pretrained representations, generates kinematic pseudo-trajectories for a large TTS corpus. These trajectories condition the filter response, while predicted pitch and energy contours parameterise the glottal source. Source and filter are predicted independently, and the source is refined by an Optimal Transport Conditional Flow Matching (OT-CFM) module before recombination into the final spectrogram. Our model achieves intelligibility and naturalness competitive with similarly sized baselines, with only a modest spectral fidelity cost in exchange for explicit control. Evaluations reveal clear source-filter disentanglement, enabling stable prosodic scaling and cross-speaker source/filter recombination, where F0 remains tied to the source speaker while vocal-tract characteristics follow the filter speaker. Finally, direct manipulation of articulatory trajectories enables fine-grained control, such as accent modification, offering a new direction for interpretable speech synthesis. Audio samples: this https URL
#### ParaCalib: Semantically Calibrated Paralinguistic Modeling for Depression Detection
 - **Authors:** Yuxin Li, Yifei Li, Yi-Wen Chao, Xiangyu Zhang, Eng Siong Chng, Cuntai Guan
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2610.01103

 - **Pdf link:** https://arxiv.org/pdf/2610.01103

 - **Abstract**
 Vocal behavior provides important signals for speech-based depression detection, but its interpretation often depends on what is being said and how it functions in context. However, existing methods typically treat acoustic cues as context-independent markers, making it difficult to distinguish vocal form from its context-dependent communicative function. We propose ParaCalib, a framework that semantically calibrates paralinguistic behavior by interpreting vocal patterns relative to utterance-level semantic context and inferred communicative function and representing them in a comparable state space. Concretely, ParaCalib uses an Audio-Language Model (ALM) to generate contextualized vocal descriptions and an LLM-based Paralinguistic State Extractor (PSE) to map these descriptions into a structured Semantically Calibrated Paralinguistic (SC-Para) representation. ParaCalib achieves the highest mean Macro-F1 among the evaluated methods, reaching 71.9\% on DAIC-WOZ and 90.5\% on MODMA. Our controlled analysis provides direct evidence of semantic calibration at the PSE stage: under a fixed caption-level acoustic description, varying the accompanying semantic context changes the inferred depression-related paralinguistic evidence. Our exploratory analysis further identifies recurring configurations of paralinguistic states associated with depression labels rather than a single uniformly dominant state.
#### FedCFM: Federated Continual Domain Generalization for Fake Speech Detection via Conditional Flow Matching
 - **Authors:** Yingjian Yu, Haiyan Guo, Tianshun Wang, Xinzhou Xu, Zirui Ge, Chi Liu, Ziheng Liu
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2610.01242

 - **Pdf link:** https://arxiv.org/pdf/2610.01242

 - **Abstract**
 The generalization ability of Fake Speech Detection (FSD) models is crucial for real-world deployment. Existing multi-dataset co-training methods rely on fixed training sets and cannot adapt to emerging spoofing types. Although con-tinual learning has been explored, many approaches overlook limited data storage at individual devices, thereby restricting practical applicability. To address this, we propose FedCFM, a Federated continual domain generalization framework via Conditional Flow Matching (CFM) for collaboration without sharing raw speech data across distributed clients facing diverse and evolving spoofing attacks. Each client trains a CFM-based generator to model spoof-type-specific embedding distributions, and cross-client generator exchange enables synthesis of unseen spoof-type embeddings for continual classifier updating through generative replay and knowledge distillation. With the same training datasets, FedCFM achieves lower EER than the eval-uated centralized and federated domain generalization baselines, demonstrating strong cross-domain generalization. Code will be released on this https URL.
#### A Federated Deepfake Speech Detection Method Based on Layer-Wise Center-Guided Weighting Aggregation
 - **Authors:** Yingjian Yu, Haiyan Guo, Tianshun Wang, Zirui Ge, Chi Liu
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2610.01259

 - **Pdf link:** https://arxiv.org/pdf/2610.01259

 - **Abstract**
 The advancement of deep learning-based speech synthesis has significantly increased the diversity of deepfake speech, posing threats to voice authentication. While centralized training is effective for deepfake speech detection (DSD), it requires considerable computational resources and raises privacy concerns. To address these issues, we propose a Federated DSD (FedDSD) method that enables collaborative model training across decentralized speech datasets without sharing raw audio. Specifically, each client trains a local model using the FedProx algorithm to mitigate the effects of data heterogeneity and uploads model parameters to a central server. To improve global model aggregation, we further propose a layer-wise center-guided weighting aggregation (L-CGWA) strategy that adjusts each client's contribution per layer based on its distance to a reference center, capturing inter-client and inter-layer discrepancies and enhancing the robustness of model aggregation. Experimental results demonstrate that models trained under the proposed FedDSD method achieve equal error rates (EERs) comparable to those obtained via centralized co-training, while significantly out-performing models trained on individual corpora. Furthermore, the proposed FedDSD method demonstrates robust generalization capabilities across diverse cross-domain datasets.
#### Code-Switching Spoken Language Identification as Multi-Label Set Prediction
 - **Authors:** Shunsuke Mitsumori, Matthew Wiesner, Shigeo Morishima, Shinji Watanabe
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Computation and Language (cs.CL)
 - **Arxiv link:** https://arxiv.org/abs/2610.01450

 - **Pdf link:** https://arxiv.org/pdf/2610.01450

 - **Abstract**
 Code-switched (CS) speech leaks through the monolingual language identification (LID) filters used to curate massive speech corpora, calling for CS-aware LID (CS-LID). We formulate utterance-level CS-LID as multi-label language-set prediction and propose a set generator that directly outputs the languages in an utterance, comparing it against atomic-pair and score-based classification baselines. Oracle Top-k is the strongest baseline, but thresholding fails because no single threshold separates CS from monolingual speech. Our set generator predicts the correct language count on unseen pairs without assuming the number of languages, but underperforms oracle Top-k in exact set accuracy. Our analysis identifies the key obstacles to robust CS-LID: oracle cardinality, threshold instability, language bias in CS training data, and the synthetic-to-real gap.
#### Teaching LLMs to Hear Who Spoke What: Metadata-Supervised Pretraining for Encoder-Free Speech-LLMs
 - **Authors:** Mohan Shi, Ruchao Fan, Sunit Sivasankaran, Keqi Deng, Jinyu Li
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2610.01695

 - **Pdf link:** https://arxiv.org/pdf/2610.01695

 - **Abstract**
 Encoder-based speech large language models (Speech-LLMs) commonly employ pretrained speech encoders that prioritize linguistic content but may discard fine-grained acoustic cues essential for speaker discrimination and paralinguistic understanding. Encoder-free Speech-LLMs instead map Mel-spectrogram features directly into the LLM input space through lightweight embedding layers, enabling the LLM to learn from low-level acoustic features. However, systematic pretraining strategies for encoder-free Speech-LLMs remain underexplored, limiting their ability to compensate for the absence of large-scale pretrained speech encoders. We propose metadata-supervised pretraining (MSP), which leverages speech attributes such as speaker identity and emotion to develop speaker-discriminative and paralinguistic capabilities. We further introduce speaker-aware utterance composition (SAUC) to strengthen speaker discrimination and apply random span masking to regularize pretraining. We primarily evaluate our approach on joint ASR and speaker diarization in multi-speaker conversations, complemented by experiments on paralinguistic speech-understanding tasks. Under matched training-data conditions, our encoder-free model outperforms its randomly initialized encoder-based counterpart. With limited metadata-annotated data, it is competitive with models using speech encoders pretrained on substantially larger corpora, outperforming them in several settings. These results demonstrate the potential of encoder-free architectures for building native multimodal LLMs that acquire diverse speech capabilities.
#### Shared-State Local Translations for Training-Free Voice Conversion
 - **Authors:** Yangyang Qu, Michele Panariello, Massimiliano Todisco, Nicholas Evans
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2610.01952

 - **Pdf link:** https://arxiv.org/pdf/2610.01952

 - **Abstract**
 In one-shot training-free voice conversion (VC), the source and reference utterances may contain different linguistic content, so reliable frame-level correspondence between them cannot be assumed. We propose StateVC, which jointly defines a common set of local regions from pooled frame-level WavLM representations of the source and reference utterances; we refer to these regions as states. These shared states are obtained by fitting a pair-specific Gaussian mixture model to the pooled representations, without explicit source--reference frame matching. Within each state, StateVC estimates a source-to-reference mean shift in the original WavLM space. Source-frame posterior probabilities then combine the state-specific shifts so that different frames can receive different local updates. For the LibriSpeech one-shot protocol, StateVC achieves the lowest word error rate (WER) and character error rate (CER) among the evaluated systems, at 8.01% and 3.22%, respectively, with a speaker similarity (SIM) of 0.9512. It also achieves the highest mean perceived speaker similarity among the evaluated systems and the highest mean naturalness among the evaluated training-free systems.
#### Multi-sample Synthetic Supervision for Accent Conversion
 - **Authors:** Yangyang Qu, Michele Panariello, Massimiliano Todisco, Nicholas Evans
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2610.01961

 - **Pdf link:** https://arxiv.org/pdf/2610.01961

 - **Abstract**
 Accent conversion (AC) requires changing accent while preserving speaker identity and linguistic content, yet parallel recordings are scarce. Speech synthesis provides an alternative source of supervision, but generated targets vary in accent realization and source preservation. We propose a multi-sample synthetic supervision framework that constructs conversion targets by jointly assessing these properties across candidate waveforms for each source--accent condition. Selected candidates provide associated discrete speech codes, representing linguistic content, prosody, and speaking style, as targets for accent-conditioned autoregressive adaptation. Compared with using one generated target per training example, our method improves target-accent classification accuracy by 2.49 percentage points, while speaker similarity remains nearly unchanged, and word error rate increases by 0.29 percentage points. Across six target accents, our method achieves 7.81% WER and the highest mean listening ratings among evaluated systems.
#### Child-Adapted Structured Phonological Representations for Interpretable Speech Sound Analysis
 - **Authors:** Abner Hernandez, Tomás Arias Vergara, Andreas Maier, Paula Andrea Pérez-Toro
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2610.00852

 - **Pdf link:** https://arxiv.org/pdf/2610.00852

 - **Abstract**
 Structured phonological representations provide an interpretable alternative to generic speech embeddings, but existing models are largely trained on adult speech. We adapt PhonoQ-2.0 to child speech using CHILDES-Aligned data and compare three alignment-supervision conditions (Adult, Adult+Child, and Child-only) across two initialization strategies (Adult PhonoQ and scratch). Generalization is evaluated against manual child-speech annotations. On 1,352 consonant targets from 58 typically developing children, child-speech adaptation improves voicing recognition across all supervision conditions, from 0.922 macro-F1 for Adult PhonoQ to 0.972--0.987 after adaptation. Manner is more sensitive to alignment supervision: Adult+Child MFA reaches 0.804 and 0.796, compared to approximately 0.70 under Adult MFA supervision. Place remains comparatively strong across systems (0.871--0.902), although per-class performance varies substantially. The velar-fronting contrast is preserved across all seven model variants. Longitudinal UltraPhonix analysis further reveals speaker-specific velar and post-alveolar changes that are largely preserved across models and broadly consistent with reported clinical progress.
#### Watch Your Speech: Text-aware Video-to-Speech Synthesis with Textual Conditioning
 - **Authors:** Gunwoo Lee, Yoori Oh, Yoseob Han
 - **Subjects:** Subjects:
Computer Vision and Pattern Recognition (cs.CV); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2610.01012

 - **Pdf link:** https://arxiv.org/pdf/2610.01012

 - **Abstract**
 Video-to-speech synthesis aims to generate natural-sounding speech from silent talking-face videos while ensuring phonetic accuracy. A fundamental challenge in this task is the inherent one-to-many mapping problem, where visual dynamics often lack sufficient information to uniquely determine the corresponding utterance. To address this, we propose Watch Your Speech (WYS), a video-to-speech synthesis framework that incorporates textual conditioning as an explicit linguistic cue to mitigate visual ambiguity. Our framework features an attention-based embedding fusion module that synergistically integrates textual context with video sequences, coupled with a conditional flow matching objective for high-fidelity speech generation. Extensive experiments on the LRS2 and LRS3 datasets demonstrate that WYS achieves superior performance, establishing new state-of-the-art results in audio-visual synchronization (LSE-C/D) while maintaining highly competitive textual accuracy (WER). Subjective evaluations further confirm that our model generates speech with near-human naturalness, validating the effectiveness of textual conditioning in content-controlled video-to-speech synthesis. Project page: this https URL
#### Q-SPT: Learnable Query-Based Compression for Low-Frame-Rate Speech Tokenization
 - **Authors:** Jeeyoung Yun, Seohwan Yun, Sungwoong Kim
 - **Subjects:** Subjects:
Sound (cs.SD); Computation and Language (cs.CL); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2610.01492

 - **Pdf link:** https://arxiv.org/pdf/2610.01492

 - **Abstract**
 Neural speech codecs increasingly serve as tokenizers for speech language models (SLMs). Lowering the frame rate reduces the computational and memory costs of SLMs, but makes it difficult to preserve both linguistic information and acoustic detail. Existing approaches rely on rule-based compression: average pooling can discard linguistic information, whereas similarity-based merging uses a fixed threshold on adjacent-frame similarity and applies the resulting boundaries to the acoustic stream. We propose Q-SPT, a low-frame-rate dual-stream speech tokenizer with separate, context-aware, learnable query-based compressors specialized for semantic and acoustic representations. In particular, queries at a fixed rate independently attend to the semantic and acoustic streams as separate key-value sources, enabling stream-specific, context-aware aggregation through two separately learned compressors. In addition, an autoregressive text loss explicitly supervises the semantic compressor to preserve linguistic information. Experimental results show that Q-SPT achieves the best reconstruction among the evaluated codecs at the same frame rate. In downstream SLMs, it yields the best speech recognition accuracy and text-to-speech perceptual quality with competitive intelligibility.
#### Beyond Decodability: Do Acoustic Factors Drive Predictions in Speech-Based Alzheimer's Assessment?
 - **Authors:** Serli Kopar, Alkis Koudounas, Roshan P. Rane, Sam Gijsen, Paula A. Perez-Toro, Kerstin Ritter
 - **Subjects:** Subjects:
Sound (cs.SD); Computation and Language (cs.CL); Machine Learning (cs.LG); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2610.01846

 - **Pdf link:** https://arxiv.org/pdf/2610.01846

 - **Abstract**
 Speech-based Alzheimer's disease (AD) assessments increasingly rely on pretrained self-supervised learning (SSL) models that learn acoustic representations directly from raw audio, exposing the model to recording factors. We ask whether such factors are merely encoded in SSL representations or can systematically alter predictions. Using ADReSSo and three large SSL backbones, we apply controlled noise and reverberation interventions to participant-speech-only, non-speech, and full-recording audio. We combine layer-wise linear decoding, input- and representation-space interventions, and geometric alignment analysis to distinguish acoustic decodability from influence on AD prediction. Our results show that controlled acoustic interventions alter AD predictions across all three SSL backbones. Noise, despite showing no significant diagnostic-group difference in the original data, produces the strongest intervention effects. Importantly, these effects are systematically structured relative to the classifier's decision direction, replicate on the held-out test set and reverse when the representation-space intervention direction is reversed. Together, these findings show that high predictive performance and the absence of a significant diagnostic-group difference in a measured acoustic factor are not sufficient for robustness. We argue that intervention-based robustness tests should become standard for trustworthy clinical speech models.


by Zyzzyva0381 (Windy). 


2026-10-02
