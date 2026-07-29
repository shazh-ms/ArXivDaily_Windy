# Showing new listings for Wednesday, 29 July 2026
Auto update papers at about 2:30am UTC (10:30am Beijing time) every weekday.


阅读 `Usage.md`了解如何使用此repo实现个性化的Arxiv论文推送

See `Usage.md` for instructions on how to personalize the repo. 


Keyword list: ['text-to-speech', 'text to speech', 'tts', 'LLM-based', 'speech', 'voice']


Excluded: []


### Today: 8papers 
#### Towards Operational Conversational Intelligence: A Speech Intelligence Framework
 - **Authors:** C. Vishnoi, S. Khurana, A. Timmapur, S. Rai, S. Mohanty
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2607.24958

 - **Pdf link:** https://arxiv.org/pdf/2607.24958

 - **Abstract**
 Body-worn camera (BWC) audio presents unique challenges including high ambient noise, variable recording conditions, and multiple overlapping speakers that make automated transcription and speaker labeling challenging. We propose a dual-path conversational intelligence framework that preprocesses raw BWC audio, separates the processing pipeline into a diarization branch and an ASR branch, and fuses their outputs. The diarization branch uses a denoising front-end (DeepFilterNet), voice activity detection (VAD), and NVIDIA's Multi-Scale Speaker Diarization Decoder (MSDD) with TitaNet embeddings. The transcription branch uses loudness normalization and WhisperX (Large-v3) with forced alignment and probability-guided speech segmentation. Finally, word-level speaker attribution is performed by assigning each recognized word to the speaker segment with the greatest temporal overlap. We evaluate the proposed framework on a curated body-worn camera dataset constructed from publicly available U.S. and U.K. police body-worn camera recordings. Experimental results demonstrate that task-specific acoustic conditioning and probability-guided speech segmentation improve speaker diarization, transcription, and word-level speaker attribution under challenging body-worn camera recording conditions. The proposed modular architecture provides an extensible foundation for future speaker-aware conversational intelligence systems.
#### Multi-Phonation Graph Learning with Self-Supervised Speech Embeddings for ALS Detection and Progression Prediction
 - **Authors:** Behrad TaghiBeyglou, Fatemeh Bagheri, Ervin Sejdic
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2607.25284

 - **Pdf link:** https://arxiv.org/pdf/2607.25284

 - **Abstract**
 Amyotrophic lateral sclerosis (ALS) progressively impairs speech motor control, making acoustic analysis a promising biomarker for severity and progression estimation. We propose a subject-level graph framework that aggregates multiple phonation recordings into a unique k-nearest-neighbor graph built from pretrained SSL embeddings of 2s segments. We compare four SSL front-ends (wav2vec 2.0, HuBERT, data2vec-audio, and UniSpeech-SAT) and five graph neural networks (GCN, residual GCN, GAT, GraphSAGE, and GIN) on the SAND dataset tasks (339 participants: 205 ALS, 134 control): 5-class dysarthria severity and 4-class ALSFRS-R progression prediction. On the official validation set, the best configuration (HuBERT+GIN) achieves macro-F$_1$ of 0.73 for Task 1 and 0.69 for Task 2, outperforming SAND validation baselines (0.61 and 0.58). These results highlight the potential of combining GNNs with pretrained cross-lingual speech representations for low-resource ALS detection and progression monitoring.
#### Self-Supervised Audio Representation Learning for Pediatric Asthma Detection in Emergency Care Using Digital Stethoscope Recordings
 - **Authors:** Fatemeh Bagheri, Thalia Pandolfi, Ervin Sejdic, Rohit Mohindra
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2607.25286

 - **Pdf link:** https://arxiv.org/pdf/2607.25286

 - **Abstract**
 Accurate diagnosis of pediatric asthma in emergency departments remains challenging due to overlapping respiratory symptoms, time constraints, and the limited feasibility of pulmonary function testing in young children. This study investigates the feasibility of pediatric asthma detection in the emergency department using breath sound recordings and machine learning. Thirty-second breath sounds were collected from six chest locations in 31 pediatric patients (10 asthmatic, 21 non-asthmatic) and analyzed using pretrained self-supervised speech representation models (HuBERT, WavLM, and Wav2Vec 2.0) for feature extraction, with patient age and sex incorporated into the feature representations. Conventional machine learning classifiers were trained and evaluated using patient-level stratified group 5-fold cross-validation and leave-one-patient-out validation to ensure the generalizability of the findings. Among the evaluated approaches, Wav2Vec 2.0 combined with histogram-based gradient boosting achieved the strongest and most consistent performance, yielding an accuracy of 0.84, sensitivity of 0.80, specificity of 0.86, and F1-score of 0.76 under both evaluation protocols. The consistency of performance across validation strategies suggests promising generalization to unseen patients. These findings suggest that pretrained self-supervised audio representations offer a promising, non-invasive approach for pediatric asthma detection in real-world emergency department settings, where objective respiratory assessment is often limited.
#### faster-enhancer.c: A Dependency-Free int8 Runtime for Streaming Speech Enhancement on Commodity CPUs
 - **Authors:** Gyeongmin Kim
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2607.25350

 - **Pdf link:** https://arxiv.org/pdf/2607.25350

 - **Abstract**
 This is an implementation and measurement study of what it costs to run a streaming speech enhancer on a CPU. We port FastEnhancer-Medium at 48 kHz to faster-enhancer.c, a C runtime with six int8 GEMM tiers selected at initialization, leaving architecture and weights untouched. One Apple M2 core reaches 0.069 real-time factor, against 0.230 for the fp32 ONNX Runtime graph on the same machine, a 3.3x speedup. A Galaxy S23+ (Snapdragon 8 Gen 2) reaches 0.096. The speedup comes from specializing every layer of the runtime around one fixed model. Activation ranges are recomputed per frame, so no calibration set is needed; the k=3 convolutions use Winograd F(2,3); cross-stage state is fp16; the GRU and the dequantization epilogues are fused; and nothing is allocated after startup. Over 824 VoiceBank-DEMAND utterances the engine tracks fp32 to within -0.006 PESQ and -0.08 dB SNR. Speed alone does not settle deployment cost. The enhancer holds a fraction of a core for as long as the microphone is open, so its real-time factor is a duty cycle. A benchmark races through a file; an audio callback does not. Pacing to the 6.67 ms deadline costs 4.2x per frame, saves 49% of the energy, and leaves the cheapest core placement missing 96% of its deadlines. All SIMD tiers within an architecture family emit byte-identical output. The runtime is released as a dependency-free library.
#### Extracting Voice Styles from Frozen TTS Models via Gradient-Based Inverse Optimization
 - **Authors:** Gyeongmin Kim
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2607.25351

 - **Pdf link:** https://arxiv.org/pdf/2607.25351

 - **Abstract**
 Some text-to-speech systems ship a synthesis model and preset style vectors but not the reference encoder that turns audio into such a vector. The model still accepts a style vector; a user with a voice of their own cannot produce one. We solve for that input directly, inverting the released pipeline by gradient descent: every weight stays frozen and only the style vector is optimized, against time-pooled WavLM statistics of one recording. Because the objective discards the time axis, the synthesized text may differ from the recording, so no transcript and no alignment are needed. On 154 speakers from two corpora, ECAPA-TDNN similarity rises from 0.132 to 0.413 and ResNet from 0.099 to 0.401, improving for every speaker; a verifier at its equal-error point accepts 53% of the recovered voices as the target, against 1% for the presets they start from.
#### VAD to the Bone: Ultra-Tiny Speech Activity Detection for Edge Deployment
 - **Authors:** Stephen Bauer, Sheila Seidel, Shanza Iftikhar, Scott Veidenheimer, Gorkem Ulkar
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Machine Learning (cs.LG)
 - **Arxiv link:** https://arxiv.org/abs/2607.25870

 - **Pdf link:** https://arxiv.org/pdf/2607.25870

 - **Abstract**
 Voice activity detection (VAD) triggers downstream speech processing in always-on systems under strict memory, latency, and compute constraints. Recent compact models report strong accuracy but rely on components that are not widely supported: learnable filterbanks, recurrent layers, or non-causal post-processing. We propose kiloVAD, designed for embedded inference using standard Mel features, CNN-only layers, and tunable context/spectral parameters. We introduce per-layer structured pruning with self-distillation and angle-based quantization-aware training (QAT) that outperforms standard QAT by 1-4%. Evaluated per-frame under causal conditions, kiloVAD achieves 0.850 AUC on AVA-Speech with 2.1 k parameters and 200 ms context, establishing a new state of the art for causal, deployment-ready VAD.
#### Depression Markers in Speech: An Approach based on Tract Variables Dynamics
 - **Authors:** Sahar Altalhi, Tanaya Guha, Alessandro Vinciarelli
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Artificial Intelligence (cs.AI)
 - **Arxiv link:** https://arxiv.org/abs/2607.25888

 - **Pdf link:** https://arxiv.org/pdf/2607.25888

 - **Abstract**
 This study identifies new depression biomarkers based on the dynamical properties of tract variables, which represent geometric features describing the configuration of the speech articulators. A key advantage of this approach lies in its ability to quantify aspects of the articulatory process that have not been previously explored in the context of depression, namely predictability, complexity, and randomness. These properties are respectively characterised using the Largest Lyapunov Exponent, the Correlation Dimension, and the Sample Entropy. Thorough experiments were conducted on the Androids Corpus, a publicly available dataset comprising 64 speakers diagnosed with depression by clinicians and 54 control speakers with no reported history of mental health conditions. The results indicate that the proposed biomarkers effectively discriminate between the depressed and control speakers, as evidenced by the high Cliffs delta values across both read and spontaneous speech.
#### CARE: A Multimodal Corpus for Studying Speech and Non-Verbal Communication Across Multiple Medical Conditions
 - **Authors:** David Gimeno-Gómez, Catarina Botelho, Carlos-D. Martínez-Hinarejos, Isabel Trancoso, Alberto Abad
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2607.25903

 - **Pdf link:** https://arxiv.org/pdf/2607.25903

 - **Abstract**
 Automatic analysis of multimodal speech has shown strong potential for computationally detecting and monitoring a wide range of neurological, psychiatric, and respiratory conditions. However, progress in this field is limited by existing publicly accessible datasets, which are often small in scale, focused on a single condition or disease, and primarily speech focused. Moreover, if key confounding variables such as education, medication use, comorbidities, or mood state are insufficiently documented, the reliability and interpretability of computational analyses are further compromised. To address these limitations, we introduce CARE v1.0, a curated multimodal English dataset of approximately 144 hours of short video interviews collected from 612 individuals across 12 medical conditions plus a control cohort. For each video, a comprehensive set of clinically relevant multimodal descriptors is provided, alongside structured metadata covering factors such as medication, life impacts, and expressed emotions. The corpus's breadth and heterogeneity support a wide range of applications, including automatic disease and symptom detection, multimodal modelling of speech and non-verbal behaviour under emotionally charged contexts, and studies of disease trajectories and coping processes.


by Zyzzyva0381 (Windy). 


2026-07-29
