# Showing new listings for Tuesday, 7 July 2026
Auto update papers at about 2:30am UTC (10:30am Beijing time) every weekday.


阅读 `Usage.md`了解如何使用此repo实现个性化的Arxiv论文推送

See `Usage.md` for instructions on how to personalize the repo. 


Keyword list: ['text-to-speech', 'text to speech', 'tts', 'LLM-based', 'speech', 'voice']


Excluded: []


### Today: 23papers 
#### Speaker-Aware Temporal Aggregation Strategies on Segment Representations for Depression Detection in Dyadic Interaction: A Benchmark Study
 - **Authors:** Anisha Pattanayak, Huang-Cheng Chou, Shrikanth Narayanan, Sudarsana Reddy Kadiri
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2607.02904

 - **Pdf link:** https://arxiv.org/pdf/2607.02904

 - **Abstract**
 Speech-based depression detection compresses features from short audio segments into one speaker-level decision, a step called temporal aggregation rarely studied on its own. Most benchmarks fix a single self-supervised encoder and a single hand-picked layer, so a reported gain may reflect the pipeline rather than the aggregation method itself. We introduce DEPOOL, a controlled benchmark that compares six aggregation architectures with six frozen speech backbones on an English and a Mandarin depression corpus, where each configuration learns which backbone layers matter rather than fixing one by hand. Across the resulting 72-configuration grid, a third of configurations collapse into predicting a single class for every speaker, a failure tied to the backbone as much as to the method, and the architecture that is most stable in a single-seed run becomes unreliable when training repeats across seeds. Robustness to backbone and seed, rather than average accuracy across a single pipeline, should be a first-class benchmarking criterion for temporal aggregation in clinical speech.
#### Layer-wise Cross-Lingual Depression Detection from Speech: Analysis with Contrastive Alignment
 - **Authors:** Anisha Pattanayak, Hanie Kang, Huang-Cheng Chou, Shrikanth Narayanan, Sudarsana Reddy Kadiri
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2607.02920

 - **Pdf link:** https://arxiv.org/pdf/2607.02920

 - **Abstract**
 Significant disparities exist in the diagnosis and clinical presentation of depression across different linguistic populations. Speech-based depression detection performs well monolingually, but cross-lingual generalization remains an open challenge. A key reason is that prior work uses segment-level random splits without speaker grouping, leading to identity leakage that inflates reported metrics. We propose CLeaD, a supervised contrastive alignment framework that maps WavLM embeddings from English and Mandarin into a shared clinical space, without parallel data or target-language fine-tuning. Evaluating 52 Mandarin speakers, contrastive alignment modestly outperforms the baseline (F1: 0.640 vs. 0.622) under leave-one-speaker-out evaluation. It also improves depressed-class recall at intermediate layers (7-8), though the small test set limits generalizability. Two findings remain robust: model scaling degrades cross-lingual performance while improving monolingual English, and speaker identity leakage artificially inflated previously reported Mandarin F1 scores to 0.954, an artifact we reproduce and quantify.
#### Open-Set Source Tracing as Compositional Factors via Structured Prototypes
 - **Authors:** Santiago Rubio, Antonio Almudévar, Antonio Miguel, Eduardo Lleida, Alfonso Ortega
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Machine Learning (cs.LG)
 - **Arxiv link:** https://arxiv.org/abs/2607.03134

 - **Pdf link:** https://arxiv.org/pdf/2607.03134

 - **Abstract**
 Recent research expands beyond binary anti-spoofing with the emergence of Source Tracing, the task of identifying the specific generative origins of synthetic speech. However, current research often equates a "source" with its generative architecture. We propose redefining a source as a compositional tuple of Architecture, Training Data, and other training factors affecting the generated speech. We propose a framework using Structured Orthonormal Prototypes to minimize class overlap and intra-class variance. Our Subspace Partitioning strategy splits the embedding into architecture and data subspaces, while a residual subspace captures stochastic variability, enabling "compositional generalization" for novel factor combinations. This approach improves performance for partially seen sources and maintains robustness in fully open-set scenarios. MLAAD evaluations for Few-Shot open-set Identification show our approach significantly outperforms angular-margin baselines.
#### An Intervention-Based Framework for Shortcut Diagnosis in Spoofing Countermeasures
 - **Authors:** Santiago Rubio, Pilar Bello, Dayana Ribas, Antonio Miguel, Eduardo Lleida, Alfonso Ortega
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Machine Learning (cs.LG)
 - **Arxiv link:** https://arxiv.org/abs/2607.03150

 - **Pdf link:** https://arxiv.org/pdf/2607.03150

 - **Abstract**
 While deepfake audio detection systems achieve high performance in controlled benchmarks, their reliability often diminishes in the wild. Prior work shows that dataset-specific artifacts contribute to this gap. Yet, systematic tools to identify which acoustic properties a model exploits as shortcuts remain limited. We propose an intervention-based diagnostic framework, grounded in a directed graphical model, that formally distinguishes confound-driven shortcut dependencies from legitimate domain shift. We operationalise this through controlled acoustic perturbations targeting non-speech structure, spectral content, and signal energy, complemented by corpus-level distributional analysis. Evaluating XLS-R-300M with RawGAT-ST across ASVspoof challenges datasets, we quantify model sensitivity to specific intervention types. Results reveal that non-speech interventions produce the largest performance shifts, confirming non-speech intervals as a dominant shortcut.
#### Deriving Benchmarking Datasets from Long-Form Recordings: Challenges and Opportunities
 - **Authors:** Kaveri K. Sheth, Lawrence Borst, Tarek Kunze, Marvin Lavechin, Okko Räsänen, Sho Tsuji, Loann Peurey, Alix Bourrée, Alejandrina Cristia
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Machine Learning (cs.LG); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2607.03201

 - **Pdf link:** https://arxiv.org/pdf/2607.03201

 - **Abstract**
 Long-form recordings (LFRs) of child-centered audio are ecologically valid sources for studying early language development, but three problems limit their use. First, LFR corpora are collected across sites with heterogeneous formats and consent structures, making cross-corpus use non-trivial. Second, without standardized benchmarks, assessing whether tools generalize across languages and conditions is hard. Third, ML workflows rarely respect privacy constraints governing sensitive child speech. This paper presents a framework addressing all three: a standardized collection of 27 child-centered datasets built with open-source tools (S1); a replicable pipeline for four speech-processing benchmarks (S2); and ELSI, a role-based ecosystem embedding ethical governance into the ML workflow (S3). We demonstrate the framework via a voice type classification case study and show the three solutions are mutually dependent.
#### QuaSR: Quality-Aware Sample Reweighting for Pacific Indigenous Speech Recognition
 - **Authors:** Yishun Li, Yang Xiao, Gongping Huang, Eun-Jung Holden, Nick Thieberger, Ting Dang
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2607.03658

 - **Pdf link:** https://arxiv.org/pdf/2607.03658

 - **Abstract**
 Training automatic speech recognition (ASR) models for low-resource languages is challenging due to limited data and highly variable supervision quality. In particular, Pacific Indigenous speech corpora often exhibit heterogeneous acoustic conditions, transcript inconsistencies, and varying degrees of acoustic-text alignment reliability, making standard fine-tuning approaches sensitive to noisy or misleading supervision signals. In this work, we propose QuaSR, a simple yet effective weighting framework that combines data-side reliability with model-side learnability to improve ASR adaptation. Specifically, we estimate data reliability from acoustic, transcription, and alignment, while measuring learnability using training loss from the model. These two complementary signals are integrated into a unified sample utility score to produce training weights for the samples. We also evaluated across four Pacific Indigenous languages, which shows that the proposed utility scores reliably correlate with adaptation performance. Furthermore, QuaSR consistently improves ASR adaptation over standard fine-tuning and alternative data selection strategies, highlighting a new way to leverage difficulty scores for low-resource speech learning.
#### TRACE-EVC: Text-Guided Relative Affective Control for Zero-Shot Emotional Voice Conversion
 - **Authors:** Zihan Zhang, Shreeram Suresh Chandra, Zongyang Du, Xiutian Zhao, Aurosweta Mahapatra, Hao Zhang, Philipp Koehn, Berrak Sisman
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Signal Processing (eess.SP)
 - **Arxiv link:** https://arxiv.org/abs/2607.03666

 - **Pdf link:** https://arxiv.org/pdf/2607.03666

 - **Abstract**
 Traditional emotional voice conversion (EVC) conditions generation on explicit target emotions like labels or references, defining the target affective state but omitting the direction or nature of the transition. We introduce instruction-guided relative emotional voice conversion, a task where natural-language instructions specify source-conditioned affective transformations (e.g., "make the speech slightly calmer" or "sound noticeably more confident") instead of fixed targets. To support this task, we construct TRACE-Instruct, a dataset of relative emotion instructions covering categorical transitions, intensity modifications, and open-ended affective changes. We propose TRACE-EVC, a zero-shot framework built around Emo-Compass, a module that models each conversion as a source-anchored rectified flow. Rather than conditioning on an explicit target, it predicts the direction and degree of the affective change. Experiments demonstrate that TRACE-EVC accurately follows relative emotion instructions while preserving speaker identity, linguistic content, and speech quality, and remains competitive with conventional EVC systems on standard categorical emotion conversion.
#### CHILDES-Aligned: A Curated Children's Speech Dataset via Multi-Model Timestamp Ensembling
 - **Authors:** Haolong Zheng, Yuanzhuo Hu, Xinyu Liang, Vishal Sunder, Dancheng Liu, Jinjun Xiong, Samuel Thomas, Brian Kingsbury, Zhizheng Wu, Mark A. Hasegawa-Johnson
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Signal Processing (eess.SP)
 - **Arxiv link:** https://arxiv.org/abs/2607.03670

 - **Pdf link:** https://arxiv.org/pdf/2607.03670

 - **Abstract**
 CHILDES is a large-scale child speech corpus containing long-form recordings of naturalistic child-adult interactions, making it a valuable resource for studying child speech and language development. However, utterance-level timestamps provided in this corpus are often noisy, incomplete, or misaligned with the audio. As a result, utterances cannot always be reliably localized within long recordings, which limits the direct use of these data for training and evaluating speech models. In this work, we propose BEACON (Boundary Estimation via Alignment CONsensus), an ensemble timestamp-curation framework that refines utterance-level timestamps by aggregating knowledge from multiple off-the-shelf ASR models. Specifically, each model's word-level timestamp predictions are first aligned to provided human transcripts, and the final utterance time boundaries are determined by a consensus voting strategy. The framework is corpus-agnostic and applies to any long-form recording paired with a trusted transcript whose timestamps are unreliable or missing, offering a general recipe for timestamp curation. Leveraging this pipeline, we curate and release a 413-hour general-purpose child-speech dataset with corrected utterance-level timestamps, together with a 283-hour quality-controlled subset for ASR training. Fine-tuning on this subset yields up to an average 19.5% relative WER reduction on four out-of-domain child-speech benchmarks.
#### Probing Low-Level Acoustic Attribute Encoding in CLAP Audio Embeddings
 - **Authors:** Héctor Martel, Joe Hennessy-Priest, Taemin Cho
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Artificial Intelligence (cs.AI); Signal Processing (eess.SP)
 - **Arxiv link:** https://arxiv.org/abs/2607.03806

 - **Pdf link:** https://arxiv.org/pdf/2607.03806

 - **Abstract**
 Audio foundation models are widely adopted as general-purpose feature extractors, yet the internal structure of their learned representations remains insufficiently understood. In this work, we analyze CLAP audio embeddings through a probing framework, studying the encoding of three fundamental perceptual dimensions: reverberation (RT60), loudness (LUFS), and spectral content, measured via spectral centroid (SC) and relative pitch (RP). Probes of increasing complexity are trained to predict each attribute from frozen embeddings across five datasets spanning noise, speech, monophonic musical notes, and music mixtures. Our primary finding is that all of these attributes are reliably recoverable from the CLAP embedding space across the examined datasets. Within this global picture, two encoding regimes emerge: RT60, LUFS, and RP are approximately linearly encoded, while SC requires non-linear probes. Both regimes generalize across eight additional audio foundation models, with the notable exception that amplitude-invariant architectures discard loudness entirely by construction. The identified linear feature directions are geometrically consistent across datasets for RT60 and LUFS, while highly domain-specific for RP. Finally, we provide a qualitative demonstration of cross-modal consistency, showing that text embeddings of acoustic descriptors align geometrically with the identified RT60 feature direction.
#### NouveauVoice: Generating Novel Pseudo Speakers for Voice Anonymization
 - **Authors:** Meiying Melissa Chen, Anastasia Kuznetsova, Zhenyu Wang, Zhiyao Duan
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Artificial Intelligence (cs.AI)
 - **Arxiv link:** https://arxiv.org/abs/2607.03985

 - **Pdf link:** https://arxiv.org/pdf/2607.03985

 - **Abstract**
 Advanced neural technologies in speech synthesis and voice conversion (VC) have introduced severe risks to personal privacy, necessitating robust Speaker Anonymization Systems (SAS). Existing SAS approaches modify voice characteristics in the hand-crafted feature space or speaker embedding space, often struggling to provide sufficient identity variance across generated voices. In this paper, we propose NouveauVoice, a novel pseudo-speaker generation framework based on a Hierarchical Deep Variational Autoencoder (NVAE). Integrated as a standalone plug-in module on top of state-of-the-art architectures (FACodec and CosyVoice2), our approach leverages tractable sampling and the Evidence Lower Bound (ELBO) objective to synthesize highly expressive pseudo-speaker embeddings with significantly enhanced speaker diversity. Evaluating our framework under a protocol similar to the VoicePrivacy Challenge alongside Maximum Mean Discrepancy (MMD) analysis, we demonstrate that NouveauVoice achieves strong identity concealment, yielding an Equal Error Rate (EER) exceeding 38% against an automatic speaker verification attacker model. Our system shows a reasonable trade-off between strict anonymity, rich pseudo-speaker diversity, and downstream speech utility, such as intelligibility and emotional expressiveness.
#### DELTA-TTS: Adapting Autoregressive Model into Diffusion Language Model for Text-to-Speech
 - **Authors:** Junwon Moon, Seungbeom Kim, Yejin Lee, Hoseong Ahn, Sewoong Park, Heeseung Kim, Kyuhong Shim
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Computation and Language (cs.CL)
 - **Arxiv link:** https://arxiv.org/abs/2607.04140

 - **Pdf link:** https://arxiv.org/pdf/2607.04140

 - **Abstract**
 Autoregressive (AR) text-to-speech (TTS) models generate discrete speech tokens sequentially, which makes inference slow and can degrade robustness by propagating local errors and hallucinations. This limitation stems from their left-to-right AR commitment: each token must be determined before future speech-token context is available. However, such ordering is not an inherent requirement for TTS, as the full input text is available before synthesis. In this paper, we introduce DELTA-TTS, a lightweight LoRA-based adaptation framework that converts a pretrained AR TTS model into a discrete diffusion language model (dLLM) for confidence-ordered speech-token decoding. To better capture the local structure of speech, DELTA-TTS incorporates a convolution module that injects local acoustic context, together with a $1/t$-weighted training objective and a time-shifted inference schedule that defer low-confidence positions to later steps. Trained on only $585$ hours of LibriTTS, DELTA-TTS achieves a $\textbf{1.75}\%$ WER on Seed-TTS test-en, outperforming its AR backbone while generating tokens $\textbf{3.3}\times$ faster. Further analysis shows that DELTA-TTS produces sharper text--speech alignment, increases overall decoding confidence, and mitigates hallucinations observed in AR generation.
#### Noisy Environment Adaptation of Neural Speech Codec via Focal Mask and Noise Feature Separation
 - **Authors:** Shaokai Li, Weiping Tu, Yuhong Yang
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2607.04195

 - **Pdf link:** https://arxiv.org/pdf/2607.04195

 - **Abstract**
 Neural speech codec has attracted extensive attention for high-quality reconstruction at low-bitrate. However, real-world noise severely degrades its performance and hinders high-quality clean speech reconstruction. To tackle this problem, we propose FocalSE, a novel speech enhancement method that performs feature denoising, noise feature separation and noise recognition in the continuous embedding space of neural speech codecs. Specifically, we develop focal modulation-based compression and decompression to capture global context and local mutual information, and generate focal masks to recover clean feature embeddings. We then separate noise embeddings from noisy embeddings to improve denoising performance. Finally, we use ResNet1D-18 to recognize noise categories for better separation effectiveness. Extensive experiments on two standard datasets, LibriTTS and ESC50, demonstrate that our method outperforms state-of-the-art approaches under low-bitrate and low-SNR conditions.
#### MOSAIC: Interpretable Multi-Token Cross-Attention of Biophonetic and Self-Supervised Representations for Unified Voice Anti-Spoofing
 - **Authors:** Yugwon Won
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2607.04314

 - **Pdf link:** https://arxiv.org/pdf/2607.04314

 - **Abstract**
 The dominant trend in voice anti-spoofing fuses self-supervised (SSL) backbones (e.g., WavLM) with handcrafted features, yet such fusion typically lacks transparency in cue-to-layer interactions, and simple concatenation limits cross-modal learning. We propose MOSAIC (Multi-token Oriented Speech Anti-spoofing via Integrated Cross-attention), an interpretable multi-token cross-attention framework that splits a 152-dimensional biophonetic feature vector into six semantic-group query tokens (Praat, phase, LFCC mean/std, sub-band mean/std) and attends them over thirteen mean-std pooled WavLM-Large transformer layers as keys/values. The resulting 6x13 attention matrix visualizes cue-to-layer alignment; a z-score analysis of the per-token activations shows that biophonetic/phase tokens activate more on bona fide speech while spectral/channel tokens activate more on spoofed speech -- yielding per-cue, per-layer attribution that extends prior fusion approaches. Trained jointly with focal loss, a dual LA/PA domain-adversarial classifier, and a bona-fide-only VAE regularizer, MOSAIC attains EER 1.93% / 1.98% on ASVspoof 2019 LA / PA -- a single unified model that approaches the PA-specialized SOTA (LFCC-CMR, 1.34%) while remaining competitive on LA -- and 9.28% / 6.21% / 40.09% on ASVspoof 2021 LA / DF / PA.
#### Weakly Guided and Autoregressive Beamformer Parameterization for Generalizable Moving Speaker Extraction in Higher-Order Ambisonics
 - **Authors:** Jakob Kienegger, Tal Peer, Sina Khanagha, Timo Gerkmann
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Machine Learning (cs.LG); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2607.04471

 - **Pdf link:** https://arxiv.org/pdf/2607.04471

 - **Abstract**
 Linear spatial filters (beamformers) enable robust, generalizable and interpretable speech enhancement with performance guarantees under ideal parameterization. Modern beamformers are often parameterized by deep neural networks, whose performance degrades in dynamic scenarios with multiple moving speakers of unknown directions. We propose a data-driven beamforming pipeline, which only requires an estimate of the target's initial direction. Building on a higher-order ambisonics representation, we show that neural temporal-spectral processing can be decoupled from linear spatial processing, and thereby achieve generalizable and array-agnostic enhancement. By incorporating autoregression into a frame-wise causal framework, we maintain consistent performance throughout fast speaker motion and long recordings. Evaluation on synthetic data demonstrates robust enhancement under challenging conditions with closely spaced and crossing speakers. Real-world recordings in a dynamic office meeting scenario complement these findings and show generalizability across varying ambisonics orders.
#### Ranking the Impact of Contextual Specialization in Neural Speech Enhancement
 - **Authors:** Peter Leer, Svend Feldt, Zheng-Hua Tan, Jan Østergaard, Jesper Jensen
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2607.04826

 - **Pdf link:** https://arxiv.org/pdf/2607.04826

 - **Abstract**
 We systematically investigate neural speech enhancement systems, ranging from very small ($\sim$10\,k parameters) to medium-large ($\sim$2-5\,M parameters), which specialize to acoustic conditions using contextual information such as speaker identity, noise type, speaker gender, spoken language, and SNR. By fine-tuning generalist models on specific data subsets, we find that specializing to a speaker's identity consistently yields the largest gains in estimated speech intelligibility and quality. In contrast, specializing to SNR, noise type, or gender offers only marginal benefits. Crucially, we show that a small model specialized to both a specific speaker and a specific noise type can match or exceed the performance of a generalist model ten times its size. Further, cross-lingual tests reveal that models specialized to a target language outperform multilingual generalists, suggesting that language is a salient feature for specialization. These findings highlight the potential of small, adaptive models for resource-constrained applications like hearing aids, which specialize on-the-fly to contextual information.
#### Towards Language-Agnostic Speech Inversion
 - **Authors:** Saba Tabatabaee, Mark Tiede, Suzanne Boyce, Liran Oren, Carol Espy-Wilson
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2607.05060

 - **Pdf link:** https://arxiv.org/pdf/2607.05060

 - **Abstract**
 Characteristic timing patterns are reflected in the acoustic speech signal, encompassing both vocal tract configuration and acoustic excitation. Previous studies have demonstrated that speech inversion (SI) systems can recover these timing patterns from speech, including oral tract variables (tongue and lip constrictions) and source information such as periodic and aperiodic energies and fundamental frequency. In this study, we develop an SI system that simultaneously estimates oral tract variables and three source information parameters trained on co-recorded American English speech audio and articulatory kinematics and investigate cross-linguistic generalizability by evaluating performance on previously unseen languages. Pearson product-moment correlation scores of 0.83 and 0.74 were achieved on untrained French and Russian respectively, across oral tract variables and source information when comparing estimated data with ground-truth measurements.
#### ProPS: Prompted Profile Synthesis for Natural Language-Conditioned Speaker Embedding Distributions
 - **Authors:** Thomas Thebaud, Junhyeok Lee, Laureano Moro-Velazquez, Jesus Villalba Lopez, Najim Dehak
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Artificial Intelligence (cs.AI)
 - **Arxiv link:** https://arxiv.org/abs/2607.05276

 - **Pdf link:** https://arxiv.org/pdf/2607.05276

 - **Abstract**
 Speaker embeddings, or x-vectors, are widely used to represent speaker identity and speaker-related attributes, but existing embedding extractors are typically descriptive rather than generative: they map an observed speech segment to an x-vector, which is then used for downstream applications. We introduce ProPS, Prompted Profile Synthesis, a framework for generating distributions of speaker embeddings conditioned on natural language prompts such as "a thirties male speaker with an Indian accent". ProPS converts human-written profile descriptions into sentence embeddings and uses a mixture density network trained on a large-scale dataset to predict a Gaussian mixture model in the x-vector space. The model is trained by maximizing the likelihood that real speaker embeddings match the requested profile, and its generated distributions are evaluated by negative log-likelihood on held-out x-vectors and by attribute classification accuracies on sampled synthetic x-vectors. Experiments show that ProPS produces profile-conditioned distributions and generates x-vectors that preserve requested speaker attributes such as age, gender, accent, and prosodic characteristics. This design enables controllable speaker-profile synthesis for speech generation systems like Text-To-Speech (TTS) or Voice Conversion (VC) while anchoring generated distributions in observed speaker-embedding structure.
#### Jointly Improving Dialect Identification and ASR in Indian Languages using Multimodal Feature Fusion
 - **Authors:** Saurabh Kumar, Amartyaveer, Prasanta Kumar Ghosh
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2607.02862

 - **Pdf link:** https://arxiv.org/pdf/2607.02862

 - **Abstract**
 Automatic Speech Recognition (ASR) and Dialect Identification (DID) are crucial for Indian languages, many of which are low-resource and exhibit significant dialectal differences. Existing methods often optimize ASR or DID individually, resulting in performance trade-offs. In this work, we propose a multimodal framework that jointly improves ASR and DID. Our method employs a Bottleneck Encoder to extract dialectal features from Conformer-based speech representations and a RoBERTa encoder to process ASR-generated CTC embeddings. A gating mechanism merges these features, followed by an attention encoder to refine the representations. The learned embeddings are concatenated with Conformer outputs to enhance ASR features. Evaluated on eight Indian languages with thirty-three dialects, our method achieves an average DID accuracy of 81.63% and average CER and WER of 4.65% and 17.73%, respectively. These results highlight the effectiveness of our method for joint ASR-DID modeling.
#### TokAN: Accent Normalization Using Self-Supervised Speech Tokens
 - **Authors:** Qibing Bai, Shuai Wang, Yuhan Du, Bohan Li, Yannan Wang, Haizhou Li
 - **Subjects:** Subjects:
Sound (cs.SD); Artificial Intelligence (cs.AI); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2607.03928

 - **Pdf link:** https://arxiv.org/pdf/2607.03928

 - **Abstract**
 Accent normalization (AN) seeks to convert non-native (L2) accented speech into standard (L1) speech while preserving speaker identity. The current techniques either require naturally recorded parallel L1-L2 speech for training, or suffer from quality degradation when supervised by synthesized targets. In this paper, we present TokAN, a token-based accent normalization framework that operates on self-supervised discrete speech tokens extracted from a L1-L2 jointly trained vector-quantization (VQ) tokenizer, without the need of synthetic supervisory speech. An autoregressive encoder-decoder model performs token-to-token conversion, translating L2-accented token sequences into the tokens of standard voice. We also introduce reinforcement learning (RL) post-training based on Group Relative Policy Optimization (GRPO), using word error rate and accent classifier confidence as complementary rewards. A non-autoregressive flow-matching synthesizer recovers the Mel-spectrogram from the converted tokens, conditioned on the source speaker embedding. We also develop a flow-matching duration predictor that supports total-duration-aware synthesis, making TokAN applicable to duration-critical tasks such as voice dubbing and live casting. Experiments on seven English accents demonstrate that TokAN reduced the word error rate from 12.40% to 9.89% after supervised fine-tuning, and further to 9.23% after RL post-training, consistently outperforming frame-to-frame, direct flow-matching, and prompt-based token-conversion baselines in terms of accent reduction and intelligibility.
#### Speaker-Disentangled Chunk-Wise Regression for Syllabic Tokenization
 - **Authors:** Ryota Komatsu, Kota Kawakita, Takuma Okamoto, Takahiro Shinozaki
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Artificial Intelligence (cs.AI); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2607.04064

 - **Pdf link:** https://arxiv.org/pdf/2607.04064

 - **Abstract**
 Unsupervised syllabic tokenization aims to learn discrete syllabic tokens that capture latent linguistic content-related structure from raw speech. Recent syllabic tokenization methods employ teacher-student distillation of the pretrained HuBERT to organize latent speech frame representations into syllabic segments. However, when trained with an utterance-level cross-entropy objective, the model predicts speaker identity rather than linguistic content, thereby compromising the purity of syllabic tokens. To address this problem, we propose a speaker-disentangled syllabic tokenizer that regresses speaker-perturbed student representations toward clean teacher targets within fixed-length chunks. Experimental results demonstrate that our proposed method achieves state-of-the-art performance in syllable boundary detection and syllabic segment clustering. Moreover, a speech language model trained on our syllabic tokens achieves a 7% relative improvement in syntactic and semantic understanding over the phone-level SpiRit-LM.
#### DuplexChat: Constructing Speaker-Separated Full-Duplex Dialogue Speech at Scale for Spoken Dialogue Language Modeling
 - **Authors:** Wataru Nakata, Yuki Saito, Hiroshi Saruwatari
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2607.04941

 - **Pdf link:** https://arxiv.org/pdf/2607.04941

 - **Abstract**
 Full-duplex spoken dialogue models are trained on conversational speech in which each speaker is represented as a separate stream, but existing large-scale public speech corpora are mostly monaural, making them unsuited for SDLM training. We present DuplexChat, an open-source corpus for full-duplex spoken dialogue models, and DuplexChat-Pipe, a pipeline for constructing speaker-separated full-duplex dialogue speech from public podcast feeds. DuplexChat-Pipe filters language-specific podcast feeds, retrieves and cleans episode audio, extracts diarization-guided two-speaker dialogue clips, and applies speech separation and restoration to produce one channel per speaker. Running this pipeline yields a speaker-separated spoken dialogue corpus covering 282,634 hours of English and 132,723 hours of Japanese. Analysis results on DuplexChat show that it contains turn-taking dynamics present in human dialogues.
#### Unified Audio Intelligence Without Regressing on Text Intelligence
 - **Authors:** Zhifeng Kong, Sang-gil Lee, Jaehyeon Kim, Boxin Wang, Zihan Liu, Sungwon Kim, Yang Chen, Arushi Goel, Rajarshi Roy, Wenliang Dai, Zhuolin Yang, Yangyi Chen, Dongfu Jiang, Sreyan Ghosh, Tuomas Rintamaki, Andrew Tao, Jonathan Raiman, Mohammad Shoeybi, Bryan Catanzaro, Wei Ping
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Artificial Intelligence (cs.AI); Machine Learning (cs.LG); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2607.05196

 - **Pdf link:** https://arxiv.org/pdf/2607.05196

 - **Abstract**
 Audio intelligence involves understanding, reasoning about, and generating both audio and speech. In this work, we introduce Nemotron-Labs-Audex-30B-A3B (Audex), a unified audio-text LLM built on Nemotron-Cascade-2-30B-A3B, a strong text-only MoE LLM. Audex adopts a simple unified design with a single Transformer decoder: audio inputs are encoded and projected into the text embedding space, while text tokens and quantized audio output tokens are treated uniformly during generation. This architecture enables strong audio-text fusion, seamless multimodal generation, and compatibility with standard LLM training and inference infrastructure. For training, we meticulously curate audio-text datasets comprising 157.4B audio tokens and 320.5B text tokens. We apply multi-stage supervised training on these datasets, followed by text-only Cascade RL and multi-domain on-policy distillation. Audex delivers state-of-the-art audio understanding, speech recognition and translation, text-to-speech, audio generation, and speech-to-speech generation, while preserving very compelling reasoning, alignment, knowledge, long-context, and agentic capabilities of its text-only LLM backbone with marginal or no regression. We release the model checkpoints to facilitate open research.
#### SPEARBench: A Benchmark for Naturalness Evaluation in Streaming Speech-to-Speech Language Models
 - **Authors:** Thomas Thebaud, Yuzhe Wang, Hao Zhang, Sathvik Manikantan Napa Ugandhar, Ashish Hallur, Georgi Tinchev, Venkatesh Ravichandran, Laureano Moro-Velazquez
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Artificial Intelligence (cs.AI); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2607.05365

 - **Pdf link:** https://arxiv.org/pdf/2607.05365

 - **Abstract**
 Streaming speech-to-speech language models aim to answer spoken queries directly with synthetic speech. However, standard speech and text benchmarks do not capture whether these systems behave naturally in conversations, where timing, turn-taking, prosody, interpersonal stance, language and dialect consistency, and relationship-aware appropriateness jointly shape perceived quality. We introduce SPEARBench, a benchmark for evaluating naturalness in speech-to-speech language models from question-answer interactions. SPEARBench constructs controlled dialogue prompts from the Seamless Interaction corpus, runs inference across multiple models, and evaluates generated answers using a multidimensional protocol that covers response latency, interruptions, speech quality, ASR robustness, language and dialect consistency, emotional naturalness, interpersonal stance, and explainable distributional baselines. The benchmark includes original human answers as a reference condition and reports results for several contemporary models. Results show that current models can achieve high signal-level quality and low ASR error while still differing from human conversational behavior in latency, overlap, dialect preservation, emotional adaptation, and interpersonal stance dynamics.


by Zyzzyva0381 (Windy). 


2026-07-07
