# Showing new listings for Thursday, 19 March 2026
Auto update papers at about 2:30am UTC (10:30am Beijing time) every weekday.


阅读 `Usage.md`了解如何使用此repo实现个性化的Arxiv论文推送

See `Usage.md` for instructions on how to personalize the repo. 


Keyword list: ['text-to-speech', 'text to speech', 'tts', 'LLM-based', 'speech', 'voice']


Excluded: []


### Today: 14papers 
#### Synthetic Data Domain Adaptation for ASR via LLM-based Text and Phonetic Respelling Augmentation
 - **Authors:** Natsuo Yamashita, Koichi Nagatsuka, Hiroaki Kokubo, Kota Dohi, Tuan Vu Ho
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2603.16920

 - **Pdf link:** https://arxiv.org/pdf/2603.16920

 - **Abstract**
 End-to-end automatic speech recognition often degrades on domain-specific data due to scarce in-domain resources. We propose a synthetic-data-based domain adaptation framework with two contributions: (1) a large language model (LLM)-based text augmentation pipeline with a filtering strategy that balances lexical diversity, perplexity, and domain-term coverage, and (2) phonetic respelling augmentation (PRA), a novel method that introduces pronunciation variability through LLM-generated orthographic pseudo-spellings. Unlike conventional acoustic-level methods such as SpecAugment, PRA provides phonetic diversity before speech synthesis, enabling synthetic speech to better approximate real-world variability. Experimental results across four domain-specific datasets demonstrate consistent reductions in word error rate, confirming that combining domain-specific lexical coverage with realistic pronunciation variation significantly improves ASR robustness.
#### Learnable Pulse Accumulation for On-Device Speech Recognition: How Much Attention Do You Need?
 - **Authors:** Yakov Pyotr Shkolnikov
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2603.16922

 - **Pdf link:** https://arxiv.org/pdf/2603.16922

 - **Abstract**
 Self-attention scales quadratically with sequence length, limiting transformer-based speech models on edge devices. We introduce the Learnable Pulse Accumulator (LPA), an O(n) replacement that substitutes key-query dot products with learned gating functions: content-dependent rectangular pulses, periodic windows, and position-dependent basis functions. An MSE diagnostic sweep determines per-layer replacement difficulty and ordering. Replacing 8 of 12 wav2vec2-base layers yields 10.61% word error rate (WER) on LibriSpeech test-clean, +7.24 percentage points (pp) over the 3.37% baseline, with 3.27x speedup at 120s audio on Apple M4 Pro via an optimized MLX inference path. Cross-domain validation on SepFormer speech enhancement shows all 16 intra-chunk attention layers can be replaced without collapse, suggesting the depth wall arises from linguistic computation rather than an LPA limitation. LPA's near-binary gates at inference enable dense GPU computation with no CPU-GPU synchronization, and all operations map to mobile neural accelerators.
#### Beyond Deep Learning: Speech Segmentation and Phone Classification with Neural Assemblies
 - **Authors:** Trevor Adelson, Vidhyasaharan Sethu, Ting Dang
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2603.16923

 - **Pdf link:** https://arxiv.org/pdf/2603.16923

 - **Abstract**
 Deep learning dominates speech processing but relies on massive datasets, global backpropagation-guided weight updates, and produces entangled representations. Assembly Calculus (AC), which models sparse neuronal assemblies via Hebbian plasticity and winner-take-all competition, offers a biologically grounded alternative, yet prior work focused on discrete symbolic inputs. We introduce an AC-based speech processing framework that operates directly on continuous speech by combining three key contributions:(i) neural encoding that converts speech into assembly-compatible spike patterns using probabilistic mel binarisation and population-coded MFCCs; (ii) a multi-area architecture organising assemblies across hierarchical timescales and classes; and (iii) cross-area update schemes for downstream tasks. Applied to two core tasks of boundary detection and segment classification, our framework detects phone (F1=0.69) and word (F1=0.61) boundaries without any weight training, and achieves 47.5% and 45.1% accuracy on phone and command recognition. These results show that AC-based dynamical systems are a viable alternative to deep learning for speech processing.
#### SimulU: Training-free Policy for Long-form Simultaneous Speech-to-Speech Translation
 - **Authors:** Amirbek Djanibekov, Luisa Bentivogli, Matteo Negri, Sara Papi
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Artificial Intelligence (cs.AI); Computation and Language (cs.CL)
 - **Arxiv link:** https://arxiv.org/abs/2603.16924

 - **Pdf link:** https://arxiv.org/pdf/2603.16924

 - **Abstract**
 Simultaneous speech-to-speech translation (SimulS2S) is essential for real-time multilingual communication, with increasing integration into meeting and streaming platforms. Despite this, SimulS2S remains underexplored in research, where current solutions often rely on resource-intensive training procedures and operate on short-form, pre-segmented utterances, failing to generalize to continuous speech. To bridge this gap, we propose SimulU, the first training-free policy for long-form SimulS2S. SimulU adopts history management and speech output selection strategies that exploit cross-attention in pre-trained end-to-end models to regulate both input history and output generation. Evaluations on MuST-C across 8 languages show that SimulU achieves a better or comparable quality-latency trade-off against strong cascaded models. By eliminating the need for ad-hoc training, SimulU offers a promising path to end-to-end SimulS2S in realistic, long-form scenarios.
#### The Voice Behind the Words: Quantifying Intersectional Bias in SpeechLLMs
 - **Authors:** Shree Harsha Bokkahalli Satish, Christoph Minixhofer, Maria Teleki, James Caverlee, Ondřej Klejch, Peter Bell, Gustav Eje Henter, Éva Székely
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Computation and Language (cs.CL); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2603.16941

 - **Pdf link:** https://arxiv.org/pdf/2603.16941

 - **Abstract**
 Speech Large Language Models (SpeechLLMs) process spoken input directly, retaining cues such as accent and perceived gender that were previously removed in cascaded pipelines. This introduces speaker identity dependent variation in responses. We present a large-scale intersectional evaluation of accent and gender bias in three SpeechLLMs using 2,880 controlled interactions across six English accents and two gender presentations, keeping linguistic content constant through voice cloning. Using pointwise LLM-judge ratings, pairwise comparisons, and Best-Worst Scaling with human validation, we detect consistent disparities. Eastern European-accented speech receives lower helpfulness scores, particularly for female-presenting voices. The bias is implicit: responses remain polite but differ in helpfulness. While LLM judges capture the directional trend of these biases, human evaluators exhibit significantly higher sensitivity, uncovering sharper intersectional disparities.
#### Over-the-air White-box Attack on the Wav2Vec Speech Recognition Neural Network
 - **Authors:** Protopopov Alexey
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Machine Learning (cs.LG); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2603.16972

 - **Pdf link:** https://arxiv.org/pdf/2603.16972

 - **Abstract**
 Automatic speech recognition systems based on neural networks are vulnerable to adversarial attacks that alter transcriptions in a malicious way. Recent works in this field have focused on making attacks work in over-the-air scenarios, however such attacks are typically detectable by human hearing, limiting their potential applications. In the present work we explore different approaches of making over-the-air attacks less detectable, as well as the impact these approaches have on the attacks' effectiveness.
#### Robust Nasality Representation Learning for Cleft Palate-Related Velopharyngeal Dysfunction Screening in Real-World Settings
 - **Authors:** Weixin Liu, Bowen Qu, Amy Stone, Maria E. Powell, Shama Dufresne, Stephane Braun, Izabela Galdyn, Michael Golinko, Bradley Malin, Zhijun Yin, Matthew E. Pontell
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2603.17383

 - **Pdf link:** https://arxiv.org/pdf/2603.17383

 - **Abstract**
 Velopharyngeal dysfunction (VPD) is characterized by inadequate velopharyngeal closure during speech and often causes hypernasality and reduced intelligibility. Although speech-based machine learning models can perform well under standardized clinical recording conditions, their performance often drops in real-world settings because of domain shift caused by differences in devices, channels, noise, and room acoustics. To improve robustness, we propose a two-stage framework for VPD screening. First, a nasality-focused speech representation is learned by supervised contrastive pre-training on an auxiliary corpus with phoneme alignments, using oral-context versus nasal-context supervision. Second, the encoder is frozen and used with lightweight classifiers on 0.5-second speech chunks, whose probabilities are aggregated to produce recording-level decisions with a fixed threshold. On an in-domain clinical cohort of 82 subjects, the proposed method achieved perfect recording-level screening performance (macro-F1 = 1.000, accuracy = 1.000). On a separate out-of-domain set of 131 heterogeneous public Internet recordings, large pretrained speech representations degraded substantially, while MFCC was the strongest baseline (macro-F1 = 0.612, accuracy = 0.641). The proposed method achieved the best out-of-domain performance (macro-F1 = 0.679, accuracy = 0.695), improving on the strongest baseline under the same evaluation protocol. These results suggest that learning a nasality-focused representation before clinical classification can reduce sensitivity to recording artifacts and improve robustness for deployable speech-based VPD screening.
#### Multi-Source Evidence Fusion for Audio Question Answering
 - **Authors:** Aivo Olev, Tanel Alumäe
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Computation and Language (cs.CL)
 - **Arxiv link:** https://arxiv.org/abs/2603.17822

 - **Pdf link:** https://arxiv.org/pdf/2603.17822

 - **Abstract**
 Large audio language models (LALMs) can answer questions about speech, music, and environmental sounds, yet their internal reasoning is largely opaque and difficult to validate. We describe TalTech's solution to the Agent Track of the Interspeech 2026 Audio Reasoning Challenge, in which systems are evaluated on reasoning process quality, specifically the factual accuracy, logical soundness, and completeness of their reasoning chains. Our multi-source ensemble pipeline uses two LALMs that generate independent observations, while a separate text-only reasoning model cross-checks these against outputs from 25 acoustic tools organized into reliability tiers. By grounding every inference step in explicit, reliability-tagged evidence, the system produces dense, verifiable reasoning chains. Our system ranked first in the challenge, outperforming all competing systems by a wide margin in challenge's reasoning quality metric.
#### The Silent Thought: Modeling Internal Cognition in Full-Duplex Spoken Dialogue Models via Latent Reasoning
 - **Authors:** Donghang Wu, Tianyu Zhang, Yuxin Li, Hexin Liu, Chen Chen, Eng Siong Chng, Yoshua Bengio
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Computation and Language (cs.CL)
 - **Arxiv link:** https://arxiv.org/abs/2603.17837

 - **Pdf link:** https://arxiv.org/pdf/2603.17837

 - **Abstract**
 During conversational interactions, humans subconsciously engage in concurrent thinking while listening to a speaker. Although this internal cognitive processing may not always manifest as explicit linguistic structures, it is instrumental in formulating high-quality responses. Inspired by this cognitive phenomenon, we propose a novel Full-duplex LAtent and Internal Reasoning method named FLAIR that conducts latent thinking simultaneously with speech perception. Unlike conventional "thinking" mechanisms in NLP, which require post-hoc generation, our approach aligns seamlessly with spoken dialogue systems: during the user's speaking phase, it recursively feeds the latent embedding output from the previous step into the next step, enabling continuous reasoning that strictly adheres to causality without introducing additional latency. To enable this latent reasoning, we design an Evidence Lower Bound-based objective that supports efficient supervised finetuning via teacher forcing, circumventing the need for explicit reasoning annotations. Experiments demonstrate the effectiveness of this think-while-listening design, which achieves competitive results on a range of speech benchmarks. Furthermore, FLAIR robustly handles conversational dynamics and attains competitive performance on full-duplex interaction metrics.
#### Rubric-Guided Fine-tuning of SpeechLLMs for Multi-Aspect, Multi-Rater L2 Reading-Speech Assessment
 - **Authors:** Aditya Kamlesh Parikh, Cristian Tejedor-Garcia, Catia Cucchiarini, Helmer Strik
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Artificial Intelligence (cs.AI); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2603.16889

 - **Pdf link:** https://arxiv.org/pdf/2603.16889

 - **Abstract**
 Reliable and interpretable automated assessment of second-language (L2) speech remains a central challenge, as large speech-language models (SpeechLLMs) often struggle to align with the nuanced variability of human raters. To address this, we introduce a rubric-guided reasoning framework that explicitly encodes multi-aspect human assessment criteria: accuracy, fluency, and prosody, while calibrating model uncertainty to capture natural rating variability. We fine-tune the Qwen2-Audio-7B-Instruct model using multi-rater human judgments and develop an uncertainty-calibrated regression approach supported by conformal calibration for interpretable confidence intervals. Our Gaussian uncertainty modeling and conformal calibration approach achieves the strongest alignment with human ratings, outperforming regression and classification baselines. The model reliably assesses fluency and prosody while highlighting the inherent difficulty of assessing accuracy. Together, these results demonstrate that rubric-guided, uncertainty-calibrated reasoning offers a principled path toward trustworthy and explainable SpeechLLM-based speech assessment.
#### Quantizer-Aware Hierarchical Neural Codec Modeling for Speech Deepfake Detection
 - **Authors:** Jinyang Wu, Zihan Pan, Qiquan Zhang, Sailor Hardik Bhupendra, Soumik Mondal
 - **Subjects:** Subjects:
Sound (cs.SD); Artificial Intelligence (cs.AI); Computation and Language (cs.CL); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2603.16914

 - **Pdf link:** https://arxiv.org/pdf/2603.16914

 - **Abstract**
 Neural audio codecs discretize speech via residual vector quantization (RVQ), forming a coarse-to-fine hierarchy across quantizers. While codec models have been explored for representation learning, their discrete structure remains underutilized in speech deepfake detection. In particular, different quantization levels capture complementary acoustic cues, where early quantizers encode coarse structure and later quantizers refine residual details that reveal synthesis artifacts. Existing systems either rely on continuous encoder features or ignore this quantizer-level hierarchy. We propose a hierarchy-aware representation learning framework that models quantizer-level contributions through learnable global weighting, enabling structured codec representations aligned with forensic cues. Keeping the speech encoder backbone frozen and updating only 4.4% additional parameters, our method achieves relative EER reductions of 46.2% on ASVspoof 2019 and 13.9% on ASVspoof5 over strong baselines.
#### CineSRD: Leveraging Visual, Acoustic, and Linguistic Cues for Open-World Visual Media Speaker Diarization
 - **Authors:** Liangbin Huang, Xiaohua Liao, Chaoqun Cui, Shijing Wang, Zhaolong Huang, Yanlong Du, Wenji Mao
 - **Subjects:** Subjects:
Computer Vision and Pattern Recognition (cs.CV); Artificial Intelligence (cs.AI); Multimedia (cs.MM); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2603.16966

 - **Pdf link:** https://arxiv.org/pdf/2603.16966

 - **Abstract**
 Traditional speaker diarization systems have primarily focused on constrained scenarios such as meetings and interviews, where the number of speakers is limited and acoustic conditions are relatively clean. To explore open-world speaker diarization, we extend this task to the visual media domain, encompassing complex audiovisual programs such as films and TV series. This new setting introduces several challenges, including long-form video understanding, a large number of speakers, cross-modal asynchrony between audio and visual cues, and uncontrolled in-the-wild variability. To address these challenges, we propose Cinematic Speaker Registration & Diarization (CineSRD), a unified multimodal framework that leverages visual, acoustic, and linguistic cues from video, speech, and subtitles for speaker annotation. CineSRD first performs visual anchor clustering to register initial speakers and then integrates an audio language model for speaker turn detection, refining annotations and supplementing unregistered off-screen speakers. Furthermore, we construct and release a dedicated speaker diarization benchmark for visual media that includes Chinese and English programs. Experimental results demonstrate that CineSRD achieves superior performance on the proposed benchmark and competitive results on conventional datasets, validating its robustness and generalizability in open-world visual media settings.
#### Collecting Prosody in the Wild: A Content-Controlled, Privacy-First Smartphone Protocol and Empirical Evaluation
 - **Authors:** Timo K. Koch, Florian Bemmann, Ramona Schoedel, Markus Buehner, Clemens Stachl
 - **Subjects:** Subjects:
Human-Computer Interaction (cs.HC); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2603.17061

 - **Pdf link:** https://arxiv.org/pdf/2603.17061

 - **Abstract**
 Collecting everyday speech data for prosodic analysis is challenging due to the confounding of prosody and semantics, privacy constraints, and participant compliance. We introduce and empirically evaluate a content-controlled, privacy-first smartphone protocol that uses scripted read-aloud sentences to standardize lexical content (including prompt valence) while capturing natural variation in prosodic delivery. The protocol performs on-device prosodic feature extraction, deletes raw audio immediately, and transmits only derived features for analysis. We deployed the protocol in a large study (N = 560; 9,877 recordings), evaluated compliance and data quality, and conducted diagnostic prediction tasks on the extracted features, predicting speaker sex and concurrently reported momentary affective states (valence, arousal). We discuss implications and directions for advancing and deploying the protocol.
#### Neuron-Level Emotion Control in Speech-Generative Large Audio-Language Models
 - **Authors:** Xiutian Zhao, Ismail Rasim Ulgen, Philipp Koehn, Björn Schuller, Berrak Sisman
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2603.17231

 - **Pdf link:** https://arxiv.org/pdf/2603.17231

 - **Abstract**
 Large audio-language models (LALMs) can produce expressive speech, yet reliable emotion control remains elusive: conversions often miss the target affect and may degrade linguistic fidelity through refusals, hallucinations, or paraphrase. We present, to our knowledge, the first neuron-level study of emotion control in speech-generative LALMs and demonstrate that compact emotion-sensitive neurons (ESNs) are causally actionable, enabling training-free emotion steering at inference time. ESNs are identified via success-filtered activation aggregation enforcing both emotion realization and content preservation. Across three LALMs (Qwen2.5-Omni-7B, MiniCPM-o 4.5, Kimi-Audio), ESN interventions yield emotion-specific gains that generalize to unseen speakers and are supported by automatic and human evaluation. Controllability depends on selector design, mask sparsity, filtering, and intervention strength. Our results establish a mechanistic framework for training-free emotion control in speech generation.


by Zyzzyva0381 (Windy). 


2026-03-19
