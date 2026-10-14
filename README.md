# Showing new listings for Wednesday, 14 October 2026
Auto update papers at about 2:30am UTC (10:30am Beijing time) every weekday.


阅读 `Usage.md`了解如何使用此repo实现个性化的Arxiv论文推送

See `Usage.md` for instructions on how to personalize the repo. 


Keyword list: ['text-to-speech', 'text to speech', 'tts', 'LLM-based', 'speech', 'voice']


Excluded: []


### Today: 19papers 
#### RUBATO: Decoupled Clocks Allow Full-Duplex Models to Think as Fast as They Can
 - **Authors:** Yi-Jen Shih, Puyuan Peng, Haohe Liu, David Harwath, Abdelrahman Mohamed
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Artificial Intelligence (cs.AI); Computation and Language (cs.CL); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2610.14299

 - **Pdf link:** https://arxiv.org/pdf/2610.14299

 - **Abstract**
 Recent Speech Full-Duplex (FDX) models are capable of engaging in naturalistic conversation by listening and speaking at the same time. Building on this foundation, recent architectures introduce internal thinking streams to enhance their intelligence. To guarantee these streams process successfully in real time, existing methods enforce a fixed decoding rate, which assumes a hardware profile capable of a specific latency. We argue this static approach is severely suboptimal. Because internal thinking is intrinsically asynchronous, an ideal FDX model should sustain real-time conversational flow while allowing each stream to decode at its own hardware-adaptive rate. Thinking streams must process as fast as the hardware allows to deliver accurate answers faster. To solve this, we propose RUBATO, a novel multistream positional encoding that assigns independent clocks to entirely decouple the streams. This structural isolation empowers the model to zero-shot interpolate across any inference decoding speed. It drastically simplifies dataset construction by requiring training on only two extreme decoding speeds while smoothly adapting to any intermediate rate during deployment. RUBATO demonstrates consistent robustness across five spoken reasoning tasks. On Spoken GSM8K, a baseline model collapses to 1.7% accuracy at interpolated speeds. Conversely, RUBATO-Full sustains 48.7% to 51.1% accuracy across all rates, leveraging faster hardware to achieve a 6.5x shorter time-to-answer with a constant time-to-first-response. This advantage holds on BIG-Bench Audio and amplifies significantly during complex full-duplex interruptions. Specifically, when a user interjects while the model is actively speaking and reasoning, RUBATO-Full achieves 40.0% to 45.1% accuracy, whereas the baseline fails completely.
#### Cross-Lingual TTS for Indian Languages: A Benchmark and Inference-Time Duration Estimation Strategies
 - **Authors:** Adarsh Arigala, Arjun Gangwar, Srinivasan Umesh
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Artificial Intelligence (cs.AI); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2610.14826

 - **Pdf link:** https://arxiv.org/pdf/2610.14826

 - **Abstract**
 Recent alignment-free Text-to-Speech (TTS) models achieve high synthesis quality by removing explicit alignment modules and simplifying training. However, they often struggle in cross-lingual scenarios due to differences in character durations across languages, leading to duration mismatch and degraded speech quality. Existing approaches typically address this issue using explicit duration predictors, increasing model complexity. Systematic evaluation of TTS performance for Indian languages in cross-lingual settings remains under-explored due to the lack of standardized evaluation benchmarks. In this work, we construct a cross-lingual evaluation benchmark covering English and nine Indian languages and propose a language-aware inference-time duration estimation strategy for alignment-free TTS that incorporates language-specific duration-per character statistics without retraining or architectural modifications. We further develop heuristic formulations that refine duration estimation based on language-dependent duration variations. Experiments on bilingual and multilingual A-DMA models demonstrate improved speech naturalness, intelligibility, and speaker similarity while reducing duration mismatch.
#### SemiCodec: Bridging Speech Tokenization and High-fidelity Reconstruction with Semi-discrete Representations
 - **Authors:** Kecan Mao, Nan Xu, Zhaoyang Meng, Zhengyao Ma, Yingming Gao, Yi Zhao, Ya Li
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2610.14834

 - **Pdf link:** https://arxiv.org/pdf/2610.14834

 - **Abstract**
 Speech representation learning requires a discrete modeling interface while preserving high waveform fidelity. Existing approaches rarely achieve both: discrete tokens ease modeling but lose fine-grained acoustic information, while continuous latents preserve fidelity but require specialized generative objectives. We introduce SemiCodec, a novel Semi-discrete neural speech Codec with synchronized discrete tokens and continuous latents. Direct joint optimization may create a reconstruction shortcut through the continuous pathway, leaving the discrete tokens insufficiently grounded. To address this issue, we propose a Preserve-then-Ground training paradigm that first establishes a high-fidelity synthesis space and then grounds SSL-guided discrete tokens within it. Experiments show that SemiCodec outperforms fully discrete codecs in reconstruction quality and intelligibility while remaining competitive with continuous representations in acoustic fidelity and speaker preservation. Audio samples are available at this https URL
#### Word-Aligned Prosody Reflects Speaker Differences More Than Depression or PTSD Severity
 - **Authors:** Hsiang-Chen Yeh, Aurosweta Mahapatra, Shreeram Suresh Chandra, Ryan L. Boyd, Berrak Sisman
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Computation and Language (cs.CL); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2610.15181

 - **Pdf link:** https://arxiv.org/pdf/2610.15181

 - **Abstract**
 Word-aligned prosody is increasingly used to detect depression and posttraumatic stress disorder (PTSD), yet the stability and clinical relevance of these measurements remain unclear. We examine stability at two levels. At the word level, we ask whether a speaker produces the same word in a similar speaker-specific way across different utterance contexts. At the speaker level, we ask whether averaging prosody across many words produces consistent speaker scores across separate recordings. Drawing on principles of psychological measurement, we compare these representations in four corpora spanning acted, spontaneous, and clinical speech. Word-specific deviations show little cross-context recurrence (0.01-0.16), whereas speaker summaries are highly consistent, reaching 0.997 in ESD, 0.73-0.90 in MSP-Podcast, and 0.84-0.996 in DAIC-WOZ and PTSD-STOP. Despite this consistency, associations with depression or PTSD severity do not exceed 0.15; equivalence tests exclude effects of 0.30 in all analyses and 0.20 in DAIC-WOZ, while within-person associations with daily PTSD symptoms are near zero ($|\rho|\leq0.04$). Thus, aggregation yields stable speaker-level prosodic measures, but these measures do not detectably track depression or PTSD severity in the studied corpora. Code is available at this https URL.
#### KOMOS: Automatic Mean Opinion Score Estimation for Korean Speech
 - **Authors:** Sangchun Ha, Junwon Moon, Kyuhong Shim
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2610.15353

 - **Pdf link:** https://arxiv.org/pdf/2610.15353

 - **Abstract**
 As speech generation and real-time spoken dialogue systems advance, neural mean opinion score (MOS) predictors are increasingly used to evaluate speech quality. However, many existing predictors are trained primarily on English read speech and generalize inconsistently across languages and speaking styles. In particular, their reliability on Korean speech has not been adequately validated. We propose KOMOS, a Korean MOS predictor for evaluating the overall quality of both read and spontaneous speech. To train and evaluate KOMOS, we collect 5,000 ratings from 29 native Korean listeners for 1,000 utterances spanning real recordings, synthesized speech, and degraded speech. KOMOS achieves an utterance-level Spearman rank correlation (SRCC) of 0.900, compared with 0.703 for UTMOS v1 and 0.767 for UTMOS v2. The inference code is publicly available at this https URL and the trained model at this https URL.
#### LiveAnon: Improving Privacy and Utility in Streaming Speaker Anonymization with Codec Language Models
 - **Authors:** Yejin Kim, Kyungsu Kim, Seungu Han, Kyogu Lee
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Machine Learning (cs.LG); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2610.15379

 - **Pdf link:** https://arxiv.org/pdf/2610.15379

 - **Abstract**
 Speaker anonymization (SA) aims to conceal speaker identity while preserving linguistic content and emotion, but existing streaming systems often lose emotion information through discrete content bottlenecks or require emotion-specific modeling. We present LiveAnon, a voice-conversion-based streaming SA system that directly conditions an autoregressive codec language model on hidden features from a pretrained streaming ASR encoder, preserving emotion without emotion-specific supervision. To reduce codec-generation latency, we introduce capped RVQ decoding, which preserves dependencies among coarse codebooks while predicting the remaining codebooks in parallel. Under VPC24, LiveAnon improves intelligibility and emotion preservation over fixed-target Stream-Voice-Anon (SVA), a strong prior streaming SA system, matches emotion-prompted SVA in emotion preservation, and achieves the strongest semi-informed privacy among streaming systems. Under VPC26, it ranks second across all metrics among the evaluated anonymization systems, demonstrating a strong privacy-utility balance.
#### Unsupervised Cross-device Speaker Diarization And Linking Through Spatiotemporal Structure
 - **Authors:** Suhas BN, Nikitha K
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2610.15978

 - **Pdf link:** https://arxiv.org/pdf/2610.15978

 - **Abstract**
 Linking speakers across independently recorded rooms is difficult because microphone, room, and codec mismatch distort speaker embeddings. We study visitor linking without visitor enrollment or identity labels in simulated outpatient workflows. The proposed method combines per-device embedding centering, concurrency constraints, room-type-specific visit constraints, and a clustering threshold calibrated from concurrent visitors and identified staff voices. On simulated days built from VOiCES recordings, with companions, return visits, walk-ins, silent visits, and acoustic leakage, it improves B$^3$-F1 over the best label-free baseline for all 11 extractors, by 5.3-14.6 points (all 95\% intervals exclude zero). ECAPA-TDNN reaches 86.6, compared with 76.4 for the best label-free baseline and 87.7 for the best oracle-informed baseline. Gains persist across the reported workflow perturbations, but depend on removing duplicated cross-room speech: without this step, the method wins for only 2 of 11 extractors. Single-visit patients also remain vulnerable to false merges. These results support structure-aware linking while identifying preprocessing and identity-safety limitations that must be addressed before clinical deployment.
#### Towards Privacy-Preserving ASR: Speaker-Level Machine Unlearning
 - **Authors:** Seaone Ok, Seungu Han, Eungbeom Kim, Kyogu Lee
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2610.16196

 - **Pdf link:** https://arxiv.org/pdf/2610.16196

 - **Abstract**
 Automatic Speech Recognition (ASR) systems are commonly designed to transcribe linguistic content independently of speaker identity. However, speaker-specific information can persist within the latent representations of ASR models, raising privacy concerns when training data includes sensitive individuals. While machine unlearning has been studied in several speech tasks, applying speaker unlearning to ASR remains challenging due to the strong entanglement of linguistic and speaker-related information in speech representations. In this work, we conduct a layer-wise analysis of Self-Supervised Learning (SSL) based ASR representations, revealing that speaker identity is primarily encoded in intermediate encoder layers. Based on this observation, we explore a feature-level unlearning strategy that targets these identity structures. Experiments on the VCTK corpus show that the proposed approach removes the influence of specific speakers while maintaining robust transcription performance.
#### FOAdapter: A Plug-and-Play Spatial Tokenizer for Low-Bitrate FOA Speech Coding
 - **Authors:** Kirak Kim, Yoonjeong Park, Juhan Nam, Sungyoung Kim, Minje Kim
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2610.16202

 - **Pdf link:** https://arxiv.org/pdf/2610.16202

 - **Abstract**
 Spatially encoded speech conveys not only linguistic information but also spatial cues. This enables listeners to distinguish and localize speakers, contributing to more natural and realistic auditory experiences in applications such as remote conferencing and immersive media. Neural speech codecs (NSCs) enable high-quality reconstruction of monaural speech at increasingly low bitrates while preserving linguistic information, but their ability to effectively preserve spatial cues remains limited. We propose FOAdapter, a plug-and-play spatial tokenizer that leverages a pretrained monaural speech codec to efficiently represent both linguistic and spatial information in first-order ambisonic (FOA) speech at low bitrates. Comprehensive evaluation demonstrates that FOAdapter consistently preserves spatial information across all evaluated codecs. Source code and audio examples are available on our demo website.
#### Spiking Neural Beamforming for Multichannel Speech Enhancement on the Edge
 - **Authors:** Hanan Beit-On, Eli Shapira, Boaz Rafaely
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2610.16430

 - **Pdf link:** https://arxiv.org/pdf/2610.16430

 - **Abstract**
 Speech enhancement with wearable and edge devices requires architectures that balance enhancement quality against power and memory resources. Spiking Neural Networks (SNNs) offer a promising path towards this goal through sparse, event-driven computation, but existing SNN-based speech enhancement methods are largely restricted to single-channel processing and therefore cannot exploit the spatial diversity available from microphone arrays. In this work, we propose a fully SNN framework for multichannel speech enhancement based on a fully connected architecture (FC-SNN) with sequential spike encoding, derived from a compact artificial neural network (ANN) baseline via leaky integrate-and-fire neurons. We characterize how enhancement performance scales with model size and spike-train length relative to ANN and classical beamforming baselines, and report the resulting parameter counts and computational complexity. We further compare direct and rate spike encoding, revealing a clear trade-off between event-driven computation and enhancement performance, and use these results to identify favorable configurations for edge devices.
#### Active noise cancellation extends the benefit of AI speech enhancement in real hearing-aid recordings
 - **Authors:** Andrew T. Sabin, Linkai Li
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2610.17287

 - **Pdf link:** https://arxiv.org/pdf/2610.17287

 - **Abstract**
 Active noise cancellation (ANC) is common in headphones but rare in hearing aids, and its value for understanding speech in noise is not established. Algorithms that improve signal-to-noise ratio (SNR) act only on the hearing aid's amplified path, while unprocessed sound also reaches the eardrum directly through the vent, limiting the SNR improvement that is realized. Our earlier simulations predicted that attenuating this direct path with ANC would substantially increase the benefit of such processing. We tested that prediction in three hardware conditions: a standard receiver-in-canal (RIC) hearing aid, and an ANC-capable RIC with ANC off and on. Each was fit to NAL-NL2 targets for an N3 hearing loss and recorded on KEMAR with an AI speech-enhancement algorithm (AI-SE) off and on, in seven realistic scenes above 70 dB SPL. Speech and noise spectra were estimated from each recording with a binary time-frequency mask derived from the clean speech. AI-SE produced a similar peak benefit of about 4 dB in the high frequencies in all three conditions, but that benefit stayed at or above 1 dB only above 1500, 600, and 130 Hz for the standard RIC, ANC RIC (ANC off), and ANC RIC (ANC on), respectively, closely tracking the measured attenuation of the direct path. With AI-SE on, ANC RIC (ANC on) improved SII-weighted SNR by 5.1 dB relative to the open ear, compared with 3.3 dB for ANC RIC (ANC off) and 1.6 dB for the standard RIC. With AI-SE off, enabling ANC added only 0.2 dB. ANC is therefore an enabling technology: on its own it added little, but by attenuating the direct path it allowed SNR-enhancing processing to deliver substantially greater benefit at the eardrum, extending that benefit to much lower frequencies.
#### Self-Supervised Source-Filter Model of Speech Production
 - **Authors:** Nani Meka (1), Raghavan Ramesh (1), Sri Rama Murty Kodukula (1) ((1) Speech Information Processing Lab, Indian Institute of Technology Hyderabad)
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2610.18120

 - **Pdf link:** https://arxiv.org/pdf/2610.18120

 - **Abstract**
 Deep learning models of speech seldom represent the source-filter process by which speech is produced, relying instead on latent features with no physical correspondence to articulation or phonation. We propose SoFilNet, a self-supervised, differentiable Source-Filter Network that estimates the parameters of speech production directly from the raw waveform: a source, specified by fundamental frequency (F0), voicing and the aperiodicity of five frequency bands, and a filter, specified by ordered formant poles, additional poles that shape the rest of the spectral envelope, and antiresonance zeros. The model is trained only to reconstruct the input speech; no formant or pitch annotation is used. Since per-bin spectral losses favor blurred spectra wherever the model is uncertain, we train F0 and voicing against autocorrelation pseudo-labels refined with the model's own filter, aperiodicity against the periodic fraction of each band, and formant sharpness against spectral contrast. On VTR-FORMANTS, the estimated formants attain the lowest mean error among the reported methods, including supervised trackers, and on the unseen PTDB-TUG corpus the F0 frame error matches the best classical tracker and is lower than that of all learned trackers. The parameters are also consistent with speech production theory: aperiodicity rises with frequency and orders phone classes by the turbulence of their excitation, and the zeros deepen on nasal consonants, the lower one rising with place of articulation.
#### PhonoVoxtLM: Exploring Single-Codebook Phonological Tokens for Speech Understanding and Generation
 - **Authors:** Mingrui Liang, Kentaro Onda, Shinji Watanabe
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2610.18469

 - **Pdf link:** https://arxiv.org/pdf/2610.18469

 - **Abstract**
 We investigate how speech token representations can support both understanding and generation within a unified speech language model. We present PhonoVoxtLM, built on the existing VoxtLM framework using phonological tokens designed to retain linguistic content and prosodic information. The existing tokenizer is jointly trained for speech recognition and resynthesis. Its single-codebook units serve as both speech inputs and generation targets, without additional speech-token prediction modules. Experiments show improvements over HuBERT-based baselines across understanding and generation under different training-data scales. Further comparisons with WavLM-based tokens reveal trade-offs between recognition and synthesis, highlighting the importance of evaluating shared representations across both capabilities. We further extend the framework to spoken question answering through fine-tuning with replay. The adapted model improves answering performance over similarly adapted HuBERT baselines while retaining advantages on the original speech tasks. Our findings support single-codebook representations designed for both content and prosody as a practical choice for unified multitask speech modeling.
#### Hearing's Blind Spots: Toward Perceptual Compute Allocation in Neural Text-to-Speech
 - **Authors:** Aayush Gupta
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2610.14056

 - **Pdf link:** https://arxiv.org/pdf/2610.14056

 - **Abstract**
 A speech synthesizer produces a detailed waveform, but a listener need not hear every detail. Strong sounds can mask weaker neighbors, creating context-dependent "blind spots" in hearing. Could neural text-to-speech (TTS) spend less computation on those details? We investigate a first step: how the placement of spectral degradation affects completed speech from Piper and Kokoro. At matched time-frequency bin-removal budgets, a simple low-energy-first policy achieves higher mean wideband PESQ than random deletion and the tested psychoacoustic masking criteria. At 50% removal in the spreading-matrix comparison, energy, masking, and random policies score 4.46, 3.60, and 1.19 on Piper, and 3.42, 2.10, and 1.22 on Kokoro. Spectrograms and masking maps expose where these perturbations fall and why a model's below-threshold regions cannot automatically be treated as inaudible. Calibration diagnostics show substantial threshold sensitivity, while a small single-listener pilot establishes no transparent removal budget. CPU profiles identify waveform decoding as the main measured kernel cost, but a tested dynamic-INT8 Kokoro decoder runs slower than FP32 on both CPUs. The results establish energy ranking as a strong oracle baseline and motivate testing perceptual allocation inside a decoder. They do not yet demonstrate saved computation: the decisions here use already synthesized audio, and quality metrics are evaluated at 16 kHz. We release scripts, results, audio examples, and a listening-study instrument.
#### BridgeCodec: Mamba Enhanced Neural Audio Codec with Schrödinger Bridge at Low Bitrate
 - **Authors:** Zijian Lin, Jing Yang, Jinghao Luo, Zhuo Wang, Fan Fan, Zhiyong Wu
 - **Subjects:** Subjects:
Sound (cs.SD); Artificial Intelligence (cs.AI); Machine Learning (cs.LG); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2610.15453

 - **Pdf link:** https://arxiv.org/pdf/2610.15453

 - **Abstract**
 Neural audio codecs achieve remarkable performance but operate as closed ecosystems with rigidly co-trained encoderdecoder pairs, severely hindering interoperability. To resolve this, we propose BridgeCodec, a versatile framework that decouples these pairs by translating latent representations between mismatched, frozen endpoints. We formulate this cross-codec translation as a Schrödinger Bridge optimal transport problem, enabling robust mapping between disparate distributions without predefined noise priors. To efficiently capture longrange speech temporal dependencies, we integrate a Mambaenhanced U-Net architecture. We validate BridgeCodec on an extreme deployment scenario: mapping a lightweight 8 kHz source encoder directly to a 48 kHz target decoder at an ultralow bitrate of 1 kbps. Experimental results demonstrate that BridgeCodec achieves superior wideband reconstruction and high perceptual quality, successfully bridging heavily bottlenecked, heterogeneous codecs.
#### Deep Speech Representations Track Word Timing along a Low-Dimensional Circular Trajectory
 - **Authors:** Burin Naowarat, Hao Tang, Sharon Goldwater
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Machine Learning (cs.LG); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2610.16273

 - **Pdf link:** https://arxiv.org/pdf/2610.16273

 - **Abstract**
 Speech timing is fundamental for speech models to recognize words accurately, but how word-timing information is represented in speech models remains underexplored. We study this question using the framework of representational geometry, which investigates whether and how concepts are represented in low-dimensional subspaces of the representation space. Prior work in cognitive neuroscience has shown that when HuBERT feature sequences are averaged across words of the same duration, they form near-circular trajectories from word onset to word offset in a two-dimensional "temporal subspace." Here we show that this circular structure of word timing is a general property of deep speech representations, by demonstrating its emergence in nine additional speech models covering both supervised and self-supervised. We then move beyond averaged word segments and show that in HuBERT, the circular trajectories are consistent enough at the level of individual words to perform word segmentation. Our approach, Rotary Segmentation (RoSeg), achieves 64.3% boundary F-score on Librispeech by detecting word boundaries when cycles complete in the temporal subspace. Although supervision is needed to identify a good subspace, nevertheless this result demonstrates much stronger implicit word knowledge in HuBERT than has previously been shown.
#### D2: High-Capability Full-Duplex Spoken Language Models from Half-Duplex Backbones
 - **Authors:** Cheng-Kuang Chang, Alexander H. Liu, Brian Kingsbury, Hilde Kuehne, Rogerio Feris, Samuel Thomas, James Glass, Kai-Wei Chang
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Artificial Intelligence (cs.AI); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2610.17124

 - **Pdf link:** https://arxiv.org/pdf/2610.17124

 - **Abstract**
 Full-duplex spoken language models (FD-SLMs) support simultaneous listening and speaking, but often lag behind half-duplex spoken language models (HD-SLMs) in general capabilities. We propose Dialogue-to-Duplex (D2), a framework that converts pretrained HD-SLMs into high-capability FD-SLMs at fine temporal granularity. D2 adapts the original audio encoder to streaming through decoder-free representation distillation and learns full-duplex interaction through supervised fine-tuning. We systematically study the effects of chunk size on full-duplex interaction, general capabilities, and inference computation. We apply D2 to two distinct HD-SLM architectures, Qwen3-Omni and LLaMA-Omni 2. At a 12.5Hz update rate, Qwen3-Omni-D2 achieves 99.2% turn-taking success with 521-ms latency, 99.5% interruption success, and a state-of-the-art VoiceBench score of 70.1.
#### SonicWorld: Source-Centric Latent Dynamics for Spatial Audio Reasoning
 - **Authors:** Chenlong Wei, Hanling Wang, Mingjin Lou, Yufan Shi, Yandong Sun, Sheldon Yu, Xiaohui Zhu, Xiaobo Jin, Yiwei Wang
 - **Subjects:** Subjects:
Sound (cs.SD); Artificial Intelligence (cs.AI); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2610.17131

 - **Pdf link:** https://arxiv.org/pdf/2610.17131

 - **Abstract**
 Dynamic spatial audio reasoning requires a model to associate changing acoustic observations with the correct source and recover its spatial state at a queried time or interval, a challenge intensified by overlapping speech and source motion. To study this problem, we introduce Sonic-Bench, comprising 2.8K first-order Ambisonics clips and 28.8K target-specific question-answer pairs across 480 simulated indoor scenes. The benchmark covers localization, distance estimation, and motion reasoning in single- and dual-source settings. We further propose SonicWorld, which integrates physical, learned acoustic, and semantic features into persistent source states through learned transitions and recurrent updates, and projects them into an Omni LLM for spatial reasoning. Compared with Spatial-Omni on the Sonic-Bench test split, SonicWorld improves direction accuracy from 31.98% to 34.69% and reduces azimuth error from 57.22° to 48.01°. In zero-shot transfer to Spatial LibriSpeech, SonicWorld retains strong localization performance, achieving 73.98% direction accuracy and 19.29° azimuth error, compared with 27.14% and 70.74° for Spatial-Omni.
#### GeoMER: A Joint Euclidean-Hyperbolic Geometry Framework for Multimodal Emotion Recognition
 - **Authors:** Ting Zhu, Huizhi Liang, Boguslaw Obara, Shufei Duan, Varun Ojha
 - **Subjects:** Subjects:
Sound (cs.SD); Artificial Intelligence (cs.AI); Machine Learning (cs.LG); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2610.18587

 - **Pdf link:** https://arxiv.org/pdf/2610.18587

 - **Abstract**
 Multimodal Emotion Recognition (MER) aims to understand human emotions by integrating information from multiple modalities such as speech and text. Psychological studies suggest that emotions exhibit latent hierarchical structured relationships in their semantic representations. However, most MER approaches rely on Euclidean representations, which capture fine-grained cross-modal interactions but do not explicitly provide a geometric prior for organizing hierarchical emotional structures. To mitigate this gap, we propose GeoMER, a joint Euclidean-Hyperbolic modeling framework that introduces hyperbolic emotion-core routing to capture higher-level, abstract emotional representations for MER. The proposed framework jointly models token-level cross-modal semantic interactions and latent emotion representations in two complementary geometric spaces. Specifically, GeoMER consists of a Euclidean Multimodal Fusion Module (EuclFusion) for multimodal representation fusion and a Hyperbolic Geometric Core Routing (HypGeoRoute) module that performs geometry-aware core assignment and aggregation in a hyperbolic manifold to capture more abstract feature representations. The HypGeoRoute module complements the Euclidean token-level modeling. We experiment on the IEMOCAP multimodal MER benchmark and demonstrate that the proposed method achieves the best performance, with 83.54% weighted accuracy and 83.35% unweighted accuracy, outperforming Euclidean baselines, including MemoCMT.


by Zyzzyva0381 (Windy). 


2026-10-14
