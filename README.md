# Showing new listings for Monday, 12 October 2026
Auto update papers at about 2:30am UTC (10:30am Beijing time) every weekday.


阅读 `Usage.md`了解如何使用此repo实现个性化的Arxiv论文推送

See `Usage.md` for instructions on how to personalize the repo. 


Keyword list: ['text-to-speech', 'text to speech', 'tts', 'LLM-based', 'speech', 'voice']


Excluded: []


### Today: 20papers 
#### Acoustic Biomarkers for Alzheimer's Disease Detection from Spontaneous Speech
 - **Authors:** Anh L Tran, Arjee Javellana Restar, Bereket Kindo
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Machine Learning (cs.LG); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2610.12500

 - **Pdf link:** https://arxiv.org/pdf/2610.12500

 - **Abstract**
 Alzheimer's disease (AD) remains a major public health challenge, and early identification is crucial for timely intervention and care planning. Recent work has shown that speech-based digital biomarkers can capture subtle changes in vocal production, prosody, and timing that often precede or accompany cognitive decline. In this study, we investigate whether acoustic features extracted from spontaneous speech can reliably distinguish individuals with AD from healthy controls. Using a structured speech dataset derived from the Cookie Theft picture-description task, we extract clinically informative low-level descriptors spanning prosodic, spectral, and temporal domains, and evaluate a range of machine learning classifiers. Our results indicate that acoustic biomarkers carry strong discriminative value, with the best-performing models achieving clinically meaningful separation between diagnostic groups. These findings support the use of non-invasive, low-cost audio-based screening tools and motivate further work on multimodal and longitudinal assessment frameworks for dementia risk detection.
#### E3-System: Expressive EmoData, Explicitly Controllable, Easy-to-Use TTS
 - **Authors:** Yiqiao Huang, Xiaolong Lin, Zhenpeng Zhan, Zhizheng Wu
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2610.12993

 - **Pdf link:** https://arxiv.org/pdf/2610.12993

 - **Abstract**
 Recent zero-shot Text-to-Speech (TTS) systems can clone unseen speakers, yet controllable expressiveness remains limited by a gap between semantic intent and acoustic realization. Text descriptions offer a natural control interface but provide weak acoustic grounding, whereas speech style prompts provide concrete prosody but entangle style with the prompt speaker's timbre. To bridge this gap, we propose E3-System, a unified intent-to-speech framework for expressive zero-shot TTS that couples three requirements: Expressive style modeling, Explicit timbre-style control, and Easy-to-use natural-language interaction. Given a target speaker and either a natural-language style description or a speech style prompt, E3-System generates expressive speech while preserving the target timbre. The key idea is to close the full control loop: building a large expressive style space, grounding user intent into usable acoustic style references, and synthesizing speech with disentangled timbre and style. To realize this loop, EmoData provides a thousand-hour multilingual emotional speech dataset, Style-RAG grounds text descriptions into high-quality acoustic style prompts from these data, and Dual-TTS transfers the grounded style to arbitrary target speakers through VC-augmented timbre-style disentanglement. Experiments show that E3-System achieves strong speech quality, speaker preservation, and explicit expressive controllability.
#### Frontend-Assisted Large Audio-Language Models for SLT2026 SmartGlasses Challenge
 - **Authors:** Yushi Wang, Zeyan Song, Xiaobin Rong, Chengwei Liu, Jun Gao, Jing Lu
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2610.13282

 - **Pdf link:** https://arxiv.org/pdf/2610.13282

 - **Abstract**
 This paper describes our systems for the Track 1 time-stamped speaker-attributed ASR task of the SmartGlasses Challenge. To handle overlapping speech, background noise and far-field conditions in two-person conversations, we employ SoulX-Transcriber, a multimodal large language model that takes full-length audio as input and generates structured transcriptions. In addition, we investigate the effect of speech frontend processing, including a rule-based Guided Source Separation method and a data-driven speech separation model, both of which improve performance on the validation set. Considering inference stability, our final submission uses the fine-tuned SoulX-Transcriber system, achieving a tcpCER of 6.22 on the test set.
#### Does Financial Speech Need Continual Learning Regularization? A Study of EWC and mini-KFAC on Earnings Call ASR
 - **Authors:** Thai Quoc On, Homayoon Beigi
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Computation and Language (cs.CL); Machine Learning (cs.LG)
 - **Arxiv link:** https://arxiv.org/abs/2610.13432

 - **Pdf link:** https://arxiv.org/pdf/2610.13432

 - **Abstract**
 Earnings call transcription presents a realistic continual learning scenario where Automatic Speech Recognition (ASR) systems are updated with new recordings but risk catastrophic forgetting of previously learned financial speech patterns. Adapting Whisper-small from a general-domain speech corpus to the Earnings22 benchmark reveals the opposite: all evaluated methods exhibit improvement on held-out financial speech from the same domain during adaptation rather than forgetting. This task complementarity makes regularization counterproductive. Nevertheless, when regularization is warranted, a layer-wise mini-Kronecker-Factored Approximate Curvature (mini-KFAC) approximation seems to achieve lower word error rates (WERs) and more stable optimization than standard diagonal Elastic Weight Consolidation (EWC). The results suggest that task complementarity should be assessed before applying continual learning constraints, and prefer mini-KFAC EWC over diagonal EWC when needed.
#### Vocal Intensity Conversion with Continuous SPL Control via Adversarial Training
 - **Authors:** Quentin Le Tellier, Albert Rilliard, Olivier Perrotin, Marc Evrard
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Machine Learning (cs.LG); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2610.13608

 - **Pdf link:** https://arxiv.org/pdf/2610.13608

 - **Abstract**
 Vocal intensity conversion renders a recorded utterance at a different vocal intensity, rather than simply scaling its amplitude, while preserving its content and speaker identity. Existing systems control intensity through a reference utterance or a few discrete categories. Instead, we control the conversion with any target sound pressure level (SPL) on a continuous dB scale. A lightweight converter, trained as a conditional cycle-consistent GAN without parallel data, transforms the features of a frozen speech encoder, from which a pretrained vocoder resynthesizes the speech. We compare three representations, log-mel spectrograms, WavLM, and SpeechTokenizer, and introduce an objective protocol that measures the achieved intensity change against the target one on a calibrated corpus. With WavLM, the converter achieves a measured 83% of the target change in intensity and also shifts the fundamental frequency. A perceptual evaluation confirms that perceived vocal intensity increases with the target level on amplitude-normalized stimuli.
#### OpenSVBench: Diagnosing Speaker Verification Generalization Across Deployment Scenarios
 - **Authors:** Qituan Shangguan, Bosen Xu, Shuai Wang
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2610.13760

 - **Pdf link:** https://arxiv.org/pdf/2610.13760

 - **Abstract**
 Speaker verification (SV) systems now achieve very low error rates on standard public protocols, but those results do not fully describe deployment generalization. VoxCeleb-style evaluations remain useful anchors for English in-the-wild media speech. Scenario-specific evaluations and challenges are also valuable. However, neither setting directly shows whether the same system remains reliable across diverse deployment scenarios. We introduce OpenSVBench, a public benchmark for cross-scenario SV evaluation. Its fixed public core contains 26 trial sets organized into 11 scenario groups, with 9.15M verification trials, frozen target/non-target keys, and metadata for scenario-wise reporting. Around this core, the online leaderboard acts as a diagnostic evaluation interface rather than only a ranked list. It accepts score packages from new systems, evaluates them with frozen keys and common scoring, and reports trial-set breakdowns, scenario-group metrics, macro EER/minDCF, model-wise profiles, and scenario-wise comparisons. We evaluate 18 public checkpoints spanning x-vector, ECAPA-TDNN, CAMPPlus, ResNet and ERes variants, ReDimNet2, SAM-ResNet, and pretrained Transformer speaker recipes. The results show clear cross-scenario gaps: systems with similar macro EER can have different scenario profiles, strong performance in one scenario does not guarantee strong performance in another, and same-scenario trial sets can still expose source and protocol gaps.
#### OpenPA: Forced-alignment-free Multi-granular Pronunciation Assessment with Mixed-transcript Adaptation
 - **Authors:** Fu-An Chao, Bi-Cheng Yan, Berlin Chen
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Computation and Language (cs.CL); Sound (cs.SD); Signal Processing (eess.SP)
 - **Arxiv link:** https://arxiv.org/abs/2610.13826

 - **Pdf link:** https://arxiv.org/pdf/2610.13826

 - **Abstract**
 Recent advances in automatic pronunciation assessment (APA) have driven a shift from scripted to free-form assessment. However, existing methods often rely on multi-stage pipelines with forced alignment and separate feature extractors, limiting their practicality across response settings. In light of this, we propose OpenPA, a novel APA framework that leverages Whisper's cross-attention mechanism to jointly predict word- and utterance-level pronunciation scores without forced alignment. Through mixed-transcript adaptation and adaptive substitution weighting, OpenPA accommodates heterogeneous transcripts across closed- and open-response settings within a single model. Trained solely on speechocean762, OpenPA achieves the strongest overall performance among evaluated systems under in-domain closed and simulated open-response conditions, while maintaining its performance zero-shot on MultiPA. An additional cross-task evaluation on oral-cancer speech suggests its potential applicability beyond non-native speech.
#### SEER: Source-Conditioned Emotion Enhancement via Retrieval for Cochlear-Implant Speech
 - **Authors:** Hsing-Hang Chou, Yun-Shao Lin, Ching-Chin Sung, Chi-Chun Lee
 - **Subjects:** Subjects:
Signal Processing (eess.SP); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2610.11131

 - **Pdf link:** https://arxiv.org/pdf/2610.11131

 - **Abstract**
 Cochlear implants (CIs) restore speech access but weaken cues needed for vocal emotion recognition. Prior CI-oriented enhancement requires parallel normal/strong recordings and intensity labels. We propose SEER, a retrieval-based framework that learns which same-emotion reference helps each source remain recognizable after CI processing. A source-conditioned retriever learns CI-aware utility from sampled emotional voice conversion outcomes, while uncertainty-guided exploration avoids exhaustive pair evaluation; neither parallel recordings nor intensity labels are required. SEER improves Source macro-F1 at N8 by 7.30 points on RAVDESS and 11.66 points on ESD, with significant ESD gains across N4/N8/N16. Sixteen-listener RAVDESS gains are significant across all conditions. Exhaustive analysis finds an aggregate benefit from stronger references but little effect from matching gender or content.
#### TTS-Q: Perception-Guided Post-Training Quantization of Text-to-Speech Diffusion Transformers
 - **Authors:** Xiaoqin Feng, Yuqi Li, Weilun Feng, Yi-Cheng Lin, Kai Li, Xianglong Wang, Kuo Yang, Chuanguang Yang, Yingli Tian, Wei Li, Zhongwei Cheng, Lin Chen
 - **Subjects:** Subjects:
Sound (cs.SD); Machine Learning (cs.LG); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2610.12474

 - **Pdf link:** https://arxiv.org/pdf/2610.12474

 - **Abstract**
 Zero-shot text-to-speech (TTS) can clone a voice from seconds of audio. Yet the quality comes from billion-parameter diffusion transformers whose size and multi-step sampling put them out of reach of the on-device, assistive, and real-time applications that need them most. Post-training quantization (PTQ) shrinks such models without retraining, but it has been built for language and vision, not speech. At 4 bits, a strong quantizer leaves intelligibility close to full precision but consistently erodes speaker timbre, so error rates stay low while the voice drifts from the target speaker. We ask why quantization hurts speech, where the damage lives, and whether it can be repaired without retraining. To find out, we run the first 4-bit weight-and-activation (W4A4) PTQ study of a TTS diffusion transformer at 1B and 3.5B. The timbre cost comes from both the weight and the activation quantization, but only the activation part has the timestep structure a precision schedule can exploit. We therefore introduce TTS-Q, which selects hard-to-quantize items from task-matched calibration candidates and applies a budget-balanced mixed-precision schedule that assigns higher precision to late-step feed-forward activations. TTS-Q recovers timbre best at both scales, at no cost to intelligibility. Separately, the deployed calibrated-W4A4 reference reduces peak memory on GPU, providing a practical reference for resource-constrained TTS deployment.
#### RDSE: Real-Time Directional Speech Enhancement Framework for Dual-Mic Commodity Smartphones
 - **Authors:** Chinmay Deo, Danish Mahajan, Soudeep Ghoshal, Himanshu Buckchash
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2610.12476

 - **Pdf link:** https://arxiv.org/pdf/2610.12476

 - **Abstract**
 Audio-visual zooming on mobile devices is fundamentally constrained by limited microphone count, compute resource, unknown array geometry, and weak assumptions on real-world acoustic conditions. We present RDSE: Real-Time Directional Speech Enhancement Framework, a hybrid pipeline that combines traditional signal processing and modern machine learning methods for dual-microphone low-end smartphones. Unlike tightly coupled AV systems that continuously use visual feeds or embeddings, RDSE decouples visual interaction from feature inference, using camera input solely to parameterize the steering vector for spatial filtering. The proposed system uses four modules: dereverberation network, a beamformer with a temporal convolutional network, and a MetricGAN based denoiser. Extensive evaluation on benchmark data and real-world recordings demonstrates effective zooming without precise geometry calibration. Comparative experiments show competitive performance to existing methods. Proposed system is rigorously tested for statistical validity, ablation study, latency profiling, and practicality through real-time live-streaming.
#### Efficient Audio-Visual Speech Enhancement via Reinforcement Learning-Based Neural Architecture Search
 - **Authors:** Khubaib Ahmed, Ahsan Adeel, Nasir Saleem, Kia Dashtipour, Ahsan Ulhaq, Amir Hussain
 - **Subjects:** Subjects:
Sound (cs.SD); Artificial Intelligence (cs.AI); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2610.12796

 - **Pdf link:** https://arxiv.org/pdf/2610.12796

 - **Abstract**
 Audio-visual speech enhancement (AVSE) improves speech intelligibility in noisy environments but often relies on computationally expensive models unsuitable for resource-constrained edge devices. This paper introduces a reinforcement learning-driven neural architecture search (RL-NAS) framework to discover compact, efficient AVSE architectures. The search space preserves temporal alignment between audio and visual streams, while a novel reward function jointly optimises speech enhancement performance, measured by SI-SNR improvement, and model compactness. Using proximal policy optimisation with constraint-aware action masking, the framework evaluates only 35 candidate architectures and identifies a model with 2.9 million parameters, compared with 22 million in the baseline. The proposed architecture achieves a 14.5 dB SI-SNR improvement, only 0.7 dB below the baseline, while reducing model size by approximately nine-fold. Evaluation on commercial mobile CPUs demonstrates a 4.3-fold reduction in real-time factor, supporting practical on-device deployment. Compared with random search, RL-NAS achieves 92% higher search efficiency. Architectural analysis reveals that shallower but wider temporal convolutional blocks and aggressive visual pathway pruning can maintain enhancement performance while substantially reducing computational requirements. These findings demonstrate the potential of reinforcement learning-guided architecture search for developing lightweight, efficient and deployable multimodal speech enhancement systems.
#### Emo-BPO: Emotion Bidirectional Preference Optimization for Diffusion-based Emotional TTS
 - **Authors:** Jiacheng Shi, Hongfei Du, Xinyuan Song, Y. Alicia Hong, Yanfu Zhang, Ye Gao
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2610.12958

 - **Pdf link:** https://arxiv.org/pdf/2610.12958

 - **Abstract**
 Diffusion-based TTS has demonstrated strong capability in emotional speech generation. However, existing emotion alignment methods primarily reinforce target emotional expressions while overlooking modeling of competing emotional modes, limiting fine-grained controllability. This issue becomes more pronounced under classifier-free guidance (CFG), which relies on contrasting conditional trajectories to shape generation. We propose Emotion Bidirectional Preference Optimization (Emo-BPO), a simple effective framework that jointly learns emotion-aligned and emotion-contrastive score functions from reordered same-text emotional pairs. Our method requires neither auxiliary reward models nor additional annotations, and can be seamlessly integrated into existing diffusion-based TTS systems to improve emotional controllability and expressiveness.
#### From 2.1M Hours of Raw Broadcast Audio to Finnish Speech SSL: Confidence Filtering and Architectural Analysis at Scale
 - **Authors:** Yaroslav Getman, Tamás Grósz, Tommi Lehtonen, Mikko Kurimo
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2610.13102

 - **Pdf link:** https://arxiv.org/pdf/2610.13102

 - **Abstract**
 Broadcast archives offer scale and diversity for self-supervised learning (SSL) but contain substantial noisy, non-speech, and multilingual content. Prior work typically relies on voice activity detection or language identification (LID) to extract speech, ensuring presence rather than quality. Whether and how to filter unlabeled speech by quality for SSL, particularly from noisy broadcasts, remains underexplored. We present a large-scale Finnish SSL study starting from 2.1M hours of unconstrained broadcast audio. We propose automatic speech recognition (ASR) confidence as a proxy for pretraining data quality and show 3-7% relative WER improvements over LID-based filtering despite a smaller training set. We further address the lack of controlled architectural comparisons in speech SSL by evaluating Transformer, Macaron-Net, Conformer, and E-Branchformer at matched parameter budgets. Our models improve upon prior Finnish SSL baselines, particularly at Base scale and out-of-domain.
#### How Should We Evaluate Naturalness for Synthetic Speech with a Foreign Accent?
 - **Authors:** Kentaro Onda, Satoru Fukayama, Daisuke Saito, Nobuaki Minematsu
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2610.13175

 - **Pdf link:** https://arxiv.org/pdf/2610.13175

 - **Abstract**
 Recent advances in speech synthesis technology have enabled the generation of speech with diverse characteristics. One such characteristic is a foreign accent. In our preliminary experiments, we found that UTMOS, a widely used automatic speech quality assessment system, tended to assign lower scores to speech with stronger foreign accents, even though it is produced by a human speaker. This observation raised a question: How do humans evaluate the naturalness of foreign-accented speech? We conducted a large-scale subjective evaluation involving both native and non-native listeners under different evaluation instructions. The results showed that humans also considered accentedness when evaluating naturalness, but that defining naturalness in terms of human-likeness and selecting raters who have greater exposure to the target accent may enable more appropriate evaluation. This study aims to provide a starting point for future discussions on fairer and more appropriate evaluation of synthetic speech with a foreign accent.
#### VoxAnon: A balanced dataset for a fair evaluation of speech anonymization
 - **Authors:** Orane Dufour, Paul Magron, Mickael Rouvier
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2610.13313

 - **Pdf link:** https://arxiv.org/pdf/2610.13313

 - **Abstract**
 Speech anonymization is evaluated with automatic speaker verification attackers whose performance depends on the attribute distribution of both their training and evaluation data, which may bias privacy estimates for some speaker groups. To alleviate this issue, we introduce VoxAnon, a dataset built from VoxCeleb1 and 2 with training and evaluation splits balanced across gender, age, accent, and pitch. We extract and cross-check attributes using two complementary methods, balance trial lists per subgroup, and draw test/enrollment utterances from distinct video sources in order to limit linguistic and session bias. Comparing VoxAnon with unbalanced training and test data across four attackers and three anonymizers, we show that attribute imbalance changes which subgroups appear hardest to re-identify, in directions that depend on the attribute. In the balanced setting, residual disparities persist, mostly tied to accent and pitch, and depend on the anonymizer/attacker pair. Code and trial lists are publicly available.
#### Oído: Open-Vocabulary Speech Recognition on a $5 Microcontroller without a Neural Accelerator
 - **Authors:** Daniel Varela Sánchez, Guillermo Martín-Coello Juárez
 - **Subjects:** Subjects:
Sound (cs.SD); Computation and Language (cs.CL); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2610.13447

 - **Pdf link:** https://arxiv.org/pdf/2610.13447

 - **Abstract**
 Speech recognition on low-cost microcontrollers is, in practice, limited to keyword spotting or a fixed list of commands. We present Oído, an inference engine that runs open-vocabulary English speech recognition entirely on an ESP32-S3 (240 MHz dual-core Xtensa LX7, 512 KB of SRAM, 8 MB of PSRAM, 16 MB of flash, no neural accelerator). Oído executes a 13M-parameter Conformer-CTC model with int8 weights, per-frame dynamic int8 activations and int8 attention on the chip's SIMD instructions, with a schedule designed around flash and PSRAM bandwidth. On LibriSpeech test-clean/test-other the on-device arithmetic reaches 3.70%/8.23% word error rate (WER), within 0.12 points of full precision, and 3.33%/7.15% with an on-chip 1.3M-parameter language model; a transducer head gives 2.99%/6.67% and an int4 variant fits in 8.3 MB. This is 2.3-2.8x fewer errors than the vendor's recognizer on the same chip, and fewer than Whisper this http URL and Moonshine tiny in full precision on a laptop, including in real noise and reverberation. Measured on a physical board, the firmware runs at 1.97x the duration of the audio on one core, with transcripts identical to an instruction-exact emulator: accurate, but not yet real time (a 4 s command is transcribed about 8.7 s after it ends). The two-core schedule gives wrong results on silicon and is disabled. A streaming variant shows partial text while the speaker talks and delivers the final text 4.1-11.1 s after clips of 3.6-10.6 s (7.2-22.4 s in utterance mode) at 4.86%/11.03% WER; a Spanish model makes 34-60% fewer errors than multilingual Whisper tiny. Code and models are released.
#### Phonology-Aware Speaker Extraction Frontend for Audio LLMs
 - **Authors:** Yanxiang Chen, Haoyuan Deng, Jinyu Wang, Hongyi Chen, Dingshan Li, Runlin Huang, Weifeng Su
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2610.13500

 - **Pdf link:** https://arxiv.org/pdf/2610.13500

 - **Abstract**
 Target speaker extraction (TSE) is a key frontend for enabling audio large language models (LLMs) to recognize and understand a designated speaker in overlapping speech. However, multilingual mixtures exhibit heterogeneous extraction demands, causing fixed-depth systems to either under-process challenging inputs or waste computation on inputs that have already converged. Existing TSE frontends generally apply uniform computation, while Audio-LLM adaptation methods neither explicitly model language-conditioned extraction demands nor preserve plug-and-play compatibility with frozen downstream models. We therefore propose LATTE, a lightweight 5.2M-parameter frontend that combines an acoustic Difficulty Estimator, a task-adaptive phonology-informed prior learned from multi-depth TSE gains, and an Adaptive Hourglass Extractor to route each mixture without requiring language identity at inference. On PolyExtract, LATTE achieves a macro SI-SNRi of 15.07 dB at 0.30 GFLOPs, retaining performance within 0.09 dB of a 0.53-GFLOP fixed-depth model while reducing computation by 43.4%, and lowers the average WER from 16.8% to 14.6% relative to TF-GridNet. By jointly modeling sample-level extraction demands and phonological structure, LATTE provides a practical framework for quality-aware, compute-efficient multilingual TSE and offers a scalable path toward robust Audio-LLM deployment.
#### Lossless Inference Acceleration for ASR via EAGLE-Style Speculative Decoding
 - **Authors:** Sen Zhang, Xianghu Yue, Yiming Chen, Wenhuan Lu, Jianguo Wei, Haizhou Li
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2610.13506

 - **Pdf link:** https://arxiv.org/pdf/2610.13506

 - **Abstract**
 Autoregressive Transformer and Conformer models have achieved state-of-the-art performance in automatic speech recognition (ASR). However, their sequential token generation mechanism imposes a significant bottleneck during decoding, resulting in substantial inference latency. While speculative decoding has emerged as a promising lossless acceleration technique, existing ASR-oriented approaches either suffer from low draft acceptance rates due to suboptimal draft quality or rely on specially designed external draft models, which limit their generality and practicality. To overcome these limitations, we propose EAGLE-ASR, a novel framework that accelerates ASR inference via EAGLE-style speculative decoding while strictly preserving lossless output quality. Specifically, we introduce Speech-Draft, a lightweight autoregressive module that serves as a universal draft model by jointly leveraging encoder acoustic features and hierarchical decoder representations from the target ASR model. Furthermore, we adopt a multi-step generation strategy with self-distilled data to train the Speech-Draft, enabling it to better approximate the target model's output distribution and maintain causal consistency during multi-token generation. Extensive experiments on representative Whisper and Conformer AED models demonstrate that EAGLE-ASR achieves significant lossless speedups of up to $1.9\times$ and $2.4\times$, respectively, consistently outperforming existing speculative decoding baselines for ASR. Our proposed framework further generalizes effectively to multilingual ASR and speech translation tasks, establishing EAGLE-ASR as a practical and universally applicable solution for accelerating modern ASR systems without compromising recognition accuracy.
#### AudioTETRIS: Token Elimination via Transitional Importance Signals for Efficient Audio LLMs
 - **Authors:** Shreya Sajid, Anna Vettoruzzo, Joaquin Vanschoren
 - **Subjects:** Subjects:
Sound (cs.SD); Artificial Intelligence (cs.AI); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2610.13600

 - **Pdf link:** https://arxiv.org/pdf/2610.13600

 - **Abstract**
 Large Audio Language Models (LALMs) achieve strong performance on speech-text tasks, but standard audio encoders produce hundreds of tokens for even short audio clips. Because self-attention scales quadratically in sequence length, this inflates inference-time compute substantially. Current training-free token compression methods prune or merge tokens based solely on coarse attention or feature similarity. However, treating audio tokens as generic, isotropic features smooths over spectral transitions, destroying fine-grained phonetic transitions essential for decoding speech. In contrast, speech signals exhibit highly structured temporal patterns where information density peaks at acoustic transitions. We propose AudioTETRIS, a training-free, two-stage framework that uses the audio encoder's representation geometry as an acoustic prior for token importance before applying attention-based compression. Instead of discarding encoder structure after the encoding stage, AudioTETRIS propagates this prior directly into the LLM decoder to compute a per-token fused score. This score drives progressive top-$k$ token selection across decoder layers without requiring additional training or auxiliary models. Evaluated on Qwen2-Audio-7B, Phi-4-Multimodal, and Audio Flamingo 3, AudioTETRIS retains near-uncompressed performance on several speech tasks using only 25% of audio tokens, matches or surpasses full-token accuracy at moderate compression rates, and consistently outperforms existing training-free baselines.
#### Enhancing Speech Recognition through Text-to-Speech and Voice Conversion Augmentation
 - **Authors:** Yunus Emre Ozkose, Ali Haznedaroglu
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2610.13662

 - **Pdf link:** https://arxiv.org/pdf/2610.13662

 - **Abstract**
 Given the challenges associated with obtaining large volumes of real-world audio, the advancement of Automatic Speech Recognition (ASR) systems increasingly depends on synthetic data. This study focuses on the call center-banking domain, providing a targeted analysis of ASR models in this context. We evaluate real and synthetic datasets, using Word Error Rate (WER) and Character Error Rate (CER) as metrics, and employ a speech quality model to assess the impact on ASR performance. Our research compares Text-to-Speech (TTS) and advanced voice conversion methods, including KNNVC, Seed-VC, and Vec2Wav2, revealing significant improvements in speech quality and ASR accuracy, particularly with Seed-VC. Additionally, our domain-specific experimentation provides insight into the unique challenges and opportunities that arise when applying ASR technologies to industry-relevant settings. This highlights the potential of voice conversion technologies to enhance ASR systems, guiding future research in diverse linguistic scenarios, and paving the way for the broader application of ASR innovations across various fields.


by Zyzzyva0381 (Windy). 


2026-10-12
