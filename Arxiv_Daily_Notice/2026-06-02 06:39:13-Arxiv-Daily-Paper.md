# Showing new listings for Tuesday, 2 June 2026
Auto update papers at about 2:30am UTC (10:30am Beijing time) every weekday.


阅读 `Usage.md`了解如何使用此repo实现个性化的Arxiv论文推送

See `Usage.md` for instructions on how to personalize the repo. 


Keyword list: ['text-to-speech', 'text to speech', 'tts', 'LLM-based', 'speech', 'voice']


Excluded: []


### Today: 18papers 
#### Privacy-preserving Prosody Representation Learning
 - **Authors:** Kevin Everson, Mari Ostendorf
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2606.00407

 - **Pdf link:** https://arxiv.org/pdf/2606.00407

 - **Abstract**
 Speech representations that capture prosodic information can be useful for both understanding and generation. However, speaker characteristics are reflected in acoustic-prosodic features (e.g., pitch). To address privacy concerns from the leakage of identity information, we propose a new self-supervised approach to learning prosody representations that incorporates speaker disentanglement strategies. We evaluate our encoder on three tasks to probe representation capabilities, including pitch reconstruction and detection of different prosodic events. Our encoder outperforms raw prosody and HuBERT-base baselines, achieving strong speaker disentanglement without adverse impact on prosody-related downstream tasks.
#### Local Diagnostics of Continuous Normalizing Flow for Out-of-Distribution Detection
 - **Authors:** Xinwei Cao, Mengxuan Lu, Torbjørn Svendsen, Giampiero Salvi
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Computation and Language (cs.CL); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2606.00684

 - **Pdf link:** https://arxiv.org/pdf/2606.00684

 - **Abstract**
 We address the problem of out-of-distribution (OOD) detection for target observations embedded in a subspace of the high dimensional data space. Using continuous normalizing flows (CNFs), we propose a Lagrangian sub-flow (LSF) framework designed to isolate and estimate the density for the relevant components in the representation and using the remaining components as context. Through experimentation with models for speech synthesis, we show that CNFs, similarly to other deep generative models (DGMs), are susceptible to the "likelihood paradox", where high likelihood is erroneously assigned to OOD samples. This is attributed to the inductive bias of DGMs that prioritize low-level structural details over high-level semantic coherence. To mitigate this phenomenon, we propose a number of geometric diagnostic signals based on the velocity field over the sub-flow trajectory. Based on these signals, we design metrics for the challenging task of zero-shot phoneme-level mispronunciation detection. Finally, we demonstrate the superiority of these metrics compared to likelihood-based methods on a real-world mispronunciation detection benchmark.
#### Context-aware child-directed speech detection from long-form recordings
 - **Authors:** Théo Charlot, Tarek Kunze, Kaveri K. Sheth, Alejandrina Cristia, Marvin Lavechin
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Machine Learning (cs.LG); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2606.01134

 - **Pdf link:** https://arxiv.org/pdf/2606.01134

 - **Abstract**
 Automatically distinguishing child-directed speech from adult-directed speech in long-form recordings is key to scalable analyses of children's language environments. Existing approaches process utterances in isolation and have been evaluated primarily on English. We address these gaps along three dimensions. First, we fine-tune and evaluate six-self supervised models on a multilingual dataset of 182 children, showing that in-domain pre-training on child-centered recordings substantially outperforms models trained on adult speech. Second, we demonstrate that incorporating surrounding context substantially improves classification, with an absolute gain of 13.8% in average F1-score. Third, we evaluate our model in a realistic end-to-end pipeline, from adult speech detection to addressee classification, showing that performance drops under automatic segmentation but still consistently outperforms a rule-based baseline.
#### RRP-Voice: A Longitudinal Dataset and Benchmark for Recurrent Respiratory Papillomatosis Detection
 - **Authors:** Wenze Ren, Ke-Han Lu, Kai-Wei Chang, Tiantian Feng, Ching Fang, Zhi-Chi Liao, Dao Thi Hai Yen, Syu-Siang Wang, Yu Tsao, Chi-Te Wang, Shih-Hau Fang
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2606.01639

 - **Pdf link:** https://arxiv.org/pdf/2606.01639

 - **Abstract**
 Deep learning has advanced pathological voice detection rapidly, yet rare laryngeal diseases remain underexplored due to data scarcity. Recurrent Respiratory Papillomatosis (RRP) exemplifies this gap: an HPV-induced disease of the larynx in which patients oscillate between recurrence and post-surgical remission over the years. RRP demands continuous voice monitoring that existing cross-sectional corpora cannot support. We introduce the first longitudinal voice dataset for RRP, comprising recordings from 26 patients with up to ten years of follow-up. Each session pairs sustained vowels with sentence-level utterances, which are annotated by otolaryngologists and confirmed synchronously with laryngoscopy. Building on this resource, we establish a systematic benchmark spanning handcrafted features, end-to-end deep networks, self-supervised pretrained models, and recent audio large language models, all evaluated under session-level cross-validation with patient-level audit. Per-subject longitudinal analyses further confirm that the cross-sectional discriminative signal reflects laryngoscopic disease state rather than stable speaker attributes. This work lays a foundation for rare longitudinal pathological voice tasks in low-resource clinical settings.
#### Kinship Verification Using Voice
 - **Authors:** Jagabandhu Mishra, Tomi H. Kinnunen
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2606.01704

 - **Pdf link:** https://arxiv.org/pdf/2606.01704

 - **Abstract**
 Kinship verification (KV) from voice, the task of determining whether two speakers are biologically related, has received only little attention. Our work establishes a foundational basis for this emerging frontier, contributing to both performance evaluation and detection methodologies. First, leveraging the speech recordings of the large-scale audio-visual dataset, KAN-AV, we propose a revised evaluation protocol that controls for various confounders and adopts a family-disjoint train--test split to address open-set KV. Second, we analyze the close connection between speaker verification and KV, showing that genealogical similarity of speaker pairs plays opposite roles in the two tasks. Third, we tackle KV using three neural speaker embedding extractors (ECAPA-TDNN, WavLM-ECAPA, and ReDimNet) combined with various back-ends. In zero-shot KV including same-speaker target trials, ReDimNet achieves the lowest equal error rate (EER) of $20.8\%$; however, performance degrades to $39.7\%$ under strict kin trials, where same-speaker target trials are excluded. Our best trainable back-end, which applies asymmetric processing of the embedding pair to mitigate age-difference effects, obtains an EER of $32.0\%$ ($18.6\%$ with speaker target trials included). These results highlight the difficulty of KV while showing that speaker embeddings encode familial cues, offering a promising foundation for voice-based kinship analysis.
#### SpeechEditBench: A Bilingual Multi-Attribute Benchmark for Instruction-Guided Speech Editing
 - **Authors:** Hanlin Zhang, Daxin Tan, Dehua Tao, Xiao Chen, Haochen Tan, Linqi Song
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2606.01804

 - **Pdf link:** https://arxiv.org/pdf/2606.01804

 - **Abstract**
 Instruction-guided speech editing requires a model to modify specified speech attributes while preserving unrelated characteristics. Despite rapid progress in Speech Large Language Models (Speech LLMs), systematic evaluation of this capability remains challenging, as existing benchmarks are fragmented across isolated editing tasks. To bridge this gap, we introduce \textbf{SpeechEditBench}, a bilingual multi-attribute benchmark for instruction-guided speech editing. SpeechEditBench encompasses seven atomic editing tasks, as well as compositional editing tasks that integrate multiple operations within a single instruction. We propose an anchor-based evaluation protocol that separately assesses the edit success of target attributes and the preservation of untargeted attributes, leading to three metrics: target success, preservation success, and joint success. Using this benchmark, we evaluate mainstream Speech LLMs and specialized speech editing systems. The results reveal three key findings: (1) no single model performs well across all editing dimensions; (2) closed-source Speech LLMs generally outperform open-source models; (3) compositional editing remains highly challenging, with even the most advanced models struggling to achieve high joint success. SpeechEditBench provides a rigorous diagnostic framework to identify bottlenecks in Speech LLMs, thereby facilitating the development of next-generation Speech LLMs with more robust and precise instruction-guided editing capabilities. Data and code will be released upon acceptance.
#### Advancing Electrolaryngeal Speech Enhancement Through Speech-Text Representation Learning
 - **Authors:** Ding Ma, Jinyi Mi, Fengji Li, Lester Phillip Violeta, Jiajun He, Wenchin Huang, Kazuhiro Kobayashi, Tomoki Toda
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2606.01905

 - **Pdf link:** https://arxiv.org/pdf/2606.01905

 - **Abstract**
 Objective: laryngectomees depend on an electromechanical device to generate electrolaryngeal (EL) speech. Compared with normal speech, EL speech suffers from severe distortion, limited phonetic variation, unnatural prosody, and temporal shifts, degrading naturalness and intelligibility. Although sequence-to-sequence (seq2seq) voice conversion (VC) based EL-speech-to-normal-speech conversion (EL2SP) is promising, substantial mismatches between EL and normal speech inevitably cause cumulative mapping errors that limit performance. To address this, we describe a novel representation learning framework integrating speech and text representations to improve mapping and reconstruction quality within a seq2seq VC model. Methods: our methodology comprises two main stages: 1) representation integration and learning, and 2) reconstruction training. A network capable of incorporating auxiliary text information is first constructed with pretrained modules to learn speech--text-based integrated representations. Then, an autoencoder-style reconstruction strategy finalizes EL2SP model to inherit these representations without increasing model complexity. We introduce three fusion strategies including middle-, input-, and hybrid-level fusion strategies that progressively enhance learning. Moreover, besides standard seq2seq VC objectives, an additional reconstruction loss on the integrated representation is introduced to refine representation transfer. Results: experiments under different EL2SP datasets consistently demonstrate that our methods, combined with data augmentations, outperform baselines relying solely on speech representations. Furthermore, progressive improvements with system design depth validate the effectiveness of our methods. Significance: the proposed methods provide an extensible and practical methodology for EL speech enhancement and assistive communication technologies.
#### Breaking the Pair: Evaluating Dyadic Interaction via Speaker Switching
 - **Authors:** Nishchay Nilabh, Neeraj Kumar Sharma
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2606.02185

 - **Pdf link:** https://arxiv.org/pdf/2606.02185

 - **Abstract**
 Speakers in dialogue continuously adapt their communicative behavior across acoustic, lexical, and semantic dimensions, a phenomenon known as conversational entrainment. Modeling this process requires representations that capture the global structure of interaction, yet prior approaches fail to disentangle dyad-specific patterns from speaker-specific traits, limiting their ability to capture true conversational adaptation. We address this with the Dyadic Distance Matrix (DDM), which encodes all pairwise similarities between the turns of two speakers over an entire conversation, capturing long-range cross-speaker dependencies. This raises a key question: does the DDM represent genuine interaction, or merely reflect individual speaker characteristics? We propose the speaker-switch test, a principled control in which one speaker's turns are replaced with those from an unrelated speaker drawn from a different conversation. This preserves turn-level statistics while disrupting the original dyadic coadaptation. The ability to distinguish real from switched DDMs thus directly evaluates whether the representation encodes interaction-specific structure. Across four embedding types and classifiers including ResNet-50 on the CANDOR corpus, real DDMs are consistently distinguishable from their switched counterparts. Comparisons with LibriSpeech show higher discriminability in read speech, highlighting the role of prosodic variability in naturalistic conversations. GradCAM analysis further reveals distinct structural signatures driving classification. These results establish the speaker-switch test as a robust diagnostic for validating representations of dyadic conversational interaction.
#### SiamCTC: Learning Speech Representations through Monotonic Temporal Alignment
 - **Authors:** SooHwan Eom, Mark Hasegawa-Johnson, ad Chang D. Yoo
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2606.02220

 - **Pdf link:** https://arxiv.org/pdf/2606.02220

 - **Abstract**
 Self-supervised speech representation learning has made significant progress through Siamese networks, which leverage different views of the same input. However, existing methods often require frame-wise alignment between these views, overlooking the broader linguistic context invariance across different speaking styles. We introduce SiamCTC, a framework that integrates Siamese networks with Connectionist Temporal Classification (CTC) to learn speech representations without strict frame-level correspondence. By employing CTC loss to establish flexible, monotonic alignments between differing temporal realizations of the same content, SiamCTC accommodates speed perturbations and other temporal augmentations. This design relaxes frame-wise constraints while preserving temporal coherence and enhancing robustness to speaking-rate variations in downstream tasks. Our experiments demonstrate that SiamCTC leads to more adaptable speech representations, particularly at diverse speaking rates.
#### Exploiting Noise Inseparability for Weakly-Supervised Discriminative Speech Denoising Using Noisy Targets
 - **Authors:** Matthew Maciejewski, Samuele Cornell
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2606.02327

 - **Pdf link:** https://arxiv.org/pdf/2606.02327

 - **Abstract**
 Speech denoising is an often necessary step not only for human listening, but also for downstream processing by systems lacking robustness to noisy, real-world acoustic conditions. Unfortunately, denoising is a problem where conventional in-domain supervised training is not trivial, as the training targets cannot be annotated by humans: producing a clean version of a naturally-noisy speech recording is itself the task to solve. Supervised training is typically performed through the artificial addition of noise to clean speech recordings, which can only be sourced from controlled domains, a significant limitation due to the poor out-of-domain generalization of neural networks. An alternative is noisy target training (NyTT), which simply replaces the clean speech with in-domain noisy recordings, with the hope that learning to remove the artificial noise will extend to the natural. Though having shown promising results, NyTT's training objective is not minimized by clean speech estimates. We show that by estimating the artificial noise in addition to the naturally-noisy speech, the undesirable optimum can actually be exploited: the residual noise in the speech estimate can be canceled by the noise estimate via simple subtraction. Crucially, the optimum is fully compatible with conventional artificial mixtures, enabling joint training using both types of data with consistent optimization targets, opening the door to improved domain adaptability. The effectiveness of our approach is demonstrated through WHAM! and CHiME-3-based benchmarks.
#### SoulX-Transcriber: A Robust End-to-End Framework for Multi-Speaker Speech Transcription
 - **Authors:** Yuhang Dai, Haopeng Lin, Zhennan Lin, Jiale Qian, Jun Wu, Hanke Xie, Hao Meng, Hanlin Wen, Chuang Ding, Shunshun Yin, Ming Tao, Lei Xie, Xinsheng Wang
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2606.02400

 - **Pdf link:** https://arxiv.org/pdf/2606.02400

 - **Abstract**
 Recent advances in Automatic Speech Recognition (ASR) and Large Language Models (LLMs) have significantly improved speech understanding capabilities. However, multi-speaker speech transcription remains challenging task, constrained by highly similar speaker voices, rapid turn-taking transitions, overlapping utterances and inaccurate speaker boundary segmentation. These challenges become particularly pronounced in real-world conversational audio, where speaker dynamics and acoustic conditions are highly variable. This technical report presents SoulX-Transcriber, a unified multi-speaker transcription system that jointly models speaker diarization (SD) and ASR within an LLM-based framework. SoulX-Transcriber adopts a two-stage training strategy to improve both speaker discrimination and transcription robustness. In the first stage, speaker-aware multi-task continuous pre-training enhances speaker representation learning and boundary perception. In the second stage, supervised fine-tuning further optimizes the model for accurate end-to-end speaker-attributed transcription under complex multi-speaker conditions. SoulX-Transcriber delivers strong performance and robustness across multiple public benchmarks, including AliMeeting, AISHELL-4, and AMI, while maintaining high adaptability to multi-domain scenarios.
#### DUET: Unified Dual-Space Emotion Control for Diffusion and Flow-Matching Driven Text-to-Speech
 - **Authors:** Xu Zhang, Longbing Cao, Zhangkai Wu
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2606.00066

 - **Pdf link:** https://arxiv.org/pdf/2606.00066

 - **Abstract**
 Diffusion and flow-matching based text-to-speech (TTS) models excel in naturalness but often lack explicit emotion control, as emotional signals remain entangled with speaker identity. We discover that emotion embedding emerges as a linearly decodable direction of frozen hidden states, nearly orthogonal to the direction embedding speaker identity. This inspires a plug-and-play framework DUET for emotion control over pretrained diffusion and flow-matching based TTS models. During generation, DUET unifies dual-space control to achieve fine-grained emotion intervention in a single per-step update: hidden space steering shifts generation along the target emotion direction, while mel-space guidance refines spectral details through gradients backpropagated from a differentiable vocoder. We validate DUET on five architecturally diverse pretrained TTS backbones across three datasets, where it outperforms 10 supervised state-of-the-art emotional TTS baselines across paradigms and achieves the highest human-rated emotion appropriateness. To further showcase its qualitative behavior, we deploy DUET on an Ameca humanoid robot, where it produces richly expressive emotional speech on the humanoid, demonstrating the strong potential for plug-and-play affective interaction for embodied agents.
#### SALSA: Speech Aware LLM Adaptation via Learned Steering Activation Vectors
 - **Authors:** Yekaterina Yegorova, Argyrios Gerogiannis, Haolong Zheng, Julia Hockenmaier, Chang D. Yoo, Mark A. Hasegawa-Johnson
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2606.00460

 - **Pdf link:** https://arxiv.org/pdf/2606.00460

 - **Abstract**
 Speech-aware large language models often generalize poorly to out-of-domain settings. We propose SALSA (Speech-Aware LLM Adaptation via Learned Steering Activations), a lightweight adaptation method that learns layer-wise steering vectors. Unlike commonly used steering approaches that rely on contrastive activation differences, SALSA directly optimizes steering vectors using a supervised objective. Across children's speech, multilingual speech, and Mandarin-English code-switching benchmarks, SALSA substantially improves performance over zero-shot inference and speech in-context learning baselines, achieving up to 46.8% relative improvements over zero-shot. Analysis further demonstrates that steering the encoder, particularly the later layers, is more effective than steering the LLM backbone. These findings suggest that steering improves downstream ASR performance by adapting higher-level acoustic and phonetic representations to better align with the pretrained language model representation space, rather than by modifying the decoder itself.
#### Sympatheia: Emotionally Adaptive Voice Assistant with Continuous Affect Conditioning
 - **Authors:** Sukru Samet Dindar, Riki Shimizu, Xilin Jiang, Nima Mesgarani
 - **Subjects:** Subjects:
Sound (cs.SD); Computation and Language (cs.CL); Human-Computer Interaction (cs.HC); Machine Learning (cs.LG); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2606.00851

 - **Pdf link:** https://arxiv.org/pdf/2606.00851

 - **Abstract**
 Empathetic spoken dialogue systems must infer a user's emotional state to respond appropriately, yet everyday speech often carries weak, neutral, or ambiguous affective cues. To address this, we introduce Sympatheia, a speech-to-speech dialogue framework conditioned on affect inferred from the user's speech and, when available, explicit affect specifications provided as a continuous valence--arousal (VA) control signal by a multimodal sensing module or user interface. To train our model, we construct Sympatheia-18k, an emotion-conditioned synthetic spoken dialogue corpus with 12 emotion anchors. This dataset includes an emotional split for learning affective speech behavior, and a neutral split that pairs emotionally neutral queries with multiple emotion-conditioned responses to isolate explicit emotion control in emotionally ambiguous cases. Empirical results show that Sympatheia outperforms speech conversational baselines in generating responses whose semantic content and spoken delivery are both emotionally appropriate. We further show that the same VA interface can integrate emotion estimates from diverse sensing modules, including facial expression, biosignals, and textual affect descriptions, improving response alignment when speech alone provides limited emotional evidence. These results suggest that continuous affect conditioning is an effective practical step for building emotionally adaptive voice assistants.
#### PolySpeech-100: A Large-Scale Benchmark for Speech Understanding Across 100+ Languages and Dialects
 - **Authors:** Sicheng Yang, Shulan Ruan, Shiwei Wu, Yu Liu, Lu Fan, Zhi Li, You He
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Artificial Intelligence (cs.AI); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2606.01016

 - **Pdf link:** https://arxiv.org/pdf/2606.01016

 - **Abstract**
 While End-to-End (E2E) Speech-Large Language Models (Speech-LLMs) are rapidly evolving, their evaluation methodologies remain limited to the era of simple transcription. Existing benchmarks suffer from three critical limitations: a pronounced bias towards high-resource languages, a focus on low-level recognition (ASR) rather than semantic reasoning, and a neglect of regional dialects. To bridge this gap, we introduce PolySpeech-100, a massive-scale benchmark designed to assess `native-level' speech comprehension across 110 linguistic variants. We employ a novel hybrid construction pipeline that augments gold-standard human recordings with instruction-driven synthetic speech, allowing us to cover 19 distinct Chinese dialects and over 80 low-resource languages. Extensive evaluation of 22 state-of-the-art models (including Gemini-3, GPT-Audio, and Qwen2.5-Omni) yields pivotal insights. First, we demonstrate that open-source E2E models outperform Cascade (ASR+LLM) systems on heavy dialects, proving that direct audio processing preserves critical paralinguistic cues and prosodic features (e.g., intonation, stress) that are often lost in standard transcription. Second, we reveal a significant performance gap: while commercial models maintain robustness, open-source models suffer catastrophic degradation on low-resource languages. Finally, counter-intuitively, we observe that under standard zero-shot settings, Chain-of-Thought prompting frequently degrades speech understanding performance for most evaluated models, revealing a potential modality alignment gap in current architectures. PolySpeech-100 establishes a rigorous standard for the next generation of inclusive, omni-capable Speech-LLMs. The data, demo, and code are publicly available at this https URL.
#### A 1000-hour EEG-EMG-audio dataset of Japanese speech production
 - **Authors:** Motoshige Sato, Ilya Horiguchi, Masakazu Inoue, Kenichi Tomeoka, Eri Hatakeyama, Yuya Kita, Atsushi Yamamoto, Ippei Fujisawa, Shuntaro Sasai
 - **Subjects:** Subjects:
Neurons and Cognition (q-bio.NC); Human-Computer Interaction (cs.HC); Sound (cs.SD); Audio and Speech Processing (eess.AS); Signal Processing (eess.SP)
 - **Arxiv link:** https://arxiv.org/abs/2606.01264

 - **Pdf link:** https://arxiv.org/pdf/2606.01264

 - **Abstract**
 We present a multimodal dataset of 1020 hours of simultaneously recorded scalp electroencephalography (EEG), facial electromyography (EMG), and speech audio from three healthy native Japanese speakers during open-vocabulary overt speech. Recordings were acquired with three EEG systems-an ultra-high-density system (this http URL) and two cap-type systems (this http URL and eegosports), spanning 62-128 channels-across many sessions over several months. Each session provides time-synchronized EEG, facial EMG, and audio, together with speech-event annotations and transcriptions. Although collected with speech decoding as a primary motivation, the dataset also supports work on multimodal signal processing, artifact modeling, longitudinal and cross-device adaptation, and EEG representation learning. Technical validation included power spectral density and event-related potential analyses across participants, devices, and tasks, which showed the expected 1/f spectral profile, task-related alpha-band attenuation, and time-locked evoked responses. The dataset is released in Brain Imaging Data Structure (BIDS) format via OpenNeuro under a CC0 waiver to support both speech-related and broader EEG research.
#### MURMUR: An Efficient Inference System for Long-Form ASR
 - **Authors:** Wei-Tzu Lee, Keisuke Kamahori, Baris Kasikci
 - **Subjects:** Subjects:
Machine Learning (cs.LG); Artificial Intelligence (cs.AI); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2606.01483

 - **Pdf link:** https://arxiv.org/pdf/2606.01483

 - **Abstract**
 Long-form automatic speech recognition (ASR) requires both high accuracy and low latency, but existing systems force a trade-off between the two. Chunk-based pipelines process audio in parallel windows for low latency, but lose cross-chunk context and need brittle heuristics to align speakers and timestamps at boundaries. Long-context ASR models resolve everything in a single pass for better accuracy, but are an order of magnitude slower. We propose Murmur, an inference system that overcomes this trade-off by operating at two levels. At the inter-chunk level, we revisit the chunk-based pipeline for modern long-context ASR, treating chunk size as a tunable hyperparameter, and show that intermediate chunk sizes strike a good balance of accuracy and latency. At the intra-chunk level, we exploit attention sparsity through a sliding window KV cache eviction policy applied to both output and speech tokens. On AMI-IHM, Murmur matches single-pass accuracy while reducing latency by 4.2x, with further gains from token eviction at less than 1% relative tcpWER degradation. The code of Murmur is available at this https URL.
#### Echo: A Joint-Embedding Predictive Architecture for Speaker Diarization and Speech Recognition in a Shared Latent Space
 - **Authors:** Louis Mouchon
 - **Subjects:** Subjects:
Sound (cs.SD); Artificial Intelligence (cs.AI); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2606.01909

 - **Pdf link:** https://arxiv.org/pdf/2606.01909

 - **Abstract**
 We present Echo, a proof-of-concept audio system built around a single 25 M-parameter ViT encoder. The encoder is pretrained with a JEPA objective and then specialised by stages to carry speaker identity, phonetic content, and dynamic source routing in the same 512-dimensional latent space, with no per-task fine-tuning at deployment. Light heads handle diarization (ArcFace + VBx) and dynamic source separation (null-target K-set prediction). On synthetic VoxCeleb2 mixtures with unknown K, the canonical stack reaches 15.00% blind DER, 97.80% PIT separation accuracy with +9.52 dB latent SI-SDR, and a +53.50-point speaker/content factorisation gap on a held-out k-NN probe. The point of Echo is not a new SOTA on any single task but the joint coexistence of three tasks on one encoder at this footprint. We document the design stage by stage, report the dead-ends, and identify the structural wall on end-to-end ASR through the VQ bottleneck that still bounds the PoC.


by Zyzzyva0381 (Windy). 


2026-06-02
