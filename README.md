# Showing new listings for Tuesday, 20 October 2026
Auto update papers at about 2:30am UTC (10:30am Beijing time) every weekday.


阅读 `Usage.md`了解如何使用此repo实现个性化的Arxiv论文推送

See `Usage.md` for instructions on how to personalize the repo. 


Keyword list: ['text-to-speech', 'text to speech', 'tts', 'LLM-based', 'speech', 'voice']


Excluded: []


### Today: 22papers 
#### SpikeAid: A Streaming Spiking Hearing Aid Framework for Joint Noise Reduction and Wide Dynamic Range Compression
 - **Authors:** Xiang Hao, Xinyi Chen, Hanwen Liu, Jibin Wu, Richard Hau Yue So, Kay Chen Tan
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2610.24696

 - **Pdf link:** https://arxiv.org/pdf/2610.24696

 - **Abstract**
 Modern hearing aids combine noise reduction (NR) with wide dynamic range compression (WDRC) to suppress background interference while compensating for the reduced auditory dynamic range associated with hearing loss. Conventional systems typically process NR and WDRC as separate stages, although the output of NR directly affects the signal levels and envelopes that determine subsequent compression. Recent end-to-end neural approaches can jointly optimize these processes, but their dense computation remains challenging for always-on hearing aids. In this work, we propose SpikeAid, a causal end-to-end spiking neural network (SNN) framework for joint NR and WDRC. SpikeAid consists of an NR-Net that constructs complementary speech and noise representations and a WDRC-Net that estimates the corresponding branch gains, with both components jointly optimized according to the final hearing aid objective. To capture the multiscale temporal structures required by joint NR and WDRC in a computationally efficient manner, we further introduce the Delayed Resonate-and-fire Neuron (DRN), which incorporates bio-inspired heterogeneous dendritic delays and resonating dynamics. Experimental results on the hearing aid task show that SpikeAid improves key objective metrics over cascaded processing and outperforms existing spiking speech-processing baselines. Evaluation on the public Intel Neuromorphic Deep Noise Suppression (N-DNS) Challenge further demonstrates state-of-the-art neuromorphic speech-denoising performance with DRN. Finally, we designed an FPGA-based neuromorphic hardware system for efficient deployment of SpikeAid, where it achieves over two orders of magnitude higher energy efficiency than conventional CPU/GPU implementations, demonstrating its potential for energy-efficient hearing aid applications.
#### Dual Task Vector Merging: Leveraging Emotion and ASR Knowledge for Low-Resource Alzheimer's Disease Detection
 - **Authors:** Yuan-Chao Li, Yin-Long Liu, Liu He, Sizhe Liu, Ji-Xuan Deng, Yin Zhang, Zhen-Hua Ling, Jiahong Yuan
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Computation and Language (cs.CL); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2610.24870

 - **Pdf link:** https://arxiv.org/pdf/2610.24870

 - **Abstract**
 Alzheimer's Disease (AD) detection from speech benefits from both paralinguistic and linguistic cues, yet model training is frequently constrained by limited data. Fine-tuning a large pretrained speech encoder can easily overfit, while a frozen probe leaves the encoder's task-relevant knowledge underutilized. Therefore, we propose Dual Task Vector Merging, a parameter-efficient method that adapts a frozen wav2vec 2.0 backbone using task knowledge learned outside the AD task. Specifically, we inject two task vectors, derived as weight deltas from emotion- and ASR-fine-tuned wav2vec 2.0 models, into each transformer layer. During AD training, only layer-wise merging coefficients and a linear classification head are optimized, updating less than 0.1% of the parameters required for full fine-tuning. Speaker-independent experiments on ADReSS and ADReSSo show that dual merging improves over frozen probes, LoRA, and full fine-tuning, in the main binary settings, and gives the best results when combined with LoRA. Evaluation on PROCESS further shows that our advantages transfer to three-class HC/MCI/dementia classification when paired with LoRA. Experiments under reduced training data (20%, 40%, and 80%) confirm reliable gains in low-resource settings.
#### Learning Speaker-Activity-Conditioned WavLM Speaker Embeddings with Real Conversational Data
 - **Authors:** Prachi Singh, Nguyen Hai Phong, Petr Pálka, Jiangyu Han, Marc Delcroix, Shota Horiguchi, Naohiro Tawara, Lukáš Burget
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2610.25183

 - **Pdf link:** https://arxiv.org/pdf/2610.25183

 - **Abstract**
 End-to-end neural diarization with vector clustering (EEND-VC) is a strong framework for long-form conversational diarization, where speaker embeddings are used to link local speaker decisions across windows. Conventional EEND-VC systems extract these embeddings by cutting out regions assigned to each local speaker, preferably using only non-overlapped speech. This works well when enough clean target-speaker speech is available, but it discards useful speaker evidence in overlapped regions and becomes unreliable when a local speaker appears mostly in overlap. We propose a speaker-activity-conditioned embedding extractor that adapts a pretrained WavLM-MHFA speaker verification model to extract target-speaker embeddings directly from full conversational windows. Inspired by the recently proposed diarization-conditioned Whisper (DiCoW) model for target-speaker ASR, frame-level diarization-dependent transformations inject local speaker-activity information into intermediate WavLM representations while preserving the pretrained initialization. Experiments on multiple conversational diarization benchmarks show that fine-tuning with limited real meeting data reduces speaker-confusion errors and improves EEND-VC diarization performance. The results further indicate that real conversational data provides an effective alternative to large-scale simulated-mixture training for diarization-aware speaker embedding extraction.
#### Probing and Mitigating Hallucinations in Speech-augmented Language Models for Automatic Speech Recognition via Small Language Models
 - **Authors:** Bi-Cheng Yan, Jhih-Rong Guo, Fu-An Chao, Berlin Chen
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2610.25351

 - **Pdf link:** https://arxiv.org/pdf/2610.25351

 - **Abstract**
 The widespread success of large language models in natural language processing has spurred the development of speech-augmented language models (SLMs) to extend their reasoning capabilities from textual inputs to spoken ones. Despite impressive performance, SLMs remain susceptible to hallucinations, posing challenges in practical utility. To this end, we first probe hallucinations in SLMs for automatic speech recognition (ASR) via the lens of causal mediation and behavioral analysis. To mitigate fabricated ASR hypotheses, we propose AudioSLM, a compact yet effective ASR framework built upon a small language model, which leverages alignment cues and cross-attention layers to synergistically bolster acoustic modeling and cross-modal interactions. Experiments on LibriSpeech reveal that hallucinations arise from multi-head self-attention modules with an excessive attention bias towards textual tokens. AudioSLM considerably eases ASR hallucinations and outperforms some LLM-based ASR models.
#### PronCFT: Pronunciation Contextual Fine-Tuning for Contextual Biasing of Out-of-Vocabulary Words in LLM-Based ASR
 - **Authors:** Yifan Xu, Kaixun Huang, Mengzhe Geng, Mingyu Cui, Shen Huang, Shidong Shang, Xunying Liu
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2610.25766

 - **Pdf link:** https://arxiv.org/pdf/2610.25766

 - **Abstract**
 LLM-based ASR systems often replace rare and out-of-vocabulary (OOV) words with familiar words that sound similar. Two factors contribute to this error: the model has not learned how an unseen word's sound maps to its spelling, and it may favor a familiar word over the one spoken, a failure we call prior suppression. Providing candidate pronunciations alone may not help: the model can treat a phoneme annotation as a marker that raises a candidate's output probability without checking its pronunciation against the speech. We propose PronCFT, which uses fixed phoneme sequences and trains the model to transcribe speech into phonemes, match them to context candidates, and output the matched spelling. During training, we alter candidate spellings while keeping their pronunciations unchanged, discouraging the model from choosing a familiar spelling without checking how the candidate sounds. A curriculum then gradually removes intermediate outputs to reduce decoding latency and prevent phoneme transcription errors from affecting the final text. The model's retained ability to use pronunciation cues suggests that this ability can be internalized. On an OOV test set filtered from open-source benchmarks, the final model, PronCFT-C, significantly improves recall from 59.89% to 83.57% (+23.68 pp). Wrong-phoneme and spelling-interference tests indicate that the gain depends on phoneme content rather than the presence of an annotation.
#### MS-Bench: A Condition-Stratified Multi-Speaker ASR Benchmark
 - **Authors:** Chunjiang He, Zheng Zi, Bingshen Mu, Yurun Li, Jingyi Wang, Zhonghua Fu, Lei Xie
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2610.25824

 - **Pdf link:** https://arxiv.org/pdf/2610.25824

 - **Abstract**
 Multi-speaker automatic speech recognition (MSASR) has advanced rapidly with end-to-end speech large language models. However, existing benchmarks cover limited ranges of specific multi-speaker conditions and scenarios, making it difficult to reveal system limitations under diverse conditions and scenarios. We introduce MS-Bench\footnote{\href{this https URL}{Demo page: this https URL}}, a condition-stratified benchmark for fine-grained MSASR evaluation. MS-Bench integrates both public and internal benchmarks and characterizes each recording along five dimensions: speaker number, overlap ratio, speaker similarity, speaker turn interval, and acoustic quality. We stratify these dimensions into condition ranges and construct MS-Bench with broad coverage of conditions, scenarios, languages, and recording devices. Evaluating representative commercial, open-source, and cascaded systems, we find that overlap affects both lexical recognition and speaker attribution, while other factors exhibit more task-specific effects on recognition, attribution, or temporal prediction. These results demonstrate that aggregate metrics alone obscure important MSASR failure modes and highlight the value of condition-stratified evaluation. MS-Bench aims to reveal condition-specific limitations of current MSASR systems and guide future improvements.
#### SECURE-AI: Multi-Signal Uncertainty Quantification for Trustworthy Acoustic Cough Detection
 - **Authors:** Sejal Solkhan, Feiya Xiang, Mayur Sanap, Geet Khatri, Edgar Lobaton
 - **Subjects:** Subjects:
Machine Learning (cs.LG); Artificial Intelligence (cs.AI); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2610.23937

 - **Pdf link:** https://arxiv.org/pdf/2610.23937

 - **Abstract**
 Acoustic models for cough detection achieve high accuracy on curated benchmarks but fail silently in deployment, where speech, environmental noise, and audio unlike the training data induce confident misclassifications. In health monitoring, a confidently wrong prediction is worse than no answer at all. We present SECURE-AI (Self-aware Estimation of Cough Uncertainty for Reliable Evaluation of AI), a multi-signal uncertainty quantification framework that wraps a frozen MobileNetV3 cough classifier and estimates when its predictions should not be trusted, combining Monte Carlo Dropout, Temperature Scaling, and Mahalanobis distance in the embedding space. On a held-out split of 12,161 Coswara recordings, the probability-space methods are individually insufficient: Monte Carlo Dropout assigns lower uncertainty to incorrect than to correct predictions, and temperature scaling improves calibration but not discrimination. The strongest signal is embedding-space distance: Mahalanobis distance separates true positives from false positives (means 44.7 versus 63.4; Cohen's $d = 0.94$; AUROC 0.86) and, among predicted non-coughs, false negatives from true negatives (AUROC 0.82), so the same signal can flag both false alarms and missed coughs. At a descriptive operating point selected from labeled test data, retaining 90 percent of true-positive detections removes about two-thirds of false positives and raises precision from 59.9 to roughly 80 percent. A rejection threshold tuned on one split does not transfer to another (validation F1 of 82 percent falls to 34 percent on test), so we report threshold-independent AUROC as the main measure. An error taxonomy shows that failures are systematic, with most false positives related to speech. The pipeline runs on Raspberry Pi hardware costing 30 to 90 US dollars. Probabilistic calibration alone is thus inadequate for reliable cough detection.
#### SynerGest: Temporal-Spatial Synergistic Gesture Generation for Dyadic Conversations
 - **Authors:** Chengxiang Guo, Jian Wang, Lanmiao Liu, Zhan Tang, Junhua Fei, Xiao Li
 - **Subjects:** Subjects:
Computer Vision and Pattern Recognition (cs.CV); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2610.24031

 - **Pdf link:** https://arxiv.org/pdf/2610.24031

 - **Abstract**
 Co-speech gesture generation in dyadic conversations for creating immersive virtual avatars remains highly challenging. Existing models often struggle to balance kinematic stability with rhythmic responsiveness and suffer from semantic entanglement, leading to either generic mean-pose regression or unnatural micro-jitters. To address these challenges, we propose SynerGest, a high-fidelity dyadic gesture generation framework grounded in Contrastive Flow Matching (CFM). Specifically, our CFM objective explicitly penalizes mismatched semantic pairs during training, discouraging spatial shortcuts and over-smoothed predictions while improving conditional semantic alignment. Furthermore, to overcome architectural bottlenecks, we introduce the Impulse-Trend Synergistic Attention (ITSA) module. Grounded in the key insight of decoupling "what moves" from "when it moves," ITSA models sharp, speech-driven rhythmic impulses separately from continuous kinematic trends, supporting precise audio-motion synchronization and global structural stability. Extensive qualitative and quantitative evaluations on two public datasets demonstrate that SynerGest achieves state-of-the-art performance, delivering contextually responsive and kinematically fluid dyadic interactions.
#### From Spectrograms to Shallow Speech SSL: Audio-Frontend Design for Generator-Disjoint and Low-Latency Audio--Visual Deepfake Detection
 - **Authors:** Qiwu Wen, Jahangir Alam, Eric Charton
 - **Subjects:** Subjects:
Sound (cs.SD); Artificial Intelligence (cs.AI); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2610.24138

 - **Pdf link:** https://arxiv.org/pdf/2610.24138

 - **Abstract**
 Video-mediated conversational systems rely on synchronized speech and facial video, making audio--visual input integrity important for reliable perception. We study audio-frontend design in a fixed MRDF (Modality-Regularization-based DeepFake)-based pipeline with AV-HuBERT visual processing and an MSCT (multi-scale cross-modal transformer) fusion encoder. We compare spectrogram-style log-filterbank features with frozen wav2vec~2.0 and WavLM representations, dual-SSL fusion, executed depth, and shallow-layer aggregation. Across three training runs, full wav2vec~2.0 improves mean macro AUROC from $84.19\%$ to $87.19\%$ on generator-disjoint MMDF. Six-layer execution further improves detection, and learned non-uniform aggregation adds $2.53$ percentage points, reaching $90.61\%$. Shallow execution reduces measured inference latency relative to full-depth wav2vec~2.0 across the tested input durations; first-second evaluation reaches $86.85\%$ macro AUROC. In a single-run comparison, dual-SSL concatenation outperforms either single stream, while one-way cross-attention yields lower detection performance and higher latency than concatenation. Layer-weight analysis shows greater emphasis on early speech layers, while removing the selector's visual input has little effect on detection performance.
#### Pronouncing Hard Words: Phoneme Adaptation of a Pretrained Conversational Text to Speech Model
 - **Authors:** Yashish M. Siriwardena, Dinesh K. Gnanasekaran, Tanmay Laud, Shinji Watanabe, Subhabrata Mukherjee, Shanil Puri
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2610.24164

 - **Pdf link:** https://arxiv.org/pdf/2610.24164

 - **Abstract**
 Voice agents built as cascaded automatic speech recognition (ASR)->large language model (LLM)-> text-to-speech (TTS) pipelines increasingly rely on conversational TTS: streaming, context conditioned models that adapt delivery to dialogue history. Such models are natural and expressive but offer little explicit control over how rare, trust-critical drug names are pronounced, because these entities are too infrequent, and their vocabulary too fast-growing, to be covered by any amount of pre-, mid-, or post-training. We retrofit phoneme controllability onto a pretrained conversational TTS model without any pre/mid-training, using an ARPAbet vocabulary exten-sion, continued phoneme-augmented supervised fine-tuning (SFT), and a training-free inference-time acoustic-biasing step. On a 7,500 clip clinical drug-pronunciation evaluation, our model raises strict Gemini-judged correctness from 54.6% (no phoneme augmentation) to 90.7%, outperforming every conversational and stateless TTS baseline on the automatic metric. Because no training stage sees a drug name, the entire evaluation is out-of-domain, so the gains reflect learned pronunciation rather than memorization. In a blinded, randomized human study, adult listeners judge 84% of our clips correct, versus 82% for the strongest baseline, with overlapping confidence intervals on a stratified subset. Across all 250 matched clips, Gemini agrees with the human majority on 84.8% of binary labels (k{appa}=0.57).
#### SENTRY: A Streaming Edge Network Trust Engine for Real-Time Audio Deepfake Detection
 - **Authors:** Chirantan Banerjee, Soumita Chatterjee, Bhargab B. Bhattacharya, Hafizur Rahaman
 - **Subjects:** Subjects:
Sound (cs.SD); Artificial Intelligence (cs.AI); Hardware Architecture (cs.AR); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2610.24468

 - **Pdf link:** https://arxiv.org/pdf/2610.24468

 - **Abstract**
 Real-time voice cloning has enabled large-scale streaming deepfake and vishing attacks, while existing Network Detection and Response (NDR) solutions rely on GPU-based post-processing that requires buffered audio and generalizes poorly to unseen speech synthesis models. This paper presents SENTRY, a hardware-oriented, streaming trust engine for inline deepfake detection at the network edge. SENTRY employs an integer lifting-wavelet front end together with a bank of multiplier-free transforms, including wavelet-packet envelopes, Walsh-Hadamard sequency profiles, waveform micro-statistics, and long-term spectral features, designed so that acoustic analysis needs only integer add, shift, compare and bitwise operations. A supervised gradient-boosted tree classifier is combined with an unsupervised Mahalanobis out-of-distribution detector trained solely on authentic speech, and their outputs are fused by a lightweight logistic layer to produce graduated LOG, CHALLENGE, or DROP actions. Under a closed-set protocol, SENTRY reaches 98.75% accuracy at a 1.25% equal-error rate (EER) on WaveFake, with perfect separation of HiFi-GAN and WaveGlow; its supervised track alone achieves an AUC of 0.9994 (fused: 0.9993). On ASVspoof 2019 LA (attacks A01-A06) over an 8 kHz mu-law telephony channel, it reaches 94.75% accuracy at a 5.05% EER, with a supervised-track AUC of 0.9948 (fused: 0.9884) and per-attack EERs below 7%. In our prototype, the feature front end runs as bit-exact integer code on one Arm Cortex-A53 core, and the classifier and fusion stage are synthesized for a Xilinx Zynq UltraScale+ MPSoC using no DSP slices, returning a decision 220 ns after a feature vector is ready. These results indicate that deepfake screening can be placed inline without GPUs or floating-point hardware; generalization to unseen generators is left to future work.
#### ASR Personalization in Pathological Voice: Accuracy, Efficiency, and Duration Effects
 - **Authors:** Giuseppe Giaimis, Sergei Katkov, Muhammad Azfar Yaqub, Antonio Liotta
 - **Subjects:** Subjects:
Sound (cs.SD); Artificial Intelligence (cs.AI); Computation and Language (cs.CL); Machine Learning (cs.LG); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2610.24572

 - **Pdf link:** https://arxiv.org/pdf/2610.24572

 - **Abstract**
 Automatic Speech recognition (ASR) struggles with Pathological speech due to persistent speaker-specific mismatch. In clinical and accessibility scenarios, only limited target-speaker data is typically available, making full model retraining impractical. We present a component-level analysis of LoRA adaptation in Whisper, investigating how adapter placement interacts with the model size, training efficiency and utterance-duration in single-speaker settings. Our experimental comparative study shows that while scaling reduces errors, it does not eliminate mismatch. Furthermore, the full attention LoRa provides the most reliable accuracy, and the best time--accuracy trade-off. Improvements vary across utterance length, with medium-length recordings benefiting the most, while short and long utterances remain challenging. These findings highlight the importance of adaptation placement and duration-dependent evaluation, under challenging, data-scarce personalization scenarios.
#### VoxDub: A Bilingual Lip-Synchronized Audio-Visual Dataset for Automatic Video Dubbing
 - **Authors:** Kaidi Wang, Wenhao Guan, Peijie Chen, Weijie Wu, Xuetao Lin, Lin Li, Qingyang Hong
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2610.25259

 - **Pdf link:** https://arxiv.org/pdf/2610.25259

 - **Abstract**
 Automatic video dubbing (AVD) requires temporally aligned speech, transcripts, and video, yet existing corpora vary in audio quality, transcription reliability, and speaking-face visibility. We introduce VoxDub, a bilingual lip-synchronized audio-visual dataset containing 766K clips and approximately 1.58K hours of in-the-wild Chinese and English recordings, covering varied speakers, scenes, speaking styles, and linguistic content. Its curation pipeline combines speech-centered segmentation with dual-ASR transcription checking, speech quality filtering, face visibility checks, and lip-speech synchronization verification. Text-to-speech (TTS) experiments show that training on VoxDub yields lower word error rates and higher speaker similarity than training on the compared audio-visual corpora, LRS3 and CN-CVS. AVD experiments show broadly comparable speech quality and subjective synchronization. These results support the utility of VoxDub for TTS and AVD training. More details are available on the project website (this https URL).
#### Does Expert Supervision Improve Text-to-Speech? An Empirical Study of Preference Optimization
 - **Authors:** Minh Duc Vu, Junchuan Zhao, Ye Wang
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2610.25292

 - **Pdf link:** https://arxiv.org/pdf/2610.25292

 - **Abstract**
 Preference optimization promises to improve text-to-speech (TTS) systems by learning which speech outputs are better. External expert speech seems a natural source of preferred responses, especially when student generations contain errors. Yet demonstration quality does not establish whether the student can benefit from that supervision, or whether better target fitting will produce better speech. We examine this gap across two TTS students using real and synthetic expert speech, comparing preference objectives and direct versus bridge supervision while tracking fixed-sequence likelihoods and generation metrics. With expert supervision held fixed, objectives differ markedly in their sensitivity to preference-score scaling and the resulting generation quality. Likelihood diagnostics characterize these responses, but cannot establish their benefit: expert likelihood can increase while intelligibility worsens, or decrease while it improves. Bridge supervision highlights a distinction between mitigating degradation and improving the student: it can recover performance lost under direct training without consistently surpassing the pretrained base. Route benefits vary across objectives, students, and expert sources; neither route consistently dominates. Together, these findings distinguish demonstration quality, target fitting, and generated-speech quality as complementary criteria for assessing expert supervision.
#### CASE: Complementary Articulation-Aware Speech Encoder to Enhance Speech-Driven 3D Facial Animation
 - **Authors:** Jaehong Myung, Seungho Eum, Taegon Kim, Hyunjung Chung, Unsang Park
 - **Subjects:** Subjects:
Computer Vision and Pattern Recognition (cs.CV); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2610.25342

 - **Pdf link:** https://arxiv.org/pdf/2610.25342

 - **Abstract**
 General-purpose self-supervised speech encoders provide strong acoustic representations, but their pretraining objectives are not explicitly aligned with the visible articulatory dynamics required by speech-driven 3D facial animation. We propose CASE (Complementary Articulation-aware Speech Encoder), a complementary speech encoder adapted by contrastively aligning short speech clips with continuous 3D lip-mesh VQ-VAE latents. After adaptation, the geometry branch is removed, and the audio-only CASE representation is fused with the baseline SSL speech representation through residual cross-attention, while retaining the motion-decoder architecture and objective. Across four VOCASET baselines, CASE reduces mean lip vertex error by 2.8-6.0%, and mean full-face vertex error decreases in every system. Under the same fusion protocol, CASE yields lower mean LVE than the unadapted AV-SE stream on all four generators, with model-dependent margins. Because adaptation uses Korean speech-geometry pairs and evaluation uses English VOCASET, the result is consistent with cross-language downstream transfer under a shared English-pretrained audio-visual prior.
#### MixiSpeech: Progressive State Distillation for One-Step Vietnamese TTS
 - **Authors:** Tran Quoc Bao Le, Binh Mai, Hung Dinh, Lam Nguyen, Cong Tran
 - **Subjects:** Subjects:
Sound (cs.SD); Multimedia (cs.MM); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2610.25394

 - **Pdf link:** https://arxiv.org/pdf/2610.25394

 - **Abstract**
 Conditional-flow text-to-speech (TTS) models can achieve high synthesis quality with relatively few sampling steps, but collapsing the remaining trajectory to a single update can still cause substantial quality degradation. We propose MixiSpeech, a progressive state distillation framework that converts a multi-step conditional-flow acoustic model into a one-step generator while preserving its architecture and speaker conditioning. Rather than training the one-step model directly, MixiSpeech progressively reduces the global sampling budget through a 10-4-2-1 curriculum, with each student trained to match local teacher-integration endpoints using a masked state-matching objective. We instantiate the framework on a 710-speaker Vietnamese Matcha-TTS model trained from scratch on PhoAudiobook. On 100 shared test text-speaker pairs, direct one-step inference substantially degrades synthesis quality, obtaining 1.6846 UTMOS22-Strong and 6.21% WER compared with 2.2923 and 4.65% for the 10-step teacher. In contrast, the distilled one-step model achieves 2.2678 UTMOS22-Strong, 4.63% WER, and 2.83% CER, recovering most of the quality lost at the one-step operating point while requiring only a single acoustic vector-field evaluation. End-to-end inference time decreases from 0.0662 s for the 10-step teacher to 0.0312 s for MixiSpeech, corresponding to a 2.12 measured speedup. These results show that progressive state distillation can make one-step conditional-flow TTS practical without relying on a large multi-step sampling budget. Project page: this https URL
#### Manifold-Constrained Residual Adaptation with Frame-Adaptive Layer Aggregation for Speaker Verification
 - **Authors:** Zezhong Jin, Xiaoyu Wang, Zhe Li, Chong-Xin Gan, Siqing Qin, Man-Wai Mak, Kong Aik Lee
 - **Subjects:** Subjects:
Sound (cs.SD); Artificial Intelligence (cs.AI); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2610.25892

 - **Pdf link:** https://arxiv.org/pdf/2610.25892

 - **Abstract**
 Large self-supervised speech models provide powerful representations for speaker verification, but full fine-tuning is expensive. Most conventional adapters insert small trainable modules to modify the output of each layer, while keeping the original residual connections unchanged. Building on our preliminary study of manifold-constrained Hyper-Connections in speaker encoders, we explore parameter-efficient adaptation of both residual paths and feature selection across layers. We first propose a Manifold-Constrained Residual Adapter (MCRA), which splits the hidden dimensions of each Transformer residual state into N groups and uses a small Sinkhorn-normalized matrix to exchange information among them. A learnable gate controls the balance between the original and mixed residual paths. The gate adds only one scalar parameter at each residual connection and initially favors the original path, allowing gradual adaptation. We further propose Frame-Adaptive Layer Aggregation (FALA), an attention-based aggregation method that uses a single learned query shared across all utterances to compute layer weights separately for each speech frame. Because the layer representations depend on the input, the resulting weights vary across both utterances and frames, unlike the fixed weights of a conventional layer-weighted sum. Compared with a frozen WavLM baseline using a fixed layer-weighted sum, our method achieves relative EER reductions of 22.4% on VoxCeleb1-O and 2.9% on CN-Celeb, with trainable adaptation parameters equivalent to only 0.0009% and 0.0016% of the WavLM-Large parameters, respectively.
#### Multilingual Voicemail and Answering Machine Detection via Pretrained Audio Embeddings and Gradient Boosting Machines
 - **Authors:** Kemal Altwlkany, Muamer Paric
 - **Subjects:** Subjects:
Sound (cs.SD); Machine Learning (cs.LG); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2610.26100

 - **Pdf link:** https://arxiv.org/pdf/2610.26100

 - **Abstract**
 Answering machine detection (AMD) is a fundamental task in telephony that classifies whether an answered call is picked up by a human or a voicemail (machine). Despite its widespread use across cloud communication platforms, existing approaches rely on proprietary systems with undisclosed architectures, or require computationally expensive models that cannot operate in real time. We propose a lightweight AMD system that combines pretrained audio embeddings with gradient boosting machines, trained to prioritize the correct detection of human-answered calls. We evaluate seven models across two datasets: three models that we propose, a proprietary industry model, two prior academic approaches, and a large language model baseline that relies on automatic speech recognition. Furthermore, we release OpenAMD, the first publicly available AMD dataset comprising of more than 500 recordings in 7 languages. Our best model achieves the highest precision across both datasets while maintaining low inference latency. We assess the robustness of models to additive noise and packet loss, and perform tests of statistical significance to discover per-language and per-gender model performance. Finally, we evaluate our model on an out-of-domain task of detecting robocalls, where it achieves 94.34% recall without any additional training, which is comparable to the industry standard.
#### Applicability and Generalization of Audio Fingerprinting Methods to Speech
 - **Authors:** Kemal Altwlkany, Elmedin Selmanovic, Sead Delalić, Emanuel Lacic
 - **Subjects:** Subjects:
Sound (cs.SD); Multimedia (cs.MM); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2610.26112

 - **Pdf link:** https://arxiv.org/pdf/2610.26112

 - **Abstract**
 Audio fingerprinting methods identify audio based on its content, even when the content has been possibly corrupted. Despite their applicability and usage in the speech domain, research and advancements in audio fingerprinting traditionally focus on music and are almost exclusively evaluated on music. We provide insights into speech-oriented audio fingerprinting by benchmarking the best performing audio fingerprinting models on speech data while taking into consideration the effects of noise, reverberation, temporal changes, pitch modifications, and lossy compression. Our findings indicate that audio fingerprinting methods based on self-supervised learning outperform classical approaches on speech data, while the performance of these models on speech is inferior compared to the performance that they achieve on music. Results also include an analysis of false positives, showing that fingerprinting methods are more likely to match with an audio of the same speaker rather than an audio of the same text, when mismatches occur.
#### AudioScorer: Learning Question-Aware Token Relevance for Large Audio Language Models
 - **Authors:** Marwan Farag, Bin Yang
 - **Subjects:** Subjects:
Sound (cs.SD); Computation and Language (cs.CL); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2610.26258

 - **Pdf link:** https://arxiv.org/pdf/2610.26258

 - **Abstract**
 Large Audio Language Models (LALMs) reason over speech, environmental sounds, music, and other acoustic information. However, their audio encoders produce dense token sequences that make long-form audio costly to process. Existing audio pruning methods reduce this cost using heuristics such as attention, representation similarity, or temporal sampling, but do not explicitly learn which tokens are most useful for a given question. We propose AudioScorer, a lightweight model that scores audio tokens according to the user question. It is trained by distilling the ranking of audio tokens under the frozen LALM's answer-gradient saliency, together with downstream answer supervision. Compared with other pruning methods, AudioScorer achieves the best or tied-best result at 10% token retention on 7 of 10 semantic and acoustic tasks, including speech recognition, emotion recognition, and audio-scene classification.
#### Label Authority in Speech Emotion Recognition: Intended and Perceived Ground Truth in Emozionalmente
 - **Authors:** Marco Giordano, Stefano Giacomelli, Claudia Rinaldi, Fabio Graziosi
 - **Subjects:** Subjects:
Sound (cs.SD); Computation and Language (cs.CL); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2610.26446

 - **Pdf link:** https://arxiv.org/pdf/2610.26446

 - **Abstract**
 Ground truth in Speech Emotion Recognition (SER) is often presented as a single categorical fact, yet an acted emotion and the emotion heard by listeners describe different aspects of the same recording. We examine this distinction in Emozionalmente, an acted Italian corpus that provides both an elicited production label and five listener judgements for every clip. Using frozen XLS-R, HuBERT, and WavLM representations with matched probes, we compare three label authorities: the expressed condition, the listeners' hard plurality, and their complete vote distribution. The results tell a consistent story. Recovering the expressed condition is easier than predicting a tie-broken listener plurality, but that gap compares two different questions rather than revealing a general loss of SER ability. As listener agreement weakens, the plurality becomes an increasingly brittle hard target. Training on the full vote distribution does not substantially change which class wins the model's hard decision, but it produces probability assignments that more closely follow the alternatives selected by listeners, especially for ambiguous clips. No authority therefore replaces the others. Expressed labels describe how speech was produced, pluralities provide an operational single-class decision, and vote distributions preserve perceptual uncertainty. For human-facing Internet of Sounds applications, evaluations should state which authority is being predicted, retain its aggregation rule and agreement information, and use distributional scores when disagreement is part of the target.
#### TeleASR-Amharic: A Benchmark for Robust Amharic ASR in Realistic Telecommunication Conditions
 - **Authors:** Dagnachew Mekonnen Marilign, Enrique Molero, Abrham Haile, Louis Powell, Israel Abebe Azime
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2610.26448

 - **Pdf link:** https://arxiv.org/pdf/2610.26448

 - **Abstract**
 The real-world usability of ASR models remains challenging, particularly for low-resource languages, where existing speech resources are often collected in clean studio environments. In this work, we introduce TeleASR-Amharic, a benchmark for evaluating Amharic ASR under various acoustic and telecommunication conditions. We analyze the major degradations observed in real telecommunications recordings and develop a benchmark that covers multiple channel conditions, environmental disturbances, and telecom-domain variations. Our results show that telecommunication degradation can substantially reduce recognition performance, making even high-performing ASR models impractical under severe conditions. For example, the WER of the best-performing model increases from 0.23 to 0.79 under severe telecommunication degradation, corresponding to a relative increase of 243% in recognition errors. We further investigate the effects of different noise and channel distortions and show that scaling training data and applying carefully designed telecommunication-aware augmentation substantially improve robustness. In particular, our best training configuration achieves a 23.6% relative WER reduction under extreme noise conditions. Together with the release of our telecommunications-domain data, these findings provide a reproducible benchmark for measuring progress in Amharic ASR while highlighting open challenges in developing robust speech technologies for real-world deployment.


by Zyzzyva0381 (Windy). 


2026-10-20
