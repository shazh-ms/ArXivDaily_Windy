# Showing new listings for Tuesday, 1 September 2026
Auto update papers at about 2:30am UTC (10:30am Beijing time) every weekday.


阅读 `Usage.md`了解如何使用此repo实现个性化的Arxiv论文推送

See `Usage.md` for instructions on how to personalize the repo. 


Keyword list: ['text-to-speech', 'text to speech', 'tts', 'LLM-based', 'speech', 'voice']


Excluded: []


### Today: 10papers 
#### Beyond Speech: Dual-Domain SSL Fusion for Unified All-Type Audio Deepfake Detection
 - **Authors:** Cunhang Fan, Junqin Cao, Tian Gao, Zhipeng Xie, Jun Xue, Zhao Lv, Xin Fang
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2608.29021

 - **Pdf link:** https://arxiv.org/pdf/2608.29021

 - **Abstract**
 Unified all-type audio deepfake detection aims to determine whether an input clip is real or fake when its audio type may be speech, environmental sound, singing voice, or music. Existing speech-centric or type-dependent solutions are insufficient for this setting because the test-time audio type is unknown, while the required output is still a single binary decision. To address these issues, this paper proposes a dual-domain SSL fusion method that maps heterogeneous audio into a shared binary authenticity space. EAT-large and wav2vec 2.0 XLS-R-300M are used as complementary SSL feature sources, providing broad acoustic and event-level representations as well as waveform-level, vocal, and speech-sensitive representations. Layer-wise weighted fusion integrates multi-level artifacts from different transformer depths, while token-level fusion forms a unified feature pool without enforcing frame-level alignment between the two SSL streams. The fused tokens are summarized by multi-head attentive statistics pooling and classified with a binary MLP head. With conservative speech refinement applied on top of this unified core detector, the submitted system achieves 95.58% Macro-F1 on the AT-ADD Track 2 evaluation set and ranks second in the challenge.
#### Weakly Supervised Tabla Stroke Transcription via an Adaptive Dynamic Rhythm Language Model (ADRM)
 - **Authors:** Rahul Bapusaheb Kodag, Vipul Arora
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2608.30314

 - **Pdf link:** https://arxiv.org/pdf/2608.30314

 - **Abstract**
 Tabla Stroke Transcription (TST) is central to the analysis of rhythmic structure in Hindustani music, yet it remains challenging due to complex and dynamic rhythmic organization and the scarcity of strongly annotated data. Existing approaches largely rely on fully supervised learning with onset-level annotations, which are costly and impractical at scale. This work addresses TST in a weakly supervised setting, using only symbolic stroke sequences without temporal alignment of onsets. We propose a framework that combines a Connectionist Temporal Classification (CTC)-based acoustic model with a sequence-level rhythmic language model for rescoring, similar to that used in automatic speech recognition. The acoustic model produces a decoding lattice, which is refined using an Adaptive Dynamic Rhythm Language Model (ADRM) that combines $t\bar{a}la$-conditioned symbolic rhythmic regularities with local stroke dynamics. Moreover, we release a new performance-recorded tabla dataset, named \emph{Tabla Improvisation Dataset}, along with a complementary synthetic dataset for sequence-level weakly supervised TST. Experiments demonstrate consistent and substantial reductions in stroke error rates with ADRM compared to those with acoustic-only decoding, confirming the benefit of incorporating symbolic rhythmic regularities during lattice rescoring for accurate transcription.
#### Towards Balanced Spectral Reconstruction: Spectrally Adaptive Loss for Streaming Speech Enhancement
 - **Authors:** Haixin Zhao, Nilesh Madhu
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2608.30739

 - **Pdf link:** https://arxiv.org/pdf/2608.30739

 - **Abstract**
 This paper proposes two spectrally weighted STFT loss functions for lightweight streaming speech enhancement, addressing the magnitude over-attenuation in mid-to-high frequency regions caused by the magnitude-phase compensation effect. The proposed sigmoid-weighted loss applies a smooth frequency-dependent modulation to the phase-aware contribution, while the signal-dependent spectrally adaptive loss further conditions the modulation on the ground-truth log-magnitude spectrogram. To evaluate the proposed objectives, we additionally design HyST-Net, a lightweight and competitive backbone with hybrid MHA-GRU spectral-temporal modelling for low-latency streaming scenarios. Experimental results exhibit consistent improvements in high-frequency spectral reconstruction for both losses. The spectrally adaptive loss further enhances the mid-frequency region, resulting in a more balanced spectral reconstruction across the full frequency range.
#### Likelihood-Constrained Acoustic Reranking for Training-Free Hallucination Mitigation in LLM-Based ASR
 - **Authors:** Jiasheng Kuang, Linru Zheng, Hongjin Song, Zhaoqi Cui, Song Li
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2608.30776

 - **Pdf link:** https://arxiv.org/pdf/2608.30776

 - **Abstract**
 Large language model (LLM)-based automatic speech recognition (ASR) systems achieve strong performance on conventional speech data by leveraging powerful linguistic priors and multilingual capabilities. However, under challenging conditions, these priors can override acoustic evidence, resulting in unintended translation, instruction execution, repetition, or catastrophic deletion. We propose Likelihood-Constrained Acoustic Reranking (LCAR), a training-free decoding method that improves acoustic grounding while preserving support from the base model. At each decoding step, LCAR first retains tokens whose base-model likelihood falls within a margin of the greedy token, then reranks them using an acoustic compatibility score computed from attention-pooled audio embeddings and the existing LM head. By restricting acoustic intervention to plausible, model-supported alternatives, LCAR requires no additional training, external detector, reference transcript, or auxiliary model at inference. We evaluate LCAR on four LLM-based ASR systems using human-audited TTS and open-source speech challenge suites. At $\delta=0.60$, LCAR removes 38.8--57.1\% of detector-identified hallucination failures while largely maintaining WER/CER on standard open-source test sets.
#### V2TATC: A Joint Voice-Trajectory Embedding Framework and Dataset for Air Traffic Controller Situational Awareness
 - **Authors:** Louis Brusset, Mathurin Petit, Jordan Kam, Alexandre Bayen
 - **Subjects:** Subjects:
Machine Learning (cs.LG); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2608.28981

 - **Pdf link:** https://arxiv.org/pdf/2608.28981

 - **Abstract**
 As air traffic volumes in the National Airspace System continue to expand, in particular in the low altitude airspaces, the need for scalable decision support tools used by air traffic controllers will also require more development. This article introduces Voice-to-Trajectory for Air Traffic Control, a joint voice communication-flight trajectory data embedding framework, that can be a component of situational awareness in congested airspaces, and assist the development of tools for ATC as they reason in real-time over Automatic Dependent Surveillance-Broadcast trajectories, or the intent expressed by pilots in natural language. We show that these data modalities are not independent and represent a common physical referent: an aircraft flying through the airspace. V2TATC maps a voice instruction and the trajectory of the addressed aircraft to nearby points in a single latent space that can be queried in both directions. It combines a self-supervised trajectory encoder, a frozen large-scale speech encoder, a contrastive joint embedding, and a bijective lifting via normalizing flows. We demonstrate V2TATC's effectiveness on the San Francisco Bay Area, for its concentration of major airports, and its mix of commercial and general aviation low altitude traffic. Lastly, we release a novel paired voice-trajectory dataset, and report experiments on cross-modal retrieval, ablations, and latent-space analysis.
#### Anchoring Speech with Semantics: A Multimodal Adapter Mechanism for Automatic Speech Recognition in Low-Resource Languages
 - **Authors:** Kuan-Tang Huang, Cheng-Yeh Yang, Chien-Chun Wang, Hung-Shin Lee, Hsin-Min Wang, Berlin Chen
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2608.29239

 - **Pdf link:** https://arxiv.org/pdf/2608.29239

 - **Abstract**
 Low-resource ASR remains difficult because scarce transcripts provide limited supervised evidence for target-side generation. To address this gap, we propose SAMA-ASR, a lightweight adapter mechanism that augments the decoder with semantic anchors from auxiliary translations and an acoustic anchor from speech; in principle, the mechanism can be applied to similar encoder--decoder multitask speech models. Through cross-modal adaptation, SAMA-ASR conditions decoder states on translation-derived semantic embeddings and a speech embedding, combining utterance-level meaning with speech-grounded evidence before token prediction. At evaluation time, these semantic anchors can be generated automatically by an upstream speech-to-text translator rather than supplied as oracle translations. Experiments on two 30-hour datasets covering the low-resource Sinitic varieties Taiwanese Hokkien and Hakka show that SAMA-ASR improves over acoustic, prior prompt-based, and semantic-only translation-guided baselines and remains effective in practical automatic semantic-anchor settings; translator-capacity analyses show that useful semantic anchors can be produced by a compact ST model.
#### Perceptually Better, Semantically Worse: Measuring Speech Enhancement Impact on LLM-Based Voice Systems
 - **Authors:** Randy Frans Fela, Pejman Mowlaee
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2608.30348

 - **Pdf link:** https://arxiv.org/pdf/2608.30348

 - **Abstract**
 Speech enhancement (SE) is commonly applied as a preprocessing step in spoken AI pipelines under the assumption that better audio quality improves downstream task performance. Whether SE-induced distortions propagate to downstream LLM task performance remains an open question. We introduce Output Divergence Rate (ODR), which measures how often SE changes an LLM's intent classification relative to clean speech, and benchmark five conditions on 2,974 SLURP clips using Whisper large-v3 and wav2vec2-large cascades. Every condition produces ODR significantly above zero ($p < 0.001$, binomial test). MetricGAN{+} more than doubles ODR versus unenhanced noisy speech (0.318 vs. 0.135) despite improving PESQ, and unmitigated echo reaches an ODR of 0.836 through speaker substitution, a failure WER cannot capture. Audio quality metrics range from near-zero to moderate correlation with ODR (SQUIM-MOS $\rho=-0.068$, PESQ $\rho=-0.467$). The MetricGAN{+} and echo results replicate across ASR architectures, indicating that standard audio quality metrics are insufficient for LLM pipeline quality.
#### Textual Acoustic Grounding for Generalizable LLM-Based Deepfake Voice Detection
 - **Authors:** Yassine El Kheir, Xin Wang, Wanqing Ge, Tim Polzehl, Sebastian Moeller, Junichi Yamagishi
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2608.30622

 - **Pdf link:** https://arxiv.org/pdf/2608.30622

 - **Abstract**
 Deepfake voice detection suffers from poor generalization across unseen domains. While Audio Large Language Models (ALLMs) show promise, the modality gap between continuous audio embeddings which capture the subtle acoustic details necessary for deepfake detection and the semantic space of LLMs remains a critical, underexplored bottleneck. We address this by benchmarking diverse audio encoders integrated with Qwen LLMs (0.5B to 7B parameters). First, we demonstrate that fine-tuning the LLM alone risks out-of-domain overfitting, making a frozen LLM a stronger, resource-efficient baseline. Second, to explicitly bridge the modality gap, we introduce a cross-modal prompting strategy that injects linguistic-knowledge-driven acoustic features (via openSMILE) as structured text tokens. This explicit textual grounding not only enhances the frozen baseline but also makes LLM fine-tuning more effective. Ultimately, our approach demonstrates state-of-the-art resilience on the out-of-domain ITW and MLAAD benchmarks, yielding over \textbf{16.2\%} absolute improvement in Macro-F1 over existing ALLM baselines while maintaining competitive in-domain performance. All models reported in this work are \href{this https URL}{publicly available}.
#### Conjoint Audio-to-Spikes Encoding and Processing for Efficient Neuromorphic Speech Recognition
 - **Authors:** Valentin M. Meunier, Amélie Gruel, Pierre Lewden, Adrien F. Vincent, Sylvain Saïghi
 - **Subjects:** Subjects:
Neural and Evolutionary Computing (cs.NE); Artificial Intelligence (cs.AI); Machine Learning (cs.LG); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2608.30792

 - **Pdf link:** https://arxiv.org/pdf/2608.30792

 - **Abstract**
 Obtaining data from neuromorphic sensors and processing it with Spiking Neural Networks is a promising solution to lower the energy cost of artificial intelligence. The current rarity of natively neuromorphic datasets promotes the development of software tools to translate input sensory data into spikes. However, highly bio-mimetic simulators can be challenging to implement on digital hardware. In this work, we evaluate the neuromorphic encoding and subsequent classification of audio into spikes using a non-learnable, high-level, programmable encoder targeting hardware implementation on FPGA. We quantify the pipeline's efficiency with hardware-agnostic metrics based on the quantitative spiking activity. Our study focuses on the simultaneous optimisation of encoder and classifier: the first provides efficient and informative data so that the latter achieves a better performance with an overall lower energy cost at learning and inference. This work introduces the first end-to-end neuromorphic spike-encoding and evaluation of the TIMIT dataset. Our simple feedforward network reaches a classification accuracy of 99.77% on a spike-encoded Heidelberg Digits, overcoming the neuromorphic state of the art on this benchmark dataset.
#### When Does Predictor-Based RL Align with Human Perception? A Study of Subjective Rewards in Codec-Based Speech Language Models
 - **Authors:** Joonyong Park, Jerry Li
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2608.31035

 - **Pdf link:** https://arxiv.org/pdf/2608.31035

 - **Abstract**
 Codec-based text-to-speech (TTS) models make language-model post-training applicable to speech generation, but it remains unclear when learned perceptual predictors can serve as reinforcement learning rewards without losing alignment with human listeners. We study this question with Group Relative Policy Optimization (GRPO) using learned rewards for anime-like speaking style, naturalness, likability, and arousal. To prevent perceptual rewards from being optimized through transcript drift, we introduce a character error rate (CER) zone constraint and compare policy optimization with Best-of-$N$ reranking under the same reward gate. Across single-reward runs, each reward primarily improves its own target metric, showing that subjective predictors are not interchangeable quality surrogates. Multi-rater A/B tests further show uneven human transfer, while a reward-gap analysis separates average transfer from within-axis calibration: signed reward gaps significantly predict listener choices in the pooled analysis, whereas residual CER gaps do not, but per-axis calibration remains heterogeneous. Best-of-8 is a strong human-level baseline and is not clearly worse than GRPO perceptually, suggesting that GRPO should be viewed as amortizing reward-selected behavior into the policy rather than uniformly outperforming reranking. These results support analyzing subjective speech rewards as predictor-axis-base tuples and provide practical diagnostics for selecting rewards before multi-reward speech post-training.


by Zyzzyva0381 (Windy). 


2026-09-01
