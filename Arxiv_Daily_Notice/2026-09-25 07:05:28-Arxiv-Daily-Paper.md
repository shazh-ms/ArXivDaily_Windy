# Showing new listings for Friday, 25 September 2026
Auto update papers at about 2:30am UTC (10:30am Beijing time) every weekday.


阅读 `Usage.md`了解如何使用此repo实现个性化的Arxiv论文推送

See `Usage.md` for instructions on how to personalize the repo. 


Keyword list: ['text-to-speech', 'text to speech', 'tts', 'LLM-based', 'speech', 'voice']


Excluded: []


### Today: 17papers 
#### Spooftral: Can Voxtral Audio-Language Model Detect Speech Spoofing?
 - **Authors:** Avishai Weizman, Yehuda Ben-Shimol, Itshak Lapidot
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Computation and Language (cs.CL)
 - **Arxiv link:** https://arxiv.org/abs/2609.28713

 - **Pdf link:** https://arxiv.org/pdf/2609.28713

 - **Abstract**
 Self-supervised learning (SSL) countermeasures (CMs) have shown strong performance in recent years. However, they often show degraded performance while facing unseen spoofing attacks and mismatched conditions. This study examines the Voxtral audio-language model (ALM) framework for spoofing detection, as a step toward combining CM capabilities within the ALM framework. We analyze how Voxtral captures spoofing cues through audio-text processing and propose an instruction-guided approach that uses label-sequence likelihoods to evaluate bonafide and spoofed speech. Experiments on the ASVspoof databases show that without task-specific adaptation, the LLM layers emphasize semantic representations, reducing the separability of spoof-discriminative acoustic cues compared to the Whisper-based audio encoder. Consequently, spoofing-related information becomes less separable after language-model processing. We also applied lightweight adaptation using weight-decomposed low-rank adaptation (DoRA) to the Voxtral model and propose the Spooftral model, achieving an equal error rate (EER) of 4.25% on the ASVspoof5 evaluation set.
#### A Harness for Synthesizing Diverse Naturalistic Full-Duplex Conversations
 - **Authors:** Matthew Sun, Vinay Kothapally, Meng Yu, Chao Huang, Hao Zhang, Yixuan Zhang, Steve Yves
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Artificial Intelligence (cs.AI)
 - **Arxiv link:** https://arxiv.org/abs/2609.28806

 - **Pdf link:** https://arxiv.org/pdf/2609.28806

 - **Abstract**
 Full-duplex dialogue systems, which listen while speaking, must distinguish a completed turn from a pause within a turn and an interruption that requests a turn from a brief acknowledgment or speech addressed to a third party. Yet existing conversational corpora provide limited control over these events and limited labels for their intent. We present a pipeline for synthesizing intent-labeled, two-channel conversational speech from relational event lists. An LLM authors each event's speaker, text, conversational act, and attachment to an earlier event without predicting absolute timestamps. Events are synthesized independently, aligned with their source text, and placed on a shared clock, so turn-taking landmarks are measured from the rendered signal while silence durations are specified or sampled from turn-taking distributions. The pipeline covers 42 phenomena across eight families in English and Mandarin, derives frame-level system actions from authored intent, and promotes diversity using small, diverse sets of prior examples and batch prompts that request alternatives with self-reported probabilities. Ablations show gains in each targeted diversity dimension. On a four-action label space for taking, holding, releasing, and not holding the conversational floor, a semantic voice-activity detector using only current and past audio reaches start-speaking and start-listening F1 scores of 0.819 and 0.802. When generating its own responses, the full-duplex speech model Moshi takes 0.85 of the reference turns after fine-tuning on the generated corpus, compared with 0.44 before fine-tuning. Its frame-level precision for predicting system-floor occupancy rises from 0.46 to 0.88. With reference context at each step, its frame-level floor F1 rises from 0.893 to 0.962. These results show that controlled synthesis can provide learnable and transferable supervision for full-duplex turn management.
#### Learning New Words from Unlabeled Test Data in Automatic Speech Recognition
 - **Authors:** Mengqi Wang, Mark A. Hasegawa-Johnson, Haolong Zheng, Chang D. Yoo
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Computation and Language (cs.CL)
 - **Arxiv link:** https://arxiv.org/abs/2609.28877

 - **Pdf link:** https://arxiv.org/pdf/2609.28877

 - **Abstract**
 New words are invented every day. A human listener can learn a new word by hearing it clearly once and inferring its usage from sentence context. This paper proposes granting ASR a similar ability to learn the contextual representations and spellings of new words from unlabeled test data at test time. A frozen CTC acoustic model provides spellings, a frozen language model provides contextual evidence for out-of-vocabulary (OOV) word detection, and an adaptation module expands the vocabulary by learning the lexical token representations with distributions over CTC-generated candidates. The spelling model of each token is optimized by minimizing a Kullback-Leibler divergence (KLD) objective. We demonstrate that the CTC-weighted language model log likelihood ratio can be interpreted as the KLD between the unknown correct ASR and the unsupervised learned ASR, and that, using a Pinsker bound, the square root of KLD can be interpreted as an upper bound on the total variation distance between the true and estimated spelling of the unknown word. Experiments show relative OOV character-error-rate reductions of up to 14.97% on LibriSpeech and 6.67% on dysarthric Speech Accessibility Project data for recurring OOV words, relative to the corresponding rescoring system.
#### Same Bit Width, Different Outcomes: Post-Training Quantization of Text-to-Speech Across Architectures
 - **Authors:** Se Un Park, Yutae Kim, Junyoung Park
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Machine Learning (cs.LG); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2609.28974

 - **Pdf link:** https://arxiv.org/pdf/2609.28974

 - **Abstract**
 Post-training quantization (PTQ) reduces the cost of on-device text-to-speech (TTS), but published evaluations cover one system or method. We evaluate PTQ across TTS architectures under one protocol with three core models, weight and activation ablations of eight more, and two held-out models quantized blind. Four-bit per-channel weights reduce UTMOS, a predicted mean opinion score, by 2.8 on Supertonic and 0.07 on Kokoro, and per-tensor scaling can cause severe degradation even at 8 bits. The same bit width yields different outcomes, because the sensitive component is model-specific and not reliably predicted from the model class. A staged ablation procedure identifies it, and per-layer GPTQ can restore it to within 0.1 UTMOS. Real int8 and int4 kernels reproduce the simulated ordering at hardware-dependent cost. On a Mac mini, a 4-bit weight kernel runs Supertonic at 0.60x the fp32 latency while int8 is slower, so each configuration requires validation on the target runtime.
#### Personalized Korean Lipreading as Visual Speech Recognition: Transfer, Census and Adaptation on OLKAVS
 - **Authors:** Se Un Park, Hakjun Kim, Taehoon Roh, Junyoung Park
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Computation and Language (cs.CL); Computer Vision and Pattern Recognition (cs.CV)
 - **Arxiv link:** https://arxiv.org/abs/2609.28988

 - **Pdf link:** https://arxiv.org/pdf/2609.28988

 - **Abstract**
 We present a personalized Korean visual speech recognition (VSR) system and quantify, on the nine-camera OLKAVS corpus, the gap between the population-level benchmark score and an individual user's error. A video-only Conformer initialized from English-trained weights attains 9.95 - 12.19% character error rate (CER) under the corpus protocol against the published 26.64, and 19.00 - 21.52 on unseen wording. Per speaker, CER spans 1.0 to 52.2%, with seen wording lowering CER by 7.0 - 9.0 points and professional delivery and spontaneous speech raising it by 8.5 - 10.5 and 12.7 points. A low-rank adapter with 4.6% of the parameters, trained on 4 to 29 minutes of the user's frontal video, lowers the CER of twelve high-error speakers by 2.13 to 3.58 points, transfers to every camera without loss, and keeps 85% of the full fine-tuning gain at 12% of its cost to other speakers. Cameras above the mouth plane add about six CER points as a constant offset that training on all views keeps small.
#### Is Broader Better? A Controlled Study of Multilingual Coverage and Pretraining Objective in Frozen SSL Encoders for Speech Deepfake Detection
 - **Authors:** Benjamin Hurt, Oscar O'Donnell
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2609.29138

 - **Pdf link:** https://arxiv.org/pdf/2609.29138

 - **Abstract**
 Frozen self-supervised (SSL) speech encoders are strong, low-cost front ends for audio deepfake detection, and recent comparisons agree that large, multilingual, discriminative encoders generalize best out of domain. These comparisons fail to control for encoder capacity, pretraining objective, and multilingual coverage together, identifying which encoder wins without isolating why. We present a controlled decomposition with a fixed pipeline and trainable capacity. We vary multilingual coverage on four wav2vec2-family encoders, matched to ~315M parameters. We isolate the pretraining objective on two encoders matched on identical data. Coverage does not help monotonically, as out-of-domain error drops sharply at the ~100-language scale (XLS-R) but does not improve further at the 1406-language extreme (MMS). We find that a mid-coverage encoder is strongest on farther out-of-domain sets, matching or surpassing a 577M-parameter model at 315M. Its lead on these far sets, statistically significant under paired bootstrap, and on the official ASVspoof 5 cost metric holds under two backends. Separately, masked-prediction pretraining generalizes better than contrastive on identical data (In-the-Wild EER 26.5% vs. 46.8%). Within this fixed frozen-encoder recipe, we find that broader and larger models are not reliably better.
#### WST-Graph: Topology-Preserving Wavelet Scattering Front-End for Speech Deepfake Detection
 - **Authors:** Kwok-Ho Ng, Tingting Song, Bingwen Feng, Zhihua Xia
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Artificial Intelligence (cs.AI); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2609.29372

 - **Pdf link:** https://arxiv.org/pdf/2609.29372

 - **Abstract**
 The acoustic front-end determines which forensic cues a speech deepfake detector can exploit. The wavelet scattering transform (WST) provides stable multiscale coefficients with explicit coordinates, yet direct flattening obscures the parent relation between paths. We introduce WST-Graph, reconstructing these paths as a sparse modulation-carrier grid for an AASIST graph backend. Modulation-level normalization and length-aware adaptive local attention pooling produce fixed relative-time representations while retaining the acoustic axes before learned adaptation. This yields a waveform-to-graph interface with a fixed, parameter-free WST. Our configurations remain competitive with AASIST while using approximately 60% fewer trainable parameters and show clear gains on selected out-of-domain benchmarks. These results underscore the value of preserving parent-child relations within the carrier-modulation topology when constructing a compact, physically grounded interface for graph-based speech deepfake detection. Code will be released at this https URL.
#### Transcript-Supervised Post-Training of Generative Speech Enhancement on Real Recordings via Reinforce Adjoint Matching
 - **Authors:** Julius Richter, Christoph Boeddeker, Yoshiki Masuyama, Kohei Saijo, Dominik Klement, Gordon Wichern, Jonathan Le Roux
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Machine Learning (cs.LG)
 - **Arxiv link:** https://arxiv.org/abs/2609.29405

 - **Pdf link:** https://arxiv.org/pdf/2609.29405

 - **Abstract**
 We adapt Reinforce Adjoint Matching (RAM), a reward-based post-training method, to generative speech enhancement (SE). Starting from a pretrained SE model, RAM tilts the model's conditional distribution toward outputs with higher reward. During training, the current model generates enhanced speech on-policy, evaluates each generated endpoint with a potentially non-differentiable reward, and analytically re-noises the endpoint to construct inputs for a reward-guided regression objective. This enables post-training directly on real recordings using weak supervision, such as text transcripts, without requiring paired clean speech targets or reward gradients. We investigate word error rate (WER)-based post-training and whether recognition performance can be improved without compromising perceptual speech quality. Experiments on real CHiME-4 recordings reduce WER by 5.08 percentage points relative to pretrained FlowSE without reducing any of the reported non-intrusive speech quality metrics. A subjective listening test at the default reward scale finds no statistically significant preference between the post-trained and pretrained models.
#### Voice Agents under Acoustic Stress: From Signal Degradation to Interaction and Action
 - **Authors:** Amir Ivry, Kai-Wei Chang, Lin Zhang, Sharon Gannot, Carlos Busso
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Human-Computer Interaction (cs.HC)
 - **Arxiv link:** https://arxiv.org/abs/2609.29452

 - **Pdf link:** https://arxiv.org/pdf/2609.29452

 - **Abstract**
 Voice agents must complete users' tasks despite noise, reverberation, and competing speech. Evaluating agents' robustness therefore requires following how acoustic conditions affect the conversation and the actions taken on the user's behalf. This overview examines what existing benchmarks reveal about agents' ability to complete tasks under acoustic stress and where further task-based evaluation is required. We then introduce TRACE, a practical workflow for designing, running, and interpreting evaluations of acoustic robustness in task-oriented human-agent interactions: the same agent attempts a specified task with an original recording and an acoustically stressed copy, and the resulting conversations are scored for task completion, wrong actions, recovery, and user effort. Finally, we explain how results from these evaluations can guide changes to an agent to prevent wrong actions and improve recovery.
#### Configurable-Bandwidth Time-Frequency Modeling for Efficient Full-Band Speech Enhancement Across Sampling Rates
 - **Authors:** Ui-Hyeop Shin, Wooseok Kim, Hyung-Min Park
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2609.29463

 - **Pdf link:** https://arxiv.org/pdf/2609.29463

 - **Abstract**
 Speech enhancement systems are often developed for a fixed sampling rate, while time-frequency models become more expensive as the number of frequency bins increases. We propose TF-Refiner, a sampling-frequency-independent model that decouples the deep analysis bandwidth from the full-band input and output. A deep encoder processes the band below a configurable cutoff, while a shallow decoder combines the encoded features with input-dependent high-band queries and predicts local complex filters applied to the original noisy STFT. A single parameter set trained at 16 and 48 kHz is evaluated at various sampling rates. On VoiceBank+DEMAND, the universal model outperforms the rate-specific counterparts in PESQ, STOI, and log-spectral distance across the evaluated rates, including rates unseen in training. Random-cutoff training enables inference-time selection of cost-quality operating points without retraining or changing the output bandwidth. These results support configurable analysis bandwidth as a practical design choice for multi-rate full-band enhancement.
#### Depth through recurrence: Looped transformers for flow-matching TTS
 - **Authors:** Jiabao Ai, Peng Han, Yuchen Song, Zhengjun Yue
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2609.29768

 - **Pdf link:** https://arxiv.org/pdf/2609.29768

 - **Abstract**
 We study how to organize Transformer depth through recurrence in flow-matching text-to-speech, varying the amount, order, and place?ment of weight reuse. Seven layouts perform 18 block calls per network evaluation under a common training objective and sam?pler. On Seed-TTS and LibriSpeech-PC, SEQUENCE applies each of nine blocks twice consecutively, retaining competitive intelligibil?ity, speaker similarity, and predicted speech quality at 32 sampling steps with 47.1% fewer parameters than the unshared baseline. Cy?cling six blocks three times further reduces model size but raises 32-step word error rates relative to cycling nine blocks twice. At matched parameter counts and executed depth, reuse order and shar?ing position produce different quality trade-offs. These comparisons depend on sampling budget: Prefix and Suffix have similar 32-step word error rates, but Suffix is worse by 3.44 and 5.97 percentage points at four steps on the two datasets, respectively. Only Middle ranks first or second in mean word error rate at 32 and four steps on both datasets. These results show that the organization of recurrent computation affects synthesis quality, and that reuse layouts should be selected for both the sampling budget and the quality dimensions of interest.
#### Beyond Model Size: Redesigning LiSenNet for embedded speech enhancement
 - **Authors:** Clément Laroche, Rasmus Kongsgaard Olsson
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Machine Learning (cs.LG); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2609.29866

 - **Pdf link:** https://arxiv.org/pdf/2609.29866

 - **Abstract**
 Deploying real-time speech enhancement on resource-constrained devices requires meeting strict latency, memory, and energy constraints. Microcontroller NPUs can accelerate neural inference under these constraints, but only through a restricted set of operators in static, integer-quantized graphs. Recent speech-enhancement networks have reduced parameter counts and MACs to levels nominally suitable for microcontrollers, but their operators and execution patterns often remain incompatible with restricted NPUs. We address this gap by redesigning LiSenNet, a 37k parameter sub-band dual-path model, for the STM32N6570-DK Neural-ART accelerator. We replace its recurrent bottleneck with convolutional frequency and temporal mixers, reformulate unsupported operations as static int8-compatible primitives, and use bounded decoder activations to preserve quality after quantization. On VoiceBank-DEMAND, the final NPU-compatible model matches or exceeds the recurrent LiSenNet baseline, reaching PESQ 3.08 versus 3.01 in FP32 and 3.01 versus 2.93 in int8. Deployed on a microcontroller, it processes each 16 ms input hop in 4.83 ms, corresponding to a real-time factor of 0.30. Stateless receptive-field recomputation is an order of magnitude slower at the same frame rate despite higher accelerator utilization. These results show that parameter count and operator compatibility, quantization range, and persistent streaming state must be co-designed to achieve efficient real-time speech enhancement on restricted NPUs.
#### Does per-frame early exit pay? A compute-matched study of dynamic depth for on-device speech enhancement
 - **Authors:** Clément Laroche, Riccardo Miccini
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Machine Learning (cs.LG); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2609.29867

 - **Pdf link:** https://arxiv.org/pdf/2609.29867

 - **Abstract**
 Deep learning-based speech enhancement is increasingly deployed on-device in hearing aids, headsets, and earbuds. Most of these devices, however, can only accelerate static int8 graphs, so a depth-varying network must be implemented as several graphs, orchestrated by a policy. In this paper, we supervise every intermediate depth of one causal model, then we fine-tune its output heads to guarantee that deeper outputs are never worse than shallower ones. Using this training protocol, we can derive a family of static models that are more Pareto-efficient than their equivalently-sized counterparts trained from scratch on the same budget. Specifically, we achieve up to 0.11 higher PESQ for equivalent compute, and match the best PESQ at 30% less compute. We then quantize the models to int8 and measure the latency-quality frontier on an STM32N6 microcontroller. On VoiceBank-DEMAND, the dynamic enhancer lies on the same frontier as the static models, rather than trading quality for dynamic execution. Running the policy on the companion Cortex-M55 takes only 26 $\mu$s per frame, while splitting the enhancer into separate NPU graphs adds 2.2% latency overhead. The cost of dynamic execution is therefore small.
#### ARIS: Low-Resource Glass-Box Neural Source-Filter Synthesis for Phonetic Stimulus Manipulation
 - **Authors:** Yiran Ding, Wenwei Xu
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2609.29923

 - **Pdf link:** https://arxiv.org/pdf/2609.29923

 - **Abstract**
 Phoneticians often need to construct stimuli in which specific acoustic cues are precisely manipulated while preserving decent speech quality. Classical synthesis and modern neural methods sit along a trade-off between precise parametric control and high fidelity, and neural synthesis typically demands more data than phoneticians can easily obtain. We present ARIS (Analytic Resonant Interpretable Synthesis), a neural source-filter model that pairs neural parameter estimation with deterministic DSP synthesis. Every control is a coefficient of the synthesizer, so F0, formants and the glottal source can be edited directly. On five small single-speaker corpora in three languages, ARIS resynthesizes speech with quality comparable to WORLD and edits single parameters more accurately than Praat KlattGrid, with negligible crosstalk between cues. Compared with HiFi-Glot, pre-trained on a large corpus and fine-tuned on the same data, ARIS scores slightly lower on predicted naturalness but reproduces the recordings more faithfully and manipulates them more precisely. Audio samples: this https URL.
#### Accent Analogy Guidance: More Speaker Similarity at Equal Accent in Cross-Lingual Voice Cloning
 - **Authors:** Yoomee Cho, Jisun Lee
 - **Subjects:** Subjects:
Sound (cs.SD); Computation and Language (cs.CL); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2609.29123

 - **Pdf link:** https://arxiv.org/pdf/2609.29123

 - **Abstract**
 In cross-lingual zero-shot text-to-speech, the accent of the reference leaks into the target speech. We propose accent analogy guidance (AAG), a training-free sampler term that subtracts an accent direction estimated from the model's own predictions for one synthetic voice rendered in both languages, so the voice cancels and only the accent remains. By a blind LLM accent judge on real dubbing data, reweighting classifier-free guidance between reference and text, and its variants, stay near one identity-accent trade-off curve; we score a method by its speaker similarity above that curve at equal accent ($\Delta$SIM). Across four open TTS models AAG lies above the curve: on OmniVoice $\Delta$SIM is +0.11 to +0.27 on three test sets (accent 3.51 to 4.28 on a 1-5 scale at speaker similarity 0.29, where reweighting keeps 0.02); MaskGCT and CosyVoice 2 also lie above their curves, and on F5-TTS it is more native than any reweighting setting. An LLM-free language-ID measure and a twelve-listener panel agree. A premise test and the reach of a model's own curve indicate in advance whether and roughly how much AAG can gain, predicting the one model where it gains nothing (X-Voice).
#### Exploring a Single Autoregressive LLM for Unified Target Speech Extraction across Synchronous and Asynchronous Cues
 - **Authors:** Wenxuan Wu, Shuhan Zhang, Shuai Wang, Haizhou Li
 - **Subjects:** Subjects:
Sound (cs.SD); Multimedia (cs.MM); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2609.29238

 - **Pdf link:** https://arxiv.org/pdf/2609.29238

 - **Abstract**
 Target speech extraction (TSE) typically trains a separate extractor per cue, and visual-cue systems often need corruption-matched training to remain robust under visual frame corruption. We show that one autoregressive LLM backbone, TSE-Omni, can serve both temporally synchronous cues (lip movements, co-speech gestures) and asynchronous cues (enrollment audio, text). TSE-Omni is driven by next-token prediction: each step predicts target speech semantic tokens from its own past outputs, which we term self-enrollment, forming a continuous target-speech context initialized by the enrollment cue (asynchronous audio or text, or a short visual prefix). This enables audio-visual compensation: the model uses synchronized visuals when intact and its token history when visual frames are missing. Under clean visuals, TSE-Omni matches strong discriminative and generative baselines (SpeechBERTScore 0.81 on VoxCeleb2 and 0.89 on LRS3 zero-shot) with higher DNSMOS. On the same VoxCeleb2 test set, after a 2 s clean visual start, removing the remaining visual frames leaves SpeechBERTScore at 0.81. It remains usable under sparse overlap and multi-speaker interference, and supports streaming inference. Project page: this https URL.
#### A Training Criterion with Token-Level Tolerance to Transcription Ambiguity for Automatic Speech Recognition
 - **Authors:** Saurabh Kumar, Diptiman Mohanta, Prasanta Kumar Ghosh
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2609.30160

 - **Pdf link:** https://arxiv.org/pdf/2609.30160

 - **Abstract**
 Automatic speech recognition is typically trained assuming that the reference transcript is the only valid labeling of an utterance, yet even nominally verbatim transcripts contain localized differences in pronunciation, spelling, or lexical realization that the acoustics do not uniquely determine. Omni-temporal Classification (OTC) tolerates such noise by adding wildcard paths to the connectionist temporal classification (CTC) alignment graph, but its word-level arcs are too coarse, since bypassing one unsupported token discards supervision for the whole word. We move wildcard arcs to token granularity so unsupported tokens can be bypassed while the rest of the word stays supervised, and we combine token- and word-level arcs as complementary escape paths. Across 19 languages and three corpora, token-level OTC improves over CTC on all 25 tasks. We also replace epoch-indexed relaxation of the wildcard weights with a predictive-entropy-indexed schedule, which performs comparably while reducing dependence on training length. Combining this schedule with the hybrid graph gives the lowest mean word error rate (WER) on every corpus and a 9.45% average relative WER reduction over CTC. Independent validator transcriptions show that token-level models place significantly more wildcard-bypass probability than CTC on disputed characters, indicating that token-level tolerance targets localized transcript ambiguity.


by Zyzzyva0381 (Windy). 


2026-09-25
