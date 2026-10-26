# Showing new listings for Monday, 26 October 2026
Auto update papers at about 2:30am UTC (10:30am Beijing time) every weekday.


阅读 `Usage.md`了解如何使用此repo实现个性化的Arxiv论文推送

See `Usage.md` for instructions on how to personalize the repo. 


Keyword list: ['text-to-speech', 'text to speech', 'tts', 'LLM-based', 'speech', 'voice']


Excluded: []


### Today: 13papers 
#### DRAMA-Eval: A Benchmark for Context-Aware, Multi-Turn, Multi-Speaker Speech Generation
 - **Authors:** Jingbin Hu, Zihang Yang, Xiaoge Peng, Haoyu Zhang, Ziyu Zhang, Qirui Zhan, Huakang Chen, Zhao Guo, Wenjie Tian, Kangxiang Xia, Wenkang Hou, Ruonan You, Zihan Liu, Chenyi Liao, Hongyu Lyu, Zihao Bai, Jinrui Liang, Shengming Lin, Yunxiang Chen, Houdun Liu, Su Feng, Bengu Wu, Liumeng Xue, Lei Xie
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2610.31796

 - **Pdf link:** https://arxiv.org/pdf/2610.31796

 - **Abstract**
 Recent advances in speech generation are shifting toward context-aware, multi-turn, and multi-speaker synthesis. While these capabilities are in high demand for applications such as audiobook and audio drama production, evaluation has not kept pace with these advances. Beyond traditional zero-shot Text-to-Speech (TTS), modern systems in this domain primarily fall into three categories: instruction-controllable generation models, multi-turn dialogue models, and the audio branches of audio-visual generation models. However, existing benchmarks evaluate these paradigms in isolation, failing to provide a unified assessment of context modeling, instruction following, and multi-speaker generation. Meanwhile, recent progress in audio understanding models has made it feasible to evaluate such complex scenarios using LLM-as-a-Judge approaches. To bridge this gap, we introduce DRAMA-Eval, a comprehensive benchmark for evaluating such systems. DRAMA-Eval utilizes structured script contexts and offers a holistic evaluation framework covering objective, local, and global aspects of generation quality, enabling the assessment of acoustic fidelity, sentence-level execution, and long-range coherence. We benchmark leading open-source and commercial models on a human?verified dataset. The results reveal critical weaknesses in long-range speaker consistency and implicit instruction following that require deep contextual understanding. To facilitate reproducible research, we release a 12-hour bilingual (Chinese-English) test set of manually corrected real-world scripts, accompanied by ground-truth audio, evaluation code, and benchmark results.
#### Concentrated Causal Influence, Distributed Diagnostic Information: Cognitive Impairment Decisions in Large Audio-Language Models
 - **Authors:** Xiutian Zhao, Luqi Sun, Björn Schuller, Berrak Sisman
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Computation and Language (cs.CL)
 - **Arxiv link:** https://arxiv.org/abs/2610.32061

 - **Pdf link:** https://arxiv.org/pdf/2610.32061

 - **Abstract**
 Large audio-language models (LALMs) can assess cognitive impairment (CI) from speech, yet their answers reveal neither which internal components control the decision nor how much CI information is accessible in their representations. We investigate both questions through CI-contrast neurons (CICNs), selected by class-contrastive activation frequency in decoder MLPs. Across eight open LALMs and two corpora (Pitt and PROCESS-2), we causally test CICNs against size- and layer-matched random masks and activity-matched controls. In three of the five models that meet our eligibility criteria, ablation of small CICN sets causes large, class-directed prediction shifts; speech-identified masks also control transcript-only decisions, and three of four primary Pitt masks retain specificity on PROCESS-2. Position-restricted interventions locate the influence before the final answer position, most strongly after the speech or transcript evidence under letter-labeled options, largely as a decision-direction shift. Beyond causal localization, supervised linear readouts of frozen CICN activations achieve 0.92--0.97 AUC within Pitt, versus 0.67--0.84 for the models' direct answers. Matched random and all-neuron readouts reach comparable performance; top-ranked CICNs show only an exploratory advantage at very small unit budgets. These results distinguish concentrated causal control over output decisions from broadly distributed diagnostic information for CI detection in LALMs.
#### NECTAR: Automatic Negative Clue Estimation via Centroid-Space Mapping for Target Speaker Extraction
 - **Authors:** Hyungjun Yoo, Yongjoon Lee, Jung-Woo Choi
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Signal Processing (eess.SP)
 - **Arxiv link:** https://arxiv.org/abs/2610.32133

 - **Pdf link:** https://arxiv.org/pdf/2610.32133

 - **Abstract**
 Target speaker extraction (TSE) aims to extract a desired speaker from a mixture using auxiliary clues such as enrollment speech. An interferer-related representation can provide a useful negative clue for TSE, but such information is not directly available from the target enrollment. To derive this representation from the observed mixture, we analyze the relationship between the speaker representations of a speech mixture and its constituent speakers. Our analysis shows that the mixture centroid lies near the midpoint of its constituent speaker centroids, whereas this relationship does not hold for single-utterance embeddings. Based on this observation, we propose NECTAR, NEgative Clue for TARget speaker extraction, which automatically derives an interferer-related negative clue from the mixture using the centroid-space relationship. We further introduce a Centroid-Space Mapper (CSM) to estimate the required centroid representations from a mixture and a target enrollment. NECTAR achieves the highest SI-SDRi on Libri2Mix among the compared systems trained on LibriSpeech train-clean-100, reaching 16.18 dB.
#### MG-REPA: Multi-Granularity Representation Alignment for Generative Target Speaker Extraction
 - **Authors:** Dohwan Kim, Jung-Woo Choi
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Artificial Intelligence (cs.AI)
 - **Arxiv link:** https://arxiv.org/abs/2610.32161

 - **Pdf link:** https://arxiv.org/pdf/2610.32161

 - **Abstract**
 Generative target speaker extraction (TSE) models achieve high-quality extraction, but suffer from considerably slow training convergence. Representation Alignment (REPA) accelerates diffusion transformer training by aligning hidden states with a pre-trained Self-Supervised Learning (SSL) teacher. However, existing alignment in the speech and audio domain operates at a single granularity. For TSE, frame-level alignment leaves the global, time-invariant characteristics of the target speech implicit, whereas utterance-level alignment leaves the time-varying content implicit, even though extracting the target requires both. We propose Multi-Granularity Representation Alignment (MG-REPA), aligning the hidden states with an SSL model at the frame level and a speaker verification model at the utterance level. We further apply temporal normalization to the frame-level target, removing its utterance-level statistics so that each branch concentrates on its granularity. On Libri2Mix, MG-REPA reaches the OVRL and PESQ of a MeanFlow-TSE baseline with $3.0\times$ and $2.1\times$ fewer epochs, respectively. Trained for 750 epochs, it is competitive with the published 2,000-epoch baseline on intrusive metrics while improving non-intrusive metrics. An ablation demonstrates that MG-REPA outperforms alignment at a single granularity.
#### Uni-Stream: A Unified Streaming Speech Language Model for Speech Synthesis and Recognition
 - **Authors:** Xuan Shi, Chang Zeng, Tiantian Feng, Ye Zhang, Jun Liu, Shih-Heng Wang, Lie Lu, Shrikanth Narayanan
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Artificial Intelligence (cs.AI)
 - **Arxiv link:** https://arxiv.org/abs/2610.32187

 - **Pdf link:** https://arxiv.org/pdf/2610.32187

 - **Abstract**
 Unified streaming speech language models aim to support speech recognition and synthesis within a single model, enabling real-time human-computer interaction. However, jointly supporting these two directions of speech-text conversion while preserving task-specific performance remains challenging. We propose Uni-Stream, a unified streaming speech language model built on compact pretrained speech models. Uni-Stream processes interleaved text and speech in a streaming format and achieves performance comparable to task-specific models through two-stage training. We also develop a data annotation pipeline for streaming training and will publicly release the generated transcriptions and word-level timestamps. We further examine Uni-Stream's representations across tasks and modalities and find that speech and text residence in distinct latent feature space even when trained in a unified model architecture, while representations within each modality become more task-dependent in later layers.
#### SURE-Tagger: Structured Speech Representations for Compositional Evaluation and Targeted Data Selection
 - **Authors:** Jing Peng, Huifei Wang, Weihan Chen, Shuo Gao, Bicheng Zhang, Junhao Du, Hanqi Li, Chaolei Liu, Wen Wen, Tao Liu, Xu Li, Yu Xi, Shuai Wang, Kai Yu
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2610.32198

 - **Pdf link:** https://arxiv.org/pdf/2610.32198

 - **Abstract**
 Speech conveys linguistic content, speaker characteristics, temporal interactions, paralinguistic cues, and acoustic environments. While recent speech captioning models describe these dimensions in natural language, inconsistent attribute coverage and ambiguous omissions hinder systematic analysis across recordings. We introduce SURE-Tagger, a framework that constructs semi-structured speech representations with explicit attribute values, supplementary descriptions, and abstention under insufficient evidence. By integrating specialist observations through language-model reasoning, SURE-Tagger enables speech conditions to be consistently identified and jointly queried across datasets. We further introduce Tag-Bench, combining real recordings with controlled, agentically synthesized speech scenes and verified attribute annotations. On Tag-Bench, SURE-Tagger achieves the highest scores on 25 of 29 evaluated dimensions for real speech and 24 of 29 for synthetic speech, outperforming caption-based baselines across structural and descriptive attributes. Beyond annotation, we use these representations to construct cross-dataset evaluation subsets and reveal condition-specific ASR errors. Using a separate candidate pool, SURE-Tagger-guided selection reduces the overall ASR error rate of Qwen3-ASR-1.7B by 14.3% relative to the unadapted model under a one-hour fine-tuning budget, while caption-guided and random selection increase error in the same setting. These results demonstrate how structured speech representations can connect rich annotation, compositional evaluation, and targeted model improvement.
#### ICASSP 2026 Radar Acoustic Speech Enhancement (RASE) Challenge
 - **Authors:** Zhi-Wei Tan, V. G. Reju, Ritesh C. Tewari, Ruotong Ding, Patrick A. Naylor, Andy W. H. Khong
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2610.32202

 - **Pdf link:** https://arxiv.org/pdf/2610.32202

 - **Abstract**
 This paper presents the ICASSP 2026 Radar Acoustic Speech Enhancement (RASE) Grand Challenge, the first community challenge dedicated to benchmarking speech enhancement from millimeter-wave (mmWave) radar vibrometry. The challenge targets realistic through-glass sensing scenarios where microphones are ineffective or impractical. We introduce a curated dataset of paired radar and clean speech recordings, standardized preprocessing, a baseline system, and a secure Codabench-based evaluation framework with unified objective metrics. The RASE Challenge establishes the first public benchmark for radar-based speech enhancement and aims to foster reproducible research in this emerging field.
#### Melody Guidance for Audio Diffusion Models Without Retraining
 - **Authors:** Cristobal Andrade, Sebastian J. Schlecht, Ben Maman, Meinard Müller
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2610.32211

 - **Pdf link:** https://arxiv.org/pdf/2610.32211

 - **Abstract**
 Controlling melody in audio generation normally requires training a conditional generative model on audio paired with pitch curves or MIDI notes. We instead guide an existing unconditional diffusion model, originally trained without melody inputs, to follow a target melody without retraining or modifying it. During inference, a differentiable pitch detector steers the sampled audio toward the desired pitches, while the diffusion model keeps it close to the distribution of realistic audio learned during training. Our main contribution is a sampling method that balances these two directions, providing melody control while preserving audio quality. To improve stability, we initialize sampling with a simple synthesized version of the target melody rather than pure noise. Experiments on piano, guitar, and singing voice show melody accuracy comparable to conditional models trained specifically for melody control. Our method therefore adds melody control to existing audio diffusion models without requiring paired melody data or task-specific retraining.
#### CoupleFlow: Null-Calibrated Auditory Attention Decoding With Time-Locked Brain-Speech Coupling and Flow-Matching Likelihood Ratios
 - **Authors:** Ali Alavi, Donald S. Williamson
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2610.32950

 - **Pdf link:** https://arxiv.org/pdf/2610.32950

 - **Abstract**
 Auditory attention decoding (AAD) seeks to identify the attended talker from brief noninvasive brain recordings by comparing neural responses with competing speech streams. Recent electroencephalography (EEG) decoders report nearly perfect accuracy from one second of data, yet we show that this performance fails to generalize to unseen speech because models exploit trial leakage, eye gaze, and acoustic cues instead of speech evoked neural activity. We introduce CoupleFlow, which decodes attention through temporal coupling between neural and acoustic embeddings. Its predictions remain at chance when the brain signal contains no informative coupling, preventing shortcuts based on speech content or talker identity. A conditional flow matching head implements a likelihood ratio decision rule based on pointwise mutual information between brain and speech. This rule is Bayes optimal when the recording depends only on the attended talker and removes preferences for speech that is likely a priori. We evaluate CoupleFlow with content disjoint splits recovered from the audio and null tests that distinguish genuine temporal coupling from trial recognition. Across nine EEG and magnetoencephalography (MEG) datasets processed through a common pipeline, spanning four languages, high density EEG recordings, listeners with hearing impairment, and a room with multiple talkers, CoupleFlow outperforms linear and deep baselines while all null tests remain at chance. Finally, we derive an evidence accumulation law that predicts accuracy across window lengths from a single parameter per dataset and predicts that parameter from the strength of neural speech tracking. This law identifies recording signal to noise ratio as a limit on decoding speed and reliability and estimates the window length required to reach a target accuracy.
#### When Do Self-Supervised Representations Help Speech Restoration?
 - **Authors:** Yongjoon Lee, Younghoo Kwon, Jung-Woo Choi
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2610.30869

 - **Pdf link:** https://arxiv.org/pdf/2610.30869

 - **Abstract**
 Self-supervised learning (SSL) representations are widely used in speech restoration (SR), and the prevailing trend is SSL feature cleansing: fine-tuning the SSL to recover clean features from degraded speech, then conditioning the restoration model on them. However, the representational capability of SSL when encoding degraded speech has been underexplored, and we ask whether cleansing always benefits SR training. Across SSL encoders and perceptual quality proxies, we discover perceptual correlation collapse---under severe degradation, cleansing degraded SSL features toward clean ones misaligns with the SR objective. We hypothesize the restoration backbone can supply the SSL with clues to resolve this collapse, since its intermediate layers already produce partially restored features. We therefore propose Early Restoration Injection (ERI), which routes intermediate features from the restoration backbone into the SSL through a lightweight conditioning network. ERI excels over the unconditioned baseline and, on the VoiceFixer benchmark, achieves competitive performance among multiple baselines.
#### X-Dubbing: Speech-to-Speech Translation under Temporal and Visual Constraints
 - **Authors:** Jeongsoo Choi, Ji-Hoon Kim, Junseok Ahn, Joon Son Chung
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Computer Vision and Pattern Recognition (cs.CV); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2610.32042

 - **Pdf link:** https://arxiv.org/pdf/2610.32042

 - **Abstract**
 Cross-lingual dubbing aims to convert source speech into target-language speech that remains synchronized with the original video. The key to this task is to produce a faithful translation that is naturally speakable within the source timing, which is non-trivial because languages differ in duration, rhythm, and articulation. In this paper, we propose X-Dubbing, which formulates the task as a joint problem of time-aware speech-to-text translation and video-aware text-to-speech. Specifically, our method consists of a Dubbing Translator that generates temporally structured scripts and a Voice Actor that renders them in synchrony with the source video, both grounded in a language-agnostic timing budget. We further refine the translator with dubbing-aware feedback from the synthesized speech, encouraging scripts that yield well-synchronized dubbing. This design requires no paired dubbing data, as the two components are trained on speech-text and monolingual audio-visual data respectively. Extensive experiments show that X-Dubbing outperforms recent baselines across diverse metrics, including isochrony and audio-visual correspondence.
#### DIAL-S2ST: Learning Latency Control for Simultaneous Speech-to-Speech Translation
 - **Authors:** Jeongsoo Choi, Junseok Ahn, Joon Son Chung
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2610.32214

 - **Pdf link:** https://arxiv.org/pdf/2610.32214

 - **Abstract**
 We propose DIAL-S2ST, a post-training method for learning inference-time latency control in simultaneous speech-to-speech translation models. Our approach reuses pretrained translation capabilities through self-distillation from lookahead predictions, while limiting the influence of large text distillation losses to account for the imitation gap. This provides supervision for earlier emission without paired target data or reward computation. To make the learned timing adjustable at inference, we jointly train a single LoRA adapter across scales with matched lookahead guidance. Experiments on French-to-English translation with Hibiki demonstrate that only 10 hours of source speech enable this adaptation, providing flexible latency control with a favorable quality-latency trade-off.
#### Noisy Target Training Meets Drifting for Weakly-Supervised Speech Enhancement
 - **Authors:** Yoshiki Masuyama, Dominik Klement, Christoph Boeddeker, Julius Richter, Kohei Saijo, Gordon Wichern, Takahiro Edo, Jonathan Le Roux
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2610.32870

 - **Pdf link:** https://arxiv.org/pdf/2610.32870

 - **Abstract**
 In speech enhancement, standard supervised training requires pairs of noisy and clean speech. Such pairs are typically created by artificially degrading clean speech, leading to a domain mismatch between synthetic training data and real recordings. Noisy target training (NyTT) mitigates this limitation by training an enhancement model to predict a noisy recording from a noisier version generated by adding extra noise. However, since the training target is noisy, NyTT might learn to leave residual noise in the enhanced output. In this paper, we address this issue in NyTT by incorporating a Sinkhorn drifting loss that matches the distribution of the enhanced speech to that of unpaired clean speech. The Sinkhorn drifting loss attracts enhanced speech toward high-density regions of the clean-speech distribution while repelling it from the current enhanced-speech distribution in latent space. This loss uses only unpaired clean speech rather than the clean counterpart of each degraded recording, preserving the weakly-supervised nature of NyTT. Experimental results on VoiceBank-DEMAND and CHiME-3 real recordings demonstrate the efficacy of the proposed combination.


by Zyzzyva0381 (Windy). 


2026-10-26
