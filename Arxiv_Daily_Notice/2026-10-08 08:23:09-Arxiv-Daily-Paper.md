# Showing new listings for Thursday, 8 October 2026
Auto update papers at about 2:30am UTC (10:30am Beijing time) every weekday.


阅读 `Usage.md`了解如何使用此repo实现个性化的Arxiv论文推送

See `Usage.md` for instructions on how to personalize the repo. 


Keyword list: ['text-to-speech', 'text to speech', 'tts', 'LLM-based', 'speech', 'voice']


Excluded: []


### Today: 23papers 
#### Is Word Error Rate Enough? Rethinking Privacy Evaluation in Speech with Entity-Aware Metrics
 - **Authors:** Anjana Rajasekhar, Jule Pohlhausen, Nayana Jacob Alappattu, Anna Leschanowsky
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Computation and Language (cs.CL); Multimedia (cs.MM); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2610.08831

 - **Pdf link:** https://arxiv.org/pdf/2610.08831

 - **Abstract**
 As the use of smart devices continues to increase, their potential to capture sensitive speech content raises growing privacy concerns. It is therefore critical to develop techniques that prevent information leakage while preserving the utility of the audio, and evaluation metrics that accurately quantify the level of privacy without overestimating it. In this work, we evaluate the effectiveness of two obfuscation techniques in protecting speech content, with particular emphasis on named entities, by adapting entity-aware privacy metrics from the Natural Language Processing field to the speech privacy domain. Further, we investigate several attack scenarios and show that fine-tuning on entity-rich data improves attack performance for some entity categories but not others. Finally, we provide guidance on metric selection based on whether the obfuscation method preserves temporal alignment.
#### Intonation Perception in Real and Synthetic Speech across Varying Familiarity Levels: A Pilot Study of Equivalence Assessment
 - **Authors:** Hanrui Zhou, Gaoyuan Zhang, Yixiang Chen, Yujie Xing, Feng Xu, Xurong Xie, Hui Chen
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Human-Computer Interaction (cs.HC); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2610.08839

 - **Pdf link:** https://arxiv.org/pdf/2610.08839

 - **Abstract**
 Language training relies on a corpus constructed by a large number linguistic materials. AI-powered voice clones provide a way to construct the corpus with relatively low cost. Singing voice conversion (SVC) model is used to generate synthetic voices. This study compares participants' performances on natural and synthetic speech in two experiments, similarity perception and intonation recognition. In the accuracy of similarity perception task, a significant interaction between speech type and intonation is found, suggesting that question may serve as a cue for speaker identification but may be influenced by synthetic features. In the accuracy of intonation recognition task, a significant interaction between speech type and familiarity is observed, indicating that speech type affects how much familiarity contributes to voice processing.
#### UZH-CL at ArA-DF 2026: Prompt-Tuned Foundation Models and Track-Adaptive Score Fusion for Arabic Speech Deepfake Detection
 - **Authors:** Aref Farhadipour, Teodora Vukovic, Petr Motlicek
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2610.08844

 - **Pdf link:** https://arxiv.org/pdf/2610.08844

 - **Abstract**
 Detecting synthetic and voice-converted speech remains difficult for low-resource languages with dialectal diversity, where systems must generalize across regional dialects and unseen acoustic channels. We present the UZH-CL submission to the ArA-DF 2026 Shared Task on Arabic speech deepfake detection, covering Track~1 (dialect generalization) and Track~2 (acoustic robustness). We freeze a W2V-BERT-2.0 backbone and adapt it with \emph{Wavelet Prompt Tuning}, updating under 1% of parameters, and aggregate multi-layer representations with cross-layer attention and a general attentive-statistics pooling head rather than a specialized graph backend. Complementary detectors are obtained by varying adaptation strategy, training data, augmentation, and encoder family. We find that the two shift types require different fusion regimes: a broad multi-window ensemble for dialect generalization, and a compact, channel-matched, center-crop ensemble for acoustic robustness. Official evaluation yields 1.96% EER on Track~1 (6th place) and 1.04% EER on Track~2 (3rd place), corresponding to 87% and 96% relative reductions over the XLS-R+AASIST baselines.
#### Unified Shared Encoder in Spoof-Aware Speaker Verification with Hybrid Wavelet Prompt Tuning
 - **Authors:** Aref Farhadipour, Srikanth Madikeri, Teodora Vukovic, Volker Dellwo, Petr Motlicek
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2610.08845

 - **Pdf link:** https://arxiv.org/pdf/2610.08845

 - **Abstract**
 Spoofing-aware speaker verification (SASV) must confirm who is speaking and that the speech is genuine, but current systems gain one at the expense of the other. Modular cascades are accurate yet run two large self-supervised encoders, whereas end-to-end models are efficient but lose accuracy when a single embedding has to serve two conflicting objectives. We show that this trade-off between efficiency and specialization is avoidable. A single frozen W2V-BERT~2.0 backbone is adapted per task by deep hybrid Wavelet Prompt Tuning (WPT), so each branch obtains its own view of the shared encoder through dedicated prompts and a task head, and the scores are combined only at inference. Training only about 7M parameters, a small fraction of those a strong two-encoder cascade requires, the system surpasses it on SpoofCeleb evaluation with 0.03% CM-EER and 0.038 min a-DCF against 0.16% and 0.047. Because the backbone is shared and frozen, new branches attach without retraining the heads and prompts already in place. On ASVspoof5, with adversarial attacks and codec distortions, it reaches 0.092 min a-DCF and 3.60% CM-EER using only the provided data, indicating that the design holds under realistic in-the-wild threats.
#### Timestamped Hindi speech transcription using Whisper
 - **Authors:** Sanat Kumar Agrawal, Srikanth Raj Chetupalli
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2610.08847

 - **Pdf link:** https://arxiv.org/pdf/2610.08847

 - **Abstract**
 Time accurate speech transcription is essential in critical appli- cations, such as medical conversation transcription, reading as- sessment, and atypical speech recognition. Encoder-decoder auto- matic speech recognition models, such as Whisper, do not produce word-level timestamps out-of-the-box for all languages. Existing approaches to word-level time stamp estimation either use an exter- nal CTC model capable of frame level phoneme/character prediction which is computationally expensive, or use the cross attention scores internal to the model which gives coarse time stamps. We consider timestamped speech transcription for Hindi in this paper. We pro- pose a CTC-based approach, in which we train a low-complexity character level CTC head directly on the Whisper encoder output, to address the complexity and granularity limitations of existing works. We evaluate the system using manually-annotated word-level times- tamps. Experiments show that the proposed internal-CTC based approach performs similar to using an external CTC model and better than the cross attention score-based approaches. Further, we study the usage of CTC-head output to detect and mitigate hallu- cinations in conversational speech. Our results show that a simple token length criterion derived from the CTC output can improve the transcription quality for short segments where hallucinations are more probable.
#### Phoneme-Guided Initialization for LLM-based Speech Recognition
 - **Authors:** Ryo Magoshi, Shinsuke Sakai, Tatsuya Kawahara
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Computation and Language (cs.CL); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2610.08994

 - **Pdf link:** https://arxiv.org/pdf/2610.08994

 - **Abstract**
 Speech large language models (speech LLMs) perform well on automatic speech recognition (ASR) when sufficient paired speech-text data is available, but their performance degrades in low-resource settings. A cascaded pipeline that performs speech-to-phoneme (S2P) conversion followed by phoneme-to-grapheme (P2G) conversion has been shown to outperform end-to-end speech LLMs in this regime, suggesting that phoneme-mediated processing is beneficial when paired data is scarce. We propose \textit{phoneme-guided initialization}, a simple method that uses this insight within an end-to-end framework: we pre-train the audio encoder on S2P and the LLM on P2G tasks, then connect them and fine-tune the full model end-to-end on the target ASR task. Experiments on Japanese (CSJ), Chinese (AISHELL-1), and two low-resource languages from Common Voice 25.0 (Tatar and Urdu) show that our method matches or outperforms both the cascaded S2P-P2G baseline and the end-to-end model without P2G initialization.
#### VM-ARRAYDPS: Virtual Microphone Augmented Diffusion Posterior Sampling for Unsupervised Blind Speech Separation
 - **Authors:** Jingqi Sun, Haozhan Tang, Shulin He, Zhong-Qiu Wang
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Machine Learning (cs.LG); Multimedia (cs.MM); Sound (cs.SD); Signal Processing (eess.SP)
 - **Arxiv link:** https://arxiv.org/abs/2610.09334

 - **Pdf link:** https://arxiv.org/pdf/2610.09334

 - **Abstract**
 Blind Source Separation(BSS) is a fundamental problem in signal processing, aiming to separate multiple source signals from their mixtures without prior knowledge of the sources or the mixing process. Traditional approaches, such as Independent Vector Analysis (IVA) exploits statistical independence of sources. Recently, diffusion-based approaches have emerged as a promising alternative by leveraging powerful generative priors. Among them, ArrayDPS formulates BSS problem as a posterior sampling problem, and utilizes a pretrained speech diffusion model to guide the recovery of clean source signals. A key factor behind its separation capability is the multi-channel consistency (MC) objective, which enforces the estimated source signals to reconstruct the observed microphone mixtures through the estimated acoustic transfer functions. However, the number of microphones in the array is often limited, which constrains the performance of ArrayDPS. To address this issue, we propose VM-ArrayDPS, a novel method that augments the microphone array with virtual microphones with higher-SNR, these microphones can offer extra MC constraints to enhance the separation performance. Experimental results demonstrate that VM-ArrayDPS significantly outperforms ArrayDPS on both 2-speaker and 3-speaker datasets, showcasing the effectiveness of virtual microphone augmentation in improving BSS performance. We also did ablation studies to show the influence of the number of virtual microphones and weight of the MC objective brought by virtual microphones.
#### Beyond Token Revision: Investigating Mask-and-Replace Diffusion for Zero-Shot Text-to-Speech
 - **Authors:** Hounsu Kim, Joonyong Park, Yuki Saito, Satoru Fukayama, Juhan Nam
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2610.09448

 - **Pdf link:** https://arxiv.org/pdf/2610.09448

 - **Abstract**
 Unlike autoregressive models, discrete diffusion-based models for zero-shot text-to-speech generate speech tokens in parallel and can revisit earlier predictions. Mask-and-replace training extends mask-only training by randomly replacing some tokens, and its gains are commonly attributed to self-correction, the ability to revise previously generated tokens. However, exposure to randomly perturbed context during training may itself improve generation, raising the question of whether these gains require inference-time token revision. To investigate this question, we use DeMaR, which combines mask-and-replace training with confidence-ranked mask-only sampling while preserving the total training corruption probability. Trained from scratch on LibriTTS, DeMaR achieves lower word error rates (WER) than autoregressive and mask-only diffusion baselines using the same speech tokenizer. This advantage persists when each token remains unchanged after first being unmasked. Matched training conditions on two heterogeneous speech tokenizers show that both noisy-context augmentation and replacement supervision improve WER under this restriction. These findings demonstrate training-side benefits of replacement beyond enabling inference-time token revision.
#### Boundary-Free Contextual Biasing: Depth-Adaptive Gating and Reading-Space Matching for Unsegmented Languages
 - **Authors:** Muhammad Huzaifah, Yu Pan, Zachary Yeo, Ningjie Bai, Guangzhao Yang
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Computation and Language (cs.CL); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2610.09467

 - **Pdf link:** https://arxiv.org/pdf/2610.09467

 - **Abstract**
 Contextual biasing supplies an ASR system with a list of expected words at inference time, but existing methods rely on word boundaries that Japanese and Chinese do not provide. We present a boundary-free biasing decoder for frozen public CTC models, built on a character-level Aho-Corasick automaton, with no training and no second pass. Two evidence-based mechanisms replace the boundary: a depth-adaptive gate that sets how hard to push from match depth, and reading-space matching for when the audio is right but the characters are wrong. On Aishell-1 NE's hard R1 subset we reach 66.5% recall, above the trained CLAS baseline (64%), transferring to WenetSpeech and to a second architecture without retuning. We release the first open Japanese contextual-biasing benchmark, where biasing lifts rare-word recall by 25 points at precision above 97%, and still by 19 and 22 points against 1,000-word lists.
#### Mitigating Accent-Language Confusion in Self-Supervised Speech Representations for Language Identification
 - **Authors:** Minu Kim, Jihwan Lee, David R. Mortensen, Shrikanth Narayanan
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Computation and Language (cs.CL); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2610.09486

 - **Pdf link:** https://arxiv.org/pdf/2610.09486

 - **Abstract**
 Spoken language identification (LID) aims to recognize the target language regardless of accent. In practice, however, LID models fine-tuned from self-supervised speech representations frequently confuse accents with languages, misclassifying non-native (L2) speech as the speaker's first language (L1). We show that non-native speech representations lie between native target-language and native L1 poles, causing systematic misclassification. To address this, we introduce a geometric projection that estimates an L1-bias direction solely from native speech and removes it before the frozen LID head. Across five MMS-LID models and non-native corpora, this projection substantially improves target language identification for L2-accented speech while preserving predictions for native speech. These results show that accent-induced L1 bias can be corrected directly within the representation space without L2 training data or model adaptation.
#### A Lightweight Hybrid Framework for Acoustic Echo Cancellation and Suppression via Constrained Recursive Training
 - **Authors:** Huawei Zhang, Rilin Chen, Hao Zhang, Meng Yu, Dong Yu
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2610.09655

 - **Pdf link:** https://arxiv.org/pdf/2610.09655

 - **Abstract**
 We propose a hybrid framework that combines deep learning with adaptive filtering for acoustic echo cancellation (AEC) and suppression. It integrates a multi-output neural network to jointly update filter parameters and perform post-processing. A learnable variant of the normalized least mean squares (NLMS) filter is introduced, incorporating a variable step size and a variable transition factor that are recursively updated for dynamic adaptation. In addition, an enhanced near-end speech estimate is generated from the filter output to suppress the residual echoes. To ensure training stability, we propose a constrained recursive training strategy that aligns with the adaptive nature of the filter, where an upper-bound constraint is introduced on the filter output. For efficiency, the framework adopts a lightweight implementation with only 253 k parameters. Experimental results show that the framework outperforms the baselines in terms of both signal quality and speech intelligibility.
#### Training-Free Instruction TTS Gender Bias Calibration Using Model-Adaptive Steering
 - **Authors:** Kuan-Yu Chen, Yi-Cheng Lin, Jeng-Lin Li, Jian-Jiun Ding
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2610.09831

 - **Pdf link:** https://arxiv.org/pdf/2610.09831

 - **Abstract**
 Instruction text-to-speech (ITTS) systems systematically encode acoustic gender skews from descriptive style prompts, such as occupations or personas, even when demographic attributes are left unspecified. Calibrating the implicit gender distribution of the synthesized voices, across heterogeneous architectures and without model retraining or overriding explicit user prompts, is an open problem. In this work, we propose \textit{model-adaptive steering}, a training-free bias calibration method that steers post-encoder conditioning representations using a group deviation vector paired with a coarse-to-fine strength search on a development set. A deterministic lexical gate bypasses intervention whenever explicit gender keywords are detected, preserving intended prompt semantics. Evaluated on a 12,800-prompt held-out benchmark across four ITTS models (three architectures), the proposed method reduces aggregate calibration error from 11.5--28.1 to 0.8--5.9 percentage points (e.g., shifting Parler-TTS Mini from 78.1\% and PromptTTS++ from 27.3\% female to 50.8--52.4\%). Output rates stay within 0.9 pp across anchor set sizes at fixed operating points, while UTMOS decreases by at most 0.08 and WER degrades by at most 3.3 pp.
#### Toward Part-Aware Choral Transcription with singing voice assignment
 - **Authors:** Hanyu Meng, Zhanhong He, Zixun Guo, Yaolong Ju
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2610.09861

 - **Pdf link:** https://arxiv.org/pdf/2610.09861

 - **Abstract**
 Single-instrument automatic music transcription (AMT) has advanced substantially, yet choral applications require soprano, alto, tenor, and bass (SATB) to be transcribed as separate parts. Recent note-level choral AMT instead produces a single merged note track, limiting rehearsal, education, and score reconstruction. To address this limitation, we introduce Part-aware Choral Transcription (PawCT), to our knowledge the first end-to-end neural framework that identifies active SATB parts from choral audio and transcribes each into a separate note-level track. PawCT combines part-specific onset, offset, and frame prediction with part-presence estimation, union-level supervision, and structured training targets using a range prior (RP) based on SATB pitch ranges and its ordered-continuity (OC) extension, which adds within-part melodic continuity and cross-part pitch ordering. On YouChorale, PawCT-RP-OC achieves a macro part-aware note F1 of 0.225 at a 50-ms onset tolerance, outperforming an adapted choral baseline (0.165) by 36.4% relative and a two-stage post-hoc assignment pipeline (0.175). Its part-agnostic variant, PagCT, achieves a 50-ms onset F1 of 0.382, compared with 0.237 for the previous state-of-the-art choral AMT model. Cross-dataset evaluations on CSD and Cantoria further assess performance under dataset shift. These results demonstrate the benefit of jointly modeling note transcription and vocal-part assignment. Code and demos are available at this https URL.
#### LIFT-SE: Linguistic Inference Followed by Flow Transformation for Generative Speech Enhancement
 - **Authors:** Haoyin Yan, Chengwei Liu, Zheng Xue, Xiaotao Liang, Jifa Cai, Zeyu Zhao, Jingjing Wang
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2610.09963

 - **Pdf link:** https://arxiv.org/pdf/2610.09963

 - **Abstract**
 Generative speech enhancement (SE) is prone to linguistic hallucination when semantic constraint is unreliable under severe noise and reverberation. Moreover, approaches that generate discrete codec tokens are bounded by the quantization error of the codec decoder, regardless of token-prediction accuracy. We propose LIFT-SE, a two-stage generative framework that decouples linguistic inference from acoustic synthesis within QRes-Codec, which exposes a quantized latent and its residual-completed continuous form. The first stage predicts clean codec tokens autoregressively, conditioned on frame-aligned features from a self-supervised front-end distilled toward clean speech. The second stage applies conditional flow matching to transport Gaussian noise to the continuous latent conditioned on the predicted tokens, and the frozen decoder reconstructs the enhanced waveform. Discrete generation provides naturalness, while continuous refinement restores signal fidelity. Experiments on the DNS1 and URGENT benchmarks show that LIFT-SE attains favorable linguistic consistency under reverberant conditions together with competitive perceptual quality, and systematic ablations verify the necessity of both stages. Code will be released in the future.
#### Source-Directed Trajectory Perturbation at First-Order Cost for Domain Generalization in Speech Deepfake Detection
 - **Authors:** Siqing Qin, Kong Aik Lee, Youzhi Tu
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2610.10094

 - **Pdf link:** https://arxiv.org/pdf/2610.10094

 - **Abstract**
 Speech deepfake detectors often lose accuracy when the distribution of the test data differs from that of the training data. Meta-learning for domain generalization (MLDG) shows promise by simulating domain shifts with episodic meta-train and meta-test splits. However, the MLDG meta-objective considers only a single clean adaptation trajectory and does not account for the local loss landscape around its endpoints. To address this limitation, we first propose an explicit worst-case MLDG variant, dubbed WC-MLDG-4P, which perturbs both the meta-train and meta-test states but requires four gradient evaluations per episode. We then introduce WC-MLDG-2P, a source-directed alternative to the explicit robust objective. It shifts the clean MLDG endpoint toward a locally higher source-loss state and evaluates the meta-test gradient there, retaining the two-evaluation cost of the first-order MLDG. A first-order expansion relates the resulting gradient change to meta-test directional curvature along the source gradient without explicitly computing a Hessian. Relative to MLDG, WC-MLDG-2P achieves relative mean-EER reductions of 23.4% with XLSR-AASIST and 7.6% with XLSR-Conformer-TCM, respectively.
#### DuRe-ST: Dual-Relation Spectro-Temporal Modeling for Speech Deepfake Detection
 - **Authors:** Shaole Li, Siqing Qin, Youzhi Tu, Kong Aik Lee
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2610.10121

 - **Pdf link:** https://arxiv.org/pdf/2610.10121

 - **Abstract**
 Previous speech deepfake detectors can adaptively capture spectro-temporal dependencies through graph attention, yet they largely overlook the co-variation between spectral and temporal representations. To address this gap, we construct a normalized affinity graph from their joint covariance and apply polynomial graph filtering to capture higher-order covariance-induced dependencies. We first develop Cov-ST to isolate the contribution of covariance-based relational modeling. Although it improves detection performance, its sensitivity to the polynomial order suggests limited robustness when covariance relations are modeled alone. We therefore propose DuRe-ST, which jointly exploits covariance-induced and graph-attention-induced relations to capture complementary second-order co-variation and adaptive spectro-temporal dependencies. Experiments show that DuRe-ST achieves an average relative EER reduction of 25.9% over XLSR-AASIST on the ASVspoof benchmarks and 28.4% across four cross-dataset benchmarks with only 4-8k additional trainable back-end parameters. It further outperforms the strongest publicly available comparison models by 2.2-13.2% in relative EER on four benchmarks, while remaining smaller than the publicly available models considered.
#### Benchmarking Adult Addressee Classification Across Child- and Adult-Directed Speech Datasets
 - **Authors:** Sofoklis Kakouros, Daniil Kocharov, Okko Räsänen
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2610.10240

 - **Pdf link:** https://arxiv.org/pdf/2610.10240

 - **Abstract**
 In this work, we present a comprehensive analysis of classification performance for distinguishing child-directed speech (CDS) from adult-directed speech (ADS) using speech data from corpora containing natural in-lab and in-the-wild CDS and ADS. We establish classification benchmarks for these datasets using self-supervised learning (SSL) representations, along with a range of time-pooled representations that go beyond first- and second-order statistics by incorporating cross-channel covariances in high-dimensional embeddings. In addition, we probe these representations to examine how different pooling methods capture prosodic information using linear probes. Overall, SSL-based representations prove particularly effective, achieving the best performance, while different pooling methods offer complementary advantages for the task.
#### Child ASR Adaptation with Adult Retention: An Empirical Study
 - **Authors:** Houssam Eddine-Othman Lachemat, Shammur Absar Chowdhury
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Artificial Intelligence (cs.AI); Machine Learning (cs.LG); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2610.08827

 - **Pdf link:** https://arxiv.org/pdf/2610.08827

 - **Abstract**
 Automatic Speech Recognition (ASR) systems often underperform for children and non-native speakers, while adapting adult ASR models to child speech can cause adult-speech forgetting. We study child ASR adaptation with adult retention across Arabic and English. We compare full fine-tuning, LoRA, and post-hoc weight-space merging across encoder--decoder, encoder--CTC, and AudioLLM-based ASR systems. Experiments use Arabic native and non-native child speech, English MyST child speech, and adult benchmarks from MGB-2 and LibriSpeech test-clean. We evaluate recognition quality with WER and quantify the adaptation--retention trade-off using Retention Index, Child Adaptation Gain, and Adaptation Recovery. Results show that child adaptation is necessary, especially for non-native Arabic and English child speech, but direct adaptation often reduces adult ASR performance. Bilingual adaptation is more stable than language-specific adaptation. Weight-space merging often improves the trade-off, especially for encoder--CTC, Whisper, and AudioLLM-based ASR, with LERP favoring adult retention and TIES recovering stronger child gains. For the encoder--decoder model, direct bilingual fine-tuning remains strongest in raw WER.\footnote{Code, and models are available at this https URL.
#### When Forgetting Looks Like Improvement: Metric Masking in Streaming Diarizer Adaptation and the Price of Rehearsal
 - **Authors:** Mo Yu, Yang Liu, Jing Qian
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2610.08828

 - **Pdf link:** https://arxiv.org/pdf/2610.08828

 - **Abstract**
 Small-data adaptation can improve speech detection while degrading speaker attribution. We study this discrepancy in a released streaming diarizer adapted on 7.5 h of two-party conversation and evaluated across six corpora. Adaptation substantially improves in-domain diarization performance and transfers to an independent corpus. However, this improvement is not consistent across evaluation scenarios as the additional confusion is mainly associated with impaired temporal identity consistency rather than speaker-count errors. A local-remapping diagnostic reveals different patterns of identity degradation across corpora, indicating that adaptation may alter how streaming models maintain speaker assignments over time. Rehearsal reduces the observed degradation but reduces the cross-domain transfer performance. These results highlight the need to jointly evaluate detection accuracy, identity consistency, and retention behavior when adapting streaming diarization systems.
#### JoyAI-Voice 2.0: A Full-Continuous Autoregressive Speech Generation Model with Semantic-Acoustic Joint Representation
 - **Authors:** Yafeng Chen, Boya Dong, Yankun Huang, Hao Li, Jingdong Li, Xiangyu Liang, Hao Ni, Wenchao Wang, Yuxuan Wang, Zhangyu Xiao, Wei Deng, Nan Duan, Yu Gu, Wenhao Guan (Intern), Weisheng Han, Yabin Li, Yuan Liu, Jiaxin Ye (Intern), Fan Yu, Lin Zhu
 - **Subjects:** Subjects:
Sound (cs.SD); Artificial Intelligence (cs.AI); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2610.08834

 - **Pdf link:** https://arxiv.org/pdf/2610.08834

 - **Abstract**
 We present JoyAI-Voice~2.0, an end-to-end anthropomorphic speech generation model built upon a fully continuous, dual-encoder architecture. Raw speech is encoded into continuous latents and partitioned into patches. Each patch is decomposed by a semantic-acoustic dual encoder into a semantically purified representation and an acoustic representation, which are fused and jointly fed to a causal autoregressive Transformer for planning. The Transformer predicts the conditioning for the next patch, and a local diffusion Transformer renders its full latents for 48\,kHz synthesis. The model is trained with a joint flow-matching and stop-prediction objective, followed by supervised fine-tuning and reinforcement learning with DiffusionNFT to improve model performance. It achieves the lowest average word error rate of 2.51\% on Seed-TTS, a 14.9\% relative reduction over the strongest baseline, and state-of-the-art attribute fidelity on InstructTTSEval in both Chinese and English, leading on 5 of 10 perceptual dimensions with the highest overall score of 0.893 on MDVD-Eval.
#### Dialect-Robust Speech Language Models with Synthetic Pseudo-Dialect Augmentation
 - **Authors:** Shunsuke Mitsumori, Tomoya Mizumoto, Yusuke Fujita
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2610.09321

 - **Pdf link:** https://arxiv.org/pdf/2610.09321

 - **Abstract**
 Speech Language Model (SLM) performance often degrades on dialects due to data scarcity. Conventional text-to-speech (TTS) augmentation struggles to cover diverse dialects as it requires a certain amount of real dialect speech. We propose synthesizing pseudo-dialect speech by converting LLM-generated dialect text via a standard-language TTS model, requiring zero real dialect speech. Additionally, we introduce intermediate standard-text prediction during training, acting as semantic normalization for downstream tasks. We evaluate dialect understanding via dialect-to-English speech translation across Japanese, German, and Chinese dialects. Compared to synthetic standard speech baselines, pseudo-dialect augmentation improves scores for Japanese (from 25.38 to 26.24) and German (from 31.57 to 32.47). Furthermore, the intermediate standard-text prediction effectively bridges the semantic gap, boosting performance to 28.26 for Japanese and from 11.67 to 16.37 for Chinese. These results suggest that our approach scales to various languages without requiring speech resources specific to each dialect.
#### CrossEdit: Cross-Modal Training Enables Rich Audio-Visual Editing
 - **Authors:** William Chen, Prem Seetharaman, Ke Chen, Oriol Nieto, Kevin Duarte, Siddharth Srinivasan Iyer, Mamshad Nayeem Rizve, Zhiwen Cao, Shinji Watanabe, Yuanjun Xiong, Jianming Zhang, Zeyu Jin, Justin Salamon
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS); Image and Video Processing (eess.IV)
 - **Arxiv link:** https://arxiv.org/abs/2610.10264

 - **Pdf link:** https://arxiv.org/pdf/2610.10264

 - **Abstract**
 Multimedia editing requires generative models to interpret complex, compositional instructions and modify only the desired elements across modalities, a capability largely beyond existing systems. Current approaches are typically limited to simple, single-attribute edits within one modality, since diverse, high-quality editing pairs are hard to obtain, especially for cross-modal tasks where aligned audiovisual (AV) data is scarce. We present CrossEdit, a unified omni-modal editing model for images, audio, and video that performs AV movie scene edits zero-shot through cross-modal transfer. Our key observation is that complex instruction-following, once learned in any modality, generalizes to others. We exploit the additive nature of acoustic signals to procedurally generate large-scale audio editing pairs with compositional instructions, and show that fine-tuning on this synthetic data and self-supervised AV masked reconstruction, alongside a targeted set of curated cross-modal tasks, induces instruction-following that transfers zero-shot to unseen modality and instruction combinations. To evaluate this capability, we release CrossEditBench, a human-annotated benchmark of AV edits on movie scenes, and propose AV-FES, a metric that jointly scores instruction following and consistency. We show that our proposed techniques improve performance on zero-shot AV movie scene editing, while maintaining or improving performance on lip-synced speech editing and standard image, video, and audio editing benchmarks. Demos are available at this https URL.
#### Steerspeech: Activation Steering For Emotion Control In Generated Speech
 - **Authors:** Afsara Benazir, Darius Pétermann, Felix Xiaozhu Lin, Salar Rahili
 - **Subjects:** Subjects:
Sound (cs.SD); Machine Learning (cs.LG); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2610.10415

 - **Pdf link:** https://arxiv.org/pdf/2610.10415

 - **Abstract**
 Pretrained text-to-speech (TTS) models can generate expressive speech, but reliable inference-time emotion control remains challenging: prompts and reference audio offer coarse, inconsistent control, whereas specialized conditioning and model adaptation require costly training. We present SteerSpeech, a lightweight activation-steering framework that controls emotion by injecting steering vectors into hidden activations. For each target emotion we train a lightweight low-rank transform, using a multi-expert objective that encourages monotonic emotion control while preserving speaker identity and linguistic content, constraining steering drift, and keeping the TTS backbone frozen. To optimize through discrete speech tokens, we introduce a two-pass generation-and-replay pipeline using a straight-through estimator to backpropagate expert supervision through sampled tokens. At inference, a target-emotion steering direction is optimized with its respective transform and injected into the base TTS model. Objective and subjective evaluations with Qwen3-TTS across seen, unseen, and accented speakers show stronger continuous emotion control with limited speaker and content degradation. SteerSpeech achieves 1.08x-7.12x baseline target-emotion scores and for a representative emotion subjectively, it receives 78.1%-96.8% intensity preference and 1.43x-1.46x speaker-identity preservation at high steering strengths.


by Zyzzyva0381 (Windy). 


2026-10-08
