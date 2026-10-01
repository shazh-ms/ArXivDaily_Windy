# Showing new listings for Thursday, 1 October 2026
Auto update papers at about 2:30am UTC (10:30am Beijing time) every weekday.


阅读 `Usage.md`了解如何使用此repo实现个性化的Arxiv论文推送

See `Usage.md` for instructions on how to personalize the repo. 


Keyword list: ['text-to-speech', 'text to speech', 'tts', 'LLM-based', 'speech', 'voice']


Excluded: []


### Today: 15papers 
#### Monotonicity-Guided Semantic Alignment for Zero-shot Multispeaker Image-to-Speech Synthesis
 - **Authors:** Lijun Wang, Yixian Lu, Shogo Okada
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2609.38440

 - **Pdf link:** https://arxiv.org/pdf/2609.38440

 - **Abstract**
 Direct image-to-speech (Img2Sp) poses an alignment challenge in mapping visual content to ordered speech sequences, as images permit multiple spoken descriptions and lack monotonic correspondence with speech sequences. We propose Monotonicity-Guided Semantic Alignment (MGSA), to the best of our knowledge, the first framework for zero-shot multispeaker Img2Sp synthesis. We use semantic speech units to provide shared content targets across speakers with reference speech for speaker conditioning. A query aligner maps semantic memory learned from visual content to speech unit positions via a soft monotonic prior, which yields position-specific conditioning states. A blockwise masked diffusion generator is employed for the speech unit generation conditioning on these states. Experiments on Flickr8k-Audio show competitive captioning performance against single-speaker baselines, while evaluation with LibriTTS-R references supports zero-shot synthesis for unseen speakers. Ablations validate the effectiveness of aligner and block diffusion. Audio samples are available at this https URL.
#### Voices as Handles: Reasoning about Speaker Identity with Frozen Text LLMs
 - **Authors:** Runqiu Xu, Zhisheng Zheng, David Harwath
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2609.38501

 - **Pdf link:** https://arxiv.org/pdf/2609.38501

 - **Abstract**
 Multi-user voice agents must track who said what across dialogue sessions. Text LLMs are attractive backbones for such agents, but transcripts alone do not expose acoustic speaker identity, leaving the model without a persistent reference for linking information to speakers across sessions. We address this gap by introducing Speaker Handles, soft-token representations that expose acoustic speaker identity to a frozen text LLM for cross-session speaker-dependent reasoning. A three-stage curriculum trains a lightweight projector, with fewer than 0.1% of the backbone's parameters, to map speaker embeddings into these handles. Establishing whether the resulting handles truly support cross-session speaker-dependent reasoning is challenging with existing benchmarks because textual cues can partially reveal fact ownership. We therefore present SpeakerBind, a controlled shared-agent benchmark in which overlapping facts across users require correct cross-session speaker attribution. Speaker Handles achieve 97.40-98.36% accuracy on VoxCeleb1 and 70.40% on SpeakerBind, close to the 71.88% topline. These results show that the proposed Speaker Handles provide an efficient way to integrate acoustic speaker identity into frozen text LLMs for speaker-content reasoning.
#### Tacit-TTS: From Autoregressive Decoding to Masked Prediction for Efficient Transcript-Free Voice Cloning
 - **Authors:** Jian Chen, You Zhang, Mark Vinton
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Computation and Language (cs.CL); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2609.38658

 - **Pdf link:** https://arxiv.org/pdf/2609.38658

 - **Abstract**
 TTS systems with autoregressive semantic modeling have demonstrated strong zero-shot voice cloning performance and rich expressive variation, but their sequential decoding incurs substantial latency. Non-autoregressive alternatives offer much faster generation, yet often rely on more restrictive reference conditioning, such as requiring transcripts of the reference speech during inference. We present Tacit-TTS, an efficient transcript-free zero-shot voice cloning system distilled from IndexTTS2. Our model replaces autoregressive text-to-semantic decoding with masked non-autoregressive generation, introduces training-free acoustic length estimation, and accelerates the flow-matching renderer through ReFlow distillation. Across two English and two Mandarin datasets, Tacit-TTS achieves competitive zero-shot quality while generating speech over 10x faster than IndexTTS2 for utterances longer than 5 seconds. Its transcript-free conditioning further supports cross-lingual and non-lexical references. We validate this capability using references from eight other languages, infant babble, and synthetic gibberish, where transcript-dependent systems often degrade or fail due to unreliable ASR transcripts.
#### A barrier or a booster? Familiarity effects on Mandarin emotion prosody recognition using AI-powered voice cloning
 - **Authors:** Feng Xu, Gaoyuan Zhang, Shanshan Xue, Yixiang Chen, Hanrui Zhou, Xurong Xie, Hui Chen
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Human-Computer Interaction (cs.HC); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2609.38794

 - **Pdf link:** https://arxiv.org/pdf/2609.38794

 - **Abstract**
 Emotion prosody perception requires simultaneous processing of acoustic cues and speaker identity. While listeners effortlessly decode natural speech, AI synthetic voices introduce cognitive complexities due to subtle acoustic atypicalities. It remains unclear how these synthetic features interact with a listener's prior social knowledge and memory of a familiar speaker. This study investigated how speech sources (human vs. AI) and speaker familiarity affect emotion recognition accuracy and cognitive load. A within-subject task with Mandarin-speaking adults evaluated behavioral (accuracy, reaction time) and physiological data (heart rate variability). Results showed that human voices yielded significantly higher accuracy and faster processing times than AI voices, while HRV did not significantly differentiate between conditions. These findings show that decoding synthetic speech is gated by top-down social cognition, highlighting limitations in current AI synthesis technologies.
#### VOSSA: Voiceprint Optimization for Streaming Speech Architectures
 - **Authors:** Mu-Ruei Tseng, Waris Quamer, Ghady Nasrallah, Ricardo Gutierrez-Osuna
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Computation and Language (cs.CL); Machine Learning (cs.LG)
 - **Arxiv link:** https://arxiv.org/abs/2609.38887

 - **Pdf link:** https://arxiv.org/pdf/2609.38887

 - **Abstract**
 Real-time voice conversion (VC) systems commonly rely on pretrained speaker embeddings from automatic speaker verification (ASV) models. While effective for speaker discrimination, these embeddings are trained to remain stable across phonetic and prosodic variations within-speaker, which may conflict with frame-level acoustic generation in streaming constraints. To address this issue, we propose VOSSA (Voiceprint Optimization for Streaming Speech Architectures), a speaker representation framework that extracts speaker information from intermediate content encoder layers and aggregates using attentive statistics pooling. The embedding is trained jointly with VC objectives, removing the need for a separate speaker encoder. Across six datasets, VOSSA improves F0 dynamics and vowel-discriminative acoustic cues while maintaining comparable NISQA-MOS, WER, and speaker similarity. Perceptual tests further indicate improvements in naturalness, speaker similarity, intelligibility, and vibrancy.
#### Improving Predicted MOS Scores, Not Perceived Quality: Multi-Predictor Test-Time Optimization of Enhanced Speech
 - **Authors:** Tsubasa Ochiai, Marc Delcroix, Nahomi Kusunoki, Rintaro Ikeshita, Naohiro Tawara, Naoyuki Kamo, Tetsuji Ogawa, Shoko Araki
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2609.39028

 - **Pdf link:** https://arxiv.org/pdf/2609.39028

 - **Abstract**
 Non-intrusive MOS predictors are widely used instead of subjective listening tests to evaluate and rank speech enhancement (SE) systems. If they accurately reflect perceived quality, raising their scores should lead to higher-quality speech. We present the first comprehensive analysis of test-time optimization for the SE task, which directly modifies the enhanced signal to raise the average of multiple MOS predictor scores. On seven systems from the URGENT 2026 challenge, we find that 1)~all the optimized predicted scores increase while reference-based metrics remain nearly unchanged, 2)~a non-optimized predicted score does not increase, and 3)~a MUSHRA listening test shows no improvement in perceived quality. These findings reveal a risk that such optimization can distort evaluations, e.g., biasing comparisons of SE systems regardless of their perceived quality. We believe these findings can inform future evaluation practices: they suggest that predictors used for optimization should not be used for evaluation, and that challenges should keep the predictors used for ranking undisclosed.
#### SURE-EVAL: A Systematic and Unified Agentic Framework for Reproducible Evaluation
 - **Authors:** Jing Peng, Junhao Du, Yixuan Wang, Bowen Wang, Hanqi Li, Chaolei Liu, Weihan Chen, Haohui Xie, Ruichen Sun, Chenghao Wang, Wen Wen, Guanyu Chen, Xiaoyu Gu, Haoyu Li, Yiwei Guo, Bohan Li, Tao Liu, Yucheng Wang, Yu Xi, Yihua Zhou, Qiang Zhou, Feng Lu, Shuai Wang, Kai Yu
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2609.39030

 - **Pdf link:** https://arxiv.org/pdf/2609.39030

 - **Abstract**
 Audio and speech models are released rapidly, but reported scores often conflate model capability with deployment and evaluation choices. The same checkpoint can produce different predictions under different runtimes, hardware, decoding settings, or fallback policies. Even fixed predictions can receive different scores under different normalization and metric implementations. Existing speech benchmarks standardize selected datasets or scoring procedures, but rarely connect heterogeneous model onboarding, controlled inference, and versioned scoring in one executable workflow. We introduce SURE-EVAL, a Systematic and Unified Agentic framework for Reproducible Evaluation of audio and speech systems. A Tool Agent Workflow converts model releases into isolated, verified callable tools. A Main Agent Workflow commits task-specific inference and scoring protocols, then delegates all score-bearing operations to versioned deterministic programs. Each result retains its runtime, protocol, pipeline nodes, predictions, and audit artifacts. Across 18 public releases covering automatic speech recognition, text-to-speech, voice conversion, speaker diarization, speaker-attributed recognition, and multi-task audio understanding, a Codex-only baseline completes 12 models in one shot, while the same agent with the SURE-EVAL Tool Agent Workflow completes all 18. We also conduct unified evaluations over seven ASR test conditions and two TTS subsets. A protocol analysis of three TTS systems finds absolute differences of 0.02-0.52 points between paper-reported and unified results, with the direction varying by model and language. These results show that reproducible evaluation requires controlling both model execution and output scoring.
#### DuSpaR: Dual-State Sparsifying Recurrent Unit with Feedback Modulation for Compute-Efficient Speech Processing
 - **Authors:** Zixiao Li, Sheng Zhou, Longbiao Cheng, Shih-Chii Liu
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2609.39237

 - **Pdf link:** https://arxiv.org/pdf/2609.39237

 - **Abstract**
 We introduce the Dual-state Sparsifying Recurrent Unit (DuSpaR) as a computationally efficient building block for speech processing models on resource-constrained edge devices. It employs dual-state recurrence to modulate its input vectors in a stateful feedback loop. Its recurrent cells sparsify the input vector operand involved in matrix-vector multiplication using ReLU activation. By skipping the zero entries dynamically, inference-time savings in multiply-accumulate operations and weight memory fetches can be achieved. We evaluate DuSpaR on three speech tasks: keyword spotting (KWS) on the Google Speech Commands dataset, spoken language understanding (SLU) on the Fluent Speech Commands dataset, and speech enhancement (SE) on the Voice Bank + Demand (VBD) dataset. At similar parameter counts, DuSpaR requires 51.0% and 68.4% less computation than Gated Recurrent Unit (GRU) on KWS and SLU, respectively, while achieving higher accuracy, and 50.1% less computation on SE while maintaining similar quality. At comparable computational cost and across a range of model sizes, DuSpaR also achieves higher KWS/SLU accuracy and better SE quality than other sparsity-aware recurrent models. Ablation studies show that compared with the single-state recurrence baseline, dual-state recurrence reduces the effective compute by factors of 3.1 to 11.3 at similar task performance.
#### Pitch Smoothing Using Relative Interval Networks
 - **Authors:** Chin-Yun Yu, Chi-Jen Peng, Li Su, György Fazekas
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2609.39852

 - **Pdf link:** https://arxiv.org/pdf/2609.39852

 - **Abstract**
 Pitch tracking systems typically couple a per-frame fundamental frequency ($F_0$) estimator with a temporal smoothing stage to obtain continuous trajectories. Conventional Viterbi smoothers enforce first-order continuity but lack long-term temporal awareness and could lock into octave errors across corrupted frames. We propose Relative Interval Networks (RIN), a trajectory smoothing framework that reconciles per-frame pitch estimates with data-driven multi-hop pitch differences. We extract robust relative pitch intervals across arbitrary frame offsets using Variable-Q Transform cross-correlation. We formulate pitch smoothing as an $L_1$-norm optimization problem and prove its equivalence to a minimum cost circulation problem, solved efficiently via linear programming. Evaluations across speech, singing, and instrumental datasets show that RIN substantially improves weak estimators, matches or outperforms Viterbi decoding at a comparable computational cost, and provides superior robustness under certain acoustic degradation.
#### Automatic estimation of verbal fluency index in people with Motor Neuron Disease using ASR alignment and pause modelling
 - **Authors:** Bahman Mirheidari, Leslie Ing, Daniel Blackburn, Sharon Abrahams, Christopher McDermott, Heidi Christensen
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Artificial Intelligence (cs.AI); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2609.38203

 - **Pdf link:** https://arxiv.org/pdf/2609.38203

 - **Abstract**
 Monitoring cognitive impairment (CI) in motor neuron disease (MND) is essential for timely treatment and care, yet challenging due to co-occurring speech difficulties. The Edinburgh Cognitive and Behavioural ALS Screen (ECAS) provides a robust metric for CI assessment, with the Verbal Fluency Index (VFI) a central element. Building on recent advances in automated speech analysis, this study proposes a system for estimating VFI. It leverages a unique MND dataset and combines ASR (WhisperX) and VAD (Silero) with refined timestamping to predict the VFI and extract several clinically interpretable measures. Our approach outperformed systems based on traditional acoustic features and self-supervised embeddings, evaluated using multiple regression algorithms. Clinically inspired features consistently outperformed the other sets, with the best models achieving strong results (P-words: R2 0.9, NRMSE 0.05; S-words: R2 0.8, NRMSE 0.08), demonstrating the feasibility of automated VFI estimation.
#### When Does a Spoken Agent Have Enough Evidence to Act? The PACT-SLM Contract Test
 - **Authors:** Mengzhe Geng
 - **Subjects:** Subjects:
Sound (cs.SD); Computation and Language (cs.CL); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2609.38232

 - **Pdf link:** https://arxiv.org/pdf/2609.38232

 - **Abstract**
 Streaming spoken agents may take an external action before the available speech supports it, yet final-turn scores do not reveal whether each observed prefix supports that action. We introduce the Partial Speech Action Contract for Turn Taking in Speech Language Models (PACT-SLM), a controlled evaluation that assigns a first valid action time and measures action identity and timing separately. The primary diagnostic contains 80 paired contrast groups from four held-out semantic families and 1,600 prefix predictions across clean and 15 dB noise renderings. After correcting a mismatch between randomized branch codes and semantic labels, a refitted WavLM Base Plus probe reaches 26.03% pooled post-onset semantic-label accuracy (95% group-bootstrap interval: 22.14%-29.68%), exposes an action on 18.99% of pre-onset prefixes, and predicts 5.94% of complete trajectories exactly. It exceeds matched text, scalar-acoustic, and shuffled-representation probes in post-onset label accuracy, but its score is at the 96th percentile of 100 within-prefix label permutations and below the 97.5th-percentile reference (26.73%). Elapsed time is more onset-exact than WavLM Base Plus (36.25% vs. 23.13%) but less accurate about action identity (9.92% vs. 26.03%). These results show that action identity and timing measure distinct aspects of partial-speech decision behavior.
#### Talk2Agent: Benchmarking Voice Interfaces for Text Agents
 - **Authors:** Terumi Chiba, Guangzhi Sun, Zheqi Yuan, Chao Zhang
 - **Subjects:** Subjects:
Artificial Intelligence (cs.AI); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2609.38867

 - **Pdf link:** https://arxiv.org/pdf/2609.38867

 - **Abstract**
 Large language model (LLM) computer-use agents are typically evaluated with clean written instructions, despite speech being an increasingly popular interface for interacting with such systems. Speech input introduces an additional failure point: transcription errors can alter task-critical entities, constraints, or targets before the agent begins reasoning, while conventional ASR metrics do not directly measure whether the information required for successful execution has been preserved. We introduce Talk2Agent, a benchmark for evaluating how effectively voice interfaces convey human-spoken instructions to LLM-based computer-use agents. Talk2Agent builds human-spoken versions of tasks from WildClawBench and OSWorld and evaluates a range of voice interfaces, including dedicated ASR models, audio-capable LLMs, contextual biasing, and LLM-based ontology repair. Because repeatedly executing long-horizon computer-use tasks is costly and stochastic, we further propose an execution-free, task-conditioned evaluation framework that projects the original task grader onto prompt-addressable intentions and measures how much task-relevant information is retained after the voice interface. On WildClawBench, Talk2Agent's execution-free native projection provides a practical, execution-grounded measure of voice-interface quality, correlating with downstream task completion and improving Pearson correlation by 0.246 over WER/CER on 32 hours of real human speech.
#### How Reliable Are Predicted MOS for Reproducing Human System-Level Preferences in Speech Enhancement?
 - **Authors:** Nahomi Kusunoki, Tsubasa Ochiai, Naohiro Tawara, Marc Delcroix, Naoyuki Kamo, Tetsuji Ogawa, Shoko Araki
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2609.39032

 - **Pdf link:** https://arxiv.org/pdf/2609.39032

 - **Abstract**
 We investigate whether predicted Mean Opinion Scores (MOS) can reliably support system-level comparisons of speech enhancement (SE) methods by introducing system-level preference accuracy (SPA). Although MOS prediction models are widely used to evaluate SE systems, their performance is typically assessed by correlation with human-rated MOS, which does not guarantee agreement on which system is better. SPA addresses this gap by directly evaluating whether predicted and human-rated MOS yield the same system preferences. Using SPA, we systematically evaluate three settings: single prediction models, ensembling, and domain adaptation. Through experiments, SPA varies substantially across single prediction models, from 9.4% to 76.8%. Even the best model disagrees with human judgments in approximately 23% of system comparisons. Ensembling yields only limited improvement, while domain adaptation tends to substantially improve SPA in the closed condition but brings only modest gains in the more practical open condition, where neither the target systems nor the speakers are known. These results suggest that SPA can reveal errors correlation-based evaluation alone does not expose, and that predicted MOS alone can lead to unreliable conclusions in practical SE system comparison.
#### From Speech to Editable Concepts: Probing Emotion Recognition with Concept Bottleneck Models
 - **Authors:** Hezhao Zhang, Thomas Hain
 - **Subjects:** Subjects:
Sound (cs.SD); Artificial Intelligence (cs.AI); Computation and Language (cs.CL); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2609.39453

 - **Pdf link:** https://arxiv.org/pdf/2609.39453

 - **Abstract**
 Speech emotion recognition (SER) is the task of assigning emotion labels to utterances. Early systems relied on acoustic features, whereas recent approaches combine multiple modalities, most commonly speech and text. Still, performance remains poor on many datasets. Large language models (LLMs) have therefore attracted interest for SER, as they can process diverse inputs jointly with instructions. However, direct audio input raises questions of explainability. To address similar questions in image classification, concept bottleneck models were introduced. This work adapts concept bottlenecks to SER to examine how individual predictions depend on transcripts, acoustic descriptions and speaker attributes. Experiments test three LLMs on CREMA-D, IEMOCAP and MELD, with concepts extracted by separate tools. On scripted corpora, LLMs are strongly biased towards the transcript in the zero-shot setting, which lowers Macro-F1 from 27.8 to 5.8 on CREMA-D. Fine-tuning removes this bias, and the transcript raises Macro-F1 from 41.8 to 45.1. Removing speech rate changes 48% of Neutral predictions to Disgust on CREMA-D; removing intensity level on MELD changes predictions despite little change in Macro-F1. These findings show that aggregate performance changes alone do not capture the effects of concept removal on individual predictions.
#### MeanVoiceFlow2: Joint Optimization of Mean Flow and Content Encoder for Fast One-Step Zero-Shot Voice Conversion
 - **Authors:** Takuhiro Kaneko, Hirokazu Kameoka, Kou Tanaka, Yuto Kondo
 - **Subjects:** Subjects:
Sound (cs.SD); Artificial Intelligence (cs.AI); Machine Learning (cs.LG); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2609.40087

 - **Pdf link:** https://arxiv.org/pdf/2609.40087

 - **Abstract**
 Flow-matching approaches to voice conversion (VC) have gained attention owing to their high speech quality and strong speaker similarity. Among them, one-step models such as MeanVoiceFlow are particularly attractive because they enable efficient inference; however, their reliance on a computationally intensive content encoder remains a bottleneck. We therefore propose MeanVoiceFlow2, a framework that jointly optimizes a flow-based conversion module and a computationally efficient content encoder. The model is trained through conversion distillation using MeanVoiceFlow and the reconstruction of real data. We further incorporate diffusion-GAN training with sample mixing and teacher-guided conditioning augmentation to enhance realism and disentanglement. Experiments on zero-shot VC showed that MeanVoiceFlow2 achieved higher perceptual quality and approximately $9\times$ faster inference than MeanVoiceFlow while maintaining comparable speaker similarity. Audio samples are available at this https URL.


by Zyzzyva0381 (Windy). 


2026-10-01
