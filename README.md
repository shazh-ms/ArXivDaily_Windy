# Showing new listings for Tuesday, 27 October 2026
Auto update papers at about 2:30am UTC (10:30am Beijing time) every weekday.


阅读 `Usage.md`了解如何使用此repo实现个性化的Arxiv论文推送

See `Usage.md` for instructions on how to personalize the repo. 


Keyword list: ['text-to-speech', 'text to speech', 'tts', 'LLM-based', 'speech', 'voice']


Excluded: []


### Today: 14papers 
#### Beyond Accuracy: What Are Speech Emotion Recognition Models Learning?
 - **Authors:** Prince Baah-Mensah, Dennis Owusu Asamoah
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2610.33952

 - **Pdf link:** https://arxiv.org/pdf/2610.33952

 - **Abstract**
 Speech Emotion Recognition (SER) is important in human-machine interactions, enabling machines to detect and classify human emotions from solely human speech. Speech emotion recognition (SER) benchmarks routinely report accuracies above 90 percent, yet these models fail when deployed or tested on diverse speakers. The papers overlook a fundamental issue: models may be learning speaker identity and tone rather than generalized emotional cues. We introduce a two-condition audit protocol that measures, for any published SER model, how much of its reported performance survives when speakers are held out. Each model is evaluated twice under identical architecture, features, and training recipe: a control condition that reproduces the common random-split practice, and a strict speaker-independent condition with disjoint speakers. We apply the protocol to seven model configurations from five published papers, spanning convolutional, recurrent, transformer, vision-transformer, and self-supervised wav2vec2 architectures, on RAVDESS, EMO-DB, and SAVEE, with a fourth corpus (EARS) as an external check. Across 21 model-dataset pairs, published headline numbers reproduce close to the control but drop by 6.9 to 63.1 percentage points under strict speaker independence. Our results quantify speaker memorization as a dominant, previously unmeasured component of published SER performance, and we recommend two-condition reporting as a community standard.
#### ADReasoner: Acoustic-Linguistic Cue-Aware Reinforcement Learning for Explainable Alzheimer's Disease Detection from Speech
 - **Authors:** Yin-Long Liu, Yuanchao Li, Jiahong Yuan, Zhen-Hua Ling
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2610.34220

 - **Pdf link:** https://arxiv.org/pdf/2610.34220

 - **Abstract**
 Speech-based Alzheimer's Disease (AD) detection holds significant promise due to its scalable and non-invasive nature. However, current Speech Large Language Models (SpeechLLMs), similar to conventional AD detection systems, still treat AD detection primarily as a simple categorical classification task. This paradigm provides limited interpretability of predictions, while leaving the inherent expressiveness and reasoning capabilities of SpeechLLMs largely underutilized. In this work, we take the first step toward reframing AD detection as a deep reasoning task via Reinforcement Learning (RL). We propose ADReasoner, an RL-enhanced SpeechLLM designed to generate accurate diagnostic predictions with interpretable rationales grounded in fine-grained acoustic-linguistic cues. To achieve this, we first systematically construct ADCoT-7K, an AD speech reasoning dataset combining real data and our quality-filtered synthetic data, where each instance features Chain-of-Thought (CoT) annotations and detailed captions. Second, observing the weak perception of AD-related acoustic-linguistic cues in current SpeechLLMs, we leverage Supervised Fine-Tuning (SFT) to develop a cue-perception-enhanced foundation model, ADReasoner-Base, thereby improving AD speech understanding. Third, alongside two standard rule-based rewards, we introduce a reasoning reward and a reasoning-label consistency reward into Group Relative Policy Optimization (GRPO) for RL. Furthermore, a shrinkage mechanism is employed to suppress misleading reasoning signals, while a gating mechanism progressively introduces both rewards to stabilize policy training. Experiments on three AD datasets (Pitt, PROCESS-1, PROCESS-2) demonstrate that ADReasoner outperforms previous state-of-the-art models in both detection performance and explanation quality, advancing AD detection toward interpretable reasoning from speech.
#### In-Context Adapter Generation for Compositional Generalization in Large Audio Language Models
 - **Authors:** Yiwei Guo, Siqi Huang, Bohan Li, Jing Peng, Hankun Wang, Zhihan Li, Kai Yu
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Artificial Intelligence (cs.AI); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2610.34413

 - **Pdf link:** https://arxiv.org/pdf/2610.34413

 - **Abstract**
 Large audio language models (LALMs) have achieved strong performance on standard speech and audio understanding tasks, yet their ability to systematically generalize to novel task compositions remains underexplored. Taking a meta-learning view, we extend Text-to-LoRA (T2L) to LALMs, generating instruction-conditioned LoRA adapters with a Transformer hypernetwork rather than learning a single fixed adapter shared across instructions. To exploit fine-grained task information, we introduce in-context adapter generation (ICAG), a token-level hypernetwork architecture; we call the resulting method T2L-ICAG. We evaluate this approach on two representative LALMs using SIFT-50M for supervised fine-tuning and construct a controlled compositional evaluation suite covering five task classes. Across two LALM backbones, T2L-ICAG consistently improves over static LoRA and also the global embedding-based T2L baseline. Coverage and capacity analyses further show that these gains are larger on structurally out-of-domain tasks and cannot be explained by parameter count alone. These results suggest that generating task-specific adapters from instructions is a promising direction for improving compositional generalization of LALMs.
#### TIMUSE: Text-Initialized Masked Semantic Token Modeling for Universal Speech Enhancement
 - **Authors:** Yiwei Guo, Hao Qiu, Mohan Ye, Zhijie Deng, Zehua Chen, Kai Yu
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2610.34814

 - **Pdf link:** https://arxiv.org/pdf/2610.34814

 - **Abstract**
 Discrete-token-based universal speech enhancement (USE) enables strong sequence modeling but often suffers from a quality--fidelity trade-off, where perceptually plausible outputs may alter linguistic content or speaker characteristics. We present TIMUSE, a two-stage USE framework combining text-initialized masked semantic restoration with acoustic reconstruction. A pretrained diffusion language model (DLM) is adapted with LoRA to recover clean semantic tokens using masked generative modeling, while a conditional flow matching model reconstructs clean 24\,kHz speech from the recovered semantic tokens and original degraded speech. Both system-level and controlled comparisons show that TIMUSE achieves a better balance between perceptual quality and fidelity than existing discrete USE models. Ablations further show that text DLM initialization brings a targeted gain under severe degradations.
#### Realizing Low-Precision Storage and Sparse Execution in a Folded FPGA Speech Enhancer
 - **Authors:** Clément Laroche
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Hardware Architecture (cs.AR); Machine Learning (cs.LG)
 - **Arxiv link:** https://arxiv.org/abs/2610.35144

 - **Pdf link:** https://arxiv.org/pdf/2610.35144

 - **Abstract**
 Hardware-aware training often optimizes FPGA resources, latency, or bit operations, but folding changes what compression saves: shared arithmetic decouples precision, weight storage, and scheduled work. We derive training objectives from the measured costs of a folded streaming speech-enhancement accelerator. Mixed-precision weights reduce on-chip storage; structured channel sparsity removes entire operations from the engine schedule. We deploy a bit-exact implementation on the Kria KV260. The best sparse model eliminates 47.3% of convolution MACs, cutting frame time by 45% (1.86 to 1.03 ms) and dynamic energy by 60% (83 to 33 $\mu$J/frame) against a storage-priced model measured in the same session, with higher speech quality. Relative to the most efficient dense build measured separately, frame time falls by 42% and energy by 48%. Reducing weight width from 8 to 6 bits lowers dynamic power by 27% but changes frame time by less than 5%. Plain channel group-lasso removes more work than cycle-cost-weighted lasso. Thus, folded-accelerator training should price storage and removable work separately, rather than rely on a single bit-operation metric.
#### Integrate the Reward, Don't Sample It: Window Likelihood for Interruptions and Backchannels in Full-Duplex Speech Models
 - **Authors:** Guoqiang Hu, Lichao Zhang, Yizhou Peng, Yuhao Luo, Haifeng Hu, Sallen Sun, Will Yang, Eng Siong Chng
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Computation and Language (cs.CL); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2610.35581

 - **Pdf link:** https://arxiv.org/pdf/2610.35581

 - **Abstract**
 A full-duplex spoken dialogue model must yield when the user cuts in and hold when the user merely backchannels. Sequence-level RL with a timing reward (GRPO) is the usual recipe; we test its whole-rollout form on four open models and find two failures. A binary window reward gives zero advantage on exactly the windows that most need learning (a zero-gradient trap), and the one score per rollout is spread over listening-state decisions the reward is not written in terms of (credit leakage), so the model talks into silence or stops answering. Both come from sampling a reward that is a known function of a few binary control decisions. We propose \emph{window likelihood} (\WL): sample a continuation per window, but integrate in closed form over the stop decisions it visits and maximise the log-probability of acting correctly---a noisy-OR for yielding, a survival product for holding. On HumDial-FDBench, \WL\ gives the best yield/hold F1 on all six structurally different open models we post-train, exceeds Kyutai's released GRPO checkpoints (trained on different data), and adds no re-opening of the floor on any model.
#### A Robust, Explainable, and Efficient Deep Metric for Speech and General Audio Quality
 - **Authors:** Konstantin Schmidt, Pablo M. Delgado, Andreas Brendel
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2610.35810

 - **Pdf link:** https://arxiv.org/pdf/2610.35810

 - **Abstract**
 We present a robust Speech and Audio Quality Metric (SAQM) that merges features of traditional quality metrics with a CNN trained on a large amount of data using contrastive learning methods. To overcome the problem of limited amounts of human-labeled training data, we propose two pretraining strategies relying on pseudo-labels. One uses an established objective metric PEAQ-2f (based on ITU-R BS.1387) as surrogate loss, the other is based on contrasting a wide range of coding modes by a rule-of-thumb ordering based on expected perceived quality. As a key novelty, our approach lets us generate saliency maps by propagating gradients to identify which feature has the greatest impact on the final metric outcome and enables our system to provide insights regarding the type and extent of distortions present in the tested audio signal. The resulting (SAQM) operates at low computational complexity, performs comparably to (SOTA) metrics on speech while still performing well on general audio content.
#### A Benchmark for Controllable Speaking-Style Captioning in Audio-Language Models
 - **Authors:** Shreeram Suresh Chandra, Ashwyn Sharma, Liang Lu, Xiutian Zhao, Leda Sari, Björn Schuller, Berrak Sisman
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Artificial Intelligence (cs.AI); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2610.33036

 - **Pdf link:** https://arxiv.org/pdf/2610.33036

 - **Abstract**
 Speaking-style captioning (SSC) aims to generate natural language descriptions of how speech is delivered, capturing paralinguistic attributes such as vocal timbre, prosody, and expressivity. Many downstream applications, including conversational AI agents, controllable speech generation, and large-scale audio annotation, require dimension-specific style captions that describe targeted aspects of speech (e.g., speaker traits, emotion, or delivery style), rather than a single undifferentiated description. However, existing work lacks a unified formulation for controllable speaking-style captioning. We address this gap by formulating SSC as an instruction-following audio-language task, where explicit instructions specify the speaking-style dimensions to be described. We introduce StyleInstructCaps, the first standardized benchmark for controllable speaking-style captioning in audio-language models. StyleInstructCaps provides a dataset and evaluation framework that measures metadata groundedness, hallucination, instruction-following ability, generalization to unseen instructions and audio datasets, and speaker-style consistency. Across nine state-of-the-art audio-language models, we find that strong instruction-following ability does not necessarily imply grounded speaking-style understanding, and that SSC-specific instruction-diverse training substantially improves groundedness, controllability, and speaker consistency.
#### Measuring Local Acoustic Contribution in Speech LLMs via Word-Span Counterfactual Interventions
 - **Authors:** Hardik B. Sailor, Qiongqiong Wang, Dheemant Rastogi, Nancy F. Chen, Ai Ti Aw
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2610.33077

 - **Pdf link:** https://arxiv.org/pdf/2610.33077

 - **Abstract**
 Correct transcription does not establish that a word depended on its local audio. We introduce the Acoustic Evidence Contribution (AEC), a word-level counterfactual measure of acoustic grounding. It is obtained from the change in reference-token log likelihood when the word's aligned acoustic span is replaced. We combine decoding outcomes on unedited audio with thresholded AEC in an outcome-contribution matrix, where correct targets are categorized according to their local acoustic contribution as high or low. Depending on the system, 33-49% of correct single-word targets fall below the twofold contribution threshold when averaged across corpora, despite no evident rank association with WER. Retention of reference words after span removal decreases monotonically across AEC quintiles, from 76-83% in the lowest quintile to approximately 1% in the highest. AEC therefore separates transcription correctness from dependence on the aligned local acoustic span while remaining an oracle-aligned diagnostic method.
#### Beyond Visible Speakers: Context-Aware Active Speaker Detection with Speaker Registration
 - **Authors:** Pu Wang, Hugo Van hamme
 - **Subjects:** Subjects:
Sound (cs.SD); Computer Vision and Pattern Recognition (cs.CV); Multimedia (cs.MM); Audio and Speech Processing (eess.AS); Image and Video Processing (eess.IV)
 - **Arxiv link:** https://arxiv.org/abs/2610.33127

 - **Pdf link:** https://arxiv.org/pdf/2610.33127

 - **Abstract**
 Active speaker detection (ASD) relies on audiovisual correspondence between speech and visible faces, yet this correspondence becomes unreliable in crowded scenes, or when visual evidence is unavailable. We introduce ROAM-ASD-C and ROAM-ASD-CR to extend ROAM-ASD with contextual and identity-aware reasoning. ROAM-ASD-C jointly reasons across simultaneously visible candidates, allowing other faces to provide complementary evidence when the target observation is ambiguous or incomplete. ROAM-ASD-CR incorporates registered-speaker representations derived from enrollment speech, allowing detection of whether a speaker is active even when that speaker is not visually observed. ROAM-ASD-C achieves state-of-the-art performance across five ASD benchmarks: 96.8% mAP on AVA, 98.9% on WASD, 88.8% on UniTalk, 98.7% on Talkies, and 99.5% on ASW. It also improves ROAM-ASD by up to 13.3 mAP under 75% target-track visual removal.
#### RoleHear: Pseudo-Labeling for Semi-Supervised Role-Attributed Streaming Speech Recognition
 - **Authors:** Zefang Liu, Chenyang Zhu, Sangwoo Cho, Xujun Peng, Rongqing Huang, Shi-Xiong Zhang, Sambit Sahu
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2610.33765

 - **Pdf link:** https://arxiv.org/pdf/2610.33765

 - **Abstract**
 Streaming automatic speech recognition transcribes at low latency but produces an unattributed word stream, while real-time applications such as customer-service analytics need each word tied to a speaker role. Role attribution is usually a separate stage, either an offline or streaming speaker-diarization model aligned to the words or a text-based large language model applied to the transcript, and role labels for training are scarce. We present RoleHear, a semi-supervised pipeline that makes a single cache-aware streaming transducer emit roles inline through lightweight in-vocabulary role tokens: a fast turn-boundary marker followed by a per-role token a few words later. An offline transducer teacher, fine-tuned into a role-attributed recognizer, relabels unlabeled audio, and a multimodal large language model supplies silver role labels where gold annotations are unavailable. The streaming student is fine-tuned on the mixture in a single pass. Across customer-service calls with gold roles and earnings calls with silver roles, RoleHear attributes roles accurately while preserving transcription quality and narrowing the gap to the offline teacher.
#### Heard Enough to Act: Early Tool Calling in Spoken Dialogue Systems
 - **Authors:** Yizhou Peng, Chen Chen, Donghang Wu, Hexin Liu, Eng Siong Chng
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2610.34653

 - **Pdf link:** https://arxiv.org/pdf/2610.34653

 - **Abstract**
 Latency-sensitive spoken agents must know not only which tool to call, but also when they have heard enough to act. Waiting for the turn to complete delays execution even when the request is already actionable. However, neither existing spoken agents nor spoken dialogue corpora define the earliest point at which a request becomes actionable. We introduce a synthetic spoken-dialogue pipeline that provides earliest-actionable-point supervision, marking when the observed speech grounds a tool and its required arguments. Through controlled task generation, speech synthesis and alignment, and gated annotation, we construct 26,694 dialogues totaling 786 hours across 48 domains, with disjoint training and test argument values. We additionally create an audio rendering of $\tau^2$-bench for external evaluation. We train an omni-modal language model on this corpus, enabling causal, chunk-level decisions and partial-transcript generation for downstream call completion. The model achieves tool-state macro-F1 scores of 0.48 and 0.47 on held-out dialogues and the external benchmark, respectively. For correct calls, the estimated mean lead over a turn-end cascade, using the same formatter, is 1.8-2.0 seconds in-domain and 1.1 seconds on $\tau^2$-bench. These results demonstrate a data-driven path toward spoken agents that act while listening. We will open-source the pipeline and the training scripts.
#### When the Persona Contradicts the Line: Diagnosing Plan-to-Synthesis Failures in Chain-of-Thought TTS
 - **Authors:** Joonyong Park, Jerry Li
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2610.35008

 - **Pdf link:** https://arxiv.org/pdf/2610.35008

 - **Abstract**
 Chain-of-thought TTS systems rely on language-model planners to translate persona and context into vocal delivery, but it is unclear where this intent is lost when the requested affect conflicts with lexical sentiment. We introduce a bilingual Japanese--English persona--line conflict benchmark in which each item pairs one line with congruent and conflicting personas and pre-declares the expected acoustic direction. We probe this direction in the planner-generated delivery phrase, the planner-written emotion vector, the QwenEmotion-generated vector, and the rendered waveform. Planner outputs are usually direction-correct, while downstream agreement can be substantially lower. Bypassing QwenEmotion provides only modest gains over the full planner-controlled Base. Supervising QwenEmotion raises the direction rate to 75% in Japanese and 84% in English. A learned text-to-latent interface reaches the highest English rate of 89% and the highest Japanese arousal rate of 88%. These results show that learned repair of the plan-to-synthesis interface can recover counter-textual delivery, although its effect differs across languages and acoustic axes.
#### Graph Neural Network for Wide-Area 6DoF Ambisonic Rendering
 - **Authors:** Gioele Greco, Filippo Marri, Mirco Pezzoli, Fabio Antonacci
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS); Signal Processing (eess.SP)
 - **Arxiv link:** https://arxiv.org/abs/2610.35741

 - **Pdf link:** https://arxiv.org/pdf/2610.35741

 - **Abstract**
 Wide-area 6DoF rendering from sparse Ambisonic recordings is commonly addressed by interpolating parametric spatial descriptors rather than the waveforms themselves. This preserves the fidelity of a nearby measured signal and avoids phase artifacts caused by mixing distant arrays. However, existing DirAC-based approaches rely on geometric interpolation weights that do not depend on the acoustic scene observed by the microphones. We propose a lightweight Graph Neural Network (GNN) that refines only this interpolation stage. The known Ambisonic arrays and the query position are represented as a graph whose nodes contain local DirAC features and whose edges encode relative distance and directional relations. The network predicts positive query-conditioned weights over the measured arrays, while the target energy, diffuseness, and direction of arrival are reconstructed by an analytical DirAC layer. Experiments on unseen rooms, unseen speech signals, and unseen array configurations show that the proposed weighting improves energy reconstruction and radiation-pattern similarity with respect to baseline interpolation.


by Zyzzyva0381 (Windy). 


2026-10-27
