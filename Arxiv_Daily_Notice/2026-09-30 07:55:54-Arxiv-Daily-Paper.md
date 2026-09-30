# Showing new listings for Wednesday, 30 September 2026
Auto update papers at about 2:30am UTC (10:30am Beijing time) every weekday.


阅读 `Usage.md`了解如何使用此repo实现个性化的Arxiv论文推送

See `Usage.md` for instructions on how to personalize the repo. 


Keyword list: ['text-to-speech', 'text to speech', 'tts', 'LLM-based', 'speech', 'voice']


Excluded: []


### Today: 13papers 
#### Model-Guided Design of Low-Context Speech Probes for Cochlear Synaptopathy
 - **Authors:** Ahsan J. Cheema, David Meng, Jorge Mejia, Sanna Hou, Sunil Puria
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2609.36272

 - **Pdf link:** https://arxiv.org/pdf/2609.36272

 - **Abstract**
 Cochlear neural degeneration (CND) can impair suprathreshold coding without elevating pure-tone thresholds, complicating its diagnosis when it coexists with hair cell loss. We present a unified comparison of temporal and noise-based probes for CND detection using low-context vowel-consonant-vowel (VCV) syllables to reduce linguistic and contextual cues. Using a phenomenological auditory nerve model, we simulated responses to 21 VCV tokens under time compression, reverberation, and speech-in-noise conditions across presentation levels and seven CND profiles. We computed mutual information (MI) between inner hair cell potentials and auditory nerve neurograms and quantified information loss relative to a normal-hearing baseline. Time compression and amplitude-modulated (AM) noise produced the largest modeled information losses. We then evaluated these stimuli in a consonant-identification study involving 36 listeners with normal audiograms, 12 of whom reported difficulty understanding speech in noise. Neither 40 percent time compression in quiet nor AM noise alone distinguished listeners with and without these difficulties. However, compressed speech presented in AM noise separated the two groups. This partial agreement between model predictions and behavior supports our MI-based stimulus design framework and motivates further evaluation of combined temporal and noise-based probes for CND detection.
#### InstCharVoice: Grounding Natural-Language Instructions for Character-Level Control in Text-to-Speech
 - **Authors:** Sihang Nie, Xueru Li, Xiaofen Xing, Deyi Tuo, Cheng-Bin Jin, Jingyuan Xing, Jinxin Ji
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2609.36287

 - **Pdf link:** https://arxiv.org/pdf/2609.36287

 - **Abstract**
 Instruction-based text-to-speech (ITTS) systems enable natural-language control of expressive speech generation, but often offer limited transparency and fine-grained control over individual text units. Character-level controllable TTS systems provide explicit acoustic control, yet typically rely on user-specified acoustic attributes. To bridge this gap, we propose InstCharVoice, a unified framework that grounds natural-language instructions in character-level acoustic control. We first construct grounded instruction annotations on the WordVoice-5A-zh corpus using Qwen3-Omni. With this supervision, we train an autoregressive model to identify instruction-relevant characters and predict their acoustic attributes before generating the corresponding speech tokens. Keyword prediction and grounding-aware loss weighting help the model focus on instruction-relevant characters and attributes. Experiments show improved instruction following and keyword-level acoustic control over representative ITTS systems, with competitive speech naturalness and explicit character-level controllability. Audio samples are available at this https URL.
#### Does a prosody-trained representation help beyond trainable fusion? A parameter-matched study with frozen HuBERT
 - **Authors:** Ki Woong Moon, Daniel Brenner
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Computation and Language (cs.CL)
 - **Arxiv link:** https://arxiv.org/abs/2609.36754

 - **Pdf link:** https://arxiv.org/pdf/2609.36754

 - **Abstract**
 Explicit prosodic cues may help automatic speech recognition (ASR) of spontaneous speech, but auxiliary representations typically require additional trainable components, making it unclear whether gains come from the auxiliary information or the fusion mechanism. We address this using a frozen HuBERT backbone and a 64-dimensional representation trained to predict log F0, voicing, Delta log F0, log energy, and spectral tilt. We compare a frozen-backbone recognizer (Baseline), trainable fusion with zero auxiliary input (Null), and the same fusion supplied with the learned representation (Learned). Across Buckeye, Switchboard, and AMI IHM, Null reduces WER by 0.71-1.45 points over Baseline, whereas Learned differs from Null by +0.07, -0.09, and +0.00 points, with no significant differences. However, removing or mismatching the representation at inference increases Learned WER. Thus, Learned depends on the representation yet shows no measurable incremental WER benefit over the parameter-matched control.
#### WenetSpeech-Min: A Large-Scale Minnan Speech Corpus with Dual Transcriptions for Dialectal Speech Processing
 - **Authors:** Haoyu Zhang, Chunjiang He, Hongtao Li, Zeyu Zhu, Qituan Shangguan, Chengyou Wang, Jingbin Hu, Ziyu Zhang, Bingshen Mu, Yanbo Wang, Shuai Wang, Jinhui Ye, Chengdong Liang, Binbin Zhang, Pengcheng Zhu, Chuang Ding, Qianze Feng, Qingyang Hong, Liumeng Xue, Lei Xie
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2609.36834

 - **Pdf link:** https://arxiv.org/pdf/2609.36834

 - **Abstract**
 Progress in dialectal speech technology is hindered by the scarcity of large-scale, real-world corpora. For Minnan speech, existing resources remain limited, and few provide paired Minnan and Mandarin transcripts at scale. To address these gaps, we introduce WenetSpeech-Min, an open-source corpus comprising around 10,000 hours of Minnan speech collected from diverse online media, with paired Minnan and Mandarin transcripts for every utterance. We further establish an automatic speech recognition (ASR) benchmark covering both Minnan and Mandarin transcripts and a text-to-speech synthesis (TTS) benchmark using Minnan transcripts, with manually verified evaluation sets for both tasks. To assess the effectiveness of the corpus, we train ASR and TTS models on WenetSpeech-Min and compare them with representative systems on the proposed benchmarks. The resulting models outperform the evaluated open-source models on most metrics and achieve competitive performance against commercial systems. We will release the corpus, benchmarks, and models to facilitate reproducible research on Minnan speech technology.
#### Louder, Longer, Livelier: Acoustic Shortcuts and Underspecified Rationales in Speech LLM Judges
 - **Authors:** Mingyue Huo, Shivam Mehta, Bhavin Jawade, Yinghong Lan, Haoqi Li
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2609.36979

 - **Pdf link:** https://arxiv.org/pdf/2609.36979

 - **Abstract**
 LLM-as-a-judge is widely used for evaluating text, but extending this paradigm to speech requires models to interpret acoustic as well as linguistic evidence. This introduces a modality-specific risk: a speech judge may treat a perceptually salient cue as evidence of quality even when that cue is irrelevant to the target criterion or receives more weight than human listeners give it. We call this behavior an acoustic shortcut. To study it, we audit six speech LLM judges using controlled manipulations of intensity, content richness, and emotional delivery. We evaluate both pointwise scoring and pairwise comparison, using human preference calibration to interpret the results. The judges consistently reward louder audio, prefer content-rich speech more strongly than human listeners do, and map emotional delivery into quality preferences. These effects are most visible in pairwise comparison, while pointwise scores often obscure them. More concerningly, the accompanying rationales rarely identify the acoustic cue that changes a judgment and instead repeatedly rely on a limited vocabulary, leaving them acoustically underspecified. Together, these findings show that reliable speech judges must both resist acoustic shortcuts and ground their rationales in the acoustic evidence behind their decisions. To support reproducibility and future audits, we also release SpeechJudgeAudit, the controlled stimuli and evaluation tools used in this study.
#### SENSE: Semantic Neural Speech Synthesis from Brain Dynamics via Spatial Graph Encoding
 - **Authors:** Jisoo Park, Seonghak Lee, Hyojin Park, Junseok Kwon
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2609.37601

 - **Pdf link:** https://arxiv.org/pdf/2609.37601

 - **Abstract**
 Reconstructing speech from non-invasive brain signals offers a promising pathway for restoring communication in individuals who are cognitively intact but unable to speak. Existing EEG-to-speech approaches formulate this task as acoustic reconstruction, optimizing waveform fidelity while ignoring whether the generated speech preserves high-level semantic content. In this work, we revisit this formulation and argue that EEG signals carry not only acoustic but also semantic information. We identify two key limitations of prior methods: (1) the neglect of spatial relationships between EEG electrodes, and (2) the failure to exploit the semantic structure of the N400 paradigm, where congruent and incongruent trials reflect distinct semantic processing. We propose SENSE(Semantic-EEG Neural Speech SynthEsis), which combines a graph-based EEG encoder over electrode geometry with EEG Semantic Conditioning (ESC), aligning EEG to a pretrained semantic space using only congruent trials. On the N400 dataset, SENSE consistently outperforms prior methods on both acoustic and semantic metrics, and model-internal channel attribution suggests distributed reliance on auditory, sensorimotor, and centro-parietal regions, consistent with known speech-perception neuroscience. In the unseen-subject setting, SENSE trained on only two subjects already surpasses the strongest baseline trained on all eighteen subjects in word error rate, and matches it on acoustic metrics with as few as eight subjects.
#### Selective Lookahead for Attention-Based Streaming ASR
 - **Authors:** Yichen Jia, Bastiaan Tamm, Hugo Van hamme
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2609.37611

 - **Pdf link:** https://arxiv.org/pdf/2609.37611

 - **Abstract**
 End-to-end attention-based speech recognition is accurate offline but hard to stream: outputs can depend on future audio, and a little future context per layer makes the lookahead grow with the number of layers. We address this with two mechanisms. A bounded-lookahead chunk encoder caps every chunk's future receptive field at a constant number of chunks, independent of the number of layers, via one age-selection rule shared by self-attention and the depthwise convolution. On this encoder, dynamic future-chunk decoding lets a per-token trigger commit a token or wait and re-decode it; we propose a learned trigger as the general mechanism, with a simple confidence threshold as an effective fallback. On full LibriSpeech test-clean the dynamic system matches the best static-lookahead accuracy (6.5%) at a median latency of 306 ms versus 860 ms for one-chunk static lookahead, and a wait budget bounds the deferral tail below the static baseline's 90th percentile at 0.1 points more WER.
#### Signal-Independent and Signal-Dependent Neural Ambisonic Matrix Encoding for Arbitrary Arrays with Variable Microphone Counts
 - **Authors:** Shichao Hu, Zhiheng Jin, Chunyang Xu, Mengyao Zhu
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2609.37691

 - **Pdf link:** https://arxiv.org/pdf/2609.37691

 - **Abstract**
 Recent neural Ambisonic encoders accommodate diverse array geometries, yet many existing neural encoders require a fixed microphone count because the number of microphone channels is embedded in the network architecture. This requirement limits deployment across devices with different microphone configurations and adaptation to changes in available channels. To address this limitation, we investigate Transformer-based matrix encoding for arbitrary microphone arrays with variable microphone counts. This is achieved through shared microphone-wise processing and masked self-attention that models inter-microphone relationships across variable-size arrays. Within this framework, we consider signal-independent (SI) encoding, which predicts encoding matrices from array transfer functions, and introduce a signal-dependent (SD) extension that additionally incorporates the observed microphone signals. Both models are trained on simulated scenes using LibriSpeech sources and extensively evaluated under changes in source type, unseen microphone counts, and increased source counts beyond those used during training. Both SI and SD outperform conventional least-squares (LS) encoding in aggregate reconstruction performance across the evaluated conditions. SD consistently achieves stronger overall performance than SI. These results demonstrate that the proposed framework enables array-agnostic Ambisonic encoding while retaining generalization across microphone counts and acoustic source conditions.
#### GLaS-JEPA: Gaussian-Regularized Speech SSL without Engineered Prediction Targets
 - **Authors:** Gaspard Botté, Séverin Baroudi, Samir Sadok, Francesco Paissan, Thomas Hueber, Xavier Alameda-Pineda, Ricard Marxer, Mirco Ravanelli
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Artificial Intelligence (cs.AI); Machine Learning (cs.LG); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2609.37798

 - **Pdf link:** https://arxiv.org/pdf/2609.37798

 - **Abstract**
 Speech self-supervised learning aims to learn general-purpose representations for downstream speech tasks. However, current approaches rely on complex, carefully designed prediction targets. We challenge this necessity with GLaS-JEPA, a framework that directly predicts the current encoder's continuous representations at masked positions, without contrastive learning, discrete targets, or separate EMA target encoders. We prevent representation collapse using SIGReg representation-space regularization, eliminating the need for engineered target-generation mechanisms. Pretrained on 960 hours of LibriSpeech, our 57M-parameter model achieves a 6.89% WER on frozen-encoder SUPERB ASR and a 25.87% CER on slot filling, outperforming the best non-distilled sub-90M baselines by 43.1% and 22.0%, respectively. These results demonstrate that highly competitive speech representations can emerge from a radically simplified training recipe.
#### FD-VAD: Semantic Endpoint Detection for Streaming Full-Duplex Speech
 - **Authors:** Puneet Mathur, Dinesh Manocha
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2609.35791

 - **Pdf link:** https://arxiv.org/pdf/2609.35791

 - **Abstract**
 Natural turn-taking in full-duplex voice interaction requires determining from partial speech whether a pause reflects hesitation or a completed conversational intent. Acoustic voice activity detection lacks this semantic information, while cascaded ASR-based endpointing introduces transcription dependence and additional processing stages. We formulate semantic endpoint detection as a causal audio-language reasoning task and introduce FD-VAD, an ASR-free streaming endpointer that maps bounded causal audio windows directly to Continue/Stop decisions. FD-VAD combines a frozen speech encoder with a lightweight modality adapter and a parameter-efficiently adapted language model, using a last-chunk training objective for streaming inference. We further introduce confidence-gated endpoint commitment to control interruption versus delay and boundary-focused hard-negative sampling to improve decisions around ambiguous turn boundaries. Across in-domain and conversational evaluations, FD-VAD outperforms strong streaming and non-streaming semantic turn classifiers, and achieves the highest EOT recall among qualifying systems on TurnBench dev set $0.853$ (at FP<=0.10) in a zero-shot setting. These results show that semantic endpointing can be performed directly from streaming audio without intermediate ASR or dialogue state tracking.
#### $τ$-Multilingual: Benchmarking Voice Agents Across Languages
 - **Authors:** Soham Ray, Edgard dos Santos Paiva, Ruben Valenzuela, Karthik Narasimhan, Keshav Dhandhania, Victor Barres
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Artificial Intelligence (cs.AI); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2609.35820

 - **Pdf link:** https://arxiv.org/pdf/2609.35820

 - **Abstract**
 English-only benchmarks expose only a narrow slice of voice-agent behavior. We introduce $\tau$-Multilingual, extending $\tau$-Voice to Spanish, Brazilian Portuguese, Hindi, Korean, and Mandarin with native-speaker review and evaluation of generated language and spoken output. Across 4,500 full-duplex calls and five voice configurations, Spanish, Portuguese, and Hindi remain within 3.2 task-completion points of English, but Korean and Mandarin fall by 14.7 and 8.4 points. The failure modes also vary: Korean systems miss more responses, Mandarin systems interrupt more often, and both struggle with tools and entities. Grok leads task completion but scores lowest on generation quality, motivating separate task, interaction, and generation reporting. We release language packs, validated judges, and tools for community-built multilingual voice-agent evaluation.
#### Reconstructing the Vocal Tract with Differentiable Acoustic Simulation
 - **Authors:** Eric Ming Chen, Jin Woo Lee, Vincent Sitzmann
 - **Subjects:** Subjects:
Sound (cs.SD); Computation and Language (cs.CL); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2609.36737

 - **Pdf link:** https://arxiv.org/pdf/2609.36737

 - **Abstract**
 The vocal tract is the region of the human body responsible for filtering one's voice to create speech. In this paper, we present a differentiable and GPU accelerated acoustic simulator for the vocal tract. The differentiable simulator synthesizes speech by propagating sound along an acoustic tube model of the vocal tract, and via its gradients, can solve the inverse problem: reconstructing the shape of the vocal tract solely from the sound it produces. Although the inverse mapping between geometry and sound is notoriously non-convex, we discover that gradient descent succeeds with three technical contributions: (1) we design a frequency domain formulation of the vocal tract's fluid dynamics that is 70x more GPU parallelizable than finite differences in time, (2) we integrate a differentiable model for turbulence to synthesize consonants, and (3) similar to prior work in implicit neural representations (INRs) and neural fields, we find that parameterizing the geometry with a neural network accelerates convergence and escapes local minima that trap discrete representations. Because the simulator is differentiable, it is readily integrated with other deep learning pipelines to enable novel linguistics and medical imaging applications. (1) We demonstrate self-supervised autoencoding of vocal tract shapes across 11 languages, and (2) we couple our simulator with a generative model of MRI (magnetic resonance imaging) images to reconstruct one's moving vocal tract from only their speech without paired data.
#### EmoRES-TTS: Residual-Enhanced Vector Steering for Emotional Speech Generation
 - **Authors:** Kuan-Po Huang, Haohe Liu, Puyuan Peng, Haibin Wu, Zhaoheng Ni, Hung-yi Lee, Jinwon Lee, Neha Chachra
 - **Subjects:** Subjects:
Sound (cs.SD); Computation and Language (cs.CL); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2609.38157

 - **Pdf link:** https://arxiv.org/pdf/2609.38157

 - **Abstract**
 Emotion-conditioned text-to-speech (TTS) models may fail to express the requested emotion reliably, and improving controllability by additional training is costly in both computation and emotion-labeled speech training data. We therefore study vector steering, a training-free approach that modifies the internal representations of a frozen model. CoCoEmo, a conventional vector steering method for emotion TTS, treats each emotion vector as an indivisible direction controlled by a single global strength, limiting adherence to the requested emotion. In this work, we first discover that an emotion vector can be decomposed into a shared component that moves speech away from neutral expression and a residual component that directs generation toward the requested emotion. Building on this finding, we propose Emotion Residual-Enhanced Steering for TTS (EmoRES), a novel method that controls the two components without retraining the backbone. On IEMOCAP, EmoRES outperforms CoCoEmo across all four objective emotion metrics on the IndexTTS-2 and CosyVoice2 backbones. Rank correlation improves by 26.13 and 12.97 percentage points, corresponding to relative gains of 118.8% and 33.1%, while emotion hit rate improves by 12.95 and 6.92 points, corresponding to relative gains of 20.1% and 9.8%. Human evaluation further shows a relative improvement up to 35.0% in the rate at which listeners correctly identified the dominant requested emotion and up to a 17.3% improvement in fidelity, while listeners prefer EmoRES for naturalness in up to 63.8% of pairwise comparisons. Component ablations further demonstrate that effective control benefits from preserving the shared component while strengthening the residual of the emotion steering vectors.


by Zyzzyva0381 (Windy). 


2026-09-30
