# Showing new listings for Friday, 10 April 2026
Auto update papers at about 2:30am UTC (10:30am Beijing time) every weekday.


阅读 `Usage.md`了解如何使用此repo实现个性化的Arxiv论文推送

See `Usage.md` for instructions on how to personalize the repo. 


Keyword list: ['text-to-speech', 'text to speech', 'tts', 'LLM-based', 'speech', 'voice']


Excluded: []


### Today: 7papers 
#### Rethinking Entropy Allocation in LLM-based ASR: Understanding the Dynamics between Speech Encoders and LLMs
 - **Authors:** Yuan Xie, Jiaqi Song, Guang Qiu, Xianliang Wang, Ming Lei, Jie Gao, Jie Wu
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Computation and Language (cs.CL); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2604.08003

 - **Pdf link:** https://arxiv.org/pdf/2604.08003

 - **Abstract**
 Integrating large language models (LLMs) into automatic speech recognition (ASR) has become a dominant paradigm. Although recent LLM-based ASR models have shown promising performance on public benchmarks, it remains challenging to balance recognition quality with latency and overhead, while hallucinations further limit real-world deployment. In this study, we revisit LLM-based ASR from an entropy allocation perspective and introduce three metrics to characterize how training paradigms allocate entropy reduction between the speech encoder and the LLM. To remedy entropy-allocation inefficiencies in prevailing approaches, we propose a principled multi-stage training strategy grounded in capability-boundary awareness, optimizing parameter efficiency and hallucination robustness. Specifically, we redesign the pretraining strategy to alleviate the speech-text modality gap, and further introduce an iterative asynchronous SFT stage between alignment and joint SFT to preserve functional decoupling and constrain encoder representation drift. Experiments on Mandarin and English benchmarks show that our method achieves competitive performance with state-of-the-art models using only 2.3B parameters, while also effectively mitigating hallucinations through our decoupling-oriented design.
#### Tracking Listener Attention: Gaze-Guided Audio-Visual Speech Enhancement Framework
 - **Authors:** Hsiang-Cheng Yang, You-Jin Li, Rong Chao, Yu Tsao, Borching Su, Shao-Yi Chien
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2604.08359

 - **Pdf link:** https://arxiv.org/pdf/2604.08359

 - **Abstract**
 This paper presents a Gaze-Guided Audio-Visual Speech Enhancement (GG-AVSE) framework to address the cocktail party problem. A major challenge in conventional AVSE is identifying the listener's intended speaker in multi-talker environments. GG-AVSE addresses this issue by exploiting gaze direction as a supervisory cue for target-speaker selection. Specifically, we propose the GG-VM module, which combines gaze signals with a YOLO5Face detector to extract the target speaker's facial features and integrates them with the pretrained AVSEMamba model through two strategies: zero-shot merging and partial visual fine-tuning. For evaluation, we introduce the AVSEC2-Gaze dataset. Experimental results show that GG-AVSE achieves substantial performance gains over gaze-free baselines: a 10.08% improvement in PESQ (2.370 to 2.609), a 5.18% improvement in STOI (0.8802 to 0.9258), and a 23.69% improvement in SI-SDR (9.16 to 11.33). These results confirm that gaze provides an effective cue for resolving target-speaker ambiguity and highlight the scalability of GG-AVSE for real-world applications.
#### TASU2: Controllable CTC Simulation for Alignment and Low-Resource Adaptation of Speech LLMs
 - **Authors:** Jing Peng, Chenghao Wang, Yi Yang, Lirong Qian, Junjie Li, Yu Xi, Shuai Wang, Kai Yu
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Artificial Intelligence (cs.AI)
 - **Arxiv link:** https://arxiv.org/abs/2604.08384

 - **Pdf link:** https://arxiv.org/pdf/2604.08384

 - **Abstract**
 Speech LLM post-training increasingly relies on efficient cross-modal alignment and robust low-resource adaptation, yet collecting large-scale audio-text pairs remains costly. Text-only alignment methods such as TASU reduce this burden by simulating CTC posteriors from transcripts, but they provide limited control over uncertainty and error rate, making curriculum design largely heuristic. We propose \textbf{TASU2}, a controllable CTC simulation framework that simulates CTC posterior distributions under a specified WER range, producing text-derived supervision that better matches the acoustic decoding interface. This enables principled post-training curricula that smoothly vary supervision difficulty without TTS. Across multiple source-to-target adaptation settings, TASU2 improves in-domain and out-of-domain recognition over TASU, and consistently outperforms strong baselines including text-only fine-tuning and TTS-based augmentation, while mitigating source-domain performance degradation.
#### Ring Mixing with Auxiliary Signal-to-Consistency-Error Ratio Loss for Unsupervised Denoising in Speech Separation
 - **Authors:** Matthew Maciejewski, Samuele Cornell
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2604.08415

 - **Pdf link:** https://arxiv.org/pdf/2604.08415

 - **Abstract**
 Noisy speech separation systems are typically trained on fully-synthetic mixtures, limiting generalization to real-world scenarios. Though training on mixtures of in-domain (thus often noisy) speech is possible, we show that this leads to undesirable optima where mixture noise is retained in the estimates, due to the inseparability of the background noises and the loss function's symmetry. To address this, we propose ring mixing, a batch strategy of using each source in two mixtures, alongside a new Signal-to-Consistency-Error Ratio (SCER) auxiliary loss penalizing inconsistent estimates of the same source from different mixtures, breaking symmetry and incentivizing denoising. On a WHAM!-based benchmark, our method can reduce residual noise by upwards of half, effectively learning to denoise from only noisy recordings. This opens the door to training more generalizable systems using in-the-wild data, which we demonstrate via systems trained using naturally-noisy speech from VoxCeleb.
#### Semantic-Emotional Resonance Embedding: A Semi-Supervised Paradigm for Cross-Lingual Speech Emotion Recognition
 - **Authors:** Ya Zhao, Yinfeng Yu, Liejun Wang
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2604.07417

 - **Pdf link:** https://arxiv.org/pdf/2604.07417

 - **Abstract**
 Cross-lingual Speech Emotion Recognition (CLSER) aims to identify emotional states in unseen languages. However, existing methods heavily rely on the semantic synchrony of complete labels and static feature stability, hindering low-resource languages from reaching high-resource performance. To address this, we propose a semi-supervised framework based on Semantic-Emotional Resonance Embedding (SERE), a cross-lingual dynamic feature paradigm that requires neither target language labels nor translation alignment. Specifically, SERE constructs an emotion-semantic structure using a small number of labeled samples. It learns human emotional experiences through an Instantaneous Resonance Field (IRF), enabling unlabeled samples to self-organize into this structure. This achieves semi-supervised semantic guidance and structural discovery. Additionally, we design a Triple-Resonance Interaction Chain (TRIC) loss to enable the model to reinforce the interaction and embedding capabilities between labeled and unlabeled samples during emotional highlights. Extensive experiments across multiple languages demonstrate the effectiveness of our method, requiring only 5-shot labeling in the source language.
#### Selective Attention System (SAS): Device-Addressed Speech Detection for Real-Time On-Device Voice AI
 - **Authors:** David Joohun Kim, Daniyal Anjum, Bonny Banerjee, Omar Abbasi
 - **Subjects:** Subjects:
Sound (cs.SD); Artificial Intelligence (cs.AI); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2604.08412

 - **Pdf link:** https://arxiv.org/pdf/2604.08412

 - **Abstract**
 We study device-addressed speech detection under pre-ASR edge deployment constraints, where systems must decide whether to forward audio before transcription under strict latency and compute limits. We show that, in multi-speaker environments with temporally ambiguous utterances, this task is more effectively modelled as a sequential routing problem over interaction history than as an utterance-local classification task. We formalize this as Sequential Device-Addressed Routing (SDAR) and present the Selective Attention System (SAS), an on-device implementation that instantiates this formulation. On a held-out 60-hour multi-speaker English test set, the primary audio-only configuration achieves F1=0.86 (precision=0.89, recall=0.83); with an optional camera, audio+video fusion raises F1 to 0.95 (precision=0.97, recall=0.93). Removing causal interaction history (Stage~3) reduced F1 from 0.95 to 0.57+/-0.03 in the audio+video configuration under our evaluation protocol. Among the tested components, this was the largest observed ablation effect, indicating that short-horizon interaction history carries substantial decision-relevant information in the evaluated setting. SAS runs fully on-device on ARM Cortex-A class hardware (<150 ms latency, <20 MB footprint). All results are from internal evaluation on a proprietary dataset evaluated primarily in English; a 5-hour evaluation subset may be shared for independent verification (Section 8.8).
#### DeepFense: A Unified, Modular, and Extensible Framework for Robust Deepfake Audio Detection
 - **Authors:** Yassine El Kheir, Arnab Das, Yixuan Xiao, Xin Wang, Feidi Kallel, Enes Erdem Erdogan, Ngoc Thang Vu, Tim Polzehl, Sebastian Moeller
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2604.08450

 - **Pdf link:** https://arxiv.org/pdf/2604.08450

 - **Abstract**
 Speech deepfake detection is a well-established research field with different models, datasets, and training strategies. However, the lack of standardized implementations and evaluation protocols limits reproducibility, benchmarking, and comparison across studies. In this work, we present DeepFense, a comprehensive, open-source PyTorch toolkit integrating the latest architectures, loss functions, and augmentation pipelines, alongside over 100 recipes. Using DeepFense, we conducted a large-scale evaluation of more than 400 models. Our findings reveal that while carefully curated training data improves cross-domain generalization, the choice of pre-trained front-end feature extractor dominates overall performance variance. Crucially, we show severe biases in high-performing models regarding audio quality, speaker gender, and language. DeepFense is expected to facilitate real-world deployment with the necessary tools to address equitable training data selection and front-end fine-tuning.


by Zyzzyva0381 (Windy). 


2026-04-10
