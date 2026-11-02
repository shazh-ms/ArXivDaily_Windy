# Showing new listings for Monday, 2 November 2026
Auto update papers at about 2:30am UTC (10:30am Beijing time) every weekday.


阅读 `Usage.md`了解如何使用此repo实现个性化的Arxiv论文推送

See `Usage.md` for instructions on how to personalize the repo. 


Keyword list: ['text-to-speech', 'text to speech', 'tts', 'LLM-based', 'speech', 'voice']


Excluded: []


### Today: 7papers 
#### VoxLock: Target-Speaker Extraction for Voice Agents Without an Enrollment Step
 - **Authors:** Arbaz Siddiqui
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2610.40727

 - **Pdf link:** https://arxiv.org/pdf/2610.40727

 - **Abstract**
 A voice agent's speech recognizer transcribes everyone near the microphone, including people other than the caller. I fine-tuned a public causal target-speaker extraction model (REAL-TSE, 27M parameters) on one GPU to work from a voice sample cut from the call itself, and released it as VoxLock. I score it by what a recognizer writes on real recordings. On Krisp's 265-recording benchmark it reaches 4.95% WER against 23.21% unprocessed and lets 9 background words into the transcript over 85 minutes of background-only speech. Its WER is not significantly different from Krisp's best filters and is significantly lower than the other commercial and open systems I tested. On 400 AMI meeting windows built and frozen before scoring, it cuts WER from 42.95% to 10.63%. The 8-bit model runs a live call on 77% of one CPU thread with 24 ms of algorithmic latency. Model, code and per-file scores are open.
#### JYVA at BeTraC: Audio-Based Medical Note Generation with SOAP Section-Specific Queries
 - **Authors:** Youness Dkhissi (LIUM), Valentin Vielzeuf, Antoine Caubrière, Jean-Emile Dartois (IRT b-com)
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2610.40695

 - **Pdf link:** https://arxiv.org/pdf/2610.40695

 - **Abstract**
 We present our submission to the Beyond Transcription Challenge (BeTraC) on medical note generation from doctor-patient audio conversations. We propose a section-aware approach based on Qwen2.5-Omni-3B, in which a lightweight Q-Former learns SOAP section-specific queries to extract compact clinical representations from speech. In the lightweight track, our system ranks fourth and substantially outperforms the challenge baseline, achieving a Concept F1 of 0.5005 on the official test set. The organizers' analysis and our heuristic evaluation suggest improved robustness to hallucinations, and better stability under difficult audio conditions. These results show that explicitly modeling SOAP structure through section-specific queries is an effective way to improve the faithfulness and reliability of audio-based medical note generation.
#### Apriel Voice 27B Preview: An Open Model for Speech, Reasoning, and Tool Use
 - **Authors:** Aman Tiwari, Akshay Kalkunte, Akintunde Oladipo, Jash Shah, Tara Bogavelli, Hoang Nguyen, Nishanth Madhusudhan, Vishal Juneja, Sridhar Nemala, Srinivas Sunkara, Sathwik Tejaswi Madhusudhan, Shruthan Radhakrishna, Masoud Hashemi, Kelechi Ogueji, Khalil Slimi, Saloni Mittal, Pulkit Pattnaik, Shashank Maiya, Dhruv Jhamb, Esakkivel Esakkiraja, Katrina Stankiewicz, Gabrielle Gauthier Melançon, Fanny Riols, Amit Saha, Lakshmi Sirisha Chodisetty, Nitin Aggarwal, Anil Madamala, Sagar Davasam, Segan Subramanian, Vikas Yadav, Sai Rajeswar Mudumba
 - **Subjects:** Subjects:
Sound (cs.SD); Computation and Language (cs.CL); Machine Learning (cs.LG); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2610.40738

 - **Pdf link:** https://arxiv.org/pdf/2610.40738

 - **Abstract**
 A capable voice assistant must do more than recognize words; it must understand requests, reason over context, sustain conversations, and act through tools. We present Apriel Voice 27B Preview, an open speech-language model that unifies these capabilities within a shared reasoning backbone while retaining text and image input modalities. Built on the post-trained Gemma 4 26B-A4B checkpoint, Apriel Voice directly processes speech and generates both responses and tool calls without requiring a separate dialogue model. We introduce a two-stage training procedure to adapt the model to speech. The first stage establishes the speech interface, while the second jointly adapts the projector and backbone through supervised fine-tuning, text replay, and source-dependent cross-entropy and forward Kullback--Leibler divergence supervision. Automatically constructed speech-conditioned dialogues provide conversational training data, while local teacher serving, overlapping teacher and student computation, and fused full-vocabulary losses enable efficient online distillation. In thinking mode, Apriel Voice achieves a score of 98.90 on Big Bench Audio, 81.13 on Speech-BFCL, and a task success rate of 65.51% on $\tau$-Voice, with a mean latency of 1.91 seconds. It also achieves 51.27% English EVA accuracy and 74.13% experience pass@1. Finally, we deploy Apriel Voice 27B Preview on a custom serving platform to support interactive voice interactions.
#### Incongruence-Aware Evaluation of Multimodal Speech Emotion Recognition in Audio Large Language Models
 - **Authors:** Fengji Jing, Guy Laban
 - **Subjects:** Subjects:
Artificial Intelligence (cs.AI); Computation and Language (cs.CL); Multimedia (cs.MM); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2610.40883

 - **Pdf link:** https://arxiv.org/pdf/2610.40883

 - **Abstract**
 Voice-interactive large language models process both what a speaker says and how they say it, yet the two channels do not always agree. We ask whether that disagreement degrades speech emotion recognition. Mapping text and audio into a shared valence-arousal-dominance space, we convert the distance between them into a continuous, threshold-free incongruence score with a two-component Gaussian mixture, and validate the score against human judgement. We evaluate three audiocapable models on acted, conversational and spontaneous speech (eNTERFACE'05, MELD and CMU-MOSEI). With the model held constant across modalities, incongruence raises the odds of error whenever audio is present, in every corpus, after controlling for the ground-truth emotion and for distance in the feature space. Supervised classifiers trained on the same features behave in the opposite way, which suggests that the difficulty lies in how these models combine raw evidence rather than in the signal itself. Text-only effects are unreliable: swapping only the model that reads the transcript reverses their sign on one corpus. Rebuilding the score without the emotion lexicon reproduces the central result, and among six simpler alternatives it is the only label-free score that predicts error in all three corpora. Code, data tables and prompts are released.
#### From Transcription to Reformulation: Defining and Benchmarking Grounded Generative Speech Recognition
 - **Authors:** Qing Wang, Qihan Hu, Hang Lv, Yan Yang, Boqing Liu, Xuechao Zhu, Zehan Li, Hongjie Chen, Jian Kang, Jie Li, Yongxiang Li, Lei Xie
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2610.41292

 - **Pdf link:** https://arxiv.org/pdf/2610.41292

 - **Abstract**
 Speech language models have expanded automatic speech recognition (ASR) from transcription to general speech-to-text generation. Yet existing benchmarks still assume a single ground-truth transcript per utterance, an assumption that struggles when models produce readable, formatted, or structured outputs. In this work, we introduce Grounded Generative Speech Recognition (GSR), reformulating speech recognition as generating semantically grounded text instead of a single transcript. We define a grounded reformulation space that characterizes all valid textual realizations supported by the input speech, and establish a four-level hierarchy of reformulation capabilities, including literal transcription, canonical reformulation, semantic reformulation and speech-grounded structuring. Based on this formulation, we present GSR-Bench, the first comprehensive benchmark designed to evaluate generative speech recognition systems beyond exact transcript matching. GSR-Bench covers diverse scenarios including refined meeting transcription, cross-dialect translation, context-aware speech understanding, and robustness under silent or noisy acoustic conditions. To address the limitations of word-level evaluation metrics, we develop a multi-dimensional evaluation framework combining speech grounding assessment and LLM-based semantic judgment, whose reliability is validated through human preference studies. Extensive experiments across representative architectures, ranging from conventional ASR systems to end-to-end Speech-LLMs, reveal fundamental trade-offs between transcription fidelity and generative reformulation ability. Our analysis further identifies critical failure modes, including acoustic hallucination, unsupported generation, and instruction drift, providing insights into the design and evaluation of future grounded generative speech systems.
#### From prediction error to sparse events: Biologically inspired signed-dyadic coding for Mandarin keyword spotting
 - **Authors:** Zijian Zhou, Rui Liu, Rongxiang Nie, Rui Ma, Ziyao Zhang, Xin Du
 - **Subjects:** Subjects:
Sound (cs.SD); Artificial Intelligence (cs.AI); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2610.41381

 - **Pdf link:** https://arxiv.org/pdf/2610.41381

 - **Abstract**
 In continuous spiking keyword spotting, temporally correlated acoustic frames can repeatedly trigger memory access and neural computation at a dense input interface, even when hidden activity is sparse. Predictive processing and sensory adaptation suggest retaining predictable input locally and responding selectively to deviations. Inspired by this mechanism, we propose Predictive Signed-Dyadic Coding (PG). PG maintains a local prediction for each input channel and transmits an event only when the residual exceeds a learned threshold. Each event carries a sign and a finite power-of-two magnitude, and the receiver reconstructs it with fixed-point shifts and additions. In Mandarin keyword spotting, INPUT-PG reaches 90.00% accuracy while reducing mean SRAM reads by 33.87% and inference cycles by 32.77%. PG provides a hardware interface for research on low-power continuous speech processing and event-based sensory systems.
#### Scalable Audio-LLM System for Speech MOS Evaluation: A Diagnostic Case Study and Adaptation Roadmap
 - **Authors:** Dimitrios Kokkinis, Panagiotis Kakoulidis, Nikolaos Ellinas, Chrysovalanti Lymperi, Alexandra Vioni, Georgios Vardaxoglou, Iakovi Alexiou, Myrsini Christidou, Jeasung Lee, Gunu Jho, Inchul Hwang, Pirros Tsiakoulis, Aimilios Chalamandaris
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2610.41539

 - **Pdf link:** https://arxiv.org/pdf/2610.41539

 - **Abstract**
 The continuous deployment of enterprise voice systems creates a critical operational bottleneck: evaluating the acoustic quality of synthetic speech across high-volume production updates. While human Mean Opinion Score (MOS) remains the quality gold standard, it introduces severe operational costs and turn-around delays that make continuous integration infeasible. In this work, we present a system design case study for a scalable, low-cost and offline evaluation pipeline capable of scoring large batches of synthetic speech at marginal cost. First, we use Qwen2-Audio-7B-Instruct in a zero-shot setting for granular MOS estimation, noticing perceptual blind spots and inherent biases. Then, we analyze SOMOS dataset and construct a diagnostic benchmark that operationalizes two prosodic dimensions associated with human naturalness ratings. Finally, we adapt Qwen2-Audio-7B-Instruct through a parameter-efficient finetuning (LoRA) on the task of MOS evaluation, observing improvements over base model and widely used neural baselines like UTMOS and establishing a practical blueprint for enterprise-grade speech quality assurance.


by Zyzzyva0381 (Windy). 


2026-11-02
