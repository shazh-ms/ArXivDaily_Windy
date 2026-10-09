# Showing new listings for Friday, 9 October 2026
Auto update papers at about 2:30am UTC (10:30am Beijing time) every weekday.


阅读 `Usage.md`了解如何使用此repo实现个性化的Arxiv论文推送

See `Usage.md` for instructions on how to personalize the repo. 


Keyword list: ['text-to-speech', 'text to speech', 'tts', 'LLM-based', 'speech', 'voice']


Excluded: []


### Today: 9papers 
#### Conversational Voice Aesthetic Model with Reinforcement Learning from Human Listeners
 - **Authors:** Xilin Jiang, Shun Zhang, Tejas Jayashankar, Yinghao Aaron Li, Osama Hanna
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Computation and Language (cs.CL); Machine Learning (cs.LG); Multimedia (cs.MM); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2610.10868

 - **Pdf link:** https://arxiv.org/pdf/2610.10868

 - **Abstract**
 We introduce Conversational Voice Aesthetic Model, a speech large language model for describing the voice aesthetics of real or synthetic speech responses in natural conversational contexts. Given a context and a response speech, CVAM describes salient moments that characterize the voice and predicts nine categorical attributes spanning gender, pitch, pacing, emotion, and delivery. The key challenge lies in perceptual fields such as emotion and delivery, which are inherently subjective and lack definitive ground truth. Therefore, we collect ~10 human annotations for each of 3k real and synthetic responses derived from the CANDOR corpus. CVAM is supervised finetuned on synthesized aesthetic descriptions and labels, then optimized with Group Relative Policy Optimization on human judgments. Experiments show that CVAM better agrees with human listeners than Gemini 3.1 Pro and open-source speech LLMs, and outperforms single-human-vs.-rest agreement. Together, we demonstrate the importance of grounding voice aesthetics in human perception and propose a principled framework for human alignment.
#### Towards Automated Clinical Behavioral Coding with Large Language Models: A Case study Using BOSCC recordings of Children
 - **Authors:** Jordan Prescott, Aditya Kommineni, Tiantian Feng, Megan Micheletti, Alyssa Viggiano, Luis Angeles, Anfeng Xu, Lynn Perry, Catherine Lord, Daniel Messinger, Shrikanth Narayanan
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2610.11106

 - **Pdf link:** https://arxiv.org/pdf/2610.11106

 - **Abstract**
 Autism spectrum disorder (ASD) is a neurodevelopmental condition characterized by differences in social communication and by restricted interests and repetitive behaviors. Treatment interventions often target social-communication skills, creating a need for reliable measures of behavioral change. The Brief Observation of Social Communication Change (BOSCC) is a validated treatment-response measure based on brief play and social-communication interactions between a child and trained examiner. The BOSCC coding process is resource-intensive and requires trained experts, motivating the automation of coding in order to improve scalability and accessibility. In this work, we evaluate general-purpose large language models (LLMs) for predicting speech-related BOSCC codes from different input representations. We compare transcript, diarized-transcript, and targeted audio conditions across 163 in-house recordings. LLMs are able to perform well in assessing verbal exchange, but do not perform as well when identifying atypical speech patterns. Additionally, performance varies considerably across scoring decisions, with no consistent pattern across diagnosis groups. An audit of model predictions indicates that applying the BOSCC coding criteria and interpreting ambiguous speech evidence remain challenges.
#### SmoothConv and DuplexConv: Complementary Mandarin Multi-Party Conversational Speech Corpora for Speech Interaction
 - **Authors:** Chengyou Wang, Mingchen Shao, Chunjiang He, Zeyu Zhu, Jierui Guo, Bingshen Mu, Zikai Liu, Hanke Xie, Yuhang Dai, Zhou Zhu, Lei Xie
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2610.11150

 - **Pdf link:** https://arxiv.org/pdf/2610.11150

 - **Abstract**
 Recent advances in large audio language models (LALMs) have driven the development of natural and intelligent speech interaction systems. Such systems need to model complex conversational behaviors, including turn-taking, overlapping speech, and speaker coordination. Multi-party conversations provide a realistic setting for studying these behaviors, yet existing Mandarin conversational corpora often lack synchronized participant-level speech tracks and comprehensive annotations. In this work, we introduce SmoothConv and DuplexConv, two complementary Mandarin multi-party conversational speech corpora totaling 2,100 hours. SmoothConv provides human-verified conversations for reliable analysis and evaluation, while DuplexConv offers large-scale automatically annotated conversations through a scalable pipeline for model training. Both corpora provide synchronized participant-level speech tracks and multi-dimensional fine-grained annotations. We further release the SmoothConv Benchmark and evaluate these resources on speech separation, multi-speaker automatic speech recognition (MSASR), and turn detection tasks. Experimental results demonstrate the utility of the proposed resources for multi-party speech interaction modeling. The datasets, benchmark, and related resources are publicly available.
#### Monaural Continuous-Radius Regional Speech Extraction with Cross-Radius Consistency Learning
 - **Authors:** Biao Dong, Jie Chen, Jianwei Fang, Wei Xiao, Jiqing Han, Yongjun He
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2610.11507

 - **Pdf link:** https://arxiv.org/pdf/2610.11507

 - **Abstract**
 Source-to-microphone distance enables speaker-independent speech extraction without prior enrollment. We propose a monaural continuous-radius regional speech extraction method that directly models the cumulative speech target within a queried radius. To realize continuous region control, the query radius is encoded as a continuous scalar and injected into a time-frequency extraction network, allowing a single model to operate over arbitrary radii within the trained range. Exploiting the nested structure of target-speaker sets across query radii, we introduce cross-radius consistency learning to stabilize predictions for adjacent radii sharing the same nonempty target-speaker set. Experiments on measured RIRs show that continuous-radius conditioning improves selective extraction over discrete conditioning. The proposed method achieves 29.25~dB SI-SDR, 4.40~dB SI-SDRi, 62.01~dB attenuation, and 0.46\% RCE, outperforming fixed-threshold and local-range baselines. It also remains effective with more speakers and additive noise.
#### SteerablePlex: Can We Steer Full-Duplex Models?
 - **Authors:** Haolong Zheng, Maike Züfle, Dominik Macháček, Peter Polák, Xulin Fan, Xavier Sumba, Siyin Wang, Ondřej Klejch, Mark Hasegawa-Johnson
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Computation and Language (cs.CL)
 - **Arxiv link:** https://arxiv.org/abs/2610.12201

 - **Pdf link:** https://arxiv.org/pdf/2610.12201

 - **Abstract**
 Full-duplex speech models can listen and speak simultaneously, enabling natural interaction, but become increasingly difficult to control as the conversation history grows. When used as user simulators, this lack of control can cause them to deviate from prescribed scenarios and produce unreliable evaluation outcomes. We introduce SimIF-Bench (Simulator Instruction-Following Benchmark), which evaluates whether a conversational model stays within a prescribed scenario and completes multiple goals in the required order. The benchmark reveals that current open-source full-duplex models struggle to follow such constraints. We then introduce a Group Reward-Decoupled Normalization Policy Optimization (GDPO)-based training recipe that enables a full-duplex model to follow textual instructions during an ongoing conversation while maintaining its turn-taking ability. By connecting the resulting SteerablePlex to an asynchronous backend language model that monitors the conversation and provides instructions when needed, we build a more controllable full-duplex user simulator that follows multi-stage constraints more reliably than existing open-source models and GPT-Realtime.
#### Disentangling Linguistic and Paralinguistic Information with Routed Sparse Autoencoders
 - **Authors:** Beimnet Bekele Guta, Xiaoyu Yang, Guangzhi Sun, Philip C. Woodland
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2610.10865

 - **Pdf link:** https://arxiv.org/pdf/2610.10865

 - **Abstract**
 Self-supervised speech encoders contain linguistic and paralinguistic information in a shared, entangled representation space. We combine a TopK sparse autoencoder with route-specific supervision and cross-factor adversaries. Across frozen SPEAR and WavLM encoders, independent probes show factor-specific retention and suppression: linguistic information remains stronger in the linguistic route, while paralinguistic factors, including speaker identity, emotion, and prosody, are retained in the paralinguistic route and substantially reduced in the linguistic route. The route organisation learned on LibriSpeech persists on MSP-Podcast without representation-side retraining. Feature-space route interventions further transfer the swapped factor while largely preserving the information carried by the unchanged route. These results show consistent route-selective separation across encoders, corpora, independent probes, and representation-level interventions.
#### Cross-Lingual Speaker Verification with Self-Supervised Pre-Trained Models
 - **Authors:** Jinghan Peng, Yu Zheng, Weiqiang Wang, Jian Liu
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2610.11099

 - **Pdf link:** https://arxiv.org/pdf/2610.11099

 - **Abstract**
 Speaker verification (SV) performance degrades under language mismatch due to the entanglement of speaker identity with language-specific acoustic cues. To address this problem, we leverage large-scale self-supervised pre-trained models (PTMs) to learn language-agnostic speaker representations. We utilize PTMs as robust front-end feature extractors, capitalizing on their rich acoustic and linguistic knowledge acquired from vast, diverse audio data. These generalized features are then used to train a downstream speaker embedding network, effectively disentangling speaker identity from language-specific characteristics. We validate our approach on the TidyVoice2026 benchmark, which benchmarks SV under language mismatch. Our proposed system (team T02) achieves equal error rates (EERs) of 2.21% on tv26_eval-A and 2.99% on tv26_eval-U.
#### Edit Who Speaks, Control How They Speak: Global Timbre Editing and Local Instruction Control for TTS
 - **Authors:** Junchuan Zhao, Chenglin Xu, Wei Zeng, Haoyang Li, Yiwen Guo, Ye Wang
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2610.11437

 - **Pdf link:** https://arxiv.org/pdf/2610.11437

 - **Abstract**
 Instruction-based text-to-speech (TTS) offers control over voice characteristics and speech expression through interfaces including voice cloning and text-based voice design. Voice cloning reproduces a reference voice, whereas text-based voice design creates a voice from a natural-language description. However, neither interface directly enables users to modify the timbre of a given reference and synthesize speech with the modified voice. Meanwhile, utterance-level expressive instructions leave changes across individual text segments underspecified. We introduce \textbf{EDICT}, a framework that unifies global timbre editing and local expressive control by using an edited acoustic reference to anchor voice identity across segments. To enable synthesis with an instruction-edited voice, EDICT combines reference audio with structured timbre edits to generate an edited reference in codec-token space. This representation serves as a shared voice anchor for a frozen TTS backbone, allowing segment-specific natural-language instructions to guide expression. To accommodate instruction changes while supporting acoustic continuity, EDICT rebuilds the KV cache at each segment boundary, refreshing instruction conditioning while retaining bounded acoustic context from previously generated speech. Evaluations on our proposed TimbreEdit-Bench and IntraTTS-Bench demonstrate improved timbre editing and a favorable balance between local instruction adherence, speaker consistency, and transition quality. Audio demos are available.
#### Beyond Speech Captions: Speech-Rewarded Style Planning for Conversational Text-to-Speech
 - **Authors:** Shiao Zhu, Lianbo Liu, Sizhen Lyu, Yuzhe Wang, Sheng Li, Takahiro Shinozaki
 - **Subjects:** Subjects:
Sound (cs.SD); Computation and Language (cs.CL); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2610.11461

 - **Pdf link:** https://arxiv.org/pdf/2610.11461

 - **Abstract**
 Natural-language style descriptions provide an interpretable interface between large language models (LLMs) and controllable text-to-speech (TTS). However, using descriptions as pseudo-labels compresses target acoustics into text, and descriptive fidelity need not imply effective control of a particular synthesizer. We empirically show that speech-text alignment only weakly predicts downstream acoustic similarity among candidate instructions for the same utterance. We therefore propose Speech-Rewarded Style Planning (SRSP), which trains a text-based style planner through a frozen downstream TTS model. Given dialogue history and response text, the planner generates candidate instructions and is optimized with group-relative policy optimization (GRPO), using the teacher-forced likelihood of target speech tokens as the reward. On an English subset of the ISCSLP 2026 CoT-TTS corpus, SRSP achieves higher speech-style and emotion similarity to target speech and lower mel-cepstral distortion than the Base LLM and target-audio-informed captioning baselines. LLM-based expressive speech evaluation further shows gains over all baselines in contextual appropriateness and reference consistency.


by Zyzzyva0381 (Windy). 


2026-10-09
