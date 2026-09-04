# Showing new listings for Friday, 4 September 2026
Auto update papers at about 2:30am UTC (10:30am Beijing time) every weekday.


阅读 `Usage.md`了解如何使用此repo实现个性化的Arxiv论文推送

See `Usage.md` for instructions on how to personalize the repo. 


Keyword list: ['text-to-speech', 'text to speech', 'tts', 'LLM-based', 'speech', 'voice']


Excluded: []


### Today: 6papers 
#### StreamWSR: Streamable and Lightweight Waveform-Domain Neural Speech Super-Resolution
 - **Authors:** Yuan Tian, Yang Ai, Hui-Peng Du, Zhen-Hua Ling
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2609.03381

 - **Pdf link:** https://arxiv.org/pdf/2609.03381

 - **Abstract**
 This paper proposes StreamWSR, a Streamable neural Waveform-domain model for speech Super-Resolution (SR). By adopting a fully causal architecture with compact frame-level waveform representation, the proposed StreamWSR supports zero-look-ahead streaming inference while avoiding vocoder-based reconstruction and explicit phase prediction. Specifically, StreamWSR downsamples the input waveform into a compact frame-level representation using strided causal convolutions. Then, a lightweight causal long-short-term modeling backbone is employed to capture both local waveform structures and long-range historical dependencies under causal constraints. Finally, the modeled output is converted back to the waveform domain through a causal transposed-convolution and combined with the input waveform via a residual connection to generate the final high-resolution speech. Experimental results on 16 kHz speech SR show that StreamWSR achieves competitive or superior speech quality and intelligibility compared with representative waveform- and spectrum-based baselines, while maintaining a zero-look-ahead streaming advantage with only 9M parameters and 2G FLOPs.
#### Summary of the ChinaVoices Challenge 2026: Data, Tasks, Baseline, and Methods
 - **Authors:** Yujie Liao, Bingshen Mu, Shuiyuan Wang, Liumeng Xue, Hexin Liu, Xian Shi, Jie Hu, Lei Xie
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2609.03471

 - **Pdf link:** https://arxiv.org/pdf/2609.03471

 - **Abstract**
 This paper summarizes the ChinaVoices Challenge 2026, which aims to establish unified task definitions and evaluation conditions for Chinese dialect speech processing and to advance multi-dialect identification and automatic speech recognition. The challenge covers 16 dialect categories and defines two tasks: Chinese Multi-Dialect Identification and Chinese Multi-Dialect Automatic Speech Recognition (ASR). It uses approximately 320 hours of speech across the Reference Set, Open Evaluation Set, and Hidden Evaluation Set. The two tasks use the same evaluation audio, and each includes restricted-data and open-data tracks. We describe the task settings, data, evaluation metrics, and Qwen3-ASR-1.7B baseline, and analyze the leaderboard results and submitted systems. In total, 28 teams submit results, 17 provide system reports, and systems from 15 teams pass the compliance review and are included in the analysis. Most eligible systems outperform the baseline, and the official top-three order remains unchanged on the Hidden Evaluation Set for both tasks. Dialect-level results show that categories with higher identification accuracy generally have lower ASR error rates, although the tasks assess related but distinct capabilities. Leading identification systems commonly exploit dialect-discriminative acoustic representations, whereas leading ASR systems emphasize data normalization, augmentation, and auxiliary CTC objectives. These results provide practical guidance for developing and evaluating Chinese multi-dialect speech processing systems.
#### Fairness Evaluation of Edge-AI Implementation for Cleft Lip and Palate Speech ASR
 - **Authors:** Susmita Bhattacharjee, Himashri Deka, H.S. Shekhawat, S.R.M. Prasanna
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2609.03982

 - **Pdf link:** https://arxiv.org/pdf/2609.03982

 - **Abstract**
 Automatic speech recognition (ASR) remains challenging for individuals with cleft lip and palate (CLP) because of limited pathological speech data and large variations in speech characteristics across speakers and severity levels. These recognition difficulties can reduce the accessibility of voice-based human-computer interaction, particularly when cloud-based ASR services are unavailable or unreliable. This work investigates a severity-aware and edge-deployable ASR framework for improving recognition of CLP speech using Whisper-small. The model was fine-tuned using different combinations of normal and CLP speech representing mild, moderate, and severe conditions, together with a CLP-only training configuration, to examine how the inclusion of different severity levels influences recognition performance and fairness across speakers. The pretrained model produced pooled word error rate (WER) and phoneme error rate (PER) values of 62.46% and 52.72%, respectively. Severity-aware fine-tuning substantially improved performance, reducing the best pooled WER to 22.72% and the best pooled PER to 18.44%. Training with a broader representation of CLP severity levels also provided the best overall balance between recognition accuracy and performance consistency across severity groups. Deployment on an NVIDIA Jetson platform demonstrated real-time inference for all fine-tuned models, with real-time factors of 0.167-0.171 and peak GPU memory usage of approximately 566 MB. The results demonstrate that incorporating severity diversity during ASR adaptation can substantially improve recognition of CLP speech while reducing performance disparities across severity groups. The proposed approach further enables low-latency, Internet-independent speech interaction on edge devices, supporting more accessible and inclusive voice-based human-computer interaction for individuals with CLP.
#### Deep Neural Compression for RIR-Characterized Acoustic Environments with Structure-Aware Constraints
 - **Authors:** Chen-Yuan Ning, Yang Ai, Hui-Peng Du, Xiao-Hang Jiang, Zhen-Hua Ling
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2609.04085

 - **Pdf link:** https://arxiv.org/pdf/2609.04085

 - **Abstract**
 Room impulse responses (RIRs) characterize the acoustic environment of a room by capturing how sound propagates and decays within an enclosed space. In applications such as immersive audio rendering, accurate acoustic reconstruction often relies on spatially densely sampled RIRs. This consequently gives rise to a large volume of RIR data, imposing a substantial burden on storage. Although recent neural audio codecs provide an effective framework for low-bitrate compression, their training objectives are mainly tailored to speech and general audio, and are therefore not well aligned with the acoustic characteristics of RIRs. Therefore, we propose an EnCodec-based neural RIR compression method, which incorporates RIR structure-aware constraints at two levels. Specifically, at the RIR level, structure-aware constraints are imposed on the global decay behavior and local energy distribution of RIRs through energy decay curve (EDC) regularization and a short-time window energy constraint, while at the reverberant-speech level, reverberant-speech supervision is further introduced to constrain the consistency of the reverberant speech generated by the reconstructed RIRs. Experimental results show that, at a low bitrate of 375 bps, the proposed method achieves lower RIR reconstruction error and better reverberant-speech perceptual consistency than audio-oriented codecs.
#### VoxReason: Listener-Free Evaluation of Source-Grounded Speech Planning Before Synthesis
 - **Authors:** Mengzhe Geng
 - **Subjects:** Subjects:
Sound (cs.SD); Computation and Language (cs.CL); Machine Learning (cs.LG); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2609.03203

 - **Pdf link:** https://arxiv.org/pdf/2609.03203

 - **Abstract**
 Expressive speech systems make a decision before any waveform is rendered: how an utterance is delivered. In dialogue agents, narration, and role-conditioned TTS, that hidden planning step sets affect, pitch, energy, rate, pause, emphasis, and stance, yet downstream audio scores rarely reveal whether those choices were licensed by the source record, a source-use failure that occurs before any waveform exists. VoxReason makes that pre-synthesis decision measurable as a listener-free task for source-grounded speech planning. Before synthesis, VoxReason measures whether delivery choices are grounded in cited source records. Systems output a source-cited speaking-plan with evidence citations, and a deterministic verifier checks citation legality, slot agreement, unsupported state, schema validity, and one-cue counterfactual locality. On 1,440 checked source-label cases, shortcut controls show why slot accuracy alone is unsafe: a key-lookup oracle reaches 1.000 plan-slot accuracy on seen keys, while an emotion prior still reaches 0.958 slot accuracy on source-key-disjoint cases without citing intensity or identity. In a separate 100-case learned source-key-disjoint comparison, a 7B locality SFT+CF repair improves plan-slot accuracy/locality from 0.684/0.141 to 0.919/1.000, and removing source records lowers citation-required grounded score by 0.488. Rendered waveform quality remains outside the present evaluation.
#### Alignment-Free Text-Audiobox for Voice Dubbing and Full-Duplex Dialogue Synthesis
 - **Authors:** Sanyuan Chen, Min-Jae Hwang, Sho Inoue, Anna Sun, Bokai Yu, David Kant, Dongmin Hyun, Dorian Desblancs, Gregory Antonovsky, Oleg Repin, Peng-Jen Chen, Xutai Ma, Zehai Tu, Juan Pino, Wei-Ning Hsu
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2609.03992

 - **Pdf link:** https://arxiv.org/pdf/2609.03992

 - **Abstract**
 We present Alignment-Free Text-Audiobox (Text-AB), a unified framework for high-quality voice dubbing and full-duplex dialogue synthesis. Building on a Diffusion Transformer trained with a flow-matching objective, Text-AB departs from the Audiobox system along three dimensions. First, it operates in a latent diffusion framework using DAC-VAE features that encode 48 kHz waveforms into a 25 Hz latent sequence, giving over 10x higher compression than previous EnCodec representations while improving resynthesis quality. Second, Text-AB is alignment-free: it consumes raw text via an off-the-shelf text encoder and learns text-speech alignment through cross-attention, removing the need for forced alignment and explicit duration prediction. Third, we scale model and data substantially, pretraining a 3B-parameter model on 480k hours of monolingual speech, followed by supervised fine-tuning on three downstream tasks: cross-lingual voice dubbing, full-duplex dialogue synthesis, and emotional full-duplex dialogue synthesis. At inference, Text-AB supports one-shot generation for up to ~1 min of speech and arbitrarily long-form generation via a multi-diffusion scheme, plus a multi-stage reranking strategy that enhances quality based on automated metrics. On a real-world dubbing benchmark, Text-AB delivers a step-change improvement over the latest internal dubbing system, with large gains in prosody similarity, voice similarity, naturalness, and shareability. For full-duplex dialogue synthesis, it approaches human recordings on short-form conversations and substantially outperforms the latest internal model on long-form human-likeness and expressivity, while natively modeling turn-taking, back-channeling, and emotional dynamics. For emotional dialogue synthesis, emotion conditioning significantly improves emotion alignment and emotional interaction quality over the unconditioned baseline.


by Zyzzyva0381 (Windy). 


2026-09-04
