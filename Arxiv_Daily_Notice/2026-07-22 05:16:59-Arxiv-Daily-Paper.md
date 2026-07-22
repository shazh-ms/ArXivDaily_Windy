# Showing new listings for Wednesday, 22 July 2026
Auto update papers at about 2:30am UTC (10:30am Beijing time) every weekday.


阅读 `Usage.md`了解如何使用此repo实现个性化的Arxiv论文推送

See `Usage.md` for instructions on how to personalize the repo. 


Keyword list: ['text-to-speech', 'text to speech', 'tts', 'LLM-based', 'speech', 'voice']


Excluded: []


### Today: 4papers 
#### Towards Array-Invariant Speech Enhancement via Geometry-Aware Dynamic Convolution
 - **Authors:** Zhenglong Liu, Wangyou Zhang, Chenda Li, Yanmin Qian
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2607.18658

 - **Pdf link:** https://arxiv.org/pdf/2607.18658

 - **Abstract**
 Multi-channel speech enhancement (SE) systems exhibit superior performance over single-channel methods but are constrained to fixed microphone array configurations. This restricts their real-world deployment across devices with diverse array geometries. While recent array-agnostic SE methods address variable microphone numbers and permutations, they largely fail to exploit explicit array geometry priors when available, missing a crucial cue for optimal spatial filtering. A Geometry-Aware Dynamic Convolution (Geo-DConv) framework is proposed, which explicitly leverages microphone coordinates to transform standard fixed-array SE models into robust array-invariant systems. Experiments are conducted on the recent real-recorded RealMAN multi-channel speech dataset. Results demonstrate that the proposed architecture enables two widely used fixed-array models to adapt to array-invariant settings, with consistent performance improvements across diverse array topologies.
#### Summary of DCASE 2026 Task 5: Audio-Dependent Question Answering
 - **Authors:** Haolin He, Renhe Sun, Zheqi Dai, Xingjian Du, Chunyat Wu, Zining Liang, Zhengxi Liu, Jiahe Lei, Runbang Wang, Jiayi Zhou, Mingru Yang, Xiquan Li, Yun Chen, Xie Chen, Zhiyao Duan, Weiqiang Wang, Mark D. Plumbley, Jian Liu, Qiuqiang Kong
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2607.18718

 - **Pdf link:** https://arxiv.org/pdf/2607.18718

 - **Abstract**
 DCASE~2026 Task~5 introduces Audio-Dependent Question Answering (ADQA), which tests whether large audio-language models answer from the audio rather than from textual priors. An Audio-Dependency Filtering (ADF) pipeline combines silent-audio probing, per-option perplexity, a large language model (LLM) commonsense check, and human review to remove items solvable from text alone. The 3000 items that pass form the ADQA-Bench evaluation set, spanning music, speech, and environmental audio. The inaugural edition draws 14 teams and 36 submissions across two tracks defined by total parameter count (up to 100B and under 10B). A Chung-Ang University ensemble of MOSS-Audio-8B-Thinking and Qwen3-Omni-30B reaches the top overall accuracy at \pct{58.33}, and a MOSS-only configuration from the same team leads the sub-10B track at \pct{57.30}. Across the 30 submissions with a comparable development score, evaluation accuracy falls by 11.91 percentage points (pp) on average (median 10.91\,pp) on the hidden evaluation split, which is designed to be harder than the development split. The most common building blocks are: the MOSS-Audio-8B-Thinking backbone (13 of 36 submissions), Low-Rank Adaptation (LoRA) fine-tuning on AudioMCQ-StrongAC, and preference or reinforcement-learning objectives -- Group Relative Policy Optimization (GRPO) in five teams, Group reward-Decoupled Normalization Policy Optimization (GDPO) in two. At test time, prompt engineering is near-universal, and majority or choice-permutation voting is common. Every system misses the same set of 233 evaluation items.
#### A Situational Speech Synthesizer for Yoruba: System Design, Phonological Rule Architecture, and Orthographic Extensions for Contour
 - **Authors:** Kola Tubosun, Adedayo Oluokun, Hafiz Adewuyi, Dadepo Aderemi
 - **Subjects:** Subjects:
Sound (cs.SD); Computation and Language (cs.CL); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2607.18317

 - **Pdf link:** https://arxiv.org/pdf/2607.18317

 - **Abstract**
 We present TTSYoruba, a rule-based concatenative diphone speech synthesizer for Yoruba, deployed at online as part of the this http URL open dictionary of Yoruba personal names. The system takes tone-marked Yoruba text as input and produces audio output by applying a hand-crafted phonological rule system to a recorded inventory of 651 diphone units spanning five tonal variants of every consonant-vowel combination in the language. We describe the phonological architecture of the system in detail, including our complete tonal file-selection logic, our treatment of the three-way nasal disambiguation problem (oral /n/, nasalized vowel, and syllabic nasal), and the derivation of contextual rising and falling tones from level-tone input. We also present, as an orthographic contribution, the adoption of the caron and circumflex, which are symbols with prior standing in Yoruba phonological transcription, as standard single-vowel contour tone markers, integrated into the TTS normalization pipeline and the WriteYoruba keyboard input tool. The system's performance was evaluated through a listener study (N=50), with detailed results on Mean Opinion Scores (MOS) presented in Section 6. Keywords: Yoruba, text-to-speech, low-resource languages, diphone synthesis, contour tones, African language NLP, rule-based synthesis
#### Staged Depth-Pruning Distillation of a Flow-Matching Text-to-Speech Teacher: A Compact Hindi Speech Synthesizer
 - **Authors:** Sivateja Trikutam
 - **Subjects:** Subjects:
Sound (cs.SD); Computation and Language (cs.CL); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2607.18662

 - **Pdf link:** https://arxiv.org/pdf/2607.18662

 - **Abstract**
 We present a practical recipe for building a compact Hindi text-to-speech (TTS) model by distilling a large flow-matching teacher (IndicF5, 337M-parameter DiT) under a severe data budget (~17.6 hours). Training a small model from scratch on this much data fails outright. Instead we warm-start the student from the teacher by pruning depth only: keeping the teacher's width, text dimension, attention heads, and mel/text I/O fixed so all non-block tensors copy one-to-one, and retaining an evenly-spaced subset of transformer blocks. We first measure how much depth the teacher tolerates (it remains near-functional at -27% blocks but collapses past -50%), then descend gradually (22 -> 16 -> 12 -> 8 -> 6 blocks), re-fine-tuning after each prune, with each step gated by an objective ASR word-error-rate (WER) check. The resulting students reach WER 0.00 on unseen sentences at 249M and 190M parameters, and remain robust down to 131M; at 102M we observe a clear capacity cliff that we attribute to the data budget rather than the recipe. We also document two train/inference feature- and library-parity failures (mel filterbank and rotary-embedding library versions) that silently degrade audio, and a version-independent fix. The method yields a high-quality Hindi voice that runs in real time on a 6 GB laptop GPU. An independent 50-sentence FLEURS benchmark compares the released 190M student against its teacher and MMS-TTS-hin.


by Zyzzyva0381 (Windy). 


2026-07-22
