# Showing new listings for Tuesday, 21 July 2026
Auto update papers at about 2:30am UTC (10:30am Beijing time) every weekday.


阅读 `Usage.md`了解如何使用此repo实现个性化的Arxiv论文推送

See `Usage.md` for instructions on how to personalize the repo. 


Keyword list: ['text-to-speech', 'text to speech', 'tts', 'LLM-based', 'speech', 'voice']


Excluded: []


### Today: 6papers 
#### AMECxSV: Adaptive Metadata-Driven Embedding-Fusion Calibration for X-Lingual Speaker Verification
 - **Authors:** Xin Wei, Shi He, Yihe Yuan, Huang-Cheng Chou, Sudarsana Reddy Kadiri, Shrikanth Narayanan
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2607.16532

 - **Pdf link:** https://arxiv.org/pdf/2607.16532

 - **Abstract**
 In X-lingual automatic speaker verification (ASV), fixed front-end scores vary in reliability with language match, duration, and score source. We propose AMECxSV, an adaptive metadata-driven embedding-fusion calibration backend for metadata-available settings. AMECxSV fuses trial scores with metadata to produce calibrated target posteriors, with optional posterior-confidence abstention; metadata serve as calibration context, not speaker evidence. On a development-derived speaker-disjoint held-out split, score+metadata heads reduce equal error rate (EER) from 3.15% to 2.42% for the official TidyVoice score source and from 0.64% to 0.43% for LI-MSV; the dual-score head reaches 0.43% full-coverage EER. At 0.79 coverage, abstention yields 0.03% accepted-trial EER, not a full-coverage metric. Matched score-only, metadata-permutation, and metadata-only controls support a calibration-context interpretation and limit claims to metadata-available scoring.
#### An Audio Language Model-Based Voice Concept Bottleneck Framework for Interpretable Health Assessment
 - **Authors:** Yu-Wen Chen, Julia Hirschberg
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2607.16967

 - **Pdf link:** https://arxiv.org/pdf/2607.16967

 - **Abstract**
 Interpretability is critical in clinical decision support. Concept bottleneck frameworks improve it by representing inputs as human-understandable concepts and restricting predictions solely on them. However, research on their use for voice-based health assessment remains limited. In this study, we propose a voice concept bottleneck framework for interpretable health assessment using an audio language model (ALM). The ALM is fine-tuned on a voice quality assessment dataset to enhance its understanding of voice concepts and serves as an independent concept extractor, producing discrete, interpretable scores for a lightweight downstream classifier. The discrete concept scores provide intuitive interpretation, while the lightweight classifier facilitates post-hoc interpretability analyses. Results on depression and dysarthria assessment tasks demonstrate that the proposed framework can flexibly adapt voice concepts to different health conditions and consistently outperforms openSMILE-based and self-supervised speech model-based baselines.
#### SALMONN-2: Advancing General-Purpose Hearing Abilities with Self-Supervised Representations
 - **Authors:** Xiaoyu Yang, Xuenan Xu, Wenyi Yu, Siyin Wang, Changli Tang, Terumi Chiba, Siyuan Hou, Ziyang Zhang, Wen Wu, Baoxiang Li, Guangzhi Sun, Chao Zhang, Philip Woodland
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2607.17079

 - **Pdf link:** https://arxiv.org/pdf/2607.17079

 - **Abstract**
 Recent audio large language models (ALLMs) are typically built upon audio encoders trained with large amounts of supervised data. Since self-supervised learning (SSL) audio encoder models are known to learn general-purpose and transferable representations, we investigate whether general-purpose SSL audio representations can serve as an effective foundation for ALLMs. We present SALMONN-2, an ALLM built upon a unified SSL encoder. To better exploit the hierarchical representations learned by SSL encoders, we propose a multi-layer feature fusion (MLF) adapter that aggregates information from all encoder layers before projecting them into the language model. Beyond conventional audio understanding tasks, we further explore multimodal in-context learning (MICL) in ALLMs and study how this capability can be acquired through contextual biasing training. Experimental results show that a general-purpose SSL encoder achieves performance comparable to, or better than, specialised supervised audio encoders while providing a more balanced capability across speech, audio, music and paralinguistic tasks. SALMONN-2 further achieves state-of-the-art performance among comparable-scale open-weight models on ALLM understanding benchmarks, obtaining the best results on MMAU-Pro, MMAR and MMSU. We also show that MICL does not emerge naturally in ALLMs, but can be effectively acquired through targeted contextual biasing training.
#### X-Translator: A Real-Time Multilingual Speaker-Aware Speech-to-Speech Translation System
 - **Authors:** Yuxiang Zhao, Yichi Zhang, Yanjie An, Yanqiao Zhu, Zhanxun Liu, Yushen Chen, Qixi Zheng, Haina Zhu, Yunchong Xiao, Keqi Deng, Shuai Fan, Kai Yu, Xie Chen
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2607.17544

 - **Pdf link:** https://arxiv.org/pdf/2607.17544

 - **Abstract**
 Real-time speech-to-speech translation (S2ST) systems must balance translation quality, latency, speech naturalness, and speaker consistency. Publicly documented S2ST systems have advanced direct, multilingual, streaming, and expressive modeling, while proprietary products and APIs increasingly expose real-time translation capabilities to users. However, practical deployment remains challenging for open and reproducible systems, especially in long-form and multi-speaker conversations where partial ASR hypotheses are unstable, turn boundaries are ambiguous, and target speech must be generated with an appropriate speaker prompt. We present X-Translator, a low-cost modular cascaded S2ST system that combines streaming ASR, machine translation, and prompt-conditioned TTS through a session-level runtime controller. The system uses incremental segment commitment to convert unstable ASR streams into translation-ready units, and an online speaker prompt manager to bind source speech spans to speaker-specific voice prompts for synthesis. We evaluate translation, speech quality, and latency with OpenSTBench, compare against proprietary speech translation APIs as behavioral baselines, measure long-form voice stability, evaluate speaker preservation in multi-speaker conversations, and assess multilingual translation quality. X-Translator provides an open platform for understanding the practical trade-offs of deployment-oriented S2ST. Code and demo are available at this https URL.
#### The tttAI System for the TSA-ASR Task of the SmartGlasses Challenge 2026
 - **Authors:** Xuanji He, Gaoyang Dong, Xiaoxiao Li, Minchuan Chen, Fengjie Zhu
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2607.17867

 - **Pdf link:** https://arxiv.org/pdf/2607.17867

 - **Abstract**
 This paper presents the tttAI system submitted to the TSA-ASR task of the SmartGlasses Challenge 2026, evaluated on both two-person dialogues (Track 1) and multi-party meetings (Track 2). The task requires time-stamped speaker-attributed speech recognition from smart-glasses recordings. This is particularly challenging due to long-form audio, multiple speakers, and frequent overlapping speech. We proposed a cascaded architecture consisting of speaker diarization, overlap detection, target-speaker extraction, post-processing, and automatic speech recognition. The diarization module extracts features via WavLM-Large, performs frame-wise speaker classification with a Conformer encoder, and then generates global speaker segments through embedding clustering. For overlapped regions, we apply a WeSep-based target-speaker extraction model with ECAPA-TDNN speaker embeddings. When the extraction is unreliable, a dominant-speaker fallback strategy is used. The final system uses FireRedASR2-AED with the first microphone channel. The submitted system has a total parameter count of approximately 1.53B. On Track 1, our system achieves a tcpCER of 7.10%. On Track 2, it achieves a tcpCER of 34.04% and ranks second on the leaderboard.
#### Robust Summarization of Doctor-Patient Conversations: TalTech Systems for the Beyond Transcription Challenge
 - **Authors:** Aivo Olev, Tanel Alumäe
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2607.17230

 - **Pdf link:** https://arxiv.org/pdf/2607.17230

 - **Abstract**
 This paper describes TalTech's submissions to the Beyond Transcription Challenge (BeTraC), which requires generating SOAP notes directly from long doctor-patient conversation recordings, without intermediate transcription. After screening open-weight speech LLMs for long-audio robustness, we adapted Voxtral Mini (lightweight track) and Voxtral Small (heavyweight track) with LoRA supervised fine-tuning followed by DAPO reinforcement learning that uses the challenge metric, Open Medical Concept F1, as its reward. Our systems ranked first in both tracks, and an independent LLM-as-a-judge evaluation showed the lowest hallucination rate among all submissions, indicating that reinforcement learning against a concept-matching metric need not compromise factual reliability. We also find that fine-tuning on text transcripts transfers well to speech input and appears to improve robustness on out-of-domain real recordings.


by Zyzzyva0381 (Windy). 


2026-07-21
