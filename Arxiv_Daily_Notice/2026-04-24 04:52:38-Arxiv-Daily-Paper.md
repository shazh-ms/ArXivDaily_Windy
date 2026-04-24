# Showing new listings for Friday, 24 April 2026
Auto update papers at about 2:30am UTC (10:30am Beijing time) every weekday.


阅读 `Usage.md`了解如何使用此repo实现个性化的Arxiv论文推送

See `Usage.md` for instructions on how to personalize the repo. 


Keyword list: ['text-to-speech', 'text to speech', 'tts', 'LLM-based', 'speech', 'voice']


Excluded: []


### Today: 3papers 
#### Full-Duplex Interaction in Spoken Dialogue Systems: A Comprehensive Study from the ICASSP 2026 HumDial Challenge
 - **Authors:** Chengyou Wang, Hongfei Yue, Guojian Li, Zhixian Zhao, Shuiyuan Wang, Shuai Wang, Xin Xu, Hui Bu, Lei Xie
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2604.21406

 - **Pdf link:** https://arxiv.org/pdf/2604.21406

 - **Abstract**
 Full-duplex interaction, where speakers and listeners converse simultaneously, is a key element of human communication often missing from traditional spoken dialogue systems. These systems, based on rigid turn-taking paradigms, struggle to respond naturally in dynamic conversations. The Full-Duplex Interaction Track of ICASSP 2026 Human-like Spoken Dialogue Systems Challenge (HumDial Challenge) aims to advance the evaluation of full-duplex systems by offering a framework for handling real-time interruptions, speech overlap, and dynamic turn negotiation. We introduce a comprehensive benchmark for full-duplex spoken dialogue systems, built from the HumDial Challenge. We release a high-quality dual-channel dataset of real human-recorded conversations, capturing interruptions, overlapping speech, and feedback mechanisms. This dataset forms the basis for the HumDial-FDBench benchmark, which assesses a system's ability to handle interruptions while maintaining conversational flow. Additionally, we create a public leaderboard to compare the performance of open-source and proprietary models, promoting transparent, reproducible evaluation. These resources support the development of more responsive, adaptive, and human-like dialogue systems.
#### DiariZen Explained: A Tutorial for the Open Source State-of-the-Art Speaker Diarization Pipeline
 - **Authors:** Nikhil Raghav
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2604.21507

 - **Pdf link:** https://arxiv.org/pdf/2604.21507

 - **Abstract**
 Speaker diarization (SD) is the task of answering "who spoke when" in a multi-speaker audio stream. Classically, an SD system clusters segments of speech belonging to an individual speaker's identity. Recent years have seen substantial progress in SD through end-to-end neural diarization (EEND) approaches. DiariZen, a hybrid SD pipeline built upon a structurally pruned WavLM-Large encoder, a Conformer backend with powerset classification, and VBx clustering, represents the leading open-source state of the art at the time of writing across multiple benchmarks. Despite its strong performance, the DiariZen architecture spans several repositories and frameworks, making it difficult for researchers and practitioners to understand, reproduce, or extend the system as a whole. This tutorial paper provides a self-contained, block-by-block explanation of the complete DiariZen pipeline, decomposing it into seven stages: (1) audio loading and sliding window segmentation, (2) WavLM feature extraction with learned layer weighting, (3) Conformer backend and powerset classification, (4) segmentation aggregation via overlap-add, (5) speaker embedding extraction with overlap exclusion, (6) VBx clustering with PLDA scoring, and (7) reconstruction and RTTM output. For each block, we provide the conceptual motivation, source code references, intermediate tensor shapes, and annotated visualizations of the actual outputs on a 30s excerpt from the AMI Meeting Corpus. The implementation is available at this https URL, which includes standalone executable scripts for each block and a Jupyter notebook that runs the complete pipeline end-to-end.
#### Dilated CNNs for Periodic Signal Processing: A Low-Complexity Approach
 - **Authors:** Eli Gildish, Michael Grebshtein, Igor Makienko
 - **Subjects:** Subjects:
Machine Learning (cs.LG); Artificial Intelligence (cs.AI); Audio and Speech Processing (eess.AS); Signal Processing (eess.SP)
 - **Arxiv link:** https://arxiv.org/abs/2604.21651

 - **Pdf link:** https://arxiv.org/pdf/2604.21651

 - **Abstract**
 Denoising of periodic signals and accurate waveform estimation are core tasks across many signal processing domains, including speech, music, medical diagnostics, radio, and sonar. Although deep learning methods have recently shown performance improvements over classical approaches, they require substantial computational resources and are usually trained separately for each signal observation. This study proposes a computationally efficient method based on DCNN and Re-sampling, termed R-DCNN, designed for operation under strict power and resource constraints. The approach targets signals with varying fundamental frequencies and requires only a single observation for training. It generalizes to additional signals via a lightweight resampling step that aligns time scales in signals with different frequencies to re-use the same network weights. Despite its low computational complexity, R-DCNN achieves performance comparable to state-of-the-art classical methods, such as autoregressive (AR)-based techniques, as well as conventional DCNNs trained individually for each observation. This combination of efficiency and performance makes the proposed method particularly well suited for deployment in resource-constrained environments without sacrificing denoising or estimation accuracy.


by Zyzzyva0381 (Windy). 


2026-04-24
