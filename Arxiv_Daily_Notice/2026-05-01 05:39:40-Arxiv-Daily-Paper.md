# Showing new listings for Friday, 1 May 2026
Auto update papers at about 2:30am UTC (10:30am Beijing time) every weekday.


阅读 `Usage.md`了解如何使用此repo实现个性化的Arxiv论文推送

See `Usage.md` for instructions on how to personalize the repo. 


Keyword list: ['text-to-speech', 'text to speech', 'tts', 'LLM-based', 'speech', 'voice']


Excluded: []


### Today: 4papers 
#### A Knowledge-Driven Approach to Target Speech Extraction in the Presence of Background Sound Effects for Cinematic Audio Source Separation (CASS)
 - **Authors:** Chun-wei Ho, Sabato Marco Siniscalchi, Kai Li, Chin-Hui Lee
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2604.27403

 - **Pdf link:** https://arxiv.org/pdf/2604.27403

 - **Abstract**
 We propose a knowledge-driven approach to speech target extraction in the presence of background sound effects already recorded in cinematic audio. The specific knowledge sources studied are manners of articulation that are detected in speech frames and adopted to form a knowledge vector as a part of features to enhance speech separation and target speech extraction because some short speech segments are often difficult to separate from mixed background sounds. Testing on the recent Sound Demixing Challenge data for cinematic audio source separation (CASS) shows that utilizing articulator-aware knowledge sources produces better separation results than those obtained without using any knowledge, especially for speech segments buried in unspecified background sound events.
#### BUT System Description for CHiME-9 MCoRec Challenge
 - **Authors:** Dominik Klement, Alexander Polok, Nguyen Hai Phong, Prachi Singh, Lukáš Burget
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Image and Video Processing (eess.IV)
 - **Arxiv link:** https://arxiv.org/abs/2604.27436

 - **Pdf link:** https://arxiv.org/pdf/2604.27436

 - **Abstract**
 Multi-talker automatic speech recognition (ASR) in conversational recordings remains an open problem, particularly in scenarios with large portion of overlapping speech where identifying and transcribing a target speaker is difficult from audio alone. Visual cues can help resolve speaker ambiguity, yet their integration into long-context audio-visual (AV) ASR systems has been limited. The CHiME-9 MCoRec task addresses this challenge by requiring transcription of audio-visual recordings of heavily-overlapped parallel conversations, followed by clustering the participants into conversational groups. In this work, we present the BUT system based on a long-context target-speaker AV-ASR model capable of processing long-form recordings in a single decoding pass. Our architecture conditions a pre-trained NVIDIA Parakeet-v2 ASR model on visual representations from a pre-trained AV-HuBERT model. To cluster participants into conversation groups, we employ Qwen3.5-122B LLM to estimate transcript topic similarity followed by hierarchical agglomerative clustering. On the MCoRec development set, the proposed system achieves 33.7% WER and a clustering F1 score of 0.97, improving over the official baseline by 16.2% WER and 0.15 F1 absolute. On the eval set, our team ranked second, being 0.16% WER and 0.5% F1 worse than the best system.
#### LRS-VoxMM: A benchmark for in-the-wild audio-visual speech recognition
 - **Authors:** Doyeop Kwak, Jeongsoo Choi, Suyeon Lee, Joon Son Chung
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Multimedia (cs.MM); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2604.27866

 - **Pdf link:** https://arxiv.org/pdf/2604.27866

 - **Abstract**
 We introduce LRS-VoxMM, an in-the-wild benchmark for audio-visual speech recognition (AVSR). The benchmark is derived from VoxMM, a dataset of diverse real-world spoken conversations with human-annotated transcriptions. We select AVSR-suitable samples and preprocess them in an LRS-style format for direct use in existing AVSR pipelines. Compared with commonly used benchmarks, LRS-VoxMM covers a more diverse range of scenarios and acoustic conditions. We also release distorted evaluation sets with additive noise, reverberation, and bandwidth limitation to support evaluation under severe acoustic degradation. Experimental results show that LRS-VoxMM is considerably harder than LRS3 and that the contribution of visual information becomes more evident as the audio signal degrades. LRS-VoxMM supports more realistic AVSR benchmarking and encourages further research on the role of visual information in challenging real-world conditions.
#### Predicting Upcoming Stuttering Events from Three-Second Audio: Stratified Evaluation Reveals Severity-Selective Precursors, and the Model Deploys Fully On-Device
 - **Authors:** Nazar Kozak
 - **Subjects:** Subjects:
Sound (cs.SD); Machine Learning (cs.LG); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2604.27279

 - **Pdf link:** https://arxiv.org/pdf/2604.27279

 - **Abstract**
 Audio-based stuttering systems to date have been trained for detection -- what disfluency is present now -- leaving prediction, the capability needed for closed-loop intervention, unstudied at deployable scale. We train a 616K-parameter CNN on SEP-28k (Apple, 20,131 three-second clips) to predict whether the next contiguous clip contains any disfluency. (1) Severity-selective precursor signal: on the episode-grouped test set, aggregate preblock AUC is modest (0.581 [0.542, 0.619]), but stratifying by upcoming event type reveals concentration on clinically severe events -- blocks 0.601 [0.554, 0.651] and sound repetitions 0.617 [0.567, 0.667] both exclude chance, while fillers (0.45) and word repetitions (0.49) are at chance. The aggregate objective converges to a severity-selective predictor because severe events carry prosodic precursors; fillers do not. (2) Cross-population transfer: without fine-tuning, the same checkpoint applied to 1,024 pediatric Children-Who-Stutter utterances (FluencyBank Teaching) attains AUC 0.674 detection and 0.655 prediction; DisfluencySpeech and LibriStutter reach 0.58-0.60 AUC. (3) Deployable on-device: lossless export to CoreML (1.19 MB), ONNX (40 KB), TFLite. Neural-Engine latency per 3 s window: 0.25 ms (iPhone 17 Pro Max, A19 Pro) to 0.55 ms (iPhone SE 3rd-gen and M1 Max). A 4 Hz streaming simulation uses 0.54% of the real-time budget. Platt-calibrated outputs (test ECE 0.010, from 0.177 raw). Five negative ablations -- output-level Future-Guided Learning, multi-clip GRU, time-axis concatenation, asymmetric focal loss, direct block-targeted training -- none improved over the vanilla baseline.


by Zyzzyva0381 (Windy). 


2026-05-01
