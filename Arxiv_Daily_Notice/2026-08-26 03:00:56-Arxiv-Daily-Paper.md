# Showing new listings for Wednesday, 26 August 2026
Auto update papers at about 2:30am UTC (10:30am Beijing time) every weekday.


阅读 `Usage.md`了解如何使用此repo实现个性化的Arxiv论文推送

See `Usage.md` for instructions on how to personalize the repo. 


Keyword list: ['text-to-speech', 'text to speech', 'tts', 'LLM-based', 'speech', 'voice']


Excluded: []


### Today: 7papers 
#### The ISCSLP 2026 Real-World Audio-Visual Speech Enhancement Challenge
 - **Authors:** Challenge Organizers
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2608.23759

 - **Pdf link:** https://arxiv.org/pdf/2608.23759

 - **Abstract**
 Audio-visual speech enhancement (AVSE) uses visual-speech cues from a target speaker to recover that speaker's speech from noisy or overlapping speech. Many widely used protocols construct mixed signals from separately recorded audio sources and assume reliable video, leaving their performance under natural overlap and visual failure insufficiently characterized. The Real-World AVSE Challenge evaluates two related settings. Track~1 comprises two scenarios: real-world mixtures recorded with two speakers speaking simultaneously, without a corresponding clean reference signal, and synthetic remixes obtained by manually mixing the separately recorded speech of two speakers, with a clean reference signal available; Track~2 reuses audio but pairs it with a degraded target video and contains additional 3-m far-field recordings. The speakers in the development and test sets are disjoint. Evaluation metrics include clean-waveform fidelity, learned quality estimates, transcription accuracy, and speaker identification. In the remix task on the development set, the baseline model achieved an SI-SDR of $-4.069$~dB and an STOI of $0.388$ on Track~1, and an SI-SDR of $-2.851$~dB and an STOI of $0.470$ on Track~2. We release the AV-ConvTasNet checkpoints, the offline evaluator, and the official baseline results on the development and test sets.
#### EmoTra-TTS: Smooth Intra-Utterance Emotion Transitions for Speech Synthesis
 - **Authors:** Tianchi Liu, Zeyang Song, Tianrui Wang, Zhipeng Li, Chenglin Xu, Yiwen Guo
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Artificial Intelligence (cs.AI)
 - **Arxiv link:** https://arxiv.org/abs/2608.23791

 - **Pdf link:** https://arxiv.org/pdf/2608.23791

 - **Abstract**
 Psychological research on emotion dynamics has established that human affect is a continuous, evolving process: emotions rise, decay, and transition within seconds. Current emotional text-to-speech (TTS) systems, however, condition on a single discrete label or static embedding per utterance, fundamentally misaligning with the temporal nature of affect. While recent LLM-based TTS systems may implicitly vary prosody through text understanding, such variation is neither explicitly controllable nor precise enough for targeted intra-utterance transitions. We address three challenges: (1) a multi-pass flow blending pipeline synthesizes frame-aligned transition audio, circumventing the scarcity of natural intra-utterance transitions; (2) dual-stage Valence-Arousal-Dominance (VAD) conditioning guides prosodic planning in the LLM and acoustic realization in the flow decoder via frame-level VAD embeddings; (3) direction-magnitude decoupled injection structurally separates emotion direction from injection magnitude, preventing content degradation. EmoTra-TTS adds only +0.43% parameters with no latency overhead, achieves 30%-87% relative improvement on emotion transition quality, corroborated by 64.4%-79.5% overall win rates in pairwise preference tests against four SOTA baselines and two commercial systems.
#### Preference Optimization for Non-Verbal Vocalization Synthesis
 - **Authors:** Haoyang Li, Chenglin Xu, Junchuan Zhao, Yuang Cao, Liumeng Xue, Yiwen Guo, Eng Siong Chng
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Artificial Intelligence (cs.AI); Machine Learning (cs.LG)
 - **Arxiv link:** https://arxiv.org/abs/2608.24163

 - **Pdf link:** https://arxiv.org/pdf/2608.24163

 - **Abstract**
 Non-verbal vocalizations (NVs), such as laughter, coughs, and sighs, are essential for expressive TTS, but the effectiveness of preference optimization for NV generation remains poorly understood. We systematically study preference optimization for NV-capable TTS, focusing on preference signals, preference-pair construction, and DPO-based optimization objectives. We formulate an NV-aware character error rate (NV-CER) by treating NV tags as distinct output symbols and computing a weighted pinyin-based CER over both verbal and non-verbal content, enabling controllable optimization of NV realization without modifying the underlying optimization algorithm. Experiments on Emilia-NV and the augmented NV-Bench covering 18 NV types reveal how different design choices affect NV realization and lexical fidelity, and establish an effective setup using standard DPO. Objective, LLM-based, and human evaluations provide converging evidence for our findings, offering practical insights into NV-aware post-training for expressive TTS.
#### Visually-Guided Spatial Audio Generation for $360^\circ$ In-the-Wild Speech Scenes
 - **Authors:** Qingyu Luo, Peng Zhang, Wenwu Wang, Philip J. B. Jackson
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2608.24579

 - **Pdf link:** https://arxiv.org/pdf/2608.24579

 - **Abstract**
 Spatial audio is a key component of immersive $360^\circ$ media, yet high-quality spatial capture remains limited in real-world speech-dominant scenes. We study visually guided First-Order Ambisonics (FOA) speech spatialization in the wild: given aligned $360^\circ$ video and an omnidirectional audio track, we recover the missing directional FOA components. To support this task, we introduce YT-SPEECH, a speech-oriented $360^\circ$ video-FOA dataset curated from YouTube. We propose a two-stage Localizer-Renderer framework, where an audio-visual segmentation backbone provides frame-wise spatial heatmaps and a conditional complex-domain U-Net reconstructs directional FOA signals from the omnidirectional channel. A confidence-based gating strategy stabilizes conditioning under ambiguous acoustic conditions. Experiments show improved reconstruction fidelity, spatial accuracy, and perceptual speech quality relative to ablated variants and prior approaches.
#### Investigating voiced and unvoiced regions of speech for audio deepfake detection
 - **Authors:** Ganesh Sivaraman, Hemlata Tak, Elie Khoury
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Signal Processing (eess.SP)
 - **Arxiv link:** https://arxiv.org/abs/2608.24639

 - **Pdf link:** https://arxiv.org/pdf/2608.24639

 - **Abstract**
 Deep neural network based deepfake detection systems have achieved high levels of accuracy on benchmark datasets and competitions. However, most models lack interpretability. It is challenging to extract reasoning from the network that can convince the human evaluator to trust the decision. Humans often rely on acoustic cues like unnatural pitch jitter, robotic intonation, acoustic artifacts, and unnatural sounding fricatives to judge the quality of the synthetic audio. This study explores the role played by the voiced and unvoiced regions of speech in discriminating synthetic from bonafide speech. A measure of signal periodicity is used to analyze speech into voiced and unvoiced components. Then, the graph attention based AASIST detection system is trained independently on each component. This work compares the accuracy of deepfake detection system using voiced and unvoiced components and analyzes the results on the MLAAD dataset. Our results show that unvoiced regions are particularly more effective in distinguishing synthetic (deepfake) speech from bonafide, and achieves an equal error rate of 6.62%. When combined with voice regions through score-level fusion, the overall performance improves further, yielding a 5.82% EER, a relative improvement of 49% over the baseline system that uses the full audio.
#### REDnet: Recursive Encoder and Decoder for Speech Separation under Unknown Number of Speakers and Variable Number of Microphones
 - **Authors:** Fulin Wu, Zhong-Qiu Wang
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2608.24659

 - **Pdf link:** https://arxiv.org/pdf/2608.24659

 - **Abstract**
 We propose $\textit{recursive encoder and decoder}$ (RED) for building a single deep neural network (DNN) model that can separate multi-speaker mixtures containing unknown numbers of speakers and variable numbers of microphones arranged in an unknown geometry, a task that has not been studied yet. The decoder of RED recursively detects whether there are active speakers left and separates one speaker at a time. It is designed to be trained in an end-to-end fashion to improve separation performance. The encoder of RED recursively encodes each microphone channel of the input mixture, sequentially incorporating spatial cues. Combining both, the DNN can be trained to separate mixtures not only with unknown numbers of speakers but also with variable numbers of microphones, achieving state-of-the-art performance on multiple public datasets.
#### Speech-to-SOAP: End-to-End Summarization of Medical Dialogues: KIT@BeTraC 2026
 - **Authors:** Enes Yavuz Ugan, Fabian Retkowski, Yuka Ko, Thai-Binh Nguyen, Maike ZÃ¼fle, Jan Niehues, Alexander Waibel
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2608.24327

 - **Pdf link:** https://arxiv.org/pdf/2608.24327

 - **Abstract**
 With the advent of Large Language Models and its instruction following capabilities a promising application is the task of summarization. Within this domain of task the extractive sub-task of clinical protocolling has emerged as a topic of particular interest as it can significantly reduce the downtime and protocolling burden of health-care workers thus enabling them to focus on their core work helping humans. A further step towards automation is the direct generation of clinical notes from speech without intermediate transcripts, reducing processing time while preserving information such as coughing or other paralinguistic cues that may be lost in transcript-based systems. To this end, we present KIT's submission to this years BeTraC challenge in the lightweight track. Our main contribution is a scalable data augmentation pipeline that unifies heterogeneous medical dialogue datasets through synthetic speech generation and automatically generated SOAP supervision, enabling robust adaptation of a speech foundation model for end-to-end speech-to-SOAP generation.


by Zyzzyva0381 (Windy). 


2026-08-26
