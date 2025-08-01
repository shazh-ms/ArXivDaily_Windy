# Showing new listings for Friday, 1 August 2025
Auto update papers at about 2:30am UTC (10:30am Beijing time) every weekday.


阅读 `Usage.md`了解如何使用此repo实现个性化的Arxiv论文推送

See `Usage.md` for instructions on how to personalize the repo. 


Keyword list: ['text-to-speech', 'text to speech', 'tts', 'LLM-based', 'speech', 'voice']


Excluded: []


### Today: 8papers 
#### Exploring Dynamic Parameters for Vietnamese Gender-Independent ASR
 - **Authors:** Sotheara Leang (CADT, M-PSI), Éric Castelli (M-PSI), Dominique Vaufreydaz (M-PSI), Sethserey Sam (CADT)
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Computation and Language (cs.CL); Sound (cs.SD); Signal Processing (eess.SP)
 - **Arxiv link:** https://arxiv.org/abs/2507.22964

 - **Pdf link:** https://arxiv.org/pdf/2507.22964

 - **Abstract**
 The dynamic characteristics of speech signal provides temporal information and play an important role in enhancing Automatic Speech Recognition (ASR). In this work, we characterized the acoustic transitions in a ratio plane of Spectral Subband Centroid Frequencies (SSCFs) using polar parameters to capture the dynamic characteristics of the speech and minimize spectral variation. These dynamic parameters were combined with Mel-Frequency Cepstral Coefficients (MFCCs) in Vietnamese ASR to capture more detailed spectral information. The SSCF0 was used as a pseudo-feature for the fundamental frequency (F0) to describe the tonal information robustly. The findings showed that the proposed parameters significantly reduce word error rates and exhibit greater gender independence than the baseline MFCCs.
#### Full-Duplex-Bench v1.5: Evaluating Overlap Handling for Full-Duplex Speech Models
 - **Authors:** Guan-Ting Lin, Shih-Yun Shan Kuan, Qirui Wang, Jiachen Lian, Tingle Li, Hung-yi Lee
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2507.23159

 - **Pdf link:** https://arxiv.org/pdf/2507.23159

 - **Abstract**
 While full-duplex speech agents promise natural, low-latency human--machine interaction by concurrently processing input and output speech, overlap management remains under-evaluated. We introduce Full-Duplex-Bench v1.5, a modular, fully automated benchmark that simulates four overlap scenarios: user interruption, listener backchannel, side conversation, and ambient speech. Our framework supports both open-sourced and commercial models, offering a comprehensive, extensible metric suite -- categorical dialogue behaviors, stop and response latency, prosodic adaptation, and perceived speech quality -- that can be tailored to application-specific criteria. Benchmarking five state-of-the-art agents reveals two principal strategies: repair-first rapid yielding versus continuity-first sustained flow, and highlights scenario-dependent performance trends. The open-sourced design enables seamless extension with new audio assets, languages, and deployment contexts, empowering practitioners to customize and accelerate the evaluation of robust full-duplex speech systems.
#### Feature Importance across Domains for Improving Non-Intrusive Speech Intelligibility Prediction in Hearing Aids
 - **Authors:** Ryandhimas E. Zezario, Sabato M. Siniscalchi, Fei Chen, Hsin-Min Wang, Yu Tsao
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2507.23223

 - **Pdf link:** https://arxiv.org/pdf/2507.23223

 - **Abstract**
 Given the critical role of non-intrusive speech intelligibility assessment in hearing aids (HA), this paper enhances its performance by introducing Feature Importance across Domains (FiDo). We estimate feature importance on spectral and time-domain acoustic features as well as latent representations of Whisper. Importance weights are calculated per frame, and based on these weights, features are projected into new spaces, allowing the model to focus on important areas early. Next, feature concatenation is performed to combine the features before the assessment module processes them. Experimental results show that when FiDo is incorporated into the improved multi-branched speech intelligibility model MBI-Net+, RMSE can be reduced by 7.62% (from 26.10 to 24.11). MBI-Net+ with FiDo also achieves a relative RMSE reduction of 3.98% compared to the best system in the 2023 Clarity Prediction Challenge. These results validate FiDo's effectiveness in enhancing neural speech assessment in HA.
#### CUHK-EE Systems for the vTAD Challenge at NCMMSC 2025
 - **Authors:** Aemon Yat Fei Chiu, Jingyu Li, Yusheng Tian, Guangyan Zhang, Tan Lee
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2507.23266

 - **Pdf link:** https://arxiv.org/pdf/2507.23266

 - **Abstract**
 This paper presents the Voice Timbre Attribute Detection (vTAD) systems developed by the Digital Signal Processing & Speech Technology Laboratory (DSP&STL) of the Department of Electronic Engineering (EE) at The Chinese University of Hong Kong (CUHK) for the 20th National Conference on Human-Computer Speech Communication (NCMMSC 2025) vTAD Challenge. The proposed systems leverage WavLM-Large embeddings with attentive statistical pooling to extract robust speaker representations, followed by two variants of Diff-Net, i.e., Feed-Forward Neural Network (FFN) and Squeeze-and-Excitation-enhanced Residual FFN (SE-ResFFN), to compare timbre attribute intensities between utterance pairs. Experimental results demonstrate that the WavLM-Large+FFN system generalises better to unseen speakers, achieving 77.96% accuracy and 21.79% EER, while the WavLM-Large+SE-ResFFN model excels in the 'Seen' setting with 94.42% accuracy and 5.49% EER. These findings highlight a trade-off between model complexity and generalisation, and underscore the importance of architectural choices in fine-grained speaker modelling. Our analysis also reveals the impact of speaker identity, annotation subjectivity, and data imbalance on system performance, pointing to future directions for improving robustness and fairness in timbre attribute detection.
#### Investigating the Invertibility of Multimodal Latent Spaces: Limitations of Optimization-Based Methods
 - **Authors:** Siwoo Park
 - **Subjects:** Subjects:
Machine Learning (cs.LG); Artificial Intelligence (cs.AI); Computer Vision and Pattern Recognition (cs.CV); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2507.23010

 - **Pdf link:** https://arxiv.org/pdf/2507.23010

 - **Abstract**
 This paper investigates the inverse capabilities and broader utility of multimodal latent spaces within task-specific AI (Artificial Intelligence) models. While these models excel at their designed forward tasks (e.g., text-to-image generation, audio-to-text transcription), their potential for inverse mappings remains largely unexplored. We propose an optimization-based framework to infer input characteristics from desired outputs, applying it bidirectionally across Text-Image (BLIP, Flux.1-dev) and Text-Audio (Whisper-Large-V3, Chatterbox-TTS) modalities. Our central hypothesis posits that while optimization can guide models towards inverse tasks, their multimodal latent spaces will not consistently support semantically meaningful and perceptually coherent inverse mappings. Experimental results consistently validate this hypothesis. We demonstrate that while optimization can force models to produce outputs that align textually with targets (e.g., a text-to-image model generating an image that an image captioning model describes correctly, or an ASR model transcribing optimized audio accurately), the perceptual quality of these inversions is chaotic and incoherent. Furthermore, when attempting to infer the original semantic input from generative models, the reconstructed latent space embeddings frequently lack semantic interpretability, aligning with nonsensical vocabulary tokens. These findings highlight a critical limitation. multimodal latent spaces, primarily optimized for specific forward tasks, do not inherently possess the structure required for robust and interpretable inverse mappings. Our work underscores the need for further research into developing truly semantically rich and invertible multimodal latent spaces.
#### Moravec's Paradox: Towards an Auditory Turing Test
 - **Authors:** David Noever, Forrest McKee
 - **Subjects:** Subjects:
Artificial Intelligence (cs.AI); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2507.23091

 - **Pdf link:** https://arxiv.org/pdf/2507.23091

 - **Abstract**
 This research work demonstrates that current AI systems fail catastrophically on auditory tasks that humans perform effortlessly. Drawing inspiration from Moravec's paradox (i.e., tasks simple for humans often prove difficult for machines, and vice versa), we introduce an auditory Turing test comprising 917 challenges across seven categories: overlapping speech, speech in noise, temporal distortion, spatial audio, coffee-shop noise, phone distortion, and perceptual illusions. Our evaluation of state-of-the-art audio models including GPT-4's audio capabilities and OpenAI's Whisper reveals a striking failure rate exceeding 93%, with even the best-performing model achieving only 6.9% accuracy on tasks that humans solved at 7.5 times higher success (52%). These results expose focusing failures in how AI systems process complex auditory scenes, particularly in selective attention, noise robustness, and contextual adaptation. Our benchmark not only quantifies the human-machine auditory gap but also provides insights into why these failures occur, suggesting that current architectures lack fundamental mechanisms for human-like auditory scene analysis. The traditional design of audio CAPTCHAs highlights common filters that humans evolved but machines fail to select in multimodal language models. This work establishes a diagnostic framework for measuring progress toward human-level machine listening and highlights the need for novel approaches integrating selective attention, physics-based audio understanding, and context-aware perception into multimodal AI systems.
#### Real-time Generation of Various Types of Nodding for Avatar Attentive Listening System
 - **Authors:** Kazushi Kato, Koji Inoue, Divesh Lala, Keiko Ochi, Tatsuya Kawahara
 - **Subjects:** Subjects:
Human-Computer Interaction (cs.HC); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2507.23298

 - **Pdf link:** https://arxiv.org/pdf/2507.23298

 - **Abstract**
 In human dialogue, nonverbal information such as nodding and facial expressions is as crucial as verbal information, and spoken dialogue systems are also expected to express such nonverbal behaviors. We focus on nodding, which is critical in an attentive listening system, and propose a model that predicts both its timing and type in real time. The proposed model builds on the voice activity projection (VAP) model, which predicts voice activity from both listener and speaker audio. We extend it to prediction of various types of nodding in a continuous and real-time manner unlike conventional models. In addition, the proposed model incorporates multi-task learning with verbal backchannel prediction and pretraining on general dialogue data. In the timing and type prediction task, the effectiveness of multi-task learning was significantly demonstrated. We confirmed that reducing the processing rate enables real-time operation without a substantial drop in accuracy, and integrated the model into an avatar attentive listening system. Subjective evaluations showed that it outperformed the conventional method, which always does nodding in sync with verbal backchannel. The code and trained models are available at this https URL.
#### Identifying Hearing Difficulty Moments in Conversational Audio
 - **Authors:** Jack Collins, Adrian Buzea, Chris Collier, Alejandro Ballesta Rosen, Julian Maclaren, Richard F. Lyon, Simon Carlile
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2507.23590

 - **Pdf link:** https://arxiv.org/pdf/2507.23590

 - **Abstract**
 Individuals regularly experience Hearing Difficulty Moments in everyday conversation. Identifying these moments of hearing difficulty has particular significance in the field of hearing assistive technology where timely interventions are key for realtime hearing assistance. In this paper, we propose and compare machine learning solutions for continuously detecting utterances that identify these specific moments in conversational audio. We show that audio language models, through their multimodal reasoning capabilities, excel at this task, significantly outperforming a simple ASR hotword heuristic and a more conventional fine-tuning approach with Wav2Vec, an audio-only input architecture that is state-of-the-art for automatic speech recognition (ASR).


by Zyzzyva0381 (Windy). 


2025-08-01
