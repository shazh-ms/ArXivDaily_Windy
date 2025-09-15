# Showing new listings for Monday, 15 September 2025
Auto update papers at about 2:30am UTC (10:30am Beijing time) every weekday.


阅读 `Usage.md`了解如何使用此repo实现个性化的Arxiv论文推送

See `Usage.md` for instructions on how to personalize the repo. 


Keyword list: ['text-to-speech', 'text to speech', 'tts', 'LLM-based', 'speech', 'voice']


Excluded: []


### Today: 10papers 
#### The MSP-Podcast Corpus
 - **Authors:** Carlos Busso, Reza Lotfian, Kusha Sridhar, Ali N. Salman, Wei-Cheng Lin, Lucas Goncalves, Srinivas Parthasarathy, Abinay Reddy Naini, Seong-Gyun Leem, Luz Martinez-Lucas, Huang-Cheng Chou, Pravin Mote
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2509.09791

 - **Pdf link:** https://arxiv.org/pdf/2509.09791

 - **Abstract**
 The availability of large, high-quality emotional speech databases is essential for advancing speech emotion recognition (SER) in real-world scenarios. However, many existing databases face limitations in size, emotional balance, and speaker diversity. This study describes the MSP-Podcast corpus, summarizing our ten-year effort. The corpus consists of over 400 hours of diverse audio samples from various audio-sharing websites, all of which have Common Licenses that permit the distribution of the corpus. We annotate the corpus with rich emotional labels, including primary (single dominant emotion) and secondary (multiple emotions perceived in the audio) emotional categories, as well as emotional attributes for valence, arousal, and dominance. At least five raters annotate these emotional labels. The corpus also has speaker identification for most samples, and human transcriptions of the lexical content of the sentences for the entire corpus. The data collection protocol includes a machine learning-driven pipeline for selecting emotionally diverse recordings, ensuring a balanced and varied representation of emotions across speakers and environments. The resulting database provides a comprehensive, high-quality resource, better suited for advancing SER systems in practical, real-world scenarios.
#### Whisper Has an Internal Word Aligner
 - **Authors:** Sung-Lin Yeh, Yen Meng, Hao Tang
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Computation and Language (cs.CL)
 - **Arxiv link:** https://arxiv.org/abs/2509.09987

 - **Pdf link:** https://arxiv.org/pdf/2509.09987

 - **Abstract**
 There is an increasing interest in obtaining accurate word-level timestamps from strong automatic speech recognizers, in particular Whisper. Existing approaches either require additional training or are simply not competitive. The evaluation in prior work is also relatively loose, typically using a tolerance of more than 200 ms. In this work, we discover attention heads in Whisper that capture accurate word alignments and are distinctively different from those that do not. Moreover, we find that using characters produces finer and more accurate alignments than using wordpieces. Based on these findings, we propose an unsupervised approach to extracting word alignments by filtering attention heads while teacher forcing Whisper with characters. Our approach not only does not require training but also produces word alignments that are more accurate than prior work under a stricter tolerance between 20 ms and 100 ms.
#### Unified Learnable 2D Convolutional Feature Extraction for ASR
 - **Authors:** Peter Vieting, Benedikt Hilmes, Ralf Schlüter, Hermann Ney
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Computation and Language (cs.CL); Machine Learning (cs.LG); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2509.10031

 - **Pdf link:** https://arxiv.org/pdf/2509.10031

 - **Abstract**
 Neural front-ends represent a promising approach to feature extraction for automatic speech recognition (ASR) systems as they enable to learn specifically tailored features for different tasks. Yet, many of the existing techniques remain heavily influenced by classical methods. While this inductive bias may ease the system design, our work aims to develop a more generic front-end for feature extraction. Furthermore, we seek to unify the front-end architecture contrasting with existing approaches that apply a composition of several layer topologies originating from different sources. The experiments systematically show how to reduce the influence of existing techniques to achieve a generic front-end. The resulting 2D convolutional front-end is parameter-efficient and suitable for a scenario with limited computational resources unlike large models pre-trained on unlabeled audio. The results demonstrate that this generic unified approach is not only feasible but also matches the performance of existing supervised learnable feature extractors.
#### Towards Data Drift Monitoring for Speech Deepfake Detection in the context of MLOps
 - **Authors:** Xin Wang, Wanying Ge, Junichi Yamagishi
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2509.10086

 - **Pdf link:** https://arxiv.org/pdf/2509.10086

 - **Abstract**
 When being delivered in applications or services on the cloud, static speech deepfake detectors that are not updated will become vulnerable to newly created speech deepfake attacks. From the perspective of machine learning operations (MLOps), this paper tries to answer whether we can monitor new and unseen speech deepfake data that drifts away from a seen reference data set. We further ask, if drift is detected, whether we can fine-tune the detector using similarly drifted data, reduce the drift, and improve the detection performance. On a toy dataset and the large-scale MLAAD dataset, we show that the drift caused by new text-to-speech (TTS) attacks can be monitored using distances between the distributions of the new data and reference data. Furthermore, we demonstrate that fine-tuning the detector using data generated by the new TTS deepfakes can reduce the drift and the detection error rates.
#### Error Analysis in a Modular Meeting Transcription System
 - **Authors:** Peter Vieting, Simon Berger, Thilo von Neumann, Christoph Boeddeker, Ralf Schlüter, Reinhold Haeb-Umbach
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Computation and Language (cs.CL); Machine Learning (cs.LG); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2509.10143

 - **Pdf link:** https://arxiv.org/pdf/2509.10143

 - **Abstract**
 Meeting transcription is a field of high relevance and remarkable progress in recent years. Still, challenges remain that limit its performance. In this work, we extend a previously proposed framework for analyzing leakage in speech separation with proper sensitivity to temporal locality. We show that there is significant leakage to the cross channel in areas where only the primary speaker is active. At the same time, the results demonstrate that this does not affect the final performance much as these leaked parts are largely ignored by the voice activity detection (VAD). Furthermore, different segmentations are compared showing that advanced diarization approaches are able to reduce the gap to oracle segmentation by a third compared to a simple energy-based VAD. We additionally reveal what factors contribute to the remaining difference. The results represent state-of-the-art performance on LibriCSS among systems that train the recognition module on LibriSpeech data only.
#### VStyle: A Benchmark for Voice Style Adaptation with Spoken Instructions
 - **Authors:** Jun Zhan, Mingyang Han, Yuxuan Xie, Chen Wang, Dong Zhang, Kexin Huang, Haoxiang Shi, DongXiao Wang, Tengtao Song, Qinyuan Cheng, Shimin Li, Jun Song, Xipeng Qiu, Bo Zheng
 - **Subjects:** Subjects:
Sound (cs.SD); Artificial Intelligence (cs.AI); Computation and Language (cs.CL); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2509.09716

 - **Pdf link:** https://arxiv.org/pdf/2509.09716

 - **Abstract**
 Spoken language models (SLMs) have emerged as a unified paradigm for speech understanding and generation, enabling natural human machine interaction. However, while most progress has focused on semantic accuracy and instruction following, the ability of SLMs to adapt their speaking style based on spoken instructions has received limited attention. We introduce Voice Style Adaptation (VSA), a new task that examines whether SLMs can modify their speaking style, such as timbre, prosody, or persona following natural language spoken commands. To study this task, we present VStyle, a bilingual (Chinese & English) benchmark covering four categories of speech generation: acoustic attributes, natural language instruction, role play, and implicit empathy. We also introduce the Large Audio Language Model as a Judge (LALM as a Judge) framework, which progressively evaluates outputs along textual faithfulness, style adherence, and naturalness, ensuring reproducible and objective assessment. Experiments on commercial systems and open source SLMs demonstrate that current models face clear limitations in controllable style adaptation, highlighting both the novelty and challenge of this task. By releasing VStyle and its evaluation toolkit, we aim to provide the community with a foundation for advancing human centered spoken interaction. The dataset and code are publicly available at \href{this https URL}{project's homepage}.
#### AI-enabled tuberculosis screening in a high-burden setting using cough sound analysis and speech foundation models
 - **Authors:** Ning Ma, Bahman Mirheidari, Guy J. Brown, Minyoi M. Maimbolwa, Nsala Sanjase, Solomon Chifwamba, Seke Muzazu, Monde Muyoyeta, Mary Kagujje
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2509.09746

 - **Pdf link:** https://arxiv.org/pdf/2509.09746

 - **Abstract**
 Background Artificial intelligence (AI) can detect disease-related acoustic patterns in cough sounds, offering a scalable approach to tuberculosis (TB) screening in high-burden, low-resource settings. Previous studies have been limited by small datasets, under-representation of symptomatic non-TB patients, reliance on simple models, and recordings collected under idealised conditions. Methods We enrolled 512 participants at two hospitals in Zambia, grouped as bacteriologically confirmed TB (TB+), symptomatic patients with other respiratory diseases (OR), and healthy controls (HC). Usable cough recordings plus demographic and clinical data were obtained from 500 participants. Deep learning classifiers based on speech foundation models were trained on cough recordings. The best-performing model, trained on 3-second segments, was further evaluated with demographic and clinical features. Findings The best audio-only classifier achieved an AUROC of 85.2% for distinguishing TB+ from all others (TB+/Rest) and 80.1% for TB+ versus OR. Adding demographic and clinical features improved performance to 92.1% (TB+/Rest) and 84.2% (TB+/OR). At a threshold of 0.38, the multimodal model reached 90.3% sensitivity and 73.1% specificity for TB+/Rest, and 80.6% and 73.1% for TB+/OR. Interpretation Cough analysis using speech foundation models, especially when combined with demographic and clinical data, showed strong potential as a TB triage tool, meeting WHO target product profile benchmarks. The model was robust to confounding factors including background noise, recording time, and device variability, indicating detection of genuine disease-related acoustic patterns. Further validation across diverse regions and case definitions, including subclinical TB, is required before clinical use.
#### DiTReducio: A Training-Free Acceleration for DiT-Based TTS via Progressive Calibration
 - **Authors:** Yanru Huo, Ziyue Jiang, Zuoli Tang, Qingyang Hong, Zhou Zhao
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2509.09748

 - **Pdf link:** https://arxiv.org/pdf/2509.09748

 - **Abstract**
 While Diffusion Transformers (DiT) have advanced non-autoregressive (NAR) speech synthesis, their high computational demands remain an limitation. Existing DiT-based text-to-speech (TTS) model acceleration approaches mainly focus on reducing sampling steps through distillation techniques, yet they remain constrained by training costs. We introduce DiTReducio, a training-free acceleration framework that compresses computations in DiT-based TTS models via progressive calibration. We propose two compression methods, Temporal Skipping and Branch Skipping, to eliminate redundant computations during inference. Moreover, based on two characteristic attention patterns identified within DiT layers, we devise a pattern-guided strategy to selectively apply the compression methods. Our method allows flexible modulation between generation quality and computational efficiency through adjustable compression thresholds. Experimental evaluations conducted on F5-TTS and MegaTTS 3 demonstrate that DiTReducio achieves a 75.4% reduction in FLOPs and improves the Real-Time Factor (RTF) by 37.1%, while preserving generation quality.
#### Combining Textual and Spectral Features for Robust Classification of Pilot Communications
 - **Authors:** Abdullah All Tanvir, Chenyu Huang, Moe Alahmad, Chuyang Yang, Xin Zhong
 - **Subjects:** Subjects:
Sound (cs.SD); Computers and Society (cs.CY); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2509.09752

 - **Pdf link:** https://arxiv.org/pdf/2509.09752

 - **Abstract**
 Accurate estimation of aircraft operations, such as takeoffs and landings, is critical for effective airport management, yet remains challenging, especially at non-towered facilities lacking dedicated surveillance infrastructure. This paper presents a novel dual pipeline machine learning framework that classifies pilot radio communications using both textual and spectral features. Audio data collected from a non-towered U.S. airport was annotated by certified pilots with operational intent labels and preprocessed through automatic speech recognition and Mel-spectrogram extraction. We evaluate a wide range of traditional classifiers and deep learning models, including ensemble methods, LSTM, and CNN across both pipelines. To our knowledge, this is the first system to classify operational aircraft intent using a dual-pipeline ML framework on real-world air traffic audio. Our results demonstrate that spectral features combined with deep architectures consistently yield superior classification performance, with F1-scores exceeding 91%. Data augmentation further improves robustness to real-world audio variability. The proposed approach is scalable, cost-effective, and deployable without additional infrastructure, offering a practical solution for air traffic monitoring at general aviation airports.
#### Prominence-aware automatic speech recognition for conversational speech
 - **Authors:** Julian Linke, Barbara Schuppler
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2509.10116

 - **Pdf link:** https://arxiv.org/pdf/2509.10116

 - **Abstract**
 This paper investigates prominence-aware automatic speech recognition (ASR) by combining prominence detection and speech recognition for conversational Austrian German. First, prominence detectors were developed by fine-tuning wav2vec2 models to classify word-level prominence. The detector was then used to automatically annotate prosodic prominence in a large corpus. Based on those annotations, we trained novel prominence-aware ASR systems that simultaneously transcribe words and their prominence levels. The integration of prominence information did not change performance compared to our baseline ASR system, while reaching a prominence detection accuracy of 85.53% for utterances where the recognized word sequence was correct. This paper shows that transformer-based models can effectively encode prosodic information and represents a novel contribution to prosody-enhanced ASR, with potential applications for linguistic research and prosody-informed dialogue systems.


by Zyzzyva0381 (Windy). 


2025-09-15
