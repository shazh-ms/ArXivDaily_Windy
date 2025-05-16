# Showing new listings for Friday, 16 May 2025
Auto update papers at about 2:30am UTC (10:30am Beijing time) every weekday.


阅读 `Usage.md`了解如何使用此repo实现个性化的Arxiv论文推送

See `Usage.md` for instructions on how to personalize the repo. 


Keyword list: ['text-to-speech', 'text to speech', 'tts', 'LLM-based', 'speech', 'voice']


Excluded: []


### Today: 5papers 
#### Who Said What WSW 2.0? Enhanced Automated Analysis of Preschool Classroom Speech
 - **Authors:** Anchen Sun, Tiantian Feng, Gabriela Gutierrez, Juan J Londono, Anfeng Xu, Batya Elbaum, Shrikanth Narayanan, Lynn K Perry, Daniel S Messinger
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Machine Learning (cs.LG)
 - **Arxiv link:** https://arxiv.org/abs/2505.09972

 - **Pdf link:** https://arxiv.org/pdf/2505.09972

 - **Abstract**
 This paper introduces an automated framework WSW2.0 for analyzing vocal interactions in preschool classrooms, enhancing both accuracy and scalability through the integration of wav2vec2-based speaker classification and Whisper (large-v2 and large-v3) speech transcription. A total of 235 minutes of audio recordings (160 minutes from 12 children and 75 minutes from 5 teachers), were used to compare system outputs to expert human annotations. WSW2.0 achieves a weighted F1 score of .845, accuracy of .846, and an error-corrected kappa of .672 for speaker classification (child vs. teacher). Transcription quality is moderate to high with word error rates of .119 for teachers and .238 for children. WSW2.0 exhibits relatively high absolute agreement intraclass correlations (ICC) with expert transcriptions for a range of classroom language features. These include teacher and child mean utterance length, lexical diversity, question asking, and responses to questions and other utterances, which show absolute agreement intraclass correlations between .64 and .98. To establish scalability, we apply the framework to an extensive dataset spanning two years and over 1,592 hours of classroom audio recordings, demonstrating the framework's robustness for broad real-world applications. These findings highlight the potential of deep learning and natural language processing techniques to revolutionize educational research by providing accurate measures of key features of preschool classroom speech, ultimately guiding more effective intervention strategies and supporting early childhood language development.
#### Spatially Selective Active Noise Control for Open-fitting Hearables with Acausal Optimization
 - **Authors:** Tong Xiao, Simon Doclo
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Signal Processing (eess.SP); Systems and Control (eess.SY)
 - **Arxiv link:** https://arxiv.org/abs/2505.10372

 - **Pdf link:** https://arxiv.org/pdf/2505.10372

 - **Abstract**
 Recent advances in active noise control have enabled the development of hearables with spatial selectivity, which actively suppress undesired noise while preserving desired sound from specific directions. In this work, we propose an improved approach to spatially selective active noise control that incorporates acausal relative impulse responses into the optimization process, resulting in significantly improved performance over the causal design. We evaluate the system through simulations using a pair of open-fitting hearables with spatially localized speech and noise sources in an anechoic environment. Performance is evaluated in terms of speech distortion, noise reduction, and signal-to-noise ratio improvement across different delays and degrees of acausality. Results show that the proposed acausal optimization consistently outperforms the causal approach across all metrics and scenarios, as acausal filters more effectively characterize the response of the desired source.
#### Quantized Approximate Signal Processing (QASP): Towards Homomorphic Encryption for audio
 - **Authors:** Tu Duyen Nguyen, Adrien Lesage, Clotilde Cantini, Rachid Riad
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Cryptography and Security (cs.CR)
 - **Arxiv link:** https://arxiv.org/abs/2505.10500

 - **Pdf link:** https://arxiv.org/pdf/2505.10500

 - **Abstract**
 Audio and speech data are increasingly used in machine learning applications such as speech recognition, speaker identification, and mental health monitoring. However, the passive collection of this data by audio listening devices raises significant privacy concerns. Fully homomorphic encryption (FHE) offers a promising solution by enabling computations on encrypted data and preserving user privacy. Despite its potential, prior attempts to apply FHE to audio processing have faced challenges, particularly in securely computing time frequency representations, a critical step in many audio tasks. Here, we addressed this gap by introducing a fully secure pipeline that computes, with FHE and quantized neural network operations, four fundamental time-frequency representations: Short-Time Fourier Transform (STFT), Mel filterbanks, Mel-frequency cepstral coefficients (MFCCs), and gammatone filters. Our methods also support the private computation of audio descriptors and convolutional neural network (CNN) classifiers. Besides, we proposed approximate STFT algorithms that lighten computation and bit use for statistical and machine learning analyses. We ran experiments on the VocalSet and OxVoc datasets demonstrating the fully private computation of our approach. We showed significant performance improvements with STFT approximation in private statistical analysis of audio markers, and for vocal exercise classification with CNNs. Our results reveal that our approximations substantially reduce error rates compared to conventional STFT implementations in FHE. We also demonstrated a fully private classification based on the raw audio for gender and vocal exercise classification. Finally, we provided a practical heuristic for parameter selection, making quantized approximate signal processing accessible to researchers and practitioners aiming to protect sensitive audio data.
#### SpecWav-Attack: Leveraging Spectrogram Resizing and Wav2Vec 2.0 for Attacking Anonymized Speech
 - **Authors:** Yuqi Li, Yuanzhong Zheng, Zhongtian Guo, Yaoxuan Wang, Jianjun Yin, Haojun Fei
 - **Subjects:** Subjects:
Sound (cs.SD); Artificial Intelligence (cs.AI); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2505.09616

 - **Pdf link:** https://arxiv.org/pdf/2505.09616

 - **Abstract**
 This paper presents SpecWav-Attack, an adversarial model for detecting speakers in anonymized speech. It leverages Wav2Vec2 for feature extraction and incorporates spectrogram resizing and incremental training for improved performance. Evaluated on librispeech-dev and librispeech-test, SpecWav-Attack outperforms conventional attacks, revealing vulnerabilities in anonymized speech systems and emphasizing the need for stronger defenses, benchmarked against the ICASSP 2025 Attacker Challenge.
#### Introducing voice timbre attribute detection
 - **Authors:** Jinghao He, Zhengyan Sheng, Liping Chen, Kong Aik Lee, Zhen-Hua Ling
 - **Subjects:** Subjects:
Sound (cs.SD); Artificial Intelligence (cs.AI); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2505.09661

 - **Pdf link:** https://arxiv.org/pdf/2505.09661

 - **Abstract**
 This paper focuses on explaining the timbre conveyed by speech signals and introduces a task termed voice timbre attribute detection (vTAD). In this task, voice timbre is explained with a set of sensory attributes describing its human perception. A pair of speech utterances is processed, and their intensity is compared in a designated timbre descriptor. Moreover, a framework is proposed, which is built upon the speaker embeddings extracted from the speech utterances. The investigation is conducted on the VCTK-RVA dataset. Experimental examinations on the ECAPA-TDNN and FACodec speaker encoders demonstrated that: 1) the ECAPA-TDNN speaker encoder was more capable in the seen scenario, where the testing speakers were included in the training set; 2) the FACodec speaker encoder was superior in the unseen scenario, where the testing speakers were not part of the training, indicating enhanced generalization capability. The VCTK-RVA dataset and open-source code are available on the website this https URL.


by Zyzzyva0381 (Windy). 


2025-05-16
