# Showing new listings for Tuesday, 10 June 2025
Auto update papers at about 2:30am UTC (10:30am Beijing time) every weekday.


阅读 `Usage.md`了解如何使用此repo实现个性化的Arxiv论文推送

See `Usage.md` for instructions on how to personalize the repo. 


Keyword list: ['text-to-speech', 'text to speech', 'tts', 'LLM-based', 'speech', 'voice']


Excluded: []


### Today: 23papers 
#### AS-ASR: A Lightweight Framework for Aphasia-Specific Automatic Speech Recognition
 - **Authors:** Chen Bao, Chuanbing Huo, Qinyu Chen, Chang Gao
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Artificial Intelligence (cs.AI)
 - **Arxiv link:** https://arxiv.org/abs/2506.06566

 - **Pdf link:** https://arxiv.org/pdf/2506.06566

 - **Abstract**
 This paper proposes AS-ASR, a lightweight aphasia-specific speech recognition framework based on Whisper-tiny, tailored for low-resource deployment on edge devices. Our approach introduces a hybrid training strategy that systematically combines standard and aphasic speech at varying ratios, enabling robust generalization, and a GPT-4-based reference enhancement method that refines noisy aphasic transcripts, improving supervision quality. We conduct extensive experiments across multiple data mixing configurations and evaluation settings. Results show that our fine-tuned model significantly outperforms the zero-shot baseline, reducing WER on aphasic speech by over 30% while preserving performance on standard speech. The proposed framework offers a scalable, efficient solution for real-world disordered speech recognition.
#### Accurate analysis of the pitch pulse-based magnitude/phase structure of natural vowels and assessment of three lightweight time/frequency voicing restoration methods
 - **Authors:** Aníbal J. S. Ferreira, Luis M. T. Jesus, Laurentino M. M. Leal, Jorge E. F. Spratley
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2506.06675

 - **Pdf link:** https://arxiv.org/pdf/2506.06675

 - **Abstract**
 Whispered speech is produced when the vocal folds are not used, either intentionally, or due to a temporary or permanent voice condition. The essential difference between natural speech and whispered speech is that periodic signal components that exist in certain regions of the former, called voiced regions, as a consequence of the vibration of the vocal folds, are missing in the latter. The restoration of natural speech from whispered speech requires delicate signal processing procedures that are especially useful if they can be implemented on low-resourced portable devices, in real-time, and on-the-fly, taking advantage of the established source-filter paradigm of voice production and related models. This paper addresses two challenges that are intertwined and are key in informing and making viable this envisioned technological realization. The first challenge involves characterizing and modeling the evolution of the harmonic phase/magnitude structure of a sequence of individual pitch periods in a voiced region of natural speech comprising sustained or co-articulated vowels. This paper proposes a novel algorithm segmenting individual pitch pulses, which is then used to obtain illustrative results highlighting important differences between sustained and co-articulated vowels, and suggesting practical synthetic voicing approaches. The second challenge involves model-based synthetic voicing. Three implementation alternatives are described that differ in their signal reconstruction approaches: frequency-domain, combined frequency and time-domain, and physiologically-inspired separate filtering of glottal excitation pulses individually generated. The three alternatives are compared objectively using illustrative examples, and subjectively using the results of listening tests involving synthetic voicing of sustained and co-articulated vowels in word context.
#### Exploring Length Generalization For Transformer-based Speech Enhancement
 - **Authors:** Qiquan Zhang, Hongxu Zhu, Xinyuan Qian, Eliathamby Ambikairajah, Haizhou Li
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2506.06697

 - **Pdf link:** https://arxiv.org/pdf/2506.06697

 - **Abstract**
 Transformer network architecture has proven effective in speech enhancement. However, as its core module, self-attention suffers from quadratic complexity, making it infeasible for training on long speech utterances. In practical scenarios, speech enhancement models are often required to perform on noisy speech at run-time that is substantially longer than the training utterances. It remains a challenge how a Transformer-based speech enhancement model can generalize to long speech utterances. In this paper, extensive empirical studies are conducted to explore the model's length generalization ability. In particular, we conduct speech enhancement experiments on four training objectives and evaluate with five metrics. Our studies establish that positional encoding is an effective instrument to dampen the effect of utterance length on speech enhancement. We first explore several existing positional encoding methods, and the results show that relative positional encoding methods exhibit a better length generalization property than absolute positional encoding methods. Additionally, we also explore a simpler and more effective positional encoding scheme, i.e. LearnLin, that uses only one trainable parameter for each attention head to scale the real relative position between time frames, which learns the different preferences on short- or long-term dependencies of these heads. The results demonstrate that our proposal exhibits excellent length generalization ability with comparable or superior performance than other state-of-the-art positional encoding strategies.
#### Rhythm Features for Speaker Identification
 - **Authors:** Nick Mehlman, Thomas Thebaud, Dani Byrd, Shri Narayanan
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2506.06834

 - **Pdf link:** https://arxiv.org/pdf/2506.06834

 - **Abstract**
 While deep learning models have demonstrated robust performance in speaker recognition tasks, they primarily rely on low-level audio features learned empirically from spectrograms or raw waveforms. However, prior work has indicated that idiosyncratic speaking styles heavily influence the temporal structure of linguistic units in speech signals (rhythm). This makes rhythm a strong yet largely overlooked candidate for a speech identity feature. In this paper, we test this hypothesis by applying deep learning methods to perform text-independent speaker identification from rhythm features. Our findings support the usefulness of rhythmic information for speaker recognition tasks but also suggest that high intra-subject variability in ad-hoc speech can degrade its effectiveness.
#### Multi-Distillation from Speech and Music Representation Models
 - **Authors:** Jui-Chiang Wei, Yi-Cheng Lin, Fabian Ritter-Gutierrez, Hung-yi Lee
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2506.07237

 - **Pdf link:** https://arxiv.org/pdf/2506.07237

 - **Abstract**
 Real-world audio often mixes speech and music, yet models typically handle only one domain. This paper introduces a multi-teacher distillation framework that unifies speech and music models into a single one while significantly reducing model size. Our approach leverages the strengths of domain-specific teacher models, such as HuBERT for speech and MERT for music, and explores various strategies to balance both domains. Experiments across diverse tasks demonstrate that our model matches the performance of domain-specific models, showing the effectiveness of cross-domain distillation. Additionally, we conduct few-shot learning experiments, highlighting the need for general models in real-world scenarios where labeled data is limited. Our results show that our model not only performs on par with specialized models but also outperforms them in few-shot scenarios, proving that a cross-domain approach is essential and effective for diverse tasks with limited data.
#### Speaker-Distinguishable CTC: Learning Speaker Distinction Using CTC for Multi-Talker Speech Recognition
 - **Authors:** Asahi Sakuma, Hiroaki Sato, Ryuga Sugano, Tadashi Kumano, Yoshihiko Kawai, Tetsuji Ogawa
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Computation and Language (cs.CL); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2506.07515

 - **Pdf link:** https://arxiv.org/pdf/2506.07515

 - **Abstract**
 This paper presents a novel framework for multi-talker automatic speech recognition without the need for auxiliary information. Serialized Output Training (SOT), a widely used approach, suffers from recognition errors due to speaker assignment failures. Although incorporating auxiliary information, such as token-level timestamps, can improve recognition accuracy, extracting such information from natural conversational speech remains challenging. To address this limitation, we propose Speaker-Distinguishable CTC (SD-CTC), an extension of CTC that jointly assigns a token and its corresponding speaker label to each frame. We further integrate SD-CTC into the SOT framework, enabling the SOT model to learn speaker distinction using only overlapping speech and transcriptions. Experimental comparisons show that multi-task learning with SD-CTC and SOT reduces the error rate of the SOT model by 26% and achieves performance comparable to state-of-the-art methods relying on auxiliary information.
#### Bayesian Learning for Domain-Invariant Speaker Verification and Anti-Spoofing
 - **Authors:** Jin Li, Man-Wai Mak, Johan Rohdin, Kong Aik Lee, Hynek Hermansky
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2506.07536

 - **Pdf link:** https://arxiv.org/pdf/2506.07536

 - **Abstract**
 The performance of automatic speaker verification (ASV) and anti-spoofing drops seriously under real-world domain mismatch conditions. The relaxed instance frequency-wise normalization (RFN), which normalizes the frequency components based on the feature statistics along the time and channel axes, is a promising approach to reducing the domain dependence in the feature maps of a speaker embedding network. We advocate that the different frequencies should receive different weights and that the weights' uncertainty due to domain shift should be accounted for. To these ends, we propose leveraging variational inference to model the posterior distribution of the weights, which results in Bayesian weighted RFN (BWRFN). This approach overcomes the limitations of fixed-weight RFN, making it more effective under domain mismatch conditions. Extensive experiments on cross-dataset ASV, cross-TTS anti-spoofing, and spoofing-robust ASV show that BWRFN is significantly better than WRFN and RFN.
#### Unified Semi-Supervised Pipeline for Automatic Speech Recognition
 - **Authors:** Nune Tadevosyan, Nikolay Karpov, Andrei Andrusenko, Vitaly Lavrukhin, Ante Jukic
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2506.07659

 - **Pdf link:** https://arxiv.org/pdf/2506.07659

 - **Abstract**
 Automatic Speech Recognition has been a longstanding research area, with substantial efforts dedicated to integrating semi-supervised learning due to the scarcity of labeled datasets. However, most prior work has focused on improving learning algorithms using existing datasets, without providing a complete public framework for large-scale semi-supervised training across new datasets or languages. In this work, we introduce a fully open-source semi-supervised training framework encompassing the entire pipeline: from unlabeled data collection to pseudo-labeling and model training. Our approach enables scalable dataset creation for any language using publicly available speech data under Creative Commons licenses. We also propose a novel pseudo-labeling algorithm, TopIPL, and evaluate it in both low-resource (Portuguese, Armenian) and high-resource (Spanish) settings. Notably, TopIPL achieves relative WER improvements of 18-40% for Portuguese, 5-16% for Armenian, and 2-8% for Spanish.
#### TESU-LLM: Training Speech-LLMs Without Speech via Unified Encoder Alignment
 - **Authors:** Taesoo Kim, Jong Hwan Ko
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Artificial Intelligence (cs.AI); Machine Learning (cs.LG); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2506.06343

 - **Pdf link:** https://arxiv.org/pdf/2506.06343

 - **Abstract**
 Recent advances in speech-enabled language models have shown promising results in building intelligent voice assistants. However, most existing approaches rely on large-scale paired speech-text data and extensive computational resources, which pose challenges in terms of scalability and accessibility. In this paper, we present \textbf{TESU-LLM}, a novel framework that enables training speech-capable language models using only text data. Our key insight is to leverage a unified encoder that maps semantically equivalent text and speech inputs to a shared latent space. By aligning the encoder output with the embedding space of a LLM via a lightweight projection network, we enable the model to generalize from text-only supervision to speech-based inference. Despite being trained exclusively on text, TESU-LLM achieves strong performance on various speech-related benchmarks, comparable to baseline methods trained with large-scale multimodal datasets and substantial computational resources. These results highlight the effectiveness and efficiency of our approach, offering a scalable path toward building speech LLMs without speech data.
#### CAtCh: Cognitive Assessment through Cookie Thief
 - **Authors:** Joseph T Colonel, Carolyn Hagler, Guiselle Wismer, Laura Curtis, Jacqueline Becker, Juan Wisnivesky, Alex Federman, Gaurav Pandey
 - **Subjects:** Subjects:
Machine Learning (cs.LG); Artificial Intelligence (cs.AI); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2506.06603

 - **Pdf link:** https://arxiv.org/pdf/2506.06603

 - **Abstract**
 Several machine learning algorithms have been developed for the prediction of Alzheimer's disease and related dementia (ADRD) from spontaneous speech. However, none of these algorithms have been translated for the prediction of broader cognitive impairment (CI), which in some cases is a precursor and risk factor of ADRD. In this paper, we evaluated several speech-based open-source methods originally proposed for the prediction of ADRD, as well as methods from multimodal sentiment analysis for the task of predicting CI from patient audio recordings. Results demonstrated that multimodal methods outperformed unimodal ones for CI prediction, and that acoustics-based approaches performed better than linguistics-based ones. Specifically, interpretable acoustic features relating to affect and prosody were found to significantly outperform BERT-based linguistic features and interpretable linguistic features, respectively. All the code developed for this study is available at this https URL.
#### A Fast and Lightweight Model for Causal Audio-Visual Speech Separation
 - **Authors:** Wendi Sang, Kai Li, Runxuan Yang, Jianqiang Huang, Xiaolin Hu
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2506.06689

 - **Pdf link:** https://arxiv.org/pdf/2506.06689

 - **Abstract**
 Audio-visual speech separation (AVSS) aims to extract a target speech signal from a mixed signal by leveraging both auditory and visual (lip movement) cues. However, most existing AVSS methods exhibit complex architectures and rely on future context, operating offline, which renders them unsuitable for real-time applications. Inspired by the pipeline of RTFSNet, we propose a novel streaming AVSS model, named Swift-Net, which enhances the causal processing capabilities required for real-time applications. Swift-Net adopts a lightweight visual feature extraction module and an efficient fusion module for audio-visual integration. Additionally, Swift-Net employs Grouped SRUs to integrate historical information across different feature spaces, thereby improving the utilization efficiency of historical information. We further propose a causal transformation template to facilitate the conversion of non-causal AVSS models into causal counterparts. Experiments on three standard benchmark datasets (LRS2, LRS3, and VoxCeleb2) demonstrated that under causal conditions, our proposed Swift-Net exhibited outstanding performance, highlighting the potential of this method for processing speech in complex environments.
#### SynHate: Detecting Hate Speech in Synthetic Deepfake Audio
 - **Authors:** Rishabh Ranjan, Kishan Pipariya, Mayank Vatsa, Richa Singh
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2506.06772

 - **Pdf link:** https://arxiv.org/pdf/2506.06772

 - **Abstract**
 The rise of deepfake audio and hate speech, powered by advanced text-to-speech, threatens online safety. We present SynHate, the first multilingual dataset for detecting hate speech in synthetic audio, spanning 37 languages. SynHate uses a novel four-class scheme: Real-normal, Real-hate, Fake-normal, and Fake-hate. Built from MuTox and ADIMA datasets, it captures diverse hate speech patterns globally and in India. We evaluate five leading self-supervised models (Whisper-small/medium, XLS-R, AST, mHuBERT), finding notable performance differences by language, with Whisper-small performing best overall. Cross-dataset generalization remains a challenge. By releasing SynHate and baseline code, we aim to advance robust, culturally sensitive, and multilingual solutions against synthetic hate speech. The dataset is available at this https URL.
#### Beyond Classification: Towards Speech Emotion Reasoning with Multitask AudioLLMs
 - **Authors:** Wenyu Zhang, Yingxu He, Geyu Lin, Zhuohan Liu, Shuo Sun, Bin Wang, Xunlong Zou, Jeremy H. M. Wong, Qiongqiong Wang, Hardik B. Sailor, Nancy F. Chen, Ai Ti Aw
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2506.06820

 - **Pdf link:** https://arxiv.org/pdf/2506.06820

 - **Abstract**
 Audio Large Language Models (AudioLLMs) have achieved strong results in semantic tasks like speech recognition and translation, but remain limited in modeling paralinguistic cues such as emotion. Existing approaches often treat emotion understanding as a classification problem, offering little insight into the underlying rationale behind predictions. In this work, we explore emotion reasoning, a strategy that leverages the generative capabilities of AudioLLMs to enhance emotion recognition by producing semantically aligned, evidence-grounded explanations. To support this in multitask AudioLLMs, we introduce a unified framework combining reasoning-augmented data supervision, dual-encoder architecture, and task-alternating training. This approach enables AudioLLMs to effectively learn different tasks while incorporating emotional reasoning. Experiments on IEMOCAP and MELD show that our approach not only improves emotion prediction accuracy but also enhances the coherence and evidential grounding of the generated responses.
#### Automatic Speech Recognition of African American English: Lexical and Contextual Effects
 - **Authors:** Hamid Mojarad, Kevin Tang
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2506.06888

 - **Pdf link:** https://arxiv.org/pdf/2506.06888

 - **Abstract**
 Automatic Speech Recognition (ASR) models often struggle with the phonetic, phonological, and morphosyntactic features found in African American English (AAE). This study focuses on two key AAE variables: Consonant Cluster Reduction (CCR) and ING-reduction. It examines whether the presence of CCR and ING-reduction increases ASR misrecognition. Subsequently, it investigates whether end-to-end ASR systems without an external Language Model (LM) are more influenced by lexical neighborhood effect and less by contextual predictability compared to systems with an LM. The Corpus of Regional African American Language (CORAAL) was transcribed using wav2vec 2.0 with and without an LM. CCR and ING-reduction were detected using the Montreal Forced Aligner (MFA) with pronunciation expansion. The analysis reveals a small but significant effect of CCR and ING on Word Error Rate (WER) and indicates a stronger presence of lexical neighborhood effect in ASR systems without LMs.
#### "In This Environment, As That Speaker": A Text-Driven Framework for Multi-Attribute Speech Conversion
 - **Authors:** Jiawei Jin, Zhuhan Yang, Yixuan Zhou, Zhiyong Wu
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2506.07036

 - **Pdf link:** https://arxiv.org/pdf/2506.07036

 - **Abstract**
 We propose TES-VC (Text-driven Environment and Speaker controllable Voice Conversion), a text-driven voice conversion framework with independent control of speaker timbre and environmental acoustics. TES-VC processes simultaneous text inputs for target voice and environment, accurately generating speech matching described timbre/environment while preserving source content. Trained on synthetic data with decoupled vocal/environment features via latent diffusion modeling, our method eliminates interference between attributes. The Retrieval-Based Timbre Control (RBTC) module enables precise manipulation using abstract descriptions without paired data. Experiments confirm TES-VC effectively generates contextually appropriate speech in both timbre and environment with high content retention and superior controllability which demonstrates its potential for widespread applications.
#### E-BATS: Efficient Backpropagation-Free Test-Time Adaptation for Speech Foundation Models
 - **Authors:** Jiaheng Dong, Hong Jia, Soumyajit Chatterjee, Abhirup Ghosh, James Bailey, Ting Dang
 - **Subjects:** Subjects:
Machine Learning (cs.LG); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2506.07078

 - **Pdf link:** https://arxiv.org/pdf/2506.07078

 - **Abstract**
 Speech Foundation Models encounter significant performance degradation when deployed in real-world scenarios involving acoustic domain shifts, such as background noise and speaker accents. Test-time adaptation (TTA) has recently emerged as a viable strategy to address such domain shifts at inference time without requiring access to source data or labels. However, existing TTA approaches, particularly those relying on backpropagation, are memory-intensive, limiting their applicability in speech tasks and resource-constrained settings. Although backpropagation-free methods offer improved efficiency, existing ones exhibit poor accuracy. This is because they are predominantly developed for vision tasks, which fundamentally differ from speech task formulations, noise characteristics, and model architecture, posing unique transferability challenges. In this paper, we introduce E-BATS, the first Efficient BAckpropagation-free TTA framework designed explicitly for speech foundation models. E-BATS achieves a balance between adaptation effectiveness and memory efficiency through three key components: (i) lightweight prompt adaptation for a forward-pass-based feature alignment, (ii) a multi-scale loss to capture both global (utterance-level) and local distribution shifts (token-level) and (iii) a test-time exponential moving average mechanism for stable adaptation across utterances. Experiments conducted on four noisy speech datasets spanning sixteen acoustic conditions demonstrate consistent improvements, with 4.1%-13.5% accuracy gains over backpropagation-free baselines and 2.0-6.4 times GPU memory savings compared to backpropagation-based methods. By enabling scalable and robust adaptation under acoustic variability, this work paves the way for developing more efficient adaptation approaches for practical speech processing systems in real-world environments.
#### Streaming Endpointer for Spoken Dialogue using Neural Audio Codecs and Label-Delayed Training
 - **Authors:** Sathvik Udupa, Shinji Watanabe, Petr Schwarz, Jan Cernocky
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2506.07081

 - **Pdf link:** https://arxiv.org/pdf/2506.07081

 - **Abstract**
 Accurate, low-latency endpointing is crucial for effective spoken dialogue systems. While traditional endpointers often rely on spectrum-based audio features, this work proposes real-time speech endpointing for multi-turn dialogues using streaming, low-bitrate Neural Audio Codec (NAC) features, building upon recent advancements in neural audio codecs. To further reduce cutoff errors, we introduce a novel label delay training scheme. At a fixed median latency of 160 ms, our combined NAC and label delay approach achieves significant relative cutoff error reductions: 42.7% for a single-stream endpointer and 37.5% for a two-stream configuration, compared to baseline methods. Finally, we demonstrate efficient integration with a codec-based pretrained speech large language model, improving its median response time by 1200 ms and reducing its cutoff error by 35%.
#### Technical Report: A Practical Guide to Kaldi ASR Optimization
 - **Authors:** Mengze Hong, Di Jiang
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2506.07149

 - **Pdf link:** https://arxiv.org/pdf/2506.07149

 - **Abstract**
 This technical report introduces innovative optimizations for Kaldi-based Automatic Speech Recognition (ASR) systems, focusing on acoustic model enhancement, hyperparameter tuning, and language model efficiency. We developed a custom Conformer block integrated with a multistream TDNN-F structure, enabling superior feature extraction and temporal modeling. Our approach includes advanced data augmentation techniques and dynamic hyperparameter optimization to boost performance and reduce overfitting. Additionally, we propose robust strategies for language model management, employing Bayesian optimization and $n$-gram pruning to ensure relevance and computational efficiency. These systematic improvements significantly elevate ASR accuracy and robustness, outperforming existing methods and offering a scalable solution for diverse speech recognition scenarios. This report underscores the importance of strategic optimizations in maintaining Kaldi's adaptability and competitiveness in rapidly evolving technological landscapes.
#### Towards Generalized Source Tracing for Codec-Based Deepfake Speech
 - **Authors:** Xuanjun Chen, I-Ming Lin, Lin Zhang, Haibin Wu, Hung-yi Lee, Jyh-Shing Roger Jang
 - **Subjects:** Subjects:
Sound (cs.SD); Cryptography and Security (cs.CR); Machine Learning (cs.LG); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2506.07294

 - **Pdf link:** https://arxiv.org/pdf/2506.07294

 - **Abstract**
 Recent attempts at source tracing for codec-based deepfake speech (CodecFake), generated by neural audio codec-based speech generation (CoSG) models, have exhibited suboptimal performance. However, how to train source tracing models using simulated CoSG data while maintaining strong performance on real CoSG-generated audio remains an open challenge. In this paper, we show that models trained solely on codec-resynthesized data tend to overfit to non-speech regions and struggle to generalize to unseen content. To mitigate these challenges, we introduce the Semantic-Acoustic Source Tracing Network (SASTNet), which jointly leverages Whisper for semantic feature encoding and Wav2vec2 with AudioMAE for acoustic feature encoding. Our proposed SASTNet achieves state-of-the-art performance on the CoSG test set of the CodecFake+ dataset, demonstrating its effectiveness for reliable source tracing.
#### Speech Recognition on TV Series with Video-guided Post-Correction
 - **Authors:** Haoyuan Yang, Yue Zhang, Liqiang Jing
 - **Subjects:** Subjects:
Sound (cs.SD); Artificial Intelligence (cs.AI); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2506.07323

 - **Pdf link:** https://arxiv.org/pdf/2506.07323

 - **Abstract**
 Automatic Speech Recognition (ASR) has achieved remarkable success with deep learning, driving advancements in conversational artificial intelligence, media transcription, and assistive technologies. However, ASR systems still struggle in complex environments such as TV series, where overlapping speech, domain-specific terminology, and long-range contextual dependencies pose significant challenges to transcription accuracy. Existing multimodal approaches fail to correct ASR outputs with the rich temporal and contextual information available in video. To address this limitation, we propose a novel multimodal post-correction framework that refines ASR transcriptions by leveraging contextual cues extracted from video. Our framework consists of two stages: ASR Generation and Video-based Post-Correction, where the first stage produces the initial transcript and the second stage corrects errors using Video-based Contextual Information Extraction and Context-aware ASR Correction. We employ the Video-Large Multimodal Model (VLMM) to extract key contextual information using tailored prompts, which is then integrated with a Large Language Model (LLM) to refine the ASR output. We evaluate our method on a multimodal benchmark for TV series ASR and demonstrate its effectiveness in improving ASR performance by leveraging video-based context to enhance transcription accuracy in complex multimedia environments.
#### Towards Energy-Efficient and Low-Latency Voice-Controlled Smart Homes: A Proposal for Offline Speech Recognition and IoT Integration
 - **Authors:** Peng Huang, Imdad Ullah, Xiaotong Wei, Tariq Ahamed Ahanger, Najm Hassan, Zawar Hussain Shah
 - **Subjects:** Subjects:
Sound (cs.SD); Computers and Society (cs.CY); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2506.07494

 - **Pdf link:** https://arxiv.org/pdf/2506.07494

 - **Abstract**
 The smart home systems, based on AI speech recognition and IoT technology, enable people to control devices through verbal commands and make people's lives more efficient. However, existing AI speech recognition services are primarily deployed on cloud platforms on the Internet. When users issue a command, speech recognition devices like ``Amazon Echo'' will post a recording through numerous network nodes, reach multiple servers, and then receive responses through the Internet. This mechanism presents several issues, including unnecessary energy consumption, communication latency, and the risk of a single-point failure. In this position paper, we propose a smart home concept based on offline speech recognition and IoT technology: 1) integrating offline keyword spotting (KWS) technologies into household appliances with limited resource hardware to enable them to understand user voice commands; 2) designing a local IoT network with decentralized architecture to manage and connect various devices, enhancing the robustness and scalability of the system. This proposal of a smart home based on offline speech recognition and IoT technology will allow users to use low-latency voice control anywhere in the home without depending on the Internet and provide better scalability and energy sustainability.
#### Generative Voice Bursts during Phone Call
 - **Authors:** Paritosh Ranjan, Surajit Majumder, Prodip Roy
 - **Subjects:** Subjects:
Sound (cs.SD); Neural and Evolutionary Computing (cs.NE); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2506.07526

 - **Pdf link:** https://arxiv.org/pdf/2506.07526

 - **Abstract**
 In critical situations, conventional mobile telephony fails to convey emergency voice messages to a callee already engaged in another call. The standard call waiting alert does not provide the urgency or content of the waiting call. This paper proposes a novel method for transmitting Generative Voice Bursts short, context aware audio messages during ongoing calls, from either preauthorized or dynamically prioritized callers. By leveraging generative AI techniques, the system automatically generates spoken messages from contextual inputs example like location, health data, images, background noise when the caller is unable to speak due to incapacitation or environmental constraints. The solution incorporates voice, text, and priority inference mechanisms, allowing high priority emergency messages to bypass conventional call waiting barriers. The approach employs models such as GPT Neo for generative text, which is synthesized into audio and delivered in configurable intervals G seconds and counts N times, ensuring minimal disruption while preserving urgency. This method holds potential for significant impact across telecom, mobile device manufacturing, and emergency communication platforms.
#### Transcript-Prompted Whisper with Dictionary-Enhanced Decoding for Japanese Speech Annotation
 - **Authors:** Rui Hu, Xiaolong Lin, Jiawang Liu, Shixi Huang, Zhenpeng Zhan
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2506.07646

 - **Pdf link:** https://arxiv.org/pdf/2506.07646

 - **Abstract**
 In this paper, we propose a method for annotating phonemic and prosodic labels on a given audio-transcript pair, aimed at constructing Japanese text-to-speech (TTS) datasets. Our approach involves fine-tuning a large-scale pre-trained automatic speech recognition (ASR) model, conditioned on ground truth transcripts, to simultaneously output phrase-level graphemes and annotation labels. To further correct errors in phonemic labeling, we employ a decoding strategy that utilizes dictionary prior knowledge. The objective evaluation results demonstrate that our proposed method outperforms previous approaches relying solely on text or audio. The subjective evaluation results indicate that the naturalness of speech synthesized by the TTS model, trained with labels annotated using our method, is comparable to that of a model trained with manual annotations.


by Zyzzyva0381 (Windy). 


2025-06-10
