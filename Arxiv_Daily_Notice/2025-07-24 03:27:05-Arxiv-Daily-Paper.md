# Showing new listings for Thursday, 24 July 2025
Auto update papers at about 2:30am UTC (10:30am Beijing time) every weekday.


阅读 `Usage.md`了解如何使用此repo实现个性化的Arxiv论文推送

See `Usage.md` for instructions on how to personalize the repo. 


Keyword list: ['text-to-speech', 'text to speech', 'tts', 'LLM-based', 'speech', 'voice']


Excluded: []


### Today: 13papers 
#### Does Language Matter for Early Detection of Parkinson's Disease from Speech?
 - **Authors:** Peter Plantinga, Briac Cordelle, Dominique Louër, Mirco Ravanelli, Denise Klein
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Machine Learning (cs.LG)
 - **Arxiv link:** https://arxiv.org/abs/2507.16832

 - **Pdf link:** https://arxiv.org/pdf/2507.16832

 - **Abstract**
 Using speech samples as a biomarker is a promising avenue for detecting and monitoring the progression of Parkinson's disease (PD), but there is considerable disagreement in the literature about how best to collect and analyze such data. Early research in detecting PD from speech used a sustained vowel phonation (SVP) task, while some recent research has explored recordings of more cognitively demanding tasks. To assess the role of language in PD detection, we tested pretrained models with varying data types and pretraining objectives and found that (1) text-only models match the performance of vocal-feature models, (2) multilingual Whisper outperforms self-supervised models whereas monolingual Whisper does worse, and (3) AudioSet pretraining improves performance on SVP but not spontaneous speech. These findings together highlight the critical role of language for the early detection of Parkinson's disease.
#### Towards Robust Speech Recognition for Jamaican Patois Music Transcription
 - **Authors:** Jordan Madden, Matthew Stone, Dimitri Johnson, Daniel Geddez
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Artificial Intelligence (cs.AI); Computation and Language (cs.CL)
 - **Arxiv link:** https://arxiv.org/abs/2507.16834

 - **Pdf link:** https://arxiv.org/pdf/2507.16834

 - **Abstract**
 Although Jamaican Patois is a widely spoken language, current speech recognition systems perform poorly on Patois music, producing inaccurate captions that limit accessibility and hinder downstream applications. In this work, we take a data-centric approach to this problem by curating more than 40 hours of manually transcribed Patois music. We use this dataset to fine-tune state-of-the-art automatic speech recognition (ASR) models, and use the results to develop scaling laws for the performance of Whisper models on Jamaican Patois audio. We hope that this work will have a positive impact on the accessibility of Jamaican Patois music and the future of Jamaican Patois language modeling.
#### Evaluating Speech-to-Text x LLM x Text-to-Speech Combinations for AI Interview Systems
 - **Authors:** Nima Yazdani, Ali Ansari, Aruj Mahajan, Amirhossein Afsharrad, Seyed Shahabeddin Mousavi
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Computation and Language (cs.CL)
 - **Arxiv link:** https://arxiv.org/abs/2507.16835

 - **Pdf link:** https://arxiv.org/pdf/2507.16835

 - **Abstract**
 Voice-based conversational AI systems increasingly rely on cascaded architectures combining speech-to-text (STT), large language models (LLMs), and text-to-speech (TTS) components. However, systematic evaluation of different component combinations in production settings remains understudied. We present a large-scale empirical comparison of STT x LLM x TTS stacks using data from over 300,000 AI-conducted job interviews. We develop an automated evaluation framework using LLM-as-a-Judge to assess conversational quality, technical accuracy, and skill assessment capabilities. Our analysis of four production configurations reveals that Google STT paired with GPT-4.1 significantly outperforms alternatives in both conversational and technical quality metrics. Surprisingly, we find that objective quality metrics correlate weakly with user satisfaction scores, suggesting that user experience in voice-based AI systems depends on factors beyond technical performance. Our findings provide practical guidance for selecting components in multimodal conversational AI systems and contribute a validated evaluation methodology for voice-based interactions.
#### From Black Box to Biomarker: Sparse Autoencoders for Interpreting Speech Models of Parkinson's Disease
 - **Authors:** Peter Plantinga, Jen-Kai Chen, Roozbeh Sattari, Mirco Ravanelli, Denise Klein
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Machine Learning (cs.LG)
 - **Arxiv link:** https://arxiv.org/abs/2507.16836

 - **Pdf link:** https://arxiv.org/pdf/2507.16836

 - **Abstract**
 Speech holds promise as a cost-effective and non-invasive biomarker for neurological conditions such as Parkinson's disease (PD). While deep learning systems trained on raw audio can find subtle signals not available from hand-crafted features, their black-box nature hinders clinical adoption. To address this, we apply sparse autoencoders (SAEs) to uncover interpretable internal representations from a speech-based PD detection system. We introduce a novel mask-based activation for adapting SAEs to small biomedical datasets, creating sparse disentangled dictionary representations. These dictionary entries are found to have strong associations with characteristic articulatory deficits in PD speech, such as reduced spectral flux and increased spectral flatness in the low-energy regions highlighted by the model attention. We further show that the spectral flux is related to volumetric measurements of the putamen from MRI scans, demonstrating the potential of SAEs to reveal clinically relevant biomarkers for disease monitoring and diagnosis.
#### Segmentation-free Goodness of Pronunciation
 - **Authors:** Xinwei Cao, Zijian Fan, Torbjørn Svendsen, Giampiero Salvi
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Artificial Intelligence (cs.AI); Computation and Language (cs.CL)
 - **Arxiv link:** https://arxiv.org/abs/2507.16838

 - **Pdf link:** https://arxiv.org/pdf/2507.16838

 - **Abstract**
 Mispronunciation detection and diagnosis (MDD) is a significant part in modern computer aided language learning (CALL) systems. Within MDD, phoneme-level pronunciation assessment is key to helping L2 learners improve their pronunciation. However, most systems are based on a form of goodness of pronunciation (GOP) which requires pre-segmentation of speech into phonetic units. This limits the accuracy of these methods and the possibility to use modern CTC-based acoustic models for their evaluation. In this study, we first propose self-alignment GOP (GOP-SA) that enables the use of CTC-trained ASR models for MDD. Next, we define a more general alignment-free method that takes all possible alignments of the target phoneme into account (GOP-AF). We give a theoretical account of our definition of GOP-AF, an implementation that solves potential numerical issues as well as a proper normalization which makes the method applicable with acoustic models with different peakiness over time. We provide extensive experimental results on the CMU Kids and Speechocean762 datasets comparing the different definitions of our methods, estimating the dependency of GOP-AF on the peakiness of the acoustic models and on the amount of context around the target phoneme. Finally, we compare our methods with recent studies over the Speechocean762 data showing that the feature vectors derived from the proposed method achieve state-of-the-art results on phoneme-level pronunciation assessment.
#### Technical report: Impact of Duration Prediction on Speaker-specific TTS for Indian Languages
 - **Authors:** Isha Pandey, Pranav Gaikwad, Amruta Parulekar, Ganesh Ramakrishnan
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Machine Learning (cs.LG)
 - **Arxiv link:** https://arxiv.org/abs/2507.16875

 - **Pdf link:** https://arxiv.org/pdf/2507.16875

 - **Abstract**
 High-quality speech generation for low-resource languages, such as many Indian languages, remains a significant challenge due to limited data and diverse linguistic structures. Duration prediction is a critical component in many speech generation pipelines, playing a key role in modeling prosody and speech rhythm. While some recent generative approaches choose to omit explicit duration modeling, often at the cost of longer training times. We retain and explore this module to better understand its impact in the linguistically rich and data-scarce landscape of India. We train a non-autoregressive Continuous Normalizing Flow (CNF) based speech model using publicly available Indian language data and evaluate multiple duration prediction strategies for zero-shot, speaker-specific generation. Our comparative analysis on speech-infilling tasks reveals nuanced trade-offs: infilling based predictors improve intelligibility in some languages, while speaker-prompted predictors better preserve speaker characteristics in others. These findings inform the design and selection of duration strategies tailored to specific languages and tasks, underscoring the continued value of interpretable components like duration prediction in adapting advanced generative architectures to low-resource, multilingual settings.
#### SLASH: Self-Supervised Speech Pitch Estimation Leveraging DSP-derived Absolute Pitch
 - **Authors:** Ryo Terashima, Yuma Shirahata, Masaya Kawamura
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2507.17208

 - **Pdf link:** https://arxiv.org/pdf/2507.17208

 - **Abstract**
 We present SLASH, a pitch estimation method of speech signals based on self-supervised learning (SSL). To enhance the performance of conventional SSL-based approaches that primarily depend on the relative pitch difference derived from pitch shifting, our method incorporates absolute pitch values by 1) introducing a prior pitch distribution derived from digital signal processing (DSP), and 2) optimizing absolute pitch through gradient descent with a loss between the target and differentiable DSP-derived spectrograms. To stabilize the optimization, a novel spectrogram generation method is used that skips complicated waveform generation. In addition, the aperiodic components in speech are accurately predicted through differentiable DSP, enhancing the method's applicability to speech signal processing. Experimental results showed that the proposed method outperformed both baseline DSP and SSL-based pitch estimation methods, attributed to the effective integration of SSL and DSP.
#### Accent Normalization Using Self-Supervised Discrete Tokens with Non-Parallel Data
 - **Authors:** Qibing Bai, Sho Inoue, Shuai Wang, Zhongjie Jiang, Yannan Wang, Haizhou Li
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2507.17735

 - **Pdf link:** https://arxiv.org/pdf/2507.17735

 - **Abstract**
 Accent normalization converts foreign-accented speech into native-like speech while preserving speaker identity. We propose a novel pipeline using self-supervised discrete tokens and non-parallel training data. The system extracts tokens from source speech, converts them through a dedicated model, and synthesizes the output using flow matching. Our method demonstrates superior performance over a frame-to-frame baseline in naturalness, accentedness reduction, and timbre preservation across multiple English accents. Through token-level phonetic analysis, we validate the effectiveness of our token-based approach. We also develop two duration preservation methods, suitable for applications such as dubbing.
#### Weak Supervision Techniques towards Enhanced ASR Models in Industry-level CRM Systems
 - **Authors:** Zhongsheng Wang, Sijie Wang, Jia Wang, Yung-I Liang, Yuxi Zhang, Jiamou Liu
 - **Subjects:** Subjects:
Sound (cs.SD); Artificial Intelligence (cs.AI); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2507.16843

 - **Pdf link:** https://arxiv.org/pdf/2507.16843

 - **Abstract**
 In the design of customer relationship management (CRM) systems, accurately identifying customer types and offering personalized services are key to enhancing customer satisfaction and loyalty. However, this process faces the challenge of discerning customer voices and intentions, and general pre-trained automatic speech recognition (ASR) models make it difficult to effectively address industry-specific speech recognition tasks. To address this issue, we innovatively proposed a solution for fine-tuning industry-specific ASR models, which significantly improved the performance of the fine-tuned ASR models in industry applications. Experimental results show that our method substantially improves the crucial auxiliary role of the ASR model in industry CRM systems, and this approach has also been adopted in actual industrial applications.
#### Triple X: A LLM-Based Multilingual Speech Recognition System for the INTERSPEECH2025 MLC-SLM Challenge
 - **Authors:** Miaomiao Gao, Xiaoxiao Xiang, Yiwen Guo
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2507.17288

 - **Pdf link:** https://arxiv.org/pdf/2507.17288

 - **Abstract**
 This paper describes our Triple X speech recognition system submitted to Task 1 of the Multi-Lingual Conversational Speech Language Modeling (MLC-SLM) Challenge. Our work focuses on optimizing speech recognition accuracy in multilingual conversational scenarios through an innovative encoder-adapter-LLM architecture. This framework harnesses the powerful reasoning capabilities of text-based large language models while incorporating domain-specific adaptations. To further enhance multilingual recognition performance, we adopted a meticulously designed multi-stage training strategy leveraging extensive multilingual audio datasets. Experimental results demonstrate that our approach achieves competitive Word Error Rate (WER) performance on both dev and test sets, obtaining second place in the challenge ranking.
#### Seed LiveInterpret 2.0: End-to-end Simultaneous Speech-to-speech Translation with Your Voice
 - **Authors:** Shanbo Cheng, Yu Bao, Zhichao Huang, Yu Lu, Ningxin Peng, Lu Xu, Runsheng Yu, Rong Cao, Ting Han, Zeyang Li, Sitong Liu, Shengtao Ma, Shiguang Pan, Jiongchen Xiao, Nuo Xu, Meng Yang, Rong Ye, Yiming Yu, Ruofei Zhang, Wanyi Zhang, Wenhao Zhu, Liehao Zou, Lu Lu, Yuxuan Wang, Yonghui Wu
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2507.17527

 - **Pdf link:** https://arxiv.org/pdf/2507.17527

 - **Abstract**
 Simultaneous Interpretation (SI) represents one of the most daunting frontiers in the translation industry, with product-level automatic systems long plagued by intractable challenges: subpar transcription and translation quality, lack of real-time speech generation, multi-speaker confusion, and translated speech inflation, especially in long-form discourses. In this study, we introduce Seed-LiveInterpret 2.0, an end-to-end SI model that delivers high-fidelity, ultra-low-latency speech-to-speech generation with voice cloning capabilities. As a fully operational product-level solution, Seed-LiveInterpret 2.0 tackles these challenges head-on through our novel duplex speech-to-speech understanding-generating framework. Experimental results demonstrate that through large-scale pretraining and reinforcement learning, the model achieves a significantly better balance between translation accuracy and latency, validated by human interpreters to exceed 70% correctness in complex scenarios. Notably, Seed-LiveInterpret 2.0 outperforms commercial SI solutions by significant margins in translation quality, while slashing the average latency of cloned speech from nearly 10 seconds to a near-real-time 3 seconds, which is around a near 70% reduction that drastically enhances practical usability.
#### BoSS: Beyond-Semantic Speech
 - **Authors:** Qing Wang, Zehan Li, Hang Lv, Hongjie Chen, Yaodong Song, Jian Kang, Jie Lian, Jie Li, Yongxiang Li, Zhongjiang He, Xuelong Li
 - **Subjects:** Subjects:
Sound (cs.SD); Computation and Language (cs.CL); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2507.17563

 - **Pdf link:** https://arxiv.org/pdf/2507.17563

 - **Abstract**
 Human communication involves more than explicit semantics, with implicit signals and contextual cues playing a critical role in shaping meaning. However, modern speech technologies, such as Automatic Speech Recognition (ASR) and Text-to-Speech (TTS) often fail to capture these beyond-semantic dimensions. To better characterize and benchmark the progression of speech intelligence, we introduce Spoken Interaction System Capability Levels (L1-L5), a hierarchical framework illustrated the evolution of spoken dialogue systems from basic command recognition to human-like social interaction. To support these advanced capabilities, we propose Beyond-Semantic Speech (BoSS), which refers to the set of information in speech communication that encompasses but transcends explicit semantics. It conveys emotions, contexts, and modifies or extends meanings through multidimensional features such as affective cues, contextual dynamics, and implicit semantics, thereby enhancing the understanding of communicative intentions and scenarios. We present a formalized framework for BoSS, leveraging cognitive relevance theories and machine learning models to analyze temporal and contextual speech dynamics. We evaluate BoSS-related attributes across five different dimensions, reveals that current spoken language models (SLMs) are hard to fully interpret beyond-semantic signals. These findings highlight the need for advancing BoSS research to enable richer, more context-aware human-machine communication.
#### Audio-Vision Contrastive Learning for Phonological Class Recognition
 - **Authors:** Daiqi Liu, Tomás Arias-Vergara, Jana Hutter, Andreas Maier, Paula Andrea Pérez-Toro
 - **Subjects:** Subjects:
Sound (cs.SD); Computer Vision and Pattern Recognition (cs.CV); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2507.17682

 - **Pdf link:** https://arxiv.org/pdf/2507.17682

 - **Abstract**
 Accurate classification of articulatory-phonological features plays a vital role in understanding human speech production and developing robust speech technologies, particularly in clinical contexts where targeted phonemic analysis and therapy can improve disease diagnosis accuracy and personalized rehabilitation. In this work, we propose a multimodal deep learning framework that combines real-time magnetic resonance imaging (rtMRI) and speech signals to classify three key articulatory dimensions: manner of articulation, place of articulation, and voicing. We perform classification on 15 phonological classes derived from the aforementioned articulatory dimensions and evaluate the system with four audio/vision configurations: unimodal rtMRI, unimodal audio signals, multimodal middle fusion, and contrastive learning-based audio-vision fusion. Experimental results on the USC-TIMIT dataset show that our contrastive learning-based approach achieves state-of-the-art performance, with an average F1-score of 0.81, representing an absolute increase of 0.23 over the unimodal baseline. The results confirm the effectiveness of contrastive representation learning for multimodal articulatory analysis. Our code and processed dataset will be made publicly available at this https URL to support future research.


by Zyzzyva0381 (Windy). 


2025-07-24
