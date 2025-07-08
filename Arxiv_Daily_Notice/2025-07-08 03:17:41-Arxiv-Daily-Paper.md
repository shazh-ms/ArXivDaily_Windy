# Showing new listings for Tuesday, 8 July 2025
Auto update papers at about 2:30am UTC (10:30am Beijing time) every weekday.


阅读 `Usage.md`了解如何使用此repo实现个性化的Arxiv论文推送

See `Usage.md` for instructions on how to personalize the repo. 


Keyword list: ['text-to-speech', 'text to speech', 'tts', 'LLM-based', 'speech', 'voice']


Excluded: []


### Today: 25papers 
#### On the Relationship between Accent Strength and Articulatory Features
 - **Authors:** Kevin Huang, Sean Foley, Jihwan Lee, Yoonjeong Lee, Dani Byrd, Shrikanth Narayanan
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Artificial Intelligence (cs.AI); Computation and Language (cs.CL)
 - **Arxiv link:** https://arxiv.org/abs/2507.03149

 - **Pdf link:** https://arxiv.org/pdf/2507.03149

 - **Abstract**
 This paper explores the relationship between accent strength and articulatory features inferred from acoustic speech. To quantify accent strength, we compare phonetic transcriptions with transcriptions based on dictionary-based references, computing phoneme-level difference as a measure of accent strength. The proposed framework leverages recent self-supervised learning articulatory inversion techniques to estimate articulatory features. Analyzing a corpus of read speech from American and British English speakers, this study examines correlations between derived articulatory parameters and accent strength proxies, associating systematic articulatory differences with indexed accent strength. Results indicate that tongue positioning patterns distinguish the two dialects, with notable differences inter-dialects in rhotic and low back vowels. These findings contribute to automated accent analysis and articulatory modeling for speech processing applications.
#### Traceable TTS: Toward Watermark-Free TTS with Strong Traceability
 - **Authors:** Yuxiang Zhao, Yunchong Xiao, Yushen Chen, Zhikang Niu, Shuai Wang, Kai Yu, Xie Chen
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2507.03887

 - **Pdf link:** https://arxiv.org/pdf/2507.03887

 - **Abstract**
 Recent advances in Text-To-Speech (TTS) technology have enabled synthetic speech to mimic human voices with remarkable realism, raising significant security concerns. This underscores the need for traceable TTS models-systems capable of tracing their synthesized speech without compromising quality or security. However, existing methods predominantly rely on explicit watermarking on speech or on vocoder, which degrades speech quality and is vulnerable to spoofing. To address these limitations, we propose a novel framework for model attribution. Instead of embedding watermarks, we train the TTS model and discriminator using a joint training method that significantly improves traceability generalization while preserving-and even slightly improving-audio quality. This is the first work toward watermark-free TTS with strong traceability. To promote progress in related fields, we will release the code upon acceptance of the paper.
#### Prosody Labeling with Phoneme-BERT and Speech Foundation Models
 - **Authors:** Tomoki Koriyama
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2507.03912

 - **Pdf link:** https://arxiv.org/pdf/2507.03912

 - **Abstract**
 This paper proposes a model for automatic prosodic label annotation, where the predicted labels can be used for training a prosody-controllable text-to-speech model. The proposed model utilizes not only rich acoustic features extracted by a self-supervised-learning (SSL)-based model or a Whisper encoder, but also linguistic features obtained from phoneme-input pretrained linguistic foundation models such as PnG BERT and PL-BERT. The concatenation of acoustic and linguistic features is used to predict phoneme-level prosodic labels. In the experimental evaluation on Japanese prosodic labels, including pitch accents and phrase break indices, it was observed that the combination of both speech and linguistic foundation models enhanced the prediction accuracy compared to using either a speech or linguistic input alone. Specifically, we achieved 89.8% prediction accuracy in accent labels, 93.2% in high-low pitch accents, and 94.3% in break indices.
#### MMMOS: Multi-domain Multi-axis Audio Quality Assessment
 - **Authors:** Yi-Cheng Lin, Jia-Hung Chen, Hung-yi Lee
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Artificial Intelligence (cs.AI); Computation and Language (cs.CL)
 - **Arxiv link:** https://arxiv.org/abs/2507.04094

 - **Pdf link:** https://arxiv.org/pdf/2507.04094

 - **Abstract**
 Accurate audio quality estimation is essential for developing and evaluating audio generation, retrieval, and enhancement systems. Existing non-intrusive assessment models predict a single Mean Opinion Score (MOS) for speech, merging diverse perceptual factors and failing to generalize beyond speech. We propose MMMOS, a no-reference, multi-domain audio quality assessment system that estimates four orthogonal axes: Production Quality, Production Complexity, Content Enjoyment, and Content Usefulness across speech, music, and environmental sounds. MMMOS fuses frame-level embeddings from three pretrained encoders (WavLM, MuQ, and M2D) and evaluates three aggregation strategies with four loss functions. By ensembling the top eight models, MMMOS shows a 20-30% reduction in mean squared error and a 4-5% increase in Kendall's {\tau} versus baseline, gains first place in six of eight Production Complexity metrics, and ranks among the top three on 17 of 32 challenge metrics.
#### The Overview of Segmental Durations Modification Algorithms on Speech Signal Characteristics
 - **Authors:** Kyeomeun Jang, Jiaying Li, Yinuo Wang
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Signal Processing (eess.SP)
 - **Arxiv link:** https://arxiv.org/abs/2507.04264

 - **Pdf link:** https://arxiv.org/pdf/2507.04264

 - **Abstract**
 This paper deeply evaluates and analyzes several mainstream algorithms that can arbitrarily modify the duration of any portion of a given speech signal without changing the essential properties (e.g., pitch contour, power spectrum, etc.) of the original signal. Arbitrary modification in this context means that the duration of any region of the signal can be changed by specifying the starting and ending time for modification or the target duration of the specified interval, which can be either a fixed value of duration in the time domain or a scaling factor of the original duration. In addition, arbitrary modification also indicates any number of intervals can be modified at the same time.
#### Long-Context Modeling Networks for Monaural Speech Enhancement: A Comparative Study
 - **Authors:** Qiquan Zhang, Moran Chen, Zeyang Song, Hexin Liu, Xiangyu Zhang, Haizhou Li
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2507.04368

 - **Pdf link:** https://arxiv.org/pdf/2507.04368

 - **Abstract**
 Advanced long-context modeling backbone networks, such as Transformer, Conformer, and Mamba, have demonstrated state-of-the-art performance in speech enhancement. However, a systematic and comprehensive comparative study of these backbones within a unified speech enhancement framework remains lacking. In addition, xLSTM, a more recent and efficient variant of LSTM, has shown promising results in language modeling and as a general-purpose vision backbone. In this paper, we investigate the capability of xLSTM in speech enhancement, and conduct a comprehensive comparison and analysis of the Transformer, Conformer, Mamba, and xLSTM backbones within a unified framework, considering both causal and noncausal configurations. Overall, xLSTM and Mamba achieve better performance than Transformer and Conformer. Mamba demonstrates significantly superior training and inference efficiency, particularly for long speech inputs, whereas xLSTM suffers from the slowest processing speed.
#### Adaptive Slimming for Scalable and Efficient Speech Enhancement
 - **Authors:** Riccardo Miccini, Minje Kim, Clément Laroche, Luca Pezzarossa, Paris Smaragdis
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Machine Learning (cs.LG); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2507.04879

 - **Pdf link:** https://arxiv.org/pdf/2507.04879

 - **Abstract**
 Speech enhancement (SE) enables robust speech recognition, real-time communication, hearing aids, and other applications where speech quality is crucial. However, deploying such systems on resource-constrained devices involves choosing a static trade-off between performance and computational efficiency. In this paper, we introduce dynamic slimming to DEMUCS, a popular SE architecture, making it scalable and input-adaptive. Slimming lets the model operate at different utilization factors (UF), each corresponding to a different performance/efficiency trade-off, effectively mimicking multiple model sizes without the extra storage costs. In addition, a router subnet, trained end-to-end with the backbone, determines the optimal UF for the current input. Thus, the system saves resources by adaptively selecting smaller UFs when additional complexity is unnecessary. We show that our solution is Pareto-optimal against individual UFs, confirming the benefits of dynamic routing. When training the proposed dynamically-slimmable model to use 10% of its capacity on average, we obtain the same or better speech quality as the equivalent static 25% utilization while reducing MACs by 29%.
#### DiceHuBERT: Distilling HuBERT with a Self-Supervised Learning Objective
 - **Authors:** Hyung Gun Chi, Zakaria Aldeneh, Tatiana Likhomanenko, Oggi Rudovic, Takuya Higuchi, Li-Wei Chen, Shinji Watanabe, Ahmed Hussen Abdelaziz
 - **Subjects:** Subjects:
Machine Learning (cs.LG); Artificial Intelligence (cs.AI); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2507.02911

 - **Pdf link:** https://arxiv.org/pdf/2507.02911

 - **Abstract**
 We introduce DiceHuBERT, a knowledge distillation framework for compressing HuBERT, a widely used self-supervised learning (SSL)-based speech foundation model. Unlike existing distillation methods that rely on layer-wise and feature-wise mapping between teacher and student models, DiceHuBERT leverages HuBERT's iterative self-distillation mechanism by directly replacing the original model with a student model. This replacement allows the student to be trained using the same SSL objective used when pre-training HuBERT, eliminating the need for additional modules or architectural constraints. Experimental results on SUPERB show that DiceHuBERT consistently outperforms existing distillation methods, improving phoneme recognition performance by over 21% and ASR performance by more than 14%. Furthermore, DiceHuBERT demonstrates competitive performance across multiple tasks, highlighting its clear advantage.
#### Audio-JEPA: Joint-Embedding Predictive Architecture for Audio Representation Learning
 - **Authors:** Ludovic Tuncay (IRIT-SAMoVA), Etienne Labbé (IRIT-SAMoVA), Emmanouil Benetos (QMUL), Thomas Pellegrini (IRIT-SAMoVA)
 - **Subjects:** Subjects:
Sound (cs.SD); Artificial Intelligence (cs.AI); Machine Learning (cs.LG); Audio and Speech Processing (eess.AS); Signal Processing (eess.SP)
 - **Arxiv link:** https://arxiv.org/abs/2507.02915

 - **Pdf link:** https://arxiv.org/pdf/2507.02915

 - **Abstract**
 Building on the Joint-Embedding Predictive Architecture (JEPA) paradigm, a recent self-supervised learning framework that predicts latent representations of masked regions in high-level feature spaces, we propose Audio-JEPA (Audio Joint-Embedding Predictive Architecture), tailored specifically for audio data. Audio-JEPA uses a simple Vision Transformer backbone to predict latent representations of masked spectrogram patches rather than reconstructing raw audio. We pre-train on unlabeled AudioSet clips (10s, 32kHz) with random patch masking on mel-spectrograms. We evaluate on the X-ARES suite covering speech, music, and environmental sound tasks. Although our implementation is a straightforward translation of the original model to audio, the results still show comparable performance to wav2vec 2.0 and data2vec while using less than one-fifth of their training data and with no hyper-parameter tuning. All code and pretrained checkpoints will be released on GitHub.
#### A Unified Speech LLM for Diarization and Speech Recognition in Multilingual Conversations
 - **Authors:** Phurich Saengthong, Boonnithi Jiaramaneepinit, Sheng Li, Manabu Okumura, Takahiro Shinozaki
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Artificial Intelligence (cs.AI); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2507.02927

 - **Pdf link:** https://arxiv.org/pdf/2507.02927

 - **Abstract**
 Speech Large Language Models (Speech LLMs) have emerged as a crucial paradigm in recent years, extending the capabilities of traditional LLMs to speech tasks such as automatic speech recognition (ASR) and spoken dialogue modeling. However, their effectiveness in real-world multilingual conversations remains limited by the scarcity of data that captures natural conversational phenomena. To address this, the MLC-SLM Challenge provides a multilingual conversational dataset and evaluates models on two tasks: ASR with oracle segmentation (Task I) and joint diarization and recognition without oracle information (Task II). In this paper, we focus on Task II and propose a unified speech LLM that jointly performs diarization and ASR in an end-to-end manner. By reformulating the training data format and modifying the inference procedure, our model addresses the ambiguity inherent in pre-segmented audio and achieves a 54.87\% relative improvement in tcpWER/tcpCER over the baseline, ranking 8th overall, despite using a smaller LLM backbone. We also report results from Task I using a fine-tuned speech LLM.
#### K-Function: Joint Pronunciation Transcription and Feedback for Evaluating Kids Language Function
 - **Authors:** Shuhe Li, Chenxu Guo, Jiachen Lian, Cheol Jun Cho, Wenshuo Zhao, Xuanru Zhou, Dingkun Zhou, Sam Wang, Grace Wang, Jingze Yang, Jingyi Xu, Ruohan Bao, Elise Brenner, Brandon In, Francesca Pei, Maria Luisa Gorno-Tempini, Gopala Anumanchipalli
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Artificial Intelligence (cs.AI); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2507.03043

 - **Pdf link:** https://arxiv.org/pdf/2507.03043

 - **Abstract**
 Early evaluation of children's language is frustrated by the high pitch, long phones, and sparse data that derail automatic speech recognisers. We introduce K-Function, a unified framework that combines accurate sub-word transcription, objective scoring, and actionable feedback. Its core, Kids-WFST, merges a Wav2Vec2 phoneme encoder with a phoneme-similarity Dysfluent-WFST to capture child-specific errors while remaining fully interpretable. Kids-WFST attains 1.39% phoneme error on MyST and 8.61% on Multitudes--absolute gains of 10.47 and 7.06 points over a greedy-search decoder. These high-fidelity transcripts power an LLM that grades verbal skills, milestones, reading, and comprehension, aligning with human proctors and supplying tongue-and-lip visualizations plus targeted advice. The results show that precise phoneme recognition cements a complete diagnostic-feedback loop, paving the way for scalable, clinician-ready language assessment.
#### Toward Efficient Speech Emotion Recognition via Spectral Learning and Attention
 - **Authors:** HyeYoung Lee, Muhammad Nadeem
 - **Subjects:** Subjects:
Sound (cs.SD); Artificial Intelligence (cs.AI); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2507.03251

 - **Pdf link:** https://arxiv.org/pdf/2507.03251

 - **Abstract**
 Speech Emotion Recognition (SER) traditionally relies on auditory data analysis for emotion classification. Several studies have adopted different methods for SER. However, existing SER methods often struggle to capture subtle emotional variations and generalize across diverse datasets. In this article, we use Mel-Frequency Cepstral Coefficients (MFCCs) as spectral features to bridge the gap between computational emotion processing and human auditory perception. To further improve robustness and feature diversity, we propose a novel 1D-CNN-based SER framework that integrates data augmentation techniques. MFCC features extracted from the augmented data are processed using a 1D Convolutional Neural Network (CNN) architecture enhanced with channel and spatial attention mechanisms. These attention modules allow the model to highlight key emotional patterns, enhancing its ability to capture subtle variations in speech signals. The proposed method delivers cutting-edge performance, achieving the accuracy of 97.49% for SAVEE, 99.23% for RAVDESS, 89.31% for CREMA-D, 99.82% for TESS, 99.53% for EMO-DB, and 96.39% for EMOVO. Experimental results show new benchmarks in SER, demonstrating the effectiveness of our approach in recognizing emotional expressions with high precision. Our evaluation demonstrates that the integration of advanced Deep Learning (DL) methods substantially enhances generalization across diverse datasets, underscoring their potential to advance SER for real-world deployment in assistive technologies and human-computer interaction.
#### SHNU Multilingual Conversational Speech Recognition System for INTERSPEECH 2025 MLC-SLM Challenge
 - **Authors:** Yuxiang Mei, Yuang Zheng, Dongxing Xu, Yanhua Long
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2507.03343

 - **Pdf link:** https://arxiv.org/pdf/2507.03343

 - **Abstract**
 This paper describes SHNU multilingual conversational speech recognition system (SHNU-mASR, team name-"maybe"), submitted to Track 1 of the INTERSPEECH 2025 MLC-SLM Challenge. Our system integrates a parallel-speech-encoder architecture with a large language model (LLM) to form a unified multilingual ASR framework. The parallel-speech-encoder consists of two pre-trained encoders, the Whisper-large-v3 encoder and mHuBERT-147 encoder. Their output embeddings are concatenated and fed into the LLM, enabling the model to leverage complementary acoustic and linguistic knowledge and achieve competitive performance. Moreover, we adopt a tri-stage training strategy to jointly update the low-rank adaptation modules and projector parameters of both the speech encoders and the LLM. In addition, we incorporate an additional language-aware prompt at the LLM input to enhance language-specific text generation. The SHNU-mASR system achieves an overall character/word error rate (CER/WER) of 11.76% on the blind evaluation set of the challenge, outperforming the official MLC-SLM baseline by 8.41 absolute CER/WER, without increasing the baseline training data.
#### Eigenvoice Synthesis based on Model Editing for Speaker Generation
 - **Authors:** Masato Murata, Koichi Miyazaki, Tomoki Koriyama, Tomoki Toda
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2507.03377

 - **Pdf link:** https://arxiv.org/pdf/2507.03377

 - **Abstract**
 Speaker generation task aims to create unseen speaker voice without reference speech. The key to the task is defining a speaker space that represents diverse speakers to determine the generated speaker trait. However, the effective way to define this speaker space remains unclear. Eigenvoice synthesis is one of the promising approaches in the traditional parametric synthesis framework, such as HMM-based methods, which define a low-dimensional speaker space using pre-stored speaker features. This study proposes a novel DNN-based eigenvoice synthesis method via model editing. Unlike prior methods, our method defines a speaker space in the DNN model parameter space. By directly sampling new DNN model parameters in this space, we can create diverse speaker voices. Experimental results showed the capability of our method to generate diverse speakers' speech. Moreover, we discovered a gender-dominant axis in the created speaker space, indicating the potential to control speaker attributes.
#### Speaker-agnostic Emotion Vector for Cross-speaker Emotion Intensity Control
 - **Authors:** Masato Murata, Koichi Miyazaki, Tomoki Koriyama
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2507.03382

 - **Pdf link:** https://arxiv.org/pdf/2507.03382

 - **Abstract**
 Cross-speaker emotion intensity control aims to generate emotional speech of a target speaker with desired emotion intensities using only their neutral speech. A recently proposed method, emotion arithmetic, achieves emotion intensity control using a single-speaker emotion vector. Although this prior method has shown promising results in the same-speaker setting, it lost speaker consistency in the cross-speaker setting due to mismatches between the emotion vector of the source and target speakers. To overcome this limitation, we propose a speaker-agnostic emotion vector designed to capture shared emotional expressions across multiple speakers. This speaker-agnostic emotion vector is applicable to arbitrary speakers. Experimental results demonstrate that the proposed method succeeds in cross-speaker emotion intensity control while maintaining speaker consistency, speech quality, and controllability, even in the unseen speaker case.
#### Robust Localization of Partially Fake Speech: Metrics, Models, and Out-of-Domain Evaluation
 - **Authors:** Hieu-Thi Luong, Inbal Rimons, Haim Permuter, Kong Aik Lee, Eng Siong Chng
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2507.03468

 - **Pdf link:** https://arxiv.org/pdf/2507.03468

 - **Abstract**
 Partial audio deepfake localization pose unique challenges and remain underexplored compared to full-utterance spoofing detection. While recent methods report strong in-domain performance, their real-world utility remains unclear. In this analysis, we critically examine the limitations of current evaluation practices, particularly the widespread use of Equal Error Rate (EER), which often obscures generalization and deployment readiness. We propose reframing the localization task as a sequential anomaly detection problem and advocate for the use of threshold-dependent metrics such as accuracy, precision, recall, and F1-score, which better reflect real-world behavior. Specifically, we analyze the performance of the open-source Coarse-to-Fine Proposal Refinement Framework (CFPRF), which achieves a 20-ms EER of 7.61% on the in-domain PartialSpoof evaluation set, but 43.25% and 27.59% on the LlamaPartialSpoof and Half-Truth out-of-domain test sets. Interestingly, our reproduced version of the same model performs worse on in-domain data (9.84%) but better on the out-of-domain sets (41.72% and 14.98%, respectively). This highlights the risks of over-optimizing for in-domain EER, which can lead to models that perform poorly in real-world scenarios. It also suggests that while deep learning models can be effective on in-domain data, they generalize poorly to out-of-domain scenarios, failing to detect novel synthetic samples and misclassifying unfamiliar bona fide audio. Finally, we observe that adding more bona fide or fully synthetic utterances to the training data often degrades performance, whereas adding partially fake utterances improves it.
#### RECA-PD: A Robust Explainable Cross-Attention Method for Speech-based Parkinson's Disease Classification
 - **Authors:** Terry Yi Zhong, Cristian Tejedor-Garcia, Martha Larson, Bastiaan R. Bloem
 - **Subjects:** Subjects:
Sound (cs.SD); Artificial Intelligence (cs.AI); Computation and Language (cs.CL); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2507.03594

 - **Pdf link:** https://arxiv.org/pdf/2507.03594

 - **Abstract**
 Parkinson's Disease (PD) affects over 10 million people globally, with speech impairments often preceding motor symptoms by years, making speech a valuable modality for early, non-invasive detection. While recent deep-learning models achieve high accuracy, they typically lack the explainability required for clinical use. To address this, we propose RECA-PD, a novel, robust, and explainable cross-attention architecture that combines interpretable speech features with self-supervised representations. RECA-PD matches state-of-the-art performance in Speech-based PD detection while providing explanations that are more consistent and more clinically meaningful. Additionally, we demonstrate that performance degradation in certain speech tasks (e.g., monologue) can be mitigated by segmenting long recordings. Our findings indicate that performance and explainability are not necessarily mutually exclusive. Future work will enhance the usability of explanations for non-experts and explore severity estimation to increase the real-world clinical relevance.
#### Improving Low-Resource Dialect Classification Using Retrieval-based Voice Conversion
 - **Authors:** Lea Fischbach, Akbar Karimi, Caroline Kleen, Alfred Lameli, Lucie Flek
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Artificial Intelligence (cs.AI); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2507.03641

 - **Pdf link:** https://arxiv.org/pdf/2507.03641

 - **Abstract**
 Deep learning models for dialect identification are often limited by the scarcity of dialectal data. To address this challenge, we propose to use Retrieval-based Voice Conversion (RVC) as an effective data augmentation method for a low-resource German dialect classification task. By converting audio samples to a uniform target speaker, RVC minimizes speaker-related variability, enabling models to focus on dialect-specific linguistic and phonetic features. Our experiments demonstrate that RVC enhances classification performance when utilized as a standalone augmentation method. Furthermore, combining RVC with other augmentation methods such as frequency masking and segment removal leads to additional performance gains, highlighting its potential for improving dialect classification in low-resource scenarios.
#### CLEP-DG: Contrastive Learning for Speech Emotion Domain Generalization via Soft Prompt Tuning
 - **Authors:** Jiacheng Shi, Yanfu Zhang, Ye Gao
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2507.04048

 - **Pdf link:** https://arxiv.org/pdf/2507.04048

 - **Abstract**
 Speech Emotion Recognition (SER) is fundamental to affective computing and human-computer interaction, yet existing models struggle to generalize across diverse acoustic conditions. While Contrastive Language-Audio Pretraining (CLAP) provides strong multimodal alignment, it lacks dedicated mechanisms for capturing emotional cues, making it suboptimal for SER. To address this, we propose CLEP-DG, a framework that enhances CLAP's robustness in emotion recognition. First, we fine-tune CLAP to obtain CLEP, adapting it on large-scale emotional speech datasets to better encode emotion-relevant features. Then, we introduce Acoustic Context Prompt Tuning (ACPT), a text-driven augmentation strategy that optimizes learnable prompt vectors to model diverse acoustic environments without additional labeled audio. Finally, leveraging cross-modal transferability, we train a classifier on text-derived embeddings and apply it to the audio encoder during inference, mitigating domain shifts between textual supervision and audio-based emotion recognition. Experiments across five benchmark datasets show that CLEP-DG outperforms prior CLAP-based approaches, achieving state-of-the-art performance in both supervised and domain generalization settings.
#### Navigating Speech Recording Collections with AI-Generated Illustrations
 - **Authors:** Sirina Håland, Trond Karlsen Strøm, Petra Galuščáková
 - **Subjects:** Subjects:
Information Retrieval (cs.IR); Computation and Language (cs.CL); Human-Computer Interaction (cs.HC); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2507.04182

 - **Pdf link:** https://arxiv.org/pdf/2507.04182

 - **Abstract**
 Although the amount of available spoken content is steadily increasing, extracting information and knowledge from speech recordings remains challenging. Beyond enhancing traditional information retrieval methods such as speech search and keyword spotting, novel approaches for navigating and searching spoken content need to be explored and developed. In this paper, we propose a novel navigational method for speech archives that leverages recent advances in language and multimodal generative models. We demonstrate our approach with a Web application that organizes data into a structured format using interactive mind maps and image generation tools. The system is implemented using the TED-LIUM~3 dataset, which comprises over 2,000 speech transcripts and audio files of TED Talks. Initial user tests using a System Usability Scale (SUS) questionnaire indicate the application's potential to simplify the exploration of large speech collections.
#### Self-supervised learning of speech representations with Dutch archival data
 - **Authors:** Nik Vaessen, David A. van Leeuwen, Roeland Ordelman
 - **Subjects:** Subjects:
Sound (cs.SD); Machine Learning (cs.LG); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2507.04554

 - **Pdf link:** https://arxiv.org/pdf/2507.04554

 - **Abstract**
 This paper explores the use of Dutch archival television broadcast data for self-supervised learning of speech foundation models, specifically wav2vec 2.0. We first study data quality assumptions for pre-training, and show how music, noise and speaker overlap affect SSL convergence and downstream fine-tuning performance. Secondly, we explore effectively pre-processing strategies to convert the noisy broadcast dataset into a qualitative dataset for pre-training, by using Whisper and WhisperX., Thirdly, we compare mono-lingual and multi-lingual pre-training with equivalent amounts of data, and show that mono-lingual pre-training is more robust to out-of-domain data. Lastly, we achieve a state-of-the-art LARGE wav2vec 2.0 model for the Dutch language, by a continuation of pre-training a wav2vec 2.0 XLS-R model checkpoint with our 55k hour archival dataset.
#### Multi-Step Prediction and Control of Hierarchical Emotion Distribution in Text-to-Speech Synthesis
 - **Authors:** Sho Inoue, Kun Zhou, Shuai Wang, Haizhou Li
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2507.04598

 - **Pdf link:** https://arxiv.org/pdf/2507.04598

 - **Abstract**
 We investigate hierarchical emotion distribution (ED) for achieving multi-level quantitative control of emotion rendering in text-to-speech synthesis (TTS). We introduce a novel multi-step hierarchical ED prediction module that quantifies emotion variance at the utterance, word, and phoneme levels. By predicting emotion variance in a multi-step manner, we leverage global emotional context to refine local emotional variations, thereby capturing the intrinsic hierarchical structure of speech emotion. Our approach is validated through its integration into a variance adaptor and an external module design compatible with various TTS systems. Both objective and subjective evaluations demonstrate that the proposed framework significantly enhances emotional expressiveness and enables precise control of emotion rendering across multiple speech granularities.
#### Fast-VGAN: Lightweight Voice Conversion with Explicit Control of F0 and Duration Parameters
 - **Authors:** Mathilde Abrassart, Nicolas Obin, Axel Roebel
 - **Subjects:** Subjects:
Sound (cs.SD); Artificial Intelligence (cs.AI); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2507.04817

 - **Pdf link:** https://arxiv.org/pdf/2507.04817

 - **Abstract**
 Precise control over speech characteristics, such as pitch, duration, and speech rate, remains a significant challenge in the field of voice conversion. The ability to manipulate parameters like pitch and syllable rate is an important element for effective identity conversion, but can also be used independently for voice transformation, achieving goals that were historically addressed by vocoder-based methods. In this work, we explore a convolutional neural network-based approach that aims to provide means for modifying fundamental frequency (F0), phoneme sequences, intensity, and speaker identity. Rather than relying on disentanglement techniques, our model is explicitly conditioned on these factors to generate mel spectrograms, which are then converted into waveforms using a universal neural vocoder. Accordingly, during inference, F0 contours, phoneme sequences, and speaker embeddings can be freely adjusted, allowing for intuitively controlled voice transformations. We evaluate our approach on speaker conversion and expressive speech tasks using both perceptual and objective metrics. The results suggest that the proposed method offers substantial flexibility, while maintaining high intelligibility and speaker similarity.
#### LAPS-Diff: A Diffusion-Based Framework for Singing Voice Synthesis With Language Aware Prosody-Style Guided Learning
 - **Authors:** Sandipan Dhar, Mayank Gupta, Preeti Rao
 - **Subjects:** Subjects:
Sound (cs.SD); Artificial Intelligence (cs.AI); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2507.04966

 - **Pdf link:** https://arxiv.org/pdf/2507.04966

 - **Abstract**
 The field of Singing Voice Synthesis (SVS) has seen significant advancements in recent years due to the rapid progress of diffusion-based approaches. However, capturing vocal style, genre-specific pitch inflections, and language-dependent characteristics remains challenging, particularly in low-resource scenarios. To address this, we propose LAPS-Diff, a diffusion model integrated with language-aware embeddings and a vocal-style guided learning mechanism, specifically designed for Bollywood Hindi singing style. We curate a Hindi SVS dataset and leverage pre-trained language models to extract word and phone-level embeddings for an enriched lyrics representation. Additionally, we incorporated a style encoder and a pitch extraction model to compute style and pitch losses, capturing features essential to the naturalness and expressiveness of the synthesized singing, particularly in terms of vocal style and pitch variations. Furthermore, we utilize MERT and IndicWav2Vec models to extract musical and contextual embeddings, serving as conditional priors to refine the acoustic feature generation process further. Based on objective and subjective evaluations, we demonstrate that LAPS-Diff significantly improves the quality of the generated samples compared to the considered state-of-the-art (SOTA) model for our constrained dataset that is typical of the low resource scenario.
#### OpenS2S: Advancing Open-Source End-to-End Empathetic Large Speech Language Model
 - **Authors:** Chen Wang, Tianyu Peng, Wen Yang, Yinan Bai, Guangfu Wang, Jun Lin, Lanpeng Jia, Lingxiang Wu, Jinqiao Wang, Chengqing Zong, Jiajun Zhang
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Artificial Intelligence (cs.AI); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2507.05177

 - **Pdf link:** https://arxiv.org/pdf/2507.05177

 - **Abstract**
 Empathetic interaction is a cornerstone of human-machine communication, due to the need for understanding speech enriched with paralinguistic cues and generating emotional and expressive responses. However, the most powerful empathetic LSLMs are increasingly closed off, leaving the crucial details about the architecture, data and development opaque to researchers. Given the critical need for transparent research into the LSLMs and empathetic behavior, we present OpenS2S, a fully open-source, transparent and end-to-end LSLM designed to enable empathetic speech interactions. Based on our empathetic speech-to-text model BLSP-Emo, OpenS2S further employs a streaming interleaved decoding architecture to achieve low-latency speech generation. To facilitate end-to-end training, OpenS2S incorporates an automated data construction pipeline that synthesizes diverse, high-quality empathetic speech dialogues at low cost. By leveraging large language models to generate empathetic content and controllable text-to-speech systems to introduce speaker and emotional variation, we construct a scalable training corpus with rich paralinguistic diversity and minimal human supervision. We release the fully open-source OpenS2S model, including the dataset, model weights, pre-training and fine-tuning codes, to empower the broader research community and accelerate innovation in empathetic speech systems. The project webpage can be accessed at this https URL


by Zyzzyva0381 (Windy). 


2025-07-08
