# Showing new listings for Thursday, 21 August 2025
Auto update papers at about 2:30am UTC (10:30am Beijing time) every weekday.


阅读 `Usage.md`了解如何使用此repo实现个性化的Arxiv论文推送

See `Usage.md` for instructions on how to personalize the repo. 


Keyword list: ['text-to-speech', 'text to speech', 'tts', 'LLM-based', 'speech', 'voice']


Excluded: []


### Today: 10papers 
#### RAG-Boost: Retrieval-Augmented Generation Enhanced LLM-based Speech Recognition
 - **Authors:** Pengcheng Wang, Sheng Li, Takahiro Shinozaki
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Computation and Language (cs.CL)
 - **Arxiv link:** https://arxiv.org/abs/2508.14048

 - **Pdf link:** https://arxiv.org/pdf/2508.14048

 - **Abstract**
 In this paper, we propose RAG-Boost (ST-ShinozakiLab Task I system), which enhances the baseline LLM-based ASR system of the MLC-SLM Challenge (task I) with a retrieval-augmented generation (RAG) module on the fly. Each partial ASR hypothesis queries a vector store of audio-text pairs and domain terms, and the retrieved results are fused with the live ASR hypotheses to fix recognition errors. The fused hypotheses are passed to the LLM, yielding improved responses.
#### MahaTTS: A Unified Framework for Multilingual Text-to-Speech Synthesis
 - **Authors:** Jaskaran Singh, Amartya Roy Chowdhury, Raghav Prabhakar, Varshul C. W
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Computation and Language (cs.CL)
 - **Arxiv link:** https://arxiv.org/abs/2508.14049

 - **Pdf link:** https://arxiv.org/pdf/2508.14049

 - **Abstract**
 Current Text-to-Speech models pose a multilingual challenge, where most of the models traditionally focus on English and European languages, thereby hurting the potential to provide access to information to many more people. To address this gap, we introduce MahaTTS-v2 a Multilingual Multi-speaker Text-To-Speech (TTS) system that has excellent multilingual expressive capabilities in Indic languages. The model has been trained on around 20K hours of data specifically focused on Indian languages. Our approach leverages Wav2Vec2.0 tokens for semantic extraction, and a Language Model (LM) for text-to-semantic modeling. Additionally, we have used a Conditional Flow Model (CFM) for semantics to melspectogram generation. The experimental results indicate the effectiveness of the proposed approach over other frameworks. Our code is available at this https URL
#### Towards Low-Latency Tracking of Multiple Speakers With Short-Context Speaker Embeddings
 - **Authors:** Taous Iatariene, Alexandre Guérin, Romain Serizel (MULTISPEECH)
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Artificial Intelligence (cs.AI); Sound (cs.SD); Signal Processing (eess.SP)
 - **Arxiv link:** https://arxiv.org/abs/2508.14115

 - **Pdf link:** https://arxiv.org/pdf/2508.14115

 - **Abstract**
 Speaker embeddings are promising identity-related features that can enhance the identity assignment performance of a tracking system by leveraging its spatial predictions, i.e, by performing identity reassignment. Common speaker embedding extractors usually struggle with short temporal contexts and overlapping speech, which imposes long-term identity reassignment to exploit longer temporal contexts. However, this increases the probability of tracking system errors, which in turn impacts negatively on identity reassignment. To address this, we propose a Knowledge Distillation (KD) based training approach for short context speaker embedding extraction from two speaker mixtures. We leverage the spatial information of the speaker of interest using beamforming to reduce overlap. We study the feasibility of performing identity reassignment over blocks of fixed size, i.e., blockwise identity reassignment, to go towards a low-latency speaker embedding based tracking system. Results demonstrate that our distilled models are effective at short-context embedding extraction and more robust to overlap. Although, blockwise reassignment results indicate that further work is needed to handle simultaneous speech more effectively.
#### EmoSLLM: Parameter-Efficient Adaptation of LLMs for Speech Emotion Recognition
 - **Authors:** Hugo Thimonier, Antony Perzo, Renaud Seguier
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Machine Learning (cs.LG)
 - **Arxiv link:** https://arxiv.org/abs/2508.14130

 - **Pdf link:** https://arxiv.org/pdf/2508.14130

 - **Abstract**
 Emotion recognition from speech is a challenging task that requires capturing both linguistic and paralinguistic cues, with critical applications in human-computer interaction and mental health monitoring. Recent works have highlighted the ability of Large Language Models (LLMs) to perform tasks outside of the sole natural language area. In particular, recent approaches have investigated coupling LLMs with other data modalities by using pre-trained backbones and different fusion mechanisms. This work proposes a novel approach that fine-tunes an LLM with audio and text representations for emotion prediction. Our method first extracts audio features using an audio feature extractor, which are then mapped into the LLM's representation space via a learnable interfacing module. The LLM takes as input (1) the transformed audio features, (2) additional features in the form of natural language (e.g., the transcript), and (3) a textual prompt describing the emotion prediction task. To efficiently adapt the LLM to this multimodal task, we employ Low-Rank Adaptation (LoRA), enabling parameter-efficient fine-tuning. Experimental results on standard emotion recognition benchmarks demonstrate that our model outperforms all but one existing Speech-Text LLMs in the literature, while requiring less than half the parameters of competing approaches. This highlights our approach's effectiveness in integrating multi-modal inputs for speech-based emotion understanding while maintaining significant computational efficiency.
#### A Study of the Scale Invariant Signal to Distortion Ratio in Speech Separation with Noisy References
 - **Authors:** Simon Dahl Jepsen, Mads Græsbøll Christensen, Jesper Rindom Jensen
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Artificial Intelligence (cs.AI); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2508.14623

 - **Pdf link:** https://arxiv.org/pdf/2508.14623

 - **Abstract**
 This paper examines the implications of using the Scale-Invariant Signal-to-Distortion Ratio (SI-SDR) as both evaluation and training objective in supervised speech separation, when the training references contain noise, as is the case with the de facto benchmark WSJ0-2Mix. A derivation of the SI-SDR with noisy references reveals that noise limits the achievable SI-SDR, or leads to undesired noise in the separated outputs. To address this, a method is proposed to enhance references and augment the mixtures with WHAM!, aiming to train models that avoid learning noisy references. Two models trained on these enhanced datasets are evaluated with the non-intrusive NISQA.v2 metric. Results show reduced noise in separated speech but suggest that processing references may introduce artefacts, limiting overall quality gains. Negative correlation is found between SI-SDR and perceived noisiness across models on the WSJ0-2Mix and Libri2Mix test sets, underlining the conclusion from the derivation.
#### Improving Resource-Efficient Speech Enhancement via Neural Differentiable DSP Vocoder Refinement
 - **Authors:** Heitor R. Guimarães, Ke Tan, Juan Azcarreta, Jesus Alvarez, Prabhav Agrawal, Ashutosh Pandey, Buye Xu
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2508.14709

 - **Pdf link:** https://arxiv.org/pdf/2508.14709

 - **Abstract**
 Deploying speech enhancement (SE) systems in wearable devices, such as smart glasses, is challenging due to the limited computational resources on the device. Although deep learning methods have achieved high-quality results, their computational cost limits their feasibility on embedded platforms. This work presents an efficient end-to-end SE framework that leverages a Differentiable Digital Signal Processing (DDSP) vocoder for high-quality speech synthesis. First, a compact neural network predicts enhanced acoustic features from noisy speech: spectral envelope, fundamental frequency (F0), and periodicity. These features are fed into the DDSP vocoder to synthesize the enhanced waveform. The system is trained end-to-end with STFT and adversarial losses, enabling direct optimization at the feature and waveform levels. Experimental results show that our method improves intelligibility and quality by 4% (STOI) and 19% (DNSMOS) over strong baselines without significantly increasing computation, making it well-suited for real-time applications.
#### Long-Context Speech Synthesis with Context-Aware Memory
 - **Authors:** Zhipeng Li, Xiaofen Xing, Jingyuan Xing, Hangrui Hu, Heng Lu, Xiangmin Xu
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2508.14713

 - **Pdf link:** https://arxiv.org/pdf/2508.14713

 - **Abstract**
 In long-text speech synthesis, current approaches typically convert text to speech at the sentence-level and concatenate the results to form pseudo-paragraph-level speech. These methods overlook the contextual coherence of paragraphs, leading to reduced naturalness and inconsistencies in style and timbre across the long-form speech. To address these issues, we propose a Context-Aware Memory (CAM)-based long-context Text-to-Speech (TTS) model. The CAM block integrates and retrieves both long-term memory and local context details, enabling dynamic memory updates and transfers within long paragraphs to guide sentence-level speech synthesis. Furthermore, the prefix mask enhances the in-context learning ability by enabling bidirectional attention on prefix tokens while maintaining unidirectional generation. Experimental results demonstrate that the proposed method outperforms baseline and state-of-the-art long-context methods in terms of prosody expressiveness, coherence and context inference cost across paragraph-level speech.
#### PadAug: Robust Speaker Verification with Simple Waveform-Level Silence Padding
 - **Authors:** Zijun Huang, Chengdong Liang, Jiadi Yao, Xiao-Lei Zhang
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2508.14732

 - **Pdf link:** https://arxiv.org/pdf/2508.14732

 - **Abstract**
 The presence of non-speech segments in utterances often leads to the performance degradation of speaker verification. Existing systems usually use voice activation detection as a preprocessing step to cut off long silence segments. However, short silence segments, particularly those between speech segments, still remain a problem for speaker verification. To address this issue, in this paper, we propose a simple wave-level data augmentation method, \textit{PadAug}, which aims to enhance the system's robustness to silence segments. The core idea of \textit{PadAug} is to concatenate silence segments with speech segments at the waveform level for model training. Due to its simplicity, it can be directly applied to the current state-of-the art architectures. Experimental results demonstrate the effectiveness of the proposed \textit{PadAug}. For example, applying \textit{PadAug} to ResNet34 achieves a relative equal error rate reduction of 5.0\% on the voxceleb dataset. Moreover, the \textit{PadAug} based systems are robust to different lengths and proportions of silence segments in the test data.
#### Systematic FAIRness Assessment of Open Voice Biomarker Datasets for Mental Health and Neurodegenerative Diseases
 - **Authors:** Ishaan Mahapatra, Nihar R. Mahapatra
 - **Subjects:** Subjects:
Sound (cs.SD); Machine Learning (cs.LG); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2508.14089

 - **Pdf link:** https://arxiv.org/pdf/2508.14089

 - **Abstract**
 Voice biomarkers--human-generated acoustic signals such as speech, coughing, and breathing--are promising tools for scalable, non-invasive detection and monitoring of mental health and neurodegenerative diseases. Yet, their clinical adoption remains constrained by inconsistent quality and limited usability of publicly available datasets. To address this gap, we present the first systematic FAIR (Findable, Accessible, Interoperable, Reusable) evaluation of 27 publicly available voice biomarker datasets focused on these disease areas. Using the FAIR Data Maturity Model and a structured, priority-weighted scoring method, we assessed FAIRness at subprinciple, principle, and composite levels. Our analysis revealed consistently high Findability but substantial variability and weaknesses in Accessibility, Interoperability, and Reusability. Mental health datasets exhibited greater variability in FAIR scores, while neurodegenerative datasets were slightly more consistent. Repository choice also significantly influenced FAIRness scores. To enhance dataset quality and clinical utility, we recommend adopting structured, domain-specific metadata standards, prioritizing FAIR-compliant repositories, and routinely applying structured FAIR evaluation frameworks. These findings provide actionable guidance to improve dataset interoperability and reuse, thereby accelerating the clinical translation of voice biomarker technologies.
#### EmoTale: An Enacted Speech-emotion Dataset in Danish
 - **Authors:** Maja J. Hjuler, Harald V. Skat-Rørdam, Line H. Clemmensen, Sneha Das
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2508.14548

 - **Pdf link:** https://arxiv.org/pdf/2508.14548

 - **Abstract**
 While multiple emotional speech corpora exist for commonly spoken languages, there is a lack of functional datasets for smaller (spoken) languages, such as Danish. To our knowledge, Danish Emotional Speech (DES), published in 1997, is the only other database of Danish emotional speech. We present EmoTale; a corpus comprising Danish and English speech recordings with their associated enacted emotion annotations. We demonstrate the validity of the dataset by investigating and presenting its predictive power using speech emotion recognition (SER) models. We develop SER models for EmoTale and the reference datasets using self-supervised speech model (SSLM) embeddings and the openSMILE feature extractor. We find the embeddings superior to the hand-crafted features. The best model achieves an unweighted average recall (UAR) of 64.1% on the EmoTale corpus using leave-one-speaker-out cross-validation, comparable to the performance on DES.


by Zyzzyva0381 (Windy). 


2025-08-21
