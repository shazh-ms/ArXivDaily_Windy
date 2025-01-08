# Showing new listings for Wednesday, 8 January 2025
Auto update papers at about 2:30am UTC (10:30am Beijing time) every weekday.


阅读 `Usage.md`了解如何使用此repo实现个性化的Arxiv论文推送

See `Usage.md` for instructions on how to personalize the repo. 


Keyword list: ['acoustic', 'reinforcement learning', 'reverb', 'meta', 'separate', 'reconstruction', 'noise', 'enhance', 'localization', 'speech']


Excluded: []


### Today: 13papers 
#### Breaking Through the Spike: Spike Window Decoding for Accelerated and Precise Automatic Speech Recognition
 - **Authors:** Wei Zhang, Tian-Hao Zhang, Chao Luo, Hui Zhou, Chao Yang, Xinyuan Qian, Xu-Cheng Yin
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Artificial Intelligence (cs.AI); Computation and Language (cs.CL); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2501.03257

 - **Pdf link:** https://arxiv.org/pdf/2501.03257

 - **Abstract**
 Recently, end-to-end automatic speech recognition has become the mainstream approach in both industry and academia. To optimize system performance in specific scenarios, the Weighted Finite-State Transducer (WFST) is extensively used to integrate acoustic and language models, leveraging its capacity to implicitly fuse language models within static graphs, thereby ensuring robust recognition while also facilitating rapid error correction. However, WFST necessitates a frame-by-frame search of CTC posterior probabilities through autoregression, which significantly hampers inference speed. In this work, we thoroughly investigate the spike property of CTC outputs and further propose the conjecture that adjacent frames to non-blank spikes carry semantic information beneficial to the model. Building on this, we propose the Spike Window Decoding algorithm, which greatly improves the inference speed by making the number of frames decoded in WFST linearly related to the number of spiking frames in the CTC output, while guaranteeing the recognition performance. Our method achieves SOTA recognition accuracy with significantly accelerates decoding speed, proven across both AISHELL-1 and large-scale In-House datasets, establishing a pioneering approach for integrating CTC output with WFST.
#### Deep Learning for Pathological Speech: A Survey
 - **Authors:** Shakeel A. Sheikh, Md. Sahidullah, Ina Kodrasi
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2501.03536

 - **Pdf link:** https://arxiv.org/pdf/2501.03536

 - **Abstract**
 Advancements in spoken language technologies for neurodegenerative speech disorders are crucial for meeting both clinical and technological needs. This overview paper is vital for advancing the field, as it presents a comprehensive review of state-of-the-art methods in pathological speech detection, automatic speech recognition, pathological speech intelligibility enhancement, intelligibility and severity assessment, and data augmentation approaches for pathological speech. It also high-lights key challenges, such as ensuring robustness, privacy, and interpretability. The paper concludes by exploring promising future directions, including the adoption of multimodal approaches and the integration of graph neural networks and large language models to further advance speech technology for neurodegenerative speech disorders
#### Towards a Generalizable Speech Marker for Parkinson's Disease Diagnosis
 - **Authors:** Maksim Siniukov, Ellie Xing, Sanaz, Attaripour Isfahani, Mohammad Soleymani
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2501.03581

 - **Pdf link:** https://arxiv.org/pdf/2501.03581

 - **Abstract**
 Parkinson's Disease (PD) is a neurodegenerative disorder characterized by motor symptoms, including altered voice production in the early stages. Early diagnosis is crucial not only to improve PD patients' quality of life but also to enhance the efficacy of potential disease-modifying therapies during early neurodegeneration, a window often missed by current diagnostic tools. In this paper, we propose a more generalizable approach to PD recognition through domain adaptation and self-supervised learning. We demonstrate the generalization capabilities of the proposed approach across diverse datasets in different languages. Our approach leverages HuBERT, a large deep neural network originally trained for speech recognition and further trains it on unlabeled speech data from a population that is similar to the target group, i.e., the elderly, in a self-supervised manner. The model is then fine-tuned and adapted for use across different datasets in multiple languages, including English, Italian, and Spanish. Evaluations on four publicly available PD datasets demonstrate the model's efficacy, achieving an average specificity of 92.1% and an average sensitivity of 91.2%. This method offers objective and consistent evaluations across large populations, addressing the variability inherent in human assessments and providing a non-invasive, cost-effective and accessible diagnostic option.
#### Universal Speaker Embedding Free Target Speaker Extraction and Personal Voice Activity Detection
 - **Authors:** Bang Zeng, Ming Li
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2501.03612

 - **Pdf link:** https://arxiv.org/pdf/2501.03612

 - **Abstract**
 Determining 'who spoke what and when' remains challenging in real-world applications. In typical scenarios, Speaker Diarization (SD) is employed to address the problem of 'who spoke when,' while Target Speaker Extraction (TSE) or Target Speaker Automatic Speech Recognition (TSASR) techniques are utilized to resolve the issue of 'who spoke what.' Although some works have achieved promising results by combining SD and TSE systems, inconsistencies remain between SD and TSE regarding both output inconsistency and scenario mismatch. To address these limitations, we propose a Universal Speaker Embedding Free Target Speaker Extraction and Personal Voice Activity Detection (USEF-TP) model that jointly performs TSE and Personal Voice Activity Detection (PVAD). USEF-TP leverages frame-level features obtained through a cross-attention mechanism as speaker-related features instead of using speaker embeddings as in traditional approaches. Additionally, a multi-task learning algorithm with a scenario-aware differentiated loss function is applied to ensure robust performance across various levels of speaker overlap. The experimental results show that our proposed USEF-TP model achieves superior performance in TSE and PVAD tasks on the LibriMix and SparseLibriMix datasets.
#### Detecting Neurocognitive Disorders through Analyses of Topic Evolution and Cross-modal Consistency in Visual-Stimulated Narratives
 - **Authors:** Jinchao Li, Yuejiao Wang, Junan Li, Jiawen Kang, Bo Zheng, Simon Wong, Brian Mak, Helene Fung, Jean Woo, Man-Wai Mak, Timothy Kwok, Vincent Mok, Xianmin Gong, Xixin Wu, Xunying Liu, Patrick Wong, Helen Meng
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Machine Learning (cs.LG)
 - **Arxiv link:** https://arxiv.org/abs/2501.03727

 - **Pdf link:** https://arxiv.org/pdf/2501.03727

 - **Abstract**
 Early detection of neurocognitive disorders (NCDs) is crucial for timely intervention and disease management. Speech analysis offers a non-intrusive and scalable screening method, particularly through narrative tasks in neuropsychological assessment tools. Traditional narrative analysis often focuses on local indicators in microstructure, such as word usage and syntax. While these features provide insights into language production abilities, they often fail to capture global narrative patterns, or microstructures. Macrostructures include coherence, thematic organization, and logical progressions, reflecting essential cognitive skills potentially critical for recognizing NCDs. Addressing this gap, we propose to investigate specific cognitive and linguistic challenges by analyzing topical shifts, temporal dynamics, and the coherence of narratives over time, aiming to reveal cognitive deficits by identifying narrative impairments, and exploring their impact on communication and cognition. The investigation is based on the CU-MARVEL Rabbit Story corpus, which comprises recordings of a story-telling task from 758 older adults. We developed two approaches: the Dynamic Topic Models (DTM)-based temporal analysis to examine the evolution of topics over time, and the Text-Image Temporal Alignment Network (TITAN) to evaluate the coherence between spoken narratives and visual stimuli. DTM-based approach validated the effectiveness of dynamic topic consistency as a macrostructural metric (F1=0.61, AUC=0.78). The TITAN approach achieved the highest performance (F1=0.72, AUC=0.81), surpassing established microstructural and macrostructural feature sets. Cross-comparison and regression tasks further demonstrated the effectiveness of proposed dynamic macrostructural modeling approaches for NCD detection.
#### Pseudo Strong Labels from Frame-Level Predictions for Weakly Supervised Sound Event Detection
 - **Authors:** Yuliang Zhang, Defeng (David)Huang, Roberto Togneri
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2501.03740

 - **Pdf link:** https://arxiv.org/pdf/2501.03740

 - **Abstract**
 Weakly Supervised Sound Event Detection (WSSED), which relies on audio tags without precise onset and offset times, has become prevalent due to the scarcity of strongly labeled data that includes exact temporal boundaries for events. This study introduces Frame-level Pseudo Strong Labeling (FPSL) to overcome the lack of temporal information in WSSED by generating pseudo strong labels from frame-level predictions. This enhances temporal localization during training and addresses the limitations of clip-wise weak supervision. We validate our approach across three benchmark datasets (DCASE2017 Task 4, DCASE2018 Task 4, and UrbanSED) and demonstrate significant improvements in key metrics such as the Polyphonic Sound Detection Scores (PSDS), event-based F1 scores, and intersection-based F1 scores. For example, Convolutional Recurrent Neural Networks (CRNNs) trained with FPSL outperform baseline models by 4.9% in PSDS1 on DCASE2017, 7.6% on DCASE2018, and 1.8% on UrbanSED, confirming the effectiveness of our method in enhancing model performance.
#### Spectral-Aware Low-Rank Adaptation for Speaker Verification
 - **Authors:** Zhe Li, Man-wai Mak, Mert Pilanci, Hung-yi Lee, Helen Meng
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2501.03829

 - **Pdf link:** https://arxiv.org/pdf/2501.03829

 - **Abstract**
 Previous research has shown that the principal singular vectors of a pre-trained model's weight matrices capture critical knowledge. In contrast, those associated with small singular values may contain noise or less reliable information. As a result, the LoRA-based parameter-efficient fine-tuning (PEFT) approach, which does not constrain the use of the spectral space, may not be effective for tasks that demand high representation capacity. In this study, we enhance existing PEFT techniques by incorporating the spectral information of pre-trained weight matrices into the fine-tuning process. We investigate spectral adaptation strategies with a particular focus on the additive adjustment of top singular vectors. This is accomplished by applying singular value decomposition (SVD) to the pre-trained weight matrices and restricting the fine-tuning within the top spectral space. Extensive speaker verification experiments on VoxCeleb1 and CN-Celeb1 demonstrate enhanced tuning performance with the proposed approach. Code is released at this https URL.
#### LHGNN: Local-Higher Order Graph Neural Networks For Audio Classification and Tagging
 - **Authors:** Shubhr Singh, Emmanouil Benetos, Huy Phan, Dan Stowell
 - **Subjects:** Subjects:
Sound (cs.SD); Artificial Intelligence (cs.AI); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2501.03464

 - **Pdf link:** https://arxiv.org/pdf/2501.03464

 - **Abstract**
 Transformers have set new benchmarks in audio processing tasks, leveraging self-attention mechanisms to capture complex patterns and dependencies within audio data. However, their focus on pairwise interactions limits their ability to process the higher-order relations essential for identifying distinct audio objects. To address this limitation, this work introduces the Local- Higher Order Graph Neural Network (LHGNN), a graph based model that enhances feature understanding by integrating local neighbourhood information with higher-order data from Fuzzy C-Means clusters, thereby capturing a broader spectrum of audio relationships. Evaluation of the model on three publicly available audio datasets shows that it outperforms Transformer-based models across all benchmarks while operating with substantially fewer parameters. Moreover, LHGNN demonstrates a distinct advantage in scenarios lacking ImageNet pretraining, establishing its effectiveness and efficiency in environments where extensive pretraining data is unavailable.
#### Effective and Efficient Mixed Precision Quantization of Speech Foundation Models
 - **Authors:** Haoning Xu, Zhaoqing Li, Zengrui Jin, Huimeng Wang, Youjun Chen, Guinan Li, Mengzhe Geng, Shujie Hu, Jiajun Deng, Xunying Liu
 - **Subjects:** Subjects:
Sound (cs.SD); Artificial Intelligence (cs.AI); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2501.03643

 - **Pdf link:** https://arxiv.org/pdf/2501.03643

 - **Abstract**
 This paper presents a novel mixed-precision quantization approach for speech foundation models that tightly integrates mixed-precision learning and quantized model parameter estimation into one single model compression stage. Experiments conducted on LibriSpeech dataset with fine-tuned wav2vec2.0-base and HuBERT-large models suggest the resulting mixed-precision quantized models increased the lossless compression ratio by factors up to 1.7x and 1.9x over the respective uniform-precision and two-stage mixed-precision quantized baselines that perform precision learning and model parameters quantization in separate and disjointed stages, while incurring no statistically word error rate (WER) increase over the 32-bit full-precision models. The system compression time of wav2vec2.0-base and HuBERT-large models is reduced by up to 1.9 and 1.5 times over the two-stage mixed-precision baselines, while both produce lower WERs. The best-performing 3.5-bit mixed-precision quantized HuBERT-large model produces a lossless compression ratio of 8.6x over the 32-bit full-precision system.
#### Unsupervised Speech Segmentation: A General Approach Using Speech Language Models
 - **Authors:** Avishai Elmakies, Omri Abend, Yossi Adi
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Artificial Intelligence (cs.AI); Machine Learning (cs.LG); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2501.03711

 - **Pdf link:** https://arxiv.org/pdf/2501.03711

 - **Abstract**
 In this paper, we introduce an unsupervised approach for Speech Segmentation, which builds on previously researched approaches, e.g., Speaker Diarization, while being applicable to an inclusive set of acoustic-semantic distinctions, paving a path towards a general Unsupervised Speech Segmentation approach. Unlike traditional speech and audio segmentation, which mainly focuses on spectral changes in the input signal, e.g., phone segmentation, our approach tries to segment the spoken utterance into chunks with differing acoustic-semantic styles, focusing on acoustic-semantic information that does not translate well into text, e.g., emotion or speaker. While most Speech Segmentation tasks only handle one style change, e.g., emotion diarization, our approach tries to handle multiple acoustic-semantic style changes. Leveraging recent advances in Speech Language Models (SLMs), we propose a simple unsupervised method to segment a given speech utterance. We empirically demonstrate the effectiveness of the proposed approach by considering several setups. Results suggest that the proposed method is superior to the evaluated baselines on boundary detection, segment purity, and over-segmentation. Code is available at this https URL.
#### Guitar-TECHS: An Electric Guitar Dataset Covering Techniques, Musical Excerpts, Chords and Scales Using a Diverse Array of Hardware
 - **Authors:** Hegel Pedroza, Wallace Abreu, Ryan M. Corey, Iran R. Roman
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2501.03720

 - **Pdf link:** https://arxiv.org/pdf/2501.03720

 - **Abstract**
 Guitar-related machine listening research involves tasks like timbre transfer, performance generation, and automatic transcription. However, small datasets often limit model robustness due to insufficient acoustic diversity and musical content. To address these issues, we introduce Guitar-TECHS, a comprehensive dataset featuring a variety of guitar techniques, musical excerpts, chords, and scales. These elements are performed by diverse musicians across various recording settings. Guitar-TECHS incorporates recordings from two stereo microphones: an egocentric microphone positioned on the performer's head and an exocentric microphone placed in front of the performer. It also includes direct input recordings and microphoned amplifier outputs, offering a wide spectrum of audio inputs and recording qualities. All signals and MIDI labels are properly synchronized. Its multi-perspective and multi-modal content makes Guitar-TECHS a valuable resource for advancing data-driven guitar research, and to develop robust guitar listening algorithms. We provide empirical data to demonstrate the dataset's effectiveness in training robust models for Guitar Tablature Transcription.
#### NeuroIncept Decoder for High-Fidelity Speech Reconstruction from Neural Activity
 - **Authors:** Owais Mujtaba Khanday, José L. Pérez-Córdoba, Mohd Yaqub Mir, Ashfaq Ahmad Najar, Jose A. Gonzalez-Lopez
 - **Subjects:** Subjects:
Sound (cs.SD); Human-Computer Interaction (cs.HC); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2501.03757

 - **Pdf link:** https://arxiv.org/pdf/2501.03757

 - **Abstract**
 This paper introduces a novel algorithm designed for speech synthesis from neural activity recordings obtained using invasive electroencephalography (EEG) techniques. The proposed system offers a promising communication solution for individuals with severe speech impairments. Central to our approach is the integration of time-frequency features in the high-gamma band computed from EEG recordings with an advanced NeuroIncept Decoder architecture. This neural network architecture combines Convolutional Neural Networks (CNNs) and Gated Recurrent Units (GRUs) to reconstruct audio spectrograms from neural patterns. Our model demonstrates robust mean correlation coefficients between predicted and actual spectrograms, though inter-subject variability indicates distinct neural processing mechanisms among participants. Overall, our study highlights the potential of neural decoding techniques to restore communicative abilities in individuals with speech disorders and paves the way for future advancements in brain-computer interface technologies.
#### Detecting the Undetectable: Assessing the Efficacy of Current Spoof Detection Methods Against Seamless Speech Edits
 - **Authors:** Sung-Feng Huang, Heng-Cheng Kuo, Zhehuai Chen, Xuesong Yang, Chao-Han Huck Yang, Yu Tsao, Yu-Chiang Frank Wang, Hung-yi Lee, Szu-Wei Fu
 - **Subjects:** Subjects:
Sound (cs.SD); Computation and Language (cs.CL); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2501.03805

 - **Pdf link:** https://arxiv.org/pdf/2501.03805

 - **Abstract**
 Neural speech editing advancements have raised concerns about their misuse in spoofing attacks. Traditional partially edited speech corpora primarily focus on cut-and-paste edits, which, while maintaining speaker consistency, often introduce detectable discontinuities. Recent methods, like A\textsuperscript{3}T and Voicebox, improve transitions by leveraging contextual information. To foster spoofing detection research, we introduce the Speech INfilling Edit (SINE) dataset, created with Voicebox. We detailed the process of re-implementing Voicebox training and dataset creation. Subjective evaluations confirm that speech edited using this novel technique is more challenging to detect than conventional cut-and-paste methods. Despite human difficulty, experimental results demonstrate that self-supervised-based detectors can achieve remarkable performance in detection, localization, and generalization across different edit methods. The dataset and related models will be made publicly available.


by Zyzzyva0381 (Windy). 


2025-01-08
