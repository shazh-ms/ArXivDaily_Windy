# Showing new listings for Monday, 7 April 2025
Auto update papers at about 2:30am UTC (10:30am Beijing time) every weekday.


阅读 `Usage.md`了解如何使用此repo实现个性化的Arxiv论文推送

See `Usage.md` for instructions on how to personalize the repo. 


Keyword list: ['acoustic', 'reinforcement learning', 'reverb', 'meta', 'separate', 'reconstruction', 'noise', 'enhance', 'localization', 'speech']


Excluded: []


### Today: 5papers 
#### Mind the Prompt: Prompting Strategies in Audio Generations for Improving Sound Classification
 - **Authors:** Francesca Ronchini, Ho-Hsiang Wu, Wei-Cheng Lin, Fabio Antonacci
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Artificial Intelligence (cs.AI); Sound (cs.SD); Signal Processing (eess.SP)
 - **Arxiv link:** https://arxiv.org/abs/2504.03329

 - **Pdf link:** https://arxiv.org/pdf/2504.03329

 - **Abstract**
 This paper investigates the design of effective prompt strategies for generating realistic datasets using Text-To-Audio (TTA) models. We also analyze different techniques for efficiently combining these datasets to enhance their utility in sound classification tasks. By evaluating two sound classification datasets with two TTA models, we apply a range of prompt strategies. Our findings reveal that task-specific prompt strategies significantly outperform basic prompt approaches in data generation. Furthermore, merging datasets generated using different TTA models proves to enhance classification performance more effectively than merely increasing the training dataset size. Overall, our results underscore the advantages of these methods as effective data augmentation techniques using synthetic data.
#### Generating Diverse Audio-Visual 360 Soundscapes for Sound Event Localization and Detection
 - **Authors:** Adrian S. Roman, Aiden Chang, Gerardo Meza, Iran R. Roman
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2504.02988

 - **Pdf link:** https://arxiv.org/pdf/2504.02988

 - **Abstract**
 We present SELDVisualSynth, a tool for generating synthetic videos for audio-visual sound event localization and detection (SELD). Our approach incorporates real-world background images to improve realism in synthetic audio-visual SELD data while also ensuring audio-visual spatial alignment. The tool creates 360 synthetic videos where objects move matching synthetic SELD audio data and its annotations. Experimental results demonstrate that a model trained with this data attains performance gains across multiple metrics, achieving superior localization recall (56.4 LR) and competitive localization error (21.9deg LE). We open-source our data generation tool for maximal use by members of the SELD research community.
#### RWKVTTS: Yet another TTS based on RWKV-7
 - **Authors:** Lin yueyu, Liu Xiao
 - **Subjects:** Subjects:
Sound (cs.SD); Computation and Language (cs.CL); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2504.03289

 - **Pdf link:** https://arxiv.org/pdf/2504.03289

 - **Abstract**
 Human-AI interaction thrives on intuitive and efficient interfaces, among which voice stands out as a particularly natural and accessible modality. Recent advancements in transformer-based text-to-speech (TTS) systems, such as Fish-Speech, CosyVoice, and MegaTTS 3, have delivered remarkable improvements in quality and realism, driving a significant evolution in the TTS domain. In this paper, we introduce RWKV-7 \cite{peng2025rwkv}, a cutting-edge RNN-based architecture tailored for TTS applications. Unlike traditional transformer models, RWKV-7 leverages the strengths of recurrent neural networks to achieve greater computational efficiency and scalability, while maintaining high-quality output. Our comprehensive benchmarks demonstrate that RWKV-7 outperforms transformer-based models across multiple key metrics, including synthesis speed, naturalness of speech, and resource efficiency. Furthermore, we explore its adaptability to diverse linguistic contexts and low-resource environments, showcasing its potential to democratize TTS technology. These findings position RWKV-7 as a powerful and innovative alternative, paving the way for more accessible and versatile voice synthesis solutions in real-world this http URL code and weights are this https URL, this https URL
#### An Efficient GPU-based Implementation for Noise Robust Sound Source Localization
 - **Authors:** Zirui Lin, Masayuki Takigahira, Naoya Terakado, Haris Gulzar, Monikka Roslianna Busto, Takeharu Eda, Katsutoshi Itoyama, Kazuhiro Nakadai, Hideharu Amano
 - **Subjects:** Subjects:
Sound (cs.SD); Robotics (cs.RO); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2504.03373

 - **Pdf link:** https://arxiv.org/pdf/2504.03373

 - **Abstract**
 Robot audition, encompassing Sound Source Localization (SSL), Sound Source Separation (SSS), and Automatic Speech Recognition (ASR), enables robots and smart devices to acquire auditory capabilities similar to human hearing. Despite their wide applicability, processing multi-channel audio signals from microphone arrays in SSL involves computationally intensive matrix operations, which can hinder efficient deployment on Central Processing Units (CPUs), particularly in embedded systems with limited CPU resources. This paper introduces a GPU-based implementation of SSL for robot audition, utilizing the Generalized Singular Value Decomposition-based Multiple Signal Classification (GSVD-MUSIC), a noise-robust algorithm, within the HARK platform, an open-source software suite. For a 60-channel microphone array, the proposed implementation achieves significant performance improvements. On the Jetson AGX Orin, an embedded device powered by an NVIDIA GPU and ARM Cortex-A78AE v8.2 64-bit CPUs, we observe speedups of 4645.1x for GSVD calculations and 8.8x for the SSL module, while speedups of 2223.4x for GSVD calculation and 8.95x for the entire SSL module on a server configured with an NVIDIA A100 GPU and AMD EPYC 7352 CPUs, making real-time processing feasible for large-scale microphone arrays and providing ample capacity for real-time processing of potential subsequent machine learning or deep learning tasks.
#### MultiMed-ST: Large-scale Many-to-many Multilingual Medical Speech Translation
 - **Authors:** Khai Le-Duc, Tuyen Tran, Bach Phan Tat, Nguyen Kim Hai Bui, Quan Dang, Hung-Phong Tran, Thanh-Thuy Nguyen, Ly Nguyen, Tuan-Minh Phan, Thi Thu Phuong Tran, Chris Ngo, Nguyen X. Khanh, Thanh Nguyen-Tang
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Artificial Intelligence (cs.AI); Machine Learning (cs.LG); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2504.03546

 - **Pdf link:** https://arxiv.org/pdf/2504.03546

 - **Abstract**
 Multilingual speech translation (ST) in the medical domain enhances patient care by enabling efficient communication across language barriers, alleviating specialized workforce shortages, and facilitating improved diagnosis and treatment, particularly during pandemics. In this work, we present the first systematic study on medical ST, to our best knowledge, by releasing MultiMed-ST, a large-scale ST dataset for the medical domain, spanning all translation directions in five languages: Vietnamese, English, German, French, Traditional Chinese and Simplified Chinese, together with the models. With 290,000 samples, our dataset is the largest medical machine translation (MT) dataset and the largest many-to-many multilingual ST among all domains. Secondly, we present the most extensive analysis study in ST research to date, including: empirical baselines, bilingual-multilingual comparative study, end-to-end vs. cascaded comparative study, task-specific vs. multi-task sequence-to-sequence (seq2seq) comparative study, code-switch analysis, and quantitative-qualitative error analysis. All code, data, and models are available online: this https URL.


by Zyzzyva0381 (Windy). 


2025-04-07
