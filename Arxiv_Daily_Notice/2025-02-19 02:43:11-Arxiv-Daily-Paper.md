# Showing new listings for Tuesday, 18 February 2025
Auto update papers at about 2:30am UTC (10:30am Beijing time) every weekday.


阅读 `Usage.md`了解如何使用此repo实现个性化的Arxiv论文推送

See `Usage.md` for instructions on how to personalize the repo. 


Keyword list: ['acoustic', 'reinforcement learning', 'reverb', 'meta', 'separate', 'reconstruction', 'noise', 'enhance', 'localization', 'speech']


Excluded: []


### Today: 12papers 
#### MoHAVE: Mixture of Hierarchical Audio-Visual Experts for Robust Speech Recognition
 - **Authors:** Sungnyun Kim, Kangwook Jang, Sangmin Bae, Sungwoo Cho, Se-Young Yun
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Computation and Language (cs.CL); Machine Learning (cs.LG)
 - **Arxiv link:** https://arxiv.org/abs/2502.10447

 - **Pdf link:** https://arxiv.org/pdf/2502.10447

 - **Abstract**
 Audio-visual speech recognition (AVSR) has become critical for enhancing speech recognition in noisy environments by integrating both auditory and visual modalities. However, existing AVSR systems struggle to scale up without compromising computational efficiency. In this study, we introduce MoHAVE (Mixture of Hierarchical Audio-Visual Experts), a novel robust AVSR framework designed to address these scalability constraints. By leveraging a Mixture-of-Experts (MoE) architecture, MoHAVE activates modality-specific expert groups, ensuring dynamic adaptation to various audio-visual inputs with minimal computational overhead. Key contributions of MoHAVE include: (1) a sparse MoE framework that efficiently scales AVSR model capacity, (2) a hierarchical gating mechanism that dynamically utilizes the expert groups based on input context, enhancing adaptability and robustness, and (3) remarkable performance across robust AVSR benchmarks, including LRS3 and MuAViC transcription and translation tasks, setting a new standard for scalable speech recognition systems.
#### Enhancing Age-Related Robustness in Children Speaker Verification
 - **Authors:** Vishwas M. Shetty, Jiusi Zheng, Steven M. Lulich, Abeer Alwan
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2502.10511

 - **Pdf link:** https://arxiv.org/pdf/2502.10511

 - **Abstract**
 One of the main challenges in children's speaker verification (C-SV) is the significant change in children's voices as they grow. In this paper, we propose two approaches to improve age-related robustness in C-SV. We first introduce a Feature Transform Adapter (FTA) module that integrates local patterns into higher-level global representations, reducing overfitting to specific local features and improving the inter-year SV performance of the system. We then employ Synthetic Audio Augmentation (SAA) to increase data diversity and size, thereby improving robustness against age-related changes. Since the lack of longitudinal speech datasets makes it difficult to measure age-related robustness of C-SV systems, we introduce a longitudinal dataset to assess inter-year verification robustness of C-SV systems. By integrating both of our proposed methods, the average equal error rate was reduced by 19.4%, 13.0%, and 6.1% in the one-year, two-year, and three-year gap inter-year evaluation sets, respectively, compared to the baseline.
#### NeuroAMP: A Novel End-to-end General Purpose Deep Neural Amplifier for Personalized Hearing Aids
 - **Authors:** Shafique Ahmed, Ryandhimas E. Zezario, Hui-Guan Yuan, Amir Hussain, Hsin-Min Wang, Wei-Ho Chung, Yu Tsao
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Artificial Intelligence (cs.AI); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2502.10822

 - **Pdf link:** https://arxiv.org/pdf/2502.10822

 - **Abstract**
 The prevalence of hearing aids is increasing. However, optimizing the amplification processes of hearing aids remains challenging due to the complexity of integrating multiple modular components in traditional methods. To address this challenge, we present NeuroAMP, a novel deep neural network designed for end-to-end, personalized amplification in hearing aids. NeuroAMP leverages both spectral features and the listener's audiogram as inputs, and we investigate four architectures: Convolutional Neural Network (CNN), Long Short-Term Memory (LSTM), Convolutional Recurrent Neural Network (CRNN), and Transformer. We also introduce Denoising NeuroAMP, an extension that integrates noise reduction along with amplification capabilities for improved performance in real-world scenarios. To enhance generalization, a comprehensive data augmentation strategy was employed during training on diverse speech (TIMIT and TMHINT) and music (Cadenza Challenge MUSIC) datasets. Evaluation using the Hearing Aid Speech Perception Index (HASPI), Hearing Aid Speech Quality Index (HASQI), and Hearing Aid Audio Quality Index (HAAQI) demonstrates that the Transformer architecture within NeuroAMP achieves the best performance, with SRCC scores of 0.9927 (HASQI) and 0.9905 (HASPI) on TIMIT, and 0.9738 (HAAQI) on the Cadenza Challenge MUSIC dataset. Notably, our data augmentation strategy maintains high performance on unseen datasets (e.g., VCTK, MUSDB18-HQ). Furthermore, Denoising NeuroAMP outperforms both the conventional NAL-R+WDRC approach and a two-stage baseline on the VoiceBank+DEMAND dataset, achieving a 10% improvement in both HASPI (0.90) and HASQI (0.59) scores. These results highlight the potential of NeuroAMP and Denoising NeuroAMP to deliver notable improvements in personalized hearing aid amplification.
#### Generalizable speech deepfake detection via meta-learned LoRA
 - **Authors:** Janne Laakkonen, Ivan Kukanov, Ville Hautamäki
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Machine Learning (cs.LG); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2502.10838

 - **Pdf link:** https://arxiv.org/pdf/2502.10838

 - **Abstract**
 Generalizable deepfake detection can be formulated as a detection problem where labels (bonafide and fake) are fixed but distributional drift affects the deepfake set. We can always train our detector with one-selected attacks and bonafide data, but an attacker can generate new attacks by just retraining his generator with a different seed. One reasonable approach is to simply pool all different attack types available in training time. Our proposed approach is to utilize meta-learning in combination with LoRA adapters to learn the structure in the training data that is common to all attack types.
#### SpeechT-RAG: Reliable Depression Detection in LLMs with Retrieval-Augmented Generation Using Speech Timing Information
 - **Authors:** Xiangyu Zhang, Hexin Liu, Qiquan Zhang, Beena Ahmed, Julien Epps
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2502.10950

 - **Pdf link:** https://arxiv.org/pdf/2502.10950

 - **Abstract**
 Large Language Models (LLMs) have been increasingly adopted for health-related tasks, yet their performance in depression detection remains limited when relying solely on text input. While Retrieval-Augmented Generation (RAG) typically enhances LLM capabilities, our experiments indicate that traditional text-based RAG systems struggle to significantly improve depression detection accuracy. This challenge stems partly from the rich depression-relevant information encoded in acoustic speech patterns information that current text-only approaches fail to capture effectively. To address this limitation, we conduct a systematic analysis of temporal speech patterns, comparing healthy individuals with those experiencing depression. Based on our findings, we introduce Speech Timing-based Retrieval-Augmented Generation, SpeechT-RAG, a novel system that leverages speech timing features for both accurate depression detection and reliable confidence estimation. This integrated approach not only outperforms traditional text-based RAG systems in detection accuracy but also enhances uncertainty quantification through a confidence scoring mechanism that naturally extends from the same temporal features. Our unified framework achieves comparable results to fine-tuned LLMs without additional training while simultaneously addressing the fundamental requirements for both accuracy and trustworthiness in mental health assessment.
#### AudioSpa: Spatializing Sound Events with Text
 - **Authors:** Linfeng Feng, Lei Zhao, Boyu Zhu, Xiao-Lei Zhang, Xuelong Li
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2502.11219

 - **Pdf link:** https://arxiv.org/pdf/2502.11219

 - **Abstract**
 Text-to-audio (TTA) systems have recently demonstrated strong performance in synthesizing monaural audio from text. However, the task of generating binaural spatial audio from text, which provides a more immersive auditory experience by incorporating the sense of spatiality, have not been explored yet. In this work, we introduce text-guided binaural audio generation. As an early effort, we focus on the scenario where a monaural reference audio is given additionally. The core problem is to associate specific sound events with their directions, thereby creating binaural spatial audio. The challenge lies in the complexity of textual descriptions and the limited availability of single-source sound event datasets. To address this, we propose AudioSpa, an end-to-end model that applies large language models to process both acoustic and textual information. We employ fusion multi-head attention (FMHA) to integrate text tokens, which enhances the generation capability of the multimodal learning. Additionally, we propose a binaural source localization model to assess the quality of the generated audio. Finally, we design a data augmentation strategy to generate diverse datasets, which enables the model to spatialize sound events across various spatial positions. Experimental results demonstrate that our model is able to put sounds at the specified locations accurately. It achieves competitive performance in both localization accuracy and signal distortion. Our demonstrations are available at this https URL.
#### LMFCA-Net: A Lightweight Model for Multi-Channel Speech Enhancement with Efficient Narrow-Band and Cross-Band Attention
 - **Authors:** Yaokai Zhang, Hanchen Pei, Wanqi Wang, Gongping Huang
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Machine Learning (cs.LG); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2502.11462

 - **Pdf link:** https://arxiv.org/pdf/2502.11462

 - **Abstract**
 Deep learning based end-to-end multi-channel speech enhancement methods have achieved impressive performance by leveraging sub-band, cross-band, and spatial information. However, these methods often demand substantial computational resources, limiting their practicality on terminal devices. This paper presents a lightweight multi-channel speech enhancement network with decoupled fully connected attention (LMFCA-Net). The proposed LMFCA-Net introduces time-axis decoupled fully-connected attention (T-FCA) and frequency-axis decoupled fully-connected attention (F-FCA) mechanisms to effectively capture long-range narrow-band and cross-band information without recurrent units. Experimental results show that LMFCA-Net performs comparably to state-of-the-art methods while significantly reducing computational complexity and latency, making it a promising solution for practical applications.
#### YNote: A Novel Music Notation for Fine-Tuning LLMs in Music Generation
 - **Authors:** Shao-Chien Lu, Chen-Chen Yeh, Hui-Lin Cho, Chun-Chieh Hsu, Tsai-Ling Hsu, Cheng-Han Wu, Timothy K. Shih, Yu-Cheng Lin
 - **Subjects:** Subjects:
Sound (cs.SD); Artificial Intelligence (cs.AI); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2502.10467

 - **Pdf link:** https://arxiv.org/pdf/2502.10467

 - **Abstract**
 The field of music generation using Large Language Models (LLMs) is evolving rapidly, yet existing music notation systems, such as MIDI, ABC Notation, and MusicXML, remain too complex for effective fine-tuning of LLMs. These formats are difficult for both machines and humans to interpret due to their variability and intricate structure. To address these challenges, we introduce YNote, a simplified music notation system that uses only four characters to represent a note and its pitch. YNote's fixed format ensures consistency, making it easy to read and more suitable for fine-tuning LLMs. In our experiments, we fine-tuned GPT-2 (124M) on a YNote-encoded dataset and achieved BLEU and ROUGE scores of 0.883 and 0.766, respectively. With just two notes as prompts, the model was able to generate coherent and stylistically relevant music. We believe YNote offers a practical alternative to existing music notations for machine learning applications and has the potential to significantly enhance the quality of music generation using LLMs.
#### FELLE: Autoregressive Speech Synthesis with Token-Wise Coarse-to-Fine Flow Matching
 - **Authors:** Hui Wang, Shujie Liu, Lingwei Meng, Jinyu Li, Yifan Yang, Shiwan Zhao, Haiyang Sun, Yanqing Liu, Haoqin Sun, Jiaming Zhou, Yan Lu, Yong Qin
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2502.11128

 - **Pdf link:** https://arxiv.org/pdf/2502.11128

 - **Abstract**
 To advance continuous-valued token modeling and temporal-coherence enforcement, we propose FELLE, an autoregressive model that integrates language modeling with token-wise flow matching. By leveraging the autoregressive nature of language models and the generative efficacy of flow matching, FELLE effectively predicts continuous-valued tokens (mel-spectrograms). For each continuous-valued token, FELLE modifies the general prior distribution in flow matching by incorporating information from the previous step, improving coherence and stability. Furthermore, to enhance synthesis quality, FELLE introduces a coarse-to-fine flow-matching mechanism, generating continuous-valued tokens hierarchically, conditioned on the language model's output. Experimental results demonstrate the potential of incorporating flow-matching techniques in autoregressive mel-spectrogram modeling, leading to significant improvements in TTS generation quality, as shown in this https URL.
#### TAPS: Throat and Acoustic Paired Speech Dataset for Deep Learning-Based Speech Enhancement
 - **Authors:** Yunsik Kim, Yonghun Song, Yoonyoung Chung
 - **Subjects:** Subjects:
Sound (cs.SD); Machine Learning (cs.LG); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2502.11478

 - **Pdf link:** https://arxiv.org/pdf/2502.11478

 - **Abstract**
 In high-noise environments such as factories, subways, and busy streets, capturing clear speech is challenging due to background noise. Throat microphones provide a solution with their noise-suppressing properties, reducing the noise while recording speech. However, a significant limitation remains: high-frequency information is attenuated as sound waves pass through skin and tissue, reducing speech clarity. Recent deep learning approaches have shown promise in enhancing throat microphone recordings, but further progress is constrained by the absence of standardized dataset. We introduce a throat and acoustic paired speech dataset (TAPS), a collection of paired utterances recorded from 60 native Korean speakers using throat and acoustic microphones. To demonstrate the TAPS's utility, we tested three baseline deep learning models and identified the mapping-based approach as superior in improving speech quality and restoring content. Additionally, we propose an optimal method to mitigate the signal mismatch between throat and acoustic microphones, ensuring model performance. These results highlight the potential of TAPS to serve as a standardized dataset and advance research in throat microphone-based speech enhancement.
#### Step-Audio: Unified Understanding and Generation in Intelligent Speech Interaction
 - **Authors:** Ailin Huang, Boyong Wu, Bruce Wang, Chao Yan, Chen Hu, Chengli Feng, Fei Tian, Feiyu Shen, Jingbei Li, Mingrui Chen, Peng Liu, Ruihang Miao, Wang You, Xi Chen, Xuerui Yang, Yechang Huang, Yuxiang Zhang, Zheng Gong, Zixin Zhang, Brian Li, Changyi Wan, Hanpeng Hu, Ranchen Ming, Song Yuan, Xuelin Zhang, Yu Zhou, Bingxin Li, Buyun Ma, Kang An, Wei Ji, Wen Li, Xuan Wen, Yuankai Ma, Yuanwei Liang, Yun Mou, Bahtiyar Ahmidi, Bin Wang, Bo Li, Changxin Miao, Chen Xu, Chengting Feng, Chenrun Wang, Dapeng Shi, Deshan Sun, Dingyuan Hu, Dula Sai, Enle Liu, Guanzhe Huang, Gulin Yan, Heng Wang, Haonan Jia, Haoyang Zhang, Jiahao Gong, Jianchang Wu, Jiahong Liu, Jianjian Sun, Jiangjie Zhen, Jie Feng, Jie Wu, Jiaoren Wu, Jie Yang, Jinguo Wang, Jingyang Zhang, Junzhe Lin, Kaixiang Li, Lei Xia, Li Zhou, Longlong Gu, Mei Chen, Menglin Wu, Ming Li, Mingxiao Li, Mingyao Liang, Na Wang, Nie Hao, Qiling Wu, Qinyuan Tan, Shaoliang Pang, Shiliang Yang, Shuli Gao, Siqi Liu, Sitong Liu, Tiancheng Cao, Tianyu Wang, Wenjin Deng, Wenqing He, Wen Sun, Xin Han, Xiaomin Deng, Xiaojia Liu, Xu Zhao, Yanan Wei, Yanbo Yu, Yang Cao, Yangguang Li, Yangzhen Ma, Yanming Xu, Yaqiang Shi, Yilei Wang, Yinmin Zhong
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Artificial Intelligence (cs.AI); Human-Computer Interaction (cs.HC); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2502.11946

 - **Pdf link:** https://arxiv.org/pdf/2502.11946

 - **Abstract**
 Real-time speech interaction, serving as a fundamental interface for human-machine collaboration, holds immense potential. However, current open-source models face limitations such as high costs in voice data collection, weakness in dynamic control, and limited intelligence. To address these challenges, this paper introduces Step-Audio, the first production-ready open-source solution. Key contributions include: 1) a 130B-parameter unified speech-text multi-modal model that achieves unified understanding and generation, with the Step-Audio-Chat version open-sourced; 2) a generative speech data engine that establishes an affordable voice cloning framework and produces the open-sourced lightweight Step-Audio-TTS-3B model through distillation; 3) an instruction-driven fine control system enabling dynamic adjustments across dialects, emotions, singing, and RAP; 4) an enhanced cognitive architecture augmented with tool calling and role-playing abilities to manage complex tasks effectively. Based on our new StepEval-Audio-360 evaluation benchmark, Step-Audio achieves state-of-the-art performance in human evaluations, especially in terms of instruction following. On open-source benchmarks like LLaMA Question, shows 9.3% average performance improvement, demonstrating our commitment to advancing the development of open-source multi-modal language technologies. Our code and models are available at this https URL.
#### NaturalL2S: End-to-End High-quality Multispeaker Lip-to-Speech Synthesis with Differential Digital Signal Processing
 - **Authors:** Yifan Liang, Fangkun Liu, Andong Li, Xiaodong Li, Chengshi Zheng
 - **Subjects:** Subjects:
Sound (cs.SD); Computer Vision and Pattern Recognition (cs.CV); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2502.12002

 - **Pdf link:** https://arxiv.org/pdf/2502.12002

 - **Abstract**
 Recent advancements in visual speech recognition (VSR) have promoted progress in lip-to-speech synthesis, where pre-trained VSR models enhance the intelligibility of synthesized speech by providing valuable semantic information. The success achieved by cascade frameworks, which combine pseudo-VSR with pseudo-text-to-speech (TTS) or implicitly utilize the transcribed text, highlights the benefits of leveraging VSR models. However, these methods typically rely on mel-spectrograms as an intermediate representation, which may introduce a key bottleneck: the domain gap between synthetic mel-spectrograms, generated from inherently error-prone lip-to-speech mappings, and real mel-spectrograms used to train vocoders. This mismatch inevitably degrades synthesis quality. To bridge this gap, we propose Natural Lip-to-Speech (NaturalL2S), an end-to-end framework integrating acoustic inductive biases with differentiable speech generation components. Specifically, we introduce a fundamental frequency (F0) predictor to capture prosodic variations in synthesized speech. The predicted F0 then drives a Differentiable Digital Signal Processing (DDSP) synthesizer to generate a coarse signal which serves as prior information for subsequent speech synthesis. Additionally, instead of relying on a reference speaker embedding as an auxiliary input, our approach achieves satisfactory performance on speaker similarity without explicitly modelling speaker characteristics. Both objective and subjective evaluation results demonstrate that NaturalL2S can effectively enhance the quality of the synthesized speech when compared to state-of-the-art methods. Our demonstration page is accessible at this https URL.


by Zyzzyva0381 (Windy). 


2025-02-19
