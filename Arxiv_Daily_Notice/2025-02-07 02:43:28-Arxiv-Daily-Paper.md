# Showing new listings for Friday, 7 February 2025
Auto update papers at about 2:30am UTC (10:30am Beijing time) every weekday.


阅读 `Usage.md`了解如何使用此repo实现个性化的Arxiv论文推送

See `Usage.md` for instructions on how to personalize the repo. 


Keyword list: ['acoustic', 'reinforcement learning', 'reverb', 'meta', 'separate', 'reconstruction', 'noise', 'enhance', 'localization', 'speech']


Excluded: []


### Today: 11papers 
#### Dementia Classification Using Acoustic Speech and Feature Selection
 - **Authors:** Marko Niemelä, Mikaela von Bonsdorff, Sami Äyrämö, Tommi Kärkkäinen
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Machine Learning (cs.LG); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2502.03484

 - **Pdf link:** https://arxiv.org/pdf/2502.03484

 - **Abstract**
 Dementia is a general term for a group of syndromes that affect cognitive functions such as memory, thinking, reasoning, and the ability to perform daily tasks. The number of dementia patients is increasing as the population ages, and it is estimated that over 10 million people develop dementia each year. Dementia progresses gradually, and the sooner a patient receives help and support, the better their chances of maintaining their functional abilities. For this reason, early diagnosis of dementia is important. In recent years, machine learning models based on naturally spoken language have been developed for the early diagnosis of dementia. These methods have proven to be user-friendly, cost-effective, scalable, and capable of providing extremely fast diagnoses. This study utilizes the well-known ADReSS challenge dataset for classifying healthy controls and Alzheimer's patients. The dataset contains speech recordings from a picture description task featuring a kitchen scene, collected from both healthy controls and dementia patients. Unlike most studies, this research does not segment the audio recordings into active speech segments; instead, acoustic features are extracted from entire recordings. The study employs Ridge linear regression, Extreme Minimal Learning Machine, and Linear Support Vector Machine machine learning models to compute feature importance scores based on model outputs. The Ridge model performed best in Leave-One-Subject-Out cross-validation, achieving a classification accuracy of 87.8%. The EMLM model, proved to be effective in both cross-validation and the classification of a separate test dataset, with accuracies of 85.3% and 79.2%, respectively. The study's results rank among the top compared to other studies using the same dataset and acoustic feature extraction for dementia diagnosis.
#### Comprehensive Layer-wise Analysis of SSL Models for Audio Deepfake Detection
 - **Authors:** Yassine El Kheir, Youness Samih, Suraj Maharjan, Tim Polzehl, Sebastian Möller
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2502.03559

 - **Pdf link:** https://arxiv.org/pdf/2502.03559

 - **Abstract**
 This paper conducts a comprehensive layer-wise analysis of self-supervised learning (SSL) models for audio deepfake detection across diverse contexts, including multilingual datasets (English, Chinese, Spanish), partial, song, and scene-based deepfake scenarios. By systematically evaluating the contributions of different transformer layers, we uncover critical insights into model behavior and performance. Our findings reveal that lower layers consistently provide the most discriminative features, while higher layers capture less relevant information. Notably, all models achieve competitive equal error rate (EER) scores even when employing a reduced number of layers. This indicates that we can reduce computational costs and increase the inference speed of detecting deepfakes by utilizing only a few lower layers. This work enhances our understanding of SSL models in deepfake detection, offering valuable insights applicable across varied linguistic and contextual settings. Our trained models and code are publicly available: this https URL.
#### DiTAR: Diffusion Transformer Autoregressive Modeling for Speech Generation
 - **Authors:** Dongya Jia, Zhuo Chen, Jiawei Chen, Chenpeng Du, Jian Wu, Jian Cong, Xiaobin Zhuang, Chumin Li, Zhen Wei, Yuping Wang, Yuxuan Wang
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Artificial Intelligence (cs.AI); Computation and Language (cs.CL); Machine Learning (cs.LG); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2502.03930

 - **Pdf link:** https://arxiv.org/pdf/2502.03930

 - **Abstract**
 Several recent studies have attempted to autoregressively generate continuous speech representations without discrete speech tokens by combining diffusion and autoregressive models, yet they often face challenges with excessive computational loads or suboptimal outcomes. In this work, we propose Diffusion Transformer Autoregressive Modeling (DiTAR), a patch-based autoregressive framework combining a language model with a diffusion transformer. This approach significantly enhances the efficacy of autoregressive models for continuous tokens and reduces computational demands. DiTAR utilizes a divide-and-conquer strategy for patch generation, where the language model processes aggregated patch embeddings and the diffusion transformer subsequently generates the next patch based on the output of the language model. For inference, we propose defining temperature as the time point of introducing noise during the reverse diffusion ODE to balance diversity and determinism. We also show in the extensive scaling analysis that DiTAR has superb scalability. In zero-shot speech generation, DiTAR achieves state-of-the-art performance in robustness, speaker similarity, and naturalness.
#### Towards Explainable Spoofed Speech Attribution and Detection:a Probabilistic Approach for Characterizing Speech Synthesizer Components
 - **Authors:** Jagabandhu Mishraa, Manasi Chhibbera, Hye-jin Shimb, Tomi H. Kinnunena
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2502.04049

 - **Pdf link:** https://arxiv.org/pdf/2502.04049

 - **Abstract**
 We propose an explainable probabilistic framework for characterizing spoofed speech by decomposing it into probabilistic attribute embeddings. Unlike raw high-dimensional countermeasure embeddings, which lack interpretability, the proposed probabilistic attribute embeddings aim to detect specific speech synthesizer components, represented through high-level attributes and their corresponding values. We use these probabilistic embeddings with four classifier back-ends to address two downstream tasks: spoofing detection and spoofing attack attribution. The former is the well-known bonafide-spoof detection task, whereas the latter seeks to identify the source method (generator) of a spoofed utterance. We additionally use Shapley values, a widely used technique in machine learning, to quantify the relative contribution of each attribute value to the decision-making process in each task. Results on the ASVspoof2019 dataset demonstrate the substantial role of duration and conversion modeling in spoofing detection; and waveform generation and speaker modeling in spoofing attack attribution. In the detection task, the probabilistic attribute embeddings achieve $99.7\%$ balanced accuracy and $0.22\%$ equal error rate (EER), closely matching the performance of raw embeddings ($99.9\%$ balanced accuracy and $0.22\%$ EER). Similarly, in the attribution task, our embeddings achieve $90.23\%$ balanced accuracy and $2.07\%$ EER, compared to $90.16\%$ and $2.11\%$ with raw embeddings. These results demonstrate that the proposed framework is both inherently explainable by design and capable of achieving performance comparable to raw CM embeddings.
#### Llasa: Scaling Train-Time and Inference-Time Compute for Llama-based Speech Synthesis
 - **Authors:** Zhen Ye, Xinfa Zhu, Chi-Min Chan, Xinsheng Wang, Xu Tan, Jiahe Lei, Yi Peng, Haohe Liu, Yizhu Jin, Zheqi DAI, Hongzhan Lin, Jianyi Chen, Xingjian Du, Liumeng Xue, Yunlin Chen, Zhifei Li, Lei Xie, Qiuqiang Kong, Yike Guo, Wei Xue
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Artificial Intelligence (cs.AI); Computation and Language (cs.CL); Multimedia (cs.MM); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2502.04128

 - **Pdf link:** https://arxiv.org/pdf/2502.04128

 - **Abstract**
 Recent advances in text-based large language models (LLMs), particularly in the GPT series and the o1 model, have demonstrated the effectiveness of scaling both training-time and inference-time compute. However, current state-of-the-art TTS systems leveraging LLMs are often multi-stage, requiring separate models (e.g., diffusion models after LLM), complicating the decision of whether to scale a particular model during training or testing. This work makes the following contributions: First, we explore the scaling of train-time and inference-time compute for speech synthesis. Second, we propose a simple framework Llasa for speech synthesis that employs a single-layer vector quantizer (VQ) codec and a single Transformer architecture to fully align with standard LLMs such as Llama. Our experiments reveal that scaling train-time compute for Llasa consistently improves the naturalness of synthesized speech and enables the generation of more complex and accurate prosody patterns. Furthermore, from the perspective of scaling inference-time compute, we employ speech understanding models as verifiers during the search, finding that scaling inference-time compute shifts the sampling modes toward the preferences of specific verifiers, thereby improving emotional expressiveness, timbre consistency, and content accuracy. In addition, we released the checkpoint and training code for our TTS model (1B, 3B, 8B) and codec model publicly available.
#### Blind Capon Beamformer Based on Independent Component Extraction: Single-Parameter Algorithm,
 - **Authors:** Zbyněk Koldovský, Jaroslav Čmejla, Stephen O'Regan
 - **Subjects:** Subjects:
Signal Processing (eess.SP); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2502.03871

 - **Pdf link:** https://arxiv.org/pdf/2502.03871

 - **Abstract**
 We consider a phase-shift mixing model for linear sensor arrays in the context of blind source extraction. We derive a blind Capon beamformer that seeks the direction where the output is independent of the other signals in the mixture. The algorithm is based on Independent Component Extraction and imposes an orthogonal constraint, thanks to which it optimizes only one real-valued parameter related to the angle of arrival. The Cramér-Rao lower bound for the mean interference-to-signal ratio is derived. The algorithm and the bound are compared with conventional blind and direction-of-arrival estimation+beamforming methods, showing improvements in terms of extraction accuracy. An application is demonstrated in frequency-domain speaker extraction in a low-reverberation room.
#### UniForm: A Unified Diffusion Transformer for Audio-Video Generation
 - **Authors:** Lei Zhao, Linfeng Feng, Dongxu Ge, Fangqiu Yi, Chi Zhang, Xiao-Lei Zhang, Xuelong Li
 - **Subjects:** Subjects:
Multimedia (cs.MM); Artificial Intelligence (cs.AI); Computer Vision and Pattern Recognition (cs.CV); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2502.03897

 - **Pdf link:** https://arxiv.org/pdf/2502.03897

 - **Abstract**
 As a natural multimodal content, audible video delivers an immersive sensory experience. Consequently, audio-video generation systems have substantial potential. However, existing diffusion-based studies mainly employ relatively independent modules for generating each modality, which lack exploration of shared-weight generative modules. This approach may under-use the intrinsic correlations between audio and visual modalities, potentially resulting in sub-optimal generation quality. To address this, we propose UniForm, a unified diffusion transformer designed to enhance cross-modal consistency. By concatenating auditory and visual information, UniForm learns to generate audio and video simultaneously within a unified latent space, facilitating the creation of high-quality and well-aligned audio-visual pairs. Extensive experiments demonstrate the superior performance of our method in joint audio-video generation, audio-guided video generation, and video-guided audio generation tasks. Our demos are available at this https URL.
#### Towards Unified Music Emotion Recognition across Dimensional and Categorical Models
 - **Authors:** Jaeyong Kang, Dorien Herremans
 - **Subjects:** Subjects:
Sound (cs.SD); Artificial Intelligence (cs.AI); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2502.03979

 - **Pdf link:** https://arxiv.org/pdf/2502.03979

 - **Abstract**
 One of the most significant challenges in Music Emotion Recognition (MER) comes from the fact that emotion labels can be heterogeneous across datasets with regard to the emotion representation, including categorical (e.g., happy, sad) versus dimensional labels (e.g., valence-arousal). In this paper, we present a unified multitask learning framework that combines these two types of labels and is thus able to be trained on multiple datasets. This framework uses an effective input representation that combines musical features (i.e., key and chords) and MERT embeddings. Moreover, knowledge distillation is employed to transfer the knowledge of teacher models trained on individual datasets to a student model, enhancing its ability to generalize across multiple tasks. To validate our proposed framework, we conducted extensive experiments on a variety of datasets, including MTG-Jamendo, DEAM, PMEmo, and EmoMusic. According to our experimental results, the inclusion of musical features, multitask learning, and knowledge distillation significantly enhances performance. In particular, our model outperforms the state-of-the-art models, including the best-performing model from the MediaEval 2021 competition on the MTG-Jamendo dataset. Our work makes a significant contribution to MER by allowing the combination of categorical and dimensional emotion labels in one unified framework, thus enabling training across datasets.
#### A data-driven two-microphone method for in-situ sound absorption measurements
 - **Authors:** Leon Emmerich, Patrik Aste, Eric Brandão, Mélanie Nolan, Jacques Cuenca, U. Peter Svensson, Marcus Maeder, Steffen Marburg, Elias Zea
 - **Subjects:** Subjects:
Sound (cs.SD); Machine Learning (cs.LG); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2502.04143

 - **Pdf link:** https://arxiv.org/pdf/2502.04143

 - **Abstract**
 This work presents a data-driven approach to estimating the sound absorption coefficient of an infinite porous slab using a neural network and a two-microphone measurement on a finite porous sample. A 1D-convolutional network predicts the sound absorption coefficient from the complex-valued transfer function between the sound pressure measured at the two microphone positions. The network is trained and validated with numerical data generated by a boundary element model using the Delany-Bazley-Miki model, demonstrating accurate predictions for various numerical samples. The method is experimentally validated with baffled rectangular samples of a fibrous material, where sample size and source height are varied. The results show that the neural network offers the possibility to reliably predict the in-situ sound absorption of a porous material using the traditional two-microphone method as if the sample were infinite. The normal-incidence sound absorption coefficient obtained by the network compares well with that obtained theoretically and in an impedance tube. The proposed method has promising perspectives for estimating the sound absorption coefficient of acoustic materials after installation and in realistic operational conditions.
#### XAttnMark: Learning Robust Audio Watermarking with Cross-Attention
 - **Authors:** Yixin Liu, Lie Lu, Jihui Jin, Lichao Sun, Andrea Fanelli
 - **Subjects:** Subjects:
Sound (cs.SD); Artificial Intelligence (cs.AI); Cryptography and Security (cs.CR); Machine Learning (cs.LG); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2502.04230

 - **Pdf link:** https://arxiv.org/pdf/2502.04230

 - **Abstract**
 The rapid proliferation of generative audio synthesis and editing technologies has raised significant concerns about copyright infringement, data provenance, and the spread of misinformation through deepfake audio. Watermarking offers a proactive solution by embedding imperceptible, identifiable, and traceable marks into audio content. While recent neural network-based watermarking methods like WavMark and AudioSeal have improved robustness and quality, they struggle to achieve both robust detection and accurate attribution simultaneously. This paper introduces Cross-Attention Robust Audio Watermark (XAttnMark), which bridges this gap by leveraging partial parameter sharing between the generator and the detector, a cross-attention mechanism for efficient message retrieval, and a temporal conditioning module for improved message distribution. Additionally, we propose a psychoacoustic-aligned temporal-frequency masking loss that captures fine-grained auditory masking effects, enhancing watermark imperceptibility. Our approach achieves state-of-the-art performance in both detection and attribution, demonstrating superior robustness against a wide range of audio transformations, including challenging generative editing with strong editing strength. The project webpage is available at this https URL.
#### Ola: Pushing the Frontiers of Omni-Modal Language Model with Progressive Modality Alignment
 - **Authors:** Zuyan Liu, Yuhao Dong, Jiahui Wang, Ziwei Liu, Winston Hu, Jiwen Lu, Yongming Rao
 - **Subjects:** Subjects:
Computer Vision and Pattern Recognition (cs.CV); Computation and Language (cs.CL); Multimedia (cs.MM); Sound (cs.SD); Audio and Speech Processing (eess.AS); Image and Video Processing (eess.IV)
 - **Arxiv link:** https://arxiv.org/abs/2502.04328

 - **Pdf link:** https://arxiv.org/pdf/2502.04328

 - **Abstract**
 Recent advances in large language models, particularly following GPT-4o, have sparked increasing interest in developing omni-modal models capable of understanding more modalities. While some open-source alternatives have emerged, there is still a notable lag behind specialized single-modality models in performance. In this paper, we present Ola, an Omni-modal language model that achieves competitive performance across image, video, and audio understanding compared to specialized counterparts. The core design of Ola lies in its progressive modality alignment strategy that extends the supporting modality of the language model progressively. Our training pipeline begins with the most distinct modalities: image and text, then gradually expands the skill sets of the model using speech data that connects language and audio knowledge, and video data that connects all modalities. The progressive learning pipeline also enables us to maintain a relatively small size of the cross-modal alignment data, making developing omni-modal from existing vision-language models easy and less costly. Moreover, to unlock an advanced interactive experience like GPT-4o, we further design a sentence-wise decoding solution for streaming speech generation. Extensive experiments demonstrate that Ola surpasses existing open omni-modal LLMs across all modalities while achieving highly competitive performance compared to state-of-the-art specialized models of similar sizes. We aim to make Ola a fully open omni-modal understanding solution to advance future research in this emerging field. Model weights, code, and data are open-sourced at this https URL.


by Zyzzyva0381 (Windy). 


2025-02-07
