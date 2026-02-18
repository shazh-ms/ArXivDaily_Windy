# Showing new listings for Wednesday, 18 February 2026
Auto update papers at about 2:30am UTC (10:30am Beijing time) every weekday.


阅读 `Usage.md`了解如何使用此repo实现个性化的Arxiv论文推送

See `Usage.md` for instructions on how to personalize the repo. 


Keyword list: ['text-to-speech', 'text to speech', 'tts', 'LLM-based', 'speech', 'voice']


Excluded: []


### Today: 5papers 
#### What Do Neurons Listen To? A Neuron-level Dissection of a General-purpose Audio Model
 - **Authors:** Takao Kawamura, Daisuke Niizumi, Nobutaka Ono
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2602.15307

 - **Pdf link:** https://arxiv.org/pdf/2602.15307

 - **Abstract**
 In this paper, we analyze the internal representations of a general-purpose audio self-supervised learning (SSL) model from a neuron-level perspective. Despite their strong empirical performance as feature extractors, the internal mechanisms underlying the robust generalization of SSL audio models remain unclear. Drawing on the framework of mechanistic interpretability, we identify and examine class-specific neurons by analyzing conditional activation patterns across diverse tasks. Our analysis reveals that SSL models foster the emergence of class-specific neurons that provide extensive coverage across novel task classes. These neurons exhibit shared responses across different semantic categories and acoustic similarities, such as speech attributes and musical pitch. We also confirm that these neurons have a functional impact on classification performance. To our knowledge, this is the first systematic neuron-level analysis of a general-purpose audio SSL model, providing new insights into its internal representation.
#### Bottleneck Transformer-Based Approach for Improved Automatic STOI Score Prediction
 - **Authors:** Amartyaveer, Murali Kadambi, Chandra Mohan Sharma, Anupam Mondal, Prasanta Kumar Ghosh
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Machine Learning (cs.LG); Signal Processing (eess.SP)
 - **Arxiv link:** https://arxiv.org/abs/2602.15484

 - **Pdf link:** https://arxiv.org/pdf/2602.15484

 - **Abstract**
 In this study, we have presented a novel approach to predict the Short-Time Objective Intelligibility (STOI) metric using a bottleneck transformer architecture. Traditional methods for calculating STOI typically requires clean reference speech, which limits their applicability in the real world. To address this, numerous deep learning-based nonintrusive speech assessment models have garnered significant interest. Many studies have achieved commendable performance, but there is room for further improvement. We propose the use of bottleneck transformer, incorporating convolution blocks for learning frame-level features and a multi-head self-attention (MHSA) layer to aggregate the information. These components enable the transformer to focus on the key aspects of the input data. Our model has shown higher correlation and lower mean squared error for both seen and unseen scenarios compared to the state-of-the-art model using self-supervised learning (SSL) and spectral features as inputs.
#### Enroll-on-Wakeup: A First Comparative Study of Target Speech Extraction for Seamless Interaction in Real Noisy Human-Machine Dialogue Scenarios
 - **Authors:** Yiming Yang, Guangyong Wang, Haixin Guan, Yanhua Long
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2602.15519

 - **Pdf link:** https://arxiv.org/pdf/2602.15519

 - **Abstract**
 Target speech extraction (TSE) typically relies on pre-recorded high-quality enrollment speech, which disrupts user experience and limits feasibility in spontaneous interaction. In this paper, we propose Enroll-on-Wakeup (EoW), a novel framework where the wake-word segment, captured naturally during human-machine interaction, is automatically utilized as the enrollment reference. This eliminates the need for pre-collected speech to enable a seamless experience. We perform the first systematic study of EoW-TSE, evaluating advanced discriminative and generative models under real diverse acoustic conditions. Given the short and noisy nature of wake-word segments, we investigate enrollment augmentation using LLM-based TTS. Results show that while current TSE models face performance degradation in EoW-TSE, TTS-based assistance significantly enhances the listening experience, though gaps remain in speech recognition accuracy.
#### ZeroSyl: Simple Zero-Resource Syllable Tokenization for Spoken Language Modeling
 - **Authors:** Nicol Visser, Simon Malan, Danel Slabbert, Herman Kamper
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2602.15537

 - **Pdf link:** https://arxiv.org/pdf/2602.15537

 - **Abstract**
 Pure speech language models aim to learn language directly from raw audio without textual resources. A key challenge is that discrete tokens from self-supervised speech encoders result in excessively long sequences, motivating recent work on syllable-like units. However, methods like Sylber and SyllableLM rely on intricate multi-stage training pipelines. We propose ZeroSyl, a simple training-free method to extract syllable boundaries and embeddings directly from a frozen WavLM model. Using L2 norms of features in WavLM's intermediate layers, ZeroSyl achieves competitive syllable segmentation performance. The resulting segments are mean-pooled, discretized using K-means, and used to train a language model. ZeroSyl outperforms prior syllabic tokenizers across lexical, syntactic, and narrative benchmarks. Scaling experiments show that while finer-grained units are beneficial for lexical tasks, our discovered syllabic units exhibit better scaling behavior for syntactic modeling.
#### UniTAF: A Modular Framework for Joint Text-to-Speech and Audio-to-Face Modeling
 - **Authors:** Qiangong Zhou, Nagasaka Tomohiro
 - **Subjects:** Subjects:
Sound (cs.SD); Computer Vision and Pattern Recognition (cs.CV); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2602.15651

 - **Pdf link:** https://arxiv.org/pdf/2602.15651

 - **Abstract**
 This work considers merging two independent models, TTS and A2F, into a unified model to enable internal feature transfer, thereby improving the consistency between audio and facial expressions generated from text. We also discuss the extension of the emotion control mechanism from TTS to the joint model. This work does not aim to showcase generation quality; instead, from a system design perspective, it validates the feasibility of reusing intermediate representations from TTS for joint modeling of speech and facial expressions, and provides engineering practice references for subsequent speech expression co-design. The project code has been open source at: this https URL


by Zyzzyva0381 (Windy). 


2026-02-18
