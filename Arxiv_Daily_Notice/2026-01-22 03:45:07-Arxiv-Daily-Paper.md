# Showing new listings for Thursday, 22 January 2026
Auto update papers at about 2:30am UTC (10:30am Beijing time) every weekday.


阅读 `Usage.md`了解如何使用此repo实现个性化的Arxiv论文推送

See `Usage.md` for instructions on how to personalize the repo. 


Keyword list: ['text-to-speech', 'text to speech', 'tts', 'LLM-based', 'speech', 'voice']


Excluded: []


### Today: 8papers 
#### Towards noise-robust speech inversion through multi-task learning with speech enhancement
 - **Authors:** Saba Tabatabaee, Carol Espy-Wilson
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2601.14516

 - **Pdf link:** https://arxiv.org/pdf/2601.14516

 - **Abstract**
 Recent studies demonstrate the effectiveness of Self Supervised Learning (SSL) speech representations for Speech Inversion (SI). However, applying SI in real-world scenarios remains challenging due to the pervasive presence of background noise. We propose a unified framework that integrates Speech Enhancement (SE) and SI models through shared SSL-based speech representations. In this framework, the SSL model is trained not only to support the SE module in suppressing noise but also to produce representations that are more informative for the SI task, allowing both modules to benefit from joint training. At a Signal-to-Noise Ratio of -5 db, our method for the SI task achieves relative improvements over the baseline of 80.95% under babble noise and 38.98% under non-babble noise, as measured by the average Pearson product-moment correlation across all estimated parameters.
#### Scaling Ambiguity: Augmenting Human Annotation in Speech Emotion Recognition with Audio-Language Models
 - **Authors:** Wenda Zhang, Hongyu Jin, Siyi Wang, Zhiqiang Wei, Ting Dang
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Artificial Intelligence (cs.AI); Machine Learning (cs.LG); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2601.14620

 - **Pdf link:** https://arxiv.org/pdf/2601.14620

 - **Abstract**
 Speech Emotion Recognition models typically use single categorical labels, overlooking the inherent ambiguity of human emotions. Ambiguous Emotion Recognition addresses this by representing emotions as probability distributions, but progress is limited by unreliable ground-truth distributions inferred from sparse human annotations. This paper explores whether Large Audio-Language Models (ALMs) can mitigate the annotation bottleneck by generating high-quality synthetic annotations. We introduce a framework leveraging ALMs to create Synthetic Perceptual Proxies, augmenting human annotations to improve ground-truth distribution reliability. We validate these proxies through statistical analysis of their alignment with human distributions and evaluate their impact by fine-tuning ALMs with the augmented emotion distributions. Furthermore, to address class imbalance and enable unbiased evaluation, we propose DiME-Aug, a Distribution-aware Multimodal Emotion Augmentation strategy. Experiments on IEMOCAP and MSP-Podcast show that synthetic annotations enhance emotion distribution, especially in low-ambiguity regions where annotation agreement is high. However, benefits diminish for highly ambiguous emotions with greater human disagreement. This work provides the first evidence that ALMs could address annotation scarcity in ambiguous emotion recognition, but highlights the need for more advanced prompting or generation strategies to handle highly ambiguous cases.
#### Inverse-Hessian Regularization for Continual Learning in ASR
 - **Authors:** Steven Vander Eeckt, Hugo Van hamme
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2601.14751

 - **Pdf link:** https://arxiv.org/pdf/2601.14751

 - **Abstract**
 Catastrophic forgetting remains a major challenge for continual learning (CL) in automatic speech recognition (ASR), where models must adapt to new domains without losing performance on previously learned conditions. Several CL methods have been proposed for ASR, and, recently, weight averaging - where models are averaged in a merging step after fine-tuning - has proven effective as a simple memory-free strategy. However, it is heuristic in nature and ignores the underlying loss landscapes of the tasks, hindering adaptability. In this work, we propose Inverse Hessian Regularization (IHR), a memory-free approach for CL in ASR that incorporates curvature information into the merging step. After fine-tuning on a new task, the adaptation is adjusted through a Kronecker-factored inverse Hessian approximation of the previous task, ensuring that the model moves primarily in directions less harmful to past performance, while keeping the method lightweight. We evaluate IHR on two CL benchmarks and show that it significantly outperforms state-of-the-art baselines, reducing forgetting while improving adaptability. Ablation studies and analyses further confirm its effectiveness.
#### Test-Time Adaptation For Speech Enhancement Via Mask Polarization
 - **Authors:** Tobias Raichle, Erfan Amini, Bin Yang
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2601.14770

 - **Pdf link:** https://arxiv.org/pdf/2601.14770

 - **Abstract**
 Adapting speech enhancement (SE) models to unseen environments is crucial for practical deployments, yet test-time adaptation (TTA) for SE remains largely under-explored due to a lack of understanding of how SE models degrade under domain shifts. We observe that mask-based SE models lose confidence under domain shifts, with predicted masks becoming flattened and losing decisive speech preservation and noise suppression. Based on this insight, we propose mask polarization (MPol), a lightweight TTA method that restores mask bimodality through distribution comparison using the Wasserstein distance. MPol requires no additional parameters beyond the trained model, making it suitable for resource-constrained edge deployments. Experimental results across diverse domain shifts and architectures demonstrate that MPol achieves very consistent gains that are competitive with significantly more complex approaches.
#### Fast-ULCNet: A fast and ultra low complexity network for single-channel speech enhancement
 - **Authors:** Nicolás Arrieta Larraza, Niels de Koeijer
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Artificial Intelligence (cs.AI)
 - **Arxiv link:** https://arxiv.org/abs/2601.14925

 - **Pdf link:** https://arxiv.org/pdf/2601.14925

 - **Abstract**
 Single-channel speech enhancement algorithms are often used in resource-constrained embedded devices, where low latency and low complexity designs gain more importance. In recent years, researchers have proposed a wide variety of novel solutions to this problem. In particular, a recent deep learning model named ULCNet is among the state-of-the-art approaches in this domain. This paper proposes an adaptation of ULCNet, by replacing its GRU layers with FastGRNNs, to reduce both computational latency and complexity. Furthermore, this paper shows empirical evidence on the performance decay of FastGRNNs in long audio signals during inference due to internal state drifting, and proposes a novel approach based on a trainable complementary filter to mitigate it. The resulting model, Fast-ULCNet, performs on par with the state-of-the-art original ULCNet architecture on a speech enhancement task, while reducing its model size by more than half and decreasing its latency by 34% on average.
#### A Cloud-Based Cross-Modal Transformer for Emotion Recognition and Adaptive Human-Computer Interaction
 - **Authors:** Ziwen Zhong, Zhitao Shu, Yue Zhao
 - **Subjects:** Subjects:
Computer Vision and Pattern Recognition (cs.CV); Artificial Intelligence (cs.AI); Human-Computer Interaction (cs.HC); Machine Learning (cs.LG); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2601.14259

 - **Pdf link:** https://arxiv.org/pdf/2601.14259

 - **Abstract**
 Emotion recognition is a fundamental component of next-generation human-computer interaction (HCI), enabling machines to perceive, understand, and respond to users' affective states. However, existing systems often rely on single-modality analysis such as facial expressions, speech tone, or textual sentiment, resulting in limited robustness and poor generalization in real-world environments. To address these challenges, this study proposes a Cloud-Based Cross-Modal Transformer (CMT) framework for multimodal emotion recognition and adaptive human-computer interaction. The proposed model integrates visual, auditory, and textual signals using pretrained encoders (Vision Transformer, Wav2Vec2, and BERT) and employs a cross-modal attention mechanism to capture complex interdependencies among heterogeneous features. By leveraging cloud computing infrastructure with distributed training on Kubernetes and TensorFlow Serving, the system enables scalable, low-latency emotion recognition for large-scale user interactions. Experiments conducted on benchmark datasets including IEMOCAP, MELD, and AffectNet demonstrate that the CMT achieves state-of-the-art performance, improving the F1-score by 3.0 percent and reducing cross-entropy loss by 12.9 percent compared to strong multimodal baselines. Additionally, cloud deployment evaluations show an average response latency of 128 ms, representing a 35 percent reduction compared with conventional transformer-based fusion systems. These results confirm that the proposed framework enables efficient, real-time emotion recognition and adaptive feedback in applications such as intelligent customer service, virtual tutoring systems, and affective computing interfaces, marking an important step toward cloud-native affective computing and emotionally intelligent interactive systems.
#### VCNAC: A Variable-Channel Neural Audio Codec for Mono, Stereo, and Surround Sound
 - **Authors:** Florian Grötschla, Arunasish Sen, Alessandro Lombardi, Guillermo Cámbara, Andreas Schwarz
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2601.14960

 - **Pdf link:** https://arxiv.org/pdf/2601.14960

 - **Abstract**
 We present VCNAC, a variable channel neural audio codec. Our approach features a single encoder and decoder parametrization that enables native inference for different channel setups, from mono speech to cinematic 5.1 channel surround audio. Channel compatibility objectives ensure that multi-channel content maintains perceptual quality when decoded to fewer channels. The shared representation enables training of generative language models on a single set of codebooks while supporting inference-time scalability across modalities and channel configurations. Evaluation using objective spatial audio metrics and subjective listening tests demonstrates that our unified approach maintains high reconstruction quality across mono, stereo, and surround audio configurations.
#### Neural Tracking of Sustained Attention, Attention Switching, and Natural Conversation in Audiovisual Environments using Mobile EEG
 - **Authors:** Johanna Wilroth, Oskar Keding, Martin A. Skoglund, Maria Sandsten, Martin Enqvist, Emina Alickovic
 - **Subjects:** Subjects:
Signal Processing (eess.SP); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2601.15097

 - **Pdf link:** https://arxiv.org/pdf/2601.15097

 - **Abstract**
 Everyday communication is dynamic and multisensory, often involving shifting attention, overlapping speech and visual cues. Yet, most neural attention tracking studies are still limited to highly controlled lab settings, using clean, often audio-only stimuli and requiring sustained attention to a single talker. This work addresses that gap by introducing a novel dataset from 24 normal-hearing participants. We used a mobile electroencephalography (EEG) system (44 scalp electrodes and 20 cEEGrid electrodes) in an audiovisual (AV) paradigm with three conditions: sustained attention to a single talker in a two-talker environment, attention switching between two talkers, and unscripted two-talker conversations with a competing single talker. Analysis included temporal response functions (TRFs) modeling, optimal lag analysis, selective attention classification with decision windows ranging from 1.1s to 35s, and comparisons of TRFs for attention to AV conversations versus side audio-only talkers. Key findings show significant differences in the attention-related P2-peak between attended and ignored speech across conditions for scalp EEG. No significant change in performance between switching and sustained attention suggests robustness for attention switches. Optimal lag analysis revealed narrower peak for conversation compared to single-talker AV stimuli, reflecting the additional complexity of multi-talker processing. Classification of selective attention was consistently above chance (55-70% accuracy) for scalp EEG, while cEEGrid data yielded lower correlations, highlighting the need for further methodological improvements. These results demonstrate that mobile EEG can reliably track selective attention in dynamic, multisensory listening scenarios and provide guidance for designing future AV paradigms and real-world attention tracking applications.


by Zyzzyva0381 (Windy). 


2026-01-22
