# Showing new listings for Friday, 11 July 2025
Auto update papers at about 2:30am UTC (10:30am Beijing time) every weekday.


阅读 `Usage.md`了解如何使用此repo实现个性化的Arxiv论文推送

See `Usage.md` for instructions on how to personalize the repo. 


Keyword list: ['text-to-speech', 'text to speech', 'tts', 'LLM-based', 'speech', 'voice']


Excluded: []


### Today: 10papers 
#### Generic Speech Enhancement with Self-Supervised Representation Space Loss
 - **Authors:** Hiroshi Sato, Tsubasa Ochiai, Marc Delcroix, Takafumi Moriya, Takanori Ashihara, Ryo Masumura
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD); Signal Processing (eess.SP)
 - **Arxiv link:** https://arxiv.org/abs/2507.07631

 - **Pdf link:** https://arxiv.org/pdf/2507.07631

 - **Abstract**
 Single-channel speech enhancement is utilized in various tasks to mitigate the effect of interfering signals. Conventionally, to ensure the speech enhancement performs optimally, the speech enhancement has needed to be tuned for each task. Thus, generalizing speech enhancement models to unknown downstream tasks has been challenging. This study aims to construct a generic speech enhancement front-end that can improve the performance of back-ends to solve multiple downstream tasks. To this end, we propose a novel training criterion that minimizes the distance between the enhanced and the ground truth clean signal in the feature representation domain of self-supervised learning models. Since self-supervised learning feature representations effectively express high-level speech information useful for solving various downstream tasks, the proposal is expected to make speech enhancement models preserve such information. Experimental validation demonstrates that the proposal improves the performance of multiple speech tasks while maintaining the perceptual quality of the enhanced signal.
#### Audio-Visual Speech Separation via Bottleneck Iterative Network
 - **Authors:** Sidong Zhang, Shiv Shankar, Trang Nguyen, Andrea Fanelli, Madalina Fiterau
 - **Subjects:** Subjects:
Sound (cs.SD); Multimedia (cs.MM); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2507.07270

 - **Pdf link:** https://arxiv.org/pdf/2507.07270

 - **Abstract**
 Integration of information from non-auditory cues can significantly improve the performance of speech-separation models. Often such models use deep modality-specific networks to obtain unimodal features, and risk being too costly or lightweight but lacking capacity. In this work, we present an iterative representation refinement approach called Bottleneck Iterative Network (BIN), a technique that repeatedly progresses through a lightweight fusion block, while bottlenecking fusion representations by fusion tokens. This helps improve the capacity of the model, while avoiding major increase in model size and balancing between the model performance and training cost. We test BIN on challenging noisy audio-visual speech separation tasks, and show that our approach consistently outperforms state-of-the-art benchmark models with respect to SI-SDRi on NTCD-TIMIT and LRS3+WHAM! datasets, while simultaneously achieving a reduction of more than 50% in training and GPU inference time across nearly all settings.
#### ViDove: A Translation Agent System with Multimodal Context and Memory-Augmented Reasoning
 - **Authors:** Yichen Lu, Wei Dai, Jiaen Liu, Ching Wing Kwok, Zongheng Wu, Xudong Xiao, Ao Sun, Sheng Fu, Jianyuan Zhan, Yian Wang, Takatomo Saito, Sicheng Lai
 - **Subjects:** Subjects:
Artificial Intelligence (cs.AI); Computation and Language (cs.CL); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2507.07306

 - **Pdf link:** https://arxiv.org/pdf/2507.07306

 - **Abstract**
 LLM-based translation agents have achieved highly human-like translation results and are capable of handling longer and more complex contexts with greater efficiency. However, they are typically limited to text-only inputs. In this paper, we introduce ViDove, a translation agent system designed for multimodal input. Inspired by the workflow of human translators, ViDove leverages visual and contextual background information to enhance the translation process. Additionally, we integrate a multimodal memory system and long-short term memory modules enriched with domain-specific knowledge, enabling the agent to perform more accurately and adaptively in real-world scenarios. As a result, ViDove achieves significantly higher translation quality in both subtitle generation and general translation tasks, with a 28% improvement in BLEU scores and a 15% improvement in SubER compared to previous state-of-the-art baselines. Moreover, we introduce DoveBench, a new benchmark for long-form automatic video subtitling and translation, featuring 17 hours of high-quality, human-annotated data. Our code is available here: this https URL
#### IML-Spikeformer: Input-aware Multi-Level Spiking Transformer for Speech Processing
 - **Authors:** Zeyang Song, Shimin Zhang, Yuhong Chou, Jibin Wu, Haizhou Li
 - **Subjects:** Subjects:
Multimedia (cs.MM); Machine Learning (cs.LG); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2507.07396

 - **Pdf link:** https://arxiv.org/pdf/2507.07396

 - **Abstract**
 Spiking Neural Networks (SNNs), inspired by biological neural mechanisms, represent a promising neuromorphic computing paradigm that offers energy-efficient alternatives to traditional Artificial Neural Networks (ANNs). Despite proven effectiveness, SNN architectures have struggled to achieve competitive performance on large-scale speech processing task. Two key challenges hinder progress: (1) the high computational overhead during training caused by multi-timestep spike firing, and (2) the absence of large-scale SNN architectures tailored to speech processing tasks. To overcome the issues, we introduce Input-aware Multi-Level Spikeformer, i.e. IML-Spikeformer, a spiking Transformer architecture specifically designed for large-scale speech processing. Central to our design is the Input-aware Multi-Level Spike (IMLS) mechanism, which simulate multi-timestep spike firing within a single timestep using an adaptive, input-aware thresholding scheme. IML-Spikeformer further integrates a Reparameterized Spiking Self-Attention (RepSSA) module with a Hierarchical Decay Mask (HDM), forming the HD-RepSSA module. This module enhances the precision of attention maps and enables modeling of multi-scale temporal dependencies in speech signals. Experiments demonstrate that IML-Spikeformer achieves word error rates of 6.0\% on AiShell-1 and 3.4\% on Librispeech-960, comparable to conventional ANN transformers while reducing theoretical inference energy consumption by 4.64$\times$ and 4.32$\times$ respectively. IML-Spikeformer marks an advance of scalable SNN architectures for large-scale speech processing in both task performance and energy efficiency.
#### DMF2Mel: A Dynamic Multiscale Fusion Network for EEG-Driven Mel Spectrogram Reconstruction
 - **Authors:** Cunhang Fan, Sheng Zhang, Jingjing Zhang, Enrui Liu, Xinhui Li, Minggang Zhao, Zhao Lv
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2507.07526

 - **Pdf link:** https://arxiv.org/pdf/2507.07526

 - **Abstract**
 Decoding speech from brain signals is a challenging research problem. Although existing technologies have made progress in reconstructing the mel spectrograms of auditory stimuli at the word or letter level, there remain core challenges in the precise reconstruction of minute-level continuous imagined speech: traditional models struggle to balance the efficiency of temporal dependency modeling and information retention in long-sequence decoding. To address this issue, this paper proposes the Dynamic Multiscale Fusion Network (DMF2Mel), which consists of four core components: the Dynamic Contrastive Feature Aggregation Module (DC-FAM), the Hierarchical Attention-Guided Multi-Scale Network (HAMS-Net), the SplineMap attention mechanism, and the bidirectional state space module (convMamba). Specifically, the DC-FAM separates speech-related "foreground features" from noisy "background features" through local convolution and global attention mechanisms, effectively suppressing interference and enhancing the representation of transient signals. HAMS-Net, based on the U-Net framework,achieves cross-scale fusion of high-level semantics and low-level details. The SplineMap attention mechanism integrates the Adaptive Gated Kolmogorov-Arnold Network (AGKAN) to combine global context modeling with spline-based local fitting. The convMamba captures long-range temporal dependencies with linear complexity and enhances nonlinear dynamic modeling capabilities. Results on the SparrKULee dataset show that DMF2Mel achieves a Pearson correlation coefficient of 0.074 in mel spectrogram reconstruction for known subjects (a 48% improvement over the baseline) and 0.048 for unknown subjects (a 35% improvement over the baseline).Code is available at: this https URL.
#### Assessing the Alignment of Audio Representations with Timbre Similarity Ratings
 - **Authors:** Haokun Tian, Stefan Lattner, Charalampos Saitis
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2507.07764

 - **Pdf link:** https://arxiv.org/pdf/2507.07764

 - **Abstract**
 Psychoacoustical so-called "timbre spaces" map perceptual similarity ratings of instrument sounds onto low-dimensional embeddings via multidimensional scaling, but suffer from scalability issues and are incapable of generalization. Recent results from audio (music and speech) quality assessment as well as image similarity have shown that deep learning is able to produce embeddings that align well with human perception while being largely free from these constraints. Although the existing human-rated timbre similarity data is not large enough to train deep neural networks (2,614 pairwise ratings on 334 audio samples), it can serve as test-only data for audio models. In this paper, we introduce metrics to assess the alignment of diverse audio representations with human judgments of timbre similarity by comparing both the absolute values and the rankings of embedding distances to human similarity ratings. Our evaluation involves three signal-processing-based representations, twelve representations extracted from pre-trained models, and three representations extracted from a novel sound matching model. Among them, the style embeddings inspired by image style transfer, extracted from the CLAP model and the sound matching model, remarkably outperform the others, showing their potential in modeling timbre similarity.
#### SecureSpeech: Prompt-based Speaker and Content Protection
 - **Authors:** Belinda Soh Hui Hui, Xiaoxiao Miao, Xin Wang
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2507.07799

 - **Pdf link:** https://arxiv.org/pdf/2507.07799

 - **Abstract**
 Given the increasing privacy concerns from identity theft and the re-identification of speakers through content in the speech field, this paper proposes a prompt-based speech generation pipeline that ensures dual anonymization of both speaker identity and spoken content. This is addressed through 1) generating a speaker identity unlinkable to the source speaker, controlled by descriptors, and 2) replacing sensitive content within the original text using a name entity recognition model and a large language model. The pipeline utilizes the anonymized speaker identity and text to generate high-fidelity, privacy-friendly speech via a text-to-speech synthesis model. Experimental results demonstrate an achievement of significant privacy protection while maintaining a decent level of content retention and audio quality. This paper also investigates the impact of varying speaker descriptions on the utility and privacy of generated speech to determine potential biases.
#### End-to-end Acoustic-linguistic Emotion and Intent Recognition Enhanced by Semi-supervised Learning
 - **Authors:** Zhao Ren, Rathi Adarshi Rammohan, Kevin Scheck, Sheng Li, Tanja Schultz
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2507.07806

 - **Pdf link:** https://arxiv.org/pdf/2507.07806

 - **Abstract**
 Emotion and intent recognition from speech is essential and has been widely investigated in human-computer interaction. The rapid development of social media platforms, chatbots, and other technologies has led to a large volume of speech data streaming from users. Nevertheless, annotating such data manually is expensive, making it challenging to train machine learning models for recognition purposes. To this end, we propose applying semi-supervised learning to incorporate a large scale of unlabelled data alongside a relatively smaller set of labelled data. We train end-to-end acoustic and linguistic models, each employing multi-task learning for emotion and intent recognition. Two semi-supervised learning approaches, including fix-match learning and full-match learning, are compared. The experimental results demonstrate that the semi-supervised learning approaches improve model performance in speech emotion and intent recognition from both acoustic and text data. The late fusion of the best models outperforms the acoustic and text baselines by joint recognition balance metrics of 12.3% and 10.4%, respectively.
#### Edge-ASR: Towards Low-Bit Quantization of Automatic Speech Recognition Models
 - **Authors:** Chen Feng, Yicheng Lin, Shaojie Zhuo, Chenzheng Su, Ramchalam Kinattinkara Ramakrishnan, Zhaocong Yuan, Xiaopeng Zhang
 - **Subjects:** Subjects:
Sound (cs.SD); Machine Learning (cs.LG); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2507.07877

 - **Pdf link:** https://arxiv.org/pdf/2507.07877

 - **Abstract**
 Recent advances in Automatic Speech Recognition (ASR) have demonstrated remarkable accuracy and robustness in diverse audio applications, such as live transcription and voice command processing. However, deploying these models on resource constrained edge devices (e.g., IoT device, wearables) still presents substantial challenges due to strict limits on memory, compute and power. Quantization, particularly Post-Training Quantization (PTQ), offers an effective way to reduce model size and inference cost without retraining. Despite its importance, the performance implications of various advanced quantization methods and bit-width configurations on ASR models remain unclear. In this work, we present a comprehensive benchmark of eight state-of-the-art (SOTA) PTQ methods applied to two leading edge-ASR model families, Whisper and Moonshine. We systematically evaluate model performances (i.e., accuracy, memory I/O and bit operations) across seven diverse datasets from the open ASR leaderboard, analyzing the impact of quantization and various configurations on both weights and activations. Built on an extension of the LLM compression toolkit, our framework integrates edge-ASR models, diverse advanced quantization algorithms, a unified calibration and evaluation data pipeline, and detailed analysis tools. Our results characterize the trade-offs between efficiency and accuracy, demonstrating that even 3-bit quantization can succeed on high capacity models when using advanced PTQ techniques. These findings provide valuable insights for optimizing ASR models on low-power, always-on edge devices.
#### Input Conditioned Layer Dropping in Speech Foundation Models
 - **Authors:** Abdul Hannan, Daniele Falavigna, Alessio Brutti
 - **Subjects:** Subjects:
Sound (cs.SD); Computer Vision and Pattern Recognition (cs.CV); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2507.07954

 - **Pdf link:** https://arxiv.org/pdf/2507.07954

 - **Abstract**
 Curating foundation speech models for edge and IoT settings, where computational resources vary over time, requires dynamic architectures featuring adaptable reduction strategies. One emerging approach is layer dropping ($\mathcal{LD}$) which skips fraction of the layers of a backbone network during inference to reduce the computational load. This allows transforming static models into dynamic ones. However, existing approaches exhibit limitations either in the mode of selecting layers or by significantly modifying the neural architecture. To this end, we propose input-driven $\mathcal{LD}$ that employs the network's input features and a lightweight layer selecting network to determine the optimum combination of processing layers. Extensive experimentation on 4 speech and audio public benchmarks, using two different pre-trained foundation models, demonstrates the effectiveness of our approach, thoroughly outperforming random dropping and producing on-par (or better) results to early exit.


by Zyzzyva0381 (Windy). 


2025-07-11
