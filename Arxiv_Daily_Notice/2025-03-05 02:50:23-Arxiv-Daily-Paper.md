# Showing new listings for Tuesday, 4 March 2025
Auto update papers at about 2:30am UTC (10:30am Beijing time) every weekday.


阅读 `Usage.md`了解如何使用此repo实现个性化的Arxiv论文推送

See `Usage.md` for instructions on how to personalize the repo. 


Keyword list: ['acoustic', 'reinforcement learning', 'reverb', 'meta', 'separate', 'reconstruction', 'noise', 'enhance', 'localization', 'speech']


Excluded: []


### Today: 14papers 
#### UL-UNAS: Ultra-Lightweight U-Nets for Real-Time Speech Enhancement via Network Architecture Search
 - **Authors:** Xiaobin Rong, Dahan Wang, Yuxiang Hu, Changbao Zhu, Kai Chen, Jing Lu
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2503.00340

 - **Pdf link:** https://arxiv.org/pdf/2503.00340

 - **Abstract**
 Lightweight models are essential for real-time speech enhancement applications. In recent years, there has been a growing trend toward developing increasingly compact models for speech enhancement. In this paper, we propose an Ultra-Lightweight U-net optimized by Network Architecture Search (UL-UNAS), which is suitable for implementation in low-footprint devices. Firstly, we explore the application of various efficient convolutional blocks within the U-Net framework to identify the most promising candidates. Secondly, we introduce two boosting components to enhance the capacity of these convolutional blocks: a novel activation function named affine PReLU and a causal time-frequency attention module. Furthermore, we leverage neural architecture search to discover an optimal architecture within our carefully designed search space. By integrating the above strategies, UL-UNAS not only significantly outperforms the latest ultra-lightweight models with the same or lower computational complexity, but also delivers competitive performance compared to recent baseline models that require substantially higher computational resources.
#### LLaSE-G1: Incentivizing Generalization Capability for LLaMA-based Speech Enhancement
 - **Authors:** Boyi Kang, Xinfa Zhu, Zihan Zhang, Zhen Ye, Mingshuai Liu, Ziqian Wang, Yike Zhu, Guobin Ma, Jun Chen, Longshuai Xiao, Chao Weng, Wei Xue, Lei Xie
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Artificial Intelligence (cs.AI); Computation and Language (cs.CL); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2503.00493

 - **Pdf link:** https://arxiv.org/pdf/2503.00493

 - **Abstract**
 Recent advancements in language models (LMs) have demonstrated strong capabilities in semantic understanding and contextual modeling, which have flourished in generative speech enhancement (SE). However, many LM-based SE approaches primarily focus on semantic information, often neglecting the critical role of acoustic information, which leads to acoustic inconsistency after enhancement and limited generalization across diverse SE tasks. In this paper, we introduce LLaSE-G1, a LLaMA-based language model that incentivizes generalization capabilities for speech enhancement. LLaSE-G1 offers the following key contributions: First, to mitigate acoustic inconsistency, LLaSE-G1 employs continuous representations from WavLM as input and predicts speech tokens from X-Codec2, maximizing acoustic preservation. Second, to promote generalization capability, LLaSE-G1 introduces dual-channel inputs and outputs, unifying multiple SE tasks without requiring task-specific IDs. Third, LLaSE-G1 outperforms prior task-specific discriminative and generative SE models, demonstrating scaling effects at test time and emerging capabilities for unseen SE tasks. Additionally, we release our code and models to support further research in this area.
#### UniWav: Towards Unified Pre-training for Speech Representation Learning and Generation
 - **Authors:** Alexander H. Liu, Sang-gil Lee, Chao-Han Huck Yang, Yuan Gong, Yu-Chiang Frank Wang, James R. Glass, Rafael Valle, Bryan Catanzaro
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Computation and Language (cs.CL); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2503.00733

 - **Pdf link:** https://arxiv.org/pdf/2503.00733

 - **Abstract**
 Pre-training and representation learning have been playing an increasingly important role in modern speech processing. Nevertheless, different applications have been relying on different foundation models, since predominant pre-training techniques are either designed for discriminative tasks or generative tasks. In this work, we make the first attempt at building a unified pre-training framework for both types of tasks in speech. We show that with the appropriate design choices for pre-training, one can jointly learn a representation encoder and generative audio decoder that can be applied to both types of tasks. We propose UniWav, an encoder-decoder framework designed to unify pre-training representation learning and generative tasks. On speech recognition, text-to-speech, and speech tokenization, UniWav achieves comparable performance to different existing foundation models, each trained on a specific task. Our findings suggest that a single general-purpose foundation model for speech can be built to replace different foundation models, reducing the overhead and cost of pre-training.
#### InspireMusic: Integrating Super Resolution and Large Language Model for High-Fidelity Long-Form Music Generation
 - **Authors:** Chong Zhang, Yukun Ma, Qian Chen, Wen Wang, Shengkui Zhao, Zexu Pan, Hao Wang, Chongjia Ni, Trung Hieu Nguyen, Kun Zhou, Yidi Jiang, Chaohong Tan, Zhifu Gao, Zhihao Du, Bin Ma
 - **Subjects:** Subjects:
Sound (cs.SD); Artificial Intelligence (cs.AI); Computation and Language (cs.CL); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2503.00084

 - **Pdf link:** https://arxiv.org/pdf/2503.00084

 - **Abstract**
 We introduce InspireMusic, a framework integrated super resolution and large language model for high-fidelity long-form music generation. A unified framework generates high-fidelity music, songs, and audio, which incorporates an autoregressive transformer with a super-resolution flow-matching model. This framework enables the controllable generation of high-fidelity long-form music at a higher sampling rate from both text and audio prompts. Our model differs from previous approaches, as we utilize an audio tokenizer with one codebook that contains richer semantic information, thereby reducing training costs and enhancing efficiency. This combination enables us to achieve high-quality audio generation with long-form coherence of up to $8$ minutes. Then, an autoregressive transformer model based on Qwen 2.5 predicts audio tokens. Next, we employ a super-resolution flow-matching model to generate high-sampling rate audio with fine-grained details learned from an acoustic codec model. Comprehensive experiments show that the InspireMusic-1.5B-Long model has a comparable performance to recent top-tier open-source systems, including MusicGen and Stable Audio 2.0, on subjective and objective evaluations. The code and pre-trained models are released at this https URL.
#### BGM2Pose: Active 3D Human Pose Estimation with Non-Stationary Sounds
 - **Authors:** Yuto Shibata, Yusuke Oumi, Go Irie, Akisato Kimura, Yoshimitsu Aoki, Mariko Isogawa
 - **Subjects:** Subjects:
Computer Vision and Pattern Recognition (cs.CV); Artificial Intelligence (cs.AI); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2503.00389

 - **Pdf link:** https://arxiv.org/pdf/2503.00389

 - **Abstract**
 We propose BGM2Pose, a non-invasive 3D human pose estimation method using arbitrary music (e.g., background music) as active sensing signals. Unlike existing approaches that significantly limit practicality by employing intrusive chirp signals within the audible range, our method utilizes natural music that causes minimal discomfort to humans. Estimating human poses from standard music presents significant challenges. In contrast to sound sources specifically designed for measurement, regular music varies in both volume and pitch. These dynamic changes in signals caused by music are inevitably mixed with alterations in the sound field resulting from human motion, making it hard to extract reliable cues for pose estimation. To address these challenges, BGM2Pose introduces a Contrastive Pose Extraction Module that employs contrastive learning and hard negative sampling to eliminate musical components from the recorded data, isolating the pose information. Additionally, we propose a Frequency-wise Attention Module that enables the model to focus on subtle acoustic variations attributable to human movement by dynamically computing attention across frequency bands. Experiments suggest that our method outperforms the existing methods, demonstrating substantial potential for real-world applications. Our datasets and code will be made publicly available.
#### PodAgent: A Comprehensive Framework for Podcast Generation
 - **Authors:** Yujia Xiao, Lei He, Haohan Guo, Fenglong Xie, Tan Lee
 - **Subjects:** Subjects:
Sound (cs.SD); Artificial Intelligence (cs.AI); Multiagent Systems (cs.MA); Multimedia (cs.MM); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2503.00455

 - **Pdf link:** https://arxiv.org/pdf/2503.00455

 - **Abstract**
 Existing Existing automatic audio generation methods struggle to generate podcast-like audio programs effectively. The key challenges lie in in-depth content generation, appropriate and expressive voice production. This paper proposed PodAgent, a comprehensive framework for creating audio programs. PodAgent 1) generates informative topic-discussion content by designing a Host-Guest-Writer multi-agent collaboration system, 2) builds a voice pool for suitable voice-role matching and 3) utilizes LLM-enhanced speech synthesis method to generate expressive conversational speech. Given the absence of standardized evaluation criteria for podcast-like audio generation, we developed comprehensive assessment guidelines to effectively evaluate the model's performance. Experimental results demonstrate PodAgent's effectiveness, significantly surpassing direct GPT-4 generation in topic-discussion dialogue content, achieving an 87.4% voice-matching accuracy, and producing more expressive speech through LLM-guided synthesis. Demo page: this https URL. Source code: this https URL.
#### Acoustic Anomaly Detection on UAM Propeller Defect with Acoustic dataset for Crack of drone Propeller (ADCP)
 - **Authors:** Juho Lee, Donghyun Yoon, Gumoon Jeong, Hyeoncheol Kim
 - **Subjects:** Subjects:
Sound (cs.SD); Emerging Technologies (cs.ET); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2503.00790

 - **Pdf link:** https://arxiv.org/pdf/2503.00790

 - **Abstract**
 The imminent commercialization of UAM requires stable, AI-based maintenance systems to ensure safety for both passengers and pedestrians. This paper presents a methodology for non-destructively detecting cracks in UAM propellers using drone propeller sound datasets. Normal operating sounds were recorded, and abnormal sounds (categorized as ripped and broken) were differentiated by varying the microphone-propeller angle and throttle power. Our novel approach integrates FFT and STFT preprocessing techniques to capture both global frequency patterns and local time-frequency variations, thereby enhancing anomaly detection performance. The constructed Acoustic Dataset for Crack of Drone Propeller (ADCP) demonstrates the potential for detecting propeller cracks and lays the groundwork for future UAM maintenance applications.
#### Unveiling Biases while Embracing Sustainability: Assessing the Dual Challenges of Automatic Speech Recognition Systems
 - **Authors:** Ajinkya Kulkarni, Atharva Kulkarni, Miguel Couceiro, Isabel Trancoso
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2503.00907

 - **Pdf link:** https://arxiv.org/pdf/2503.00907

 - **Abstract**
 In this paper, we present a bias and sustainability focused investigation of Automatic Speech Recognition (ASR) systems, namely Whisper and Massively Multilingual Speech (MMS), which have achieved state-of-the-art (SOTA) performances. Despite their improved performance in controlled settings, there remains a critical gap in understanding their efficacy and equity in real-world scenarios. We analyze ASR biases w.r.t. gender, accent, and age group, as well as their effect on downstream tasks. In addition, we examine the environmental impact of ASR systems, scrutinizing the use of large acoustic models on carbon emission and energy consumption. We also provide insights into our empirical analyses, offering a valuable contribution to the claims surrounding bias and sustainability in ASR systems.
#### Exploiting Vulnerabilities in Speech Translation Systems through Targeted Adversarial Attacks
 - **Authors:** Chang Liu, Haolin Wu, Xi Yang, Kui Zhang, Cong Wu, Weiming Zhang, Nenghai Yu, Tianwei Zhang, Qing Guo, Jie Zhang
 - **Subjects:** Subjects:
Sound (cs.SD); Artificial Intelligence (cs.AI); Cryptography and Security (cs.CR); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2503.00957

 - **Pdf link:** https://arxiv.org/pdf/2503.00957

 - **Abstract**
 As speech translation (ST) systems become increasingly prevalent, understanding their vulnerabilities is crucial for ensuring robust and reliable communication. However, limited work has explored this issue in depth. This paper explores methods of compromising these systems through imperceptible audio manipulations. Specifically, we present two innovative approaches: (1) the injection of perturbation into source audio, and (2) the generation of adversarial music designed to guide targeted translation, while also conducting more practical over-the-air attacks in the physical world. Our experiments reveal that carefully crafted audio perturbations can mislead translation models to produce targeted, harmful outputs, while adversarial music achieve this goal more covertly, exploiting the natural imperceptibility of music. These attacks prove effective across multiple languages and translation models, highlighting a systemic vulnerability in current ST architectures. The implications of this research extend beyond immediate security concerns, shedding light on the interpretability and robustness of neural speech processing systems. Our findings underscore the need for advanced defense mechanisms and more resilient architectures in the realm of audio systems. More details and samples can be found at this https URL.
#### Talking Turns: Benchmarking Audio Foundation Models on Turn-Taking Dynamics
 - **Authors:** Siddhant Arora, Zhiyun Lu, Chung-Cheng Chiu, Ruoming Pang, Shinji Watanabe
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2503.01174

 - **Pdf link:** https://arxiv.org/pdf/2503.01174

 - **Abstract**
 The recent wave of audio foundation models (FMs) could provide new capabilities for conversational modeling. However, there have been limited efforts to evaluate these audio FMs comprehensively on their ability to have natural and interactive conversations. To engage in meaningful conversation with the end user, we would want the FMs to additionally perform a fluent succession of turns without too much overlapping speech or long stretches of silence. Inspired by this, we ask whether the recently proposed audio FMs can understand, predict, and perform turn-taking events? To answer this, we propose a novel evaluation protocol that can assess spoken dialog system's turn-taking capabilities using a supervised model as a judge that has been trained to predict turn-taking events in human-human conversations. Using this protocol, we present the first comprehensive user study that evaluates existing spoken dialogue systems on their ability to perform turn-taking events and reveal many interesting insights, such as they sometimes do not understand when to speak up, can interrupt too aggressively and rarely backchannel. We further evaluate multiple open-source and proprietary audio FMs accessible through APIs on carefully curated test benchmarks from Switchboard to measure their ability to understand and predict turn-taking events and identify significant room for improvement. We will open source our evaluation platform to promote the development of advanced conversational AI systems.
#### Voice Cloning for Dysarthric Speech Synthesis: Addressing Data Scarcity in Speech-Language Pathology
 - **Authors:** Birger Moell, Fredrik Sand Aronsson
 - **Subjects:** Subjects:
Sound (cs.SD); Artificial Intelligence (cs.AI); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2503.01266

 - **Pdf link:** https://arxiv.org/pdf/2503.01266

 - **Abstract**
 This study explores voice cloning to generate synthetic speech replicating the unique patterns of individuals with dysarthria. Using the TORGO dataset, we address data scarcity and privacy challenges in speech-language pathology. Our contributions include demonstrating that voice cloning preserves dysarthric speech characteristics, analyzing differences between real and synthetic data, and discussing implications for diagnostics, rehabilitation, and communication. We cloned voices from dysarthric and control speakers using a commercial platform, ensuring gender-matched synthetic voices. A licensed speech-language pathologist (SLP) evaluated a subset for dysarthria, speaker gender, and synthetic indicators. The SLP correctly identified dysarthria in all cases and speaker gender in 95% but misclassified 30% of synthetic samples as real, indicating high realism. Our results suggest synthetic speech effectively captures disordered characteristics and that voice cloning has advanced to produce high-quality data resembling real speech, even to trained professionals. This has critical implications for healthcare, where synthetic data can mitigate data scarcity, protect privacy, and enhance AI-driven diagnostics. By enabling the creation of diverse, high-quality speech datasets, voice cloning can improve generalizable models, personalize therapy, and advance assistive technologies for dysarthria. We publicly release our synthetic dataset to foster further research and collaboration, aiming to develop robust models that improve patient outcomes in speech-language pathology.
#### Streaming Piano Transcription Based on Consistent Onset and Offset Decoding with Sustain Pedal Detection
 - **Authors:** Weixing Wei, Jiahao Zhao, Yulun Wu, Kazuyoshi Yoshii
 - **Subjects:** Subjects:
Sound (cs.SD); Information Retrieval (cs.IR); Multimedia (cs.MM); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2503.01362

 - **Pdf link:** https://arxiv.org/pdf/2503.01362

 - **Abstract**
 This paper describes a streaming audio-to-MIDI piano transcription approach that aims to sequentially translate a music signal into a sequence of note onset and offset events. The sequence-to-sequence nature of this task may call for the computationally-intensive transformer model for better performance, which has recently been used for offline transcription benchmarks and could be extended for streaming transcription with causal attention mechanisms. We assume that the performance limitation of this naive approach lies in the decoder. Although time-frequency features useful for onset detection are considerably different from those for offset detection, the single decoder is trained to output a mixed sequence of onset and offset events without guarantee of the correspondence between the onset and offset events of the same note. To overcome this limitation, we propose a streaming encoder-decoder model that uses a convolutional encoder aggregating local acoustic features, followed by an autoregressive Transformer decoder detecting a variable number of onset events and another decoder detecting the offset events for the active pitches with validation of the sustain pedal at each time frame. Experiments using the MAESTRO dataset showed that the proposed streaming method performed comparably with or even better than the state-of-the-art offline methods while significantly reducing the computational cost.
#### FlowDec: A flow-based full-band general audio codec with high perceptual quality
 - **Authors:** Simon Welker, Matthew Le, Ricky T. Q. Chen, Wei-Ning Hsu, Timo Gerkmann, Alexander Richard, Yi-Chiao Wu
 - **Subjects:** Subjects:
Sound (cs.SD); Machine Learning (cs.LG); Audio and Speech Processing (eess.AS); Signal Processing (eess.SP)
 - **Arxiv link:** https://arxiv.org/abs/2503.01485

 - **Pdf link:** https://arxiv.org/pdf/2503.01485

 - **Abstract**
 We propose FlowDec, a neural full-band audio codec for general audio sampled at 48 kHz that combines non-adversarial codec training with a stochastic postfilter based on a novel conditional flow matching method. Compared to the prior work ScoreDec which is based on score matching, we generalize from speech to general audio and move from 24 kbit/s to as low as 4 kbit/s, while improving output quality and reducing the required postfilter DNN evaluations from 60 to 6 without any fine-tuning or distillation techniques. We provide theoretical insights and geometric intuitions for our approach in comparison to ScoreDec as well as another recent work that uses flow matching, and conduct ablation studies on our proposed components. We show that FlowDec is a competitive alternative to the recent GAN-dominated stream of neural codecs, achieving FAD scores better than those of the established GAN-based codec DAC and listening test scores that are on par, and producing qualitatively more natural reconstructions for speech and harmonic structures in music.
#### Spark-TTS: An Efficient LLM-Based Text-to-Speech Model with Single-Stream Decoupled Speech Tokens
 - **Authors:** Xinsheng Wang, Mingqi Jiang, Ziyang Ma, Ziyu Zhang, Songxiang Liu, Linqin Li, Zheng Liang, Qixi Zheng, Rui Wang, Xiaoqin Feng, Weizhen Bian, Zhen Ye, Sitong Cheng, Ruibin Yuan, Zhixian Zhao, Xinfa Zhu, Jiahao Pan, Liumeng Xue, Pengcheng Zhu, Yunlin Chen, Zhifei Li, Xie Chen, Lei Xie, Yike Guo, Wei Xue
 - **Subjects:** Subjects:
Sound (cs.SD); Artificial Intelligence (cs.AI); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2503.01710

 - **Pdf link:** https://arxiv.org/pdf/2503.01710

 - **Abstract**
 Recent advancements in large language models (LLMs) have driven significant progress in zero-shot text-to-speech (TTS) synthesis. However, existing foundation models rely on multi-stage processing or complex architectures for predicting multiple codebooks, limiting efficiency and integration flexibility. To overcome these challenges, we introduce Spark-TTS, a novel system powered by BiCodec, a single-stream speech codec that decomposes speech into two complementary token types: low-bitrate semantic tokens for linguistic content and fixed-length global tokens for speaker attributes. This disentangled representation, combined with the Qwen2.5 LLM and a chain-of-thought (CoT) generation approach, enables both coarse-grained control (e.g., gender, speaking style) and fine-grained adjustments (e.g., precise pitch values, speaking rate). To facilitate research in controllable TTS, we introduce VoxBox, a meticulously curated 100,000-hour dataset with comprehensive attribute annotations. Extensive experiments demonstrate that Spark-TTS not only achieves state-of-the-art zero-shot voice cloning but also generates highly customizable voices that surpass the limitations of reference-based synthesis. Source code, pre-trained models, and audio samples are available at this https URL.


by Zyzzyva0381 (Windy). 


2025-03-05
