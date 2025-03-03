# Showing new listings for Friday, 28 February 2025
Auto update papers at about 2:30am UTC (10:30am Beijing time) every weekday.


阅读 `Usage.md`了解如何使用此repo实现个性化的Arxiv论文推送

See `Usage.md` for instructions on how to personalize the repo. 


Keyword list: ['acoustic', 'reinforcement learning', 'reverb', 'meta', 'separate', 'reconstruction', 'noise', 'enhance', 'localization', 'speech']


Excluded: []


### Today: 10papers 
#### PrimeK-Net: Multi-scale Spectral Learning via Group Prime-Kernel Convolutional Neural Networks for Single Channel Speech Enhancement
 - **Authors:** Zizhen Lin, Junyu Wang, Ruili Li, Fei Shen, Xi Xuan
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2502.19906

 - **Pdf link:** https://arxiv.org/pdf/2502.19906

 - **Abstract**
 Single-channel speech enhancement is a challenging ill-posed problem focused on estimating clean speech from degraded signals. Existing studies have demonstrated the competitive performance of combining convolutional neural networks (CNNs) with Transformers in speech enhancement tasks. However, existing frameworks have not sufficiently addressed computational efficiency and have overlooked the natural multi-scale distribution of the spectrum. Additionally, the potential of CNNs in speech enhancement has yet to be fully realized. To address these issues, this study proposes a Deep Separable Dilated Dense Block (DSDDB) and a Group Prime Kernel Feedforward Channel Attention (GPFCA) module. Specifically, the DSDDB introduces higher parameter and computational efficiency to the Encoder/Decoder of existing frameworks. The GPFCA module replaces the position of the Conformer, extracting deep temporal and frequency features of the spectrum with linear complexity. The GPFCA leverages the proposed Group Prime Kernel Feedforward Network (GPFN) to integrate multi-granularity long-range, medium-range, and short-range receptive fields, while utilizing the properties of prime numbers to avoid periodic overlap effects. Experimental results demonstrate that PrimeK-Net, proposed in this study, achieves state-of-the-art (SOTA) performance on the VoiceBank+Demand dataset, reaching a PESQ score of 3.61 with only 1.41M parameters.
#### CleanMel: Mel-Spectrogram Enhancement for Improving Both Speech Quality and ASR
 - **Authors:** Nian Shao, Rui Zhou, Pengyu Wang, Xian Li, Ying Fang, Yujie Yang, Xiaofei Li
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Artificial Intelligence (cs.AI); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2502.20040

 - **Pdf link:** https://arxiv.org/pdf/2502.20040

 - **Abstract**
 In this work, we propose CleanMel, a single-channel Mel-spectrogram denoising and dereverberation network for improving both speech quality and automatic speech recognition (ASR) performance. The proposed network takes as input the noisy and reverberant microphone recording and predicts the corresponding clean Mel-spectrogram. The enhanced Mel-spectrogram can be either transformed to speech waveform with a neural vocoder or directly used for ASR. The proposed network is composed of interleaved cross-band and narrow-band processing in the Mel-frequency domain, for learning the full-band spectral pattern and the narrow-band properties of signals, respectively. Compared to linear-frequency domain or time-domain speech enhancement, the key advantage of Mel-spectrogram enhancement is that Mel-frequency presents speech in a more compact way and thus is easier to learn, which will benefit both speech quality and ASR. Experimental results on four English and one Chinese datasets demonstrate a significant improvement in both speech quality and ASR performance achieved by the proposed model. Code and audio examples of our model are available online in this https URL.
#### UniCodec: Unified Audio Codec with Single Domain-Adaptive Codebook
 - **Authors:** Yidi Jiang, Qian Chen, Shengpeng Ji, Yu Xi, Wen Wang, Chong Zhang, Xianghu Yue, ShiLiang Zhang, Haizhou Li
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2502.20067

 - **Pdf link:** https://arxiv.org/pdf/2502.20067

 - **Abstract**
 The emergence of audio language models is empowered by neural audio codecs, which establish critical mappings between continuous waveforms and discrete tokens compatible with language model paradigms. The evolutionary trends from multi-layer residual vector quantizer to single-layer quantizer are beneficial for language-autoregressive decoding. However, the capability to handle multi-domain audio signals through a single codebook remains constrained by inter-domain distribution discrepancies. In this work, we introduce UniCodec, a unified audio codec with a single codebook to support multi-domain audio data, including speech, music, and sound. To achieve this, we propose a partitioned domain-adaptive codebook method and domain Mixture-of-Experts strategy to capture the distinct characteristics of each audio domain. Furthermore, to enrich the semantic density of the codec without auxiliary modules, we propose a self-supervised mask prediction modeling approach. Comprehensive objective and subjective evaluations demonstrate that UniCodec achieves excellent audio reconstruction performance across the three audio domains, outperforming existing unified neural codecs with a single codebook, and even surpasses state-of-the-art domain-specific codecs on both acoustic and semantic representation capabilities.
#### Filtro Adaptativo y Modulo de Grabacion en Dispositivo Para Mejora en la Calidad de Audicion
 - **Authors:** Carlos Elihu Palomino Torres, Francisco Claudio Chichipe Mondragon, Frank Antonio Siesquen Rodriguez, Mariana Alexandra Huaynate Leon
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2502.19444

 - **Pdf link:** https://arxiv.org/pdf/2502.19444

 - **Abstract**
 This project presents the development of a real-time auditory enhancement system utilizing an ESP32, an LMS adaptive filter, and artificial intelligence techniques. An I2S INMP44 microphone captures the sound, which is dynamically processed to suppress noise before being played through a MAX98357 speaker. The system continuously adapts to varying acoustic environments, ensuring improved speech clarity and an optimized listening experience
#### When Large Language Models Meet Speech: A Survey on Integration Approaches
 - **Authors:** Zhengdong Yang, Shuichiro Shimizu, Yahan Yu, Chenhui Chu
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2502.19548

 - **Pdf link:** https://arxiv.org/pdf/2502.19548

 - **Abstract**
 Recent advancements in large language models (LLMs) have spurred interest in expanding their application beyond text-based tasks. A large number of studies have explored integrating other modalities with LLMs, notably speech modality, which is naturally related to text. This paper surveys the integration of speech with LLMs, categorizing the methodologies into three primary approaches: text-based, latent-representation-based, and audio-token-based integration. We also demonstrate how these methods are applied across various speech-related applications and highlight the challenges in this field to offer inspiration for
#### Does Your Voice Assistant Remember? Analyzing Conversational Context Recall and Utilization in Voice Interaction Models
 - **Authors:** Heeseung Kim, Che Hyun Lee, Sangkwon Park, Jiheum Yeom, Nohil Park, Sangwon Yu, Sungroh Yoon
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2502.19759

 - **Pdf link:** https://arxiv.org/pdf/2502.19759

 - **Abstract**
 Recent advancements in multi-turn voice interaction models have improved user-model communication. However, while closed-source models effectively retain and recall past utterances, whether open-source models share this ability remains unexplored. To fill this gap, we systematically evaluate how well open-source interaction models utilize past utterances using ContextDialog, a benchmark we proposed for this purpose. Our findings show that speech-based models have more difficulty than text-based ones, especially when recalling information conveyed in speech, and even with retrieval-augmented generation, models still struggle with questions about past utterances. These insights highlight key limitations in open-source models and suggest ways to improve memory retention and retrieval robustness.
#### DiffCSS: Diverse and Expressive Conversational Speech Synthesis with Diffusion Models
 - **Authors:** Weihao wu, Zhiwei Lin, Yixuan Zhou, Jingbei Li, Rui Niu, Qinghua Wu, Songjun Cao, Long Ma, Zhiyong Wu
 - **Subjects:** Subjects:
Sound (cs.SD); Artificial Intelligence (cs.AI); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2502.19924

 - **Pdf link:** https://arxiv.org/pdf/2502.19924

 - **Abstract**
 Conversational speech synthesis (CSS) aims to synthesize both contextually appropriate and expressive speech, and considerable efforts have been made to enhance the understanding of conversational context. However, existing CSS systems are limited to deterministic prediction, overlooking the diversity of potential responses. Moreover, they rarely employ language model (LM)-based TTS backbones, limiting the naturalness and quality of synthesized speech. To address these issues, in this paper, we propose DiffCSS, an innovative CSS framework that leverages diffusion models and an LM-based TTS backbone to generate diverse, expressive, and contextually coherent speech. A diffusion-based context-aware prosody predictor is proposed to sample diverse prosody embeddings conditioned on multimodal conversational context. Then a prosody-controllable LM-based TTS backbone is developed to synthesize high-quality speech with sampled prosody embeddings. Experimental results demonstrate that the synthesized speech from DiffCSS is more diverse, contextually coherent, and expressive than existing CSS systems
#### DIN-CTS: Low-Complexity Depthwise-Inception Neural Network with Contrastive Training Strategy for Deepfake Speech Detection
 - **Authors:** Lam Pham, Dat Tran, Florian Skopik, Alexander Schindler, Silvia Poletti, Fischinger David, Martin Boyer
 - **Subjects:** Subjects:
Sound (cs.SD); Cryptography and Security (cs.CR); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2502.20225

 - **Pdf link:** https://arxiv.org/pdf/2502.20225

 - **Abstract**
 In this paper, we propose a deep neural network approach for deepfake speech detection (DSD) based on a lowcomplexity Depthwise-Inception Network (DIN) trained with a contrastive training strategy (CTS). In this framework, input audio recordings are first transformed into spectrograms using Short-Time Fourier Transform (STFT) and Linear Filter (LF), which are then used to train the DIN. Once trained, the DIN processes bonafide utterances to extract audio embeddings, which are used to construct a Gaussian distribution representing genuine speech. Deepfake detection is then performed by computing the distance between a test utterance and this distribution to determine whether the utterance is fake or bonafide. To evaluate our proposed systems, we conducted extensive experiments on the benchmark dataset of ASVspoof 2019 LA. The experimental results demonstrate the effectiveness of combining the Depthwise-Inception Network with the contrastive learning strategy in distinguishing between fake and bonafide utterances. We achieved Equal Error Rate (EER), Accuracy (Acc.), F1, AUC scores of 4.6%, 95.4%, 97.3%, and 98.9% respectively using a single, low-complexity DIN with just 1.77 M parameters and 985 M FLOPS on short audio segments (4 seconds). Furthermore, our proposed system outperforms the single-system submissions in the ASVspoof 2019 LA challenge, showcasing its potential for real-time applications.
#### Adapting Automatic Speech Recognition for Accented Air Traffic Control Communications
 - **Authors:** Marcus Yu Zhe Wee, Justin Juin Hng Wong, Lynus Lim, Joe Yu Wei Tan, Prannaya Gupta, Dillion Lim, En Hao Tew, Aloysius Keng Siew Han, Yong Zhi Lim
 - **Subjects:** Subjects:
Machine Learning (cs.LG); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2502.20311

 - **Pdf link:** https://arxiv.org/pdf/2502.20311

 - **Abstract**
 Effective communication in Air Traffic Control (ATC) is critical to maintaining aviation safety, yet the challenges posed by accented English remain largely unaddressed in Automatic Speech Recognition (ASR) systems. Existing models struggle with transcription accuracy for Southeast Asian-accented (SEA-accented) speech, particularly in noisy ATC environments. This study presents the development of ASR models fine-tuned specifically for Southeast Asian accents using a newly created dataset. Our research achieves significant improvements, achieving a Word Error Rate (WER) of 0.0982 or 9.82% on SEA-accented ATC speech. Additionally, the paper highlights the importance of region-specific datasets and accent-focused training, offering a pathway for deploying ASR systems in resource-constrained military operations. The findings emphasize the need for noise-robust training techniques and region-specific datasets to improve transcription accuracy for non-Western accents in ATC communications.
#### On Adversarial Attacks In Acoustic Drone Localization
 - **Authors:** Tamir Shor, Chaim Baskin, Alex Bronstein
 - **Subjects:** Subjects:
Sound (cs.SD); Robotics (cs.RO); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2502.20325

 - **Pdf link:** https://arxiv.org/pdf/2502.20325

 - **Abstract**
 Multi-rotor aerial autonomous vehicles (MAVs, more widely known as "drones") have been generating increased interest in recent years due to their growing applicability in a vast and diverse range of fields (e.g., agriculture, commercial delivery, search and rescue). The sensitivity of visual-based methods to lighting conditions and occlusions had prompted growing study of navigation reliant on other modalities, such as acoustic sensing. A major concern in using drones in scale for tasks in non-controlled environments is the potential threat of adversarial attacks over their navigational systems, exposing users to mission-critical failures, security breaches, and compromised safety outcomes that can endanger operators and bystanders. While previous work shows impressive progress in acoustic-based drone localization, prior research in adversarial attacks over drone navigation only addresses visual sensing-based systems. In this work, we aim to compensate for this gap by supplying a comprehensive analysis of the effect of PGD adversarial attacks over acoustic drone localization. We furthermore develop an algorithm for adversarial perturbation recovery, capable of markedly diminishing the affect of such attacks in our setting. The code for reproducing all experiments will be released upon publication.


by Zyzzyva0381 (Windy). 


2025-03-03
