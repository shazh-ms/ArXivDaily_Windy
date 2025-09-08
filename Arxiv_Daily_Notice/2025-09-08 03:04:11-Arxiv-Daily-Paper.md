# Showing new listings for Monday, 8 September 2025
Auto update papers at about 2:30am UTC (10:30am Beijing time) every weekday.


阅读 `Usage.md`了解如何使用此repo实现个性化的Arxiv论文推送

See `Usage.md` for instructions on how to personalize the repo. 


Keyword list: ['text-to-speech', 'text to speech', 'tts', 'LLM-based', 'speech', 'voice']


Excluded: []


### Today: 7papers 
#### DarkStream: real-time speech anonymization with low latency
 - **Authors:** Waris Quamer, Ricardo Gutierrez-Osuna
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Computation and Language (cs.CL); Machine Learning (cs.LG)
 - **Arxiv link:** https://arxiv.org/abs/2509.04667

 - **Pdf link:** https://arxiv.org/pdf/2509.04667

 - **Abstract**
 We propose DarkStream, a streaming speech synthesis model for real-time speaker anonymization. To improve content encoding under strict latency constraints, DarkStream combines a causal waveform encoder, a short lookahead buffer, and transformer-based contextual layers. To further reduce inference time, the model generates waveforms directly via a neural vocoder, thus removing intermediate mel-spectrogram conversions. Finally, DarkStream anonymizes speaker identity by injecting a GAN-generated pseudo-speaker embedding into linguistic features from the content encoder. Evaluations show our model achieves strong anonymization, yielding close to 50% speaker verification EER (near-chance performance) on the lazy-informed attack scenario, while maintaining acceptable linguistic intelligibility (WER within 9%). By balancing low-latency, robust privacy, and minimal intelligibility degradation, DarkStream provides a practical solution for privacy-preserving real-time speech communication.
#### Say More with Less: Variable-Frame-Rate Speech Tokenization via Adaptive Clustering and Implicit Duration Coding
 - **Authors:** Rui-Chen Zheng, Wenrui Liu, Hui-Peng Du, Qinglin Zhang, Chong Deng, Qian Chen, Wen Wang, Yang Ai, Zhen-Hua Ling
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2509.04685

 - **Pdf link:** https://arxiv.org/pdf/2509.04685

 - **Abstract**
 Existing speech tokenizers typically assign a fixed number of tokens per second, regardless of the varying information density or temporal fluctuations in the speech signal. This uniform token allocation mismatches the intrinsic structure of speech, where information is distributed unevenly over time. To address this, we propose VARSTok, a VAriable-frame-Rate Speech Tokenizer that adapts token allocation based on local feature similarity. VARSTok introduces two key innovations: (1) a temporal-aware density peak clustering algorithm that adaptively segments speech into variable-length units, and (2) a novel implicit duration coding scheme that embeds both content and temporal span into a single token index, eliminating the need for auxiliary duration predictors. Extensive experiments show that VARSTok significantly outperforms strong fixed-rate baselines. Notably, it achieves superior reconstruction naturalness while using up to 23% fewer tokens than a 40 Hz fixed-frame-rate baseline. VARSTok further yields lower word error rates and improved naturalness in zero-shot text-to-speech synthesis. To the best of our knowledge, this is the first work to demonstrate that a fully dynamic, variable-frame-rate acoustic speech tokenizer can be seamlessly integrated into downstream speech language models. Speech samples are available at this https URL.
#### Layer-wise Analysis for Quality of Multilingual Synthesized Speech
 - **Authors:** Erica Cooper, Takuma Okamoto, Yamato Ohtani, Tomoki Toda, Hisashi Kawai
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2509.04830

 - **Pdf link:** https://arxiv.org/pdf/2509.04830

 - **Abstract**
 While supervised quality predictors for synthesized speech have demonstrated strong correlations with human ratings, their requirement for in-domain labeled training data hinders their generalization ability to new domains. Unsupervised approaches based on pretrained self-supervised learning (SSL) based models and automatic speech recognition (ASR) models are a promising alternative; however, little is known about how these models encode information about speech quality. Towards the goal of better understanding how different aspects of speech quality are encoded in a multilingual setting, we present a layer-wise analysis of multilingual pretrained speech models based on reference modeling. We find that features extracted from early SSL layers show correlations with human ratings of synthesized speech, and later layers of ASR models can predict quality of non-neural systems as well as intelligibility. We also demonstrate the importance of using well-matched reference data.
#### Lightweight DNN for Full-Band Speech Denoising on Mobile Devices: Exploiting Long and Short Temporal Patterns
 - **Authors:** Konstantinos Drossos, Mikko Heikkinen, Paschalis Tsiaflakis
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Machine Learning (cs.LG); Sound (cs.SD); Signal Processing (eess.SP)
 - **Arxiv link:** https://arxiv.org/abs/2509.05079

 - **Pdf link:** https://arxiv.org/pdf/2509.05079

 - **Abstract**
 Speech denoising (SD) is an important task of many, if not all, modern signal processing chains used in devices and for everyday-life applications. While there are many published and powerful deep neural network (DNN)-based methods for SD, few are optimized for resource-constrained platforms such as mobile devices. Additionally, most DNN-based methods for SD are not focusing on full-band (FB) signals, i.e. having 48 kHz sampling rate, and/or low latency cases. In this paper we present a causal, low latency, and lightweight DNN-based method for full-band SD, leveraging both short and long temporal patterns. The method is based on a modified UNet architecture employing look-back frames, temporal spanning of convolutional kernels, and recurrent neural networks for exploiting short and long temporal patterns in the signal and estimated denoising mask. The DNN operates on a causal frame-by-frame basis taking as an input the STFT magnitude, utilizes inverted bottlenecks inspired by MobileNet, employs causal instance normalization for channel-wise normalization, and achieves a real-time factor below 0.02 when deployed on a modern mobile phone. The proposed method is evaluated using established speech denoising metrics and publicly available datasets, demonstrating its effectiveness in achieving an (SI-)SDR value that outperforms existing FB and low latency SD methods.
#### MEAN-RIR: Multi-Modal Environment-Aware Network for Robust Room Impulse Response Estimation
 - **Authors:** Jiajian Chen, Jiakang Chen, Hang Chen, Qing Wang, Yu Gao, Jun Du
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2509.05205

 - **Pdf link:** https://arxiv.org/pdf/2509.05205

 - **Abstract**
 This paper presents a Multi-Modal Environment-Aware Network (MEAN-RIR), which uses an encoder-decoder framework to predict room impulse response (RIR) based on multi-level environmental information from audio, visual, and textual sources. Specifically, reverberant speech capturing room acoustic properties serves as the primary input, which is combined with panoramic images and text descriptions as supplementary inputs. Each input is processed by its respective encoder, and the outputs are fed into cross-attention modules to enable effective interaction between different modalities. The MEAN-RIR decoder generates two distinct components: the first component captures the direct sound and early reflections, while the second produces masks that modulate learnable filtered noise to synthesize the late reverberation. These two components are mixed to reconstruct the final RIR. The results show that MEAN-RIR significantly improves RIR estimation, with notable gains in acoustic parameters.
#### Serialized Output Prompting for Large Language Model-based Multi-Talker Speech Recognition
 - **Authors:** Hao Shi, Yusuke Fujita, Tomoya Mizumoto, Lianbo Liu, Atsushi Kojima, Yui Sudo
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Artificial Intelligence (cs.AI); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2509.04488

 - **Pdf link:** https://arxiv.org/pdf/2509.04488

 - **Abstract**
 Prompts are crucial for task definition and for improving the performance of large language models (LLM)-based systems. However, existing LLM-based multi-talker (MT) automatic speech recognition (ASR) systems either omit prompts or rely on simple task-definition prompts, with no prior work exploring the design of prompts to enhance performance. In this paper, we propose extracting serialized output prompts (SOP) and explicitly guiding the LLM using structured prompts to improve system performance (SOP-MT-ASR). A Separator and serialized Connectionist Temporal Classification (CTC) layers are inserted after the speech encoder to separate and extract MT content from the mixed speech encoding in a first-speaking-first-out manner. Subsequently, the SOP, which serves as a prompt for LLMs, is obtained by decoding the serialized CTC outputs using greedy search. To train the model effectively, we design a three-stage training strategy, consisting of serialized output training (SOT) fine-tuning, serialized speech information extraction, and SOP-based adaptation. Experimental results on the LibriMix dataset show that, although the LLM-based SOT model performs well in the two-talker scenario, it fails to fully leverage LLMs under more complex conditions, such as the three-talker scenario. The proposed SOP approach significantly improved performance under both two- and three-talker conditions.
#### Quantum Fourier Transform Based Denoising: Unitary Filtering for Enhanced Speech Clarity
 - **Authors:** Rajeshwar Tripathi, Sahil Tomar, Sandeep Kumar, Monika Aggarwal
 - **Subjects:** Subjects:
Sound (cs.SD); Emerging Technologies (cs.ET); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2509.04851

 - **Pdf link:** https://arxiv.org/pdf/2509.04851

 - **Abstract**
 This paper introduces a quantum-inspired denoising framework that integrates the Quantum Fourier Transform (QFT) into classical audio enhancement pipelines. Unlike conventional Fast Fourier Transform (FFT) based methods, QFT provides a unitary transformation with global phase coherence and energy preservation, enabling improved discrimination between speech and noise. The proposed approach replaces FFT in Wiener and spectral subtraction filters with a QFT operator, ensuring consistent hyperparameter settings for fair comparison. Experiments on clean speech, synthetic tones, and noisy mixtures across diverse signal to noise ratio (SNR) conditions, demonstrate statistically significant gains in SNR, with up to 15 dB improvement and reduced artifact generation. Results confirm that QFT based denoising offers robustness under low SNR and nonstationary noise scenarios without additional computational overhead, highlighting its potential as a scalable pathway toward quantum-enhanced speech processing.


by Zyzzyva0381 (Windy). 


2025-09-08
