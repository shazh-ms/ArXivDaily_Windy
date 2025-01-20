# Showing new listings for Monday, 20 January 2025
Auto update papers at about 2:30am UTC (10:30am Beijing time) every weekday.


阅读 `Usage.md`了解如何使用此repo实现个性化的Arxiv论文推送

See `Usage.md` for instructions on how to personalize the repo. 


Keyword list: ['acoustic', 'reinforcement learning', 'reverb', 'meta', 'separate', 'reconstruction', 'noise', 'enhance', 'localization', 'speech']


Excluded: []


### Today: 7papers 
#### CLAP-S: Support Set Based Adaptation for Downstream Fiber-optic Acoustic Recognition
 - **Authors:** Jingchen Sun, Shaobo Han, Wataru Kohno, Changyou Chen
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Machine Learning (cs.LG)
 - **Arxiv link:** https://arxiv.org/abs/2501.09877

 - **Pdf link:** https://arxiv.org/pdf/2501.09877

 - **Abstract**
 Contrastive Language-Audio Pretraining (CLAP) models have demonstrated unprecedented performance in various acoustic signal recognition tasks. Fiber-optic-based acoustic recognition is one of the most important downstream tasks and plays a significant role in environmental sensing. Adapting CLAP for fiber-optic acoustic recognition has become an active research area. As a non-conventional acoustic sensor, fiber-optic acoustic recognition presents a challenging, domain-specific, low-shot deployment environment with significant domain shifts due to unique frequency response and noise characteristics. To address these challenges, we propose a support-based adaptation method, CLAP-S, which linearly interpolates a CLAP Adapter with the Support Set, leveraging both implicit knowledge through fine-tuning and explicit knowledge retrieved from memory for cross-domain generalization. Experimental results show that our method delivers competitive performance on both laboratory-recorded fiber-optic ESC-50 datasets and a real-world fiber-optic gunshot-firework dataset. Our research also provides valuable insights for other downstream acoustic recognition tasks. The code and gunshot-firework dataset are available at this https URL.
#### Unsupervised Rhythm and Voice Conversion of Dysarthric to Healthy Speech for ASR
 - **Authors:** Karl El Hajal, Enno Hermann, Ajinkya Kulkarni, Mathew Magimai.-Doss
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Artificial Intelligence (cs.AI); Machine Learning (cs.LG); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2501.10256

 - **Pdf link:** https://arxiv.org/pdf/2501.10256

 - **Abstract**
 Automatic speech recognition (ASR) systems are well known to perform poorly on dysarthric speech. Previous works have addressed this by speaking rate modification to reduce the mismatch with typical speech. Unfortunately, these approaches rely on transcribed speech data to estimate speaking rates and phoneme durations, which might not be available for unseen speakers. Therefore, we combine unsupervised rhythm and voice conversion methods based on self-supervised speech representations to map dysarthric to typical speech. We evaluate the outputs with a large ASR model pre-trained on healthy speech without further fine-tuning and find that the proposed rhythm conversion especially improves performance for speakers of the Torgo corpus with more severe cases of dysarthria. Code and audio samples are available at this https URL .
#### On Ambisonic Source Separation with Spatially Informed Non-negative Tensor Factorization
 - **Authors:** Mateusz Guzik, Konrad Kowalczyk
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2501.10305

 - **Pdf link:** https://arxiv.org/pdf/2501.10305

 - **Abstract**
 This article presents a Non-negative Tensor Factorization based method for sound source separation from Ambisonic microphone signals. The proposed method enables the use of prior knowledge about the Directions-of-Arrival (DOAs) of the sources, incorporated through a constraint on the Spatial Covariance Matrix (SCM) within a Maximum a Posteriori (MAP) framework. Specifically, this article presents a detailed derivation of four algorithms that are based on two types of cost functions, namely the squared Euclidean distance and the Itakura-Saito divergence, which are then combined with two prior probability distributions on the SCM, that is the Wishart and the Inverse Wishart. The experimental evaluation of the baseline Maximum Likelihood (ML) and the proposed MAP methods is primarily based on first-order Ambisonic recordings, using four different source signal datasets, three with musical pieces and one containing speech utterances. We consider under-determined, determined, as well as over-determined scenarios by separating two, four and six sound sources, respectively. Furthermore, we evaluate the proposed algorithms for different spherical harmonic orders and at different reverberation time levels, as well as in non-ideal prior knowledge conditions, for increasingly more corrupted DOAs. Overall, in comparison with beamforming and a state-of-the-art separation technique, as well as the baseline ML methods, the proposed MAP approach offers superior separation performance in a variety of scenarios, as shown by the analysis of the experimental evaluation results, in terms of the standard objective separation measures, such as the SDR, ISR, SIR and SAR.
#### HiFi-SR: A Unified Generative Transformer-Convolutional Adversarial Network for High-Fidelity Speech Super-Resolution
 - **Authors:** Shengkui Zhao, Kun Zhou, Zexu Pan, Yukun Ma, Chong Zhang, Bin Ma
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2501.10045

 - **Pdf link:** https://arxiv.org/pdf/2501.10045

 - **Abstract**
 The application of generative adversarial networks (GANs) has recently advanced speech super-resolution (SR) based on intermediate representations like mel-spectrograms. However, existing SR methods that typically rely on independently trained and concatenated networks may lead to inconsistent representations and poor speech quality, especially in out-of-domain scenarios. In this work, we propose HiFi-SR, a unified network that leverages end-to-end adversarial training to achieve high-fidelity speech super-resolution. Our model features a unified transformer-convolutional generator designed to seamlessly handle both the prediction of latent representations and their conversion into time-domain waveforms. The transformer network serves as a powerful encoder, converting low-resolution mel-spectrograms into latent space representations, while the convolutional network upscales these representations into high-resolution waveforms. To enhance high-frequency fidelity, we incorporate a multi-band, multi-scale time-frequency discriminator, along with a multi-scale mel-reconstruction loss in the adversarial training process. HiFi-SR is versatile, capable of upscaling any input speech signal between 4 kHz and 32 kHz to a 48 kHz sampling rate. Experimental results demonstrate that HiFi-SR significantly outperforms existing speech SR methods across both objective metrics and ABX preference tests, for both in-domain and out-of-domain scenarios (this https URL).
#### Conditional Latent Diffusion-Based Speech Enhancement Via Dual Context Learning
 - **Authors:** Shengkui Zhao, Zexu Pan, Kun Zhou, Yukun Ma, Chong Zhang, Bin Ma
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2501.10052

 - **Pdf link:** https://arxiv.org/pdf/2501.10052

 - **Abstract**
 Recently, the application of diffusion probabilistic models has advanced speech enhancement through generative approaches. However, existing diffusion-based methods have focused on the generation process in high-dimensional waveform or spectral domains, leading to increased generation complexity and slower inference speeds. Additionally, these methods have primarily modelled clean speech distributions, with limited exploration of noise distributions, thereby constraining the discriminative capability of diffusion models for speech enhancement. To address these issues, we propose a novel approach that integrates a conditional latent diffusion model (cLDM) with dual-context learning (DCL). Our method utilizes a variational autoencoder (VAE) to compress mel-spectrograms into a low-dimensional latent space. We then apply cLDM to transform the latent representations of both clean speech and background noise into Gaussian noise by the DCL process, and a parameterized model is trained to reverse this process, conditioned on noisy latent representations and text embeddings. By operating in a lower-dimensional space, the latent representations reduce the complexity of the generation process, while the DCL process enhances the model's ability to handle diverse and unseen noise environments. Our experiments demonstrate the strong performance of the proposed approach compared to existing diffusion-based methods, even with fewer iterative steps, and highlight the superior generalization capability of our models to out-of-domain noise datasets (this https URL).
#### AI-Generated Music Detection and its Challenges
 - **Authors:** Darius Afchar, Gabriel Meseguer-Brocal, Romain Hennequin
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2501.10111

 - **Pdf link:** https://arxiv.org/pdf/2501.10111

 - **Abstract**
 In the face of a new era of generative models, the detection of artificially generated content has become a matter of utmost importance. In particular, the ability to create credible minute-long synthetic music in a few seconds on user-friendly platforms poses a real threat of fraud on streaming services and unfair competition to human artists. This paper demonstrates the possibility (and surprising ease) of training classifiers on datasets comprising real audio and artificial reconstructions, achieving a convincing accuracy of 99.8%. To our knowledge, this marks the first publication of a AI-music detector, a tool that will help in the regulation of synthetic media. Nevertheless, informed by decades of literature on forgery detection in other fields, we stress that getting a good test score is not the end of the story. We expose and discuss several facets that could be problematic with such a deployed detector: robustness to audio manipulation, generalisation to unseen models. This second part acts as a position for future research steps in the field and a caveat to a flourishing market of artificial content checkers.
#### Towards An Integrated Approach for Expressive Piano Performance Synthesis from Music Scores
 - **Authors:** Jingjing Tang, Erica Cooper, Xin Wang, Junichi Yamagishi, George Fazekas
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2501.10222

 - **Pdf link:** https://arxiv.org/pdf/2501.10222

 - **Abstract**
 This paper presents an integrated system that transforms symbolic music scores into expressive piano performance audio. By combining a Transformer-based Expressive Performance Rendering (EPR) model with a fine-tuned neural MIDI synthesiser, our approach directly generates expressive audio performances from score inputs. To the best of our knowledge, this is the first system to offer a streamlined method for converting score MIDI files lacking expression control into rich, expressive piano performances. We conducted experiments using subsets of the ATEPP dataset, evaluating the system with both objective metrics and subjective listening tests. Our system not only accurately reconstructs human-like expressiveness, but also captures the acoustic ambience of environments such as concert halls and recording studios. Additionally, the proposed system demonstrates its ability to achieve musical expressiveness while ensuring good audio quality in its outputs.


by Zyzzyva0381 (Windy). 


2025-01-20
