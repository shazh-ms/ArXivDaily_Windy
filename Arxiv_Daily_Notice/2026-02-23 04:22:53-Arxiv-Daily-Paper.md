# Showing new listings for Monday, 23 February 2026
Auto update papers at about 2:30am UTC (10:30am Beijing time) every weekday.


阅读 `Usage.md`了解如何使用此repo实现个性化的Arxiv论文推送

See `Usage.md` for instructions on how to personalize the repo. 


Keyword list: ['text-to-speech', 'text to speech', 'tts', 'LLM-based', 'speech', 'voice']


Excluded: []


### Today: 3papers 
#### SIRUP: A diffusion-based virtual upmixer of steering vectors for highly-directive spatialization with first-order ambisonics
 - **Authors:** Emilio Picard (RIKEN AIP, UP1 EMS), Diego Di Carlo (RIKEN AIP, IP Paris), Aditya Arie Nugraha (RIKEN AIP), Mathieu Fontaine (LTCI, IP Paris), Kazuyoshi Yoshii (RIKEN AIP)
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD); Signal Processing (eess.SP)
 - **Arxiv link:** https://arxiv.org/abs/2602.17732

 - **Pdf link:** https://arxiv.org/pdf/2602.17732

 - **Abstract**
 This paper presents virtual upmixing of steering vectors captured by a fewer-channel spherical microphone array. This challenge has conventionally been addressed by recovering the directions and signals of sound sources from first-order ambisonics (FOA) data, and then rendering the higher-order ambisonics (HOA) data using a physics-based acoustic simulator. This approach, however, struggles to handle the mutual dependency between the spatial directivity of source estimation and the spatial resolution of FOA ambisonics data. Our method, named SIRUP, employs a latent diffusion model architecture. Specifically, a variational autoencoder (VAE) is used to learn a compact encoding of the HOA data in a latent space and a diffusion model is then trained to generate the HOA embeddings, conditioned by the FOA data. Experimental results showed that SIRUP achieved a significant improvement compared to FOA systems for steering vector upmixing, source localization, and speech denoising.
#### Rethinking Flow and Diffusion Bridge Models for Speech Enhancement
 - **Authors:** Dahan Wang, Jun Gao, Tong Lei, Yuxiang Hu, Changbao Zhu, Kai Chen, Jing Lu
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2602.18355

 - **Pdf link:** https://arxiv.org/pdf/2602.18355

 - **Abstract**
 Flow matching and diffusion bridge models have emerged as leading paradigms in generative speech enhancement, modeling stochastic processes between paired noisy and clean speech signals based on principles such as flow matching, score matching, and SchrÃ¶dinger bridge. In this paper, we present a framework that unifies existing flow and diffusion bridge models by interpreting them as constructions of Gaussian probability paths with varying means and variances between paired data. Furthermore, we investigate the underlying consistency between the training/inference procedures of these generative models and conventional predictive models. Our analysis reveals that each sampling step of a well-trained flow or diffusion bridge model optimized with a data prediction loss is theoretically analogous to executing predictive speech enhancement. Motivated by this insight, we introduce an enhanced bridge model that integrates an effective probability path design with key elements from predictive paradigms, including improved network architecture, tailored loss functions, and optimized training strategies. Experiments on denoising and dereverberation tasks demonstrate that the proposed method outperforms existing flow and diffusion baselines with fewer parameters and reduced computational complexity. The results also highlight that the inherently predictive nature of this generative framework imposes limitations on its achievable upper-bound performance.
#### MeanVoiceFlow: One-step Nonparallel Voice Conversion with Mean Flows
 - **Authors:** Takuhiro Kaneko, Hirokazu Kameoka, Kou Tanaka, Yuto Kondo
 - **Subjects:** Subjects:
Sound (cs.SD); Artificial Intelligence (cs.AI); Machine Learning (cs.LG); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2602.18104

 - **Pdf link:** https://arxiv.org/pdf/2602.18104

 - **Abstract**
 In voice conversion (VC) applications, diffusion and flow-matching models have exhibited exceptional speech quality and speaker similarity performances. However, they are limited by slow conversion owing to their iterative inference. Consequently, we propose MeanVoiceFlow, a novel one-step nonparallel VC model based on mean flows, which can be trained from scratch without requiring pretraining or distillation. Unlike conventional flow matching that uses instantaneous velocity, mean flows employ average velocity to more accurately compute the time integral along the inference path in a single step. However, training the average velocity requires its derivative to compute the target velocity, which can cause instability. Therefore, we introduce a structural margin reconstruction loss as a zero-input constraint, which moderately regularizes the input-output behavior of the model without harmful statistical averaging. Furthermore, we propose conditional diffused-input training in which a mixture of noise and source data is used as input to the model during both training and inference. This enables the model to effectively leverage source information while maintaining consistency between training and inference. Experimental results validate the effectiveness of these techniques and demonstrate that MeanVoiceFlow achieves performance comparable to that of previous multi-step and distillation-based models, even when trained from scratch. Audio samples are available at this https URL.


by Zyzzyva0381 (Windy). 


2026-02-23
