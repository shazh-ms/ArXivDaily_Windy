# Showing new listings for Monday, 30 March 2026
Auto update papers at about 2:30am UTC (10:30am Beijing time) every weekday.


阅读 `Usage.md`了解如何使用此repo实现个性化的Arxiv论文推送

See `Usage.md` for instructions on how to personalize the repo. 


Keyword list: ['text-to-speech', 'text to speech', 'tts', 'LLM-based', 'speech', 'voice']


Excluded: []


### Today: 5papers 
#### Sommelier: Scalable Open Multi-turn Audio Pre-processing for Full-duplex Speech Language Models
 - **Authors:** Kyudan Jung, Jihwan Kim, Soyoon Kim, Jeongoon Kim, Jaegul Choo, Cheonbok Park
 - **Subjects:** Subjects:
Sound (cs.SD); Artificial Intelligence (cs.AI); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2603.25750

 - **Pdf link:** https://arxiv.org/pdf/2603.25750

 - **Abstract**
 As the paradigm of AI shifts from text-based LLMs to Speech Language Models (SLMs), there is a growing demand for full-duplex systems capable of real-time, natural human-computer interaction. However, the development of such models is constrained by the scarcity of high-quality, multi-speaker conversational data, as existing large-scale resources are predominantly single-speaker or limited in volume. Addressing the complex dynamics of natural dialogue, such as overlapping and back-channeling remains a challenge, with standard processing pipelines suffering from diarization errors and ASR hallucinations. To bridge this gap, we present a robust and scalable open-source data processing pipeline designed for full-duplex model.
#### Unlocking Strong Supervision: A Data-Centric Study of General-Purpose Audio Pre-Training Methods
 - **Authors:** Xuanru Zhou, Yiwen Shao, Wei-Cheng Tseng, Dong Yu
 - **Subjects:** Subjects:
Sound (cs.SD); Artificial Intelligence (cs.AI); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2603.25767

 - **Pdf link:** https://arxiv.org/pdf/2603.25767

 - **Abstract**
 Current audio pre-training seeks to learn unified representations for broad audio understanding tasks, but it remains fragmented and is fundamentally bottlenecked by its reliance on weak, noisy, and scale-limited labels. Drawing lessons from vision's foundational pre-training blueprint, we argue that the audio field must first establish its own large-scale, strong supervision framework. We introduce a new data-centric pipeline that leverages a high-fidelity captioner to create SOTA-quality captions and the first Unified Tag System (UTS) that bridges speech, music, and environmental sounds. We then conduct a systematic comparative study of different pre-training objectives on these strong source data. Our experiments suggest that data quality and coverage are the primary drivers of performance, while the choice of objective dictates downstream task specialization.
#### Cinematic Audio Source Separation Using Visual Cues
 - **Authors:** Kang Zhang, Suyeon Lee, Arda Senocak, Joon Son Chung
 - **Subjects:** Subjects:
Multimedia (cs.MM); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2603.26113

 - **Pdf link:** https://arxiv.org/pdf/2603.26113

 - **Abstract**
 Cinematic Audio Source Separation (CASS) aims to decompose mixed film audio into speech, music, and sound effects, enabling applications like dubbing and remastering. Existing CASS approaches are audio-only, overlooking the inherent audio-visual nature of films, where sounds often align with visual cues. We present the first framework for audio-visual CASS (AV-CASS), leveraging visual context to enhance separation quality. Our method formulates CASS as a conditional generative modeling problem using conditional flow matching, enabling multimodal audio source separation. To address the lack of cinematic datasets with isolated sound tracks, we introduce a training data synthesis pipeline that pairs in-the-wild audio and video streams (e.g., facial videos for speech, scene videos for effects) and design a dedicated visual encoder for this dual-stream setup. Trained entirely on synthetic data, our model generalizes effectively to real-world cinematic content and achieves strong performance on synthetic, real-world, and audio-only CASS benchmarks. Code and demo are available at \url{this https URL}.
#### Distilling Conversations: Abstract Compression of Conversational Audio Context for LLM-based ASR
 - **Authors:** Shashi Kumar, Esaú Villatoro-Tello, Sergio Burdisso, Kadri Hacioglu, Thibault Bañeras-Roux, Hasindri Watawana, Dairazalia Sanchez-Cortes, Srikanth Madikeri, Petr Motlicek, Andreas Stolcke
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Artificial Intelligence (cs.AI); Machine Learning (cs.LG); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2603.26246

 - **Pdf link:** https://arxiv.org/pdf/2603.26246

 - **Abstract**
 Standard LLM-based speech recognition systems typically process utterances in isolation, limiting their ability to leverage conversational context. In this work, we study whether multimodal context from prior turns improves LLM-based ASR and how to represent that context efficiently. We find that, after supervised multi-turn training, conversational context mainly helps with the recognition of contextual entities. However, conditioning on raw context is expensive because the prior-turn audio token sequence grows rapidly with conversation length. To address this, we propose Abstract Compression, which replaces the audio portion of prior turns with a fixed number of learned latent tokens while retaining corresponding transcripts explicitly. On both in-domain and out-of-domain test sets, the compressed model recovers part of the gains of raw-context conditioning with a smaller prior-turn audio footprint. We also provide targeted analyses of the compression setup and its trade-offs.
#### A Power-Weighted Noncentral Complex Gaussian Distribution
 - **Authors:** Toru Nakashika
 - **Subjects:** Subjects:
Machine Learning (stat.ML); Machine Learning (cs.LG); Sound (cs.SD); Audio and Speech Processing (eess.AS); Signal Processing (eess.SP)
 - **Arxiv link:** https://arxiv.org/abs/2603.26344

 - **Pdf link:** https://arxiv.org/pdf/2603.26344

 - **Abstract**
 The complex Gaussian distribution has been widely used as a fundamental spectral and noise model in signal processing and communication. However, its Gaussian structure often limits its ability to represent the diverse amplitude characteristics observed in individual source signals. On the other hand, many existing non-Gaussian amplitude distributions derived from hyperspherical models achieve good empirical fit due to their power-law structures, while they do not explicitly account for the complex-plane geometry inherent in complex-valued observations. In this paper, we propose a new probabilistic model for complex-valued random variables, which can be interpreted as a power-weighted noncentral complex Gaussian distribution. Unlike conventional hyperspherical amplitude models, the proposed model is formulated directly on the complex plane and preserves the geometric structure of complex-valued observations while retaining a higher-dimensional interpretation. The model introduces a nonlinear phase diffusion through a single shape parameter, enabling continuous control of the distributional geometry from arc-shaped diffusion along the phase direction to concentration of probability mass toward the origin. We formulate the proposed distribution and analyze the statistical properties of the induced amplitude distribution. The derived amplitude and power distributions provide a unified framework encompassing several widely used distributions in signal modeling, including the Rice, Nakagami, and gamma distributions. Experimental results on speech power spectra demonstrate that the proposed model consistently outperforms conventional distributions in terms of log-likelihood.


by Zyzzyva0381 (Windy). 


2026-03-30
