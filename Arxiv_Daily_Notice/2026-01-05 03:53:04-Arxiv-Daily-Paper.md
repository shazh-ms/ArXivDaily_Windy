# Showing new listings for Monday, 5 January 2026
Auto update papers at about 2:30am UTC (10:30am Beijing time) every weekday.


阅读 `Usage.md`了解如何使用此repo实现个性化的Arxiv论文推送

See `Usage.md` for instructions on how to personalize the repo. 


Keyword list: ['text-to-speech', 'text to speech', 'tts', 'LLM-based', 'speech', 'voice']


Excluded: []


### Today: 3papers 
#### Learning Speech Representations with Variational Predictive Coding
 - **Authors:** Sung-Lin Yeh, Peter Bell, Hao Tang
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Computation and Language (cs.CL)
 - **Arxiv link:** https://arxiv.org/abs/2601.00100

 - **Pdf link:** https://arxiv.org/pdf/2601.00100

 - **Abstract**
 Despite being the best known objective for learning speech representations, the HuBERT objective has not been further developed and improved. We argue that it is the lack of an underlying principle that stalls the development, and, in this paper, we show that predictive coding under a variational view is the principle behind the HuBERT objective. Due to its generality, our formulation provides opportunities to improve parameterization and optimization, and we show two simple modifications that bring immediate improvements to the HuBERT objective. In addition, the predictive coding formulation has tight connections to various other objectives, such as APC, CPC, wav2vec, and BEST-RQ. Empirically, the improvement in pre-training brings significant improvements to four downstream tasks: phone classification, f0 tracking, speaker recognition, and automatic speech recognition, highlighting the importance of the predictive coding interpretation.
#### IKFST: IOO and KOO Algorithms for Accelerated and Precise WFST-based End-to-End Automatic Speech Recognition
 - **Authors:** Zhuoran Zhuang, Ye Chen, Chao Luo, Tian-Hao Zhang, Xuewei Zhang, Jian Ma, Jiatong Shi, Wei Zhang
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2601.00160

 - **Pdf link:** https://arxiv.org/pdf/2601.00160

 - **Abstract**
 End-to-end automatic speech recognition has become the dominant paradigm in both academia and industry. To enhance recognition performance, the Weighted Finite-State Transducer (WFST) is widely adopted to integrate acoustic and language models through static graph composition, providing robust decoding and effective error correction. However, WFST decoding relies on a frame-by-frame autoregressive search over CTC posterior probabilities, which severely limits inference efficiency. Motivated by establishing a more principled compatibility between WFST decoding and CTC modeling, we systematically study the two fundamental components of CTC outputs, namely blank and non-blank frames, and identify a key insight: blank frames primarily encode positional information, while non-blank frames carry semantic content. Building on this observation, we introduce Keep-Only-One and Insert-Only-One, two decoding algorithms that explicitly exploit the structural roles of blank and non-blank frames to achieve significantly faster WFST-based inference without compromising recognition accuracy. Experiments on large-scale in-house, AISHELL-1, and LibriSpeech datasets demonstrate state-of-the-art recognition accuracy with substantially reduced decoding latency, enabling truly efficient and high-performance WFST decoding in modern speech recognition systems.
#### Latent Flow Matching for Expressive Singing Voice Synthesis
 - **Authors:** Minhyeok Yun, Yong-Hoon Choi
 - **Subjects:** Subjects:
Sound (cs.SD); Artificial Intelligence (cs.AI); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2601.00217

 - **Pdf link:** https://arxiv.org/pdf/2601.00217

 - **Abstract**
 Conditional variational autoencoder (cVAE)-based singing voice synthesis provides efficient inference and strong audio quality by learning a score-conditioned prior and a recording-conditioned posterior latent space. However, because synthesis relies on prior samples while training uses posterior latents inferred from real recordings, imperfect distribution matching can cause a prior-posterior mismatch that degrades fine-grained expressiveness such as vibrato and micro-prosody. We propose FM-Singer, which introduces conditional flow matching (CFM) in latent space to learn a continuous vector field transporting prior latents toward posterior latents along an optimal-transport-inspired path. At inference time, the learned latent flow refines a prior sample by solving an ordinary differential equation (ODE) before waveform generation, improving expressiveness while preserving the efficiency of parallel decoding. Experiments on Korean and Chinese singing datasets demonstrate consistent improvements over strong baselines, including lower mel-cepstral distortion and fundamental-frequency error and higher perceptual scores on the Korean dataset. Code, pretrained checkpoints, and audio demos are available at this https URL


by Zyzzyva0381 (Windy). 


2026-01-05
