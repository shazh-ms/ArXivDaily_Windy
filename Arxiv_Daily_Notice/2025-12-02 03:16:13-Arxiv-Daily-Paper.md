# Showing new listings for Monday, 1 December 2025
Auto update papers at about 2:30am UTC (10:30am Beijing time) every weekday.


阅读 `Usage.md`了解如何使用此repo实现个性化的Arxiv论文推送

See `Usage.md` for instructions on how to personalize the repo. 


Keyword list: ['text-to-speech', 'text to speech', 'tts', 'LLM-based', 'speech', 'voice']


Excluded: []


### Today: 4papers 
#### Group-Aware Partial Model Merging for Children's Automatic Speech Recognition
 - **Authors:** Thomas Rolland, Alberto Abad
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2511.23098

 - **Pdf link:** https://arxiv.org/pdf/2511.23098

 - **Abstract**
 Automatic Speech Recognition (ASR) for children remains challenging, primarily due to large acoustic variability and limited availability of training data. While supervised fine-tuning of adult pre-trained models has shown promise, it often fails to capture group-specific characteristics variations among children. To address this, we introduce GRoup-Aware PARtial model Merging (GRAPAM), a parameter-efficient approach that combines unsupervised clustering, partial fine-tuning, and model merging. Our approach adapts adult-pre-trained models to children by first grouping the children's data based on acoustic similarity. Each group is used to partially fine-tune an adult pre-trained model, and the resulting models are merged at the parameter level. Experiments conducted on the MyST children's speech corpus indicate that GRAPAM achieves a relative improvement of 6% of Word Error Rate (WER), using the same amount of data, outperforming full fine-tuning while training fewer parameters. These results highlight the promise of model merging as a scalable and effective strategy for children's ASR.
#### GLA-Grad++: An Improved Griffin-Lim Guided Diffusion Model for Speech Synthesis
 - **Authors:** Teysir Baoueb, Xiaoyu Bie, Mathieu Fontaine, Gaël Richard
 - **Subjects:** Subjects:
Sound (cs.SD); Machine Learning (cs.LG); Audio and Speech Processing (eess.AS); Signal Processing (eess.SP)
 - **Arxiv link:** https://arxiv.org/abs/2511.22293

 - **Pdf link:** https://arxiv.org/pdf/2511.22293

 - **Abstract**
 Recent advances in diffusion models have positioned them as powerful generative frameworks for speech synthesis, demonstrating substantial improvements in audio quality and stability. Nevertheless, their effectiveness in vocoders conditioned on mel spectrograms remains constrained, particularly when the conditioning diverges from the training distribution. The recently proposed GLA-Grad model introduced a phase-aware extension to the WaveGrad vocoder that integrated the Griffin-Lim algorithm (GLA) into the reverse process to reduce inconsistencies between generated signals and conditioning mel spectrogram. In this paper, we further improve GLA-Grad through an innovative choice in how to apply the correction. Particularly, we compute the correction term only once, with a single application of GLA, to accelerate the generation process. Experimental results demonstrate that our method consistently outperforms the baseline models, particularly in out-of-domain scenarios.
#### Joint Speech and Text Training for LLM-Based End-to-End Spoken Dialogue State Tracking
 - **Authors:** Katia Vendrame, Bolaji Yusuf, Santosh Kesiraju, Šimon Sedláček, Oldřich Plchot, Jan Černocký
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2511.22503

 - **Pdf link:** https://arxiv.org/pdf/2511.22503

 - **Abstract**
 End-to-end spoken dialogue state tracking (DST) is made difficult by the tandem of having to handle speech input and data scarcity. Combining speech foundation encoders and large language models has been proposed in recent work as to alleviate some of this difficulty. Although this approach has been shown to result in strong spoken DST models, achieving state-of-the-art performance in realistic multi-turn DST, it struggles to generalize across domains and requires annotated spoken DST training data for each domain of interest. However, collecting such data for every target domain is both costly and difficult. Noting that textual DST data is more easily obtained for various domains, in this work, we propose jointly training on available spoken DST data and written textual data from other domains as a way to achieve cross-domain generalization. We conduct experiments which show the efficacy of our proposed method for getting good cross-domain DST performance without relying on spoken training data from the target domains.
#### PURE Codec: Progressive Unfolding of Residual Entropy for Speech Codec Learning
 - **Authors:** Jiatong Shi, Haoran Wang, William Chen, Chenda Li, Wangyou Zhang, Jinchuan Tian, Shinji Watanabe
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2511.22687

 - **Pdf link:** https://arxiv.org/pdf/2511.22687

 - **Abstract**
 Neural speech codecs have achieved strong performance in low-bitrate compression, but residual vector quantization (RVQ) often suffers from unstable training and ineffective decomposition, limiting reconstruction quality and efficiency. We propose PURE Codec (Progressive Unfolding of Residual Entropy), a novel framework that guides multi-stage quantization using a pre-trained speech enhancement model. The first quantization stage reconstructs low-entropy, denoised speech embeddings, while subsequent stages encode residual high-entropy components. This design improves training stability significantly. Experiments demonstrate that PURE consistently outperforms conventional RVQ-based codecs in reconstruction and downstream speech language model-based text-to-speech, particularly under noisy training conditions.


by Zyzzyva0381 (Windy). 


2025-12-02
