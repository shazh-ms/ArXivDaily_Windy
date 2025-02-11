# Showing new listings for Monday, 10 February 2025
Auto update papers at about 2:30am UTC (10:30am Beijing time) every weekday.


阅读 `Usage.md`了解如何使用此repo实现个性化的Arxiv论文推送

See `Usage.md` for instructions on how to personalize the repo. 


Keyword list: ['acoustic', 'reinforcement learning', 'reverb', 'meta', 'separate', 'reconstruction', 'noise', 'enhance', 'localization', 'speech']


Excluded: []


### Today: 7papers 
#### GenVC: Self-Supervised Zero-Shot Voice Conversion
 - **Authors:** Zexin Cai, Henry Li Xinyuan, Ashi Garg, Leibny Paola García-Perera, Kevin Duh, Sanjeev Khudanpur, Matthew Wiesner, Nicholas Andrews
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Machine Learning (cs.LG)
 - **Arxiv link:** https://arxiv.org/abs/2502.04519

 - **Pdf link:** https://arxiv.org/pdf/2502.04519

 - **Abstract**
 Zero-shot voice conversion has recently made substantial progress, but many models still depend on external supervised systems to disentangle speaker identity and linguistic content. Furthermore, current methods often use parallel conversion, where the converted speech inherits the source utterance's temporal structure, restricting speaker similarity and privacy. To overcome these limitations, we introduce GenVC, a generative zero-shot voice conversion model. GenVC learns to disentangle linguistic content and speaker style in a self-supervised manner, eliminating the need for external models and enabling efficient training on large, unlabeled datasets. Experimental results show that GenVC achieves state-of-the-art speaker similarity while maintaining naturalness competitive with leading approaches. Its autoregressive generation also allows the converted speech to deviate from the source utterance's temporal structure. This feature makes GenVC highly effective for voice anonymization, as it minimizes the preservation of source prosody and speaker characteristics, enhancing privacy protection.
#### FocalCodec: Low-Bitrate Speech Coding via Focal Modulation Networks
 - **Authors:** Luca Della Libera, Francesco Paissan, Cem Subakan, Mirco Ravanelli
 - **Subjects:** Subjects:
Machine Learning (cs.LG); Artificial Intelligence (cs.AI); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2502.04465

 - **Pdf link:** https://arxiv.org/pdf/2502.04465

 - **Abstract**
 Large language models have revolutionized natural language processing through self-supervised pretraining on massive datasets. Inspired by this success, researchers have explored adapting these methods to speech by discretizing continuous audio into tokens using neural audio codecs. However, existing approaches face limitations, including high bitrates, the loss of either semantic or acoustic information, and the reliance on multi-codebook designs when trying to capture both, which increases architectural complexity for downstream tasks. To address these challenges, we introduce FocalCodec, an efficient low-bitrate codec based on focal modulation that utilizes a single binary codebook to compress speech between 0.16 and 0.65 kbps. FocalCodec delivers competitive performance in speech resynthesis and voice conversion at lower bitrates than the current state-of-the-art, while effectively handling multilingual speech and noisy environments. Evaluation on downstream tasks shows that FocalCodec successfully preserves sufficient semantic and acoustic information, while also being well-suited for generative modeling. Demo samples, code and checkpoints are available at this https URL.
#### ADIFF: Explaining audio difference using natural language
 - **Authors:** Soham Deshmukh, Shuo Han, Rita Singh, Bhiksha Raj
 - **Subjects:** Subjects:
Sound (cs.SD); Artificial Intelligence (cs.AI); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2502.04476

 - **Pdf link:** https://arxiv.org/pdf/2502.04476

 - **Abstract**
 Understanding and explaining differences between audio recordings is crucial for fields like audio forensics, quality assessment, and audio generation. This involves identifying and describing audio events, acoustic scenes, signal characteristics, and their emotional impact on listeners. This paper stands out as the first work to comprehensively study the task of explaining audio differences and then propose benchmark, baselines for the task. First, we present two new datasets for audio difference explanation derived from the AudioCaps and Clotho audio captioning datasets. Using Large Language Models (LLMs), we generate three levels of difference explanations: (1) concise descriptions of audio events and objects, (2) brief sentences about audio events, acoustic scenes, and signal properties, and (3) comprehensive explanations that include semantics and listener emotions. For the baseline, we use prefix tuning where audio embeddings from two audio files are used to prompt a frozen language model. Our empirical analysis and ablation studies reveal that the naive baseline struggles to distinguish perceptually similar sounds and generate detailed tier 3 explanations. To address these limitations, we propose ADIFF, which introduces a cross-projection module, position captioning, and a three-step training process to enhance the model's ability to produce detailed explanations. We evaluate our model using objective metrics and human evaluation and show our model enhancements lead to significant improvements in performance over naive baseline and SoTA Audio-Language Model (ALM) Qwen Audio. Lastly, we conduct multiple ablation studies to study the effects of cross-projection, language model parameters, position captioning, third stage fine-tuning, and present our findings. Our benchmarks, findings, and strong baseline pave the way for nuanced and human-like explanations of audio differences.
#### Dynamic Frequency-Adaptive Knowledge Distillation for Speech Enhancement
 - **Authors:** Xihao Yuan, Siqi Liu, Hanting Chen, Lu Zhou, Jian Li, Jie Hu
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2502.04711

 - **Pdf link:** https://arxiv.org/pdf/2502.04711

 - **Abstract**
 Deep learning-based speech enhancement (SE) models have recently outperformed traditional techniques, yet their deployment on resource-constrained devices remains challenging due to high computational and memory demands. This paper introduces a novel dynamic frequency-adaptive knowledge distillation (DFKD) approach to effectively compress SE models. Our method dynamically assesses the model's output, distinguishing between high and low-frequency components, and adapts the learning objectives to meet the unique requirements of different frequency bands, capitalizing on the SE task's inherent characteristics. To evaluate the DFKD's efficacy, we conducted experiments on three state-of-the-art models: DCCRN, ConTasNet, and DPTNet. The results demonstrate that our method not only significantly enhances the performance of the compressed model (student model) but also surpasses other logit-based knowledge distillation methods specifically for SE tasks.
#### Evaluating Standard and Dialectal Frisian ASR: Multilingual Fine-tuning and Language Identification for Improved Low-resource Performance
 - **Authors:** Reihaneh Amooie, Wietse de Vries, Yun Hao, Jelske Dijkstra, Matt Coler, Martijn Wieling
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Machine Learning (cs.LG); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2502.04883

 - **Pdf link:** https://arxiv.org/pdf/2502.04883

 - **Abstract**
 Automatic Speech Recognition (ASR) performance for low-resource languages is still far behind that of higher-resource languages such as English, due to a lack of sufficient labeled data. State-of-the-art methods deploy self-supervised transfer learning where a model pre-trained on large amounts of data is fine-tuned using little labeled data in a target low-resource language. In this paper, we present and examine a method for fine-tuning an SSL-based model in order to improve the performance for Frisian and its regional dialects (Clay Frisian, Wood Frisian, and South Frisian). We show that Frisian ASR performance can be improved by using multilingual (Frisian, Dutch, English and German) fine-tuning data and an auxiliary language identification task. In addition, our findings show that performance on dialectal speech suffers substantially, and, importantly, that this effect is moderated by the elicitation approach used to collect the dialectal data. Our findings also particularly suggest that relying solely on standard language data for ASR evaluation may underestimate real-world performance, particularly in languages with substantial dialectal variation.
#### Latent Swap Joint Diffusion for Long-Form Audio Generation
 - **Authors:** Yusheng Dai, Chenxi Wang, Chang Li, Chen Wang, Jun Du, Kewei Li, Ruoyu Wang, Jiefeng Ma, Lei Sun, Jianqing Gao
 - **Subjects:** Subjects:
Sound (cs.SD); Artificial Intelligence (cs.AI); Computer Vision and Pattern Recognition (cs.CV); Multimedia (cs.MM); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2502.05130

 - **Pdf link:** https://arxiv.org/pdf/2502.05130

 - **Abstract**
 Previous work on long-form audio generation using global-view diffusion or iterative generation demands significant training or inference costs. While recent advancements in multi-view joint diffusion for panoramic generation provide an efficient option, they struggle with spectrum generation with severe overlap distortions and high cross-view consistency costs. We initially explore this phenomenon through the connectivity inheritance of latent maps and uncover that averaging operations excessively smooth the high-frequency components of the latent map. To address these issues, we propose Swap Forward (SaFa), a frame-level latent swap framework that synchronizes multiple diffusions to produce a globally coherent long audio with more spectrum details in a forward-only manner. At its core, the bidirectional Self-Loop Latent Swap is applied between adjacent views, leveraging stepwise diffusion trajectory to adaptively enhance high-frequency components without disrupting low-frequency components. Furthermore, to ensure cross-view consistency, the unidirectional Reference-Guided Latent Swap is applied between the reference and the non-overlap regions of each subview during the early stages, providing centralized trajectory guidance. Quantitative and qualitative experiments demonstrate that SaFa significantly outperforms existing joint diffusion methods and even training-based long audio generation models. Moreover, we find that it also adapts well to panoramic generation, achieving comparable state-of-the-art performance with greater efficiency and model generalizability. Project page is available at this https URL.
#### Meta Audiobox Aesthetics: Unified Automatic Quality Assessment for Speech, Music, and Sound
 - **Authors:** Andros Tjandra, Yi-Chiao Wu, Baishan Guo, John Hoffman, Brian Ellis, Apoorv Vyas, Bowen Shi, Sanyuan Chen, Matt Le, Nick Zacharov, Carleigh Wood, Ann Lee, Wei-Ning Hsu
 - **Subjects:** Subjects:
Sound (cs.SD); Machine Learning (cs.LG); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2502.05139

 - **Pdf link:** https://arxiv.org/pdf/2502.05139

 - **Abstract**
 The quantification of audio aesthetics remains a complex challenge in audio processing, primarily due to its subjective nature, which is influenced by human perception and cultural context. Traditional methods often depend on human listeners for evaluation, leading to inconsistencies and high resource demands. This paper addresses the growing need for automated systems capable of predicting audio aesthetics without human intervention. Such systems are crucial for applications like data filtering, pseudo-labeling large datasets, and evaluating generative audio models, especially as these models become more sophisticated. In this work, we introduce a novel approach to audio aesthetic evaluation by proposing new annotation guidelines that decompose human listening perspectives into four distinct axes. We develop and train no-reference, per-item prediction models that offer a more nuanced assessment of audio quality. Our models are evaluated against human mean opinion scores (MOS) and existing methods, demonstrating comparable or superior performance. This research not only advances the field of audio aesthetics but also provides open-source models and datasets to facilitate future work and benchmarking. We release our code and pre-trained model at: this https URL


by Zyzzyva0381 (Windy). 


2025-02-11
