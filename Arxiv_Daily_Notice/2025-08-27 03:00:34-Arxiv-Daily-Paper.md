# Showing new listings for Wednesday, 27 August 2025
Auto update papers at about 2:30am UTC (10:30am Beijing time) every weekday.


阅读 `Usage.md`了解如何使用此repo实现个性化的Arxiv论文推送

See `Usage.md` for instructions on how to personalize the repo. 


Keyword list: ['text-to-speech', 'text to speech', 'tts', 'LLM-based', 'speech', 'voice']


Excluded: []


### Today: 8papers 
#### Toward Responsible ASR for African American English Speakers: A Scoping Review of Bias and Equity in Speech Technology
 - **Authors:** Jay L. Cunningham, Adinawa Adjagbodjou, Jeffrey Basoah, Jainaba Jawara, Kowe Kadoma, Aaleyah Lewis
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Artificial Intelligence (cs.AI); Computation and Language (cs.CL); Human-Computer Interaction (cs.HC); Machine Learning (cs.LG)
 - **Arxiv link:** https://arxiv.org/abs/2508.18288

 - **Pdf link:** https://arxiv.org/pdf/2508.18288

 - **Abstract**
 This scoping literature review examines how fairness, bias, and equity are conceptualized and operationalized in Automatic Speech Recognition (ASR) and adjacent speech and language technologies (SLT) for African American English (AAE) speakers and other linguistically diverse communities. Drawing from 44 peer-reviewed publications across Human-Computer Interaction (HCI), Machine Learning/Natural Language Processing (ML/NLP), and Sociolinguistics, we identify four major areas of inquiry: (1) how researchers understand ASR-related harms; (2) inclusive data practices spanning collection, curation, annotation, and model training; (3) methodological and theoretical approaches to linguistic inclusion; and (4) emerging practices and design recommendations for more equitable systems. While technical fairness interventions are growing, our review highlights a critical gap in governance-centered approaches that foreground community agency, linguistic justice, and participatory accountability. We propose a governance-centered ASR lifecycle as an emergent interdisciplinary framework for responsible ASR development and offer implications for researchers, practitioners, and policymakers seeking to address language marginalization in speech AI systems.
#### On the Application of Diffusion Models for Simultaneous Denoising and Dereverberation
 - **Authors:** Adrian Meise, Tobias Cord-Landwehr, Reinhold Haeb-Umbach
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2508.18833

 - **Pdf link:** https://arxiv.org/pdf/2508.18833

 - **Abstract**
 Diffusion models have been shown to achieve natural-sounding enhancement of speech degraded by noise or reverberation. However, their simultaneous denoising and dereverberation capability has so far not been studied much, although this is arguably the most common scenario in a practical application. In this work, we investigate different approaches to enhance noisy and/or reverberant speech. We examine the cascaded application of models, each trained on only one of the distortions, and compare it with a single model, trained either solely on data that is both noisy and reverberated, or trained on data comprising subsets of purely noisy, of purely reverberated, and of noisy reverberant speech. Tests are performed both on artificially generated and real recordings of noisy and/or reverberant data. The results show that, when using the cascade of models, satisfactory results are only achieved if they are applied in the order of the dominating distortion. If only a single model is desired that can operate on all distortion scenarios, the best compromise appears to be a model trained on the aforementioned three subsets of degraded speech data.
#### A Framework for Robust Speaker Verification in Highly Noisy Environments Leveraging Both Noisy and Enhanced Audio
 - **Authors:** Adam Katav, Yair Moshe, Israel Cohen
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2508.18913

 - **Pdf link:** https://arxiv.org/pdf/2508.18913

 - **Abstract**
 Recent advancements in speaker verification techniques show promise, but their performance often deteriorates significantly in challenging acoustic environments. Although speech enhancement methods can improve perceived audio quality, they may unintentionally distort speaker-specific information, which can affect verification accuracy. This problem has become more noticeable with the increasing use of generative deep neural networks (DNNs) for speech enhancement. While these networks can produce intelligible speech even in conditions of very low signal-to-noise ratio (SNR), they may also severely alter distinctive speaker characteristics. To tackle this issue, we propose a novel neural network framework that effectively combines speaker embeddings extracted from both noisy and enhanced speech using a Siamese architecture. This architecture allows us to leverage complementary information from both sources, enhancing the robustness of speaker verification under severe noise conditions. Our framework is lightweight and agnostic to specific speaker verification and speech enhancement techniques, enabling the use of a wide range of state-of-the-art solutions without modification. Experimental results demonstrate the superior performance of our proposed framework.
#### MOSA: Mixtures of Simple Adapters Outperform Monolithic Approaches in LLM-based Multilingual ASR
 - **Authors:** Junjie Li, Jing Peng, Yangui Fang, Shuai Wang, Kai Yu
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2508.18998

 - **Pdf link:** https://arxiv.org/pdf/2508.18998

 - **Abstract**
 End-to-end multilingual ASR aims to transcribe speech from different languages into corresponding text, but is often limited by scarce multilingual data. LLM-based ASR aligns speech encoder outputs with LLM input space via a projector and has achieved notable success. However, prior work mainly improves performance by increasing data, with little focus on cross-lingual knowledge sharing. Moreover, a single complex projector struggles to capture both shared and language-specific features effectively. In this work, we propose MOSA (Mixture of Simple Adapters), leveraging a Mixture-of-Experts mechanism to combine lightweight adapters that learn shared and language-specific knowledge. This enables better utilization of high-resource language data to support low-resource languages, mitigating data scarcity issues. Experimental results show that MOSA-Base achieves a 15.4\% relative reduction in average WER compared to the Baseline-Base and consistently outperforms it across all languages. Remarkably, MOSA-Base surpasses the Baseline-Base even when trained with only 60\% of its parameters. Similarly, MOSA-Large outperforms the Baseline-Large in average WER and demonstrates greater robustness to data imbalance. Ablation studies further indicate that MOSA is more effective at handling individual languages and learning both language-specific and shared linguistic knowledge. These findings support that, in LLM-based ASR, a mixture of simple adapters is more effective than a single, complex adapter design.
#### CLEAR: Continuous Latent Autoregressive Modeling for High-quality and Low-latency Speech Synthesis
 - **Authors:** Chun Yat Wu, Jiajun Deng, Guinan Li, Qiuqiang Kong, Simon Lui
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2508.19098

 - **Pdf link:** https://arxiv.org/pdf/2508.19098

 - **Abstract**
 Autoregressive (AR) language models have emerged as powerful solutions for zero-shot text-to-speech (TTS) synthesis, capable of generating natural speech from a few seconds of audio prompts. However, conventional AR-based TTS systems relying on discrete audio tokens face the challenge of lossy compression during tokenization, requiring longer discrete token sequences to capture the same information as continuous ones, which adds inference latency and complicates AR modeling. To address this challenge, this paper proposes the Continuous Latent Autoregressive model (CLEAR), a unified zero-shot TTS framework that directly models continuous audio representations. More specifically, CLEAR introduces an enhanced variational autoencoder with shortcut connections, which achieves a high compression ratio to map waveforms into compact continuous latents. A lightweight MLP-based rectified flow head that operates independently for each hidden state is presented to model the continuous latent probability distribution, and trained jointly with the AR model within a single-stage framework. Experiments show that the proposed zero-shot CLEAR TTS can synthesize high-quality speech with low latency. Compared to state-of-the-art (SOTA) TTS models, CLEAR delivers competitive performance in robustness, speaker similarity and naturalness, while offering a lower real-time factor (RTF). In particular, CLEAR achieves SOTA results on the LibriSpeech test-clean dataset, with a word error rate of 1.88\% and an RTF of 0.29. Moreover, CLEAR facilitates streaming speech synthesis with a first-frame delay of 96ms, while maintaining high-quality speech synthesis.
#### MDD: a Mask Diffusion Detector to Protect Speaker Verification Systems from Adversarial Perturbations
 - **Authors:** Yibo Bai, Sizhou Chen, Michele Panariello, Xiao-Lei Zhang, Massimiliano Todisco, Nicholas Evans
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2508.19180

 - **Pdf link:** https://arxiv.org/pdf/2508.19180

 - **Abstract**
 Speaker verification systems are increasingly deployed in security-sensitive applications but remain highly vulnerable to adversarial perturbations. In this work, we propose the Mask Diffusion Detector (MDD), a novel adversarial detection and purification framework based on a \textit{text-conditioned masked diffusion model}. During training, MDD applies partial masking to Mel-spectrograms and progressively adds noise through a forward diffusion process, simulating the degradation of clean speech features. A reverse process then reconstructs the clean representation conditioned on the input transcription. Unlike prior approaches, MDD does not require adversarial examples or large-scale pretraining. Experimental results show that MDD achieves strong adversarial detection performance and outperforms prior state-of-the-art methods, including both diffusion-based and neural codec-based approaches. Furthermore, MDD effectively purifies adversarially-manipulated speech, restoring speaker verification performance to levels close to those observed under clean conditions. These findings demonstrate the potential of diffusion-based masking strategies for secure and reliable speaker verification systems.
#### Interpolating Speaker Identities in Embedding Space for Data Expansion
 - **Authors:** Tianchi Liu, Ruijie Tao, Qiongqiong Wang, Yidi Jiang, Hardik B. Sailor, Ke Zhang, Jingru Lin, Haizhou Li
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Artificial Intelligence (cs.AI)
 - **Arxiv link:** https://arxiv.org/abs/2508.19210

 - **Pdf link:** https://arxiv.org/pdf/2508.19210

 - **Abstract**
 The success of deep learning-based speaker verification systems is largely attributed to access to large-scale and diverse speaker identity data. However, collecting data from more identities is expensive, challenging, and often limited by privacy concerns. To address this limitation, we propose INSIDE (Interpolating Speaker Identities in Embedding Space), a novel data expansion method that synthesizes new speaker identities by interpolating between existing speaker embeddings. Specifically, we select pairs of nearby speaker embeddings from a pretrained speaker embedding space and compute intermediate embeddings using spherical linear interpolation. These interpolated embeddings are then fed to a text-to-speech system to generate corresponding speech waveforms. The resulting data is combined with the original dataset to train downstream models. Experiments show that models trained with INSIDE-expanded data outperform those trained only on real data, achieving 3.06\% to 5.24\% relative improvements. While INSIDE is primarily designed for speaker verification, we also validate its effectiveness on gender classification, where it yields a 13.44\% relative improvement. Moreover, INSIDE is compatible with other augmentation techniques and can serve as a flexible, scalable addition to existing training pipelines.
#### Improving Noise Robust Audio-Visual Speech Recognition via Router-Gated Cross-Modal Feature Fusion
 - **Authors:** DongHoon Lim, YoungChae Kim, Dong-Hyun Kim, Da-Hee Yang, Joon-Hyuk Chang
 - **Subjects:** Subjects:
Computer Vision and Pattern Recognition (cs.CV); Artificial Intelligence (cs.AI); Multimedia (cs.MM); Audio and Speech Processing (eess.AS); Signal Processing (eess.SP)
 - **Arxiv link:** https://arxiv.org/abs/2508.18734

 - **Pdf link:** https://arxiv.org/pdf/2508.18734

 - **Abstract**
 Robust audio-visual speech recognition (AVSR) in noisy environments remains challenging, as existing systems struggle to estimate audio reliability and dynamically adjust modality reliance. We propose router-gated cross-modal feature fusion, a novel AVSR framework that adaptively reweights audio and visual features based on token-level acoustic corruption scores. Using an audio-visual feature fusion-based router, our method down-weights unreliable audio tokens and reinforces visual cues through gated cross-attention in each decoder layer. This enables the model to pivot toward the visual modality when audio quality deteriorates. Experiments on LRS3 demonstrate that our approach achieves an 16.51-42.67% relative reduction in word error rate compared to AV-HuBERT. Ablation studies confirm that both the router and gating mechanism contribute to improved robustness under real-world acoustic noise.


by Zyzzyva0381 (Windy). 


2025-08-27
