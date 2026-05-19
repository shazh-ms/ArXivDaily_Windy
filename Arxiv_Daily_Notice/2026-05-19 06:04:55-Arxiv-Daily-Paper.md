# Showing new listings for Tuesday, 19 May 2026
Auto update papers at about 2:30am UTC (10:30am Beijing time) every weekday.


阅读 `Usage.md`了解如何使用此repo实现个性化的Arxiv论文推送

See `Usage.md` for instructions on how to personalize the repo. 


Keyword list: ['text-to-speech', 'text to speech', 'tts', 'LLM-based', 'speech', 'voice']


Excluded: []


### Today: 7papers 
#### SemaVoice: Semantic-Aware Continuous Autoregressive Speech Synthesis
 - **Authors:** Huimeng Wang, Hui Lu, Jiajun Deng, Haoning Xu, Youjun Chen, Xueyuan Chen, Zhaoqing Li, Shuhai Peng, Shiyin Kang, Xunying Liu
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2605.16964

 - **Pdf link:** https://arxiv.org/pdf/2605.16964

 - **Abstract**
 Continuous autoregressive speech synthesis has recently emerged as a promising direction for zero-shot text-to-speech (TTS). However, existing methods still suffer from a fundamental mismatch between semantic-prosodic modeling and reconstruction-driven continuous speech representations. This mismatch causes TTS models to focus excessively on low-level acoustic textures at the expense of high-level semantic coherence, further exacerbating error accumulation in autoregressive generation. To address this challenge, we propose SemaVoice, a semantic-aware continuous autoregressive framework for high-fidelity zero-shot TTS. SemaVoice introduces a Speech Foundation Model (SFM) guided alignment mechanism that refines continuous speech representations to better capture both local semantic consistency and global structural relationships. These representations condition a patch-wise diffusion head within the autoregressive framework for high-quality speech synthesis. Experimental results on the Seed-TTS benchmark show that SemaVoice achieves an English WER of 1.71\% and remains highly competitive with state-of-the-art open-source systems in both objective and subjective evaluations. The effectiveness of SFM guided alignment is further confirmed by significant improvements under varying representation granularities with a fixed information-rate constraint.
#### Robust Soft-Constrained Spatially Selective Active Noise Control for Hearables Under Secondary Path Variations
 - **Authors:** Tong Xiao, Reinhild Roden, Matthias Blau, Simon Doclo
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Signal Processing (eess.SP); Systems and Control (eess.SY)
 - **Arxiv link:** https://arxiv.org/abs/2605.17407

 - **Pdf link:** https://arxiv.org/pdf/2605.17407

 - **Abstract**
 Spatially selective active noise control (SSANC) hearables aim to attenuate noise from certain directions at the eardrum while preserving desired speech arriving from selected directions. Existing SSANC systems typically assume an accurate estimate of the secondary path from the loudspeaker to the inner error microphone. In practice, however, this path varies across users and device fits, which can degrade performance and compromise system stability. This paper proposes a robust soft-constrained optimization framework that computes a single control filter by minimizing the average cost over a set of secondary path estimates derived from human measurements. Simulations and experiments on a real-time control platform show that the proposed approach slightly reduces mean performance relative to the matched case but substantially narrows the performance spread under secondary path mismatch. The proposed framework therefore provides a practical design strategy when accurate secondary path estimates are unavailable.
#### UrduSpeech: A 156-Hour Urdu Speech Corpus with 12-Dimension Paralinguistic Annotations
 - **Authors:** Attia Nafees ul Haq, Zeyu Zhu, Jingbin Hu, ChunJiang He, Lei Xie
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2605.17846

 - **Pdf link:** https://arxiv.org/pdf/2605.17846

 - **Abstract**
 Despite 230 million speakers, Urdu remains critically under-resourced in speech technology. We introduce UrduSpeech: a large high-fidelity Urdu corpus comprising 156 hours of audio with 12-dimension paralinguistic metadata, encompassing US-Std, US-CS, US-EngPk. To address Right-to-Left script constraints and frequent code-switching, we developed UrduSpeech, a LLM-driven pipeline to curate data across 12 diverse categories, including news, drama, and rare literary forms like Bait-Bazi. We also release a 9-hour US-Benchmark set, manually corrected by native annotators to serve as a standard. Human quality assessment of the primary 156-hour corpus yielded a Mean Opinion Score (MOS) of 4.6 (std = 0.7) with inter-rater reliability confirmed by a 0.68 Cohen's Kappa, validating our curation pipeline's 97.6% confidence score. The corpus maintains a 60-40 gender balance across 71,792 utterances. Our work represents a significant leap toward linguistic inclusivity in global AI. The corpus and code are open-sourced, and a demo page is available.
#### Contextual Biasing for Streaming ASR via CTC-based Word Spotting
 - **Authors:** Kai-Chen Tsai, Tien-Hong Lo, Yun-Ting Sun, Berlin Chen
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2605.18222

 - **Pdf link:** https://arxiv.org/pdf/2605.18222

 - **Abstract**
 Contextual biasing is essential to improving the recognition of rare and domain-specific words in an automatic speech recognition (ASR) system. While numerous methods have been proposed in recent years, most of them focus on offline settings and do not explicitly address the challenges of streaming ASR. For example, CTC-based word spotting (CTC-WS) have demonstrated strong performance by directly detecting keywords from CTC log-probabilities, but they are limited to offline processing and require access to the full utterance. In This work, we present a streaming extension of CTC-WS for real-time contextual biasing. Our method maintains active keyword paths across audio chunks using a stateful token passing algorithm, enabling the detection of keywords that span multiple chunks. To ensure low latency and stable output, we introduce an incremental commitment mechanism that only emits segments guaranteed not to be affected by future audio, while deferring uncertain regions. This method naturally integrates with streaming ASR pipelines and does not require modifications to the underlying acoustic model or additional training, making it practical for real-world deployment. Experimental results show that our method reduces overall WER and effectively improves keyword F-score, demonstrating its effectiveness for real-time ASR applications.
#### Taming Audio VAEs via Target-KL Regularization
 - **Authors:** Prem Seetharaman, Rithesh Kumar
 - **Subjects:** Subjects:
Sound (cs.SD); Machine Learning (cs.LG); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2605.17085

 - **Pdf link:** https://arxiv.org/pdf/2605.17085

 - **Abstract**
 Latent diffusion models have emerged as the dominant paradigm for many generation tasks including audio generation such as text-to-audio, text-to-music and text-to-speech. A key component of latent diffusion is an autoencoder (VAE) that compresses high-dimensional signals into a low frame rate continuous representation that is conducive for downstream prediction. Regularizing these VAEs is challenging, as there is a trade-off between over-regularized (poor output quality) and under-regularized (difficult to predict) latent representations. We propose a framework for studying this trade-off through compression and train Audio VAEs at specific bitrates via target-KL regularization. This allows direct comparison to well-studied discrete neural audio codec models, and the construction of rate-distortion curves for audio VAEs. We evaluate the impact of target-KL regularization on text-to-sound generation and find that sweeping compression rates is helpful in identifying the optimal generation setting.
#### Analyzing Error Propagation in Korean Spoken QA with ASR-LLM Cascades
 - **Authors:** Donghyuk Jung, Youngwon Choi
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2605.17443

 - **Pdf link:** https://arxiv.org/pdf/2605.17443

 - **Abstract**
 We analyze how automatic speech recognition (ASR) errors propagate through ASR-LLM cascades in Korean spoken question answering (SQA), focusing on downstream semantic failures that conventional ASR metrics cannot fully capture. Our analysis shows that the relative downstream degradation caused by ASR errors is consistent across LLMs with different absolute performance, suggesting that cascade degradation largely tracks ASR-stage information loss. We further identify single-character Korean ASR errors as a distinct semantic-failure channel, where the gold answer becomes entirely absent from the downstream prediction despite only a minimal transcription difference. Finally, an auxiliary comparison shows that a large audio language model outperforms an ASR-LLM pipeline with a matched language backbone in noisy Korean SQA, indicating the potential of direct audio input to mitigate transcript-induced information loss.
#### Sometin Beta Pass Notin (SBPN): Improving Multilingual ASR for Nigerian Languages via Knowledge Distillation
 - **Authors:** Sewade Ogun
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2605.17710

 - **Pdf link:** https://arxiv.org/pdf/2605.17710

 - **Abstract**
 Although modern multilingual Automatic Speech Recognition (ASR) systems support several Nigerian languages, their performance consistently lags behind high-resource languages like English and French. Nigerian languages present unique modelling hurdles, including acute data scarcity, inconsistent orthography, tonal diacritics, diverse accents, frequent code-switching, and localized named entities. To address these challenges, we developed a multilingual ASR framework utilizing a two-stage distillation process. First, we employ student-teacher knowledge distillation from existing monolingual models, conditioned on robust language-specific N-gram language models. Second, we perform iterative self improvement using pseudo-labelled data to further refine accuracy. Our method significantly bridges the performance gap, achieving on average a relative Word Error Rate (WER) reduction of 29 % over monolingual baselines. Our models also outperform state-of-the-art multilingual models across major benchmarks, including Common Voice and Fleurs. We introduce Sometin Beta Pass Notin (SBPN), a foundational multilingual ASR model covering Yorùbá, Hausa, Igbo, Nigerian Pidgin, and Nigerian English. SBPN is released in two sizes: SBPN-Base (120 M parameters) and SBPN-Large (600 M parameters). By releasing these as open foundation models, we aim to provide ASR resources for further research into the rich phonetic and cultural landscape of the region.


by Zyzzyva0381 (Windy). 


2026-05-19
