# Showing new listings for Thursday, 26 March 2026
Auto update papers at about 2:30am UTC (10:30am Beijing time) every weekday.


阅读 `Usage.md`了解如何使用此repo实现个性化的Arxiv论文推送

See `Usage.md` for instructions on how to personalize the repo. 


Keyword list: ['text-to-speech', 'text to speech', 'tts', 'LLM-based', 'speech', 'voice']


Excluded: []


### Today: 7papers 
#### Crab: Multi Layer Contrastive Supervision to Improve Speech Emotion Recognition Under Both Acted and Natural Speech Condition
 - **Authors:** Lucas H. Ueda, João G. T. Lima, Paula D. P. Costa
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2603.23673

 - **Pdf link:** https://arxiv.org/pdf/2603.23673

 - **Abstract**
 Speech Emotion Recognition (SER) in real-world scenarios remains challenging due to severe class imbalance and the prevalence of spontaneous, natural speech. While recent approaches leverage self-supervised learning (SSL) representations and multimodal fusion of speech and text, most existing methods apply supervision only at the final classification layer, limiting the discriminative power of intermediate representations. In this work, we propose Crab (Contrastive Representation and Multimodal Aligned Bottleneck), a bimodal Cross-Modal Transformer architecture that integrates speech representations from WavLM and textual representations from RoBERTa, together with a novel \textit{Multi Layer Contrastive Supervision} (MLCS) strategy. MLCS injects multi-positive contrastive learning signals at multiple layers of the network, encouraging emotionally discriminative representations throughout the model without introducing additional parameters at inference time. To further address data imbalance, we adopt weighted cross-entropy during training. We evaluate the proposed approach on three benchmark datasets covering different degrees of emotional naturalness: IEMOCAP, MELD, and MSP-Podcast 2.0. Experimental results demonstrate that Crab consistently outperforms strong unimodal and multimodal baselines across all datasets, with particularly large gains under naturalistic and highly imbalanced conditions. These findings highlight the effectiveness of \textit{Multi Layer Contrastive Supervision} as a general and robust strategy for SER. Official implementation can be found in this https URL.
#### Autoregressive Guidance of Deep Spatially Selective Filters using Bayesian Tracking for Efficient Extraction of Moving Speakers
 - **Authors:** Jakob Kienegger, Timo Gerkmann
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Machine Learning (cs.LG); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2603.23723

 - **Pdf link:** https://arxiv.org/pdf/2603.23723

 - **Abstract**
 Deep spatially selective filters achieve high-quality enhancement with real-time capable architectures for stationary speakers of known directions. To retain this level of performance in dynamic scenarios when only the speakers' initial directions are given, accurate, yet computationally lightweight tracking algorithms become necessary. Assuming a frame-wise causal processing style, temporal feedback allows for leveraging the enhanced speech signal to improve tracking performance. In this work, we investigate strategies to incorporate the enhanced signal into lightweight tracking algorithms and autoregressively guide deep spatial filters. Our proposed Bayesian tracking algorithms are compatible with arbitrary deep spatial filters. To increase the realism of simulated trajectories during development and evaluation, we propose and publish a novel dataset based on the social force model. Results validate that the autoregressive incorporation significantly improves the accuracy of our Bayesian trackers, resulting in superior enhancement with none or only negligibly increased computational overhead. Real-world recordings complement these findings and demonstrate the generalizability of our methods to unseen, challenging acoustic conditions.
#### ACAVCaps: Enabling large-scale training for fine-grained and diverse audio understanding
 - **Authors:** Yadong Niu, Tianzi Wang, Heinrich Dinkel, Xingwei Sun, Jiahao Zhou, Gang Li, Jizhong Liu, Junbo Zhang, Jian Luan
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2603.24038

 - **Pdf link:** https://arxiv.org/pdf/2603.24038

 - **Abstract**
 General audio understanding is a fundamental goal for large audio-language models, with audio captioning serving as a cornerstone task for their development. However, progress in this domain is hindered by existing datasets, which lack the scale and descriptive granularity required to train truly versatile models. To address this gap, we introduce ACAVCaps, a new large-scale, fine-grained, and multi-faceted audio captioning dataset. Derived from the ACAV100M collection, ACAVCaps is constructed using a multi-expert pipeline that analyzes audio from diverse perspectives-including speech, music, and acoustic properties-which are then synthesized into rich, detailed descriptions by a large language model. Experimental results demonstrate that models pre-trained on ACAVCaps exhibit substantially stronger generalization capabilities on various downstream tasks compared to those trained on other leading captioning datasets. The dataset is available at this https URL.
#### How Open is Open TTS? A Practical Evaluation of Open Source TTS Tools for Romanian
 - **Authors:** Teodora Răgman, Adrian Bogdan Stânea, Horia Cucu, Adriana Stan
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2603.24116

 - **Pdf link:** https://arxiv.org/pdf/2603.24116

 - **Abstract**
 Open-source text-to-speech (TTS) frameworks have emerged as highly adaptable platforms for developing speech synthesis systems across a wide range of languages. However, their applicability is not uniform -- particularly when the target language is under-resourced or when computational resources are constrained. In this study, we systematically assess the feasibility of building novel TTS models using four widely adopted open-source architectures: FastPitch, VITS, Grad-TTS, and Matcha-TTS. Our evaluation spans multiple dimensions, including qualitative aspects such as ease of installation, dataset preparation, and hardware requirements, as well as quantitative assessments of synthesis quality for Romanian. We employ both objective metrics and subjective listening tests to evaluate intelligibility, speaker similarity, and naturalness of the generated speech. The results reveal significant challenges in tool chain setup, data preprocessing, and computational efficiency, which can hinder adoption in low-resource contexts. By grounding the analysis in reproducible protocols and accessible evaluation criteria, this work aims to inform best practices and promote more inclusive, language-diverse TTS development. All information needed to reproduce this study (i.e. code and data) are available in our git repository: this https URL
#### ArrayDPS-Refine: Generative Refinement of Discriminative Multi-Channel Speech Enhancement
 - **Authors:** Zhongweiyang Xu, Ashutosh Pandey, Juan Azcarreta, Zhaoheng Ni, Sanjeel Parekh, Buye Xu
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2603.24385

 - **Pdf link:** https://arxiv.org/pdf/2603.24385

 - **Abstract**
 Multi-channel speech enhancement aims to recover clean speech from noisy multi-channel recordings. Most deep learning methods employ discriminative training, which can lead to non-linear distortions from regression-based objectives, especially under challenging environmental noise conditions. Inspired by ArrayDPS for unsupervised multi-channel source separation, we introduce ArrayDPS-Refine, a method designed to enhance the outputs of discriminative models using a clean speech diffusion prior. ArrayDPS-Refine is training-free, generative, and array-agnostic. It first estimates the noise spatial covariance matrix (SCM) from the enhanced speech produced by a discriminative model, then uses this estimated noise SCM for diffusion posterior sampling. This approach allows direct refinement of any discriminative model's output without retraining. Our results show that ArrayDPS-Refine consistently improves the performance of various discriminative models, including state-of-the-art waveform and STFT domain models. Audio demos are provided at this https URL.
#### YingMusic-Singer: Controllable Singing Voice Synthesis with Flexible Lyric Manipulation and Annotation-free Melody Guidance
 - **Authors:** Chunbo Hao, Junjie Zheng, Guobin Ma, Yuepeng Jiang, Huakang Chen, Wenjie Tian, Gongyu Chen, Zihao Chen, Lei Xie
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2603.24589

 - **Pdf link:** https://arxiv.org/pdf/2603.24589

 - **Abstract**
 Regenerating singing voices with altered lyrics while preserving melody consistency remains challenging, as existing methods either offer limited controllability or require laborious manual alignment. We propose YingMusic-Singer, a fully diffusion-based model enabling melody-controllable singing voice synthesis with flexible lyric manipulation. The model takes three inputs: an optional timbre reference, a melody-providing singing clip, and modified lyrics, without manual alignment. Trained with curriculum learning and Group Relative Policy Optimization, YingMusic-Singer achieves stronger melody preservation and lyric adherence than Vevo2, the most comparable baseline supporting melody control without manual alignment. We also introduce LyricEditBench, the first benchmark for melody-preserving lyric modification evaluation. The code, weights, benchmark, and demos are publicly available at this https URL.
#### Semantic-Aware Interruption Detection in Spoken Dialogue Systems: Benchmark, Metric, and Model
 - **Authors:** Kangxiang Xia, Bingshen Mu, Xian Shi, Jin Xu, Lei Xie
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2603.24144

 - **Pdf link:** https://arxiv.org/pdf/2603.24144

 - **Abstract**
 Achieving natural full-duplex interaction in spoken dialogue systems (SDS) remains a challenge due to the difficulty of accurately detecting user interruptions. Current solutions are polarized between "trigger-happy" VAD-based methods that misinterpret backchannels and robust end-to-end models that exhibit unacceptable response delays. Moreover, the absence of real-world benchmarks and holistic metrics hinders progress in the field. This paper presents a comprehensive frame-work to overcome these limitations. We first introduce SID-Bench, the first benchmark for semantic-aware interruption detection built entirely from real-world human dialogues. To provide a rigorous assessment of the responsiveness-robustness trade-off, we propose the Average Penalty Time (APT) metric, which assigns a temporal cost to both false alarms and late responses. Building on this framework, we design an LLM-based detection model optimized through a novel training paradigm to capture subtle semantic cues of intent. Experimental results show that our model significantly outperforms mainstream baselines, achieving a nearly threefold reduction in APT. By successfully resolving the long-standing tension between speed and stability, our work establishes a new state-of-the-art for intelligent interruption handling in SDS. To facilitate future research, SID-Bench and the associated code are available at: this https URL.


by Zyzzyva0381 (Windy). 


2026-03-26
