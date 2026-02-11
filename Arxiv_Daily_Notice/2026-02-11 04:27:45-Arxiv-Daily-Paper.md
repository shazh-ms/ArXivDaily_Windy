# Showing new listings for Wednesday, 11 February 2026
Auto update papers at about 2:30am UTC (10:30am Beijing time) every weekday.


阅读 `Usage.md`了解如何使用此repo实现个性化的Arxiv论文推送

See `Usage.md` for instructions on how to personalize the repo. 


Keyword list: ['text-to-speech', 'text to speech', 'tts', 'LLM-based', 'speech', 'voice']


Excluded: []


### Today: 8papers 
#### Soft Clustering Anchors for Self-Supervised Speech Representation Learning in Joint Embedding Prediction Architectures
 - **Authors:** Georgios Ioannides, Adrian Kieback, Judah Goldfeder, Linsey Pang, Aman Chadha, Aaron Elkins, Yann LeCun, Ravid Shwartz-Ziv
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Artificial Intelligence (cs.AI); Machine Learning (cs.LG); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2602.09040

 - **Pdf link:** https://arxiv.org/pdf/2602.09040

 - **Abstract**
 Joint Embedding Predictive Architectures (JEPA) offer a promising approach to self-supervised speech representation learning, but suffer from representation collapse without explicit grounding. We propose GMM-Anchored JEPA, which fits a Gaussian Mixture Model once on log-mel spectrograms and uses its frozen soft posteriors as auxiliary targets throughout training. A decaying supervision schedule allows GMM regularization to dominate early training before gradually yielding to the JEPA objective. Unlike HuBERT and WavLM, which require iterative re-clustering, our approach clusters input features once with soft rather than hard assignments. On ~50k hours of speech, GMM anchoring improves ASR (28.68% vs. 33.22% WER), emotion recognition (67.76% vs. 65.46%), and slot filling (64.7% vs. 59.1% F1) compared to a WavLM-style baseline with matched compute. Cluster analysis shows GMM-anchored representations achieve up to 98% entropy compared to 31% for WavLM-style, indicating substantially more uniform cluster utilization. Code is made available at this https URL.
#### Windowed SummaryMixing: An Efficient Fine-Tuning of Self-Supervised Learning Models for Low-resource Speech Recognition
 - **Authors:** Aditya Srinivas Menon, Kumud Tripathi, Raj Gohil, Pankaj Wasnik
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Machine Learning (cs.LG); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2602.09043

 - **Pdf link:** https://arxiv.org/pdf/2602.09043

 - **Abstract**
 Self-supervised learning (SSL) has advanced speech processing but suffers from quadratic complexity due to self-attention. To address this, SummaryMixing (SM) has been proposed as a linear-time alternative that summarizes entire utterances using mean pooling but lacks sufficient local context. In this work, we introduce Windowed SummaryMixing (WSM), which enhances SM by integrating local neighborhood summaries alongside the global summary, maintaining efficiency while improving temporal dependencies. Additionally, we introduce a selective fine-tuning approach, replacing self-attention layers in SSL models with WSM blocks and fine-tuning only these blocks in low-resource settings. Our approach improves ASR performance while reducing peak VRAM usage by 40\% in the SSL models. WSM blocks have linear-time complexity with enhanced context awareness. Selectively replacing some attention layers reduces compute, memory, and latency, making it ideal for low-resource speech recognition.
#### Beyond the Utterance: An Empirical Study of Very Long Context Speech Recognition
 - **Authors:** Robert Flynn, Anton Ragni
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2602.09044

 - **Pdf link:** https://arxiv.org/pdf/2602.09044

 - **Abstract**
 Automatic speech recognition (ASR) models are normally trained to operate over single utterances, with a short duration of less than 30 seconds. This choice has been made in part due to computational constraints, but also reflects a common, but often inaccurate, modelling assumption that treats utterances as independent and identically distributed samples. When long-format audio recordings are available, to work with such systems, these recordings must first be segmented into short utterances and processed independently. In this work, we show that due to recent algorithmic and hardware advances, this is no longer necessary, and current attention-based approaches can be used to train ASR systems that operate on sequences of over an hour in length. Therefore, to gain a better understanding of the relationship between the training/evaluation sequence length and performance, we train ASR models on large-scale data using 10 different sequence lengths from 10 seconds up to 1 hour. The results show a benefit from using up to 21.8 minutes of context, with up to a 14.2% relative improvement from a short context baseline in our primary experiments. Through modifying various architectural components, we find that the method of encoding positional information and the model's width/depth are important factors when working with long sequences. Finally, a series of evaluations using synthetic data are constructed to help analyse the model's use of context. From these results, it is clear that both linguistic and acoustic aspects of the distant context are being used by the model.
#### TVTSyn: Content-Synchronous Time-Varying Timbre for Streaming Voice Conversion and Anonymization
 - **Authors:** Waris Quamer, Mu-Ruei Tseng, Ghady Nasrallah, Ricardo Gutierrez-Osuna
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Computation and Language (cs.CL); Machine Learning (cs.LG); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2602.09389

 - **Pdf link:** https://arxiv.org/pdf/2602.09389

 - **Abstract**
 Real-time voice conversion and speaker anonymization require causal, low-latency synthesis without sacrificing intelligibility or naturalness. Current systems have a core representational mismatch: content is time-varying, while speaker identity is injected as a static global embedding. We introduce a streamable speech synthesizer that aligns the temporal granularity of identity and content via a content-synchronous, time-varying timbre (TVT) representation. A Global Timbre Memory expands a global timbre instance into multiple compact facets; frame-level content attends to this memory, a gate regulates variation, and spherical interpolation preserves identity geometry while enabling smooth local changes. In addition, a factorized vector-quantized bottleneck regularizes content to reduce residual speaker leakage. The resulting system is streamable end-to-end, with <80 ms GPU latency. Experiments show improvements in naturalness, speaker transfer, and anonymization compared to SOTA streaming baselines, establishing TVT as a scalable approach for privacy-preserving and expressive speech synthesis under strict latency budgets.
#### BioME: A Resource-Efficient Bioacoustic Foundational Model for IoT Applications
 - **Authors:** Heitor R. Guimarães, Abhishek Tiwari, Mahsa Abdollahi, Anderson R. Avila, Tiago H. Falk
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2602.09970

 - **Pdf link:** https://arxiv.org/pdf/2602.09970

 - **Abstract**
 Passive acoustic monitoring has become a key strategy in biodiversity assessment, conservation, and behavioral ecology, especially as Internet-of-Things (IoT) devices enable continuous in situ audio collection at scale. While recent self-supervised learning (SSL)-based audio encoders, such as BEATs and AVES, have shown strong performance in bioacoustic tasks, their computational cost and limited robustness to unseen environments hinder deployment on resource-constrained platforms. In this work, we introduce BioME, a resource-efficient audio encoder designed for bioacoustic applications. BioME is trained via layer-to-layer distillation from a high-capacity teacher model, enabling strong representational transfer while reducing the parameter count by 75%. To further improve ecological generalization, the model is pretrained on multi-domain data spanning speech, environmental sounds, and animal vocalizations. A key contribution is the integration of modulation-aware acoustic features via FiLM conditioning, injecting a DSP-inspired inductive bias that enhances feature disentanglement in low-capacity regimes. Across multiple bioacoustic tasks, BioME matches or surpasses the performance of larger models, including its teacher, while being suitable for resource-constrained IoT deployments. For reproducibility, code and pretrained checkpoints are publicly available.
#### DSFlow: Dual Supervision and Step-Aware Architecture for One-Step Flow Matching Speech Synthesis
 - **Authors:** Bin Lin, Peng Yang, Chao Yan, Xiaochen Liu, Wei Wang, Boyong Wu, Pengfei Tan, Xuerui Yang
 - **Subjects:** Subjects:
Sound (cs.SD); Artificial Intelligence (cs.AI); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2602.09041

 - **Pdf link:** https://arxiv.org/pdf/2602.09041

 - **Abstract**
 Flow-matching models have enabled high-quality text-to-speech synthesis, but their iterative sampling process during inference incurs substantial computational cost. Although distillation is widely used to reduce the number of inference steps, existing methods often suffer from process variance due to endpoint error accumulation. Moreover, directly reusing continuous-time architectures for discrete, fixed-step generation introduces structural parameter inefficiencies. To address these challenges, we introduce DSFlow, a modular distillation framework for few-step and one-step synthesis. DSFlow reformulates generation as a discrete prediction task and explicitly adapts the student model to the target inference regime. It improves training stability through a dual supervision strategy that combines endpoint matching with deterministic mean-velocity alignment, enforcing consistent generation trajectories across inference steps. In addition, DSFlow improves parameter efficiency by replacing continuous-time timestep conditioning with lightweight step-aware tokens, aligning model capacity with the significantly reduced timestep space of the discrete task. Extensive experiments across diverse flow-based text-to-speech architectures demonstrate that DSFlow consistently outperforms standard distillation approaches, achieving strong few-step and one-step synthesis quality while reducing model parameters and inference cost.
#### Gencho: Room Impulse Response Generation from Reverberant Speech and Text via Diffusion Transformers
 - **Authors:** Jackie Lin, Jiaqi Su, Nishit Anand, Zeyu Jin, Minje Kim, Paris Smaragdis
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2602.09233

 - **Pdf link:** https://arxiv.org/pdf/2602.09233

 - **Abstract**
 Blind room impulse response (RIR) estimation is a core task for capturing and transferring acoustic properties; yet existing methods often suffer from limited modeling capability and degraded performance under unseen conditions. Moreover, emerging generative audio applications call for more flexible impulse response generation methods. We propose Gencho, a diffusion-transformer-based model that predicts complex spectrogram RIRs from reverberant speech. A structure-aware encoder leverages isolation between early and late reflections to encode the input audio into a robust representation for conditioning, while the diffusion decoder generates diverse and perceptually realistic impulse responses from it. Gencho integrates modularly with standard speech processing pipelines for acoustic matching. Results show richer generated RIRs than non-generative baselines while maintaining strong performance in standard RIR metrics. We further demonstrate its application to text-conditioned RIR generation, highlighting Gencho's versatility for controllable acoustic simulation and generative audio tasks.
#### Covo-Audio Technical Report
 - **Authors:** Wenfu Wang, Chenxing Li, Liqiang Zhang, Yiyang Zhao, Yuxiang Zou, Hanzhao Li, Mingyu Cui, Hao Zhang, Kun Wei, Le Xu, Zikang Huang, Jiajun Xu, Jiliang Hu, Xiang He, Zeyu Xie, Jiawen Kang, Youjun Chen, Meng Yu, Dong Yu, Rilin Chen, Linlin Di, Shulin Feng, Na Hu, Yang Liu, Bang Wang, Shan Yang
 - **Subjects:** Subjects:
Sound (cs.SD); Computation and Language (cs.CL); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2602.09823

 - **Pdf link:** https://arxiv.org/pdf/2602.09823

 - **Abstract**
 In this work, we present Covo-Audio, a 7B-parameter end-to-end LALM that directly processes continuous audio inputs and generates audio outputs within a single unified architecture. Through large-scale curated pretraining and targeted post-training, Covo-Audio achieves state-of-the-art or competitive performance among models of comparable scale across a broad spectrum of tasks, including speech-text modeling, spoken dialogue, speech understanding, audio understanding, and full-duplex voice interaction. Extensive evaluations demonstrate that the pretrained foundation model exhibits strong speech-text comprehension and semantic reasoning capabilities on multiple benchmarks, outperforming representative open-source models of comparable scale. Furthermore, Covo-Audio-Chat, the dialogue-oriented variant, demonstrates strong spoken conversational abilities, including understanding, contextual reasoning, instruction following, and generating contextually appropriate and empathetic responses, validating its applicability to real-world conversational assistant scenarios. Covo-Audio-Chat-FD, the evolved full-duplex model, achieves substantially superior performance on both spoken dialogue capabilities and full-duplex interaction behaviors, demonstrating its competence in practical robustness. To mitigate the high cost of deploying end-to-end LALMs for natural conversational systems, we propose an intelligence-speaker decoupling strategy that separates dialogue intelligence from voice rendering, enabling flexible voice customization with minimal text-to-speech (TTS) data while preserving dialogue performance. Overall, our results highlight the strong potential of 7B-scale models to integrate sophisticated audio intelligence with high-level semantic reasoning, and suggest a scalable path toward more capable and versatile LALMs.


by Zyzzyva0381 (Windy). 


2026-02-11
