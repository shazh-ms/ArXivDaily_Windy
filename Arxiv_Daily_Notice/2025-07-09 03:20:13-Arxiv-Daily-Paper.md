# Showing new listings for Wednesday, 9 July 2025
Auto update papers at about 2:30am UTC (10:30am Beijing time) every weekday.


阅读 `Usage.md`了解如何使用此repo实现个性化的Arxiv论文推送

See `Usage.md` for instructions on how to personalize the repo. 


Keyword list: ['text-to-speech', 'text to speech', 'tts', 'LLM-based', 'speech', 'voice']


Excluded: []


### Today: 11papers 
#### Parametric Object Coding in IVAS: Efficient Coding of Multiple Audio Objects at Low Bit Rates
 - **Authors:** Andrea Eichenseer, Srikanth Korse, Guillaume Fuchs, Markus Multrus
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2507.05409

 - **Pdf link:** https://arxiv.org/pdf/2507.05409

 - **Abstract**
 The recently standardized 3GPP codec for Immersive Voice and Audio Services (IVAS) includes a parametric mode for efficiently coding multiple audio objects at low bit rates. In this mode, parametric side information is obtained from both the object metadata and the input audio objects. The side information comprises directional information, indices of two dominant objects, and the power ratio between these two dominant objects. It is transmitted to the decoder along with a stereo downmix. In IVAS, parametric object coding allows for transmitting three or four arbitrarily placed objects at bit rates of 24.4 or 32 kbit/s and faithfully reconstructing the spatial image of the original audio scene. Subjective listening tests confirm that IVAS provides a comparable immersive experience at lower bit rate and complexity compared to coding the audio objects independently using Enhanced Voice Services (EVS).
#### MMW: Side Talk Rejection Multi-Microphone Whisper on Smart Glasses
 - **Authors:** Yang Liu, Li Wan, Yiteng Huang, Yong Xu, yangyang shi, Saurabh Adya, ming sun, Florian Metze
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2507.05609

 - **Pdf link:** https://arxiv.org/pdf/2507.05609

 - **Abstract**
 Smart glasses are increasingly positioned as the next-generation interface for ubiquitous access to large language models (LLMs). Nevertheless, achieving reliable interaction in real-world noisy environments remains a major challenge, particularly due to interference from side speech. In this work, we introduce a novel side-talk rejection multi-microphone Whisper (MMW) framework for smart glasses, incorporating three key innovations. First, we propose a Mix Block based on a Tri-Mamba architecture to effectively fuse multi-channel audio at the raw waveform level, while maintaining compatibility with streaming processing. Second, we design a Frame Diarization Mamba Layer to enhance frame-level side-talk suppression, facilitating more efficient fine-tuning of Whisper models. Third, we employ a Multi-Scale Group Relative Policy Optimization (GRPO) strategy to jointly optimize frame-level and utterance-level side speech suppression. Experimental evaluations demonstrate that the proposed MMW system can reduce the word error rate (WER) by 4.95\% in noisy conditions.
#### Frequency-Specific Neural Response and Cross-Correlation Analysis of Envelope Following Responses to Native Speech and Music Using Multichannel EEG Signals: A Case Study
 - **Authors:** Md. Mahbub Hasan, Md Rakibul Hasan, Md Zakir Hossain, Tom Gedeon
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD); Signal Processing (eess.SP); Systems and Control (eess.SY)
 - **Arxiv link:** https://arxiv.org/abs/2507.05635

 - **Pdf link:** https://arxiv.org/pdf/2507.05635

 - **Abstract**
 Although native speech and music envelope following responses (EFRs) play a crucial role in auditory processing and cognition, their frequency profile, such as the dominating frequency and spectral coherence, is largely unknown. We have assumed that the auditory pathway - which transmits envelope components of speech and music to the scalp through time-varying neurophysiological processes - is a linear time-varying system, with the envelope and the multi-channel EEG responses as excitation and response, respectively. This paper investigates the transfer function of this system through two analytical techniques - time-averaged spectral responses and cross-spectral density - in the frequency domain at four different positions of the human scalp. Our findings suggest that alpha (8-11 Hz), lower gamma (53-56 Hz), and higher gamma (78-81 Hz) bands are the peak responses of the system. These frequently appearing dominant frequency responses may be the key components of familiar speech perception, maintaining attention, binding acoustic features, and memory processing. The cross-spectral density, which reflects the spatial neural coherence of the human brain, shows that 10-13 Hz, 27-29 Hz, and 62-64 Hz are common for all channel pairs. As neural coherences are frequently observed in these frequencies among native participants, we suggest that these distributed neural processes are also dominant in native speech and music perception.
#### Robust One-step Speech Enhancement via Consistency Distillation
 - **Authors:** Liang Xu, Longfei Felix Yan, W. Bastiaan Kleijn
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2507.05688

 - **Pdf link:** https://arxiv.org/pdf/2507.05688

 - **Abstract**
 Diffusion models have shown strong performance in speech enhancement, but their real-time applicability has been limited by multi-step iterative sampling. Consistency distillation has recently emerged as a promising alternative by distilling a one-step consistency model from a multi-step diffusion-based teacher model. However, distilled consistency models are inherently biased towards the sampling trajectory of the teacher model, making them less robust to noise and prone to inheriting inaccuracies from the teacher model. To address this limitation, we propose ROSE-CD: Robust One-step Speech Enhancement via Consistency Distillation, a novel approach for distilling a one-step consistency model. Specifically, we introduce a randomized learning trajectory to improve the model's robustness to noise. Furthermore, we jointly optimize the one-step model with two time-domain auxiliary losses, enabling it to recover from teacher-induced errors and surpass the teacher model in overall performance. This is the first pure one-step consistency distillation model for diffusion-based speech enhancement, achieving 54 times faster inference speed and superior performance compared to its 30-step teacher model. Experiments on the VoiceBank-DEMAND dataset demonstrate that the proposed model achieves state-of-the-art performance in terms of speech quality. Moreover, its generalization ability is validated on both an out-of-domain dataset and real-world noisy recordings.
#### ContextASR-Bench: A Massive Contextual Speech Recognition Benchmark
 - **Authors:** He Wang, Linhan Ma, Dake Guo, Xiong Wang, Lei Xie, Jin Xu, Junyang Lin
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Computation and Language (cs.CL); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2507.05727

 - **Pdf link:** https://arxiv.org/pdf/2507.05727

 - **Abstract**
 Automatic Speech Recognition (ASR) has been extensively investigated, yet prior evaluative efforts have largely been restricted to contextless paradigms. This constraint stems from the limited proficiency of conventional ASR models in context modeling and their deficiency in memory and reasoning based on world knowledge. Recent breakthroughs in the development of Large Language Models (LLMs) and corresponding Large Audio Language Models (LALMs) have markedly enhanced the visibility of general artificial intelligence capabilities. Consequently, there exists a compelling need for a benchmark that can evaluate both the generality and intelligence of ASR systems. To address this gap, we propose ContextASR-Bench: a comprehensive, large-scale benchmark designed to assess contextual speech recognition. This benchmark encompasses up to 40,000 data entries across over 10 domains, enabling a thorough evaluation of model performance in scenarios that omit or incorporate coarse-grained or fine-grained contextual information. Moreover, diverging from conventional ASR evaluations, our benchmark includes an analysis of model efficacy in recognizing named entities mentioned within the auditory input. Our extensive evaluation highlights that LALMs, with strong world knowledge and context learning capabilities, outperform conventional ASR models by a large margin. The dataset and evaluation code have been released at this https URL.
#### Dynamic Slimmable Networks for Efficient Speech Separation
 - **Authors:** Mohamed Elminshawi, Srikanth Raj Chetupalli, EmanuÃ«l A. P. Habets
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2507.06179

 - **Pdf link:** https://arxiv.org/pdf/2507.06179

 - **Abstract**
 Recent progress in speech separation has been largely driven by advances in deep neural networks, yet their high computational and memory requirements hinder deployment on resource-constrained devices. A significant inefficiency in conventional systems arises from using static network architectures that maintain constant computational complexity across all input segments, regardless of their characteristics. This approach is sub-optimal for simpler segments that do not require intensive processing, such as silence or non-overlapping speech. To address this limitation, we propose a dynamic slimmable network (DSN) for speech separation that adaptively adjusts its computational complexity based on the input signal. The DSN combines a slimmable network, which can operate at different network widths, with a lightweight gating module that dynamically determines the required width by analyzing the local input characteristics. To balance performance and efficiency, we introduce a signal-dependent complexity loss that penalizes unnecessary computation based on segmental reconstruction error. Experiments on clean and noisy two-speaker mixtures from the WSJ0-2mix and WHAM! datasets show that the DSN achieves a better performance-efficiency trade-off than individually trained static networks of different sizes.
#### Omni-Router: Sharing Routing Decisions in Sparse Mixture-of-Experts for Speech Recognition
 - **Authors:** Zijin Gu, Tatiana Likhomanenko, Navdeep Jaitly
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Artificial Intelligence (cs.AI); Machine Learning (cs.LG); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2507.05724

 - **Pdf link:** https://arxiv.org/pdf/2507.05724

 - **Abstract**
 Mixture-of-experts (MoE) architectures have expanded from language modeling to automatic speech recognition (ASR). Traditional MoE methods, such as the Switch Transformer, route experts independently within each layer. Our analysis reveals that routers in most layers make expert choices that are not strongly correlated with the choices of the routers in other layers. To increase the cooperation between experts in different layers and encourage greater specialization, we use a shared router across different MoE layers. We call this model \emph{Omni-router Transformer}. Extensive experiments on a large-scale pseudo-labeled dataset and evaluations across 10 diverse, out-of-domain ASR benchmarks demonstrate that the Omni-router Transformer is able to achieve lower training loss and consistently outperform dense and Switch Transformer models, reducing average word error rates by 11.2% and 8.2%, respectively, while providing structured expert usage and improved robustness to diverse data.
#### Non-Intrusive Binaural Speech Intelligibility Prediction Using Mamba for Hearing-Impaired Listeners
 - **Authors:** Katsuhiko Yamamoto, Koichi Miyazaki
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2507.05729

 - **Pdf link:** https://arxiv.org/pdf/2507.05729

 - **Abstract**
 Speech intelligibility prediction (SIP) models have been used as objective metrics to assess intelligibility for hearing-impaired (HI) listeners. In the Clarity Prediction Challenge 2 (CPC2), non-intrusive binaural SIP models based on transformers showed high prediction accuracy. However, the self-attention mechanism theoretically incurs high computational and memory costs, making it a bottleneck for low-latency, power-efficient devices. This may also degrade the temporal processing of binaural SIPs. Therefore, we propose Mamba-based SIP models instead of transformers for the temporal processing blocks. Experimental results show that our proposed SIP model achieves competitive performance compared to the baseline while maintaining a relatively small number of parameters. Our analysis suggests that the SIP model based on bidirectional Mamba effectively captures contextual and spatial speech information from binaural signals.
#### How to Evaluate Automatic Speech Recognition: Comparing Different Performance and Bias Measures
 - **Authors:** Tanvina Patel, Wiebke Hutiri, Aaron Yi Ding, Odette Scharenborg
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2507.05885

 - **Pdf link:** https://arxiv.org/pdf/2507.05885

 - **Abstract**
 There is increasingly more evidence that automatic speech recognition (ASR) systems are biased against different speakers and speaker groups, e.g., due to gender, age, or accent. Research on bias in ASR has so far primarily focused on detecting and quantifying bias, and developing mitigation approaches. Despite this progress, the open question is how to measure the performance and bias of a system. In this study, we compare different performance and bias measures, from literature and proposed, to evaluate state-of-the-art end-to-end ASR systems for Dutch. Our experiments use several bias mitigation strategies to address bias against different speaker groups. The findings reveal that averaged error rates, a standard in ASR research, alone is not sufficient and should be supplemented by other measures. The paper ends with recommendations for reporting ASR performance and bias to better represent a system's performance for diverse speaker groups, and overall system bias.
#### Differentiable Reward Optimization for LLM based TTS system
 - **Authors:** Changfeng Gao, Zhihao Du, Shiliang Zhang
 - **Subjects:** Subjects:
Sound (cs.SD); Artificial Intelligence (cs.AI); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2507.05911

 - **Pdf link:** https://arxiv.org/pdf/2507.05911

 - **Abstract**
 This paper proposes a novel Differentiable Reward Optimization (DiffRO) method aimed at enhancing the performance of neural codec language models based text-to-speech (TTS) systems. In contrast to conventional reinforcement learning from human feedback (RLHF) approaches applied to TTS, DiffRO directly compute the rewards based on neural codec tokens, rather than relying on synthesized audio. Furthermore, we employ the Gumbel-Softmax technique to render the reward function differentiable, thereby streamlining the RLHF training process. Additionally, we introduce a multi-task reward (MTR) model which can provide feedback from different perspectives and find that it can augment the system's capability to follow instructions this http URL results indicate that DiffRO significantly improves the pronunciation accuracy of the TTS system, achieving state-of-the-art (SOTA) WER results on the seed-tts-eval benchmark. Moreover, with the integration of the MTR model, we demonstrate the ability to control emotional and quality attributes in a zero-shot manner.
#### Speech Quality Assessment Model Based on Mixture of Experts: System-Level Performance Enhancement and Utterance-Level Challenge Analysis
 - **Authors:** Xintong Hu, Yixuan Chen, Rui Yang, Wenxiang Guo, Changhao Pan
 - **Subjects:** Subjects:
Sound (cs.SD); Artificial Intelligence (cs.AI); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2507.06116

 - **Pdf link:** https://arxiv.org/pdf/2507.06116

 - **Abstract**
 Automatic speech quality assessment plays a crucial role in the development of speech synthesis systems, but existing models exhibit significant performance variations across different granularity levels of prediction tasks. This paper proposes an enhanced MOS prediction system based on self-supervised learning speech models, incorporating a Mixture of Experts (MoE) classification head and utilizing synthetic data from multiple commercial generation models for data augmentation. Our method builds upon existing self-supervised models such as wav2vec2, designing a specialized MoE architecture to address different types of speech quality assessment tasks. We also collected a large-scale synthetic speech dataset encompassing the latest text-to-speech, speech conversion, and speech enhancement systems. However, despite the adoption of the MoE architecture and expanded dataset, the model's performance improvements in sentence-level prediction tasks remain limited. Our work reveals the limitations of current methods in handling sentence-level quality assessment, provides new technical pathways for the field of automatic speech quality assessment, and also delves into the fundamental causes of performance differences across different assessment granularities.


by Zyzzyva0381 (Windy). 


2025-07-09
