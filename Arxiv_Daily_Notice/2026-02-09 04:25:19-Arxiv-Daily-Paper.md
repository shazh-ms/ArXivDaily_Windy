# Showing new listings for Monday, 9 February 2026
Auto update papers at about 2:30am UTC (10:30am Beijing time) every weekday.


阅读 `Usage.md`了解如何使用此repo实现个性化的Arxiv论文推送

See `Usage.md` for instructions on how to personalize the repo. 


Keyword list: ['text-to-speech', 'text to speech', 'tts', 'LLM-based', 'speech', 'voice']


Excluded: []


### Today: 5papers 
#### From Hallucination to Articulation: Language Model-Driven Losses for Ultra Low-Bitrate Neural Speech Coding
 - **Authors:** Jayeon Yi, Minje Kim
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2602.06213

 - **Pdf link:** https://arxiv.org/pdf/2602.06213

 - **Abstract**
 ``Phoneme Hallucinations (PH)'' commonly occur in low-bitrate DNN-based codecs. It is the generative decoder's attempt to synthesize plausible outputs from excessively compressed tokens missing some semantic information. In this work, we propose language model-driven losses (LM loss) and show they may alleviate PHs better than a semantic distillation (SD) objective in very-low-bitrate settings. The proposed LM losses build upon language models pretrained to associate speech with text. When ground-truth transcripts are unavailable, we propose to modify a popular automatic speech recognition (ASR) model, Whisper, to compare the decoded utterance against the ASR-inferred transcriptions of the input speech. Else, we propose to use the timed-text regularizer (TTR) to compare WavLM representations of the decoded utterance against BERT representations of the ground-truth transcriptions. We test and compare LM losses against an SD objective, using a reference codec whose three-stage training regimen was designed after several popular codecs. Subjective and objective evaluations conclude that LM losses may provide stronger guidance to extract semantic information from self-supervised speech representations, boosting human-perceived semantic adherence while preserving overall output quality. Demo samples, code, and checkpoints are available online.
#### B-GRPO: Unsupervised Speech Emotion Recognition based on Batched-Group Relative Policy Optimization
 - **Authors:** Yingying Gao, Shilei Zhang, Runyan Yang, Zihao Cui, Junlan Feng
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2602.06290

 - **Pdf link:** https://arxiv.org/pdf/2602.06290

 - **Abstract**
 Unsupervised speech emotion recognition (SER) focuses on addressing the problem of data sparsity and annotation bias of emotional speech. Reinforcement learning (RL) is a promising method which enhances the performance through rule-based or model-based verification functions rather than human annotations. We treat the sample selection during the learning process as a long-term procedure and whether to select a sample as the action to make policy, thus achieving the application of RL to measure sample quality in SER. We propose a modified Group Relative Policy Optimization (GRPO) to adapt it to classification problems, which takes the samples in a batch as a group and uses the average reward of these samples as the baseline to calculate the advantage. And rather than using a verifiable reward function as in GRPO, we put forward self-reward functions and teacher-reward functions to encourage the model to produce high-confidence outputs. Experiments indicate that the proposed method improves the performance of baseline without RL by 19.8%.
#### The Combination of Several Decorrelation Methods to Improve Acoustic Feedback Cancellation
 - **Authors:** Klaus Linhard, Philipp Bulling
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2602.06921

 - **Pdf link:** https://arxiv.org/pdf/2602.06921

 - **Abstract**
 This paper extends an acoustic feedback cancellation system by incorporating multiple decorrelation methods. The baseline system is based on a frequency-domain Kalman filter implemented in a multi-delay structure. The proposed extensions include a variable time delay line, prediction, distortion compensation, and a simplified reverberation model. Each extension is analyzed, and a practical parameter range is defined. While existing literature often focuses on a single extension, such as prediction, to describe an optimal system, this work demonstrates that each individual extension contributes to performance improvements. Furthermore, the combination of all proposed extensions results in a superior system. The evaluation is conducted using publicly available datasets, with performance assessed through system distance metrics and the objective speech quality measure PSEQ.
#### Scaling Speech Tokenizers with Diffusion Autoencoders
 - **Authors:** Yuancheng Wang, Zhenyu Tang, Yun Wang, Arthur Hinsvark, Yingru Liu, Yinghao Li, Kainan Peng, Junyi Ao, Mingbo Ma, Mike Seltzer, Qing He, Xubo Liu
 - **Subjects:** Subjects:
Sound (cs.SD); Artificial Intelligence (cs.AI); Machine Learning (cs.LG); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2602.06602

 - **Pdf link:** https://arxiv.org/pdf/2602.06602

 - **Abstract**
 Speech tokenizers are foundational to speech language models, yet existing approaches face two major challenges: (1) balancing trade-offs between encoding semantics for understanding and acoustics for reconstruction, and (2) achieving low bit rates and low token rates. We propose Speech Diffusion Tokenizer (SiTok), a diffusion autoencoder that jointly learns semantic-rich representations through supervised learning and enables high-fidelity audio reconstruction with diffusion. We scale SiTok to 1.6B parameters and train it on 2 million hours of speech. Experiments show that SiTok outperforms strong baselines on understanding, reconstruction and generation tasks, at an extremely low token rate of $12.5$ Hz and a bit-rate of 200 bits-per-second.
#### AI-Generated Music Detection in Broadcast Monitoring
 - **Authors:** David Lopez-Ayala, Asier Cabello, Pablo Zinemanas, Emilio Molina, Martin Rocamora
 - **Subjects:** Subjects:
Sound (cs.SD); Artificial Intelligence (cs.AI); Audio and Speech Processing (eess.AS); Signal Processing (eess.SP)
 - **Arxiv link:** https://arxiv.org/abs/2602.06823

 - **Pdf link:** https://arxiv.org/pdf/2602.06823

 - **Abstract**
 AI music generators have advanced to the point where their outputs are often indistinguishable from human compositions. While detection methods have emerged, they are typically designed and validated in music streaming contexts with clean, full-length tracks. Broadcast audio, however, poses a different challenge: music appears as short excerpts, often masked by dominant speech, conditions under which existing detectors fail. In this work, we introduce AI-OpenBMAT, the first dataset tailored to broadcast-style AI-music detection. It contains 3,294 one-minute audio excerpts (54.9 hours) that follow the duration patterns and loudness relations of real television audio, combining human-made production music with stylistically matched continuations generated with Suno v3.5. We benchmark a CNN baseline and state-of-the-art SpectTTTra models to assess SNR and duration robustness, and evaluate on a full broadcast scenario. Across all settings, models that excel in streaming scenarios suffer substantial degradation, with F1-scores dropping below 60% when music is in the background or has a short duration. These results highlight speech masking and short music length as critical open challenges for AI music detection, and position AI-OpenBMAT as a benchmark for developing detectors capable of meeting industrial broadcast requirements.


by Zyzzyva0381 (Windy). 


2026-02-09
