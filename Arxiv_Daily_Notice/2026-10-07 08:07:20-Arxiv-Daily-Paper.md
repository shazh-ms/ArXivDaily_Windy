# Showing new listings for Wednesday, 7 October 2026
Auto update papers at about 2:30am UTC (10:30am Beijing time) every weekday.


阅读 `Usage.md`了解如何使用此repo实现个性化的Arxiv论文推送

See `Usage.md` for instructions on how to personalize the repo. 


Keyword list: ['text-to-speech', 'text to speech', 'tts', 'LLM-based', 'speech', 'voice']


Excluded: []


### Today: 10papers 
#### GIVE-KWS: Gated Injection of Visual Evidence for Noise-Robust Query-by-Example Keyword Spotting
 - **Authors:** Ming-Hsiang Hu, Kuan-Tang Huang, Hung-Shin Lee, Berlin Chen
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Computation and Language (cs.CL); Multimedia (cs.MM); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2610.07046

 - **Pdf link:** https://arxiv.org/pdf/2610.07046

 - **Abstract**
 Visual speech promises noise-robust keyword spotting, yet a visual stream is not necessarily used. On a tri-modal query-by-example keyword spotting (QbyE-KWS) benchmark, we find that a system with a task-trained visual encoder comes within 2 percentage points of a text-and-audio system in equal error rate (EER) at -10 dB, and link this gap to the encoder's lack of phonemic information. We present GIVE-KWS, whose fusion stage, GIVE (Gated Injection of Visual Evidence), conditions query audio on lip motion through gated cross-attention. We show that visual robustness depends on two interacting conditions: a phoneme-bearing visual representation, and fusion that injects visual evidence rather than rescaling audio features. Under a phoneme-bearing encoder, injection yields an effective SNR gain of 4.0-9.3 dB over masking at -10 dB, whereas under a phoneme-poor one it nearly vanishes. Relative to the benchmark system, GIVE-KWS reduces unseen-keyword EER by 72.9% at -10 dB and 62.8% on average.
#### SEAL: Mixture-Closed Additive Reconstruction and Refinement-Aware Expert Routing for Efficient Speech Separation
 - **Authors:** Shao-Chun Hu, Zi-Xiang Lin, Jeih-Weih Hung, Hung-Shin Lee
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Computation and Language (cs.CL); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2610.07047

 - **Pdf link:** https://arxiv.org/pdf/2610.07047

 - **Abstract**
 Compact time-frequency separators that mask the mixture and refine through a shared cell face two limits. First, a bounded multiplicative mask only scales a mixture bin, so where overlapping components cancel, the estimate stays small. Second, a shared cell applies the same weights to every time-frequency token at every step, so enlarging it adds compute everywhere. We present SEAL (Sparse Expert routing with Additive Latent reconstruction) to address both. For reconstruction, a zero-sum additive residual bounded by the local mixture amplitude lets estimates be nonzero where components cancel yet still sum to the mixture. For routing, a query built from acoustic and inter-step evidence sends each token to one of six residual experts, and a norm cap keeps the step cue from overriding clear acoustic evidence. On EchoSet, SEAL (small) surpasses TIGER (small) by 0.31 dB SI-SDRi with 28% fewer parameters and 2.9 times fewer MACs, and SEAL (large) is within 0.07 dB SI-SDRi of TIGER (large) at 3.1 times fewer MACs.
#### Region-Aware Masking for Accent-Robust Cross-Lingual Text-to-Speech
 - **Authors:** Haoqi Li, Shivam Mehta, Ravi Teja Gadde, Yinghong Lan
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2610.07524

 - **Pdf link:** https://arxiv.org/pdf/2610.07524

 - **Abstract**
 Accent leakage remains a critical challenge in cross-lingual zero-shot text-to-speech (TTS), where models inadvertently transfer both the speaker's timbre and source-language accent into the target-language output. In mask-reconstruction TTS this is amplified by a train--inference mismatch: training reconstructs from same-language context, while inference conditions on cross-lingual context. We close this gap by concatenating utterances from two languages and placing the reconstruction mask relative to the language boundary, so training presents the cross-lingual prompting pattern the model encounters at inference. This requires no language IDs, accent labels, parallel same-speaker recordings, or architectural changes---only the mask geometry differs. In a listening study with native speakers, region-aware masking reduces perceived accent from 4.3 to 0.5 on a 0-5 scale against the unadapted baseline, and by 18-51\% in seen regimes and 35-55\% on unseen source languages against bilingual adaptation alone, with no measurable loss in intelligibility or naturalness. Comparing mask geometries shows that placement, not the amount masked, governs the balance between accent suppression and speaker preservation: masks confined to the reconstruction region suppress accent most but retain the reference speaker least reliably, whereas two-region masking achieves both. Mask geometry is thus a simple, effective lever for accent robustness.
#### HINTT Submission to the 2nd MLC-SLM Challenge: Comparing Cascaded and Unified Approaches to Diarization and ASR
 - **Authors:** Takanori Ashihara, Kohei Matsuura, Masato Mimura
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Computation and Language (cs.CL)
 - **Arxiv link:** https://arxiv.org/abs/2610.08063

 - **Pdf link:** https://arxiv.org/pdf/2610.08063

 - **Abstract**
 This paper presents the HINTT system submitted to the 2nd Challenge and Workshop on Multilingual Conversational Speech Language Model (MLC-SLM). We address multilingual speaker-attributed ASR, where systems must determine who spoke when and what was spoken. We investigate two modeling strategies for this problem: a cascaded pipeline that combines speaker diarization with speech-LLM-based ASR, and a unified speech LLM that directly generates speaker labels, timestamps, and transcriptions. Our final submission is based on the cascaded pipeline, consisting of a fine-tuned DiariZen diarization model, a fine-tuned Qwen3-ASR model, and LLM-based generative error correction. For comparison, we also fine-tune VibeVoice-ASR as a unified model using the same official training data. All task-specific fine-tuning and model selection are performed using only the official MLC-SLM data, without external data or pseudo-labels. Experimental results demonstrate that the cascaded system remains more reliable under the MLC-SLM Task 1 conditions, while unified speech LLMs offer a promising direction for future speaker-attributed ASR.
#### Conversation Is a Two-Body Problem: Dyadic Evaluation of Full-Duplex Dialogue Models
 - **Authors:** Sungnyun Kim, Sungwoo Cho, Jihwan Oh, Se-Young Yun
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Computation and Language (cs.CL)
 - **Arxiv link:** https://arxiv.org/abs/2610.08125

 - **Pdf link:** https://arxiv.org/pdf/2610.08125

 - **Abstract**
 Full-duplex spoken dialogue models listen and speak at the same time, enabling voice agents to have natural, low-latency interactions that turn-based systems cannot offer. However, they are commonly evaluated against single-sided interlocutors: pre-recorded audio that cannot react, or an automated examiner that reacts in real time but only administers a fixed sequence of tests and is never graded. These single-sided frameworks evaluate only half of a two-body problem, where turn-taking, overlap, and interruption are joint products of two coupled speakers. We propose DyaFDB, a framework that evaluates full-duplex models in a dyadic setup: two models converse directly under assigned roles with cooperative or conflicting goals, and both sides are scored offline with an external judge. DyaFDB probes how the two models behave toward each other, such as how they take turns or carry an assigned role under different interests. We instantiate four tasks as 140 scenarios and record 7,560 conversations, covering six self- and cross-play pairings. Throughout the experiments, we observe that how a model behaves continually reshapes its partner. We thus demonstrate that each model must be both the examiner and examinee of the other, and no single fixed interlocutor can play both parts. We will release the scenarios, role prompts, and recording protocols between two full-duplex models, without any pre-recorded audio.
#### Voice Anonymization Made Simple: Training-Free Anonymization with Projected Classifier-Free Guidance
 - **Authors:** Xiang Shi, Han Zhu, Ming Li, Xiaoxiao Miao
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2610.08276

 - **Pdf link:** https://arxiv.org/pdf/2610.08276

 - **Abstract**
 Voice anonymization aims to conceal speaker identity while preserving linguistic and paralinguistic information, yet many existing methods require dedicated training. We propose a training-free, two-stage anonymization method for pretrained flow-matching voice conversion systems. In the condition-construction stage, the source content representation is regenerated under a randomly selected speaker context to reduce residual source-speaker information while preserving linguistic content. In the speech-generation stage, the source speech is used to identify the generation direction associated with the original speaker, and the source-aligned component is suppressed while useful content guidance is retained. The proposed method modifies only inference-time conditioning and guidance, without updating any pretrained parameters. Experiments on VoicePrivacy 2026 Track 1 show that the two-stage method improves speaker privacy while maintaining competitive word error rate and emotion recognition performance.
#### Audiovisual joint learning for end-to-end hearing aids
 - **Authors:** You-Jin Li, Yu Tsao, Borching Su, Kuan-Chung Ting, Fan-Gang Zeng
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2610.08579

 - **Pdf link:** https://arxiv.org/pdf/2610.08579

 - **Abstract**
 Speech understanding in noise remains challenging for hearing-aid users, particularly in the presence of competing speakers. Conventional hearing aids typically perform speech enhancement (SE) and hearing-loss compensation in separate stages, which may cause enhancement errors and signal distortions to carry over to the amplification stage. Moreover, audio-only SE often provides limited benefits under competing-speech conditions because the target and interfering speech share similar acoustic characteristics, making them difficult to separate. To address these limitations, we propose AV-NeuroAMP, an end-to-end audiovisual framework that integrates noisy speech, target-talker video, and the listener's audiogram to jointly perform SE, personalized amplification, and dynamic-range compression. We further introduce audiogram-conditioned feature-wise linear modulation (AC-FiLM) to effectively incorporate listener-specific hearing profiles. Objective evaluations showed that AV-NeuroAMP outperformed conventional amplification, the audio-only NeuroAMP model, and two-stage systems on an in-domain English test set, and that these improvements were retained on an unseen Mandarin test set. Listening tests involving normal-hearing participants under simulated hearing loss and listeners with hearing loss further demonstrated improvements in speech quality and intelligibility, with the greatest benefits observed under competing-speech conditions. These findings support end-to-end audiovisual personalized amplification as a promising approach for improving hearing-aid performance in challenging acoustic environments.
#### Neural Representations, Natural Connections: What Transfers From Human Speech Foundation Models to Animal Vocalizations?
 - **Authors:** Tomás Arias-Vergara, Christopher Hauer, Héloïse Brotier, Elmar Nöth, Andreas Maier, Lee Koren
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2610.07107

 - **Pdf link:** https://arxiv.org/pdf/2610.07107

 - **Abstract**
 Speech-pretrained models have shown promise in animal bioacoustics, but the factors governing their cross-species transfer remain poorly understood. We evaluate 15 frozen encoders spanning monolingual and multilingual speech, speaker verification, animal bioacoustics, and general audio on caller identification across four species and call-type classification across three. Differences between the strongest speech- and animal-pretrained representations range from $-0.027$ to $+0.119$ UAR; speech is significantly better in three of seven settings ($p<0.01$) and not significantly different in the remaining four. Controlled comparisons show no systematic advantage from increased language coverage or animal-domain pretraining, while frozen speaker-verification embeddings transfer poorly. The results also show that transfer is strongly layer-dependent: raw-waveform models peak early, whereas patch-spectrogram models peak deeper.
#### AccentCL: Robust Accent Classification with Incremental Expansion
 - **Authors:** Mu-Ruei Tseng, Waris Quamer, Ghady Nasrallah, Ricardo Gutierrez-Osuna
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Machine Learning (cs.LG); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2610.07426

 - **Pdf link:** https://arxiv.org/pdf/2610.07426

 - **Abstract**
 Accent classifiers are typically trained with a fixed label inventory and cannot accommodate new accent categories as new data becomes available. Moreover, accented speech corpora often exhibit substantial class imbalance and/or domain shift due to differences in recording conditions across corpora. We present AccentCL, a class-incremental learning framework for English accent classification that is robust to class imbalance and cross-corpus domain shift. AccentCL extracts multi-layer representations from a frozen Whisper-Large-v3 encoder, optimized with an imbalance-aware cross-entropy loss to reduce bias toward the majority accent classes and a domain mean alignment loss that minimizes distributional mean shift across training corpora. The label space is then expanded via replay-based continual learning, using the frozen base model for knowledge retention and an old-to-new margin loss to reduce overprediction on newly added classes. On a five-class accent classification task, AccentCL achieves 77.1% balanced accuracy and a 76.9% macro-averaged F1 score. We further evaluate the model's ability to incrementally incorporate two new accent categories: Spanish-accented and Chinese-accented English. When adding Spanish-accented English to the pretrained model, AccentCL attains an F1 of 83.3% on the new class while retaining 77.3% balanced accuracy on the base classes. When subsequently adding Chinese-accented English, it achieves 61.8% F1 on the new class while preserving 77.6% balanced accuracy on the previously learned classes. These results show that AccentCL enables robust regional accent classification while allowing new accent categories to be added without full retraining.
#### Pronunciation-Oriented Reinforcement Learning for Japanese Text-to-Speech with Kana-Domain ASR Rewards
 - **Authors:** Shiao Zhu, Lianbo Liu, Kai Washizaki, Koki Nikaido, Yui Sudo
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2610.07575

 - **Pdf link:** https://arxiv.org/pdf/2610.07575

 - **Abstract**
 Character error rate (CER) computed by automatic speech recognition (ASR) is widely used as an intelligibility reward for reinforcement learning (RL) post-training of text-to-speech (TTS) systems. For Japanese, however, orthographic CER introduces a representation mismatch for pronunciation-oriented optimization: distinct kanji readings may collapse to the same orthographic representation, while equivalent pronunciations may admit different orthographic forms. We instead compute CER in the kana domain using a kana-transcribing ASR model and reference readings (Kana-CER). Under matched group relative policy optimization (GRPO) conditions, Kana-CER reduces target-kanji reading error by approximately 26% relative to the orthographic CER reward, while maintaining comparable orthographic CER and similar speaker similarity and objective speech quality. It also reaches its best validation performance in substantially fewer optimization steps (4k vs. 18k). We further observe severe output elongation under unregularized Kana-CER optimization, which is substantially suppressed by KL regularization.


by Zyzzyva0381 (Windy). 


2026-10-07
