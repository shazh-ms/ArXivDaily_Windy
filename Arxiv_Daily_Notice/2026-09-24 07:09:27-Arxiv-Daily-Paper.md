# Showing new listings for Thursday, 24 September 2026
Auto update papers at about 2:30am UTC (10:30am Beijing time) every weekday.


阅读 `Usage.md`了解如何使用此repo实现个性化的Arxiv论文推送

See `Usage.md` for instructions on how to personalize the repo. 


Keyword list: ['text-to-speech', 'text to speech', 'tts', 'LLM-based', 'speech', 'voice']


Excluded: []


### Today: 8papers 
#### A Temporal-Envelope Frontend with Learnable Per-Channel Energy Normalization for Whisper-Based Children's ASR
 - **Authors:** Edem Ahadzi, Ruchi Pandey, Tomi H. Kinnunen
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2609.26937

 - **Pdf link:** https://arxiv.org/pdf/2609.26937

 - **Abstract**
 Temporal envelopes carry cues critical to speech intelligibility, yet ASR frontends based on log-mel spectrograms do not explicitly model continuous sub-band envelope structure. This limitation is particularly acute for children's speech, where high acoustic variability demands robust feature representations. We propose a modular time-domain frontend that decomposes speech into sub-band envelopes using mel-spaced windowed-sinc filters and the Hilbert transform, with learnable per-channel energy normalization (PCEN) jointly optimized with the Whisper model. On the MyST children's speech corpus, systematic ablations identify full-band windowed-sinc filters, Hilbert envelopes, a 25 Hz smoothing cutoff, and learnable PCEN as the best configuration. Under the same Whisper-small fine-tuning setup, the frontend reduces WER from 13.16% to 11.08%, a 15.8% relative reduction over the log-mel baseline, and outperforms the evaluated Kid-Whisper checkpoint on the same cleaned test split. These results show that temporal-envelope representations and learnable frontend normalization are effective complements to backend adaptation for children's ASR.
#### One-Step Voice Conversion by Learning kNN Transport in WavLM Space
 - **Authors:** Anton Selitskiy, David Millard
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2609.27230

 - **Pdf link:** https://arxiv.org/pdf/2609.27230

 - **Abstract**
 Voice conversion (VC) systems fall into two families: non-parametric embedding-space methods, which need no trained model but degrade on short target utterances, and spectrogram-based neural architectures, which achieve strong quality via multi-module pipelines with tens of millions of parameters. We propose kNN-FM-VC, a single conditional flow-matching network that learns to approximate the kNN-VC mapping between WavLM embedding distributions of source and target speakers, replacing explicit pointwise kNN matching with a neural regressor trained on kNN-generated pairs. The model is conditioned on the target speaker via cross-attention and FiLM, and trained under three Gaussian conditional paths (Schrödinger bridge, straight line, and constant-variance Gaussian tube), enabling few-step sampling. Unlike Phoneme Hallucinator, which uses an upsampling stage followed by kNN matching, our 13M-parameter model performs conversion with a single learned network and supports one-step inference. On LibriSpeech, the one-step Gaussian Bridge achieves lower WER and higher estimated speech quality than FreeVC and Phoneme Hallucinator. Relative to kNN and kDOT, it substantially reduces WER.
#### JASPER: Joint Audio and Speech Pre-trained Encoder Representations
 - **Authors:** Geeth George, Ameenudeen P E, Hrishikesh H Pillai, Sriram Ganapathy
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2609.27260

 - **Pdf link:** https://arxiv.org/pdf/2609.27260

 - **Abstract**
 Self-supervised learning (SSL) for speech and audio has largely progressed along separate tracks: speech models emphasise time-domain prediction, whereas audio representation learning has focused on time-frequency patterns. This separation creates a compatibility gap, limiting cross-domain generalization. In this work, we introduce JASPER, Joint Audio and Speech Pre-trained Encoder Representations, a framework that augments speech-pretrained models with time-frequency objectives. Specifically, JASPER performs masked prediction of temporal and spectral targets over long audio segments, enabling spectro-temporal representation learning of speech and audio signals. The proposed method consistently outperforms multiple baselines and existing speech/audio encoders on diverse speech, audio, and music tasks, demonstrating the effectiveness of unified spectro-temporal modeling.
#### Beyond DER: Speaker Counting in Crowded End-to-End Diarization
 - **Authors:** Lahiru Samarakoon, Hongxu Zhu
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2609.27315

 - **Pdf link:** https://arxiv.org/pdf/2609.27315

 - **Abstract**
 Speaker diarization must solve two problems: counting how many speakers are present in a given conversation, and assigning speech to each one. This becomes harder in crowded conversations with five or more speakers. We evaluate how end-to-end neural diarization models count speakers, and find systematic under-counting in crowded recordings. The standard diarization error rate (DER) hides this failure, because it is duration-weighted and barely penalizes the dropped, low-activity speakers. We therefore also report the Jaccard error rate (JER) and explicit counting metrics. We propose a gated loss that couples speaker existence with frame activity. This loss can be computed in two ways, duration-weighted or speaker-weighted, and the speaker-weighted variant, used as a regularizer, reduces the under-count. On real-world crowded recordings our method clearly improves the counting metrics, lowering JER by about 6% relative and DER by about 9% relative, while leaving sparse recordings unharmed.
#### The Second MLC-SLM Challenge: Multilingual Conversational Speech Diarization, Recognition, and Understanding
 - **Authors:** Bingshen Mu, Mingchen Shao, Zhennan Lin, Liumeng Xue, Hexin Liu, Lei Xie, Eng Siong Chng, Longshuai Xiao, Qiangze Feng, Daliang Wang
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2609.27514

 - **Pdf link:** https://arxiv.org/pdf/2609.27514

 - **Abstract**
 This paper summarizes the Interspeech2026 second Multilingual Conversational Speech Language Model (MLC-SLM) Challenge, which aims to advance the development of effective multilingual conversational speech language models. We describe the two challenge tasks: multilingual conversational speech diarization and recognition, and multilingual conversational speech understanding, together with the released real-world conversational speech dataset, evaluation protocols, and baseline systems. The challenge attracted 91 teams worldwide, with 704 valid leaderboard results and 14 technical reports across the two tasks. Based on the participating systems, we summarize representative approaches and distill practical insights into multilingual conversational speech recognition and understanding to support future research in the community.
#### EmphTTS: an emphasis-control TTS with reinforcement learning
 - **Authors:** Zirui Li, Rech Silas, Lauri Juvela, Tom Backstrom, Mikko Kurimo
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2609.27599

 - **Pdf link:** https://arxiv.org/pdf/2609.27599

 - **Abstract**
 Generating controllable and human-like emphasis remains an open challenge in text-to-speech, even when explicit emphasis control signals are provided in the text input, limiting the communicative accuracy of synthetic speech in real-world applications. Reinforcement learning has recently shown promise for post-training TTS systems to align with human preference, yet existing methods have not been applied to word-level prosodic control. We present EmphTTS, a non-autoregressive TTS system that applies Group Relative Policy Optimization (GRPO) to the duration predictor with an emphasis localization reward, enabling direct optimization for word-level emphasis. Evaluations show that EmphTTS achieves the best emphasis controllability and performs the best in emphasis objective evaluation. In subjective preference tests, EmphTTS is significantly preferred over synthetic groundtruth and most baselines. Ablation studies show that GRPO improves emphasis realization beyond supervised-finetuning-based duration modeling and simple speaking-rate adjustment, while alleviating the mismatch between the independently trained duration predictor and TTS model.
#### Subjective Evaluation of DNN AND Auditory-Model-Based Hearing-Loss Compensation
 - **Authors:** Chuan Wen, Brent Nissens, Nele De Poortere, Morgan Thienpont, Matthias Inghels, Attila Fráter, Guy Torfs, Sarah Verhulst
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2609.28033

 - **Pdf link:** https://arxiv.org/pdf/2609.28033

 - **Abstract**
 Outer-hair-cell (OHC) loss is a primary deficit of sensorineural hearing loss (SNHL), impairing cochlear amplification and frequency selectivity and thereby elevating hearing thresholds. Biophysically-inspired DNN-based hearing-aid (HA) algorithms have been proposed to compensate for OHC deficits and have shown clear benefits in objective speech intelligibility and quality metrics (e.g. HASPI, HASQI). However, comprehensive subjective validation of these benefits in human listeners is still missing. In this work, we present a subjective evaluation of a biophysically-inspired HA model targeting OHC deficits. The cochlear module of an auditory model was individualized based on each listener's pure-tone audiogram and integrated into a trainable system, which includes the personalized model and a normal-hearing reference model, and the resulting trained HA was evaluated using a Matrix test comparing intelligibility scores for unprocessed and HA-processed noisy speech. The results revealed a significant benefit of the HA model over the unprocessed condition in the range of +1 to +27%, providing behavioral confirmation of the efficacy of the HA model. This study closes the gap between objective and perceptual evidence for this new generation of DNN-based HA algorithms, paving the way for their integration into next-generation DNN-accelerated chips for hearables and hearing aids.
#### Text Scores Can Miss Waveform Use: A Qwen2-Audio Quantization Case Study
 - **Authors:** Mengzhe Geng, Jinxi Jin, Junhao Xu
 - **Subjects:** Subjects:
Sound (cs.SD); Computation and Language (cs.CL); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2609.26823

 - **Pdf link:** https://arxiv.org/pdf/2609.26823

 - **Abstract**
 Post-training quantization of speech language models is often summarized with text-output scores and nominal bit widths. Those numbers alone do not establish behavior that depends on information missing from a transcript, or efficiency for a particular runtime. We introduce an evaluation protocol that separately tests lexical output, a transcript-insufficient endpoint, and a measured packed implementation. In a Qwen2-Audio case study, a translation-selected 6-bit allocation improves chrF by 2.36 on a frozen English-to-German replay, with paired 95% bootstrap interval [1.04, 3.62], but loses 3.91 percentage points on speaker-disjoint emotion recognition. At the same 6-bit budget, the uniform structural control reaches higher emotion accuracy than the selected allocation, and the front-layer control is also higher by point estimate on the same frozen set. At 7 bits, chrF improves by 3.28 with interval [2.08, 4.59], the emotion interval against FP16 includes zero, and a same-budget front-layer control still exceeds the selected allocation. A separate matched-budget 4.08-bit study finds roughly 10-point emotion deficits for every tested low-bit allocation and no selected-allocation advantage over frozen controls. Finally, a dequantized average-6-bit simulation retains the FP16 peak memory. This case study identifies a precision-dependent mismatch between lexical output, waveform-dependent behavior, and nominal precision. It does not establish a general failure of low-bit speech models or a deployment benefit for the selected allocation.


by Zyzzyva0381 (Windy). 


2026-09-24
