# Showing new listings for Wednesday, 4 March 2026
Auto update papers at about 2:30am UTC (10:30am Beijing time) every weekday.


阅读 `Usage.md`了解如何使用此repo实现个性化的Arxiv论文推送

See `Usage.md` for instructions on how to personalize the repo. 


Keyword list: ['text-to-speech', 'text to speech', 'tts', 'LLM-based', 'speech', 'voice']


Excluded: []


### Today: 14papers 
#### Quality of Automatic Speech Recognition -- Polish Language case study -- from Wav2Vec to Scribe ElevenLabs
 - **Authors:** Marcin Pietroń, Szymon Piórkowski, Kamil Faber, Dominik Żurek, Michał Karwatowski, Jerzy Duda, Hubert Zieliński, Piotr Lipnicki, Mikołaj Leszczuk
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2603.02246

 - **Pdf link:** https://arxiv.org/pdf/2603.02246

 - **Abstract**
 This article concerns comparative studies on the Automatic Speech Recognition (ASR) model incorporated with the Large Language Model (LLM) used for medical interviews. The proposed solution is tested on polish language benchmarks and dataset with medical interviews. The latest ASR technologies are based on convolutional neural networks (CNNs), recurrent neural networks (RNNs) and Transformers. Most of them work as end-to-end solutions. The presented approach in the case of the Whisper model shows a two-stage solution with End-To-End ASR and LLM working together in a pipeline. The ASR output is an input for LLM. The LLM is a component by which the output from ASR is corrected and improved. Comparative studies for automatic recognition of the Polish language between modern End-To-End deep learning architectures and the ASR hybrid model were performed. The medical interview tests were performed with two state-of-the-art ASR models: OpenAI Whisper incorporated with LLM and Scribe ElevenLabs. Additionally, the results were compared with five more end-to-end models (QuartzNet, FastConformer, Wav2Vec 2.0 XLSR and ESPnet Model Zoo) on Mozilla Common Voice and VoxPopuli databases. Tests were conducted for clean audio signal, signal with bandwidth limitation, and degraded. The tested models were evaluated on the basis of Word Error Rate (WER) and Character Error Rate (CER). The results show that the Whisper model performs by far the best among the open-source models. ElevenLabs Scribe model, on the other hand, performs best for Polish on both general benchmark and medical data.
#### Whisper-RIR-Mega: A Paired Clean-Reverberant Speech Benchmark for ASR Robustness to Room Acoustics
 - **Authors:** Mandip Goswami
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Artificial Intelligence (cs.AI); Machine Learning (cs.LG); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2603.02252

 - **Pdf link:** https://arxiv.org/pdf/2603.02252

 - **Abstract**
 We introduce Whisper-RIR-Mega, a benchmark dataset of paired clean and reverberant speech for evaluating automatic speech recognition (ASR) robustness to room acoustics. Each sample pairs a clean LibriSpeech utterance with the same utterance convolved with a real room impulse response from the RIR-Mega corpus, with stratified splits by reverberation time (RT60) and direct-to-reverberant ratio (DRR). We evaluate five Whisper models (tiny through large-v3) on 1600 test samples and report word error rate (WER) and character error rate (CER) under clean and reverberant conditions. Reverberation consistently degrades performance across all model sizes; the reverb penalty in WER ranges from 0.12 to 1.07 percentage points depending on the model. We release the dataset, evaluation code, and baseline results to support reproducible research on robust ASR.
#### Benchmarking Speech Systems for Frontline Health Conversations: The DISPLACE-M Challenge
 - **Authors:** Dhanya E, Ankita Meena, Manas Nanivadekar, Noumida A, Victor Azad, Ashwini Nagaraj Shenoy, Pratik Roy Chowdhuri, Shobhit Banga, Vanshika Chhabra, Chitralekha Bhat, Shareef babu Kalluri, Srikanth Raj Chetupalli, Deepu Vijayasenan, Sriram Ganapathy
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2603.02813

 - **Pdf link:** https://arxiv.org/pdf/2603.02813

 - **Abstract**
 The DIarization and Speech Processing for LAnguage understanding in Conversational Environments - Medical (DISPLACE-M) challenge introduces a conversational AI benchmark focused on understanding goal-oriented, real-world medical dialogues collected in the field. The challenge addresses multi-speaker interactions between healthcare workers and seekers characterized by spontaneous, noisy and overlapping speech across Indian languages and dialects. As part of the challenge, medical conversational dataset comprising 25 hours of development data and 10 hours of blind evaluation recordings was released. We provided baseline systems within a unified end-to-end pipeline across 4 tasks - speaker diarization, automatic speech recognition, topic identification and dialogue summarization - to enable consistent benchmarking. System performance is evaluated using established metrics such as diarization error rate (DER), time-constrained minimum-permutation word error rate (tcpWER), and ROUGE-L. During this evaluation (Phase-I), 12 teams, across the globe, actively participated pushing the baseline systems on these metrics. However, even with a 6-8 week dedicated effort from various participants, the task is shown to be substantially challenging, and the existing systems are significantly short of healthcare deployment readiness.
#### DBMIF: a deep balanced multimodal iterative fusion framework for air- and bone-conduction speech enhancement
 - **Authors:** Yilei Wu, Changyan Zheng, Xingyu Zhang, Yakun Zhang, Chengshi Zheng, Shuang Yang, Ye Yan, Erwei Yin
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2603.02877

 - **Pdf link:** https://arxiv.org/pdf/2603.02877

 - **Abstract**
 The performance of conventional speech enhancement systems degrades sharply in extremely low signal-to-noise ratio (SNR) environments where air-conduction (AC) microphones are overwhelmed by ambient noise. Although bone-conduction (BC) sensors offer complementary, noise-tolerant information, existing fusion approaches struggle to maintain consistent performance across a wide range of SNR conditions. To address this limitation, we propose the Deep Balanced Multimodal Iterative Fusion Framework (DBMIF), a three-branch architecture designed to reconstruct high-fidelity speech through rigorous cross-modal interaction. Specifically, grounded in a multi-scale interactive encoder-decoder backbone, the framework orchestrates an iterative attention module and a cross-branch gated module to facilitate adaptive weighting and bidirectional exchange. To complement this dynamic interaction, a balanced-interaction bottleneck is further integrated to learn a compact, stable fused representation. Extensive experiments demonstrate that DBMIF achieves competitive performance compared with recent unimodal and multimodal baselines in both speech quality and intelligibility across diverse noise types. In downstream ASR tasks, the proposed method reduces the character error rate by at least 2.5 percent compared to competing approaches. These results confirm that DBMIF effectively harnesses the robustness of BC speech while preserving the naturalness of AC speech, ensuring reliability in real-world scenarios. The source code is publicly available at this http URL.
#### Does Fine-tuning by Reinforcement Learning Improve Generalization in Binary Speech Deepfake Detection?
 - **Authors:** Xin Wang, Ge Wanying, Junichi Yamagishi
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2603.02914

 - **Pdf link:** https://arxiv.org/pdf/2603.02914

 - **Abstract**
 Building speech deepfake detection models that are generalizable to unseen attacks remains a challenging problem. Although the field has shifted toward a pre-training and fine-tuning paradigm using speech foundation models, most approaches rely solely on supervised fine-tuning (SFT). Inspired by the field of large language models, wherein reinforcement learning (RL) is used for model fine-tuning, we investigate the impact of RL, specifically Group Relative Policy Optimization (GRPO). The results from experiments using multiple detectors and test sets indicate that pure GRPO-based fine-tuning improves performance on out-of-domain test sets while maintaining performance on target-domain test data. This approach outperforms both SFT-only and hybrid setups. Our ablation studies further suggest that the negative reward in GRPO may be a key factor in this improvement.
#### Bias and Fairness in Self-Supervised Acoustic Representations for Cognitive Impairment Detection
 - **Authors:** Kashaf Gulzar, Korbinian Riedhammer, Elmar Nöth, Andreas K. Maier, Paula Andrea Pérez-Toro
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Machine Learning (cs.LG)
 - **Arxiv link:** https://arxiv.org/abs/2603.02937

 - **Pdf link:** https://arxiv.org/pdf/2603.02937

 - **Abstract**
 Speech-based detection of cognitive impairment (CI) offers a promising non-invasive approach for early diagnosis, yet performance disparities across demographic and clinical subgroups remain underexplored, raising concerns around fairness and generalizability. This study presents a systematic bias analysis of acoustic-based CI and depression classification using the DementiaBank Pitt Corpus. We compare traditional acoustic features (MFCCs, eGeMAPS) with contextualized speech embeddings from Wav2Vec 2.0 (W2V2), and evaluate classification performance across gender, age, and depression-status subgroups. For CI detection, higher-layer W2V2 embeddings outperform baseline features (UAR up to 80.6\%), but exhibit performance disparities; specifically, females and younger participants demonstrate lower discriminative power (\(AUC\): 0.769 and 0.746, respectively) and substantial specificity disparities (\(\Delta_{spec}\) up to 18\% and 15\%, respectively), leading to a higher risk of misclassifications than their counterparts. These disparities reflect representational biases, defined as systematic differences in model performance across demographic or clinical subgroups. Depression detection within CI subjects yields lower overall performance, with mild improvements from low and mid-level W2V2 layers. Cross-task generalization between CI and depression classification is limited, indicating that each task depends on distinct representations. These findings emphasize the need for fairness-aware model evaluation and subgroup-specific analysis in clinical speech applications, particularly in light of demographic and clinical heterogeneity in real-world applications.
#### Interpreting Speaker Characteristics in the Dimensions of Self-Supervised Speech Features
 - **Authors:** Kyle Janse van Rensburg, Benjamin van Niekerk, Herman Kamper
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Computation and Language (cs.CL)
 - **Arxiv link:** https://arxiv.org/abs/2603.03096

 - **Pdf link:** https://arxiv.org/pdf/2603.03096

 - **Abstract**
 How do speech models trained through self-supervised learning structure their representations? Previous studies have looked at how information is encoded in feature vectors across different layers. But few studies have considered whether speech characteristics are captured within individual dimensions of SSL features. In this paper we specifically look at speaker information using PCA on utterance-averaged representations. Using WavLM, we find that the principal dimension that explains most variance encodes pitch and associated characteristics like gender. Other individual principal dimensions correlate with intensity, noise levels, the second formant, and higher frequency characteristics. Finally, in synthesis experiments we show that most characteristics can be controlled by changing the corresponding dimensions. This provides a simple method to control characteristics of the output voice in synthesis applications.
#### SGPA: Spectrogram-Guided Phonetic Alignment for Feasible Shapley Value Explanations in Multimodal Large Language Models
 - **Authors:** Paweł Pozorski, Jakub Muszyński, Maria Ganzha
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2603.02250

 - **Pdf link:** https://arxiv.org/pdf/2603.02250

 - **Abstract**
 Explaining the behavior of end-to-end audio language models via Shapley value attribution is intractable under native tokenization: a typical utterance yields over $150$ encoder frames, inflating the coalition space by roughly $10^{42}$ relative to text; individual audio frames lack standalone meaning; and token boundaries that bisect phonetic transitions introduce masking artifacts. We introduce Spectrogram-Guided Phonetic Alignment (SGPA), a four-stage pipeline that combines Connectionist Temporal Classification forced alignment with spectral boundary refinement to produce acoustically stable, word-aligned audio segments. Controlled diagnostics on LFM2-Audio-1.5B with VoiceBench show that SGPA yields a 43$\times$ reduction in model evaluations. Statistical testing confirms that SGPA significantly alters attribution concentration while preserving the global cumulative profile, establishing it as a feasibility-enabling layer for audio explainability.
#### MEBM-Phoneme: Multi-scale Enhanced BrainMagic for End-to-End MEG Phoneme Classification
 - **Authors:** Liang Jinghua, Zhang Zifeng, Li Songyi, Zheng Linze
 - **Subjects:** Subjects:
Sound (cs.SD); Artificial Intelligence (cs.AI); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2603.02254

 - **Pdf link:** https://arxiv.org/pdf/2603.02254

 - **Abstract**
 We propose MEBM-Phoneme, a multi-scale enhanced neural decoder for phoneme classification from non-invasive magnetoencephalography (MEG) signals. Built upon the BrainMagic backbone, MEBM-Phoneme integrates a short-term multi-scale convolutional module to augment the native mid-term encoder, with fused representations via depthwise separable convolution for efficient cross-scale integration. A convolutional attention layer dynamically weights temporal dependencies to refine feature aggregation. To address class imbalance and session-specific distributional shifts, we introduce a stacking-based local validation set alongside weighted cross-entropy loss and random temporal augmentation. Comprehensive evaluations on LibriBrain Competition 2025 Track2 demonstrate robust generalization, achieving competitive phoneme decoding accuracy on the validation and official test leaderboard. These results underscore the value of hierarchical temporal modeling and training stabilization for advancing MEG-based speech perception analysis.
#### MEBM-Speech: Multi-scale Enhanced BrainMagic for Robust MEG Speech Detection
 - **Authors:** Li Songyi, Zheng Linze, Liang Jinghua, Zhang Zifeng
 - **Subjects:** Subjects:
Sound (cs.SD); Artificial Intelligence (cs.AI); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2603.02255

 - **Pdf link:** https://arxiv.org/pdf/2603.02255

 - **Abstract**
 We propose MEBM-Speech, a multi-scale enhanced neural decoder for speech activity detection from non-invasive magnetoencephalography (MEG) signals. Built upon the BrainMagic backbone, MEBM-Speech integrates three complementary temporal modeling mechanisms: a multi-scale convolutional module for short-term pattern extraction, a bidirectional LSTM (BiLSTM) for long-range context modeling, and a depthwise separable convolutional layer for efficient cross-scale feature fusion. A lightweight temporal jittering strategy and average pooling further improve onset robustness and boundary stability. The model performs continuous probabilistic decoding of MEG signals, enabling fine-grained detection of speech versus silence states - an ability crucial for both cognitive neuroscience and clinical applications. Comprehensive evaluations on the LibriBrain Competition 2025 Track1 benchmark demonstrate strong performance, achieving an average F1 macro of 89.3% on the validation set and comparable results on the official test leaderboard. These findings highlight the effectiveness of multi-scale temporal representation learning for robust MEG-based speech decoding.
#### Sequence-Level Unsupervised Training in Speech Recognition: A Theoretical Study
 - **Authors:** Zijian Yang, Jörg Barkoczi, Ralf Schlüter, Hermann Ney
 - **Subjects:** Subjects:
Sound (cs.SD); Machine Learning (cs.LG); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2603.02285

 - **Pdf link:** https://arxiv.org/pdf/2603.02285

 - **Abstract**
 Unsupervised speech recognition is a task of training a speech recognition model with unpaired data. To determine when and how unsupervised speech recognition can succeed, and how classification error relates to candidate training objectives, we develop a theoretical framework for unsupervised speech recognition grounded in classification error bounds. We introduce two conditions under which unsupervised speech recognition is possible. The necessity of these conditions are also discussed. Under these conditions, we derive a classification error bound for unsupervised speech recognition and validate this bound in simulations. Motivated by this bound, we propose a single-stage sequence-level cross-entropy loss for unsupervised speech recognition.
#### When Spoof Detectors Travel: Evaluation Across 66 Languages in the Low-Resource Language Spoofing Corpus
 - **Authors:** Kirill Borodin, Vasiliy Kudryavtsev, Maxim Maslov, Mikhail Gorodnichev, Grach Mkrtchian
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2603.02364

 - **Pdf link:** https://arxiv.org/pdf/2603.02364

 - **Abstract**
 We introduce LRLspoof, a large-scale multilingual synthetic-speech corpus for cross-lingual spoof detection, comprising 2,732 hours of audio generated with 24 open-source TTS systems across 66 languages, including 45 low-resource languages under our operational definition. To evaluate robustness without requiring target-domain bonafide speech, we benchmark 11 publicly available countermeasures using threshold transfer: for each model we calibrate an EER operating point on pooled external benchmarks and apply the resulting threshold, reporting spoof rejection rate (SRR). Results show model-dependent cross-lingual disparity, with spoof rejection varying markedly across languages even under controlled conditions, highlighting language as an independent source of domain shift in spoof detection. The dataset is publicly available at \href{this https URL}{\textbf{\underline{\textit{HuggingFace}}}} and \href{this https URL}{\textbf{\underline{\textit{ModelScope}}}}
#### Differentiable Time-Varying IIR Filtering for Real-Time Speech Denoising
 - **Authors:** Riccardo Rota, Kiril Ratmanski, Jozef Coldenhoff, Milos Cernak
 - **Subjects:** Subjects:
Sound (cs.SD); Artificial Intelligence (cs.AI); Machine Learning (cs.LG); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2603.02794

 - **Pdf link:** https://arxiv.org/pdf/2603.02794

 - **Abstract**
 We present TVF (Time-Varying Filtering), a low-latency speech enhancement model with 1 million parameters. Combining the interpretability of Digital Signal Processing (DSP) with the adaptability of deep learning, TVF bridges the gap between traditional filtering and modern neural speech modeling. The model utilizes a lightweight neural network backbone to predict the coefficients of a differentiable 35-band IIR filter cascade in real time, allowing it to adapt dynamically to non-stationary noise. Unlike ``black-box'' deep learning approaches, TVF offers a completely interpretable processing chain, where spectral modifications are explicit and adjustable. We demonstrate the efficacy of this approach on a speech denoising task using the Valentini-Botinhao dataset and compare the results to a static DDSP approach and a fully deep-learning-based solution, showing that TVF achieves effective adaptation to changing noise conditions.
#### DLIOS: An LLM-Augmented Real-Time Multi-Modal Interactive Enhancement Overlay System for Douyin Live Streaming
 - **Authors:** Shuide Wen, Sungil Seok, Beier Ku, Richee Li, Yubin He, Bowen Qu, Yang Yang, Ping Su, Can Jiao
 - **Subjects:** Subjects:
Image and Video Processing (eess.IV); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2603.03060

 - **Pdf link:** https://arxiv.org/pdf/2603.03060

 - **Abstract**
 We present DLIOS, a Large Language Model (LLM)-augmented real-time multi-modal interactive enhancement overlay system for Douyin (TikTok) live streaming. DLIOS employs a three-layer transparent window architecture for independent rendering of danmaku (scrolling text), gift and like particle effects, and VIP entrance animations, built around an event-driven WebView2 capture pipeline and a thread-safe event bus. On top of this foundation we contribute an LLM broadcast automation framework comprising: (1) a per-song four-segment prompt scheduling system (T1 opening/transition, T2 empathy, T3 era story/production notes, T4 closing) that generates emotionally coherent radio-style commentary from lyric metadata; (2) a JSON-serializable RadioPersonaConfig schema supporting hot-swap multi-persona broadcasting; (3) a real-time danmaku quick-reaction engine with keyword routing to static urgent speech or LLM-generated empathetic responses; and (4) the Suwan Li AI singer-songwriter persona case study -- over 100 AI-generated songs produced with Suno. A 36-hour stress test demonstrates: zero danmaku overlap, zero deadlock crashes, gift effect P95 latency <= 180 ms, LLM-to-TTS segment P95 latency <= 2.1 s, and TTS integrated loudness gain of 9.5 LUFS. live streaming; danmaku; large language model; prompt engineering; virtual persona; WebView2; WINMM; TTS; Suno; loudness normalization; real-time scheduling


by Zyzzyva0381 (Windy). 


2026-03-04
