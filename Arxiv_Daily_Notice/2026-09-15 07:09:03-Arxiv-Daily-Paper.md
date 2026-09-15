# Showing new listings for Tuesday, 15 September 2026
Auto update papers at about 2:30am UTC (10:30am Beijing time) every weekday.


阅读 `Usage.md`了解如何使用此repo实现个性化的Arxiv论文推送

See `Usage.md` for instructions on how to personalize the repo. 


Keyword list: ['text-to-speech', 'text to speech', 'tts', 'LLM-based', 'speech', 'voice']


Excluded: []


### Today: 27papers 
#### Building a Production Greek-English Speech Recognizer
 - **Authors:** Christos Petrocheilos, Cleopatra Papadopoulou, Chris Porikis, Ioakeim Perros, Ayoub Kirouane, Themistoklis Nikolis
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Artificial Intelligence (cs.AI); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2609.13498

 - **Pdf link:** https://arxiv.org/pdf/2609.13498

 - **Abstract**
 We report a multi-month engineering program to build Sophea, a production bilingual Greek-English automatic speech recognition system. We evaluate the system against nine production gates covering Greek and English word error rate, language identification, and hallucinations on non-speech audio. Across twenty-three training iterations and two model architectures, no training-data composition passed all nine gates simultaneously. Meeting the Greek noisy-environment target required about 1,500 steps of dense domain exposure, while preserving English language identification tolerated only about 250 steps, or about 1,250 with a rebalanced mix that reduced Greek accuracy. We describe a six-stage data pipeline in which calibrating an audio-quality filter against in-domain anchors reduced the discarded share of scored Greek audio from 98.7 percent to 10.6 percent. A pre-registered ablation isolated a hallucination defect to one training-data package. A three-model ROVER ensemble increased gate coverage from 4-7 of 9 for individual models to 9 of 9 and reduced overlapping-speech WER from 53.35 percent to 37.87 percent, a 29 percent relative improvement. A separate learned per-clip arbiter over two models is listed as sophea/asr-k1 (preview) on the public Open ASR Leaderboard, with 4.26 percent average WER across eight public English test sets, and reaches 25.88 percent WER on live Greek noisy-environment traffic. We also document five cases in which a measurement tool produced a plausible but incorrect result and seven substantial approaches that were evaluated but not shipped. No model weights or training data are released; we report methodology and quantitative results only.
#### Subphonetic Acoustic Modeling via Optimal Transport for Pronunciation Assessment
 - **Authors:** Haopeng Geng, Jiun-Ting Li, Daisuke Saito, Nobuaki Minematsu
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2609.13694

 - **Pdf link:** https://arxiv.org/pdf/2609.13694

 - **Abstract**
 Pronunciation assessment requires acoustic evidence that is temporally precise, diagnostically meaningful, and faithful to the learner's actual production. However, existing acoustic models often struggle to provide recognition and segmentation evidence simultaneously. CTC-based phone recognizers can predict phone sequences flexibly, but their sparse and peaky posteriors often miss phone boundaries and fine-grained pronunciation cues. In contrast, text-dependent forced aligners provide reliable temporal information when transcripts are available, but are not directly applicable to reference-free pronunciation analysis. In this work, we propose a topology-aware frame-wise acoustic model that learns dense ordered state posteriors within each phone. The key idea is to recover phone-internal state structure in a neural acoustic model by combining ordered subphonetic states with optimal temporal transport classification (OTTC). This combination encourages dense monotonic frame-level state discrimination while preserving phone recognition ability. Experiments on read, spontaneous, and L2 speech show improved segmentation over neural baselines with competitive recognition performance. Downstream evaluations further show gains in mispronunciation detection and automatic pronunciation assessment. Probing analysis suggests that the learned states capture phoneme-dependent acoustic structure rather than arbitrary frame-level distributions.
#### DualSpecSE: A Dual-Path Speech Enhancement Network Integrating Mel and Complex Spectrograms
 - **Authors:** Xingchen Li, Ziqian Wang, Zikai Liu, Yike Zhu, Zihan Zhang, Longshuai Xiao, Lei Xie
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2609.13911

 - **Pdf link:** https://arxiv.org/pdf/2609.13911

 - **Abstract**
 In this paper, we propose DualSpecSE, a speech enhancement framework that jointly models Mel-spectrogram and complex spectrogram in a dual-path architecture for improved ASR performance and higher-quality speech reconstruction. The Mel branch learns coarse-grained acoustic representations and produces enhanced Mel-spectrograms for direct ASR usage, while the complex branch refines fine-grained spectral details for high-fidelity waveform reconstruction. Built upon the cross-band and narrow-band blocks from CleanMel, DualSpecSE introduces an interaction module and a fusion module to enable effective information exchange between the two branches. The model simultaneously outputs enhanced Mel and complex spectrogram without requiring a pretrained vocoder. Experimental results demonstrate consistent improvements in speech fidelity, perceptual quality, and ASR performance. Codes and audio samples are available.
#### Modeling, Scaling, and Decoding: Optimizing Controllable Speech Generation with Nonverbal Vocalizations
 - **Authors:** Ziyu Zhang, Yun Chen, Taihui Wang, Hanzhao Li, Qicong Xie, Rilin Chen, Zhixian Zhao, Lei Xie
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Artificial Intelligence (cs.AI)
 - **Arxiv link:** https://arxiv.org/abs/2609.14231

 - **Pdf link:** https://arxiv.org/pdf/2609.14231

 - **Abstract**
 Controllable synthesis of nonverbal vocalizations (NVVs) is es- sential for natural and expressive speech, but remains challeng- ing due to their acoustic diversity and imbalanced distribution in existing corpora. To address these challenges, we develop an NVV-aware DiTAR system that models continuous speech latents, encodes the 16 target NVV categories as dedicated to- kens, and adapts stop prediction to distinguish mid-utterance vocalizations from utterance boundaries. Training begins with large-scale bilingual pre-training on diverse NVV speech, fol- lowed by continued supervised fine-tuning on a corpus en- hanced through targeted synthetic augmentation and frequency- aware rebalancing. At inference time, we select the acoustic prompt, tune the LM-guidance and noise-injection scales, and apply Best-of-N sampling with multi-metric selection to re- duce generation failures. The final system achieves an official weighted bilingual score of 62.786, ranking first in Mandarin, second in English, and first overall among participating systems in Track 2 of the ISCSLP 2026 NVVSpeech Challenge. Ab- lation studies show that targeted augmentation benefits under- represented NVV categories the most, while robust candidate selection requires balancing NVV correctness, lexical fidelity, and perceptual quality.
#### Exploring Multimodal Turn-Taking Cues in Face-to-Face Conversation using Voice Activity Projection
 - **Authors:** Willem Berner, Julio Cesar Cavalcanti, Kalle Åström, Gabriel Skantze
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Computation and Language (cs.CL); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2609.14666

 - **Pdf link:** https://arxiv.org/pdf/2609.14666

 - **Abstract**
 Turn-taking is a fundamental component of spoken interaction, and while humans naturally rely on both verbal and non-verbal signals, dialogue systems usually depend on audio cues alone. This paper investigates whether visual features from face-to-face conversations can enhance turn-taking prediction beyond what is achievable from audio-only. We extend the Voice Activity Projection (VAP) model, a self-supervised transformer-based model for predicting future voice activity, by incorporating visual features extracted from the large-scale Meta Seamless Interaction dataset of dyadic face-to-face conversations. The visual features include gaze direction, head movement, body and hand pose, and facial action units (FAU). For incorporating the visual features, we explore concatenation, cross-attention fusion, delta features, and trainable gating mechanisms. Results show that visual information improves performance over the audio-only baseline, with FAU being significantly more informative than other feature groups. Body and gaze features nevertheless contribute complementary information, as the model combining all features performs best. Furthermore, results indicate that performance on specific tasks varies depending on whether training and test data come from improvised (acted) or naturalistic (non-acted) conversations.
#### Bridging Data, Reasoning, and Alignment: A Unified Framework for Context-Aware Instruction-Following TTS
 - **Authors:** Jingbin Hu, Luyu Wang, Wenjie Tian, Kangxiang Xia, Qirui Zhan, Haoyu Zhang, Yunxiang Chen, Houdun Liu, Lei Xie, Liumeng Xue
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2609.14740

 - **Pdf link:** https://arxiv.org/pdf/2609.14740

 - **Abstract**
 The ISCSLP 2026 CoT-TTS Challenge requires TTS systems to generate Chain-of-Thought (CoT) reasoning from dialogue history before synthesizing contextually appropriate speech. While the official baseline establishes a unified architecture, it remains constrained by limited contextual comprehension, weak instruction fidelity, and suboptimal audio quality. We present a systematic optimization pipeline to address these limitations. First, we develop a data process framework that cleans raw data via FullSubNet denoising, Qwen3-ASR re-transcription, and Qwen3.5-35B-A3B-based history-CoT consistency analysis, while distilling 545K high-fidelity instruction samples using Qwen3-TTS and Seed-VC under strict quality filtration. Second, we propose a Context-Aware Direct Preference Optimization (CA-DPO) method. By employing a cascaded filtering strategy, ASR prescreening, LLM tournament ranking, and speaker similarity verification, we obtain high-confidence preference pairs that significantly enhance holistic ``Context$\rightarrow$CoT$\rightarrow$Speech'' consistency during DPO training. Third, we establish an evaluation method featuring a 500-sample test set and an LLM-as-Judge framework to independently assess reasoning and execution fidelity. Experiments demonstrate that our system significantly outperforms the baseline across all objective and subjective metrics, validating our data governance and alignment strategies.
#### Word Timestamps and Speaker Attribution with a Non-Autoregressive LLM
 - **Authors:** Zvi Kons, Avihu Dekel, Hagai Aronowitz, Vishal Sunder, Ron Hoory
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2609.15218

 - **Pdf link:** https://arxiv.org/pdf/2609.15218

 - **Abstract**
 Timestamps and speaker attribution are useful additions to speech recognition, creating a rich text transcript. This information can either be extracted during transcription or aligned to a given transcript. In this paper we present models that add timestamps and speaker information to a given transcript using a non-autoregressive LLM-based architecture. Compared to an autoregressive model built from similar components, the models are more accurate and annotate a given transcript one to two orders of magnitude faster. Compared to other models, our models achieve state-of-the-art timestamp accuracy and the best cpWER for speaker attribution.
#### Reducing the Output-Mode Gap in Speech Language Models via Joint-Output On-Policy Distillation
 - **Authors:** Daxin Tan, Dehua Tao, Chengxi Deng, Hanlin Zhang, Xiao Chen
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Computation and Language (cs.CL); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2609.15313

 - **Pdf link:** https://arxiv.org/pdf/2609.15313

 - **Abstract**
 Autoregressive generation of interleaved text and acoustic tokens is a common approach to spoken-response generation in speech large language models. Although this design enables streaming generation with explicit textual guidance, generated acoustic tokens become part of the context for subsequent text predictions. Given identical speech inputs, we observe markedly lower answer accuracy for the internal text generated in speech-to-text-and-speech (S2TS) mode than for speech-to-text (S2T) responses. We term this discrepancy the \emph{output-mode gap} (OMG). To reduce OMG, we propose \emph{Joint-Output On-Policy Distillation} (JO-OPD), which distills the model's stronger S2T policy into joint generation using student-generated S2TS trajectories. At each text position, the S2T teacher provides soft targets from a text-only projection of the student's preceding outputs, while the student predicts from the corresponding full interleaved history. A preservation objective further regularizes native non-text predictions. Experiments on Step-Audio-2-mini and Baichuan-Audio-Instruct reveal OMG across two interleaved generation architectures. On Step-Audio-2-mini, JO-OPD reduces OMG from 42.87 to 16.26 percentage points on Spoken-MQA and from 29.72 to 13.04 points on speech-rendered GSM8K, with little change in S2T accuracy and substantially larger reductions than matched SFT baselines. ASR-based evaluation further shows a 7.49-point improvement in spoken-answer accuracy on Spoken-MQA.
#### OpenEnded: An Open-Response Speech Corpus for Speaking Proficiency Assessment with Human Annotations and ALM Supervision
 - **Authors:** Yu-Wen Chen, Eric Zhou, Evelyn Ding, Tianyi Shen, Zhou Yu, Julia Hirschberg
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2609.15666

 - **Pdf link:** https://arxiv.org/pdf/2609.15666

 - **Abstract**
 The development of automated speaking assessment (ASA) is limited by the scarcity of public datasets, with most existing work relying on read-aloud speech, which limits applicability to real-world communication scenarios. In this work, we introduce OpenEnded, a corpus of English practice speech from Mandarin speakers in open-response tasks. Unlike prior open-response datasets that provide only holistic proficiency scores, OpenEnded offers utterance-level assessments of accuracy, fluency, and prosody. Approximately 10,000 utterances are collected and annotated using a hybrid framework: 1,000 are manually labeled via multi-rater scoring with discrepancy resolution to form a high-quality test set, while the remaining are pseudo-labeled by an audio language model (ALM) for training and development sets. We evaluate ALMs and existing ASA models on the OpenEnded test set and introduce VoxPA as an additional baseline. Results show that ALM-generated pseudo-labels improve training over original ALM scoring, while VoxPA achieves the best performance among all baselines.
#### Directivity-Conditioned Low-Latency Neural Filtering for Speech Enhancement in Hearing Aids
 - **Authors:** Lennart Uphaus, André Merboldt, Markus Hofbauer, Timo Gerkmann
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2609.15760

 - **Pdf link:** https://arxiv.org/pdf/2609.15760

 - **Abstract**
 Latest advances in neural directional filtering show exceptional results in adapting the direction and shape of directivity patterns during the inference phase. However, in the existing methods for adapting directivity patterns during inference, important real-world constraints have been disregarded. Particularly for hearing devices, scenarios are often much more dynamic, microphone positions vary with head diameter and hearing aid placement, head-shadow effects occur, and strict latency constraints apply. In this work, we propose a novel low-latency (10 ms) deep neural network (DNN) taking the above requirements of hearing devices into account. As in recent work, we use feature-wise linear modulation (FiLM) to steer the directivity patterns during testing. To preserve the desired directivity pattern, a loss function is proposed that maintains the spectral cross-channel relationships. Interestingly, we are able to achieve similar results to methods with relaxed latency constraints.
#### The Limits of Reference-Free Speech Quality Metrics as Evaluators and Rewards on Modern Text-to-Speech
 - **Authors:** Antonis Asonitis, Juan Pablo Zuluaga Gomez, Francesco Verdini, Aref Farhadipour, Marzieh Razavi, Pierre-Edouard Honnet, Vijeta Avijeet
 - **Subjects:** Subjects:
Sound (cs.SD); Computation and Language (cs.CL); Machine Learning (cs.LG); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2609.13150

 - **Pdf link:** https://arxiv.org/pdf/2609.13150

 - **Abstract**
 Reference-free quality predictors such as UTMOS, DNSMOS and SCOREQ are the de facto automatic evaluators for text-to-speech (TTS) and are increasingly adopted as reward signals for preference optimization. Both roles presuppose that the predicted score tracks human preference. In this work, we test this assumption across six human-rated corpora spanning the quality range from artifact-rich to defect-free TTS, evaluating each predictor on a pairwise task that asks whether the clip it scores higher is the clip listeners prefer, and we subject interpretable prosodic and signal-processing features to the same protocol. When one clip carries audible defects the predictors tend to agree with listeners. Once both clips are clean, no single predictor reliably identifies the preferred sample, and several fall below the accuracy of simply picking the longest-duration clip. A calibrated composite of complementary signals is the strongest evaluator we test, though on the cleanest audio it recovers only part of the gap to the human ceiling. Additionally, using even an equal-weighted ensemble of metrics helps as a post-training reward, where no calibration data is available. Optimizing a single score with policy optimization induces reward hacking, driving the metric toward its optimum while independent held-out judges and a human listening test deteriorate. The composite reward resists this behavior and tends to improve the model. Our contribution is the evaluation protocol, the predictor scores across these corpora, and the diagnosis of when and why single scores fail.
#### Machine Unlearning for Speech Question Answering in Large Audio-Language Models
 - **Authors:** Zhe Liu
 - **Subjects:** Subjects:
Machine Learning (cs.LG); Computation and Language (cs.CL); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2609.13195

 - **Pdf link:** https://arxiv.org/pdf/2609.13195

 - **Abstract**
 Large Audio-Language Models (LALMs) have recently shown strong capabilities in speech understanding and question answering (QA), but they also inherit privacy risks from large-scale training data, including the unintended memorization of sensitive information. In this work, we study machine unlearning for speech QA in LALMs, a setting that is more challenging than prior work on text-based Large Language Models (LLMs) or Automatic Speech Recognition (ASR) due to the tight coupling between acoustic perception and factual knowledge. We present and evaluate multiple unlearning strategies, including gradient ascent, task arithmetic, and alignment-based fine-tuning methods that enforce safe refusal responses, to remove private knowledge while still preserving performance on core capabilities. Through extensive experiments on speech QA datasets, we show that these unlearning methods can reduce the privacy leakage rate by up to 80% while maintaining near-neutral performance on non-private speech QA and general speech understanding benchmarks.
#### From Masking to Merging: Rethinking SpecAugment for Efficient Audio Spectrogram Transformer
 - **Authors:** Minhee Park, Hyowon Ahn, Chanwoo Kim
 - **Subjects:** Subjects:
Sound (cs.SD); Machine Learning (cs.LG); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2609.13260

 - **Pdf link:** https://arxiv.org/pdf/2609.13260

 - **Abstract**
 This paper proposes SpecAugment-Patch Merging, a simple yet effective method to accelerate Audio Spectrogram Transformer (AST) training. We first apply SpecAugment to mask input spectrograms at the patch level, and after positional embeddings are added, the method selects r pairs of masked patches and merges them, reducing the number of tokens processed by the Transformer. Increasing the number of merged pairs r from 0 to 100 keeps mAP on AudioSet nearly unchanged (34.07 to 34.08) while throughput increases from 43.3 to 49.3 samples/sec, which is a relatively 13.9% improvement. Similar patterns appear on ESC-50 and Speech Commands V2, where throughput steadily improves with only minor accuracy changes, demonstrating that this merging approach provides faster training with minimal performance loss.
#### CVSS-X: A Multilingual Speech-to-Speech Translation Corpus for 28 Languages
 - **Authors:** Lucas Rafael Stefanel Gris, Alef Iury Siqueira Ferreira, Frederico Santos de Oliveira, Augusto Seben da Rosa, Alexandre Costa Ferro Filho, Arlindo Rodrigues Galvão Filho, Anderson da Silva Soares
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2609.13413

 - **Pdf link:** https://arxiv.org/pdf/2609.13413

 - **Abstract**
 We introduce CVSS-X, a large-scale synthetic speech-to-speech translation corpus that extends CVSS by reversing the translation direction. While CVSS translates from 21 languages into English, CVSS-X enables translation from English into 28 target languages spanning 12 language families. The corpus comprises approximately 240,000 parallel speech pairs per language, totaling over 16,000 hours, eight times larger than CVSS. We provide two variants: CVSS-X-C with two canonical voices per language, and CVSS-X-T with cross-lingual voice cloning, both fully generated. Evaluation shows comparable translation quality to CVSS with consistent performance across typologically diverse languages. Combined with CVSS, this enables research on bidirectional and multilingual speech-to-speech translation. The code is available at this https URL and the dataset under CC-BY-NC 4.0 license at this https URL.
#### The VoiceMOS Challenge 2026: Evaluating Speech Enhancement, Emotional TTS and Accented TTS Systems
 - **Authors:** Wen-Chin Huang, Wei Wang, Marvin Sach, Xiaoxue Gao, Nicholas Sanders, Erica Cooper, Toda Tomoki
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2609.13792

 - **Pdf link:** https://arxiv.org/pdf/2609.13792

 - **Abstract**
 We present the results of the VoiceMOS Challenge 2026, the fifth edition of a scientific challenge on automatic prediction of subjective speech assessments. After expanding the scope to music and general audio in 2025, we refocused the evaluation target on speech and organized three tracks: prediction of absolute and comparative category ratings for enhanced speech, prediction of naturalness and emotion-related tasks for emotional text-to-speech systems, and prediction of speaker and accent similarity for codec-based speech synthesis systems. The challenge attracted a total of 18 teams worldwide, with most teams successfully surpassing the provided baselines. We summarize the challenge results, representative top-performing systems, participant feedback, and directions for future editions.
#### Realtime-Venus: A full-duplex interaction system with asynchronous delegation
 - **Authors:** Ruixiang Zhao, Hualei Wang, Renhe Sun, Enzhi Zhou, Jincenzi Wu, Xujie Song, Kexin Shi, Zihang Liu, Pengcheng Zhu, Jiayi Zhou, Baoyue Zhang, Changhao Zhang, Zitong Wang, Jinhong Wang, Tong Niu, Jingjing Liu, Junan Lin, Haolin He, Hengshuo Chu, Yuhui Chen, Jian Liu, Yuge Huang, Junliang Xing, Yuntao Wang, Weiqiang Wang, Chun Yu, Yuanchun Shi
 - **Subjects:** Subjects:
Computer Vision and Pattern Recognition (cs.CV); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2609.13814

 - **Pdf link:** https://arxiv.org/pdf/2609.13814

 - **Abstract**
 Natural interaction in digital and physical environments requires continuous perception and timely responses. Spoken dialogue relies on acoustic and linguistic cues, while video interaction also requires grounding the conversation in evolving visual context. We present Realtime-Venus, a proactive full-duplex interaction system with two separately trained 9B models: Realtime-Venus-Omni for audio-visual interaction and Realtime-Venus-Audio for spoken interaction. Each model serves as a complete conversational frontend, integrating continuous perception, conversational control, and native speech generation through a shared causal timeline for user inputs, model outputs, and delegation events. A dual-loop runtime coordinates live interaction with background reasoning and tool execution. Foreground interaction continues while Realtime-Venus-Harness executes tasks asynchronously and returns results for integration into the ongoing dialogue. Both models follow a common post-training recipe combining offline understanding, proactive full-duplex trajectories, and delegation workflows. Among the evaluated online models, Realtime-Venus-Omni achieves the highest scores on six of eight video benchmarks, including StreamingBench (70.2%), OVO-Bench (64.7%), and Daily-Omni (81.3%). Across eight audio understanding and spoken question answering benchmarks, Realtime-Venus-Audio leads the compared models on MMAU (78.0%), MMAU-Pro (63.2%), Llama Questions (83.8%), and Speech CMMLU (67.8%), while matching the best VoiceBench AlpacaEval score of 4.81. On Full-Duplex-Bench v1.5, Realtime-Venus-Audio responds to 75% of user interruptions and achieves continuation rates of 97%, 88%, and 86% under backchannels, other-directed speech, and background speech, respectively, exceeding Gemini 3.1 Live and GPT-4o on all three continuation metrics.
#### CRAF: Cross-View Residual-Aware Fusion for Deepfake Speech Detection
 - **Authors:** Minh-Xuan Phan, Khalid Zaman, Candy Olivia Mawalim, Masashi Unoki
 - **Subjects:** Subjects:
Sound (cs.SD); Artificial Intelligence (cs.AI); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2609.13842

 - **Pdf link:** https://arxiv.org/pdf/2609.13842

 - **Abstract**
 Recent advances in speech synthesis and voice conversion have made deepfake speech increasingly realistic, making generalization to unseen spoofing attacks a critical challenge. Pretrained speech and audio models offer a promising direction for improving robustness to such unseen attacks. Self-supervised learning (SSL) models capture fine-grained, low-level acoustic characteristics, whereas Auditory Large Language Models (ALLMs) provide higher-level contextual representations. These complementary views can provide useful cues for improving generalization to unseen attacks. However, direct fusion does not explicitly disentangle information shared across the two views from view-specific complementary information, limiting effective cross-view integration. To address this, we propose CRAF, a cross-view residual-aware fusion framework that uses ALLM-guided cross-view attention to enrich SSL representations and adopts ALLM as a high-level reference to separate ALLM-explainable information from complementary SSL residual information. The residual is selectively refined through adaptive gating and integrated through SSL-primary fusion. Experiments on ASVspoof 5 show that CRAF with Kimi-Audio achieves an EER of 5.96% and a minDCF of 0.1192, demonstrating robustness to unseen spoofing attacks.
#### DiTAR+: Dual Optimization for Robust Autoregressive Diffusion Speech Synthesis
 - **Authors:** Ziyu Zhang, Tianlun Zuo, Hanzhao Li, Haoyu Zhang, Lei Xie
 - **Subjects:** Subjects:
Sound (cs.SD); Artificial Intelligence (cs.AI); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2609.13909

 - **Pdf link:** https://arxiv.org/pdf/2609.13909

 - **Abstract**
 Continuous-latent Autoregressive Diffusion Transformer (AR-DiT) models have demonstrated immense potential in zero-shot speech generation. However, they still suffer from limited decoding stability when synthesizing long utterances or complex linguistic structures. This instability primarily stems from a restricted historical receptive field and an acoustic inertia dependency within the diffusion decoder, which causes the model to ignore semantic conditions. To address these challenges, we propose DiTAR+, a dual-optimization framework. First, we introduce Dilated Context Sampling to expand the macro-level historical receptive field without violating physical temporal continuity, thereby preventing cumulative error propagation. Second, we propose Hierarchical Acoustic Masking to prevent shallow layers from attending to acoustic pre-context, explicitly decoupling semantic alignment from acoustic detail reconstruction. Extensive experiments show that our framework effectively mitigates pronunciation errors and semantic hallucinations, enhances generation robustness on challenging sentences, and maintains exceptionally high speaker similarity throughout the entirety of long-form utterances. On the linguistically challenging ZH-Hard set, DiTAR+ reduces the word error rate from 12.478% to 9.893%, and on extended utterances of 25 to 35 seconds it improves speaker similarity from 0.741 to 0.759 while simultaneously lowering the word error rate from 2.778% to 2.173%, outperforming both discrete-token and pure flow-matching baselines.
#### StepAudio 3 Realtime Technical Report
 - **Authors:** Bin Lin, Bo Zhao, Boyang Zhang, Boyong Wu, Chao Yan, Chen Geng, Chen Wu, Cheng Yi, Chengli Feng, Chenglin Zhu, Chengting Feng, Chengyuan Yao, Daijiao Liu, DanNi Wan, Daxin Jiang, Dongjian Li, Dongqing Pang, Fei Tian, Feng Tian, Future Li, Gang Yu, Guanglong Yang, Haoyang Zhang, Hongyuan Wang, Jia Peng, Jiahao Song, Jialong Xue, Jiamin Fan, Jiangjie Zhen, Jianzheng Gao, Jincheng Wen, Jinghua Liang, Jinglan Gong, Jun Chen, Li Xie, Liang Zhao, Lifang Zhang, Lingli Ji, Lun Cai, Min Xu, Peilin Li, Peng Yang, Pengfei Tan, Qingjian Lin, Qinxin Du, Ruijie Xiong, Runze Li, Shenghua Hu, Shengqian Qin, Shi Qiu, Siqi Tu, Siyi Zhou, Tianjiao Deng, Wanying Lu, Weiming Niu, Wen Sun, WenWen Qu, Xiangyu Zhang, Xianwei Zhang, Xiaosu Su, Xing Chen, Xinyu Liu, Xuerui Yang, Yan Wu, Yang Li, Yang Yang, Yechang Huang, Yibo Zhu, Yifan Zhang, Yinuo Yan, Youjun Chen, Yu Fu, Yu Luo, Yu Zhou, Yujie Chen, Yumang Wang, Yunzhou Ju, Yuxiang Yang, Yuxin Li, Yuxin Zhang, Zekai Liu, Zengwei Yao, Zhaoxin Yuan, Zhenwei Mou, Zhiquan Zhang, Zhiyue Wu, Zichao Li, Zichao Zhou, Ziqi Ren, Zixuan Wang
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2609.14005

 - **Pdf link:** https://arxiv.org/pdf/2609.14005

 - **Abstract**
 Realtime spoken interaction demands deep reasoning, prompt responses, and fluid turn-taking. We present StepAudio 3 Realtime, an audio-language foundation model organized around a continuous listen-converse-think-act loop. Deep Perception captures rich acoustic cues to interpret user intent, while Seamless Duplex models synchronized audio streams to handle pauses, backchannels, and interruptions naturally. Crucially, we resolve the tension between deep deliberation and latency via Think-While-Speaking, executing private reasoning in parallel with spoken delivery. In reasoning mode, StepAudio 3 reaches a 73.0 macro average on StepAudioChat. With Think-While-Speaking, it achieves dialogue and reasoning performance comparable to dedicated reasoning models while speaking in real time. Furthermore, an integrated Voice Agent handles asynchronous tool execution without disrupting the dialogue flow. StepAudio 3 Realtime achieves top-tier performance across key dimensions: an exceptional 90.6 on the MMSU benchmark, 98.9 Overall on the Artificial Analysis Full-Duplex Bench, and a 56.0% macro task-success rate on $\tau$-Voice.
#### Robust Cross-Domain Speech-Based Alzheimer's Disease Detection via Iterative Adversarial Self-Training
 - **Authors:** Luqi Sun, Shreeram Suresh Chandra, Aurosweta Mahapatra, Emily Mower Provost, Brian MacWhinney, Berrak Sisman
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2609.14139

 - **Pdf link:** https://arxiv.org/pdf/2609.14139

 - **Abstract**
 As Alzheimer's disease (AD) has increasingly become a major global public health issue, speech-based AD detection has attracted widespread attention. However, most existing methods are trained and evaluated on a single dataset, often leading to severe cross-domain performance degradation due to reliance on dataset-specific artifacts rather than disease-related speech cues. In real-world applications, reliable Alzheimer's disease detection requires models that are robust to variations in recording environments, speakers and data collection conditions. To address this challenge, this paper adopts unsupervised domain adaptation to learn robust, domain-invariant feature representations in the absence of target-domain diagnosis labels. On this basis, a novel unsupervised domain adaptation method, Iterative Adversarial Self-Training (IAST), is proposed. Results demonstrate that IAST significantly improves the generalization ability and robustness under various cross-domain settings.
#### A New Transformer-Based Approach for Audio-Based Kinship Verification and a New Uncontrolled Mandarin Kinship Speech Dataset
 - **Authors:** Qiyang Sun, Langqing Zhang, Yupei Li, Björn Schuller
 - **Subjects:** Subjects:
Sound (cs.SD); Artificial Intelligence (cs.AI); Machine Learning (cs.LG); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2609.14145

 - **Pdf link:** https://arxiv.org/pdf/2609.14145

 - **Abstract**
 Kinship verification is a task involving determining whether two individuals share a first-order kin relation. To tackle this task, we propose CONVTRAP-TN, a new architecture for audio-based kinship verification, and conduct an ablation study on the proposed model. To the best of our knowledge, we are the first to apply the successful transformer architecture to the task of audio-based kinship verification. Furthermore, we also collect a custom speech dataset, ARKIN, which accurately reflects everyday recording conditions. We do this because only a few speech datasets with kinship labels currently exist, all of which either source extremely noisy in-the-wild data from the internet, or instruct speakers to record in specific environments. These settings fail to reflect real-world scenarios where users record on personal devices under unrestrained conditions. Additionally, we perform a series of preliminary baseline experiments on the collected dataset, including speaker verification and recognition, speech recognition, age estimation, and kinship verification, as well as cross-dataset kinship verification experiments to show that existing methods are not robust across datasets.
#### Exploiting Speech LLM Representations for Multilingual and Cross-Lingual Parkinson's Disease Detection
 - **Authors:** Sarthak Giri, Zi Haur Pang, Tatsuya Kawahara
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2609.14431

 - **Pdf link:** https://arxiv.org/pdf/2609.14431

 - **Abstract**
 Speech Large Language Models (Speech LLMs) have shown strong performance across diverse tasks, yet their utility for pathological speech analysis remains underexplored. In this work, we investigate the effectiveness of internal representations from encoder and decoder components of Speech LLMs for Parkinson's Disease (PD) detection across multilingual and cross-lingual settings. Our findings reveal that encoder representations consistently outperform their decoder counterparts in most models and settings and that pathological cues may be progressively attenuated as audio representations are projected into the language model space. We further show that generative outputs are less reliable for clinical tasks compared to internal representations. To leverage information spread across multiple layers, we propose a Squeeze-and-Excitation (SE)-based dynamic layer aggregation framework, which surpasses best-layer selection in multiple experiments, suggesting that PD-relevant acoustic cues are distributed across transformer layers rather than concentrated in one.
#### Grounded in Sound: Reinforcement Learning with a Frozen Acoustic Judge to Curb ASR Insertion Hallucinations
 - **Authors:** Tingzhen Xiong, Rilin Chen, Weiwei Li, Wentao Zhang, Qicong Xie
 - **Subjects:** Subjects:
Sound (cs.SD); Multimedia (cs.MM); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2609.14455

 - **Pdf link:** https://arxiv.org/pdf/2609.14455

 - **Abstract**
 When reinforcement learning (RL) is used for post-training automatic speech recognition (ASR), the reward almost always lives in the text space: it compares a hypothesis with the reference and never checks whether the hypothesis is supported by the audio. On highly regular speech this licenses a shortcut - guessing from a strong language prior rather than listening. Once the acoustics degrade, the shortcut runs unchecked and emits fluent but ungrounded words, i.e., insertion errors. We propose an acoustic-fidelity reward: a GRPO reward augmented with a separately pretrained, permanently frozen, non-autoregressive character-level wav2vec2-CTC acoustic judge, used strictly at training and absent at inference, where a single model decodes greedily. Trained on LibriSpeech and evaluated across a six-tier difficulty gradient including real AMI meeting speech (33,282 utterance-condition instances), the method reduces insertion errors by 28.3% on close-talking AMI-IHM and 22.3% on far-field AMI-SDM, while lowering WER on AMI-SDM from 35.89% to 34.71% and showing no detectable WER difference on the other five tiers, against a schedule-matched WER-GRPO baseline. The insertion reduction holds under a meeting-level clustered bootstrap. Four prespecified analyses support content-conditioned insertion calibration: output collapses 85-90% on unintelligible audio that preserves energy and voice activity; the gain is not recovered by the evaluated 32-best CTC rescoring configuration, yet RL internalizes it into a single greedy decoding run; and policy-only confidence yields lower insertion-AURC in all four evaluated settings. We frame this as a mechanism paper, demonstrated in one instantiation: a 7B speech LLM with a 0.3B CTC judge.
#### Neyshekar: An Open Persian Read-Speech Corpus for Automatic Speech Recognition
 - **Authors:** Ahmad Amirivojdan, Farzad Nadiri, Abolfazl Alizadeh, Shaghayegh Yaraghi
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2609.14542

 - **Pdf link:** https://arxiv.org/pdf/2609.14542

 - **Abstract**
 Neyshekar is presented as an open Persian read-speech corpus designed for coverage of both formal and informal language, named entities, and longer utterances. In version 6, 62,279 validated recordings totalling 99.02 hours are provided from 190 contributors, with 34,541 distinct recorded prompts. The prompt pool was assembled from human-written material, contextualised homographs, and reviewed language-model-generated text. Text entries were normalised with the shekar library, which supports both formal and informal Persian, and every submitted recording was reviewed against a common validation rubric. About 24% of released clips are classified as informal by an automatic classifier; these register labels are not human-validated. Item-level rater labels are provided for reproducible agreement estimation, opaque per-clip contributor identifiers make the speaker-disjoint partitioning auditable and support contributor-clustered uncertainty estimates, and a text-disjoint test subset is included for evaluation beyond previously seen prompts. Per-contributor recording load and reference-free signal quality are characterised for every released clip. Corpus characteristics are compared with Persian Common Voice under shared processing. Utility is assessed through two ASR architectures, three optimisation seeds, WER and CER, and independent evaluation on the public PSRB sample. Against duration-matched Common Voice training at approximately 32 hours, in-domain WER is reduced by 9.5 points for Whisper and 11.6 points for XLS-R, and by approximately eight points for both architectures on the independent PSRB sample. Transfer and mixture benefits are not consistently observed across architectures and training budgets. The corpus is released under CC0; code and data are made available through the project repository at this https URL.
#### CCMAN: Cognitive Instability-Aware Cross-Modal Attention Network for Interpretable Temporal Biomarkers of Verbal Fluency Speech
 - **Authors:** Madhurananda Pahar, Caitlin Illingworth, Dorota Braun, Daniel Blackburn, Heidi Christensen
 - **Subjects:** Subjects:
Sound (cs.SD); Machine Learning (cs.LG); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2609.14764

 - **Pdf link:** https://arxiv.org/pdf/2609.14764

 - **Abstract**
 Early detection of cognitive decline from speech offers a scalable and non-invasive alternative to conventional clinical assessment. Verbal fluency tasks are particularly informative, but most automated approaches aggregate features across an entire recording, overlooking temporal speech dynamics. We propose the Cognitive Instability-Aware Cross-Modal Attention Network (CCMAN), a transfer learning framework that learns task-agnostic cognitive speech representations from multiple memory-probing tasks before fine-tuning on a minute-long semantic and phonemic verbal fluency task. CCMAN integrates semantic, acoustic, and linguistic information through bidirectional cross-attention, gated multimodal fusion, and transformer-based temporal modelling to derive interpretable biomarkers of cognitive decline. Experiments were conducted on 165.44 hours of speech from 843 participants (498 healthy controls, 245 with mild cognitive impairment, and 100 with dementia). CCMAN achieved Macro-F1 scores of 0.81 and 0.59 for binary and multiclass semantic fluency classification, and 0.77 and 0.53 for phonemic fluency, consistently outperforming strong static and temporal baselines. Statistical analyses showed that semantic drift variance and pause variance, but not mean semantic drift, were significantly elevated in both MCI and dementia relative to healthy controls, while pause duration increased progressively over the task with the steepest slope in dementia, supporting global and progressive temporal speech instability as interpretable biomarkers. Evaluation on the independent PROCESS-2 benchmark further demonstrated the generalisability of the proposed framework, improving the baseline Macro-F1 by up to 9%. These findings support temporal speech instability as a dynamic speech biomarker for robust, interpretable, and generalisable early detection of cognitive decline.
#### CAL-MOS: Bridging Layers with Adapters for Robust MOS Prediction Across Speech Foundation Models
 - **Authors:** Alef Iury Siqueira Ferreira, Pedro Lustosa Rege Botelho, Fernanda Silva, Daniel Casanova, Rafael Faustino, Frederico Oliveira, Arlindo Galvão Filho, Anderson da Silva Soares
 - **Subjects:** Subjects:
Sound (cs.SD); Artificial Intelligence (cs.AI); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2609.14956

 - **Pdf link:** https://arxiv.org/pdf/2609.14956

 - **Abstract**
 Speech Quality Assessment (SQA) is essential for modern speech technologies, and recent non-intrusive SQA predictors increasingly rely on Speech Foundation Models (SFMs). However, because SFMs expose representations from many layers, it remains unclear which depths are most informative for MOS prediction and how multi-layer information should be combined reliably across backbones and datasets. We benchmark ten SFMs on four MOS datasets under three regimes: full fine-tuning, last-layer probing with a frozen encoder, and naive cross-layer weighted aggregation. We find that the best layer is strongly backbone- and dataset-dependent, and that naive weighted fusion can be unstable across settings. We further evaluate a layer-calibrated aggregation variant that applies per-layer adapters before pooling, which improves the robustness of multi-layer fusion and narrows the gap to full fine-tuning while keeping the backbone frozen.
#### Typhoon ASR Streaming: Steerable Low-Latency Thai Speech Recognition with Real-Time Shallow Fusion
 - **Authors:** Warit Sirichotedumrong, Tanawin Samutsin, Shah Faisal Wani, Sittipong Sripaisarnmongkol, Kunat Pipatanakul
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2609.14991

 - **Pdf link:** https://arxiv.org/pdf/2609.14991

 - **Abstract**
 Open Thai automatic speech recognition (ASR) is dominated by offline, Whisper-based models that read the whole utterance before transcribing, ruling out low-latency uses such as live captioning and voice agents. We present a deployable system for streaming Thai ASR that lets a user steer its vocabulary at decode time, without retraining. A widely used open Thai model, trained with full context, collapses when run as a true stream; we restore streaming with a cache-aware encoder, by converting it or adapting a natively streaming one, and add a shallow-fusion layer that re-ranks candidates inside the streaming decoder with a GPU n-gram language model and phrase boosting. Across two Thai benchmarks and two model sizes, the streaming models stay usable where the full-context model fails, cutting character error rate 4.3-4.5x at a one-second look-ahead while running faster than real time. Decode-time steering then lifts keyword recall from 16.6% to 20.7% at no accuracy cost and negligible overhead; most of the gain comes from an n-gram over ordinary training transcripts, which resolves the written form of code-switched words the model hears but spells inconsistently, with phrase boosting adding targeted control over rare domain terms.


by Zyzzyva0381 (Windy). 


2026-09-15
