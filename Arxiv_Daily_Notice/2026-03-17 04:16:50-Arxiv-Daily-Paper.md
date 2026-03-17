# Showing new listings for Tuesday, 17 March 2026
Auto update papers at about 2:30am UTC (10:30am Beijing time) every weekday.


阅读 `Usage.md`了解如何使用此repo实现个性化的Arxiv论文推送

See `Usage.md` for instructions on how to personalize the repo. 


Keyword list: ['text-to-speech', 'text to speech', 'tts', 'LLM-based', 'speech', 'voice']


Excluded: []


### Today: 17papers 
#### BrainWhisperer: Leveraging Large-Scale ASR Models for Neural Speech Decoding
 - **Authors:** Tommaso Boccato, Michal Olak, Matteo Ferrante
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2603.13321

 - **Pdf link:** https://arxiv.org/pdf/2603.13321

 - **Abstract**
 Decoding continuous speech from intracortical recordings is a central challenge for brain-computer interfaces (BCIs), with transformative potential for individuals with conditions that impair their ability to speak. While recent microelectrode array (MEA) decoders achieve impressive accuracy, their performance is fundamentally limited by the small size of existing datasets, they remain brittle to session-to-session variability, and their ability to generalize across participants remains unexplored. We introduce BrainWhisperer, a neural speech decoder that integrates high-resolution MEA recordings with a large pretrained automatic speech recognition (ASR) model. Building on interpretability findings showing that Whisper's encoder learns phoneme-selective representations with localized attention, we train a customized version of Whisper, modified to process neural features, using a hybrid objective that combines CTC loss on phonemes--predicted from the third encoder layer--and cross-entropy loss on word tokens. We introduce domain-informed modifications including windowed self-attention to capture articulatory continuity, hierarchical month/day-specific low-rank projections to address non-stationarity, and subject-specific embedders enabling cross-subject training. Evaluated on a publicly available MEA dataset (Card et al.), BrainWhisperer matches or outperforms prior state-of-the-art decoders. Critically, cross-dataset training improves performance even on individual datasets without fine-tuning, demonstrating unprecedented generalization. The model supports dual decoding paths: a high-accuracy phoneme-based path with external language model rescoring, and a fast direct text generation path enabling sub-100ms inference with minimal hardware requirements.
#### Understanding the strengths and weaknesses of SSL models for audio deepfake model attribution
 - **Authors:** Gabriel Pîrlogeanu, Adriana Stan, Horia Cucu
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2603.13488

 - **Pdf link:** https://arxiv.org/pdf/2603.13488

 - **Abstract**
 Audio deepfake model attribution aims to mitigate the misuse of synthetic speech by identifying the source model responsible for generating a given audio sample, enabling accountability and informing vendors. The task is challenging, but self-supervised learning (SSL)-derived acoustic features have demonstrated state-of-the-art attribution capabilities, yet the underlying factors driving their success and the limits of their discriminative power remain unclear. In this paper, we systematically investigate how SSL-derived features capture architectural signatures in audio deepfakes. By controlling multiple dimensions of the audio generation process we reveal how subtle perturbations in model checkpoints, text prompts, vocoders, or speaker identity influence attribution. Our results provide new insights into the robustness, biases, and limitations of SSL-based deepfake attribution, highlighting both its strengths and vulnerabilities in realistic scenarios.
#### VoXtream2: Full-stream TTS with dynamic speaking rate control
 - **Authors:** Nikita Torgashov, Gustav Eje Henter, Gabriel Skantze
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Computation and Language (cs.CL); Human-Computer Interaction (cs.HC); Machine Learning (cs.LG); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2603.13518

 - **Pdf link:** https://arxiv.org/pdf/2603.13518

 - **Abstract**
 Full-stream text-to-speech (TTS) for interactive systems must start speaking with minimal delay while remaining controllable as text arrives incrementally. We present VoXtream2, a zero-shot full-stream TTS model with dynamic speaking-rate control that can be updated mid-utterance on the fly. VoXtream2 combines a distribution matching mechanism over duration states with classifier-free guidance across conditioning signals to improve controllability and synthesis quality. Prompt-text masking enables textless audio prompting, removing the need for prompt transcription. Across standard zero-shot benchmarks and a dedicated speaking-rate test set, VoXtream2 achieves competitive objective and subjective results against public baselines despite a smaller model and less training data. In full-stream mode, it runs 4 times faster than real time with 74 ms first-packet latency on a consumer GPU.
#### Beyond Two-stage Diffusion TTS: Joint Structure and Content Refinement via Jump Diffusion
 - **Authors:** Jiabao Ai, Minghui Zhao, Anton Ragni
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2603.14032

 - **Pdf link:** https://arxiv.org/pdf/2603.14032

 - **Abstract**
 Diffusion and flow matching TTS faces a tension between discrete temporal structure and continuous spectral modeling. Two-stage models diffuse on fixed alignments, often collapsing to mean prosody; single-stage models avoid explicit durations but suffer alignment instability. We propose a jump-diffusion framework where discrete jumps model temporal structure and continuous diffusion refines spectral content within one process. Even in its one-shot degenerate form, our framework achieves 3.37% WER vs. 4.38% for Grad-TTS with improved UTMOSv2 on LJSpeech. The full iterative UDD variant further enables adaptive prosody, autonomously inserting natural pauses in out-of-distribution slow speech rather than stretching uniformly. Audio samples are available at this https URL.
#### Controllable Accent Normalization via Discrete Diffusion
 - **Authors:** Qibing Bai, Yuhan Du, Tom Ko, Shuai Wang, Yannan Wang, Haizhou Li
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Artificial Intelligence (cs.AI); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2603.14275

 - **Pdf link:** https://arxiv.org/pdf/2603.14275

 - **Abstract**
 Existing accent normalization methods do not typically offer control over accent strength, yet many applications-such as language learning and dubbing-require tunable accent retention. We propose DLM-AN, a controllable accent normalization system built on masked discrete diffusion over self-supervised speech tokens. A Common Token Predictor identifies source tokens that likely encode native pronunciation; these tokens are selectively reused to initialize the reverse diffusion process. This provides a simple yet effective mechanism for controlling accent strength: reusing more tokens preserves more of the original accent. DLM-AN further incorporates a flow-matching Duration Ratio Predictor that automatically adjusts the total duration to better match the native rhythm. Experiments on multi-accent English data show that DLM-AN achieves the lowest word error rate among all compared systems while delivering competitive accent reduction and smooth, interpretable accent strength control.
#### SoulX-Duplug: Plug-and-Play Streaming State Prediction Module for Realtime Full-Duplex Speech Conversation
 - **Authors:** Ruiqi Yan, Wenxi Chen, Zhanxun Liu, Ziyang Ma, Haopeng Lin, Hanlin Wen, Hanke Xie, Jun Wu, Yuzhe Liang, Yuxiang Zhao, Pengchao Feng, Jiale Qian, Hao Meng, Yuhang Dai, Shunshun Yin, Ming Tao, Lei Xie, Kai Yu, Xinsheng Wang, Xie Chen
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2603.14877

 - **Pdf link:** https://arxiv.org/pdf/2603.14877

 - **Abstract**
 Recent advances in spoken dialogue systems have brought increased attention to human-like full-duplex voice interactions. However, our comprehensive review of this field reveals several challenges, including the difficulty in obtaining training data, catastrophic forgetting, and limited scalability. In this work, we propose SoulX-Duplug, a plug-and-play streaming state prediction module for full-duplex spoken dialogue systems. By jointly performing streaming ASR, SoulX-Duplug explicitly leverages textual information to identify user intent, effectively serving as a semantic VAD. To promote fair evaluation, we introduce SoulX-Duplug-Eval, extending widely used benchmarks with improved bilingual coverage. Experimental results show that SoulX-Duplug enables low-latency streaming dialogue state control, and the system built upon it outperforms existing full-duplex models in overall turn management and latency performance. We have open-sourced SoulX-Duplug and SoulX-Duplug-Eval.
#### Modeling and Benchmarking Spoken Dialogue Rewards with Modality and Colloquialness
 - **Authors:** Jingyu Lu, Yuhan Wang, Fan Zhuo, Xize Cheng, Changhao Pan, Xueyi Pu, Yifu Chen, Chenyuhao Wen, Tianle Liang, Zhou Zhao
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Computation and Language (cs.CL); Machine Learning (cs.LG)
 - **Arxiv link:** https://arxiv.org/abs/2603.14889

 - **Pdf link:** https://arxiv.org/pdf/2603.14889

 - **Abstract**
 The rapid evolution of end-to-end spoken dialogue systems demands transcending mere textual semantics to incorporate paralinguistic nuances and the spontaneous nature of human conversation. However, current methods struggle with two critical gaps: the modality gap, involving prosody and emotion, and the colloquialness gap, distinguishing written scripts from natural speech. To address these challenges, we introduce SDiaReward, an end-to-end multi-turn reward model trained on SDiaReward-Dataset, a novel collection of episode-level preference pairs explicitly targeting these gaps. It operates directly on full multi-turn speech episodes and is optimized with pairwise preference supervision, enabling joint assessment of modality and colloquialness in a single evaluator. We further establish ESDR-Bench, a stratified benchmark for robust episode-level evaluation. Experiments demonstrate that SDiaReward achieves state-of-the-art pairwise preference accuracy, significantly outperforming general-purpose audio LLMs. Further analysis suggests that SDiaReward captures relative conversational expressiveness beyond superficial synthesis cues, improving generalization across domains and recording conditions. Code, data, and demos are available at this https URL.
#### Spectrogram features for audio and speech analysis
 - **Authors:** Ian McLoughlin, Lam Pham, Yan Song, Xiaoxiao Miao, Huy Phan, Pengfei Cai, Qing Gu, Jiang Nan, Haoyu Song, Donny Soh
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Artificial Intelligence (cs.AI); Machine Learning (cs.LG); Signal Processing (eess.SP)
 - **Arxiv link:** https://arxiv.org/abs/2603.14917

 - **Pdf link:** https://arxiv.org/pdf/2603.14917

 - **Abstract**
 Spectrogram-based representations have grown to dominate the feature space for deep learning audio analysis systems, and are often adopted for speech analysis also. Initially, the primary motivator for spectrogram-based representations was their ability to present sound as a two dimensional signal in the time-frequency plane, which not only provides an interpretable physical basis for analysing sound, but also unlocks the use of a wide range of machine learning techniques such as convolutional neural networks, that had been developed for image processing. A spectrogram is a matrix characterised by the resolution and span of its two dimensions, as well as by the representation and scaling of each element. Many possibilities for these three characteristics have been explored by researchers across numerous application areas, with different settings showing affinity for various tasks. This paper reviews the use of spectrogram-based representations and surveys the state-of-the-art to question how front-end feature representation choice allies with back-end classifier architecture for different tasks.
#### Deep Filter Estimation from Inter-Frame Correlations for Monaural Speech Dereverberation
 - **Authors:** Ui-Hyeop Shin, Jun Hyung Kim, Jangyeon Kim, Wooseok Kim, Hyung-Min Park
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2603.14986

 - **Pdf link:** https://arxiv.org/pdf/2603.14986

 - **Abstract**
 Speech dereverberation in distant-microphone scenarios remains challenging due to the high correlation between reverberation and target signals, often leading to poor generalization in real-world environments. We propose IF-CorrNet, a correlation-to-filter architecture designed for robustness against acoustic variability. Unlike conventional black-box mapping methods that directly estimate complex spectra, IF-CorrNet explicitly exploits inter-frame STFT correlations to estimate multi-frame deep filters for each time-frequency bin. By shifting the learning objective from direct mapping to filter estimation, the network effectively constrains the solution space, which simplifies the training process and mitigates overfitting to synthetic data. Experimental results on the REVERB Challenge dataset demonstrate that IF-CorrNet achieves a substantial gain in the SRMR metric on RealData, confirming its robustness in suppressing reverberation and noise in practical, non-synthetic environments.
#### LLMs and Speech: Integration vs. Combination
 - **Authors:** Robin Schmitt, Albert Zeyer, Mohammad Zeineldeen, Ralf Schlüter, Hermann Ney
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2603.15045

 - **Pdf link:** https://arxiv.org/pdf/2603.15045

 - **Abstract**
 In this work, we study how to best utilize pre-trained LLMs for automatic speech recognition. Specifically, we compare the tight integration of an acoustic model (AM) with the LLM ("speech LLM") to the traditional way of combining AM and LLM via shallow fusion. For tight integration, we provide ablations on the effect of different label units, fine-tuning strategies, LLM sizes and pre-training data, attention interfaces, encoder downsampling, text prompts, and length normalization. Additionally, we investigate joint recognition with a CTC model to mitigate hallucinations of speech LLMs and present effective optimizations for this joint recognition. For shallow fusion, we investigate the effect of fine-tuning the LLM on the transcriptions using different label units, and we compare rescoring AM hypotheses to single-pass recognition with label-wise or delayed fusion of AM and LLM scores. We train on Librispeech and Loquacious and evaluate our models on the HuggingFace ASR leaderboard.
#### How Attention Shapes Emotion: A Comparative Study of Attention Mechanisms for Speech Emotion Recognition
 - **Authors:** Marc Casals-Salvador, Federico Costa, Rodolfo Zevallos, Javier Hernando
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2603.15120

 - **Pdf link:** https://arxiv.org/pdf/2603.15120

 - **Abstract**
 Speech Emotion Recognition (SER) plays a key role in advancing human-computer interaction. Attention mechanisms have become the dominant approach for modeling emotional speech due to their ability to capture long-range dependencies and emphasize salient information. However, standard self-attention suffers from quadratic computational and memory complexity, limiting its scalability. In this work, we present a systematic benchmark of optimized attention mechanisms for SER, including RetNet, LightNet, GSA, FoX, and KDA. Experiments on both MSP-Podcast benchmark versions show that while standard self-attention achieves the strongest recognition performance across test sets, efficient attention variants dramatically improve scalability, reducing inference latency and memory usage by up to an order of magnitude. These results highlight a critical trade-off between accuracy and efficiency, providing practical insights for designing scalable SER systems.
#### spINAch: A Diachronic Corpus of French Broadcast Speech Controlled for Speakers' Age and Gender
 - **Authors:** Simon Devauchelle, David Doukhan, Rémi Uro, Lucas Ondel Yang, Valentin Pelloin, Olympia Imbert-Brégégère, Véronique Lefort, Kévin Picard, Emeline Seignobos, Albert Rilliard
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2603.15516

 - **Pdf link:** https://arxiv.org/pdf/2603.15516

 - **Abstract**
 We present spINAch, a large diachronic corpus of French speech from radio and television archives, balanced by speakers' gender, age (20-95 years old), and spanning 60 years from 1955 to 2015. The dataset includes over 320 hours of recordings from more than two thousand speakers. The methodology for building the corpus is described, focusing on the quality of collected samples in acoustic terms. The data were automatically transcribed and phonetically aligned to allow studies at a phonemic level. More than 3 million oral vowels have been analyzed to propose their fundamental frequency and formants. The corpus, available to the community for research purposes, is valuable for describing the evolution of Parisian French through the representation of gender and age. The presented analyses also demonstrate that the diachronic nature of the corpus allows the observation of various phonetic phenomena, such as the evolution of voice pitch over time (which does not differ by gender in our data) and the neutralization of the /a/-/$a$/ opposition in Parisian French during this period.
#### LLM-Guided Reinforcement Learning for Audio-Visual Speech Enhancement
 - **Authors:** Chih-Ning Chen, Jen-Cheng Hou, Hsin-Min Wang, Shao-Yi Chien, Yu Tsao, Fan-Gang Zeng
 - **Subjects:** Subjects:
Sound (cs.SD); Artificial Intelligence (cs.AI); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2603.13952

 - **Pdf link:** https://arxiv.org/pdf/2603.13952

 - **Abstract**
 In existing Audio-Visual Speech Enhancement (AVSE) methods, objectives such as Scale-Invariant Signal-to-Noise Ratio (SI-SNR) and Mean Squared Error (MSE) are widely used; however, they often correlate poorly with perceptual quality and provide limited interpretability for optimization. This work proposes a reinforcement learning-based AVSE framework with a Large Language Model (LLM)-based interpretable reward model. An audio LLM generates natural language descriptions of enhanced speech, which are converted by a sentiment analysis model into a 1-5 rating score serving as the PPO reward for fine-tuning a pretrained AVSE model. Compared with scalar metrics, LLM-generated feedback is semantically rich and explicitly describes improvements in speech quality. Experiments on the 4th COG-MHEAR AVSE Challenge (AVSEC-4) dataset show that the proposed method outperforms a supervised baseline and a DNSMOS-based RL baseline in PESQ, STOI, neural quality metrics, and subjective listening tests.
#### What Counts as Real? Speech Restoration and Voice Quality Conversion Pose New Challenges to Deepfake Detection
 - **Authors:** Shree Harsha Bokkahalli Satish, Harm Lameris, Joakim Gustafson, Éva Székely
 - **Subjects:** Subjects:
Sound (cs.SD); Artificial Intelligence (cs.AI); Machine Learning (cs.LG); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2603.14033

 - **Pdf link:** https://arxiv.org/pdf/2603.14033

 - **Abstract**
 Audio anti-spoofing systems are typically formulated as binary classifiers distinguishing bona fide from spoofed speech. This assumption fails under layered generative processing, where benign transformations introduce distributional shifts that are misclassified as spoofing. We show that phonation-modifying voice conversion and speech restoration are treated as out-of-distribution despite preserving speaker authenticity. Using a multi-class setup separating bona fide, converted, spoofed, and converted-spoofed speech, we analyse model behaviour through self-supervised learning (SSL) embeddings and acoustic correlates. The benign transformations induce a drift in the SSL space, compressing bona fide and spoofed speech and reducing classifier separability. Reformulating anti-spoofing as a multi-class problem improves robustness to benign shifts while preserving spoof detection, suggesting binary systems model the distribution of raw speech rather than authenticity itself.
#### CodecMOS-Accent: A MOS Benchmark of Resynthesized and TTS Speech from Neural Codecs Across English Accents
 - **Authors:** Wen-Chin Huang, Nicholas Sanders, Erica Cooper
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2603.14328

 - **Pdf link:** https://arxiv.org/pdf/2603.14328

 - **Abstract**
 We present the CodecMOS-Accent dataset, a mean opinion score (MOS) benchmark designed to evaluate neural audio codec (NAC) models and the large language model (LLM)-based text-to-speech (TTS) models trained upon them, especially across non-standard speech like accented speech. The dataset comprises 4,000 codec resynthesis and TTS samples from 24 systems, featuring 32 speakers spanning ten accents. A large-scale subjective test was conducted to collect 19,600 annotations from 25 listeners across three dimensions: naturalness, speaker similarity, and accent similarity. This dataset does not only represent an up-to-date study of recent speech synthesis system performance but reveals insights including a tight relationship between speaker and accent similarity, the predictive power of objective metrics, and a perceptual bias when listeners share the same accent with the speaker. This dataset is expected to foster research on more human-centric evaluation for NAC and accented TTS.
#### Nudging Hidden States: Training-Free Model Steering for Chain-of-Thought Reasoning in Large Audio-Language Models
 - **Authors:** Lok-Lam Ieong, Chia-Chien Chen, Chih-Kai Yang, Yu-Han Huang, An-Yu Cheng, Hung-yi Lee
 - **Subjects:** Subjects:
Sound (cs.SD); Artificial Intelligence (cs.AI); Computation and Language (cs.CL); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2603.14636

 - **Pdf link:** https://arxiv.org/pdf/2603.14636

 - **Abstract**
 Chain-of-thought (CoT) prompting has been extended to large audio-language models (LALMs) to elicit reasoning, yet enhancing its effectiveness without training remains challenging. We study inference-time model steering as a training-free approach to improve LALM reasoning. We introduce three strategies using diverse information sources and evaluate them across four LALMs and four benchmarks. Results show general accuracy gains up to 4.4% over CoT prompting. Notably, we identify a cross-modal transfer where steering vectors derived from few text samples effectively guide speech-based reasoning, demonstrating high data efficiency. We also examine hyperparameter sensitivity to understand the robustness of these approaches. Our findings position model steering as a practical direction for strengthening LALM reasoning.
#### NV-Bench: Benchmark of Nonverbal Vocalization Synthesis for Expressive Text-to-Speech Generation
 - **Authors:** Qinke Ni, Huan Liao, Dekun Chen, Yuxiang Wang, Zhizheng Wu
 - **Subjects:** Subjects:
Sound (cs.SD); Artificial Intelligence (cs.AI); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2603.15352

 - **Pdf link:** https://arxiv.org/pdf/2603.15352

 - **Abstract**
 While recent text-to-speech (TTS) systems increasingly integrate nonverbal vocalizations (NVs), their evaluations lack standardized metrics and reliable ground-truth references. To bridge this gap, we propose NV-Bench, the first benchmark grounded in a functional taxonomy that treats NVs as communicative acts rather than acoustic artifacts. NV-Bench comprises 1,651 multi-lingual, in-the-wild utterances with paired human reference audio, balanced across 14 NV categories. We introduce a dual-dimensional evaluation protocol: (1) Instruction Alignment, utilizing the proposed paralinguistic character error rate (PCER) to assess controllability, (2) Acoustic Fidelity, measuring the distributional gap to real recordings to assess acoustic realism. We evaluate diverse TTS models and develop two baselines. Experimental results demonstrate a strong correlation between our objective metrics and human perception, establishing NV-Bench as a standardized evaluation framework.


by Zyzzyva0381 (Windy). 


2026-03-17
