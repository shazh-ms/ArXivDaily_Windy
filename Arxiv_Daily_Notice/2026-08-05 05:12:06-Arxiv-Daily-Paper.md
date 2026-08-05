# Showing new listings for Wednesday, 5 August 2026
Auto update papers at about 2:30am UTC (10:30am Beijing time) every weekday.


阅读 `Usage.md`了解如何使用此repo实现个性化的Arxiv论文推送

See `Usage.md` for instructions on how to personalize the repo. 


Keyword list: ['text-to-speech', 'text to speech', 'tts', 'LLM-based', 'speech', 'voice']


Excluded: []


### Today: 6papers 
#### Towards Real-world Environment-aware Zero-shot Text-to-speech Synthesis via Disentangled Audio Infilling
 - **Authors:** Ye-Xin Lu, Xin Wang, Yang Ai, Hui-Peng Du, Zhen-Hua Ling, Junichi Yamagishi
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Signal Processing (eess.SP)
 - **Arxiv link:** https://arxiv.org/abs/2608.03011

 - **Pdf link:** https://arxiv.org/pdf/2608.03011

 - **Abstract**
 Recent zero-shot text-to-speech (TTS) systems achieve remarkable naturalness and speaker similarity but typically require high-quality speaker prompts and either strip away or entangle the acoustic environment with speaker characteristics, limiting their real-world applicability. We present an extended DAIEN-TTS, an environment-aware zero-shot TTS framework that disentangles and jointly models speech, background noise, and reverberation, enabling independent control over timbre and acoustic environment through separate speaker and environment prompts. Built upon the flow-matching-based F5-TTS, it uses a speech-environment separation module to decompose environmental speech into speech, noise, and reverberation components, which are injected into the Diffusion Transformer for environment-aware generation. Training uses simulated data constructed by mixing clean speech with noise and room impulse responses, together with a cross-speaker conditioning strategy that suppresses speaker information leakage from the environment branch. When real-world data are available, the system can be further fine-tuned to bridge the simulated-to-real domain this http URL inference, a triple classifier-free guidance mechanism enables fine-grained control over speech, noise, and reverberation, and a signal-to-noise-ratio adaptation strategy aligns the synthesized speech with the environment prompt. Experiments on simulated and real-world test sets show that DAIEN-TTS generates environmental personalized speech with high naturalness, strong speaker similarity, and faithful noise and reverberation reproduction, while offering controllability beyond prior environment-aware TTS systems.
#### GROW: Group-Relative Advantage-Weighted On-Policy Reinforcement Learning of Autoregressive-Diffusion Text-to-Speech model
 - **Authors:** Guanrou Yang, Tian Tan, Qian Chen, Ziyang Ma, Yakun Song, Zhikang Niu, Qi Chen, Wenming Tu, Haitao Li, Shan Yang, Xie Chen
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Artificial Intelligence (cs.AI); Computation and Language (cs.CL)
 - **Arxiv link:** https://arxiv.org/abs/2608.03215

 - **Pdf link:** https://arxiv.org/pdf/2608.03215

 - **Abstract**
 Reinforcement learning for flow-matching text-to-speech is complicated by deterministic ODE sampling: trajectory-level policy-gradient methods typically convert the ODE into an SDE and track per-step likelihood ratios, introducing stochastic perturbations and substantial overhead. We propose GROW, a group-relative advantage-weighted on-policy RL method that acts directly on the standard flow-matching objective. For each prompt, GROW samples a group of on-policy utterances, separately standardizes intelligibility and speaker-similarity rewards within the group, and combines them to reweight flow-matching regression. A Wasserstein-2 velocity penalty anchors the updated model to a frozen pretrained reference. A group-mean reward baseline is introduced to convert reward weighting into advantage weighting. For strong pretrained TTS models with concentrated rewards, positive exponential weighting is dominated by reward-agnostic self-imitation, whereas a zero-mean signed advantage preserves effective within-group credit assignment. Instantiated on DiTAR and evaluated on LibriSpeech and Seed-TTS EN/ZH, GROW reduces average WER from 2.016 to 1.558 and raises speaker similarity from 0.676 to 0.715 while keeping UTMOS. With 10-NFE training rollouts and 32-NFE evaluation, GROW retains comparable performance while training 2.9x faster than 32-NFE DiTAR-GRPO. We will open-source complete GROW codes, faithful DiTAR reproduction, and all model checkpoints.
#### Speaker Verification Under Real Classroom Conditions for English Speech
 - **Authors:** Saba Tabatabaee, Jing Liu, Megh Krishnaswamy, Carol Espy-Wilson
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2608.03623

 - **Pdf link:** https://arxiv.org/pdf/2608.03623

 - **Abstract**
 Developing speaker verification (SV) models that are robust to classroom noise and effective across both children and adult speakers is critical for AI tools supporting educational environments. In this study, we use a real-world English-speaking classrooms dataset containing partial speaker identity annotations, with most recordings remaining unlabeled. We adapt the WavLM-TDNN model for classroom SV, achieving average relative reductions in Equal Error Rate (EER) of 23.99% and 6.32% compared to the ECAPA-TDNN baseline and the ECAPA-TDNN model trained on classroom data, respectively. Additionally, we investigate two training strategies for SV in classroom settings: self-supervised learning (SSL) and a two-stage approach that first pre-trains with SSL and then fine-tunes with limited annotated data. Five-fold cross-validation demonstrates that the two-stage strategy consistently outperforms the SSL-only approach, achieving an average relative EER reduction of 13.39%.
#### Identity-Faithful Audio-Visual Target Speaker Extraction with QIANGDA and VOXBLINK2-AVSE
 - **Authors:** Peijun Yang, Zhan Jin, Juan Liu, Ming Li
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2608.03964

 - **Pdf link:** https://arxiv.org/pdf/2608.03964

 - **Abstract**
 Audio-visual target speaker extraction should return the speaker indicated by the video, yet a separator can ignore the visual cue and repeatedly output the acoustically dominant voice. We introduce QIANGDA, a Mandarin AV-TSE benchmark of jointly recorded real two-speaker mixtures with synchronized multi-view video. Each scene also contains preceding A-only and B-only stages that provide in-scene speaker references. It contains 77 scenes and 7,598 clips (11.84 hours), including 6,042 dual-annotated mixtures. After processing there leave 6,038 evaluable mixtures and 12,076 target-speaker rows. We additionally curate VOXBLINK2-AVSE from VoxBlink2, comprising 250,828 synchronized audio--lip-ROI pairs from 28,421 identities and 766.17 hours of speech. Our extractor uses frozen, 1,280-dimensional projected AV-HuBERT features, target-conditioned training, and layer-wise feature modulation. We jointly evaluate content with Qwen3-ASR-1.7B CER and target identity with WeSpeaker ResNet34 plus Overlapped Speech Detection (OSD). On the complete manifest, the best archived checkpoint obtains 0.2261 CER, 82.22% strict output correctness, and 69.53% both-output strict success.
#### dots.tts.edit: Precisely Controlled Speech Editing with a Continuous Autoregressive Model
 - **Authors:** Hankun Wang, Bohan Li, Shi Lian, Xiaoyu Gu, Jing Peng, Da Zheng, Colin Zhang, Kai Yu
 - **Subjects:** Subjects:
Sound (cs.SD); Artificial Intelligence (cs.AI); Computation and Language (cs.CL); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2608.02673

 - **Pdf link:** https://arxiv.org/pdf/2608.02673

 - **Abstract**
 Speech editing for content creation requires precise control over both what an edit should do and where it should apply. Free-form natural language provides a flexible interface for expressing edit requests, but its ambiguity may leave the intended operation, parameters, or target region underspecified. We study a precise and explicit interface for speech editing: a transcript-grounded structural edit instruction with XML-style tags explicitly specifies typed operations and localizes them to transcript spans or boundaries. This semantic timeline avoids explicit timestamp alignment and provides an externally inspectable contract for compositional edits. We instantiate the interface in this http URL, an editor adapted from the continuous autoregressive this http URL foundation model. Four representative speech-creation controls cover lexical content, affective expression, pitch and speaking-rate delivery, and temporal phrasing through text, emotion, prosody, and pause editing. Task-specific data pipelines construct operation- and scope-controlled pairs while retaining source-derived context outside each target region. We further introduce doteBench, a bilingual evaluation suite that measures precise instruction following, local preservation, and audio quality across the four controls and their composition. Experiments show leading overall instruction following and local preservation across its five editing categories, while audio quality remains comparable to existing open-source systems. Across three Seed-TTS-Eval shards, the model shows negligible differences from the base model in zero-shot TTS recognition error rate and speaker similarity. The code and model will be released soon.
#### Language-Specialized Multi-Teacher On-Policy Distillation for Multilingual LLM-Based ASR
 - **Authors:** Yuan Xie, Jiaqi Song, Xianliang Wang, Ming Lei, Jie Gao, Jie Wu
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2608.03610

 - **Pdf link:** https://arxiv.org/pdf/2608.03610

 - **Abstract**
 Modern LLM-based ASR systems have established multilingual capability as a standard feature, leveraging large-scale multilingual corpora and LLMs' cross-lingual knowledge to achieve competitive performance across multilingual benchmarks. However, joint modeling of languages with heterogeneous acoustic, phonological, and lexical characteristics inevitably introduces optimization conflicts, undermining language-wise specialization. To address this challenge, we propose Language-Specialized Multi-Teacher On-Policy Distillation (LS-MOPD), which decouples language-specific knowledge acquisition from multilingual capability integration: language-specialized teachers are independently optimized via reinforcement learning (RL), after which their expertise is integrated into a generalist multilingual student through language routing and token-level multi-teacher distillation, thereby reducing direct cross-lingual optimization conflicts. We further explore two acoustic-prefix configurations, static and dynamic, to examine how teacher--student prefix consistency influences the efficacy of on-policy distillation. Experiments on benchmarks covering Mandarin, Mandarin subdialects, Cantonese, and English demonstrate that LS-MOPD substantially outperforms RL baselines and consistently surpasses the empirical performance envelope defined by best-performing RL teachers, revealing its potential to generalize beyond all teachers in multilingual ASR.


by Zyzzyva0381 (Windy). 


2026-08-05
