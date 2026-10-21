# Showing new listings for Wednesday, 21 October 2026
Auto update papers at about 2:30am UTC (10:30am Beijing time) every weekday.


阅读 `Usage.md`了解如何使用此repo实现个性化的Arxiv论文推送

See `Usage.md` for instructions on how to personalize the repo. 


Keyword list: ['text-to-speech', 'text to speech', 'tts', 'LLM-based', 'speech', 'voice']


Excluded: []


### Today: 10papers 
#### The Impact of Side Information on Voice Privacy
 - **Authors:** Eulalie Thiombiano, Martha Larson, Vincent Colotte, Emmanuel Vincent
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Multimedia (cs.MM); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2610.26884

 - **Pdf link:** https://arxiv.org/pdf/2610.26884

 - **Abstract**
 Currently, voice-related privacy risks are studied under the assumption that the attacker has access only to spoken audio. In contrast, real-world collections often pair the speech signal with side information; yet to our knowledge, no publicly available dataset provides sufficiently detailed and directly linkable speech and side-information pairs to evaluate the combined risk. We propose a distribution-constrained method to build a semi-synthetic dataset and study singling out risk when geographic side information is assumed to accompany anonymized speech. We instantiate it using the English portion of Mozilla Common Voice and the 2021 UK Census, under ignorant and semi-informed attackers, and introduce a conditional reference that isolates the contribution of the audio beyond the candidate-set reduction induced by the side information. In our controlled geographic case study, for single-utterance speakers, the semi-informed attacker reaches an observed isolation probability of 0.718 in the finest OA-plus-household setting, compared with 0.380 in the signal-only setting; its excess over the conditional reference increases from 0.012 to 0.340. The ignorant attacker shows the same trend at a lower level. The effect is largest at fine granularity and short recordings, precisely the setting in which signal-only evaluation looks safest. Our results show that geographic side information can increase the singling out risk of anonymized speech, and we recommend including attacker settings with side information in future studies of voice privacy.
#### Disentangling Depression from Cognitive Decline in Elderly Speech Using Concurrent Clinical Assessments
 - **Authors:** Woori Jeon, Seunghee Ha, Sang-Kyu Lee, Ji Hye Yoon, Tae-Jin Yoon, Seung Jin Lee, Woojae Han, Jungmin So
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Artificial Intelligence (cs.AI); Computation and Language (cs.CL); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2610.26890

 - **Pdf link:** https://arxiv.org/pdf/2610.26890

 - **Abstract**
 Detecting depression from speech in elderly patients with mild cognitive impairment (MCI) is complicated by overlapping acoustic effects of cognitive decline. Without disentangling these, classifiers risk learning cognitive rather than depression-specific patterns. We present a Korean elderly speech corpus of 89 MCI speakers collected over three years with concurrent depression (SGDS) and cognitive (MMSE) assessments. Both linear mixed-effects models and partial correlations identify the same pattern: of seven feature groups, only formant features are associated with depression after controlling for cognitive function, while widely used F0 features carry no signal. The F1+F2 subset achieves UAR 0.760 with just 12 features, outperforming both the full eGeMAPS baseline and self-supervised representations. These findings show that concurrent clinical assessments can identify interpretable, depression-specific acoustic markers in populations where cognitive and affective symptoms co-occur.
#### PW-Pruning: Phase-Aware Expert Selection for Speech-Centric Omni-Modal Models
 - **Authors:** Junjie Cao, Fengping Tian, Wenjie Zhou, Xuanfan Ni, Chenyang Lyu, Siyou Liu, Longyue Wang, Weihua Luo
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Artificial Intelligence (cs.AI); Computation and Language (cs.CL); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2610.26891

 - **Pdf link:** https://arxiv.org/pdf/2610.26891

 - **Abstract**
 Sparse routing reduces per-token computation but leaves the full expert bank resident in memory. We study expert pruning for speech-centric omni-modal models, where pooling calibration statistics can obscure the different expert demands of acoustic-input processing and text-answer generation. We propose PW-Pruning, a phase-aware extension of DO-ACP that constructs separate importance-weighted kernels for acoustic-prefill and teacher-forced answer tokens, then combines phase-weighted log-determinant marginal gains to select one static expert bank. On Qwen3-Omni, retaining 32 of 128 experts per layer improves MMAU-mini and MMSU accuracy over matched pooled selection by 3.20 and 4.38 percentage points, respectively, after supervised fine-tuning, averaged over three runs. Single-phase and shuffled-label controls on this public backbone suggest that semantic phase information contributes to these gains. Experiments on a non-public Qwen3.5-Omni backbone and English-only zero-shot evaluation of Nemotron provide complementary evidence. Across the tested Qwen budgets, AU gains are larger under stronger pruning, without inference-time mask switching.
#### ShlokaDhara: A Multi-Speaker Sanskrit Speech Corpus for Learning Meter-Aware Prosody in Text-to-Recitation
 - **Authors:** Harshal Kausadikar, Onkar Kishor Susladkar, Prathmesh Santosh Choudhari, Vedangi Deepak Deshpande, Ismini Lorentzu, Sparsh Mittal
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Machine Learning (cs.LG); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2610.26929

 - **Pdf link:** https://arxiv.org/pdf/2610.26929

 - **Abstract**
 Sanskrit verse recitation is governed by $\textit{Chandas}$, a strict prosodic framework defined by rhythmic $\textit{Laghu}$ (light) and $\textit{Guru}$ (heavy) syllables. Because existing Sanskrit speech datasets prioritize ASR or conversational TTS, they lack the metrical fidelity required for authentic verse synthesis. We introduce $\textbf{\textit{ShlokaDhara}}$, a 43-hour, multi-speaker Sanskrit speech corpus capturing diverse meters, recitation styles, and melodic variations. We specifically focus on $\textit{Text-to-Recite}$, a synthesis task explicitly requiring both prosodic accuracy and metrical correctness. Benchmarking state-of-the-art TTS architectures via an automated UTMOS (neural MOS) predictor demonstrates that XTTS-v2 achieves the highest perceptual quality. We further validate the XTTS-v2 generated samples using two paradigms: subjective assessments by 16 Sanskrit experts and novel empirical metrics, specifically, syllable-level $\textit{Laghu-Guru}$ duration analysis to objectively quantify structural metrical awareness. Ultimately, this work establishes a robust dataset and evaluation benchmark for prosody-aware computational Sanskrit recitation. Generated Samples: this https URL
#### RealComm: Benchmarking and Adapting Audio Deepfake Detection over Real Communication Channels
 - **Authors:** Li Wang, Jindong Wang, Wan Lin, Kunyu Feng, Yufan Lin, Lei Wang, Rizhao Cai, Ce Fang, Jinzhe Xue, Jie Shi, Haizhou Li, Zhizheng Wu
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2610.27459

 - **Pdf link:** https://arxiv.org/pdf/2610.27459

 - **Abstract**
 Acoustic capture, device processing, and transmission coding in real calls jointly alter speech signals and may affect the reliability of audio deepfake detection. To systematically evaluate these effects, we introduce RealComm, a dataset pairing digital speech with recordings of the same utterances transmitted through real mobile calls. The acquisition design covers acoustic and wired injection, handset and speakerphone operation, and caller-side denoising, with bona fide and synthetic speech recorded under matched conditions. The dataset includes RealCommBench for evaluation and RealCommTrain for adaptation. We further develop RealComm-Aug, a staged waveform augmentation pipeline for simulating acquisition and transmission effects, and investigate its combination with real-call training. Experiments with existing detectors reveal substantial degradation after call transmission, even for models with strong digital-domain performance; acoustic injection is particularly challenging. Paired quality and spectral analyses show that quality degradation is associated with poorer detection across recording conditions for the evaluated detectors, while high-frequency attenuation alone does not explain detection difficulty. Simulated augmentation and real-call training provide complementary benefits, with their combination achieving the lowest call-domain error for each of three evaluated detectors. For Teffic-Audio, the equal error rate decreases from 37.94% to 22.37%. The demo and benchmark access information are provided through the project repository.
#### VISH-ASR: Viterbi-Guided Selection of Hotwords for Speech-LLM Contextual ASR
 - **Authors:** Xiaoyu Gu, Yu Xi, Chencheng Wang, Zonghan Zhou, Haoyu Li, Jing Peng, Kai Yu
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Artificial Intelligence (cs.AI); Human-Computer Interaction (cs.HC)
 - **Arxiv link:** https://arxiv.org/abs/2610.27867

 - **Pdf link:** https://arxiv.org/pdf/2610.27867

 - **Abstract**
 Prompt-based contextual ASR requires selecting a compact hotword subset from large dynamic inventories, yet existing pre-selection methods rely either on discrete ASR hypotheses that may discard acoustic evidence or on separately trained retrieval models. We propose Viterbi-guided Selection of Hotwords (VISH), a training-free, plug-and-play framework that directly verifies candidate-specific acoustic support over frame-level CTC posteriors. VISH formulates pre-selection as constrained posterior search; M-VISH makes large-list verification scalable through frame-synchronous shared-prefix Viterbi search with bounded top-K pruning, while Bi-MVISH exploits complementary backward evidence for robust verification. Across five benchmarks, M-VISH achieves up to 11.7% absolute improvement in Best-F1 over a strong learned-retrieval baseline and reduces candidate-search latency by 75.2% at 2,000 candidates relative to hypothesis-based filtering. These results demonstrate the effectiveness of posterior-level verification for accurate and scalable hotword pre-selection without additional model training.
#### Multimodal Ambivalence and Hesitancy Recognition with Bidirectional Audio--Text Cross-Attention
 - **Authors:** TingTing Gu, JinCheng Luo, WenQiu Guo
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2610.26868

 - **Pdf link:** https://arxiv.org/pdf/2610.26868

 - **Abstract**
 Ambivalence and hesitancy (A/H) are subtle behavioural states that can be expressed through both what a person says and how it is spoken. This paper presents an audio--text framework for video-level A/H recognition. Full-video transcripts and speech waveforms are encoded using pre-trained multilingual BERT and wav2vec~2.0, respectively, and projected into a shared representation space. Two stacked bidirectional cross-attention blocks allow linguistic tokens to attend to acoustic frames and acoustic frames to attend to linguistic context without requiring explicit temporal alignment. The resulting sequences are aggregated through masked mean pooling and combined using concatenation, element-wise products, and absolute differences for binary classification. We evaluate the model on the subject-independent split of the Behavioural Ambivalence/Hesitancy (BAH) dataset. The proposed system achieves 72.19\% accuracy, 75.00\% precision, 81.13\% recall, 77.95\% A/H F1-score, and 70.16\% Macro-F1 on the test set. These results demonstrate the potential of jointly modelling linguistic and acoustic information for automatic A/H this http URL at:this https URL.
#### Commanding at the Speed of Sound: Audio-Centric Referring Expression Segmentation for Aerial Scenes
 - **Authors:** Fan Liu, Ziyun Chen, Yijun Wang, Chuanyi Zhang, Liang Yao, Yuexuan An, Xiang Gu, Pai Peng
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS); Image and Video Processing (eess.IV)
 - **Arxiv link:** https://arxiv.org/abs/2610.26872

 - **Pdf link:** https://arxiv.org/pdf/2610.26872

 - **Abstract**
 Referring expression segmentation (RES) aims to localize targets at the pixel level guided by human instructions. However, current models predominantly follow a text-centric paradigm that treats discrete text as the primary interaction medium for these human instructions. This assumption is especially limiting in unmanned aerial vehicle (UAV) applications, where rapid response, hands-free operation, and natural communication are often critical. In this paper, we reconstruct UAV referring expression segmentation through an audio-centric perspective. Grounded in information theory, we demonstrate that raw audio provides a higher upper bound for extracting rich semantic features than transcribed text. To achieve this, we construct an audio-driven UAV referring expression segmentation benchmark named Aerial-AudioRES by augmenting VisDrone with synthesized speech, forming "audio-text-image-mask" quadruples. Specifically, we employ an ambiguity filtering mechanism to ensure referential uniqueness, and incorporate eight distinct timbres alongside 4 realistic noises to simulate authentic environments. To drive the paradigm shift from text-centric to audio-centric, we propose the end-to-end Audio-centric Referring Expression Segmentation Network (ARES-Net). Distinct from discrete text-visual matching, our approach learns to establish a direct soft alignment between continuous acoustic manifolds and multi-scale visual representations. Comprehensive experiments show that our method obtains a peak accuracy of 64.23% in oIoU and demonstrates robust generalization on a collected real-human test set. Furthermore, it significantly reduces total inference latency to one-third of the text-based method. Our code is publicly available.
#### Mixture of Phonetic Experts Based Low-Rank Adaptation of Conformer Models for Accented English Speech Recognition
 - **Authors:** Santosh Dahal, Anmol Guragain, Tianchi Liu, Luis Fernando D'Haro, Kiran Chandra Dahal
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2610.26888

 - **Pdf link:** https://arxiv.org/pdf/2610.26888

 - **Abstract**
 We present Mixture of Phonetic Experts based Low-Rank Adaptation (MoPE-LoRA), a parameter-efficient adaptation framework for accented English speech recognition. Recent mixture-of-experts approaches typically allocate experts per accent, implicitly treating accent identity as the primary axis of variation. In contrast, we organize adaptation along phonetic categories, motivated by the observation that accent differences are often expressed as systematic distortions of phoneme realization. MoPE-LoRA introduces a fixed set of six low-rank experts aligned with manner-of-articulation classes and performs frame-level routing through a hybrid mechanism that combines phoneme supervision with learned acoustic gating. Because the expert set is shared across accents, the model does not require accent-specific modules and naturally extends to previously unseen accents. Experiments on L2-ARCTIC show that MoPE-LoRA consistently improves over single LoRA and full fine-tuning. In zero-shot evaluation where one accent is held out during training, the proposed method yields a 12.3% relative improvement over single LoRA. These results indicate that phonetic-category-based decomposition offers a scalable and transferable alternative to accent-level adaptation.
#### Beyond Two Tracks: Speaker-Anchored Full-Duplex Dialogue Recovery
 - **Authors:** Yuxuan Chen, Yang Xiao, Jack Sun, Chak Sun, Brick Li, Sam
 - **Subjects:** Subjects:
Sound (cs.SD); Computation and Language (cs.CL); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2610.27109

 - **Pdf link:** https://arxiv.org/pdf/2610.27109

 - **Abstract**
 Spoken interaction is moving from alternating turns to full-duplex conversation, where an agent listens while speaking. Learning this coordination requires natural dialogue with persistent speaker-resolved tracks, a data requirement that remains difficult to satisfy at scale. Generative recovery offers a route from ordinary recordings, but recovering two voices is not enough: their track assignments can drift as the conversation unfolds. We address this missing link with speaker-anchored routing before diffusion: a 1.84M-parameter module matches the separator's coarse streams to fixed enrollment identities and orders them before a frozen generator reconstructs the speech, without retraining the generator or altering the reconstructed audio. On 25-minute remote conversations, routing cuts wrong-track time from 43.9% to 4.1%, below the 6.2% of matched-enrollment waveform re-identification, with its largest advantage when one participant holds the floor (17.8% versus 28.6%), and raises fixed-track activity F1 from 0.46 to 0.84 while per-chunk reconstruction quality is unchanged. The same router transfers to unseen face-to-face and telephone recordings, and a margin-gated verification step brings their wrong-track time to 0.9-1.7% while embedding 20-36% of the output, where full re-identification embeds all of it. Recovered voices thus become the participant-consistent tracks that full-duplex dialogue modeling needs. Audio examples: this https URL


by Zyzzyva0381 (Windy). 


2026-10-21
