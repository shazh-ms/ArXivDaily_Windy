# Showing new listings for Thursday, 5 March 2026
Auto update papers at about 2:30am UTC (10:30am Beijing time) every weekday.


阅读 `Usage.md`了解如何使用此repo实现个性化的Arxiv论文推送

See `Usage.md` for instructions on how to personalize the repo. 


Keyword list: ['text-to-speech', 'text to speech', 'tts', 'LLM-based', 'speech', 'voice']


Excluded: []


### Today: 7papers 
#### The PARLO Dementia Corpus: A German Multi-Center Resource for Alzheimer's Disease
 - **Authors:** Franziska Braun, Christopher Witzl, Florian Hönig, Elmar Nöth, Tobias Bocklet, Korbinian Riedhammer
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2603.03471

 - **Pdf link:** https://arxiv.org/pdf/2603.03471

 - **Abstract**
 Early and accessible detection of Alzheimer's disease (AD) remains a major challenge, as current diagnostic methods often rely on costly and invasive biomarkers. Speech and language analysis has emerged as a promising non-invasive and scalable approach to detecting cognitive impairment, but research in this area is hindered by the lack of publicly available datasets, especially for languages other than English. This paper introduces the PARLO Dementia Corpus (PDC), a new multi-center, clinically validated German resource for AD collected across nine academic memory clinics in Germany. The dataset comprises speech recordings from individuals with AD-related mild cognitive impairment and mild to moderate dementia, as well as cognitively healthy controls. Speech was elicited using a standardized test battery of eight neuropsychological tasks, including confrontation naming, verbal fluency, word repetition, picture description, story reading, and recall tasks. In addition to audio recordings, the dataset includes manually verified transcriptions and detailed demographic, clinical, and biomarker metadata. Baseline experiments on ASR benchmarking, automated test evaluation, and LLM-based classification illustrate the feasibility of automatic, speech-based cognitive assessment and highlight the diagnostic value of recall-driven speech production. The PDC thus establishes the first publicly available German benchmark for multi-modal and cross-lingual research on neurodegenerative diseases.
#### Cyclostationarity Analysis as a Complement to Self-Supervised Representations for Speech Deepfake Detection
 - **Authors:** Cemal Hanilçi, Md Sahidullah, Tomi Kinnunen
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2603.03921

 - **Pdf link:** https://arxiv.org/pdf/2603.03921

 - **Abstract**
 Speech deepfake detection (SDD) is essential for maintaining trust in voice-driven technologies and digital media. Although recent SDD systems increasingly rely on self-supervised learning (SSL) representations that capture rich contextual information, complementary signal-driven acoustic features remain important for modeling fine-grained structural properties of speech. Most existing acoustic front ends are based on time-frequency representations, which do not fully exploit higher-order spectral dependencies inherent in speech signals. We introduce a cyclostationarity-inspired acoustic feature extraction framework for SDD based on spectral correlation density (SCD). The proposed features model periodic statistical structures in speech by capturing spectral correlations between frequency components. In particular, we propose temporally structured SCD features that characterize the evolution of spectral and cyclic-frequency components over time. The effectiveness and complementarity of the proposed features are evaluated using multiple countermeasure architectures, including convolutional neural networks, SSL-based embedding systems, and hybrid fusion models. Experiments on ASVspoof 2019 LA, ASVspoof 2021 DF, and ASVspoof 5 demonstrate that SCD-based features provide complementary discriminative information to SSL embeddings and conventional acoustic representations. In particular, fusion of SSL and SCD embeddings reduces the equal error rate on ASVspoof 2019 LA from $8.28\%$ to $0.98\%$, and yields consistent improvements on the challenging ASVspoof 5 dataset. The results highlight cyclostationary signal analysis as a theoretically grounded and effective front end for speech deepfake detection.
#### FlowW2N: Whispered-to-Normal Speech Conversion via Flow-Matching
 - **Authors:** Fabian Ritter-Gutierrez, Md Asif Jalal, Pablo Peso Parada, Karthikeyan Saravanan, Yusun Shul, Minseung Kim, Gun-Woo Lee, Han-Gil Moon
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2603.04296

 - **Pdf link:** https://arxiv.org/pdf/2603.04296

 - **Abstract**
 Whispered-to-normal (W2N) speech conversion aims to reconstruct missing phonation from whispered input while preserving content and speaker identity. This task is challenging due to temporal misalignment between whisper and voiced recordings and lack of paired data. We propose FlowW2N, a conditional flow matching approach that trains exclusively on synthetic, time-aligned whisper-normal pairs and conditions on domain-invariant features. We exploit high-level ASR embeddings that exhibits strong invariance between synthetic and real whispered speech, enabling generalization to real whispers despite never observing it during training. We verify this invariance across ASR layers and propose a selection criterion optimizing content informativeness and cross-domain invariance. Our method achieves SOTA intelligibility on the CHAINS and wTIMIT datasets, reducing Word Error Rate by 26-46% relative to prior work while using only 10 steps at inference and requiring no real paired data.
#### Automated Measurement of Geniohyoid Muscle Thickness During Speech Using Deep Learning and Ultrasound
 - **Authors:** Alisher Myrgyyassov, Bruce Xiao Wang, Yu Sun, Shuming Huang, Zhen Song, Min Ney Wong, Yongping Zheng
 - **Subjects:** Subjects:
Quantitative Methods (q-bio.QM); Machine Learning (cs.LG); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2603.03350

 - **Pdf link:** https://arxiv.org/pdf/2603.03350

 - **Abstract**
 Manual measurement of muscle morphology from ultrasound during speech is time-consuming and limits large-scale studies. We present SMMA, a fully automated framework that combines deep-learning segmentation with skeleton-based thickness quantification to analyze geniohyoid (GH) muscle dynamics. Validation demonstrates near-human-level accuracy (Dice = 0.9037, MAE = 0.53 mm, r = 0.901). Application to Cantonese vowel production (N = 11) reveals systematic patterns: /a:/ shows significantly greater GH thickness (7.29 mm) than /i:/ (5.95 mm, p < 0.001, Cohen's d > 1.3), suggesting greater GH activation during production of /a:/ than /i:/, consistent with its role in mandibular depression. Sex differences (5-8% greater in males) reflect anatomical scaling. SMMA achieves expert-validated accuracy while eliminating the need for manual annotation, enabling scalable investigations of speech motor control and objective assessment of speech and swallowing disorders.
#### ACES: Accent Subspaces for Coupling, Explanations, and Stress-Testing in Automatic Speech Recognition
 - **Authors:** Swapnil Parekh
 - **Subjects:** Subjects:
Sound (cs.SD); Artificial Intelligence (cs.AI); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2603.03359

 - **Pdf link:** https://arxiv.org/pdf/2603.03359

 - **Abstract**
 ASR systems exhibit persistent performance disparities across accents, yet the internal mechanisms underlying these gaps remain poorly understood. We introduce ACES, a representation-centric audit that extracts accent-discriminative subspaces and uses them to probe model fragility and disparity. Analyzing Wav2Vec2-base with five English accents, we find that accent information concentrates in a low-dimensional early-layer subspace (layer 3, k=8). Projection magnitude correlates with per-utterance WER (r=0.26), and crucially, subspace-constrained perturbations yield stronger coupling between representation shift and degradation (r=0.32) than random-subspace controls (r=0.15). Finally, linear attenuation of this subspace however does not reduce disparity and slightly worsens it. Our findings suggest that accent-relevant features are deeply entangled with recognition-critical cues, positioning accent subspaces as vital diagnostic tools rather than simple "erasure" levers for fairness.
#### Robust LLM-based Audio-Visual Speech Recognition with Sparse Modality Alignment and Visual Unit-Guided Refinement
 - **Authors:** Fei Su, Cancan Li, Juan Liu, Wei Ju, Hongbin Suo, Ming Li
 - **Subjects:** Subjects:
Sound (cs.SD); Multimedia (cs.MM); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2603.03811

 - **Pdf link:** https://arxiv.org/pdf/2603.03811

 - **Abstract**
 Audio-Visual Speech Recognition (AVSR) integrates acoustic and visual information to enhance robustness in adverse acoustic conditions. Recent advances in Large Language Models (LLMs) have yielded competitive automatic speech recognition performance and shown effectiveness for AVSR. However, prior approaches project audio and visual features independently or apply shallow fusion, limiting cross-modal alignment and complementary exchange while increasing the LLM's computational load. To address this, we propose AVUR-LLM, an LLM-based Audio-Visual Speech Recognition via Sparse Modality Alignment and Visual Unit-Guided Refinement. Experiments on LRS3 demonstrate state-of-the-art results for AVSR. Under additive-noise conditions at 0 dB SNR, it achieves 37% relative improvement over the baseline system.
#### ZeSTA: Zero-Shot TTS Augmentation with Domain-Conditioned Training for Data-Efficient Personalized Speech Synthesis
 - **Authors:** Youngwon Choi, Jinwoo Oh, Hwayeon Kim, Hyeonyu Kim
 - **Subjects:** Subjects:
Sound (cs.SD); Artificial Intelligence (cs.AI); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2603.04219

 - **Pdf link:** https://arxiv.org/pdf/2603.04219

 - **Abstract**
 We investigate the use of zero-shot text-to-speech (ZS-TTS) as a data augmentation source for low-resource personalized speech synthesis. While synthetic augmentation can provide linguistically rich and phonetically diverse speech, naively mixing large amounts of synthetic speech with limited real recordings often leads to speaker similarity degradation during fine-tuning. To address this issue, we propose ZeSTA, a simple domain-conditioned training framework that distinguishes real and synthetic speech via a lightweight domain embedding, combined with real-data oversampling to stabilize adaptation under extremely limited target data, without modifying the base architecture. Experiments on LibriTTS and an in-house dataset with two ZS-TTS sources demonstrate that our approach improves speaker similarity over naive synthetic augmentation while preserving intelligibility and perceptual quality.


by Zyzzyva0381 (Windy). 


2026-03-05
