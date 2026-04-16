# Showing new listings for Thursday, 16 April 2026
Auto update papers at about 2:30am UTC (10:30am Beijing time) every weekday.


阅读 `Usage.md`了解如何使用此repo实现个性化的Arxiv论文推送

See `Usage.md` for instructions on how to personalize the repo. 


Keyword list: ['text-to-speech', 'text to speech', 'tts', 'LLM-based', 'speech', 'voice']


Excluded: []


### Today: 3papers 
#### ProSDD: Learning Prosodic Representations for Speech Deepfake Detection against Expressive and Emotional Attacks
 - **Authors:** Aurosweta Mahapatra, Ismail Rasim Ulgen, Kong Aik Lee, Nicholas Andrews, Berrak Sisman
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2604.13229

 - **Pdf link:** https://arxiv.org/pdf/2604.13229

 - **Abstract**
 Speech deepfake detection (SDD) systems perform well on standard benchmarks datasets but often fail to generalize to expressive and emotional spoofing attacks. Many methods rely on spoof-heavy training data, learning dataset-specific artifacts rather than transferable cues of natural speech. In contrast, humans internalize variability in real speech and detect fakes as deviations from it. We introduce ProSDD, a two-stage framework that enriches model embeddings through supervised masked prediction of speaker-conditioned prosodic variation based on pitch, voice activity, and energy. Stage I learns prosodic variability from real speech, and Stage II jointly optimizes this objective with spoof classification. ProSDD consistently outperforms baselines under both ASVspoof 2019 and 2024 training, reducing ASVspoof 2024 EER from 25.43% to 16.14% (2019-trained) and from 39.62% to 7.38% (2024-trained), while achieving 50% relative reductions on EmoFake and EmoSpoof-TTS.
#### Classical Machine Learning Baselines for Deepfake Audio Detection on the Fake-or-Real Dataset
 - **Authors:** Faheem Ahmad, Ajan Ahmed, Masudul Imtiaz
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2604.13400

 - **Pdf link:** https://arxiv.org/pdf/2604.13400

 - **Abstract**
 Deep learning has enabled highly realistic synthetic speech, raising concerns about fraud, impersonation, and disinformation. Despite rapid progress in neural detectors, transparent baselines are needed to reveal which acoustic cues reliably separate real from synthetic speech. This paper presents an interpretable classical machine learning baseline for deepfake audio detection using the Fake-or-Real (FoR) dataset. We extract prosodic, voice-quality, and spectral features from two-second clips at 44.1 kHz (high-fidelity) and 16 kHz (telephone-quality) sampling rates. Statistical analysis (ANOVA, correlation heatmaps) identifies features that differ significantly between real and fake speech. We then train multiple classifiers -- Logistic Regression, LDA, QDA, Gaussian Naive Bayes, SVMs, and GMMs -- and evaluate performance using accuracy, ROC-AUC, EER, and DET curves. Pairwise McNemar's tests confirm statistically significant differences between models. The best model, an RBF SVM, achieves ~93% test accuracy and ~7% EER on both sampling rates, while linear models reach ~75% accuracy. Feature analysis reveals that pitch variability and spectral richness (spectral centroid, bandwidth) are key discriminative cues. These results provide a strong, interpretable baseline for future deepfake audio detectors.
#### Few-Shot and Pseudo-Label Guided Speech Quality Evaluation with Large Language Models
 - **Authors:** Ryandhimas E. Zezario, Dyah A. M. G. Wisnu, Szu-Wei Fu, Sabato Marco Siniscalchi, Hsin-Min Wang, Yu Tsao
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2604.13528

 - **Pdf link:** https://arxiv.org/pdf/2604.13528

 - **Abstract**
 In this paper, we introduce GatherMOS, a novel framework that leverages large language models (LLM) as meta-evaluators to aggregate diverse signals into quality predictions. GatherMOS integrates lightweight acoustic descriptors with pseudo-labels from DNSMOS and VQScore, enabling the LLM to reason over heterogeneous inputs and infer perceptual mean opinion scores (MOS). We further explore both zero-shot and few-shot in-context learning setups, showing that zero-shot GatherMOS maintains stable performance across diverse conditions, while few-shot guidance yields large gains when support samples match the test conditions. Experiments on the VoiceBank-DEMAND dataset demonstrate that GatherMOS consistently outperforms DNSMOS, VQScore, naive score averaging, and even learning-based models such as CNN-BLSTM and MOS-SSL when trained under limited labeled-data conditions. These results highlight the potential of LLM-based aggregation as a practical strategy for non-intrusive speech quality evaluation.


by Zyzzyva0381 (Windy). 


2026-04-16
