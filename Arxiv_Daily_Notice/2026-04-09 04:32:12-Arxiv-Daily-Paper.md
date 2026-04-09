# Showing new listings for Thursday, 9 April 2026
Auto update papers at about 2:30am UTC (10:30am Beijing time) every weekday.


阅读 `Usage.md`了解如何使用此repo实现个性化的Arxiv论文推送

See `Usage.md` for instructions on how to personalize the repo. 


Keyword list: ['text-to-speech', 'text to speech', 'tts', 'LLM-based', 'speech', 'voice']


Excluded: []


### Today: 4papers 
#### Harf-Speech: A Clinically Aligned Framework for Arabic Phoneme-Level Speech Assessment
 - **Authors:** Asif Azad, MD Sadik Hossain Shanto, Mohammad Sadat Hossain, Bdour Alwuqaysi, Sabri Boughorbel, Yahya Bokhari, Abdulrhman Aljouie, Ayah Othman Sindi, Ehsan Hoque
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Artificial Intelligence (cs.AI); Computation and Language (cs.CL); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2604.06191

 - **Pdf link:** https://arxiv.org/pdf/2604.06191

 - **Abstract**
 Automated phoneme-level pronunciation assessment is vital for scalable speech therapy and language learning, yet validated tools for Arabic remain scarce. We present Harf-Speech, a modular system scoring Arabic pronunciation at the phoneme level on a clinical scale. It combines an MSA phonetizer, a fine-tuned speech-to-phoneme model, Levenshtein alignment, and a blended scorer using longest common subsequence and edit-distance metrics. We fine-tune three ASR architectures on Arabic phoneme data and benchmark them with zero-shot multimodal models; the best, OmniASR-CTC-1B-v2, achieves 8.92\% phoneme error rate. Three certified speech-language pathologists independently scored 40 utterances for clinical validation. Harf-Speech attains a Pearson correlation of 0.791 and ICC(2,1) of 0.659 with mean expert scores, outperforming existing end-to-end assessment frameworks. These results show Harf-Speech yields clinically aligned, interpretable scores comparable to inter-rater expert agreement.
#### ULTRAS -- Unified Learning of Transformer Representations for Audio and Speech Signals
 - **Authors:** Ameenudeen P E, Charumathi Narayanan, Sriram Ganapathy
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2604.06702

 - **Pdf link:** https://arxiv.org/pdf/2604.06702

 - **Abstract**
 Self-supervised learning (SSL) has driven impressive advances in speech processing by adopting time-domain prediction objectives, while audio representation learning frameworks operate on time-frequency spectrograms. Models optimized for one paradigm struggle to transfer to the other, highlighting the need for a joint framework. We propose Unified Learning of Transformer Representations for Audio and Speech (ULTRAS), where the masking and predictive modeling is performed over long patches of the data. The model, based on the transformer architecture, encodes spectral-patches of log-mel spectrogram features. The predictive modeling of masked segments is performed on spectral and temporal targets using a combined loss-function, forcing the representations to encode time and frequency traits. Experiments are performed on a variety of speech and audio tasks, where we illustrate that the ULTRAS framework achieves improved performance over other established baselines.
#### DAT-CFTNet: Speech Enhancement for Cochlear Implant Recipients using Attention-based Dual-Path Recurrent Neural Network
 - **Authors:** Nursadul Mamun, John H.L. Hansen
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2604.06744

 - **Pdf link:** https://arxiv.org/pdf/2604.06744

 - **Abstract**
 The human auditory system has the ability to selectively focus on key speech elements in an audio stream while giving secondary attention to less relevant areas such as noise or distortion within the background, dynamically adjusting its attention over time. Inspired by the recent success of attention models, this study introduces a dual-path attention module in the bottleneck layer of a concurrent speech enhancement network. Our study proposes an attention-based dual-path RNN (DAT-RNN), which, when combined with the modified complex-valued frequency transformation network (CFTNet), forms the DAT-CFTNet. This attention mechanism allows for precise differentiation between speech and noise in time-frequency (T-F) regions of spectrograms, optimizing both local and global context information processing in the CFTNet. Our experiments suggest that the DAT-CFTNet leads to consistently improved performance over the existing models, including CFTNet and DCCRN, in terms of speech intelligibility and quality. Moreover, the proposed model exhibits superior performance in enhancing speech intelligibility for cochlear implant (CI) recipients, who are known to have severely limited T-F hearing restoration (e.g., >10%) in CI listener studies in noisy settings show the proposed solution is capable of suppressing non-stationary noise, avoiding the musical artifacts often seen in traditional speech enhancement methods. The implementation of the proposed model will be publicly available.
#### EvoTSE: Evolving Enrollment for Target Speaker Extraction
 - **Authors:** Zikai Liu, Ziqian Wang, Xingchen Li, Yike Zhu, Shuai Wang, Longshuai Xiao, Lei Xie
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2604.06810

 - **Pdf link:** https://arxiv.org/pdf/2604.06810

 - **Abstract**
 Target Speaker Extraction (TSE) aims to isolate a specific speaker's voice from a mixture, guided by a pre-recorded enrollment. While TSE bypasses the global permutation ambiguity of blind source separation, it remains vulnerable to speaker confusion, where models mistakenly extract the interfering speaker. Furthermore, conventional TSE relies on static inference pipeline, where performance is limited by the quality of the fixed enrollment. To overcome these limitations, we propose EvoTSE, an evolving TSE framework in which the enrollment is continuously updated through reliability-filtered retrieval over high-confidence historical estimates. This mechanism reduces speaker confusion and relaxes the quality requirements for pre-recorded enrollment without relying on additional annotated data. Experiments across multiple benchmarks demonstrate that EvoTSE achieves consistent improvements, especially when evaluated on out-of-domain (OOD) scenarios. Our code and checkpoints are available.


by Zyzzyva0381 (Windy). 


2026-04-09
