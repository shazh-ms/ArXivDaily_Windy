# Showing new listings for Friday, 22 May 2026
Auto update papers at about 2:30am UTC (10:30am Beijing time) every weekday.


阅读 `Usage.md`了解如何使用此repo实现个性化的Arxiv论文推送

See `Usage.md` for instructions on how to personalize the repo. 


Keyword list: ['text-to-speech', 'text to speech', 'tts', 'LLM-based', 'speech', 'voice']


Excluded: []


### Today: 4papers 
#### Effective User-defined Keyword Spotting with Dual-stage Matching, Multi-modal Enrollment, and Continual Adaptation
 - **Authors:** Zhiqi Ai, Han Cheng, Shiyi Mu, Xinnuo Li, Yongjin Zhou, Shugong Xu
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2605.22120

 - **Pdf link:** https://arxiv.org/pdf/2605.22120

 - **Abstract**
 User-defined keyword spotting (KWS) is crucial for personalized voice interaction, yet existing methods face several challenges: (1) insufficient discriminability among confusable words, (2) performance inconsistency across speakers with varying pronunciations, and (3) high data cost to ensure reliable wake-word performance. In this paper, we introduce DMA-KWS, an efficient and robust framework for user-defined keyword spotting. First, it adopts a dual-stage matching pipeline: CTC decoding with streaming phoneme search to locate candidate segments, followed by QbyT with a phoneme matcher for fine-grained verification, enabling it to better distinguish confusable words. Next, multi-modal enrollment fuses user-specific speech with text embeddings to further improve accuracy for registered users. Finally, a parameter-efficient continual adaptation mechanism performs lightweight updates using synthetic and real data. Extensive experiments demonstrate the superior performance of DMA-KWS. On the LibriPhrase Hard subset, it achieves 97.85% AUC and 6.13% EER, reaching state-of-the-art performance. In speaker-dependent settings, DMA-KWS consistently outperforms text-only enrollment, demonstrating significant performance gains. Moreover, the proposed parameter-efficient fine-tuning mechanism adapts DMA-KWS with only 187k updated parameters, further enhancing KWS performance while ensuring suitability for on-device deployment.
#### RobustSpeechFlow: Learning Robust Text-to-Speech Trajectories via Augmentation-based Contrastive Flow Matching
 - **Authors:** Jinhyeok Yang, Hyeongju Kim, Yechan Yu, Joon Byun, Frederik Bous, Juheon Lee
 - **Subjects:** Subjects:
Sound (cs.SD); Machine Learning (cs.LG); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2605.22083

 - **Pdf link:** https://arxiv.org/pdf/2605.22083

 - **Abstract**
 While flow-matching text-to-speech (TTS) achieves strong zero-shot speaker similarity and naturalness, it remains susceptible to content fidelity issues, particularly skip and repeat errors from imperfect alignment. We propose RobustSpeechFlow, a training strategy that improves alignment robustness by extending contrastive flow matching with length-preserving repeat and skip latent augmentations. Requiring no external aligners or preference data, our method directly penalizes realistic failure modes and readily integrates into existing pipelines. On Seed-TTS-eval, it reduces the word error rate (WER) from 1.44 to 1.38 using only 0.06B parameters. On our ZERO500 benchmark, it delivers consistent intelligibility improvements across diverse speaker and prosody conditions; at NFE=24, it reduces English character error rate (CER) from 0.48\% to 0.35\% and Korean CER from 0.81\% to 0.57\%. Audio samples: this https URL
#### Beyond Acoustic Emotion Recognition: Multimodal Pathos Analysis in Political Speech Using LLM-Based and Acoustic Emotion Models
 - **Authors:** Juergen Dietrich
 - **Subjects:** Subjects:
Artificial Intelligence (cs.AI); Computation and Language (cs.CL); Human-Computer Interaction (cs.HC); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2605.22732

 - **Pdf link:** https://arxiv.org/pdf/2605.22732

 - **Abstract**
 We investigate whether acoustic emotion recognition models can serve as proxies for the Pathos dimension in political speech analysis, as operationalised by the TRUST multi-agent large language model (LLM) pipeline. Using a Bundestag plenary speech by Felix Banaszak (51 segments, 245 s) as a case study, we compare three analysis modalities: (1) emotion2vec_plus_large, an acoustic speech emotion recognition (SER) model whose continuous Arousal and Valence values are derived via post-hoc Russell Circumplex projection; (2) Gemini 2.5 Flash, an LLM analysing the full speech audio together with its transcript in an open-ended, context-aware fashion; and (3) TRUST-Pathos scores from a three-advocate LLM supervisor ensemble. Spearman rank correlations reveal that Gemini Valence correlates strongly with TRUST-Pathos (rho = +0.664, p < 0.001), whereas emotion2vec Valence does not (rho = +0.097, p = 0.499). We further demonstrate, via a systematic quality evaluation of the Berlin Database of Emotional Speech (EMO-DB) using Gemini in an open-ended annotation paradigm, that standard SER benchmark corpora suffer from acted speech, cultural bias, and category incompatibility. Our results suggest that LLM-based multimodal analysis captures semantically defined political emotion substantially better than acoustic models alone, while acoustic features remain informative for low-level Arousal estimation. Future work will extend this approach to video-based analysis incorporating facial expression and gaze.
#### Plug-in Losses for Evidential Deep Learning: A Simplified Framework for Uncertainty Estimation that Includes the Softmax Classifier
 - **Authors:** Berk Hayta, Hannah Laus, Simon Mittermaier, Felix Krahmer
 - **Subjects:** Subjects:
Machine Learning (cs.LG); Audio and Speech Processing (eess.AS); Machine Learning (stat.ML)
 - **Arxiv link:** https://arxiv.org/abs/2605.22746

 - **Pdf link:** https://arxiv.org/pdf/2605.22746

 - **Abstract**
 Real-world sensor-based learning systems require uncertainty estimation that is both reliable and computationally efficient. Evidential Deep Learning (EDL) provides single-pass uncertainty estimation by modeling the class probabilities via Dirichlet distributions, where the Dirichlet parameters are predicted by a learned neural network mapping. However, this approach can lead to computational challenges, as Dirichlet expected objectives are more complex than standard supervised learning losses, complicating their analysis and implementation. We address this issue by approximating the objective of the first-order empirical risk minimization problem induced by EDL with a plug-in loss evaluated at the Dirichlet mean and show that, under mild assumptions, the approximation error decays with growing evidence for a broad class of loss functions, including mean-squared error and cross-entropy loss. As a special case, our analysis provides justification for the use of softmax in the context of uncertainty estimation, since under a particular evidence-to-Dirichlet mapping, our framework includes the standard softmax classifier. We validate the proposed simplified objectives on the Google Speech Commands dataset and show that they achieve predictive accuracy and selective prediction performance comparable to classical EDL, while being simpler to implement using standard deep learning losses and training pipelines. To the best of our knowledge, this empirical analysis is the first to obtain coverage-accuracy trade-offs for speech recognition tasks through EDL.


by Zyzzyva0381 (Windy). 


2026-05-22
