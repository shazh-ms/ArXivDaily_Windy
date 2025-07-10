# Showing new listings for Thursday, 10 July 2025
Auto update papers at about 2:30am UTC (10:30am Beijing time) every weekday.


阅读 `Usage.md`了解如何使用此repo实现个性化的Arxiv论文推送

See `Usage.md` for instructions on how to personalize the repo. 


Keyword list: ['text-to-speech', 'text to speech', 'tts', 'LLM-based', 'speech', 'voice']


Excluded: []


### Today: 9papers 
#### Pronunciation-Lexicon Free Training for Phoneme-based Crosslingual ASR via Joint Stochastic Approximation
 - **Authors:** Saierdaer Yusuyin, Te Ma, Hao Huang, Zhijian Ou
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Artificial Intelligence (cs.AI); Computation and Language (cs.CL)
 - **Arxiv link:** https://arxiv.org/abs/2507.06249

 - **Pdf link:** https://arxiv.org/pdf/2507.06249

 - **Abstract**
 Recently, pre-trained models with phonetic supervision have demonstrated their advantages for crosslingual speech recognition in data efficiency and information sharing across languages. However, a limitation is that a pronunciation lexicon is needed for such phoneme-based crosslingual speech recognition. In this study, we aim to eliminate the need for pronunciation lexicons and propose a latent variable model based method, with phonemes being treated as discrete latent variables. The new method consists of a speech-to-phoneme (S2P) model and a phoneme-to-grapheme (P2G) model, and a grapheme-to-phoneme (G2P) model is introduced as an auxiliary inference model. To jointly train the three models, we utilize the joint stochastic approximation (JSA) algorithm, which is a stochastic extension of the EM (expectation-maximization) algorithm and has demonstrated superior performance particularly in estimating discrete latent variable models. Based on the Whistle multilingual pre-trained S2P model, crosslingual experiments are conducted in Polish (130 h) and Indonesian (20 h). With only 10 minutes of phoneme supervision, the new method, JSA-SPG, achieves 5\% error rate reductions compared to the best crosslingual fine-tuning approach using subword or full phoneme supervision. Furthermore, it is found that in language domain adaptation (i.e., utilizing cross-domain text-only data), JSA-SPG outperforms the standard practice of language model fusion via the auxiliary support of the G2P model by 9% error rate reductions. To facilitate reproducibility and encourage further exploration in this field, we open-source the JSA-SPG training code and complete pipeline.
#### Open-Set Source Tracing of Audio Deepfake Systems
 - **Authors:** Nicholas Klein, Hemlata Tak, Elie Khoury
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2507.06470

 - **Pdf link:** https://arxiv.org/pdf/2507.06470

 - **Abstract**
 Existing research on source tracing of audio deepfake systems has focused primarily on the closed-set scenario, while studies that evaluate open-set performance are limited to a small number of unseen systems. Due to the large number of emerging audio deepfake systems, robust open-set source tracing is critical. We leverage the protocol of the Interspeech 2025 special session on source tracing to evaluate methods for improving open-set source tracing performance. We introduce a novel adaptation to the energy score for out-of-distribution (OOD) detection, softmax energy (SME). We find that replacing the typical temperature-scaled energy score with SME provides a relative average improvement of 31% in the standard FPR95 (false positive rate at true positive rate of 95%) measure. We further explore SME-guided training as well as copy synthesis, codec, and reverberation augmentations, yielding an FPR95 of 8.3%.
#### Training Strategies for Modality Dropout Resilient Multi-Modal Target Speaker Extraction
 - **Authors:** Srikanth Korse, Mohamed Elminshawi, Emanuel A. P. Habets, Srikanth Raj Chetupalli
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2507.06566

 - **Pdf link:** https://arxiv.org/pdf/2507.06566

 - **Abstract**
 The primary goal of multi-modal TSE (MTSE) is to extract a target speaker from a speech mixture using complementary information from different modalities, such as audio enrolment and visual feeds corresponding to the target speaker. MTSE systems are expected to perform well even when one of the modalities is unavailable. In practice, the systems often suffer from modality dominance, where one of the modalities outweighs the others, thereby limiting robustness. Our study investigates training strategies and the effect of architectural choices, particularly the normalization layers, in yielding a robust MTSE system in both non-causal and causal configurations. In particular, we propose the use of modality dropout training (MDT) as a superior strategy to standard and multi-task training (MTT) strategies. Experiments conducted on two-speaker mixtures from the LRS3 dataset show the MDT strategy to be effective irrespective of the employed normalization layer. In contrast, the models trained with the standard and MTT strategies are susceptible to modality dominance, and their performance depends on the chosen normalization layer. Additionally, we demonstrate that the system trained with MDT strategy is robust to using extracted speech as the enrollment signal, highlighting its potential applicability in scenarios where the target speaker is not enrolled.
#### Deep Feed-Forward Neural Network for Bangla Isolated Speech Recognition
 - **Authors:** Dipayan Bhadra, Mehrab Hosain, Fatema Alam
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2507.07068

 - **Pdf link:** https://arxiv.org/pdf/2507.07068

 - **Abstract**
 As the most important human-machine interfacing tool, an insignificant amount of work has been carried out on Bangla Speech Recognition compared to the English language. Motivated by this, in this work, the performance of speaker-independent isolated speech recognition systems has been implemented and analyzed using a dataset that is created containing both isolated Bangla and English spoken words. An approach using the Mel Frequency Cepstral Coefficient (MFCC) and Deep Feed-Forward Fully Connected Neural Network (DFFNN) of 7 layers as a classifier is proposed in this work to recognize isolated spoken words. This work shows 93.42% recognition accuracy which is better compared to most of the works done previously on Bangla speech recognition considering the number of classes and dataset size.
#### Incremental Averaging Method to Improve Graph-Based Time-Difference-of-Arrival Estimation
 - **Authors:** Klaus Brümann, Kouei Yamaoka, Nobutaka Ono, Simon Doclo
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Signal Processing (eess.SP)
 - **Arxiv link:** https://arxiv.org/abs/2507.07087

 - **Pdf link:** https://arxiv.org/pdf/2507.07087

 - **Abstract**
 Estimating the position of a speech source based on time-differences-of-arrival (TDOAs) is often adversely affected by background noise and reverberation. A popular method to estimate the TDOA between a microphone pair involves maximizing a generalized cross-correlation with phase transform (GCC-PHAT) function. Since the TDOAs across different microphone pairs satisfy consistency relations, generally only a small subset of microphone pairs are used for source position estimation. Although the set of microphone pairs is often determined based on a reference microphone, recently a more robust method has been proposed to determine the set of microphone pairs by computing the minimum spanning tree (MST) of a signal graph of GCC-PHAT function reliabilities. To reduce the influence of noise and reverberation on the TDOA estimation accuracy, in this paper we propose to compute the GCC-PHAT functions of the MST based on an average of multiple cross-power spectral densities (CPSDs) using an incremental method. In each step of the method, we increase the number of CPSDs over which we average by considering CPSDs computed indirectly via other microphones from previous steps. Using signals recorded in a noisy and reverberant laboratory with an array of spatially distributed microphones, the performance of the proposed method is evaluated in terms of TDOA estimation error and 2D source position estimation error. Experimental results for different source and microphone configurations and three reverberation conditions show that the proposed method considering multiple CPSDs improves the TDOA estimation and source position estimation accuracy compared to the reference microphone- and MST-based methods that rely on a single CPSD as well as steered-response power-based source position estimation.
#### Super Kawaii Vocalics: Amplifying the "Cute" Factor in Computer Voice
 - **Authors:** Yuto Mandai, Katie Seaborn, Tomoyasu Nakano, Xin Sun, Yijia Wang, Jun Kato
 - **Subjects:** Subjects:
Human-Computer Interaction (cs.HC); Artificial Intelligence (cs.AI); Computation and Language (cs.CL); Computers and Society (cs.CY); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2507.06235

 - **Pdf link:** https://arxiv.org/pdf/2507.06235

 - **Abstract**
 "Kawaii" is the Japanese concept of cute, which carries sociocultural connotations related to social identities and emotional responses. Yet, virtually all work to date has focused on the visual side of kawaii, including in studies of computer agents and social robots. In pursuit of formalizing the new science of kawaii vocalics, we explored what elements of voice relate to kawaii and how they might be manipulated, manually and automatically. We conducted a four-phase study (grand N = 512) with two varieties of computer voices: text-to-speech (TTS) and game character voices. We found kawaii "sweet spots" through manipulation of fundamental and formant frequencies, but only for certain voices and to a certain extent. Findings also suggest a ceiling effect for the kawaii vocalics of certain voices. We offer empirical validation of the preliminary kawaii vocalics model and an elementary method for manipulating kawaii perceptions of computer voice.
#### STARS: A Unified Framework for Singing Transcription, Alignment, and Refined Style Annotation
 - **Authors:** Wenxiang Guo, Yu Zhang, Changhao Pan, Zhiyuan Zhu, Ruiqi Li, Zhetao Chen, Wenhao Xu, Fei Wu, Zhou Zhao
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2507.06670

 - **Pdf link:** https://arxiv.org/pdf/2507.06670

 - **Abstract**
 Recent breakthroughs in singing voice synthesis (SVS) have heightened the demand for high-quality annotated datasets, yet manual annotation remains prohibitively labor-intensive and resource-intensive. Existing automatic singing annotation (ASA) methods, however, primarily tackle isolated aspects of the annotation pipeline. To address this fundamental challenge, we present STARS, which is, to our knowledge, the first unified framework that simultaneously addresses singing transcription, alignment, and refined style annotation. Our framework delivers comprehensive multi-level annotations encompassing: (1) precise phoneme-audio alignment, (2) robust note transcription and temporal localization, (3) expressive vocal technique identification, and (4) global stylistic characterization including emotion and pace. The proposed architecture employs hierarchical acoustic feature processing across frame, word, phoneme, note, and sentence levels. The novel non-autoregressive local acoustic encoders enable structured hierarchical representation learning. Experimental validation confirms the framework's superior performance across multiple evaluation dimensions compared to existing annotation approaches. Furthermore, applications in SVS training demonstrate that models utilizing STARS-annotated data achieve significantly enhanced perceptual naturalness and precise style control. This work not only overcomes critical scalability challenges in the creation of singing datasets but also pioneers new methodologies for controllable singing voice synthesis. Audio samples are available at this https URL.
#### Revealing the Hidden Temporal Structure of HubertSoft Embeddings based on the Russian Phonetic Corpus
 - **Authors:** Anastasia Ananeva, Anton Tomilov, Marina Volkova
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2507.06794

 - **Pdf link:** https://arxiv.org/pdf/2507.06794

 - **Abstract**
 Self-supervised learning (SSL) models such as Wav2Vec 2.0 and HuBERT have shown remarkable success in extracting phonetic information from raw audio without labelled data. While prior work has demonstrated that SSL embeddings encode phonetic features at the frame level, it remains unclear whether these models preserve temporal structure, specifically, whether embeddings at phoneme boundaries reflect the identity and order of adjacent phonemes. This study investigates the extent to which boundary-sensitive embeddings from HubertSoft, a soft-clustering variant of HuBERT, encode phoneme transitions. Using the CORPRES Russian speech corpus, we labelled 20 ms embedding windows with triplets of phonemes corresponding to their start, centre, and end segments. A neural network was trained to predict these positions separately, and multiple evaluation metrics, such as ordered, unordered accuracy and a flexible centre accuracy, were used to assess temporal sensitivity. Results show that embeddings extracted at phoneme boundaries capture both phoneme identity and temporal order, with especially high accuracy at segment boundaries. Confusion patterns further suggest that the model encodes articulatory detail and coarticulatory effects. These findings contribute to our understanding of the internal structure of SSL speech representations and their potential for phonological analysis and fine-grained transcription tasks.
#### A Novel Hybrid Deep Learning Technique for Speech Emotion Detection using Feature Engineering
 - **Authors:** Shahana Yasmin Chowdhury, Bithi Banik, Md Tamjidul Hoque, Shreya Banerjee
 - **Subjects:** Subjects:
Sound (cs.SD); Artificial Intelligence (cs.AI); Machine Learning (cs.LG); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2507.07046

 - **Pdf link:** https://arxiv.org/pdf/2507.07046

 - **Abstract**
 Nowadays, speech emotion recognition (SER) plays a vital role in the field of human-computer interaction (HCI) and the evolution of artificial intelligence (AI). Our proposed DCRF-BiLSTM model is used to recognize seven emotions: neutral, happy, sad, angry, fear, disgust, and surprise, which are trained on five datasets: RAVDESS (R), TESS (T), SAVEE (S), EmoDB (E), and Crema-D (C). The model achieves high accuracy on individual datasets, including 97.83% on RAVDESS, 97.02% on SAVEE, 95.10% for CREMA-D, and a perfect 100% on both TESS and EMO-DB. For the combined (R+T+S) datasets, it achieves 98.82% accuracy, outperforming previously reported results. To our knowledge, no existing study has evaluated a single SER model across all five benchmark datasets (i.e., R+T+S+C+E) simultaneously. In our work, we introduce this comprehensive combination and achieve a remarkable overall accuracy of 93.76%. These results confirm the robustness and generalizability of our DCRF-BiLSTM framework across diverse datasets.


by Zyzzyva0381 (Windy). 


2025-07-10
