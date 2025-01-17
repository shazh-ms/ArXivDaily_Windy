# Showing new listings for Friday, 17 January 2025
Auto update papers at about 2:30am UTC (10:30am Beijing time) every weekday.


阅读 `Usage.md`了解如何使用此repo实现个性化的Arxiv论文推送

See `Usage.md` for instructions on how to personalize the repo. 


Keyword list: ['acoustic', 'reinforcement learning', 'reverb', 'meta', 'separate', 'reconstruction', 'noise', 'enhance', 'localization', 'speech']


Excluded: []


### Today: 8papers 
#### persoDA: Personalized Data Augmentation forPersonalized ASR
 - **Authors:** Pablo Peso Parada, Spyros Fontalis, Md Asif Jalal, Karthikeyan Saravanan, Anastasios Drosou, Mete Ozay, Gil Ho Lee, Jungin Lee, Seokyeong Jung
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2501.09113

 - **Pdf link:** https://arxiv.org/pdf/2501.09113

 - **Abstract**
 Data augmentation (DA) is ubiquitously used in training of Automatic Speech Recognition (ASR) models. DA offers increased data variability, robustness and generalization against different acoustic distortions. Recently, personalization of ASR models on mobile devices has been shown to improve Word Error Rate (WER). This paper evaluates data augmentation in this context and proposes persoDA; a DA method driven by user's data utilized to personalize ASR. persoDA aims to augment training with data specifically tuned towards acoustic characteristics of the end-user, as opposed to standard augmentation based on Multi-Condition Training (MCT) that applies random reverberation and noises. Our evaluation with an ASR conformer-based baseline trained on Librispeech and personalized for VOICES shows that persoDA achieves a 13.9% relative WER reduction over using standard data augmentation (using random noise & reverberation). Furthermore, persoDA shows 16% to 20% faster convergence over MCT.
#### Beyond Speaker Identity: Text Guided Target Speech Extraction
 - **Authors:** Mingyue Huo, Abhinav Jain, Cong Phuoc Huynh, Fanjie Kong, Pichao Wang, Zhu Liu, Vimal Bhat
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2501.09169

 - **Pdf link:** https://arxiv.org/pdf/2501.09169

 - **Abstract**
 Target Speech Extraction (TSE) traditionally relies on explicit clues about the speaker's identity like enrollment audio, face images, or videos, which may not always be available. In this paper, we propose a text-guided TSE model StyleTSE that uses natural language descriptions of speaking style in addition to the audio clue to extract the desired speech from a given mixture. Our model integrates a speech separation network adapted from SepFormer with a bi-modality clue network that flexibly processes both audio and text clues. To train and evaluate our model, we introduce a new dataset TextrolMix with speech mixtures and natural language descriptions. Experimental results demonstrate that our method effectively separates speech based not only on who is speaking, but also on how they are speaking, enhancing TSE in scenarios where traditional audio clues are absent. Demos are at: this https URL
#### Quantum-Enhanced Transformers for Robust Acoustic Scene Classification in IoT Environments
 - **Authors:** Minh K. Quan, Mayuri Wijayasundara, Sujeeva Setunge, Pubudu N. Pathirana
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Artificial Intelligence (cs.AI); Machine Learning (cs.LG); Performance (cs.PF); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2501.09394

 - **Pdf link:** https://arxiv.org/pdf/2501.09394

 - **Abstract**
 The proliferation of Internet of Things (IoT) devices equipped with acoustic sensors necessitates robust acoustic scene classification (ASC) capabilities, even in noisy and data-limited environments. Traditional machine learning methods often struggle to generalize effectively under such conditions. To address this, we introduce Q-ASC, a novel Quantum-Inspired Acoustic Scene Classifier that leverages the power of quantum-inspired transformers. By integrating quantum concepts like superposition and entanglement, Q-ASC achieves superior feature learning and enhanced noise resilience compared to classical models. Furthermore, we introduce a Quantum Variational Autoencoder (QVAE) based data augmentation technique to mitigate the challenge of limited labeled data in IoT deployments. Extensive evaluations on the Tampere University of Technology (TUT) Acoustic Scenes 2016 benchmark dataset demonstrate that Q-ASC achieves remarkable accuracy between 68.3% and 88.5% under challenging conditions, outperforming state-of-the-art methods by over 5% in the best case. This research paves the way for deploying intelligent acoustic sensing in IoT networks, with potential applications in smart homes, industrial monitoring, and environmental surveillance, even in adverse acoustic environments.
#### A Non-autoregressive Model for Joint STT and TTS
 - **Authors:** Vishal Sunder, Brian Kingsbury, George Saon, Samuel Thomas, Slava Shechtman Hagai Aronowitz, Eric Fosler-Lussier, Luis Lastras
 - **Subjects:** Subjects:
Sound (cs.SD); Artificial Intelligence (cs.AI); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2501.09104

 - **Pdf link:** https://arxiv.org/pdf/2501.09104

 - **Abstract**
 In this paper, we take a step towards jointly modeling automatic speech recognition (STT) and speech synthesis (TTS) in a fully non-autoregressive way. We develop a novel multimodal framework capable of handling the speech and text modalities as input either individually or together. The proposed model can also be trained with unpaired speech or text data owing to its multimodal nature. We further propose an iterative refinement strategy to improve the STT and TTS performance of our model such that the partial hypothesis at the output can be fed back to the input of our model, thus iteratively improving both STT and TTS predictions. We show that our joint model can effectively perform both STT and TTS tasks, outperforming the STT-specific baseline in all tasks and performing competitively with the TTS-specific baseline across a wide range of evaluation metrics.
#### Delayed Fusion: Integrating Large Language Models into First-Pass Decoding in End-to-end Speech Recognition
 - **Authors:** Takaaki Hori, Martin Kocour, Adnan Haider, Erik McDermott, Xiaodan Zhuang
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2501.09258

 - **Pdf link:** https://arxiv.org/pdf/2501.09258

 - **Abstract**
 This paper presents an efficient decoding approach for end-to-end automatic speech recognition (E2E-ASR) with large language models (LLMs). Although shallow fusion is the most common approach to incorporate language models into E2E-ASR decoding, we face two practical problems with LLMs. (1) LLM inference is computationally costly. (2) There may be a vocabulary mismatch between the ASR model and the LLM. To resolve this mismatch, we need to retrain the ASR model and/or the LLM, which is at best time-consuming and in many cases not feasible. We propose "delayed fusion," which applies LLM scores to ASR hypotheses with a delay during decoding and enables easier use of pre-trained LLMs in ASR tasks. This method can reduce not only the number of hypotheses scored by the LLM but also the number of LLM inference calls. It also allows re-tokenizion of ASR hypotheses during decoding if ASR and LLM employ different tokenizations. We demonstrate that delayed fusion provides improved decoding speed and accuracy compared to shallow fusion and N-best rescoring using the LibriHeavy ASR corpus and three public LLMs, OpenLLaMA 3B & 7B and Mistral 7B.
#### LAVCap: LLM-based Audio-Visual Captioning using Optimal Transport
 - **Authors:** Kyeongha Rho, Hyeongkeun Lee, Valentio Iverson, Joon Son Chung
 - **Subjects:** Subjects:
Multimedia (cs.MM); Artificial Intelligence (cs.AI); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2501.09291

 - **Pdf link:** https://arxiv.org/pdf/2501.09291

 - **Abstract**
 Automated audio captioning is a task that generates textual descriptions for audio content, and recent studies have explored using visual information to enhance captioning quality. However, current methods often fail to effectively fuse audio and visual data, missing important semantic cues from each modality. To address this, we introduce LAVCap, a large language model (LLM)-based audio-visual captioning framework that effectively integrates visual information with audio to improve audio captioning performance. LAVCap employs an optimal transport-based alignment loss to bridge the modality gap between audio and visual features, enabling more effective semantic extraction. Additionally, we propose an optimal transport attention module that enhances audio-visual fusion using an optimal transport assignment map. Combined with the optimal training strategy, experimental results demonstrate that each component of our framework is effective. LAVCap outperforms existing state-of-the-art methods on the AudioCaps dataset, without relying on large datasets or post-processing. Code is available at this https URL.
#### Multimodal Marvels of Deep Learning in Medical Diagnosis: A Comprehensive Review of COVID-19 Detection
 - **Authors:** Md Shofiqul Islama, Khondokar Fida Hasanc, Hasibul Hossain Shajeebd, Humayan Kabir Ranae, Md Saifur Rahmand, Md Munirul Hasanb, AKM Azadf, Ibrahim Abdullahg, Mohammad Ali Moni
 - **Subjects:** Subjects:
Machine Learning (cs.LG); Sound (cs.SD); Audio and Speech Processing (eess.AS); Image and Video Processing (eess.IV)
 - **Arxiv link:** https://arxiv.org/abs/2501.09506

 - **Pdf link:** https://arxiv.org/pdf/2501.09506

 - **Abstract**
 This study presents a comprehensive review of the potential of multimodal deep learning (DL) in medical diagnosis, using COVID-19 as a case example. Motivated by the success of artificial intelligence applications during the COVID-19 pandemic, this research aims to uncover the capabilities of DL in disease screening, prediction, and classification, and to derive insights that enhance the resilience, sustainability, and inclusiveness of science, technology, and innovation systems. Adopting a systematic approach, we investigate the fundamental methodologies, data sources, preprocessing steps, and challenges encountered in various studies and implementations. We explore the architecture of deep learning models, emphasising their data-specific structures and underlying algorithms. Subsequently, we compare different deep learning strategies utilised in COVID-19 analysis, evaluating them based on methodology, data, performance, and prerequisites for future research. By examining diverse data types and diagnostic modalities, this research contributes to scientific understanding and knowledge of the multimodal application of DL and its effectiveness in diagnosis. We have implemented and analysed 11 deep learning models using COVID-19 image, text, and speech (ie, cough) data. Our analysis revealed that the MobileNet model achieved the highest accuracy of 99.97% for COVID-19 image data and 93.73% for speech data (i.e., cough). However, the BiGRU model demonstrated superior performance in COVID-19 text classification with an accuracy of 99.89%. The broader implications of this research suggest potential benefits for other domains and disciplines that could leverage deep learning techniques for image, text, and speech analysis.
#### Metric Learning with Progressive Self-Distillation for Audio-Visual Embedding Learning
 - **Authors:** Donghuo Zeng, Kazushi Ikeda
 - **Subjects:** Subjects:
Sound (cs.SD); Artificial Intelligence (cs.AI); Computer Vision and Pattern Recognition (cs.CV); Information Retrieval (cs.IR); Multimedia (cs.MM); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2501.09608

 - **Pdf link:** https://arxiv.org/pdf/2501.09608

 - **Abstract**
 Metric learning projects samples into an embedded space, where similarities and dissimilarities are quantified based on their learned representations. However, existing methods often rely on label-guided representation learning, where representations of different modalities, such as audio and visual data, are aligned based on annotated labels. This approach tends to underutilize latent complex features and potential relationships inherent in the distributions of audio and visual data that are not directly tied to the labels, resulting in suboptimal performance in audio-visual embedding learning. To address this issue, we propose a novel architecture that integrates cross-modal triplet loss with progressive self-distillation. Our method enhances representation learning by leveraging inherent distributions and dynamically refining soft audio-visual alignments -- probabilistic alignments between audio and visual data that capture the inherent relationships beyond explicit labels. Specifically, the model distills audio-visual distribution-based knowledge from annotated labels in a subset of each batch. This self-distilled knowledge is used t


by Zyzzyva0381 (Windy). 


2025-01-17
