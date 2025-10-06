# Showing new listings for Monday, 6 October 2025
Auto update papers at about 2:30am UTC (10:30am Beijing time) every weekday.


阅读 `Usage.md`了解如何使用此repo实现个性化的Arxiv论文推送

See `Usage.md` for instructions on how to personalize the repo. 


Keyword list: ['text-to-speech', 'text to speech', 'tts', 'LLM-based', 'speech', 'voice']


Excluded: []


### Today: 7papers 
#### WEE-Therapy: A Mixture of Weak Encoders Framework for Psychological Counseling Dialogue Analysis
 - **Authors:** Yongqi Kang, Yong Zhao
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Computation and Language (cs.CL); Machine Learning (cs.LG); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2510.02320

 - **Pdf link:** https://arxiv.org/pdf/2510.02320

 - **Abstract**
 The advancement of computational psychology requires AI tools capable of deeply understanding counseling dialogues. Existing audio language models (AudioLLMs) often rely on single speech encoders pre-trained on general data, struggling to capture domain-specific features like complex emotions and professional techniques. To address this, we propose WEE-Therapy, a multi-task AudioLLM incorporating a Weak Encoder Ensemble (WEE) mechanism. This supplements a powerful base encoder with a pool of lightweight, specialized encoders. A novel dual-routing strategy combines stable, data-independent domain knowledge with dynamic, data-dependent expert selection. Evaluated on emotion recognition, technique classification, risk detection, and summarization, WEE-Therapy achieves significant performance gains across all tasks with minimal parameter overhead, demonstrating strong potential for AI-assisted clinical analysis.
#### SpeechCT-CLIP: Distilling Text-Image Knowledge to Speech for Voice-Native Multimodal CT Analysis
 - **Authors:** Lukas Buess, Jan Geier, David Bani-Harouni, Chantal Pellegrini, Matthias Keicher, Paula Andrea Perez-Toro, Nassir Navab, Andreas Maier, Tomas Arias-Vergara
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Computation and Language (cs.CL)
 - **Arxiv link:** https://arxiv.org/abs/2510.02322

 - **Pdf link:** https://arxiv.org/pdf/2510.02322

 - **Abstract**
 Spoken communication plays a central role in clinical workflows. In radiology, for example, most reports are created through dictation. Yet, nearly all medical AI systems rely exclusively on written text. In this work, we address this gap by exploring the feasibility of learning visual-language representations directly from spoken radiology reports. Specifically, we synthesize a large-scale dataset (Speech-RATE) of spoken radiology reports and train SpeechCT-CLIP, a contrastive model that aligns speech and 3D CT volumes in a shared representation space. While naive speech-based models underperform compared to text-trained counterparts, we show that knowledge distillation from a pretrained text-image CLIP model effectively transfers semantic alignment capabilities from text to speech, substantially narrowing this gap. Experiments demonstrate improved zero-shot classification F1 from 0.623 to 0.705, recovering 88% of the performance difference, and strong retrieval results without requiring text at inference. These findings highlight speech as a practical alternative to text in multimodal pretraining and open the door to voice-driven diagnostic support tools in clinical practice.
#### When Voice Matters: Evidence of Gender Disparity in Positional Bias of SpeechLLMs
 - **Authors:** Shree Harsha Bokkahalli Satish, Gustav Eje Henter, Éva Székely
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2510.02398

 - **Pdf link:** https://arxiv.org/pdf/2510.02398

 - **Abstract**
 The rapid development of SpeechLLM-based conversational AI systems has created a need for robustly benchmarking these efforts, including aspects of fairness and bias. At present, such benchmarks typically rely on multiple choice question answering (MCQA). In this paper, we present the first token-level probabilistic evaluation and response-based study of several issues affecting the use of MCQA in SpeechLLM benchmarking: 1) we examine how model temperature and prompt design affect gender and positional bias on an MCQA gender-bias benchmark; 2) we examine how these biases are affected by the gender of the input voice; and 3) we study to what extent observed trends carry over to a second gender-bias benchmark. Our results show that concerns about positional bias from the text domain are equally valid in the speech domain. We also find the effect to be stronger for female voices than for male voices. To our knowledge, this is the first study to isolate positional bias effects in SpeechLLM-based gender-bias benchmarks. We conclude that current MCQA benchmarks do not account for speech-based bias and alternative strategies are needed to ensure fairness towards all users.
#### STSM-FiLM: A FiLM-Conditioned Neural Architecture for Time-Scale Modification of Speech
 - **Authors:** Dyah A. M. G. Wisnu, Ryandhimas E. Zezario, Stefano Rini, Fo-Rui Li, Yan-Tsung Peng, Hsin-Min Wang, Yu Tsao
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2510.02672

 - **Pdf link:** https://arxiv.org/pdf/2510.02672

 - **Abstract**
 Time-Scale Modification (TSM) of speech aims to alter the playback rate of audio without changing its pitch. While classical methods like Waveform Similarity-based Overlap-Add (WSOLA) provide strong baselines, they often introduce artifacts under non-stationary or extreme stretching conditions. We propose STSM-FILM - a fully neural architecture that incorporates Feature-Wise Linear Modulation (FiLM) to condition the model on a continuous speed factor. By supervising the network using WSOLA-generated outputs, STSM-FILM learns to mimic alignment and synthesis behaviors while benefiting from representations learned through deep learning. We explore four encoder--decoder variants: STFT-HiFiGAN, WavLM-HiFiGAN, Whisper-HiFiGAN, and EnCodec, and demonstrate that STSM-FILM is capable of producing perceptually consistent outputs across a wide range of time-scaling factors. Overall, our results demonstrate the potential of FiLM-based conditioning to improve the generalization and flexibility of neural TSM models.
#### Evaluation of preprocessing pipelines in the creation of in-the-wild TTS datasets
 - **Authors:** Matías Di Bernardo, Emmanuel Misley, Ignacio Correa, Mateo García Iacovelli, Simón Mellino, Gala Lucía Gonzalez Barrios
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2510.03111

 - **Pdf link:** https://arxiv.org/pdf/2510.03111

 - **Abstract**
 This work introduces a reproducible, metric-driven methodology to evaluate preprocessing pipelines for in-the-wild TTS corpora generation. We apply a custom low-cost pipeline to the first in-the-wild Argentine Spanish collection and compare 24 pipeline configurations combining different denoising and quality filtering variants. Evaluation relies on complementary objective measures (PESQ, SI-SDR, SNR), acoustic descriptors (T30, C50), and speech-preservation metrics (F0-STD, MCD). Results expose trade-offs between dataset size, signal quality, and voice preservation; where denoising variants with permissive filtering provide the best overall compromise for our testbed. The proposed methodology allows selecting pipeline configurations without training TTS models for each subset, accelerating and reducing the cost of preprocessing development for low-resource settings.
#### KAME: Tandem Architecture for Enhancing Knowledge in Real-Time Speech-to-Speech Conversational AI
 - **Authors:** So Kuroki, Yotaro Kubo, Takuya Akiba, Yujin Tang
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Artificial Intelligence (cs.AI); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2510.02327

 - **Pdf link:** https://arxiv.org/pdf/2510.02327

 - **Abstract**
 Real-time speech-to-speech (S2S) models excel at generating natural, low-latency conversational responses but often lack deep knowledge and semantic understanding. Conversely, cascaded systems combining automatic speech recognition, a text-based Large Language Model (LLM), and text-to-speech synthesis offer superior knowledge representation at the cost of high latency, which disrupts the flow of natural interaction. This paper introduces a novel hybrid architecture that bridges the gap between these two paradigms. Our framework processes user speech through an S2S transformer for immediate responsiveness while concurrently relaying the query to a powerful back-end LLM. The LLM's text-based response is then injected in real time to guide the S2S model's speech generation, effectively infusing its output with rich knowledge without the full latency penalty of a cascaded system. We evaluated our method using a speech-synthesized variant of the MT-Bench benchmark that consists of multi-turn question-answering sessions. The results demonstrate that our system substantially outperforms a baseline S2S model in response correctness, approaching that of a cascaded system, while maintaining a latency on par with the baseline.
#### WavInWav: Time-domain Speech Hiding via Invertible Neural Network
 - **Authors:** Wei Fan, Kejiang Chen, Xiangkun Wang, Weiming Zhang, Nenghai Yu
 - **Subjects:** Subjects:
Sound (cs.SD); Artificial Intelligence (cs.AI); Cryptography and Security (cs.CR); Machine Learning (cs.LG); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2510.02915

 - **Pdf link:** https://arxiv.org/pdf/2510.02915

 - **Abstract**
 Data hiding is essential for secure communication across digital media, and recent advances in Deep Neural Networks (DNNs) provide enhanced methods for embedding secret information effectively. However, previous audio hiding methods often result in unsatisfactory quality when recovering secret audio, due to their inherent limitations in the modeling of time-frequency relationships. In this paper, we explore these limitations and introduce a new DNN-based approach. We use a flow-based invertible neural network to establish a direct link between stego audio, cover audio, and secret audio, enhancing the reversibility of embedding and extracting messages. To address common issues from time-frequency transformations that degrade secret audio quality during recovery, we implement a time-frequency loss on the time-domain signal. This approach not only retains the benefits of time-frequency constraints but also enhances the reversibility of message recovery, which is vital for practical applications. We also add an encryption technique to protect the hidden data from unauthorized access. Experimental results on the VCTK and LibriSpeech datasets demonstrate that our method outperforms previous approaches in terms of subjective and objective metrics and exhibits robustness to various types of noise, suggesting its utility in targeted secure communication scenarios.


by Zyzzyva0381 (Windy). 


2025-10-06
