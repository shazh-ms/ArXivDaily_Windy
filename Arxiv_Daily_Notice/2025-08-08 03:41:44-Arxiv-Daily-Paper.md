# Showing new listings for Friday, 8 August 2025
Auto update papers at about 2:30am UTC (10:30am Beijing time) every weekday.


阅读 `Usage.md`了解如何使用此repo实现个性化的Arxiv论文推送

See `Usage.md` for instructions on how to personalize the repo. 


Keyword list: ['text-to-speech', 'text to speech', 'tts', 'LLM-based', 'speech', 'voice']


Excluded: []


### Today: 12papers 
#### Keyword Spotting with Hyper-Matched Filters for Small Footprint Devices
 - **Authors:** Yael Segal-Feldman, Ann R. Bradlow, Matthew Goldrick, Joseph Keshet
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Machine Learning (cs.LG); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2508.04857

 - **Pdf link:** https://arxiv.org/pdf/2508.04857

 - **Abstract**
 Open-vocabulary keyword spotting (KWS) refers to the task of detecting words or terms within speech recordings, regardless of whether they were included in the training data. This paper introduces an open-vocabulary keyword spotting model with state-of-the-art detection accuracy for small-footprint devices. The model is composed of a speech encoder, a target keyword encoder, and a detection network. The speech encoder is either a tiny Whisper or a tiny Conformer. The target keyword encoder is implemented as a hyper-network that takes the desired keyword as a character string and generates a unique set of weights for a convolutional layer, which can be considered as a keyword-specific matched filter. The detection network uses the matched-filter weights to perform a keyword-specific convolution, which guides the cross-attention mechanism of a Perceiver module in determining whether the target term appears in the recording. The results indicate that our system achieves state-of-the-art detection performance and generalizes effectively to out-of-domain conditions, including second-language (L2) speech. Notably, our smallest model, with just 4.2 million parameters, matches or outperforms models that are several times larger, demonstrating both efficiency and robustness.
#### REF-VC: Robust, Expressive and Fast Zero-Shot Voice Conversion with Diffusion Transformers
 - **Authors:** Yuepeng Jiang, Ziqian Ning, Shuai Wang, Chengjia Wang, Mengxiao Bi, Pengcheng Zhu, Lei Xie, Zhonghua Fu
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2508.04996

 - **Pdf link:** https://arxiv.org/pdf/2508.04996

 - **Abstract**
 In real-world voice conversion applications, environmental noise in source speech and user demands for expressive output pose critical challenges. Traditional ASR-based methods ensure noise robustness but suppress prosody, while SSL-based models improve expressiveness but suffer from timbre leakage and noise sensitivity. This paper proposes REF-VC, a noise-robust expressive voice conversion system. Key innovations include: (1) A random erasing strategy to mitigate the information redundancy inherent in SSL feature, enhancing noise robustness and expressiveness; (2) Implicit alignment inspired by E2TTS to suppress non-essential feature reconstruction; (3) Integration of Shortcut Models to accelerate flow matching inference, significantly reducing to 4 steps. Experimental results demonstrate that our model outperforms baselines such as Seed-VC in zero-shot scenarios on the noisy set, while also performing comparably to Seed-VC on the clean set. In addition, REF-VC can be compatible with singing voice conversion within one model.
#### MOVER: Combining Multiple Meeting Recognition Systems
 - **Authors:** Naoyuki Kamo, Tsubasa Ochiai, Marc Delcroix, Tomohiro Nakatani
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2508.05055

 - **Pdf link:** https://arxiv.org/pdf/2508.05055

 - **Abstract**
 In this paper, we propose Meeting recognizer Output Voting Error Reduction (MOVER), a novel system combination method for meeting recognition tasks. Although there are methods to combine the output of diarization (e.g., DOVER) or automatic speech recognition (ASR) systems (e.g., ROVER), MOVER is the first approach that can combine the outputs of meeting recognition systems that differ in terms of both diarization and ASR. MOVER combines hypotheses with different time intervals and speaker labels through a five-stage process that includes speaker alignment, segment grouping, word and timing combination, etc. Experimental results on the CHiME-8 DASR task and the multi-channel track of the NOTSOFAR-1 task demonstrate that MOVER can successfully combine multiple meeting recognition systems with diverse diarization and recognition outputs, achieving relative tcpWER improvements of 9.55 % and 8.51 % over the state-of-the-art systems for both tasks.
#### Fairness in Dysarthric Speech Synthesis: Understanding Intrinsic Bias in Dysarthric Speech Cloning using F5-TTS
 - **Authors:** Anuprabha M, Krishna Gurugubelli, Anil Kumar Vuppala
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Artificial Intelligence (cs.AI)
 - **Arxiv link:** https://arxiv.org/abs/2508.05102

 - **Pdf link:** https://arxiv.org/pdf/2508.05102

 - **Abstract**
 Dysarthric speech poses significant challenges in developing assistive technologies, primarily due to the limited availability of data. Recent advances in neural speech synthesis, especially zero-shot voice cloning, facilitate synthetic speech generation for data augmentation; however, they may introduce biases towards dysarthric speech. In this paper, we investigate the effectiveness of state-of-the-art F5-TTS in cloning dysarthric speech using TORGO dataset, focusing on intelligibility, speaker similarity, and prosody preservation. We also analyze potential biases using fairness metrics like Disparate Impact and Parity Difference to assess disparities across dysarthric severity levels. Results show that F5-TTS exhibits a strong bias toward speech intelligibility over speaker and prosody preservation in dysarthric speech synthesis. Insights from this study can help integrate fairness-aware dysarthric speech synthesis, fostering the advancement of more inclusive speech technologies.
#### Speech LLMs in Low-Resource Scenarios: Data Volume Requirements and the Impact of Pretraining on High-Resource Languages
 - **Authors:** Seraphina Fong, Marco Matassoni, Alessio Brutti
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Artificial Intelligence (cs.AI); Computation and Language (cs.CL)
 - **Arxiv link:** https://arxiv.org/abs/2508.05149

 - **Pdf link:** https://arxiv.org/pdf/2508.05149

 - **Abstract**
 Large language models (LLMs) have demonstrated potential in handling spoken inputs for high-resource languages, reaching state-of-the-art performance in various tasks. However, their applicability is still less explored in low-resource settings. This work investigates the use of Speech LLMs for low-resource Automatic Speech Recognition using the SLAM-ASR framework, where a trainable lightweight projector connects a speech encoder and a LLM. Firstly, we assess training data volume requirements to match Whisper-only performance, re-emphasizing the challenges of limited data. Secondly, we show that leveraging mono- or multilingual projectors pretrained on high-resource languages reduces the impact of data scarcity, especially with small training sets. Using multilingual LLMs (EuroLLM, Salamandra) with whisper-large-v3-turbo, we evaluate performance on several public benchmarks, providing insights for future research on optimizing Speech LLMs for low-resource languages and multilinguality.
#### Privacy Disclosure of Similarity in Speech and Language Processing
 - **Authors:** Tom Bäckström, Mohammad Hassan Vali, My Nguyen, Silas Rech
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2508.05250

 - **Pdf link:** https://arxiv.org/pdf/2508.05250

 - **Abstract**
 Speaker, author, and other biometric identification applications often compare a sample's similarity to a database of templates to determine the identity. Given that data may be noisy and similarity measures can be inaccurate, such a comparison may not reliably identify the true identity as the most similar. Still, even the similarity rank based on an inaccurate similarity measure can disclose private information about the true identity. We propose a methodology for quantifying the privacy disclosure of such a similarity rank by estimating its probability distribution. It is based on determining the histogram of the similarity rank of the true speaker, or when data is scarce, modeling the histogram with the beta-binomial distribution. We express the disclosure in terms of entropy (bits), such that the disclosure from independent features are additive. Our experiments demonstrate that all tested speaker and author characterizations contain personally identifying information (PII) that can aid in identification, with embeddings from speaker recognition algorithms containing the most information, followed by phone embeddings, linguistic embeddings, and fundamental frequency. Our initial experiments show that the disclosure of PII increases with the length of test samples, but it is bounded by the length of database templates. The provided metric, similarity rank disclosure, provides a way to compare the disclosure of PII between biometric features and merge them to aid identification. It can thus aid in the holistic evaluation of threats to privacy in speech and other biometric technologies.
#### Investigation of Speech and Noise Latent Representations in Single-channel VAE-based Speech Enhancement
 - **Authors:** Jiatong Li, Simon Doclo
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2508.05293

 - **Pdf link:** https://arxiv.org/pdf/2508.05293

 - **Abstract**
 Recently, a variational autoencoder (VAE)-based single-channel speech enhancement system using Bayesian permutation training has been proposed, which uses two pretrained VAEs to obtain latent representations for speech and noise. Based on these pretrained VAEs, a noisy VAE learns to generate speech and noise latent representations from noisy speech for speech enhancement. Modifying the pretrained VAE loss terms affects the pretrained speech and noise latent representations. In this paper, we investigate how these different representations affect speech enhancement performance. Experiments on the DNS3, WSJ0-QUT, and VoiceBank-DEMAND datasets show that a latent space where speech and noise representations are clearly separated significantly improves performance over standard VAEs, which produce overlapping speech and noise representations.
#### Toward Low-Latency End-to-End Voice Agents for Telecommunications Using Streaming ASR, Quantized LLMs, and Real-Time TTS
 - **Authors:** Vignesh Ethiraj, Ashwath David, Sidhanth Menon, Divya Vijay
 - **Subjects:** Subjects:
Sound (cs.SD); Artificial Intelligence (cs.AI); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2508.04721

 - **Pdf link:** https://arxiv.org/pdf/2508.04721

 - **Abstract**
 We introduce a low-latency telecom AI voice agent pipeline for real-time, interactive telecommunications use, enabling advanced voice AI for call center automation, intelligent IVR (Interactive Voice Response), and AI-driven customer support. The solution is built for telecom, combining four specialized models by NetoAI: TSLAM, a 4-bit quantized Telecom-Specific Large Language Model (LLM); T-VEC, a Telecom-Specific Embedding Model; TTE, a Telecom-Specific Automatic Speech Recognition (ASR) model; and T-Synth, a Telecom-Specific Text-to-Speech (TTS) model. These models enable highly responsive, domain-adapted voice AI agents supporting knowledge-grounded spoken interactions with low latency. The pipeline integrates streaming ASR (TTE), conversational intelligence (TSLAM), retrieval augmented generation (RAG) over telecom documents, and real-time TTS (T-Synth), setting a new benchmark for telecom voice assistants. To evaluate the system, we built a dataset of 500 human-recorded telecom questions from RFCs, simulating real telecom agent queries. This framework allows analysis of latency, domain relevance, and real-time performance across the stack. Results show that TSLAM, TTE, and T-Synth deliver real-time factors (RTF) below 1.0, supporting enterprise, low-latency telecom deployments. These AI agents -- powered by TSLAM, TTE, and T-Synth -- provide a foundation for next-generation telecom AI, enabling automated customer support, diagnostics, and more.
#### Pitch Accent Detection improves Pretrained Automatic Speech Recognition
 - **Authors:** David Sasu, Natalie Schluter
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2508.04814

 - **Pdf link:** https://arxiv.org/pdf/2508.04814

 - **Abstract**
 We show the performance of Automatic Speech Recognition (ASR) systems that use semi-supervised speech representations can be boosted by a complimentary pitch accent detection module, by introducing a joint ASR and pitch accent detection model. The pitch accent detection component of our model achieves a significant improvement on the state-of-the-art for the task, closing the gap in F1-score by 41%. Additionally, the ASR performance in joint training decreases WER by 28.3% on LibriSpeech, under limited resource fine-tuning. With these results, we show the importance of extending pretrained speech models to retain or re-learn important prosodic cues such as pitch accent.
#### REINA: Regularized Entropy Information-Based Loss for Efficient Simultaneous Speech Translation
 - **Authors:** Nameer Hirschkind, Joseph Liu, Mahesh Kumar Nandwana, Xiao Yu
 - **Subjects:** Subjects:
Machine Learning (cs.LG); Computation and Language (cs.CL); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2508.04946

 - **Pdf link:** https://arxiv.org/pdf/2508.04946

 - **Abstract**
 Simultaneous Speech Translation (SimulST) systems stream in audio while simultaneously emitting translated text or speech. Such systems face the significant challenge of balancing translation quality and latency. We introduce a strategy to optimize this tradeoff: wait for more input only if you gain information by doing so. Based on this strategy, we present Regularized Entropy INformation Adaptation (REINA), a novel loss to train an adaptive policy using an existing non-streaming translation model. We derive REINA from information theory principles and show that REINA helps push the reported Pareto frontier of the latency/quality tradeoff over prior works. Utilizing REINA, we train a SimulST model on French, Spanish and German, both from and into English. Training on only open source or synthetically generated data, we achieve state-of-the-art (SOTA) streaming results for models of comparable size. We also introduce a metric for streaming efficiency, quantitatively showing REINA improves the latency/quality trade-off by as much as 21% compared to prior approaches, normalized against non-streaming baseline BLEU scores.
#### A Scalable Pipeline for Enabling Non-Verbal Speech Generation and Understanding
 - **Authors:** Runchuan Ye, Yixuan Zhou, Renjie Yu, Zijian Lin, Kehan Li, Xiang Li, Xin Liu, Guoyang Zeng, Zhiyong Wu
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2508.05385

 - **Pdf link:** https://arxiv.org/pdf/2508.05385

 - **Abstract**
 Human spoken communication involves not only lexical content but also non-verbal vocalizations (NVs) such as laughter, sighs, and coughs, which convey emotions, intentions, and social signals. However, most existing speech systems focus solely on verbal content and lack the ability to understand and generate such non-verbal cues, reducing the emotional intelligence and communicative richness of spoken interfaces. In this work, we introduce $\textbf{NonVerbalSpeech-38K}$, a large and diverse dataset for non-verbal speech generation and understanding, collected from real-world media and annotated using an automatic pipeline. The dataset contains 38,718 samples (about 131 hours) with 10 categories of non-verbal cues, such as laughter, sniff, and throat clearing. We further validate the dataset by fine-tuning state-of-the-art models, including F5-TTS and Qwen2-Audio, demonstrating its effectiveness in non-verbal speech generation and understanding tasks. Our contributions are threefold: (1) We propose a practical pipeline for building natural and diverse non-verbal speech datasets; (2) We release a large-scale dataset to advance research on non-verbal speech generation and understanding; (3) We validate the dataset's effectiveness by demonstrating improvements in both non-verbal speech synthesis and captioning, thereby facilitating richer human-computer interaction.
#### SPGISpeech 2.0: Transcribed multi-speaker financial audio for speaker-tagged transcription
 - **Authors:** Raymond Grossman, Taejin Park, Kunal Dhawan, Andrew Titus, Sophia Zhi, Yulia Shchadilova, Weiqing Wang, Jagadeesh Balam, Boris Ginsburg
 - **Subjects:** Subjects:
Sound (cs.SD); Computation and Language (cs.CL); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2508.05554

 - **Pdf link:** https://arxiv.org/pdf/2508.05554

 - **Abstract**
 We introduce SPGISpeech 2.0, a dataset suitable for speaker-tagged transcription in the financial domain. SPGISpeech 2.0 improves the diversity of applicable modeling tasks while maintaining the core characteristic of the original SPGISpeech dataset: audio snippets and their corresponding fully formatted text transcriptions, usable for end-to-end automatic speech recognition (ASR). SPGISpeech 2.0 consists of 3,780 additional hours of professionally transcribed earnings calls. Furthermore, the dataset contains call and speaker information for each audio snippet facilitating multi-talker ASR. We validate the utility of SPGISpeech 2.0 through improvements in speaker-tagged ASR performance of popular speech recognition models after fine-tuning on SPGISpeech 2.0. Released free for non-commercial use, we expect SPGISpeech 2.0 to foster advancements in speech recognition technologies and inspire a wide range of research applications.


by Zyzzyva0381 (Windy). 


2025-08-08
