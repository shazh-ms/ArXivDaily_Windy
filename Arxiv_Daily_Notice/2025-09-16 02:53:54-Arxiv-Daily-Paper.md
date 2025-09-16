# Showing new listings for Tuesday, 16 September 2025
Auto update papers at about 2:30am UTC (10:30am Beijing time) every weekday.


阅读 `Usage.md`了解如何使用此repo实现个性化的Arxiv论文推送

See `Usage.md` for instructions on how to personalize the repo. 


Keyword list: ['text-to-speech', 'text to speech', 'tts', 'LLM-based', 'speech', 'voice']


Excluded: []


### Today: 5papers 
#### Length-Aware Rotary Position Embedding for Text-Speech Alignment
 - **Authors:** Hyeongju Kim, Juheon Lee, Jinhyeok Yang, Jacob Morton
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Artificial Intelligence (cs.AI); Computation and Language (cs.CL); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2509.11084

 - **Pdf link:** https://arxiv.org/pdf/2509.11084

 - **Abstract**
 Many recent text-to-speech (TTS) systems are built on transformer architectures and employ cross-attention mechanisms for text-speech alignment. Within these systems, rotary position embedding (RoPE) is commonly used to encode positional information in text and speech representations. In this work, we introduce length-aware RoPE (LARoPE), a simple yet effective extension of RoPE that improves text-speech alignment. Unlike RoPE, which relies on absolute indices, LARoPE computes relative distances between query and key positions using length-normalized indices. Experimental results show that LARoPE consistently outperforms RoPE, offering faster loss convergence, more accurate text-speech alignment, and higher overall TTS quality. Furthermore, LARoPE demonstrates greater resilience to variations in utterance duration and maintains stable performance in extended speech generation up to 30 seconds, whereas RoPE suffers from notable degradation. Notably, our method achieves a state-of-the-art word error rate on a standard zero-shot TTS benchmark.
#### EEND-SAA: Enrollment-Less Main Speaker Voice Activity Detection Using Self-Attention Attractors
 - **Authors:** Wen-Yung Wu, Pei-Chin Hsieh, Tai-Shih Chi
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2509.11957

 - **Pdf link:** https://arxiv.org/pdf/2509.11957

 - **Abstract**
 Voice activity detection (VAD) is essential in speech-based systems, but traditional methods detect only speech presence without identifying speakers. Target-speaker VAD (TS-VAD) extends this by detecting the speech of a known speaker using a short enrollment utterance, but this assumption fails in open-domain scenarios such as meetings or customer service calls, where the main speaker is unknown. We propose EEND-SAA, an enrollment-less, streaming-compatible framework for main-speaker VAD, which identifies the primary speaker without prior knowledge. Unlike TS-VAD, our method determines the main speaker as the one who talks more steadily and clearly, based on speech continuity and volume. We build our model on EEND using two self-attention attractors in a Transformer and apply causal masking for real-time use. Experiments on multi-speaker LibriSpeech mixtures show that EEND-SAA reduces main-speaker DER from 6.63% to 3.61% and improves F1 from 0.9667 to 0.9818 over the SA-EEND baseline, achieving state-of-the-art performance under conditions involving speaker overlap and noise.
#### Multimodal Deep Learning for ATCO Command Lifecycle Modeling and Workload Prediction
 - **Authors:** Kaizhen Tan
 - **Subjects:** Subjects:
Machine Learning (cs.LG); Artificial Intelligence (cs.AI); Computer Vision and Pattern Recognition (cs.CV); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2509.10522

 - **Pdf link:** https://arxiv.org/pdf/2509.10522

 - **Abstract**
 Air traffic controllers (ATCOs) issue high-intensity voice commands in dense airspace, where accurate workload modeling is critical for safety and efficiency. This paper proposes a multimodal deep learning framework that integrates structured data, trajectory sequences, and image features to estimate two key parameters in the ATCO command lifecycle: the time offset between a command and the resulting aircraft maneuver, and the command duration. A high-quality dataset was constructed, with maneuver points detected using sliding window and histogram-based methods. A CNN-Transformer ensemble model was developed for accurate, generalizable, and interpretable predictions. By linking trajectories to voice commands, this work offers the first model of its kind to support intelligent command generation and provides practical value for workload assessment, staffing, and scheduling.
#### FuseCodec: Semantic-Contextual Fusion and Supervision for Neural Codecs
 - **Authors:** Md Mubtasim Ahasan, Rafat Hasan Khan, Tasnim Mohiuddin, Aman Chadha, Tariq Iqbal, M Ashraful Amin, Amin Ahsan Ali, Md Mofijul Islam, A K M Mahbubur Rahman
 - **Subjects:** Subjects:
Sound (cs.SD); Artificial Intelligence (cs.AI); Computation and Language (cs.CL); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2509.11425

 - **Pdf link:** https://arxiv.org/pdf/2509.11425

 - **Abstract**
 Speech tokenization enables discrete representation and facilitates speech language modeling. However, existing neural codecs capture low-level acoustic features, overlooking the semantic and contextual cues inherent to human speech. While recent efforts introduced semantic representations from self-supervised speech models or incorporated contextual representations from pre-trained language models, challenges remain in aligning and unifying the semantic and contextual representations. We introduce FuseCodec, which unifies acoustic, semantic, and contextual representations through strong cross-modal alignment and globally informed supervision. We propose three complementary techniques: (i) Latent Representation Fusion, integrating semantic and contextual features directly into the encoder latent space for robust and unified representation learning; (ii) Global Semantic-Contextual Supervision, supervising discrete tokens with globally pooled and broadcasted representations to enhance temporal consistency and cross-modal alignment; and (iii) Temporally Aligned Contextual Supervision, strengthening alignment by dynamically matching contextual and speech tokens within a local window for fine-grained token-level supervision. We further introduce FuseCodec-TTS, demonstrating our methodology's applicability to zero-shot speech synthesis. Empirically, FuseCodec achieves state-of-the-art performance in LibriSpeech, surpassing EnCodec, SpeechTokenizer, and DAC in transcription accuracy, perceptual quality, intelligibility, and speaker similarity. Results highlight the effectiveness of contextually and semantically guided tokenization for speech tokenization and downstream tasks. Code and pretrained models are available at this https URL.
#### Room acoustics affect communicative success in hybrid meeting spaces: a pilot study
 - **Authors:** Robert Einig, Stefan Janscha, Jonas Schuster, Julian Koch, Martin Hagmueller, Barbara Schuppler
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2509.11709

 - **Pdf link:** https://arxiv.org/pdf/2509.11709

 - **Abstract**
 Since the COVID-19 pandemic in 2020, universities and companies have increasingly integrated hybrid features into their meeting spaces, or even created dedicated rooms for this purpose. While the importance of a fast and stable internet connection is often prioritized, the acoustic design of seminar rooms is frequently overlooked. Poor acoustics, particularly excessive reverberation, can lead to issues such as misunderstandings, reduced speech intelligibility or cognitive and vocal fatigue. This pilot study investigates whether room acoustic interventions in a seminar room at Graz University of Technology support better communication in hybrid meetings. For this purpose, we recorded two groups of persons twice, once before and once after improving the acoustics of the room. Our findings -- despite not reaching statistical significance due to the small sample size - indicate clearly that our spatial interventions improve communicative success in hybrid meetings. To make the paper accessible also for readers from the speech communication community, we explain room acoustics background, relevant for the interpretation of our results.


by Zyzzyva0381 (Windy). 


2025-09-16
