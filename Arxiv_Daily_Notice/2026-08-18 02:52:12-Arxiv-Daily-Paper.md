# Showing new listings for Monday, 17 August 2026
Auto update papers at about 2:30am UTC (10:30am Beijing time) every weekday.


阅读 `Usage.md`了解如何使用此repo实现个性化的Arxiv论文推送

See `Usage.md` for instructions on how to personalize the repo. 


Keyword list: ['text-to-speech', 'text to speech', 'tts', 'LLM-based', 'speech', 'voice']


Excluded: []


### Today: 4papers 
#### VoiceDesigner: Text-to-Voice Generation and Editing via Unified Diffusion Modeling and Data Augmentation
 - **Authors:** Jiarui Hai, Karan Thakkar, Ke Chen, Yunyun Wang, Jiaqi Su, Rithesh Kumar, Mounya Elhilali, Zeyu Jin
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Machine Learning (cs.LG)
 - **Arxiv link:** https://arxiv.org/abs/2608.13613

 - **Pdf link:** https://arxiv.org/pdf/2608.13613

 - **Abstract**
 Recent breakthroughs in generative models have made text-to-voice generation (TTV) possible, enabling the synthesis of speech directly from textual voice descriptions. However, existing systems face two key challenges. First, they struggle to generate a diverse range of voices, spanning real-world human speakers and fictional characters. Second, they lack robust and flexible voice editing capabilities, such as voice cloning and the ability to modify attributes like emotion and tone. In this paper, we propose VoiceDesigner, a unified framework for text-to-voice generation and editing that supports diverse and controllable voice design. To tackle the above challenges, we propose solutions from two perspectives. First, we develop a hybrid data pipeline that leverages digital signal processing techniques and speech generation models to construct a diverse voice dataset covering both real-world and fictional voices. Second, we introduce a diffusion transformer with architectural improvements to better handle complex conditioning and enhance multi-task performance, enabling unified voice generation and editing. Through subjective and objective evaluations, VoiceDesigner achieves superior prompt alignment with both voice descriptions and editing instructions, while maintaining competitive perceptual quality and voice usability compared to state-of-the-art TTV models.
#### Trajectory Dynamics in Self-Supervised Learning Latent Space for Audio Deepfake Detection
 - **Authors:** Tomás Andrade Weber
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Machine Learning (cs.LG); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2608.13817

 - **Pdf link:** https://arxiv.org/pdf/2608.13817

 - **Abstract**
 Human speech production is constrained by physiology, giving rise to characteristic temporal structure on acoustic signals. We hypothesise that these constraints manifest as structured trajectory dynamics in the latent space of Self-Supervised Learning (SSL) models, and that synthetic speech violates them detectably. To test this hypothesis, we train a causal Long Short-Term Memory (LSTM) next-frame predictor on bonafide speech only (Stage 1), using the deepfake-specialised SSL backbone Wav2Vec2-Large-AntiDeepfake, and compare against a static global-average-pooling baseline using identical features, thus isolating the contribution of temporal modelling. A supervised Stage 2, which trains a Multi-Layer Perceptron on the frozen LSTM internal states using labelled data, is included to characterise the role of spoof supervision. Our system achieves competitive or state-of-the-art performance across six benchmarks: ASVspoof 2019/2021, Codecfake, In-the-Wild, MLAAD-EN, and Deepfake-Eval-2024, including best published EER on ASVspoof 2021 (0.75\%) and, notably, Stage 1 trained on bonafide speech only surpasses the published supervised baseline from the same backbone on DE2024 (30.35\%). On near-domain benchmarks, static and dynamic approaches perform comparably. On harder cross-corpus benchmarks with diverse synthesis methods, trajectory dynamics provide substantial gains, confirming that temporal physiological constraints carry detection signal beyond utterance-level statistics.
#### VoiceChat-TTS: A Low-Latency Continuous Speech Synthesis Model for Interactive Agents
 - **Authors:** Edresson Casanova, Jaehyeon Kim, Mariana Graterol Fuenmayor, Shehzeen Hussain, Viacheslav Klimkov, Valentin Mendelev, Mikyas Desta, Paarth Neekhara, Piotr Zelasko, Chen Chen, Elena Rastorgueva, Ke Hu, Ankita Pasad, Xuesong Yang, Aya Alja'fari, Rajarshi Roy, Rohan Badlani, Jason Roche, Jason Li, Zhehuai Chen
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Computation and Language (cs.CL)
 - **Arxiv link:** https://arxiv.org/abs/2608.13831

 - **Pdf link:** https://arxiv.org/pdf/2608.13831

 - **Abstract**
 Spoken dialogue is a natural form of human--computer interaction, yet most speech language models remain limited to turn-based operation and lack real-time adaptability, such as user barge-in. Recent duplex speech-to-speech and speech-to-text models reduce latency by replacing multi-stage pipelines, but often compromise speech quality because accurate ASR, interruption handling, and high-fidelity synthesis must be optimized jointly. We propose VoiceChat-TTS, a low-latency, continuous, and streamable text-to-speech model for interactive agents. VoiceChat-TTS is driven directly by LLM text-token streams, supports explicit interruption via control tokens, and produces silence when no textual input is available. The model enables always-on, responsive speech generation while preserving modularity and high speech quality, and it supports mid-utterance interruptions without resetting the KV cache.
#### StreamHear: Domain-Adapted Pseudo-Labeling for Semi-Supervised Streaming Speech Recognition
 - **Authors:** Zefang Liu, Chenyang Zhu, Sangwoo Cho, Xujun Peng, Shi-Xiong Zhang, Sambit Sahu
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2608.13717

 - **Pdf link:** https://arxiv.org/pdf/2608.13717

 - **Abstract**
 Streaming automatic speech recognition (ASR) underperforms on domain-shifted target audio, where labeled in-domain data is costly to prepare while unlabeled audio is abundant. We present StreamHear, a semi-supervised pipeline that adapts a pretrained streaming student by fine-tuning an offline transducer teacher on the labeled training set, generating pseudo-labels on the unlabeled portion, and fine-tuning the student on the mixture. We further introduce a prior-regularized dynamic-programming realignment step that fixes chunk-level word placement using an ASR-hypothesis anchor. Across four datasets spanning financial calls, prepared read speech, and phone-quality dialogue, StreamHear consistently outperforms supervised student fine-tuning and narrows the gap to the offline teacher.


by Zyzzyva0381 (Windy). 


2026-08-18
