# Showing new listings for Thursday, 29 January 2026
Auto update papers at about 2:30am UTC (10:30am Beijing time) every weekday.


阅读 `Usage.md`了解如何使用此repo实现个性化的Arxiv论文推送

See `Usage.md` for instructions on how to personalize the repo. 


Keyword list: ['text-to-speech', 'text to speech', 'tts', 'LLM-based', 'speech', 'voice']


Excluded: []


### Today: 11papers 
#### RIR-Mega-Speech: A Reverberant Speech Corpus with Comprehensive Acoustic Metadata and Reproducible Evaluation
 - **Authors:** Mandip Goswami
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Computation and Language (cs.CL); Sound (cs.SD); Signal Processing (eess.SP)
 - **Arxiv link:** https://arxiv.org/abs/2601.19949

 - **Pdf link:** https://arxiv.org/pdf/2601.19949

 - **Abstract**
 Despite decades of research on reverberant speech, comparing methods remains difficult because most corpora lack per-file acoustic annotations or provide limited documentation for reproduction. We present RIR-Mega-Speech, a corpus of approximately 117.5 hours created by convolving LibriSpeech utterances with roughly 5,000 simulated room impulse responses from the RIR-Mega collection. Every file includes RT60, direct-to-reverberant ratio (DRR), and clarity index ($C_{50}$) computed from the source RIR using clearly defined, reproducible procedures. We also provide scripts to rebuild the dataset and reproduce all evaluation results. Using Whisper small on 1,500 paired utterances, we measure 5.20% WER (95% CI: 4.69--5.78) on clean speech and 7.70% (7.04--8.35) on reverberant versions, corresponding to a paired increase of 2.50 percentage points (2.06--2.98). This represents a 48% relative degradation. WER increases monotonically with RT60 and decreases with DRR, consistent with prior perceptual studies. While the core finding that reverberation harms recognition is well established, we aim to provide the community with a standardized resource where acoustic conditions are transparent and results can be verified independently. The repository includes one-command rebuild instructions for both Windows and Linux environments.
#### VoxPrivacy: A Benchmark for Evaluating Interactional Privacy of Speech Language Models
 - **Authors:** Yuxiang Wang, Hongyu Liu, Dekun Chen, Xueyao Zhang, Zhizheng Wu
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Artificial Intelligence (cs.AI); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2601.19956

 - **Pdf link:** https://arxiv.org/pdf/2601.19956

 - **Abstract**
 As Speech Language Models (SLMs) transition from personal devices to shared, multi-user environments such as smart homes, a new challenge emerges: the model is expected to distinguish between users to manage information flow appropriately. Without this capability, an SLM could reveal one user's confidential schedule to another, a privacy failure we term interactional privacy. Thus, the ability to generate speaker-aware responses becomes essential for SLM safe deployment. Current SLM benchmarks test dialogue ability but overlook speaker identity. Multi-speaker benchmarks check who said what without assessing whether SLMs adapt their responses. Privacy benchmarks focus on globally sensitive data (e.g., bank passwords) while neglecting contextual privacy-sensitive information (e.g., a user's private appointment). To address this gap, we introduce VoxPrivacy, the first benchmark designed to evaluate interactional privacy in SLMs. VoxPrivacy spans three tiers of increasing difficulty, from following direct secrecy commands to proactively protecting privacy. Our evaluation of nine SLMs on a 32-hour bilingual dataset reveals a widespread vulnerability: most open-source models perform close to random chance (around 50% accuracy) on conditional privacy decisions, while even strong closed-source systems fall short on proactive privacy inference. We further validate these findings on Real-VoxPrivacy, a human-recorded subset, confirming that failures observed on synthetic data persist in real speech. Finally, we demonstrate a viable path forward: by fine-tuning on a new 4,000-hour training set, we improve privacy-preserving abilities while maintaining robustness. To support future work, we release the VoxPrivacy benchmark, the large-scale training set, and the fine-tuned model to foster the development of safer and more context-aware SLMs.
#### Do we really need Self-Attention for Streaming Automatic Speech Recognition?
 - **Authors:** Youness Dkhissi (LIUM), Valentin Vielzeuf, Elys Allesiardo, Anthony Larcher (LIUM)
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Artificial Intelligence (cs.AI); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2601.19960

 - **Pdf link:** https://arxiv.org/pdf/2601.19960

 - **Abstract**
 Transformer-based architectures are the most used architectures in many deep learning fields like Natural Language Processing, Computer Vision or Speech processing. It may encourage the direct use of Transformers in the constrained tasks, without questioning whether it will yield the same benefits as in standard tasks.  Given specific constraints, it is essential to evaluate the relevance of transformer models. This work questions the suitability of transformers for specific domains. We argue that the high computational requirements and latency issues associated with these models do not align well with streaming applications. Our study promotes the search for alternative strategies to improve efficiency without sacrificing performance.  In light of this observation, our paper critically examines the usefulness of transformer architecture in such constrained environments. As a first attempt, we show that the computational cost for Streaming Automatic Speech Recognition (ASR) can be reduced using deformable convolution instead of Self-Attention. Furthermore, we show that Self-Attention mechanisms can be entirely removed and not replaced, without observing significant degradation in the Word Error Rate.
#### T-Mimi: A Transformer-based Mimi Decoder for Real-Time On-Phone TTS
 - **Authors:** Haibin Wu, Bach Viet Do, Naveen Suda, Julian Chan, Madhavan C R, Gene-Ping Yang, Yi-Chiao Wu, Naoyuki Kanda, Yossef Adi, Xin Lei, Yue Liu, Florian Metze, Yuzong Liu
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2601.20094

 - **Pdf link:** https://arxiv.org/pdf/2601.20094

 - **Abstract**
 Neural audio codecs provide promising acoustic features for speech synthesis, with representative streaming codecs like Mimi providing high-quality acoustic features for real-time Text-to-Speech (TTS) applications. However, Mimi's decoder, which employs a hybrid transformer and convolution architecture, introduces significant latency bottlenecks on edge devices due to the the compute intensive nature of deconvolution layers which are not friendly for mobile-CPUs, such as the most representative framework XNNPACK. This paper introduces T-Mimi, a novel modification of the Mimi codec decoder that replaces its convolutional components with a purely transformer-based decoder, inspired by the TS3-Codec architecture. This change dramatically reduces on-device TTS latency from 42.1ms to just 4.4ms. Furthermore, we conduct quantization aware training and derive a crucial finding: the final two transformer layers and the concluding linear layers of the decoder, which are close to the waveform, are highly sensitive to quantization and must be preserved at full precision to maintain audio quality.
#### ASR for Affective Speech: Investigating Impact of Emotion and Speech Generative Strategy
 - **Authors:** Ya-Tse Wu, Chi-Chun Lee
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2601.20319

 - **Pdf link:** https://arxiv.org/pdf/2601.20319

 - **Abstract**
 This work investigates how emotional speech and generative strategies affect ASR performance. We analyze speech synthesized from three emotional TTS models and find that substitution errors dominate, with emotional expressiveness varying across models. Based on these insights, we introduce two generative strategies: one using transcription correctness and another using emotional salience, to construct fine-tuning subsets. Results show consistent WER improvements on real emotional datasets without noticeable degradation on clean LibriSpeech utterances. The combined strategy achieves the strongest gains, particularly for expressive speech. These findings highlight the importance of targeted augmentation for building emotion-aware ASR systems.
#### Erasing Your Voice Before It's Heard: Training-free Speaker Unlearning for Zero-shot Text-to-Speech
 - **Authors:** Myungjin Lee, Eunji Shin, Jiyoung Lee
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2601.20481

 - **Pdf link:** https://arxiv.org/pdf/2601.20481

 - **Abstract**
 Modern zero-shot text-to-speech (TTS) models offer unprecedented expressivity but also pose serious crime risks, as they can synthesize voices of individuals who never consented. In this context, speaker unlearning aims to prevent the generation of specific speaker identities upon request. Existing approaches, reliant on retraining, are costly and limited to speakers seen in the training set. We present TruS, a training-free speaker unlearning framework that shifts the paradigm from data deletion to inference-time control. TruS steers identity-specific hidden activations to suppress target speakers while preserving other attributes (e.g., prosody and emotion). Experimental results show that TruS effectively prevents voice generation on both seen and unseen opt-out speakers, establishing a scalable safeguard for speech synthesis. The demo and code are available on this http URL.
#### Decoding Speech Envelopes from Electroencephalogram with a Contrastive Pearson Correlation Coefficient Loss
 - **Authors:** Yayun Liang, Yuanming Zhang, Fei Chen, Jing Lu, Zhibin Lin
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2601.20542

 - **Pdf link:** https://arxiv.org/pdf/2601.20542

 - **Abstract**
 Recent advances in reconstructing speech envelopes from Electroencephalogram (EEG) signals have enabled continuous auditory attention decoding (AAD) in multi-speaker environments. Most Deep Neural Network (DNN)-based envelope reconstruction models are trained to maximize the Pearson correlation coefficients (PCC) between the attended envelope and the reconstructed envelope (attended PCC). While the difference between the attended PCC and the unattended PCC plays an essential role in auditory attention decoding, existing methods often focus on maximizing the attended PCC. We therefore propose a contrastive PCC loss which represents the difference between the attended PCC and the unattended PCC. The proposed approach is evaluated on three public EEG AAD datasets using four DNN architectures. Across many settings, the proposed objective improves envelope separability and AAD accuracy, while also revealing dataset- and architecture-dependent failure cases.
#### LTS-VoiceAgent: A Listen-Think-Speak Framework for Efficient Streaming Voice Interaction via Semantic Triggering and Incremental Reasoning
 - **Authors:** Wenhao Zou, Yuwei Miao, Zhanyu Ma, Jun Xu, Jiuchong Gao, Jinghua Hao, Renqing He, Jingwen Xu
 - **Subjects:** Subjects:
Sound (cs.SD); Artificial Intelligence (cs.AI); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2601.19952

 - **Pdf link:** https://arxiv.org/pdf/2601.19952

 - **Abstract**
 Real-time voice agents face a dilemma: end-to-end models often lack deep reasoning, while cascaded pipelines incur high latency by executing ASR, LLM reasoning, and TTS strictly in sequence, unlike human conversation where listeners often start thinking before the speaker finishes. Since cascaded architectures remain the dominant choice for complex tasks, existing cascaded streaming strategies attempt to reduce this latency via mechanical segmentation (e.g., fixed chunks, VAD-based splitting) or speculative generation, but they frequently either break semantic units or waste computation on predictions that must be rolled back. To address these challenges, we propose LTS-VoiceAgent, a Listen-Think-Speak framework that explicitly separates when to think from how to reason incrementally. It features a Dynamic Semantic Trigger to detect meaningful prefixes, and a Dual-Role Stream Orchestrator that coordinates a background Thinker (for state maintenance) and a foreground Speaker (for speculative solving). This parallel design enables "thinking while speaking" without blocking responses. We also introduce a Pause-and-Repair benchmark containing natural disfluencies to stress-test streaming robustness. Experiments across VERA, Spoken-MQA, BigBenchAudio, and our benchmark show that LTS-VoiceAgent achieves a stronger accuracy-latency-efficiency trade-off than serial cascaded baselines and existing streaming strategies.
#### Mind the Shift: Using Delta SSL Embeddings to Enhance Child ASR
 - **Authors:** Zilai Wang, Natarajan Balaji Shankar, Kaiyuan Zhang, Zihan Wang, Abeer Alwan
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2601.20142

 - **Pdf link:** https://arxiv.org/pdf/2601.20142

 - **Abstract**
 Self-supervised learning (SSL) models have achieved impressive results across many speech tasks, yet child automatic speech recognition (ASR) remains challenging due to limited data and pretraining domain mismatch. Fine-tuning SSL models on child speech induces shifts in the representation space. We hypothesize that delta SSL embeddings, defined as the differences between embeddings from a finetuned model and those from its pretrained counterpart, encode task-specific information that complements finetuned features from another SSL model. We evaluate multiple fusion strategies on the MyST childrens corpus using different models. Results show that delta embedding fusion with WavLM yields up to a 10 percent relative WER reduction for HuBERT and a 4.4 percent reduction for W2V2, compared to finetuned embedding fusion. Notably, fusing WavLM with delta W2V2 embeddings achieves a WER of 9.64, setting a new state of the art among SSL models on the MyST corpus. These findings demonstrate the effectiveness of delta embeddings and highlight feature fusion as a promising direction for advancing child ASR.
#### MiLorE-SSL: Scaling Multilingual Capabilities in Self-Supervised Models without Forgetting
 - **Authors:** Jing Xu, Minglin Wu, Xueyuan Chen, Xixin Wu, Helen Meng
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2601.20300

 - **Pdf link:** https://arxiv.org/pdf/2601.20300

 - **Abstract**
 Self-supervised learning (SSL) has greatly advanced speech representation learning, but multilingual SSL models remain constrained to languages encountered during pretraining. Retraining from scratch to incorporate new languages is computationally expensive, while sequential training without migitation strategies often leads to catastrophic forgetting. To address this, we propose MiLorE-SSL, a lightweight framework that combines LoRA modules with a soft mixture-of-experts (MoE) mechanism for efficient continual multilingual training. LoRA provides efficient low-rank adaptation, while soft MoE promotes flexible expert sharing across languages, reducing cross-lingual interference. To further mitigate forgetting, we introduce limited replay data from existing languages, avoiding reliance on large historical corpora. Experiments on ML-SUPERB demonstrate that MiLorE-SSL achieves strong performance in new languages and improves the ability in existing ones with only 2.14% trainable parameters.
#### Audio Deepfake Detection in the Age of Advanced Text-to-Speech models
 - **Authors:** Robin Singh, Aditya Yogesh Nair, Fabio Palumbo, Florian Barbaro, Anna Dyka, Lohith Rachakonda
 - **Subjects:** Subjects:
Sound (cs.SD); Artificial Intelligence (cs.AI); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2601.20510

 - **Pdf link:** https://arxiv.org/pdf/2601.20510

 - **Abstract**
 Recent advances in Text-to-Speech (TTS) systems have substantially increased the realism of synthetic speech, raising new challenges for audio deepfake detection. This work presents a comparative evaluation of three state-of-the-art TTS models--Dia2, Maya1, and MeloTTS--representing streaming, LLM-based, and non-autoregressive architectures. A corpus of 12,000 synthetic audio samples was generated using the Daily-Dialog dataset and evaluated against four detection frameworks, including semantic, structural, and signal-level approaches. The results reveal significant variability in detector performance across generative mechanisms: models effective against one TTS architecture may fail against others, particularly LLM-based synthesis. In contrast, a multi-view detection approach combining complementary analysis levels demonstrates robust performance across all evaluated models. These findings highlight the limitations of single-paradigm detectors and emphasize the necessity of integrated detection strategies to address the evolving landscape of audio deepfake threats.


by Zyzzyva0381 (Windy). 


2026-01-29
