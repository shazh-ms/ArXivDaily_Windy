# Showing new listings for Tuesday, 14 July 2026
Auto update papers at about 2:30am UTC (10:30am Beijing time) every weekday.


阅读 `Usage.md`了解如何使用此repo实现个性化的Arxiv论文推送

See `Usage.md` for instructions on how to personalize the repo. 


Keyword list: ['text-to-speech', 'text to speech', 'tts', 'LLM-based', 'speech', 'voice']


Excluded: []


### Today: 14papers 
#### CoFi-Lite: Pushing the Limits of Ultra-Lightweight Speech Enhancement
 - **Authors:** Leyan Yang, Dahan Wang, Xiaobin Rong, Jiadong Zhao, Jing Lu
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2607.10142

 - **Pdf link:** https://arxiv.org/pdf/2607.10142

 - **Abstract**
 Ultra-lightweight models are essential for the deployment of deep learning-based speech enhancement algorithms on edge devices. Although recent approaches have achieved a certain balance between computational complexity and performance, pushing the complexity limits further demands more sophisticated designs. In this letter, we propose CoFi-Lite, a highly efficient model that decouples spectral modeling into coarse- and fine-grained streams. By leveraging two parallel and symmetric encoder-decoder paths, it simultaneously extracts full-band envelopes and low-frequency details for complementary enhancement. In addition, a novel Cross-Path Fusion (CPF) module is introduced to bridge the distinct paths, facilitating efficient feature interaction. Remarkably, CoFi-Lite requires extremely low computational resources, featuring only 12.87M MACs/s and 83.12k parameters. Experimental results demonstrate that our proposed model outperforms the ultra-lightweight baseline GTCRN while requiring only 40.26% of its computational complexity. Its scaled-up variant also delivers performance on par with that of the SOTA ultra-lightweight model AdaptCRN alongside a 19.34% reduction in computational cost. Audio examples are available at this https URL.
#### Evaluating SSL and ViViT Architectures for Cross-Corpus Audio MOS Prediction via LODO Validation
 - **Authors:** Mustafa Ozan Duman, Ahmet Emir Dirik
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2607.10146

 - **Pdf link:** https://arxiv.org/pdf/2607.10146

 - **Abstract**
 Automatic Mean Opinion Score (MOS) prediction is essential for evaluating large-scale synthetic speech and audio enhancement systems, yet models frequently struggle with domain shift. This study presents a comprehensive benchmarking of three architectural frameworks: Frozen Self-Supervised Learning (SSL-FRZ), Fine-Tuned SSL (SSL-FT), and a Video Vision Transformer (ViViT). Evaluation is conducted in two phases: Part I utilizes a consolidated corpus of 130,000 samples across 19 diverse datasets, while Part II focuses on a purified 17-dataset English-only corpus. To assess robustness, a systematic Leave-One-Dataset-Out (LODO) protocol is employed to quantify the generalization gap between seen and unseen distributions. Finally, the top-performing model is benchmarked against 18 state-of-the-art (SOTA) metrics using the ARECHO framework. Results demonstrate that an English-only purified corpus consistently yields higher predictive precision across all architectures. While SSL-FT achieves the highest performance on seen validation data, the SSL-FRZ model provides superior robustness on unseen distributions, achieving a competitive Mean Squared Error (MSE) of 0.36 on the URGENT 2024 benchmark-closely matching domain-optimized SOTA metrics (MSE 0.30). Although the ViViT architecture remains below SSL-based models in total capacity, it delivers stable results in English-only trials. LODO results confirm that while models perform significantly better on seen samples, frozen SSL embeddings combined with deep Transformer encoders offer the most stable and scalable solution for universal speech quality assessment. To support further research, the top-performing English-only SSL-Transformer model and weights are made publicly available via Hugging Face.
#### Hearing Like Humans? Sound Symbolism and Perceptual Alignment in Speech Language Models
 - **Authors:** Yun-Shao Tsai, Chun-Wei Chen, Chee-En Yu, Yi-Cheng Lin, Hung-yi Lee
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Computation and Language (cs.CL)
 - **Arxiv link:** https://arxiv.org/abs/2607.10162

 - **Pdf link:** https://arxiv.org/pdf/2607.10162

 - **Abstract**
 Sound symbolism, the human tendency to map speech sounds to perceptual qualities such as roundness or sharpness, arises primarily from the acoustics of speech rather than spelling. Whether Speech Language Models (SLMs) share this tendency remains open, as prior evaluations rely on text or images rather than real speech. We study it using genuine human speech recordings, comparing model judgments against human data across the auditory, crossmodal, and visual components of the effect. We find that SLMs' auditory judgments align poorly with human perception and miss the acoustic cues, such as spectral tilt, that drive human intuitions, and open-weight models cannot reliably link a heard sound to its corresponding shape. With a visual-only control ruling out shape perception, the weakness localizes to how speech is represented, suggesting that perceptual alignment depends not on stronger vision but on speech representations that capture the cues humans hear.
#### GigaAM Multilingual: Foundation Model for Underrepresented Languages
 - **Authors:** Andrei Kuzmenko, Alexandr Maximenko, Aleksandr Kutsakov, Georgii Gospodinov, Dmitrii Bolotov, Oleg Kutuzov, Pavel Bogomolov, Fyodor Minkin
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Computation and Language (cs.CL)
 - **Arxiv link:** https://arxiv.org/abs/2607.10371

 - **Pdf link:** https://arxiv.org/pdf/2607.10371

 - **Abstract**
 Despite recent scaling successes, multilingual ASR performance remains highly uneven, with long-tail languages suffering from severe data scarcity. This work addresses the challenge of building robust foundation models for underrepresented Central Asian languages (Kazakh, Kyrgyz, Uzbek). We present GigaAM Multilingual, a Conformer encoder pre-trained on 2M hours of audio using a HuBERT-style objective. Crucially, we introduce a cluster-level data balancing strategy during pre-training and a domain-aware sampling method during fine-tuning to mitigate head-language dominance. In controlled comparisons, our approach outperforms strong open pretrained encoders (Whisper Large v3, Omnilingual-1B) on target languages, achieving significant gains on spontaneous speech while maintaining efficiency. We release the foundation encoder and ASR model, offering a proven recipe for effective multilingual adaptation under realistic data imbalance.
#### An Objective Intelligibility Metric Evaluation on Spanish Speech
 - **Authors:** Iván López-Espejo, Jesper Jensen
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2607.10619

 - **Pdf link:** https://arxiv.org/pdf/2607.10619

 - **Abstract**
 Objective intelligibility metrics (OIMs) enable fast and low-cost evaluation of speech intelligibility and are widely used in speech technology assessment. In this study, we evaluate five reference-based OIMs (STOI, ESTOI, STGI, HASPI, and SIIB) and two deep learning-based no-reference metrics (MOSA-Net+ and W2V-SIP) on SpInt, a new Spanish speech intelligibility dataset. Our results show that reference-based OIMs consistently outperform modern data-driven no-reference approaches, which degrade notably under training-test acoustic mismatches such as language mismatch. This effect is particularly relevant in our scenario, as none of the evaluated metrics were exposed to Spanish speech data during development. Consequently, to foster research on more robust and generalizable no-reference OIMs, SpInt is released publicly.
#### Data Augmentation for L2 English Speaking Assessment using TTS
 - **Authors:** Stefano Bannò, Penny Karanasou, Mengjie Qian, Kate M. Knill, Mark J. F. Gales
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2607.10790

 - **Pdf link:** https://arxiv.org/pdf/2607.10790

 - **Abstract**
 Automated assessment of second language (L2) speaking proficiency relies on large-scale annotated speech data, which remains scarce compared to widely available written learner corpora. A promising direction for addressing this imbalance is to use text-to-speech (TTS) and voice cloning to convert written L2 production into synthetic speech. However, written and spoken L2 differ fundamentally: spontaneous speech includes disfluencies and discourse markers, while writing is more planned and complex. This raises the question of what is required to generate synthetic L2 speech suitable for assessment. We address this through a systematic analysis of speaker-text relationships using COREFL, a publicly available corpus containing paired spoken and written responses from the same L2 learners to the same questions across modalities. In our proposed framework, we first address the structural differences between written and spoken language by transforming written responses into spoken-style transcripts ("speechification") using a large language model. These transcripts are then converted into speech using a TTS/voice-cloning model. To assign a voice to each synthetic response, we investigate different speaker-text pairing strategies based on shared learner attributes (proficiency level, first language, both, or neither). We evaluate our data augmentation techniques on the language assessment task, with improvements shown in both wav2vec2 (audio-based) and ModernBERT (text-based) scoring systems. Results show that matching speakers and texts by proficiency level yields the most robust synthetic speech. Moreover, raw written text leads to a strong mismatch with spoken language, while speechification substantially reduces this gap and improves grading performance.
#### Where Speech Enhancement Hurts Recognition: An Inference Time Polar Projection Diagnosis
 - **Authors:** Mingyue Huo, Yuheng Zhang, Hao Zhang
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2607.11157

 - **Pdf link:** https://arxiv.org/pdf/2607.11157

 - **Abstract**
 Speech enhancement (SE) can substantially improve perceptual quality, yet enhanced speech does not necessarily improve automatic speech recognition (ASR). Existing remedies, such as retraining the enhancer jointly with recognizer or interpolating enhanced speech with the noisy input, can mitigate this mismatch, but common explanations such as artifacts and over-suppression remain qualitative and do not localize which enhancement component harms recognition. We propose inference time polar projection, a diagnosis for STFT domain enhancement. Given a mask $M=Ae^{j\phi}$, polar projection forms $M_{\alpha,\gamma}=A^\alpha e^{j\gamma\phi}$, where $\alpha$ controls magnitude strength and $\gamma$ controls phase correction. Sweeping these controls on frozen SE and ASR models turns ASR degradation into measurable magnitude and phase effects. Our projection analysis shows that magnitude strength is the operative axis, while estimated phase correction provides no recognition benefit. The optimal magnitude strength is recognizer dependent: waveform-input wav2vec2.0 favors strong correction, whereas log-Mel-input, noise-robust Whisper prefers weaker correction. Finally, the projection provides a simple mitigation for any SE front end in the STFT mask domain, without retraining either the enhancer or the recognizer, making it directly useful for voice assistants and agents that rely on enhanced speech.
#### Semantic Sampling via Learnable Observation Front Ends
 - **Authors:** Yuxuan Liu, Guangming Shi, Pengfei He, Shuai Ma, Xiang Cheng
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2607.11260

 - **Pdf link:** https://arxiv.org/pdf/2607.11260

 - **Abstract**
 Sampling determines the form of information available to downstream reconstruction systems. Conventional lowrate sampling forms finite-dimensional observations directly from the raw waveform, with the sampling rule mainly guided by bandwidth, sparsity, or fixed signal-level structures. For acoustic signals such as speech, however, reconstruction-relevant information is often expressed through content-related spectral-temporal structures rather than waveform samples alone. This paper proposes semantic sampling via learnable observation front ends, where finite-dimensional observations are generated from learned signal responses instead of directly subsampled waveform points. The proposed front end consists of a semantic feature filterbank, a constrained semantic observation matrix, and a low-rate readout module. The filterbank maps the input waveform into multiple acoustic response channels, the observation matrix combines these responses into a small number of observation channels, and the readout module produces low-rate finite-dimensional samples. A reconstruction network is then used to recover the signal from the resulting observations. Experiments on low-rate speech reconstruction show that, under the same observation budget, the proposed semantic sampling front end provides more informative observations than fixed low-rate sampling and neural restoration methods based on predetermined low-rate waveforms. The improvements in waveform fidelity, spectral consistency, and perceptual quality show that learnable observation front ends preserve more useful information for acoustic signal reconstruction under the same observation budget.
#### Qwen-Audio-VAE Technical Report
 - **Authors:** Ziyue Jiang, Dake Guo, Zekai Zhang, Hangrui Hu, Ting He, Xinfa Zhu, Xiong Wang, Yongqi Wang, Jiapeng Wang, Wenxiang Guo, Zhifang Guo, Chenfei Wu, Dayiheng Liu, Jin Xu
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2607.11738

 - **Pdf link:** https://arxiv.org/pdf/2607.11738

 - **Abstract**
 We introduce \textbf{Qwen-Audio-VAE}, a suite of low-bitrate, fast-encoding continuous audio autoencoders designed for scalable general audio generation. The model is built around a simple but important principle: an audio VAE should not only reconstruct diverse audio with high fidelity, but also produce compact latent representations fast enough to support large-scale text-to-audio training. Qwen-Audio-VAE combines a causal encoder-decoder, window Transformer blocks, and multi-discriminator training to achieve a strong balance between reconstruction quality and compression rate. The model is trained at scale on 5 million hours of multi-domain audio, enabling robust reconstruction across heterogeneous acoustic conditions. To further improve computational efficiency, we adopt an asymmetric encoder-decoder backbone and introduce latency-aware encoder pruning to maximize encoding throughput. Experiments on public speech, music, and sound reconstruction benchmarks show that Qwen-Audio-VAE generalizes well across diverse audio domains and is particularly efficient, requiring only 541 ms to encode 32 minutes of audio. Overall, Qwen-Audio-VAE provides a high-quality, compact, and high-throughput representation backbone for efficient general audio generation.
#### Synchronized Three-Dimensional Vocal-Tract Motion for Speech Synchronization via Joint-Embedding Predictive Architecture Alignment
 - **Authors:** Sheng Li, Takahiro Shinozaki
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2607.11772

 - **Pdf link:** https://arxiv.org/pdf/2607.11772

 - **Abstract**
 Modern neural speech systems can generate intelligible waveforms, but they usually hide the physical speech-production state that produced the sound. Conversely, biomechanical vocal-tract models expose articulatory structure, contact behavior, airflow routing, and geometric constraints, but direct physical waveform synthesis remains less robust than modern neural vocoders. A duration-preserving acoustic carrier supplies the listening waveform, while a corrected three-dimensional vocal-tract model supplies synchronized jaw, lip, tongue, velum, laryngeal, oral-airflow, and nasal-airflow motion. A joint-embedding predictive architecture (JEPA)-style representation and a reinforcement learning/cross-entropy method (RL/CEM) trajectory-selection loop align articulatory actions to the acoustic carrier and to physical-plausibility constraints. The evaluation contains 12 3D recordings covering 24 minimal-pair stimuli. On the 24-word set, the carrier obtains good automatic speech recognition (ASR) results (an 8.33\% WER, a 4.17\% CER), a UTMOS score of 3.174, a mean JEPA score of 0.864, and a mean timbre-guard score of 0.947.
#### Which Languages Transfer Best to Warlpiri? A Similarity-Based Study for Low-Resource ASR
 - **Authors:** Pravina Mylvaganam, Eliathamby Ambikairajah, Ting Dang, Vidhyasaharan Sethu, Tuende Szalay
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2607.10256

 - **Pdf link:** https://arxiv.org/pdf/2607.10256

 - **Abstract**
 This paper investigates how language similarity can improve cross-lingual transfer for automatic speech recognition (ASR) in extremely low-resource settings. Warlpiri, an Australian Aboriginal language, has very limited transcribed speech data, making transfer learning essential. We propose a framework combining acoustic similarity from pre-trained speech models with linguistic similarity based on typology, phoneme inventories, grammatical, and syntactic features to rank high-resource source languages and evaluate their effectiveness for ASR transfer to Warlpiri. Experiments with Whisper show that acoustically and typologically similar languages outperform monolingual and multilingual baselines. Assamese and Hindi achieve substantial reductions in word and character error rates. Correlation analysis further indicates that acoustic similarity is the strongest predictor of fine-tuning performance, while phoneme inventory and typological similarity better explain zero-shot transfer.
#### Simple Features and Honest Calibration for Ambivalence and Hesitancy Recognition in Video
 - **Authors:** Vikas Kumar, Aditya Mishra, Haroon R. Lone
 - **Subjects:** Subjects:
Computer Vision and Pattern Recognition (cs.CV); Computation and Language (cs.CL); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2607.11120

 - **Pdf link:** https://arxiv.org/pdf/2607.11120

 - **Abstract**
 We address ambivalence and hesitancy (A/H) recognition in the ABAW 2026 BAH Challenge: given a short interview video, predict whether the person shows signs of A/H. Our system combines affect-specialised text, audio, and visual representations with a small set of readable linguistic hesitation cues, fused by a reliability gate we call Affective Marker Fusion (AMF), and finished with a simple AP-weighted ensemble at a fixed decision threshold. We also introduce \emph{ASR-erased time}: speech recognisers delete fillers and hesitation pauses from the transcript, but the chunk timestamps keep the time those events took, and sixteen features built from these gaps form the strongest and most independent non-verbal channel we measured (AP $0.718$, correlation $0.11$--$0.36$ with all other members). Across controlled experiments we find three things: cross-modal conflict design does not reliably help on BAH; language is by far the strongest channel while affect-specialised audio is a useful second; and calibration matters more than architecture. Fitting ensemble weights and a threshold on the small validation split overfits: it scores $0.741$ macro-F1 on validation but only $0.690$ on the untouched test set. AP-weighting at a fixed threshold instead reaches $\mathbf{0.731}$ on test.
#### Teaching Speech Enhancement Models to Sing: Domain Adaptation from Speech Enhancement to Singing Voice Separation
 - **Authors:** Paul A. Bereuter, Mark D. Plumbley, Alois Sontacchi
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2607.11630

 - **Pdf link:** https://arxiv.org/pdf/2607.11630

 - **Abstract**
 State-of-the-art speech enhancement models benefit from large-scale labeled datasets, whereas singing voice separation models suffer from limited available training data. To address this limitation, we formulate singing voice separation as domain adaptation from speech enhancement to singing voice separation. We investigate two fine-tuning strategies: full fine-tuning and parameter-efficient fine-tuning using Low-Rank Adaptation (LoRA) on a discriminative and a generative model. Models with either adaptation strategy outperform the same architectures trained from scratch by 0.29-1.8 dB in Signal-to-Distortion-Ratio. Full fine-tuning yields the highest singing voice separation performance, but catastrophic forgetting degrades speech enhancement performance. LoRA fine-tuning achieves competitive singing voice separation performance while preserving the original speech enhancement capability with only 6-12% additional parameters compared to the base speech enhancement model. Furthermore, the generative model shows improved generalization to an unseen test set. The results demonstrate that adapting pretrained speech enhancement models is an effective strategy for training singing voice separation models in data-scarce scenarios.
#### Casting Everything to Online API Services? A Survey of Integrating Localized Speech Recognition Models in Robotic Systems
 - **Authors:** Sheng Li, Jing Li, Felix Schijve, Jun Hu, Emilia Barakova
 - **Subjects:** Subjects:
Robotics (cs.RO); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2607.11792

 - **Pdf link:** https://arxiv.org/pdf/2607.11792

 - **Abstract**
 Automatic speech recognition (ASR) has become a critical component of modern robotic systems because it is one of the most natural and intuitive ways for humans to interact with robots. A commonly used method is to directly use API services online. But is that all we can do? This article provides an overview of how ASR technologies are integrated into various intelligent robots and machines. We discuss the evolution of speech recognition from established approaches to state-of-the-art deep learning models, such as OpenAI's Whisper. We also list large-scale datasets and open source toolkits that have been widely used in both industry and academia. We structure the survey around ASR model families, deployment strategies in robotics (especially ROS-based, cloud-based, and hybrid solutions), and several real-world robotic platforms. Finally, we outline the challenges of deploying robust speech recognition in robots and discuss future directions, including multimodal interaction in diverse and dynamic environments. This paper can help social robotics researchers better navigate the emerging domain of language-based natural human-robot interaction.


by Zyzzyva0381 (Windy). 


2026-07-14
