# Showing new listings for Monday, 18 August 2025
Auto update papers at about 2:30am UTC (10:30am Beijing time) every weekday.


阅读 `Usage.md`了解如何使用此repo实现个性化的Arxiv论文推送

See `Usage.md` for instructions on how to personalize the repo. 


Keyword list: ['text-to-speech', 'text to speech', 'tts', 'LLM-based', 'speech', 'voice']


Excluded: []


### Today: 11papers 
#### Expressive Speech Retrieval using Natural Language Descriptions of Speaking Style
 - **Authors:** Wonjune Kang, Deb Roy
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Computation and Language (cs.CL); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2508.11187

 - **Pdf link:** https://arxiv.org/pdf/2508.11187

 - **Abstract**
 We introduce the task of expressive speech retrieval, where the goal is to retrieve speech utterances spoken in a given style based on a natural language description of that style. While prior work has primarily focused on performing speech retrieval based on what was said in an utterance, we aim to do so based on how something was said. We train speech and text encoders to embed speech and text descriptions of speaking styles into a joint latent space, which enables using free-form text prompts describing emotions or styles as queries to retrieve matching expressive speech segments. We perform detailed analyses of various aspects of our proposed framework, including encoder architectures, training criteria for effective cross-modal alignment, and prompt augmentation for improved generalization to arbitrary text queries. Experiments on multiple datasets encompassing 22 speaking styles demonstrate that our approach achieves strong retrieval performance as measured by Recall@k.
#### EmoSSLSphere: Multilingual Emotional Speech Synthesis with Spherical Vectors and Discrete Speech Tokens
 - **Authors:** Joonyong Park, Kenichi Nakamura
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2508.11273

 - **Pdf link:** https://arxiv.org/pdf/2508.11273

 - **Abstract**
 This paper introduces EmoSSLSphere, a novel framework for multilingual emotional text-to-speech (TTS) synthesis that combines spherical emotion vectors with discrete token features derived from self-supervised learning (SSL). By encoding emotions in a continuous spherical coordinate space and leveraging SSL-based representations for semantic and acoustic modeling, EmoSSLSphere enables fine-grained emotional control, effective cross-lingual emotion transfer, and robust preservation of speaker identity. We evaluate EmoSSLSphere on English and Japanese corpora, demonstrating significant improvements in speech intelligibility, spectral fidelity, prosodic consistency, and overall synthesis quality. Subjective evaluations further confirm that our method outperforms baseline models in terms of naturalness and emotional expressiveness, underscoring its potential as a scalable solution for multilingual emotional TTS.
#### MoE-TTS: Enhancing Out-of-Domain Text Understanding for Description-based TTS via Mixture-of-Experts
 - **Authors:** Heyang Xue, Xuchen Song, Yu Tang, Jianyu Chen, Yanru Chen, Yang Li, Yahui Zhou
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2508.11326

 - **Pdf link:** https://arxiv.org/pdf/2508.11326

 - **Abstract**
 Description-based text-to-speech (TTS) models exhibit strong performance on in-domain text descriptions, i.e., those encountered during training. However, in real-world applications, the diverse range of user-generated descriptions inevitably introduces numerous out-of-domain inputs that challenge the text understanding capabilities of these systems. To address this issue, we propose MoE-TTS, a description-based TTS model designed to enhance the understanding of out-of-domain text descriptions. MoE-TTS employs a modality-based mixture-of-experts (MoE) approach to augment a pre-trained textual large language model (LLM) with a set of specialized weights adapted to the speech modality while maintaining the original LLM frozen during training. This approach allows MoE-TTS to effectively leverage the pre-trained knowledge and text understanding abilities of textual LLMs. Our experimental results indicate that: first, even the most advanced closed-source commercial products can be challenged by carefully designed out-of-domain description test sets; second, MoE-TTS achieves superior performance in generating speech that more accurately reflects the descriptions. We encourage readers to listen to the demos at this https URL.
#### Enhancing In-the-Wild Speech Emotion Conversion with Resynthesis-based Duration Modeling
 - **Authors:** Navin Raj Prabhu, Danilo de Oliveira, Nale Lehmann-Willenbrock, Timo Gerkmann
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2508.11535

 - **Pdf link:** https://arxiv.org/pdf/2508.11535

 - **Abstract**
 Speech Emotion Conversion aims to modify the emotion expressed in input speech while preserving lexical content and speaker identity. Recently, generative modeling approaches have shown promising results in changing local acoustic properties such as fundamental frequency, spectral envelope and energy, but often lack the ability to control the duration of sounds. To address this, we propose a duration modeling framework using resynthesis-based discrete content representations, enabling modification of speech duration to reflect target emotions and achieve controllable speech rates without using parallel data. Experimental results reveal that the inclusion of the proposed duration modeling framework significantly enhances emotional expressiveness, in the in-the-wild MSP-Podcast dataset. Analyses show that low-arousal emotions correlate with longer durations and slower speech rates, while high-arousal emotions produce shorter, faster speech.
#### Emphasis Sensitivity in Speech Representations
 - **Authors:** Shaun Cassini, Thomas Hain, Anton Ragni
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Computation and Language (cs.CL)
 - **Arxiv link:** https://arxiv.org/abs/2508.11566

 - **Pdf link:** https://arxiv.org/pdf/2508.11566

 - **Abstract**
 This work investigates whether modern speech models are sensitive to prosodic emphasis - whether they encode emphasized and neutral words in systematically different ways. Prior work typically relies on isolated acoustic correlates (e.g., pitch, duration) or label prediction, both of which miss the relational structure of emphasis. This paper proposes a residual-based framework, defining emphasis as the difference between paired neutral and emphasized word representations. Analysis on self-supervised speech models shows that these residuals correlate strongly with duration changes and perform poorly at word identity prediction, indicating a structured, relational encoding of prosodic emphasis. In ASR fine-tuned models, residuals occupy a subspace up to 50% more compact than in pre-trained models, further suggesting that emphasis is encoded as a consistent, low-dimensional transformation that becomes more structured with task-specific learning.
#### Perturbed Public Voices (P$^{2}$V): A Dataset for Robust Audio Deepfake Detection
 - **Authors:** Chongyang Gao, Marco Postiglione, Isabel Gortner, Sarit Kraus, V.S. Subrahmanian
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2508.10949

 - **Pdf link:** https://arxiv.org/pdf/2508.10949

 - **Abstract**
 Current audio deepfake detectors cannot be trusted. While they excel on controlled benchmarks, they fail when tested in the real world. We introduce Perturbed Public Voices (P$^{2}$V), an IRB-approved dataset capturing three critical aspects of malicious deepfakes: (1) identity-consistent transcripts via LLMs, (2) environmental and adversarial noise, and (3) state-of-the-art voice cloning (2020-2025). Experiments reveal alarming vulnerabilities of 22 recent audio deepfake detectors: models trained on current datasets lose 43% performance when tested on P$^{2}$V, with performance measured as the mean of F1 score on deepfake audio, AUC, and 1-EER. Simple adversarial perturbations induce up to 16% performance degradation, while advanced cloning techniques reduce detectability by 20-30%. In contrast, P$^{2}$V-trained models maintain robustness against these attacks while generalizing to existing datasets, establishing a new benchmark for robust audio deepfake detection. P$^{2}$V will be publicly released upon acceptance by a conference/journal.
#### Novel Parasitic Dual-Scale Modeling for Efficient and Accurate Multilingual Speech Translation
 - **Authors:** Chenyang Le, Yinfeng Xia, Huiyan Li, Manhong Wang, Yutao Sun, Xingyang Ma, Yanmin Qian
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2508.11189

 - **Pdf link:** https://arxiv.org/pdf/2508.11189

 - **Abstract**
 Recent advancements in speech-to-text translation have led to the development of multilingual models capable of handling multiple language pairs simultaneously. However, these unified models often suffer from large parameter sizes, making it challenging to balance inference efficiency and performance, particularly in local deployment scenarios. We propose an innovative Parasitic Dual-Scale Approach, which combines an enhanced speculative sampling method with model compression and knowledge distillation techniques. Building on the Whisper Medium model, we enhance it for multilingual speech translation into whisperM2M, and integrate our novel KVSPN module, achieving state-of-the-art (SOTA) performance across six popular languages with improved inference efficiency. KVSPN enables a 40\% speedup with no BLEU score degradation. Combined with distillation methods, it represents a 2.6$\times$ speedup over the original Whisper Medium with superior performance.
#### Benchmarking Prosody Encoding in Discrete Speech Tokens
 - **Authors:** Kentaro Onda, Satoru Fukayama, Daisuke Saito, Nobuaki Minematsu
 - **Subjects:** Subjects:
Sound (cs.SD); Computation and Language (cs.CL); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2508.11224

 - **Pdf link:** https://arxiv.org/pdf/2508.11224

 - **Abstract**
 Recently, discrete tokens derived from self-supervised learning (SSL) models via k-means clustering have been actively studied as pseudo-text in speech language models and as efficient intermediate representations for various tasks. However, these discrete tokens are typically learned in advance, separately from the training of language models or downstream tasks. As a result, choices related to discretization, such as the SSL model used or the number of clusters, must be made heuristically. In particular, speech language models are expected to understand and generate responses that reflect not only the semantic content but also prosodic features. Yet, there has been limited research on the ability of discrete tokens to capture prosodic information. To address this gap, this study conducts a comprehensive analysis focusing on prosodic encoding based on their sensitivity to the artificially modified prosody, aiming to provide practical guidelines for designing discrete tokens.
#### Speech Emotion Recognition Using Fine-Tuned DWFormer:A Study on Track 1 of the IERPChallenge 2024
 - **Authors:** Honghong Wang, Xupeng Jia, Jing Deng, Rong Zheng
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2508.11371

 - **Pdf link:** https://arxiv.org/pdf/2508.11371

 - **Abstract**
 The field of artificial intelligence has a strong interest in the topic of emotion recognition. The majority of extant emotion recognition models are oriented towards enhancing the precision of discrete emotion label prediction. Given the direct relationship between human personality and emotion, as well as the significant inter-individual differences in subjective emotional expression, the IERP Challenge 2024 incorporates personality traits into emotion recognition research. This paper presents the Fosafer submissions to the Track 1 of the IERP Challenge 2024. This task primarily concerns the recognition of emotions in audio, while also providing text and audio features. In Track 1, we utilized exclusively audio-based features and fine-tuned a pre-trained speech emotion recognition model, DWFormer, through the integration of data augmentation and score fusion strategies, thereby achieving the first place among the participating teams.
#### Representing Speech Through Autoregressive Prediction of Cochlear Tokens
 - **Authors:** Greta Tuckute, Klemen Kotar, Evelina Fedorenko, Daniel L.K. Yamins
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2508.11598

 - **Pdf link:** https://arxiv.org/pdf/2508.11598

 - **Abstract**
 We introduce AuriStream, a biologically inspired model for encoding speech via a two-stage framework inspired by the human auditory processing hierarchy. The first stage transforms raw audio into a time-frequency representation based on the human cochlea, from which we extract discrete \textbf{cochlear tokens}. The second stage applies an autoregressive sequence model over the cochlear tokens. AuriStream learns meaningful phoneme and word representations, and state-of-the-art lexical semantics. AuriStream shows competitive performance on diverse downstream SUPERB speech tasks. Complementing AuriStream's strong representational capabilities, it generates continuations of audio which can be visualized in a spectrogram space and decoded back into audio, providing insights into the model's predictions. In summary, we present a two-stage framework for speech representation learning to advance the development of more human-like models that efficiently handle a range of speech-based tasks.
#### Pretrained Conformers for Audio Fingerprinting and Retrieval
 - **Authors:** Kemal Altwlkany, Elmedin Selmanovic, Sead Delalic
 - **Subjects:** Subjects:
Sound (cs.SD); Artificial Intelligence (cs.AI); Information Retrieval (cs.IR); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2508.11609

 - **Pdf link:** https://arxiv.org/pdf/2508.11609

 - **Abstract**
 Conformers have shown great results in speech processing due to their ability to capture both local and global interactions. In this work, we utilize a self-supervised contrastive learning framework to train conformer-based encoders that are capable of generating unique embeddings for small segments of audio, generalizing well to previously unseen data. We achieve state-of-the-art results for audio retrieval tasks while using only 3 seconds of audio to generate embeddings. Our models are almost completely immune to temporal misalignments and achieve state-of-the-art results in cases of other audio distortions such as noise, reverb or extreme temporal stretching. Code and models are made publicly available and the results are easy to reproduce as we train and test using popular and freely available datasets of different sizes.


by Zyzzyva0381 (Windy). 


2025-08-18
