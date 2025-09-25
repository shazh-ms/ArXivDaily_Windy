# Showing new listings for Thursday, 25 September 2025
Auto update papers at about 2:30am UTC (10:30am Beijing time) every weekday.


阅读 `Usage.md`了解如何使用此repo实现个性化的Arxiv论文推送

See `Usage.md` for instructions on how to personalize the repo. 


Keyword list: ['text-to-speech', 'text to speech', 'tts', 'LLM-based', 'speech', 'voice']


Excluded: []


### Today: 13papers 
#### Frame-Stacked Local Transformers For Efficient Multi-Codebook Speech Generation
 - **Authors:** Roy Fejgin, Paarth Neekhara, Xuesong Yang, Edresson Casanova, Ryan Langman Jaehyeon Kim, Subhankar Ghosh, Shehzeen Hussain, Jason Li
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Artificial Intelligence (cs.AI); Computation and Language (cs.CL); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2509.19592

 - **Pdf link:** https://arxiv.org/pdf/2509.19592

 - **Abstract**
 Speech generation models based on large language models (LLMs) typically operate on discrete acoustic codes, which differ fundamentally from text tokens due to their multicodebook structure. At each timestep, models must predict N codebook entries jointly, introducing dependencies that challenge simple parallel prediction approaches. Parallel prediction assumes independence among codebooks, yielding efficient decoding but often at the cost of reduced fidelity. To address this, hierarchical strategies employ a local transformer (LT) to refine predictions and capture intra-timestep dependencies. In this work, we systematically investigate two LT architectures: an autoregressive transformer that generates codebooks sequentially, and a MaskGIT-based transformer that performs iterative masked prediction. Both designs further enable frame stacking, where the primary transformer predicts multiple frames jointly, and the LT decodes their codebooks, offering improvements in speed without compromising perceptual quality. Through extensive analysis, we characterize the tradeoffs between parallel and iterative sampling strategies across different throughput and quality regimes. Finally, we propose practical guidelines for selecting decoding strategies based on deployment priorities such as computational efficiency and synthesis fidelity.
#### Advancing Speech Summarization in Multi-modal LLMs with Reinforcement Learning
 - **Authors:** Shaoshi Ling, Gang Liu, Guoli Ye, Jinyu Li
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Artificial Intelligence (cs.AI); Computation and Language (cs.CL)
 - **Arxiv link:** https://arxiv.org/abs/2509.19631

 - **Pdf link:** https://arxiv.org/pdf/2509.19631

 - **Abstract**
 Speech summarization is a critical component of spoken content understanding, particularly in the era of rapidly growing spoken and audiovisual data. Recent advances in multi-modal large language models (MLLMs), leveraging the power of LLMs, enable generating textual summaries directly from speech without intermediate transcriptions, while supporting controllable styles and zero-shot generalization. However, open-source MLLMs continue to lag behind the state-of-the-art text-based LLMs, limiting their practical deployment for speech summarization. In this work, we present a novel multi-stage reinforcement learning training framework to enhance the speech summarization capabilities in MLLMs. Our model delivers substantial improvements over strong baselines, outperforms much larger MLLMs, and significantly narrows the gap with state-of-the-art text-based LLMs.
#### Selective Classifier-free Guidance for Zero-shot Text-to-speech
 - **Authors:** John Zheng, Farhad Maleki
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Artificial Intelligence (cs.AI); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2509.19668

 - **Pdf link:** https://arxiv.org/pdf/2509.19668

 - **Abstract**
 In zero-shot text-to-speech, achieving a balance between fidelity to the target speaker and adherence to text content remains a challenge. While classifier-free guidance (CFG) strategies have shown promising results in image generation, their application to speech synthesis are underexplored. Separating the conditions used for CFG enables trade-offs between different desired characteristics in speech synthesis. In this paper, we evaluate the adaptability of CFG strategies originally developed for image generation to speech synthesis and extend separated-condition CFG approaches for this domain. Our results show that CFG strategies effective in image generation generally fail to improve speech synthesis. We also find that we can improve speaker similarity while limiting degradation of text adherence by applying standard CFG during early timesteps and switching to selective CFG only in later timesteps. Surprisingly, we observe that the effectiveness of a selective CFG strategy is highly text-representation dependent, as differences between the two languages of English and Mandarin can lead to different results even with the same model.
#### MMedFD: A Real-world Healthcare Benchmark for Multi-turn Full-Duplex Automatic Speech Recognition
 - **Authors:** Hongzhao Chen, XiaoYang Wang, Jing Lan, Hexiao Ding, Yufeng Jiang MingHui Yang, DanHui Xu, Jun Luo, Nga-Chun Ng, Gerald W.Y. Cheng, Yunlin Mao, Jung Sun Yoo
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2509.19817

 - **Pdf link:** https://arxiv.org/pdf/2509.19817

 - **Abstract**
 Automatic speech recognition (ASR) in clinical dialogue demands robustness to full-duplex interaction, speaker overlap, and low-latency constraints, yet open benchmarks remain scarce. We present MMedFD, the first real-world Chinese healthcare ASR corpus designed for multi-turn, full-duplex settings. Captured from a deployed AI assistant, the dataset comprises 5,805 annotated sessions with synchronized user and mixed-channel views, RTTM/CTM timing, and role labels. We introduce a model-agnostic pipeline for streaming segmentation, speaker attribution, and dialogue memory, and fine-tune Whisper-small on role-concatenated audio for long-context recognition. ASR evaluation includes WER, CER, and HC-WER, which measures concept-level accuracy across healthcare settings. LLM-generated responses are assessed using rubric-based and pairwise protocols. MMedFD establishes a reproducible framework for benchmarking streaming ASR and end-to-end duplex agents in healthcare deployment. The dataset and related resources are publicly available at this https URL
#### Weakly Supervised Phonological Features for Pathological Speech Analysis
 - **Authors:** Jenthe Thienpondt, Geoffroy Vanderreydt, Abdessalem Hammami, Kris Demuynck
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2509.19879

 - **Pdf link:** https://arxiv.org/pdf/2509.19879

 - **Abstract**
 Paralinguistic properties of speech are essential in analyzing and choosing optimal treatment options for patients with speech disorders. However, automatic modeling of these characteristics is difficult due to the lack of labeled speech datasets describing paralinguistic properties, especially at the frame-level. In this paper, we propose a weakly supervised training method which exploits the known acoustic properties of phonemes by training an ASR model with an interpretable frame-level phonological feature bottleneck layer. Subsequently, we assess the viability of these phonological features in speech pathology analysis by developing corresponding models for intelligibility prediction and speech pathology classification. Models using our proposed phonological features perform similar to other state-of-the-art acoustic features on both tasks with a classification accuracy of 75% and a 8.43 RMSE on speech intelligibility prediction. In contrast to others, our phonological features are text-independent and highly interpretable, providing potentially useful insights for speech therapists.
#### MAGE: A Coarse-to-Fine Speech Enhancer with Masked Generative Model
 - **Authors:** The Hieu Pham, Tan Dat Nguyen, Phuong Thanh Tran, Joon Son Chun, Duc Dung Nguyen
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2509.19881

 - **Pdf link:** https://arxiv.org/pdf/2509.19881

 - **Abstract**
 Speech enhancement remains challenging due to the trade-off between efficiency and perceptual quality. In this paper, we introduce MAGE, a Masked Audio Generative Enhancer that advances generative speech enhancement through a compact and robust design. Unlike prior masked generative models with random masking, MAGE employs a scarcity-aware coarse-to-fine masking strategy that prioritizes frequent tokens in early steps and rare tokens in later refinements, improving efficiency and generalization. We also propose a lightweight corrector module that further stabilizes inference by detecting low-confidence predictions and re-masking them for refinement. Built on BigCodec and finetuned from Qwen2.5-0.5B, MAGE is reduced to 200M parameters through selective layer retention. Experiments on DNS Challenge and noisy LibriSpeech show that MAGE achieves state-of-the-art perceptual quality and significantly reduces word error rate for downstream recognition, outperforming larger baselines. Audio examples are available at this https URL.
#### Voice Privacy Preservation with Multiple Random Orthogonal Secret Keys: Attack Resistance Analysis
 - **Authors:** Kohei Tanaka, Hitoshi Kiya, Sayaka Shiota
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2509.19906

 - **Pdf link:** https://arxiv.org/pdf/2509.19906

 - **Abstract**
 Recently, opportunities to transmit speech data to deep learning models executed in the cloud have increased. This has led to growing concerns about speech privacy, including both speaker-specific information and the linguistic content of utterances. As an approach to preserving speech privacy, a speech privacy-preserving method based on encryption using a secret key with a random orthogonal matrix has been proposed. This method enables cloud-based model inference while concealing both the speech content and the speaker identity. However, the method has limited attack resistance and is constrained in terms of the deep learning models to which the encryption can be applied. In this work, we propose a method that enhances the attack resistance of the conventional speech privacy-preserving technique by employing multiple random orthogonal matrices as secret keys. We also introduce approaches to relax the model constraints, enabling the application of our method to a broader range of deep learning models. Furthermore, we investigate the robustness of the proposed method against attacks using extended attack scenarios based on the scenarios employed in the Voice Privacy Challenge. Our experimental results confirmed that the proposed method maintains privacy protection performance for speaker concealment, even under more powerful attack scenarios not considered in prior work.
#### Measuring Prosody Diversity in Zero-Shot TTS: A New Metric, Benchmark, and Exploration
 - **Authors:** Yifan Yang, Bing Han, Hui Wang, Long Zhou, Wei Wang, Mingyu Cui, Xu Tan, Xie Chen
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2509.19928

 - **Pdf link:** https://arxiv.org/pdf/2509.19928

 - **Abstract**
 Prosody diversity is essential for achieving naturalness and expressiveness in zero-shot text-to-speech (TTS). However, frequently used acoustic metrics capture only partial views of prosodic variation and correlate poorly with human perception, leaving the problem of reliably quantifying prosody diversity underexplored. To bridge this gap, we introduce ProsodyEval, a prosody diversity assessment dataset that provides Prosody Mean Opinion Score (PMOS) alongside conventional acoustic metrics. ProsodyEval comprises 1000 speech samples derived from 7 mainstream TTS systems, with 2000 human ratings. Building on this, we propose the Discretized Speech Weighted Edit Distance (DS-WED), a new objective diversity metric that quantifies prosodic variation via weighted edit distance over semantic tokens. Experiments on ProsodyEval show that DS-WED achieves substantially higher correlation with human judgments than existing acoustic metrics, while remaining highly robust in speech tokenization from HuBERT and WavLM. Leveraging DS-WED, we benchmark state-of-the-art open-source TTS systems on LibriSpeech test-clean and Seed-TTS test-en, and further explorations uncover several factors that influence prosody diversity, including generative modeling paradigms, duration control, and reinforcement learning. Moreover, we find that current large audio language models (LALMs) remain limited in capturing prosodic variations. Audio samples are available at this https URL.
#### Evaluating pretrained speech embedding systems for dysarthria detection across heterogenous datasets
 - **Authors:** Lovisa Wihlborg, Jemima Goodall, David Wheatley, Jacob J. Webber, Johnny Tam, Christine Weaver, Suvankar Pal, Siddharthan Chandran, Sohan Seth, Oliver Watts, Cassia Valentini-Botinhao
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2509.19946

 - **Pdf link:** https://arxiv.org/pdf/2509.19946

 - **Abstract**
 We present a comprehensive evaluation of pretrained speech embedding systems for the detection of dysarthric speech using existing accessible data. Dysarthric speech datasets are often small and can suffer from recording biases as well as data imbalance. To address these we selected a range of datasets covering related conditions and adopt the use of several cross-validations runs to estimate the chance level. To certify that results are above chance, we compare the distribution of scores across these runs against the distribution of scores of a carefully crafted null hypothesis. In this manner, we evaluate 17 publicly available speech embedding systems across 6 different datasets, reporting the cross-validation performance on each. We also report cross-dataset results derived when training with one particular dataset and testing with another. We observed that within-dataset results vary considerably depending on the dataset, regardless of the embedding used, raising questions about which datasets should be used for benchmarking. We found that cross-dataset accuracy is, as expected, lower than within-dataset, highlighting challenges in the generalization of the systems. These findings have important implications for the clinical validity of systems trained and tested on the same dataset.
#### Discrete Diffusion for Generative Modeling of Text-Aligned Speech Tokens
 - **Authors:** Pin-Jui Ku, He Huang, Jean-Marie Lemercier, Subham Sekhar Sahoo, Zhehuai Chen, Ante Jukić
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2509.20060

 - **Pdf link:** https://arxiv.org/pdf/2509.20060

 - **Abstract**
 This paper introduces a discrete diffusion model (DDM) framework for text-aligned speech tokenization and reconstruction. By replacing the auto-regressive speech decoder with a discrete diffusion counterpart, our model achieves significantly better reconstruction quality, stronger ASR performance, and faster inference. We provide a comprehensive analysis of applying DDMs to speech reconstruction, examining sampler choices, inference steps, and robustness to length-scale estimation errors. Furthermore, we improve the original TASTE by systematically comparing vector quantization modules, showing that FSQ yields up to a 35% relative WER reduction and +0.14 UT-MOS improvement over RVQ for AR models, while also enhancing DDM performance. Our model generates speech in just 10 denoising steps and even supports single-step generation with only minor quality degradation.
#### Retrieval Augmented Generation based context discovery for ASR
 - **Authors:** Dimitrios Siskos, Stavros Papadopoulos, Pablo Peso Parada, Jisi Zhang, Karthikeyan Saravanan, Anastasios Drosou
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2509.19567

 - **Pdf link:** https://arxiv.org/pdf/2509.19567

 - **Abstract**
 This work investigates retrieval augmented generation as an efficient strategy for automatic context discovery in context-aware Automatic Speech Recognition (ASR) system, in order to improve transcription accuracy in the presence of rare or out-of-vocabulary terms. However, identifying the right context automatically remains an open challenge. This work proposes an efficient embedding-based retrieval approach for automatic context discovery in ASR. To contextualize its effectiveness, two alternatives based on large language models (LLMs) are also evaluated: (1) large language model (LLM)-based context generation via prompting, and (2) post-recognition transcript correction using LLMs. Experiments on the TED-LIUMv3, Earnings21 and SPGISpeech demonstrate that the proposed approach reduces WER by up to 17% (percentage difference) relative to using no-context, while the oracle context results in a reduction of up to 24.1%.
#### Z-Scores: A Metric for Linguistically Assessing Disfluency Removal
 - **Authors:** Maria Teleki, Sai Janjur, Haoran Liu, Oliver Grabner, Ketan Verma, Thomas Docog, Xiangjue Dong, Lingfeng Shi, Cong Wang, Stephanie Birkelbach, Jason Kim, Yin Zhang, James Caverlee
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Artificial Intelligence (cs.AI); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2509.20319

 - **Pdf link:** https://arxiv.org/pdf/2509.20319

 - **Abstract**
 Evaluating disfluency removal in speech requires more than aggregate token-level scores. Traditional word-based metrics such as precision, recall, and F1 (E-Scores) capture overall performance but cannot reveal why models succeed or fail. We introduce Z-Scores, a span-level linguistically-grounded evaluation metric that categorizes system behavior across distinct disfluency types (EDITED, INTJ, PRN). Our deterministic alignment module enables robust mapping between generated text and disfluent transcripts, allowing Z-Scores to expose systematic weaknesses that word-level metrics obscure. By providing category-specific diagnostics, Z-Scores enable researchers to identify model failure modes and design targeted interventions -- such as tailored prompts or data augmentation -- yielding measurable performance improvements. A case study with LLMs shows that Z-Scores uncover challenges with INTJ and PRN disfluencies hidden in aggregate F1, directly informing model refinement strategies.
#### DRES: Benchmarking LLMs for Disfluency Removal
 - **Authors:** Maria Teleki, Sai Janjur, Haoran Liu, Oliver Grabner, Ketan Verma, Thomas Docog, Xiangjue Dong, Lingfeng Shi, Cong Wang, Stephanie Birkelbach, Jason Kim, Yin Zhang, James Caverlee
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Artificial Intelligence (cs.AI); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2509.20321

 - **Pdf link:** https://arxiv.org/pdf/2509.20321

 - **Abstract**
 Disfluencies -- such as "um," "uh," interjections, parentheticals, and edited statements -- remain a persistent challenge for speech-driven systems, degrading accuracy in command interpretation, summarization, and conversational agents. We introduce DRES (Disfluency Removal Evaluation Suite), a controlled text-level benchmark that establishes a reproducible semantic upper bound for this task. DRES builds on human-annotated Switchboard transcripts, isolating disfluency removal from ASR errors and acoustic variability. We systematically evaluate proprietary and open-source LLMs across scales, prompting strategies, and architectures. Our results reveal that (i) simple segmentation consistently improves performance, even for long-context models; (ii) reasoning-oriented models tend to over-delete fluent tokens; and (iii) fine-tuning achieves near state-of-the-art precision and recall but harms generalization abilities. We further present a set of LLM-specific error modes and offer nine practical recommendations (R1-R9) for deploying disfluency removal in speech-driven pipelines. DRES provides a reproducible, model-agnostic foundation for advancing robust spoken-language systems.


by Zyzzyva0381 (Windy). 


2025-09-25
