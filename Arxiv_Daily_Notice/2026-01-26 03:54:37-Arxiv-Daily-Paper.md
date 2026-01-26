# Showing new listings for Monday, 26 January 2026
Auto update papers at about 2:30am UTC (10:30am Beijing time) every weekday.


阅读 `Usage.md`了解如何使用此repo实现个性化的Arxiv论文推送

See `Usage.md` for instructions on how to personalize the repo. 


Keyword list: ['text-to-speech', 'text to speech', 'tts', 'LLM-based', 'speech', 'voice']


Excluded: []


### Today: 13papers 
#### ES4R: Speech Encoding Based on Prepositive Affective Modeling for Empathetic Response Generation
 - **Authors:** Zhuoyue Gao, Xiaohui Wang, Xiaocui Yang, Wen Zhang, Daling Wang, Shi Feng, Yifei Zhang
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Artificial Intelligence (cs.AI); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2601.16225

 - **Pdf link:** https://arxiv.org/pdf/2601.16225

 - **Abstract**
 Empathetic speech dialogue requires not only understanding linguistic content but also perceiving rich paralinguistic information such as prosody, tone, and emotional intensity for affective understandings. Existing speech-to-speech large language models either rely on ASR transcription or use encoders to extract latent representations, often weakening affective information and contextual coherence in multi-turn dialogues. To address this, we propose \textbf{ES4R}, a framework for speech-based empathetic response generation. Our core innovation lies in explicitly modeling structured affective context before speech encoding, rather than relying on implicit learning by the encoder or explicit emotion supervision. Specifically, we introduce a dual-level attention mechanism to capture turn-level affective states and dialogue-level affective dynamics. The resulting affective representations are then integrated with textual semantics through speech-guided cross-modal attention to generate empathetic responses. For speech output, we employ energy-based strategy selection and style fusion to achieve empathetic speech synthesis. ES4R consistently outperforms strong baselines in both automatic and human evaluations and remains robust across different LLM backbones.
#### Zero-Shot Speech LLMs for Multi-Aspect Evaluation of L2 Speech: Challenges and Opportunities
 - **Authors:** Aditya Kamlesh Parikh, Cristian Tejedor-Garcia, Catia Cucchiarini, Helmer Strik
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Artificial Intelligence (cs.AI); Computation and Language (cs.CL); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2601.16230

 - **Pdf link:** https://arxiv.org/pdf/2601.16230

 - **Abstract**
 An accurate assessment of L2 English pronunciation is crucial for language learning, as it provides personalized feedback and ensures a fair evaluation of individual progress. However, automated scoring remains challenging due to the complexity of sentence-level fluency, prosody, and completeness. This paper evaluates the zero-shot performance of Qwen2-Audio-7B-Instruct, an instruction-tuned speech-LLM, on 5,000 Speechocean762 utterances. The model generates rubric-aligned scores for accuracy, fluency, prosody, and completeness, showing strong agreement with human ratings within +-2 tolerance, especially for high-quality speech. However, it tends to overpredict low-quality speech scores and lacks precision in error detection. These findings demonstrate the strong potential of speech LLMs in scalable pronunciation assessment and suggest future improvements through enhanced prompting, calibration, and phonetic integration to advance Computer-Assisted Pronunciation Training.
#### Test-Time Adaptation for Speech Emotion Recognition
 - **Authors:** Jiaheng Dong, Hong Jia, Ting Dang
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Machine Learning (cs.LG); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2601.16240

 - **Pdf link:** https://arxiv.org/pdf/2601.16240

 - **Abstract**
 The practical utility of Speech Emotion Recognition (SER) systems is undermined by their fragility to domain shifts, such as speaker variability, the distinction between acted and naturalistic emotions, and cross-corpus variations. While domain adaptation and fine-tuning are widely studied, they require either source data or labelled target data, which are often unavailable or raise privacy concerns in SER. Test-time adaptation (TTA) bridges this gap by adapting models at inference using only unlabeled target data. Yet, having been predominantly designed for image classification and speech recognition, the efficacy of TTA for mitigating the unique domain shifts in SER has not been investigated. In this paper, we present the first systematic evaluation and comparison covering 11 TTA methods across three representative SER tasks. The results indicate that backpropagation-free TTA methods are the most promising. Conversely, entropy minimization and pseudo-labeling generally fail, as their core assumption of a single, confident ground-truth label is incompatible with the inherent ambiguity of emotional expression. Further, no single method universally excels, and its effectiveness is highly dependent on the distributional shifts and tasks.
#### TidyVoice: A Curated Multilingual Dataset for Speaker Verification Derived from Common Voice
 - **Authors:** Aref Farhadipour, Jan Marquenie, Srikanth Madikeri, Eleanor Chodroff
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2601.16358

 - **Pdf link:** https://arxiv.org/pdf/2601.16358

 - **Abstract**
 The development of robust, multilingual speaker recognition systems is hindered by a lack of large-scale, publicly available and multilingual datasets, particularly for the read-speech style crucial for applications like anti-spoofing. To address this gap, we introduce the TidyVoice dataset derived from the Mozilla Common Voice corpus after mitigating its inherent speaker heterogeneity within the provided client IDs. TidyVoice currently contains training and test data from over 212,000 monolingual speakers (Tidy-M) and around 4,500 multilingual speakers (Tidy-X) from which we derive two distinct conditions. The Tidy-M condition contains target and non-target trials from monolingual speakers across 81 languages. The Tidy-X condition contains target and non-target trials from multilingual speakers in both same- and cross-language trials. We employ two architectures of ResNet models, achieving a 0.35% EER by fine-tuning on our comprehensive Tidy-M partition. Moreover, we show that this fine-tuning enhances the model's generalization, improving performance on unseen conversational interview data from the CANDOR corpus. The complete dataset, evaluation trials, and our models are publicly released to provide a new resource for the community.
#### FlowSE-GRPO: Training Flow Matching Speech Enhancement via Online Reinforcement Learning
 - **Authors:** Haoxu Wang, Biao Tian, Yiheng Jiang, Zexu Pan, Shengkui Zhao, Bin Ma, Daren Chen, Xiangang Li
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2601.16483

 - **Pdf link:** https://arxiv.org/pdf/2601.16483

 - **Abstract**
 Generative speech enhancement offers a promising alternative to traditional discriminative methods by modeling the distribution of clean speech conditioned on noisy inputs. Post-training alignment via reinforcement learning (RL) effectively aligns generative models with human preferences and downstream metrics in domains such as natural language processing, but its use in speech enhancement remains limited, especially for online RL. Prior work explores offline methods like Direct Preference Optimization (DPO); online methods such as Group Relative Policy Optimization (GRPO) remain largely uninvestigated. In this paper, we present the first successful integration of online GRPO into a flow-matching speech enhancement framework, enabling efficient post-training alignment to perceptual and task-oriented metrics with few update steps. Unlike prior GRPO work on Large Language Models, we adapt the algorithm to the continuous, time-series nature of speech and to the dynamics of flow-matching generative models. We show that optimizing a single reward yields rapid metric gains but often induces reward hacking that degrades audio fidelity despite higher scores. To mitigate this, we propose a multi-metric reward optimization strategy that balances competing objectives, substantially reducing overfitting and improving overall performance. Our experiments validate online GRPO for speech enhancement and provide practical guidance for RL-based post-training of generative audio models.
#### SoundBreak: A Systematic Study of Audio-Only Adversarial Attacks on Trimodal Models
 - **Authors:** Aafiya Hussain, Gaurav Srivastava, Alvi Ishmam, Zaber Hakim, Chris Thomas
 - **Subjects:** Subjects:
Sound (cs.SD); Artificial Intelligence (cs.AI); Computation and Language (cs.CL); Machine Learning (cs.LG); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2601.16231

 - **Pdf link:** https://arxiv.org/pdf/2601.16231

 - **Abstract**
 Multimodal foundation models that integrate audio, vision, and language achieve strong performance on reasoning and generation tasks, yet their robustness to adversarial manipulation remains poorly understood. We study a realistic and underexplored threat model: untargeted, audio-only adversarial attacks on trimodal audio-video-language models. We analyze six complementary attack objectives that target different stages of multimodal processing, including audio encoder representations, cross-modal attention, hidden states, and output likelihoods. Across three state-of-the-art models and multiple benchmarks, we show that audio-only perturbations can induce severe multimodal failures, achieving up to 96% attack success rate. We further show that attacks can be successful at low perceptual distortions (LPIPS <= 0.08, SI-SNR >= 0) and benefit more from extended optimization than increased data scale. Transferability across models and encoders remains limited, while speech recognition systems such as Whisper primarily respond to perturbation magnitude, achieving >97% attack success under severe distortion. These results expose a previously overlooked single-modality attack surface in multimodal systems and motivate defenses that enforce cross-modal consistency.
#### Contrastive Knowledge Distillation for Embedding Refinement in Personalized Speech Enhancement
 - **Authors:** Thomas Serre (LTCI, IP Paris), Mathieu Fontaine (LTCI, IP Paris), Éric Benhaim, Slim Essid (IDS, S2A, LTCI)
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS); Signal Processing (eess.SP)
 - **Arxiv link:** https://arxiv.org/abs/2601.16235

 - **Pdf link:** https://arxiv.org/pdf/2601.16235

 - **Abstract**
 Personalized speech enhancement (PSE) has shown convincing results when it comes to extracting a known target voice among interfering ones. The corresponding systems usually incorporate a representation of the target voice within the enhancement system, which is extracted from an enrollment clip of the target voice with upstream models. Those models are generally heavy as the speaker embedding's quality directly affects PSE performances. Yet, embeddings generated beforehand cannot account for the variations of the target voice during inference time. In this paper, we propose to perform on-thefly refinement of the speaker embedding using a tiny speaker encoder. We first introduce a novel contrastive knowledge distillation methodology in order to train a 150k-parameter encoder from complex embeddings. We then use this encoder within the enhancement system during inference and show that the proposed method greatly improves PSE performances while maintaining a low computational load.
#### The CMU-AIST submission for the ICME 2025 Audio Encoder Challenge
 - **Authors:** Shikhar Bharadwaj, Samuele Cornell, Kwanghee Choi, Hye-jin Shim, Soham Deshmukh, Satoru Fukayama, Shinji Watanabe
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2601.16273

 - **Pdf link:** https://arxiv.org/pdf/2601.16273

 - **Abstract**
 This technical report describes our submission to the ICME 2025 audio encoder challenge. Our submitted system is built on BEATs, a masked speech token prediction based audio encoder. We extend the BEATs model using 74,000 hours of data derived from various speech, music, and sound corpora and scale its architecture upto 300 million parameters. We experiment with speech-heavy and balanced pre-training mixtures to study the impact of different domains on final performance. Our submitted system consists of an ensemble of the Dasheng 1.2 billion model with two custom scaled-up BEATs models trained on the aforementioned pre-training data mixtures. We also propose a simple ensembling technique that retains the best capabilities of constituent models and surpasses both the baseline and Dasheng 1.2B. For open science, we publicly release our trained checkpoints via huggingface at this https URL and this https URL.
#### Auditory Attention Decoding without Spatial Information: A Diotic EEG Study
 - **Authors:** Masahiro Yoshino, Haruki Yokota, Junya Hara, Yuichi Tanaka, Hiroshi Higashi
 - **Subjects:** Subjects:
Signal Processing (eess.SP); Human-Computer Interaction (cs.HC); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2601.16442

 - **Pdf link:** https://arxiv.org/pdf/2601.16442

 - **Abstract**
 Auditory attention decoding (AAD) identifies the attended speech stream in multi-speaker environments by decoding brain signals such as electroencephalography (EEG). This technology is essential for realizing smart hearing aids that address the cocktail party problem and for facilitating objective audiometry systems. Existing AAD research mainly utilizes dichotic environments where different speech signals are presented to the left and right ears, enabling models to classify directional attention rather than speech content. However, this spatial reliance limits applicability to real-world scenarios, such as the "cocktail party" situation, where speakers overlap or move dynamically. To address this challenge, we propose an AAD framework for diotic environments where identical speech mixtures are presented to both ears, eliminating spatial cues. Our approach maps EEG and speech signals into a shared latent space using independent encoders. We extract speech features using wav2vec 2.0 and encode them with a 2-layer 1D convolutional neural network (CNN), while employing the BrainNetwork architecture for EEG encoding. The model identifies the attended speech by calculating the cosine similarity between EEG and speech representations. We evaluate our method on a diotic EEG dataset and achieve 72.70% accuracy, which is 22.58% higher than the state-of-the-art direction-based AAD method.
#### Do Models Hear Like Us? Probing the Representational Alignment of Audio LLMs and Naturalistic EEG
 - **Authors:** Haoyun Yang, Xin Xiao, Jiang Zhong, Yu Tian, Dong Xiaohua, Yu Mao, Hao Wu, Kaiwen Wei
 - **Subjects:** Subjects:
Sound (cs.SD); Artificial Intelligence (cs.AI); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2601.16540

 - **Pdf link:** https://arxiv.org/pdf/2601.16540

 - **Abstract**
 Audio Large Language Models (Audio LLMs) have demonstrated strong capabilities in integrating speech perception with language understanding. However, whether their internal representations align with human neural dynamics during naturalistic listening remains largely unexplored. In this work, we systematically examine layer-wise representational alignment between 12 open-source Audio LLMs and Electroencephalogram (EEG) signals across 2 datasets. Specifically, we employ 8 similarity metrics, such as Spearman-based Representational Similarity Analysis (RSA), to characterize within-sentence representational geometry. Our analysis reveals 3 key findings: (1) we observe a rank-dependence split, in which model rankings vary substantially across different similarity metrics; (2) we identify spatio-temporal alignment patterns characterized by depth-dependent alignment peaks and a pronounced increase in RSA within the 250-500 ms time window, consistent with N400-related neural dynamics; (3) we find an affective dissociation whereby negative prosody, identified using a proposed Tri-modal Neighborhood Consistency (TNC) criterion, reduces geometric similarity while enhancing covariance-based dependence. These findings provide new neurobiological insights into the representational mechanisms of Audio LLMs.
#### Omni-directional attention mechanism based on Mamba for speech separation
 - **Authors:** Ke Xue, Chang Sun, Rongfei Fan, Jing Wang, Han Hu
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2601.16603

 - **Pdf link:** https://arxiv.org/pdf/2601.16603

 - **Abstract**
 Mamba, a selective state-space model (SSM), has emerged as an efficient alternative to Transformers for speech modeling, enabling long-sequence processing with linear complexity. While effective in speech separation, existing approaches, whether in the time or time-frequency domain, typically decompose the input along a single dimension into short one-dimensional sequences before processing them with Mamba, which restricts it to local 1D modeling and limits its ability to capture global dependencies across the 2D spectrogram. In this work, we propose an efficient omni-directional attention (OA) mechanism built upon unidirectional Mamba, which models global dependencies from ten different directions on the spectrogram. We expand the proposed mechanism into two baseline separation models and evaluate on three public datasets. Experimental results show that our approach consistently achieves significant performance gains over the baselines while preserving linear complexity, outperforming existing state-of-the-art (SOTA) systems.
#### E2E-AEC: Implementing an end-to-end neural network learning approach for acoustic echo cancellation
 - **Authors:** Yiheng Jiang, Biao Tian, Haoxu Wang, Shengkui Zhao, Bin Ma, Daren Chen, Xiangang Li
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2601.16774

 - **Pdf link:** https://arxiv.org/pdf/2601.16774

 - **Abstract**
 We propose a novel neural network-based end-to-end acoustic echo cancellation (E2E-AEC) method capable of streaming inference, which operates effectively without reliance on traditional linear AEC (LAEC) techniques and time delay estimation. Our approach includes several key strategies: First, we introduce and refine progressive learning to gradually enhance echo suppression. Second, our model employs knowledge transfer by initializing with a pre-trained LAECbased model, harnessing the insights gained from LAEC training. Third, we optimize the attention mechanism with a loss function applied on attention weights to achieve precise time alignment between the reference and microphone signals. Lastly, we incorporate voice activity detection to enhance speech quality and improve echo removal by masking the network output when near-end speech is absent. The effectiveness of our approach is validated through experiments conducted on public datasets.
#### A Novel Transfer Learning Approach for Mental Stability Classification from Voice Signal
 - **Authors:** Rafiul Islam, Md. Taimur Ahad
 - **Subjects:** Subjects:
Sound (cs.SD); Neural and Evolutionary Computing (cs.NE); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2601.16793

 - **Pdf link:** https://arxiv.org/pdf/2601.16793

 - **Abstract**
 This study presents a novel transfer learning approach and data augmentation technique for mental stability classification using human voice signals and addresses the challenges associated with limited data availability. Convolutional neural networks (CNNs) have been employed to analyse spectrogram images generated from voice recordings. Three CNN architectures, VGG16, InceptionV3, and DenseNet121, were evaluated across three experimental phases: training on non-augmented data, augmented data, and transfer learning. This proposed transfer learning approach involves pre-training models on the augmented dataset and fine-tuning them on the non-augmented dataset while ensuring strict data separation to prevent data leakage. The results demonstrate significant improvements in classification performance compared to the baseline approach. Among three CNN architectures, DenseNet121 achieved the highest accuracy of 94% and an AUC score of 99% using the proposed transfer learning approach. This finding highlights the effectiveness of combining data augmentation and transfer learning to enhance CNN-based classification of mental stability using voice spectrograms, offering a promising non-invasive tool for mental health diagnostics.


by Zyzzyva0381 (Windy). 


2026-01-26
