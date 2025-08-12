# Showing new listings for Tuesday, 12 August 2025
Auto update papers at about 2:30am UTC (10:30am Beijing time) every weekday.


阅读 `Usage.md`了解如何使用此repo实现个性化的Arxiv论文推送

See `Usage.md` for instructions on how to personalize the repo. 


Keyword list: ['text-to-speech', 'text to speech', 'tts', 'LLM-based', 'speech', 'voice']


Excluded: []


### Today: 19papers 
#### FlowSE: Flow Matching-based Speech Enhancement
 - **Authors:** Seonggyu Lee, Sein Cheong, Sangwook Han, Jong Won Shin
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Signal Processing (eess.SP)
 - **Arxiv link:** https://arxiv.org/abs/2508.06840

 - **Pdf link:** https://arxiv.org/pdf/2508.06840

 - **Abstract**
 Diffusion probabilistic models have shown impressive performance for speech enhancement, but they typically require 25 to 60 function evaluations in the inference phase, resulting in heavy computational complexity. Recently, a fine-tuning method was proposed to correct the reverse process, which significantly lowered the number of function evaluations (NFE). Flow matching is a method to train continuous normalizing flows which model probability paths from known distributions to unknown distributions including those described by diffusion processes. In this paper, we propose a speech enhancement based on conditional flow matching. The proposed method achieved the performance comparable to those for the diffusion-based speech enhancement with the NFE of 60 when the NFE was 5, and showed similar performance with the diffusion model correcting the reverse process at the same NFE from 1 to 5 without additional fine tuning procedure. We also have shown that the corresponding diffusion model derived from the conditional probability path with a modified optimal transport conditional vector field demonstrated similar performances with the NFE of 5 without any fine-tuning procedure.
#### Speech Enhancement based on cascaded two flow
 - **Authors:** Seonggyu Lee, Sein Cheong, Sangwook Han, Kihyuk Kim, Jong Won Shi
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Signal Processing (eess.SP)
 - **Arxiv link:** https://arxiv.org/abs/2508.06842

 - **Pdf link:** https://arxiv.org/pdf/2508.06842

 - **Abstract**
 Speech enhancement (SE) based on diffusion probabilistic models has exhibited impressive performance, while requiring a relatively high number of function evaluations (NFE). Recently, SE based on flow matching has been proposed, which showed competitive performance with a small NFE. Early approaches adopted the noisy speech as the only conditioning variable. There have been other approaches which utilize speech enhanced with a predictive model as another conditioning variable and to sample an initial value, but they require a separate predictive model on top of the generative SE model. In this work, we propose to employ an identical model based on flow matching for both SE and generating enhanced speech used as an initial starting point and a conditioning variable. Experimental results showed that the proposed method required the same or fewer NFEs even with two cascaded generative methods while achieving equivalent or better performances to the previous baselines.
#### TurboBias: Universal ASR Context-Biasing powered by GPU-accelerated Phrase-Boosting Tree
 - **Authors:** Andrei Andrusenko, Vladimir Bataev, Lilit Grigoryan, Vitaly Lavrukhin, Boris Ginsburg
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Artificial Intelligence (cs.AI); Computation and Language (cs.CL); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2508.07014

 - **Pdf link:** https://arxiv.org/pdf/2508.07014

 - **Abstract**
 Recognizing specific key phrases is an essential task for contextualized Automatic Speech Recognition (ASR). However, most existing context-biasing approaches have limitations associated with the necessity of additional model training, significantly slow down the decoding process, or constrain the choice of the ASR system type. This paper proposes a universal ASR context-biasing framework that supports all major types: CTC, Transducers, and Attention Encoder-Decoder models. The framework is based on a GPU-accelerated word boosting tree, which enables it to be used in shallow fusion mode for greedy and beam search decoding without noticeable speed degradation, even with a vast number of key phrases (up to 20K items). The obtained results showed high efficiency of the proposed method, surpassing the considered open-source context-biasing approaches in accuracy and decoding speed. Our context-biasing framework is open-sourced as a part of the NeMo toolkit.
#### ParaNoise-SV: Integrated Approach for Noise-Robust Speaker Verification with Parallel Joint Learning of Speech Enhancement and Noise Extraction
 - **Authors:** Minu Kim, Kangwook Jang, Hoirin Kim
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2508.07219

 - **Pdf link:** https://arxiv.org/pdf/2508.07219

 - **Abstract**
 Noise-robust speaker verification leverages joint learning of speech enhancement (SE) and speaker verification (SV) to improve robustness. However, prevailing approaches rely on implicit noise suppression, which struggles to separate noise from speaker characteristics as they do not explicitly distinguish noise from speech during training. Although integrating SE and SV helps, it remains limited in handling noise effectively. Meanwhile, recent SE studies suggest that explicitly modeling noise, rather than merely suppressing it, enhances noise resilience. Reflecting this, we propose ParaNoise-SV, with dual U-Nets combining a noise extraction (NE) network and a speech enhancement (SE) network. The NE U-Net explicitly models noise, while the SE U-Net refines speech with guidance from NE through parallel connections, preserving speaker-relevant features. Experimental results show that ParaNoise-SV achieves a relatively 8.4% lower equal error rate (EER) than previous joint SE-SV models.
#### Lessons Learnt: Revisit Key Training Strategies for Effective Speech Emotion Recognition in the Wild
 - **Authors:** Jing-Tong Tzeng, Bo-Hao Su, Ya-Tse Wu, Hsing-Hang Chou, Chi-Chun Lee
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2508.07282

 - **Pdf link:** https://arxiv.org/pdf/2508.07282

 - **Abstract**
 In this study, we revisit key training strategies in machine learning often overlooked in favor of deeper architectures. Specifically, we explore balancing strategies, activation functions, and fine-tuning techniques to enhance speech emotion recognition (SER) in naturalistic conditions. Our findings show that simple modifications improve generalization with minimal architectural changes. Our multi-modal fusion model, integrating these optimizations, achieves a valence CCC of 0.6953, the best valence score in Task 2: Emotional Attribute Regression. Notably, fine-tuning RoBERTa and WavLM separately in a single-modality setting, followed by feature fusion without training the backbone extractor, yields the highest valence performance. Additionally, focal loss and activation functions significantly enhance performance without increasing complexity. These results suggest that refining core components, rather than deepening models, leads to more robust SER in-the-wild.
#### A Survey on Non-Intrusive ASR Refinement: From Output-Level Correction to Full-Model Distillation
 - **Authors:** Mohammad Reza Peyghan, Fatemeh Rajabi, Saman Soleimani Roudi, Saeedreza Zouashkiani, Sajjad Amini, Shahrokh Ghaemmaghami
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2508.07285

 - **Pdf link:** https://arxiv.org/pdf/2508.07285

 - **Abstract**
 Automatic Speech Recognition (ASR) has become an integral component of modern technology, powering applications such as voice-activated assistants, transcription services, and accessibility tools. Yet ASR systems continue to struggle with the inherent variability of human speech, such as accents, dialects, and speaking styles, as well as environmental interference, including background noise. Moreover, domain-specific conversations often employ specialized terminology, which can exacerbate transcription errors. These shortcomings not only degrade raw ASR accuracy but also propagate mistakes through subsequent natural language processing pipelines. Because redesigning an ASR model is costly and time-consuming, non-intrusive refinement techniques that leave the model's architecture unchanged have become increasingly popular. In this survey, we systematically review current non-intrusive refinement approaches and group them into five classes: fusion, re-scoring, correction, distillation, and training adjustment. For each class, we outline the main methods, advantages, drawbacks, and ideal application scenarios. Beyond method classification, this work surveys adaptation techniques aimed at refining ASR in domain-specific contexts, reviews commonly used evaluation datasets along with their construction processes, and proposes a standardized set of metrics to facilitate fair comparisons. Finally, we identify open research gaps and suggest promising directions for future work. By providing this structured overview, we aim to equip researchers and practitioners with a clear foundation for developing more robust, accurate ASR refinement pipelines.
#### XEmoRAG: Cross-Lingual Emotion Transfer with Controllable Intensity Using Retrieval-Augmented Generation
 - **Authors:** Tianlun Zuo, Jingbin Hu, Yuke Li, Xinfa Zhu, Hai Li, Ying Yan, Junhui Liu, Danming Xie, Lei Xie
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2508.07302

 - **Pdf link:** https://arxiv.org/pdf/2508.07302

 - **Abstract**
 Zero-shot emotion transfer in cross-lingual speech synthesis refers to generating speech in a target language, where the emotion is expressed based on reference speech from a different source this http URL, this task remains challenging due to the scarcity of parallel multilingual emotional corpora, the presence of foreign accent artifacts, and the difficulty of separating emotion from language-specific prosodic this http URL this paper, we propose XEmoRAG, a novel framework to enable zero-shot emotion transfer from Chinese to Thai using a large language model (LLM)-based model, without relying on parallel emotional this http URL extracts language-agnostic emotional embeddings from Chinese speech and retrieves emotionally matched Thai utterances from a curated emotional database, enabling controllable emotion transfer without explicit emotion labels. Additionally, a flow-matching alignment module minimizes pitch and duration mismatches, ensuring natural prosody. It also blends Chinese timbre into the Thai synthesis, enhancing rhythmic accuracy and emotional expression, while preserving speaker characteristics and emotional this http URL results show that XEmoRAG synthesizes expressive and natural Thai speech using only Chinese reference audio, without requiring explicit emotion this http URL results highlight XEmoRAG's capability to achieve flexible and low-resource emotional transfer across this http URL demo is available at this https URL.
#### FlexCTC: GPU-powered CTC Beam Decoding with advanced Contextual Abilities
 - **Authors:** Lilit Grigoryan, Vladimir Bataev, Nikolay Karpov, Andrei Andrusenko, Vitaly Lavrukhin, Boris Ginsburg
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Artificial Intelligence (cs.AI); Computation and Language (cs.CL); Machine Learning (cs.LG); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2508.07315

 - **Pdf link:** https://arxiv.org/pdf/2508.07315

 - **Abstract**
 While beam search improves speech recognition quality over greedy decoding, standard implementations are slow, often sequential, and CPU-bound. To fully leverage modern hardware capabilities, we present a novel open-source FlexCTC toolkit for fully GPU-based beam decoding, designed for Connectionist Temporal Classification (CTC) models. Developed entirely in Python and PyTorch, it offers a fast, user-friendly, and extensible alternative to traditional C++, CUDA, or WFST-based decoders. The toolkit features a high-performance, fully batched GPU implementation with eliminated CPU-GPU synchronization and minimized kernel launch overhead via CUDA Graphs. It also supports advanced contextualization techniques, including GPU-powered N-gram language model fusion and phrase-level boosting. These features enable accurate and efficient decoding, making them suitable for both research and production use.
#### KLASSify to Verify: Audio-Visual Deepfake Detection Using SSL-based Audio and Handcrafted Visual Features
 - **Authors:** Ivan Kukanov, Jun Wah Ng
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Computer Vision and Pattern Recognition (cs.CV)
 - **Arxiv link:** https://arxiv.org/abs/2508.07337

 - **Pdf link:** https://arxiv.org/pdf/2508.07337

 - **Abstract**
 The rapid development of audio-driven talking head generators and advanced Text-To-Speech (TTS) models has led to more sophisticated temporal deepfakes. These advances highlight the need for robust methods capable of detecting and localizing deepfakes, even under novel, unseen attack scenarios. Current state-of-the-art deepfake detectors, while accurate, are often computationally expensive and struggle to generalize to novel manipulation techniques. To address these challenges, we propose multimodal approaches for the AV-Deepfake1M 2025 challenge. For the visual modality, we leverage handcrafted features to improve interpretability and adaptability. For the audio modality, we adapt a self-supervised learning (SSL) backbone coupled with graph attention networks to capture rich audio representations, improving detection robustness. Our approach strikes a balance between performance and real-world deployment, focusing on resilience and potential interpretability. On the AV-Deepfake1M++ dataset, our multimodal system achieves AUC of 92.78% for deepfake classification task and IoU of 0.3536 for temporal localization using only the audio modality.
#### Scalable Controllable Accented TTS
 - **Authors:** Henry Li Xinyuan, Zexin Cai, Ashi Garg, Kevin Duh, Leibny Paola García-Perera, Sanjeev Khudanpur, Nicholas Andrews, Matthew Wiesner
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2508.07426

 - **Pdf link:** https://arxiv.org/pdf/2508.07426

 - **Abstract**
 We tackle the challenge of scaling accented TTS systems, expanding their capabilities to include much larger amounts of training data and a wider variety of accent labels, even for accents that are poorly represented or unlabeled in traditional TTS datasets. To achieve this, we employ two strategies: 1. Accent label discovery via a speech geolocation model, which automatically infers accent labels from raw speech data without relying solely on human annotation; 2. Timbre augmentation through kNN voice conversion to increase data diversity and model robustness. These strategies are validated on CommonVoice, where we fine-tune XTTS-v2 for accented TTS with accent labels discovered or enhanced using geolocation. We demonstrate that the resulting accented TTS model not only outperforms XTTS-v2 fine-tuned on self-reported accent labels in CommonVoice, but also existing accented TTS benchmarks.
#### UniFlow: Unifying Speech Front-End Tasks via Continuous Generative Modeling
 - **Authors:** Ziqian Wang, Zikai Liu, Yike Zhu, Xingchen Li, Boyi Kang, Jixun Yao, Xianjun Xia, Chuanzeng Huang, Lei Xie
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2508.07558

 - **Pdf link:** https://arxiv.org/pdf/2508.07558

 - **Abstract**
 Generative modeling has recently achieved remarkable success across image, video, and audio domains, demonstrating powerful capabilities for unified representation learning. Yet speech front-end tasks such as speech enhancement (SE), target speaker extraction (TSE), acoustic echo cancellation (AEC), and language-queried source separation (LASS) remain largely tackled by disparate, task-specific solutions. This fragmentation leads to redundant engineering effort, inconsistent performance, and limited extensibility. To address this gap, we introduce UniFlow, a unified framework that employs continuous generative modeling to tackle diverse speech front-end tasks in a shared latent space. Specifically, UniFlow utilizes a waveform variational autoencoder (VAE) to learn a compact latent representation of raw audio, coupled with a Diffusion Transformer (DiT) that predicts latent updates. To differentiate the speech processing task during the training, learnable condition embeddings indexed by a task ID are employed to enable maximal parameter sharing while preserving task-specific adaptability. To balance model performance and computational efficiency, we investigate and compare three generative objectives: denoising diffusion, flow matching, and mean flow within the latent domain. We validate UniFlow on multiple public benchmarks, demonstrating consistent gains over state-of-the-art baselines. UniFlow's unified latent formulation and conditional design make it readily extensible to new tasks, providing an integrated foundation for building and scaling generative speech processing pipelines. To foster future research, we will open-source our codebase.
#### Is GAN Necessary for Mel-Spectrogram-based Neural Vocoder?
 - **Authors:** Hui-Peng Du, Yang Ai, Rui-Chen Zheng, Ye-Xin Lu, Zhen-Hua Ling
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2508.07711

 - **Pdf link:** https://arxiv.org/pdf/2508.07711

 - **Abstract**
 Recently, mainstream mel-spectrogram-based neural vocoders rely on generative adversarial network (GAN) for high-fidelity speech generation, e.g., HiFi-GAN and BigVGAN. However, the use of GAN restricts training efficiency and model complexity. Therefore, this paper proposes a novel FreeGAN vocoder, aiming to answer the question of whether GAN is necessary for mel-spectrogram-based neural vocoders. The FreeGAN employs an amplitude-phase serial prediction framework, eliminating the need for GAN training. It incorporates amplitude prior input, SNAKE-ConvNeXt v2 backbone and frequency-weighted anti-wrapping phase loss to compensate for the performance loss caused by the absence of GAN. Experimental results confirm that the speech quality of FreeGAN is comparable to that of advanced GAN-based vocoders, while significantly improving training efficiency and complexity. Other explicit-phase-prediction-based neural vocoders can also work without GAN, leveraging our proposed methods.
#### G-IFT: A Gated Linear Unit adapter with Iterative Fine-Tuning for Low-Resource Children's Speaker Verification
 - **Authors:** Vishwas M. Shetty, Jiusi Zheng, Abeer Alwan
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Machine Learning (cs.LG)
 - **Arxiv link:** https://arxiv.org/abs/2508.07836

 - **Pdf link:** https://arxiv.org/pdf/2508.07836

 - **Abstract**
 Speaker Verification (SV) systems trained on adults speech often underperform on children's SV due to the acoustic mismatch, and limited children speech data makes fine-tuning not very effective. In this paper, we propose an innovative framework, a Gated Linear Unit adapter with Iterative Fine-Tuning (G-IFT), to enhance knowledge transfer efficiency between the high-resource adults speech domain and the low-resource children's speech domain. In this framework, a Gated Linear Unit adapter is first inserted between the pre-trained speaker embedding model and the classifier. Then the classifier, adapter, and pre-trained speaker embedding model are optimized sequentially in an iterative way. This framework is agnostic to the type of the underlying architecture of the SV system. Our experiments on ECAPA-TDNN, ResNet, and X-vector architectures using the OGI and MyST datasets demonstrate that the G-IFT framework yields consistent reductions in Equal Error Rates compared to baseline methods.
#### MSU-Bench: Towards Understanding the Conversational Multi-talker Scenarios
 - **Authors:** Shuai Wang, Zhaokai Sun, Zhennan Lin, Chengyou Wang, Zhou Pan, Lei Xie
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2508.08155

 - **Pdf link:** https://arxiv.org/pdf/2508.08155

 - **Abstract**
 Spoken Language Understanding (SLU) has progressed from traditional single-task methods to large audio language model (LALM) solutions. Yet, most existing speech benchmarks focus on single-speaker or isolated tasks, overlooking the challenges posed by multi-speaker conversations that are common in real-world scenarios. We introduce MSU-Bench, a comprehensive benchmark for evaluating multi-speaker conversational understanding with a speaker-centric design. Our hierarchical framework covers four progressive tiers: single-speaker static attribute understanding, single-speaker dynamic attribute understanding, multi-speaker background understanding, and multi-speaker interaction understanding. This structure ensures all tasks are grounded in speaker-centric contexts, from basic perception to complex reasoning across multiple speakers. By evaluating state-of-the-art models on MSU-Bench, we demonstrate that as task complexity increases across the benchmark's tiers, all models exhibit a significant performance decline. We also observe a persistent capability gap between open-source models and closed-source commercial ones, particularly in multi-speaker interaction reasoning. These findings validate the effectiveness of MSU-Bench for assessing and advancing conversational understanding in realistic multi-speaker environments. Demos can be found in the supplementary material.
#### How Does a Deep Neural Network Look at Lexical Stress?
 - **Authors:** Itai Allouche, Itay Asael, Rotem Rousso, Vered Dassa, Ann Bradlow, Seung-Eun Kim, Matthew Goldrick, Joseph Keshet
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Machine Learning (cs.LG); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2508.07229

 - **Pdf link:** https://arxiv.org/pdf/2508.07229

 - **Abstract**
 Despite their success in speech processing, neural networks often operate as black boxes, prompting the question: what informs their decisions, and how can we interpret them? This work examines this issue in the context of lexical stress. A dataset of English disyllabic words was automatically constructed from read and spontaneous speech. Several Convolutional Neural Network (CNN) architectures were trained to predict stress position from a spectrographic representation of disyllabic words lacking minimal stress pairs (e.g., initial stress WAllet, final stress exTEND), achieving up to 92% accuracy on held-out test data. Layerwise Relevance Propagation (LRP), a technique for CNN interpretability analysis, revealed that predictions for held-out minimal pairs (PROtest vs. proTEST ) were most strongly influenced by information in stressed versus unstressed syllables, particularly the spectral properties of stressed vowels. However, the classifiers also attended to information throughout the word. A feature-specific relevance analysis is proposed, and its results suggest that our best-performing classifier is strongly influenced by the stressed vowel's first and second formants, with some evidence that its pitch and third formant also contribute. These results reveal deep learning's ability to acquire distributed cues to stress from naturally occurring data, extending traditional phonetic work based around highly controlled stimuli.
#### Incorporating Contextual Paralinguistic Understanding in Large Speech-Language Models
 - **Authors:** Qiongqiong Wang, Hardik B. Sailor, Jeremy H. M. Wong, Tianchi Liu, Shuo Sun, Wenyu Zhang, Muhammad Huzaifah, Nancy Chen, Ai Ti Aw
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Artificial Intelligence (cs.AI); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2508.07273

 - **Pdf link:** https://arxiv.org/pdf/2508.07273

 - **Abstract**
 Current large speech language models (Speech-LLMs) often exhibit limitations in empathetic reasoning, primarily due to the absence of training datasets that integrate both contextual content and paralinguistic cues. In this work, we propose two approaches to incorporate contextual paralinguistic information into model training: (1) an explicit method that provides paralinguistic metadata (e.g., emotion annotations) directly to the LLM, and (2) an implicit method that automatically generates novel training question-answer (QA) pairs using both categorical and dimensional emotion annotations alongside speech transcriptions. Our implicit method boosts performance (LLM-judged) by 38.41% on a human-annotated QA benchmark, reaching 46.02% when combined with the explicit approach, showing effectiveness in contextual paralinguistic understanding. We also validate the LLM judge by demonstrating its correlation with classification metrics, providing support for its reliability.
#### Keyword Mamba: Spoken Keyword Spotting with State Space Models
 - **Authors:** Hanyu Ding, Wenlong Dong, Qirong Mao
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2508.07363

 - **Pdf link:** https://arxiv.org/pdf/2508.07363

 - **Abstract**
 Keyword spotting (KWS) is an essential task in speech processing. It is widely used in voice assistants and smart devices. Deep learning models like CNNs, RNNs, and Transformers have performed well in KWS. However, they often struggle to handle long-term patterns and stay efficient at the same time. In this work, we present Keyword Mamba, a new architecture for KWS. It uses a neural state space model (SSM) called Mamba. We apply Mamba along the time axis and also explore how it can replace the self-attention part in Transformer models. We test our model on the Google Speech Commands datasets. The results show that Keyword Mamba reaches strong accuracy with fewer parameters and lower computational cost. To our knowledge, this is the first time a state space model has been used for KWS. These results suggest that Mamba has strong potential in speech-related tasks.
#### Think Before You Talk: Enhancing Meaningful Dialogue Generation in Full-Duplex Speech Language Models with Planning-Inspired Text Guidance
 - **Authors:** Wenqian Cui, Lei Zhu, Xiaohui Li, Zhihan Guo, Haoli Bai, Lu Hou, Irwin King
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2508.07375

 - **Pdf link:** https://arxiv.org/pdf/2508.07375

 - **Abstract**
 Full-Duplex Speech Language Models (FD-SLMs) are specialized foundation models designed to enable natural, real-time spoken interactions by modeling complex conversational dynamics such as interruptions, backchannels, and overlapping speech, and End-to-end (e2e) FD-SLMs leverage real-world double-channel conversational data to capture nuanced two-speaker dialogue patterns for human-like interactions. However, they face a critical challenge -- their conversational abilities often degrade compared to pure-text conversation due to prolonged speech sequences and limited high-quality spoken dialogue data. While text-guided speech generation could mitigate these issues, it suffers from timing and length issues when integrating textual guidance into double-channel audio streams, disrupting the precise time alignment essential for natural interactions. To address these challenges, we propose TurnGuide, a novel planning-inspired approach that mimics human conversational planning by dynamically segmenting assistant speech into dialogue turns and generating turn-level text guidance before speech output, which effectively resolves both insertion timing and length challenges. Extensive experiments demonstrate our approach significantly improves e2e FD-SLMs' conversational abilities, enabling them to generate semantically meaningful and coherent speech while maintaining natural conversational flow. Demos are available at this https URL. Code will be available at this https URL.
#### Bridging ASR and LLMs for Dysarthric Speech Recognition: Benchmarking Self-Supervised and Generative Approaches
 - **Authors:** Ahmed Aboeitta, Ahmed Sharshar, Youssef Nafea, Shady Shehata
 - **Subjects:** Subjects:
Sound (cs.SD); Artificial Intelligence (cs.AI); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2508.08027

 - **Pdf link:** https://arxiv.org/pdf/2508.08027

 - **Abstract**
 Speech Recognition (ASR) due to phoneme distortions and high variability. While self-supervised ASR models like Wav2Vec, HuBERT, and Whisper have shown promise, their effectiveness in dysarthric speech remains unclear. This study systematically benchmarks these models with different decoding strategies, including CTC, seq2seq, and LLM-enhanced decoding (BART,GPT-2, Vicuna). Our contributions include (1) benchmarking ASR architectures for dysarthric speech, (2) introducing LLM-based decoding to improve intelligibility, (3) analyzing generalization across datasets, and (4) providing insights into recognition errors across severity levels. Findings highlight that LLM-enhanced decoding improves dysarthric ASR by leveraging linguistic constraints for phoneme restoration and grammatical correction.


by Zyzzyva0381 (Windy). 


2025-08-12
