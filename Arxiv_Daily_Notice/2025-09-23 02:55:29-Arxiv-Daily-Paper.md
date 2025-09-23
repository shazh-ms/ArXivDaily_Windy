# Showing new listings for Tuesday, 23 September 2025
Auto update papers at about 2:30am UTC (10:30am Beijing time) every weekday.


阅读 `Usage.md`了解如何使用此repo实现个性化的Arxiv论文推送

See `Usage.md` for instructions on how to personalize the repo. 


Keyword list: ['text-to-speech', 'text to speech', 'tts', 'LLM-based', 'speech', 'voice']


Excluded: []


### Today: 29papers 
#### Investigating Polyglot Speech Foundation Models for Learning Collective Emotion from Crowds
 - **Authors:** Orchid Chetia Phukan, Girish, Mohd Mujtaba Akhtar, Panchal Nayak, Priyabrata Mallick, Swarup Ranjan Behera, Parabattina Bhagath, Pailla Balakrishna Reddy, Arun Balaji Buduru
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2509.16329

 - **Pdf link:** https://arxiv.org/pdf/2509.16329

 - **Abstract**
 This paper investigates the polyglot (multilingual) speech foundation models (SFMs) for Crowd Emotion Recognition (CER). We hypothesize that polyglot SFMs, pre-trained on diverse languages, accents, and speech patterns, are particularly adept at navigating the noisy and complex acoustic environments characteristic of crowd settings, thereby offering a significant advantage for CER. To substantiate this, we perform a comprehensive analysis, comparing polyglot, monolingual, and speaker recognition SFMs through extensive experiments on a benchmark CER dataset across varying audio durations (1 sec, 500 ms, and 250 ms). The results consistently demonstrate the superiority of polyglot SFMs, outperforming their counterparts across all audio lengths and excelling even with extremely short-duration inputs. These findings pave the way for adaptation of SFMs in setting up new benchmarks for CER.
#### Harmonic Summation-Based Robust Pitch Estimation in Noisy and Reverberant Environments
 - **Authors:** Anup Singh, Kris Demuynck
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2509.16480

 - **Pdf link:** https://arxiv.org/pdf/2509.16480

 - **Abstract**
 Accurate pitch estimation is essential for numerous speech processing applications, yet it remains challenging in high-distortion environments. This paper proposes a robust pitch estimation method that delivers robust pitch estimates in challenging noise environments. Our approach computes the Normalized Average Magnitude Difference Function (NAMDF), transforms it into a likelihood function, and generates probabilistic pitch states for frames at each sample shift. To enhance noise robustness, we aggregate likelihood values across integer multiples of the pitch period and neighboring frames. Furthermore, we introduce a simple yet effective continuity constraint in the Viterbi algorithm to refine pitch selection among multiple candidates. Experimental results show that our method consistently achieves lower Gross Pitch Error (GPE) and Voicing Decision Error (VDE) across various SNR levels, outperforming existing methods in both noisy and reverberant conditions.
#### TF-CorrNet: Leveraging Spatial Correlation for Continuous Speech Separation
 - **Authors:** Ui-Hyeop Shin, Bon Hyeok Ku, Hyung-Min Park
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2509.16481

 - **Pdf link:** https://arxiv.org/pdf/2509.16481

 - **Abstract**
 In general, multi-channel source separation has utilized inter-microphone phase differences (IPDs) concatenated with magnitude information in time-frequency domain, or real and imaginary components stacked along the channel axis. However, the spatial information of a sound source is fundamentally contained in the differences between microphones, specifically in the correlation between them, while the power of each microphone also provides valuable information about the source spectrum, which is why the magnitude is also included. Therefore, we propose a network that directly leverages a correlation input with phase transform (PHAT)-beta to estimate the separation filter. In addition, the proposed TF-CorrNet processes the features alternately across time and frequency axes as a dual-path strategy in terms of spatial information. Furthermore, we add a spectral module to model source-related direct time-frequency patterns for improved speech separation. Experimental results demonstrate that the proposed TF-CorrNet effectively separates the speech sounds, showing high performance with a low computational cost in the LibriCSS dataset.
#### Audio-Conditioned Diffusion LLMs for ASR and Deliberation Processing
 - **Authors:** Mengqi Wang, Zhan Liu, Zengrui Jin, Guangzhi Sun, Chao Zhang, Philip C. Woodland
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Artificial Intelligence (cs.AI); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2509.16622

 - **Pdf link:** https://arxiv.org/pdf/2509.16622

 - **Abstract**
 Diffusion-based large language models (DLLMs) have recently attracted growing interest as an alternative to autoregressive decoders. In this work, we present an empirical study on using the diffusion-based large language model LLaDA for automatic speech recognition (ASR). We first investigate its use as an external deliberation-based processing module for Whisper-LLaMA transcripts. By leveraging the bidirectional attention and denoising capabilities of LLaDA, we explore random masking, low-confidence masking, and semi-autoregressive strategies, showing that Whisper-LLaDA substantially reduces WER compared with the baseline. On LibriSpeech, the best cascade system achieves 2.25%/4.94% WER on test-clean/test-other, representing a 12.3% relative improvement over the Whisper-LLaMA baseline on the test-other split. In contrast, a plain-text LLaDA without acoustic features fails to improve accuracy, highlighting the importance of audio-conditioned embeddings. We further evaluate Whisper-LLaDA as a standalone decoder for ASR with diffusion-based and semi-autoregressive decoding. Most experimental configurations achieve faster inference than the Whisper-LLaMA baseline, although recognition accuracy is slightly lower. These findings offer an empirical view of diffusion-based LLMs for ASR and point to promising directions for improvements.
#### Reverse Attention for Lightweight Speech Enhancement on Edge Devices
 - **Authors:** Shuubham Ojha, Felix Gervits, Carol Espy-Wilson
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2509.16705

 - **Pdf link:** https://arxiv.org/pdf/2509.16705

 - **Abstract**
 This paper introduces a lightweight deep learning model for real-time speech enhancement, designed to operate efficiently on resource-constrained devices. The proposed model leverages a compact architecture that facilitates rapid inference without compromising performance. Key contributions include infusing soft attention-based attention gates in the U-Net architecture which is known to perform well for segmentation tasks and is optimized for GPUs. Experimental evaluations demonstrate that the model achieves competitive speech quality and intelligibility metrics, such as PESQ and Word Error Rates (WER), improving the performance of similarly sized baseline models. We are able to achieve a 6.24% WER improvement and a 0.64 PESQ score improvement over un-enhanced waveforms.
#### QASTAnet: A DNN-based Quality Metric for Spatial Audio
 - **Authors:** Adrien Llave, Emma Granier, Grégory Pallone
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Machine Learning (cs.LG)
 - **Arxiv link:** https://arxiv.org/abs/2509.16715

 - **Pdf link:** https://arxiv.org/pdf/2509.16715

 - **Abstract**
 In the development of spatial audio technologies, reliable and shared methods for evaluating audio quality are essential. Listening tests are currently the standard but remain costly in terms of time and resources. Several models predicting subjective scores have been proposed, but they do not generalize well to real-world signals. In this paper, we propose QASTAnet (Quality Assessment for SpaTial Audio network), a new metric based on a deep neural network, specialized on spatial audio (ambisonics and binaural). As training data is scarce, we aim for the model to be trainable with a small amount of data. To do so, we propose to rely on expert modeling of the low-level auditory system and use a neurnal network to model the high-level cognitive function of the quality judgement. We compare its performance to two reference metrics on a wide range of content types (speech, music, ambiance, anechoic, reverberated) and focusing on codec artifacts. Results demonstrate that QASTAnet overcomes the aforementioned limitations of the existing methods. The strong correlation between the proposed metric prediction and subjective scores makes it a good candidate for comparing codecs in their development.
#### DroFiT: A Lightweight Band-fused Frequency Attention Toward Real-time UAV Speech Enhancement
 - **Authors:** Jeongmin Lee, Chanhong Jeon, Hyungjoo Seo, Taewook Kang
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2509.16945

 - **Pdf link:** https://arxiv.org/pdf/2509.16945

 - **Abstract**
 This paper proposes DroFiT (Drone Frequency lightweight Transformer for speech enhancement, a single microphone speech enhancement network for severe drone self-noise. DroFit integrates a frequency-wise Transformer with a full/sub-band hybrid encoder-decoder and a TCN back-end for memory-efficient streaming. A learnable skip-and-gate fusion with a combined spectral-temporal loss further refines reconstruction. The model is trained on VoiceBank-DEMAND mixed with recorded drone noise (-5 to -25 dB SNR) and evaluate using standard speech enhancement metrics and computational efficiency. Experimental results show that DroFiT achieves competitive enhancement performance while significantly reducing computational and memory demands, paving the way for real-time processing on resource-constrained UAV platforms. Audio demo samples are available on our demo page.
#### MaskVCT: Masked Voice Codec Transformer for Zero-Shot Voice Conversion With Increased Controllability via Multiple Guidances
 - **Authors:** Junhyeok Lee, Helin Wang, Yaohan Guan, Thomas Thebaud, Laureano Moro-Velazquez, Jesús Villalba, Najim Dehak
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Artificial Intelligence (cs.AI)
 - **Arxiv link:** https://arxiv.org/abs/2509.17143

 - **Pdf link:** https://arxiv.org/pdf/2509.17143

 - **Abstract**
 We introduce MaskVCT, a zero-shot voice conversion (VC) model that offers multi-factor controllability through multiple classifier-free guidances (CFGs). While previous VC models rely on a fixed conditioning scheme, MaskVCT integrates diverse conditions in a single model. To further enhance robustness and control, the model can leverage continuous or quantized linguistic features to enhance intellgibility and speaker similarity, and can use or omit pitch contour to control prosody. These choices allow users to seamlessly balance speaker identity, linguistic content, and prosodic factors in a zero-shot VC setting. Extensive experiments demonstrate that MaskVCT achieves the best target speaker and accent similarities while obtaining competitive word and character error rates compared to existing baselines. Audio samples are available at this https URL.
#### Reference-aware SFM layers for intrusive intelligibility prediction
 - **Authors:** Hanlin Yu, Haoshuai Zhou, Boxuan Cao, Changgeng Mo, Linkai Li, Shan X. Wang
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2509.17270

 - **Pdf link:** https://arxiv.org/pdf/2509.17270

 - **Abstract**
 Intrusive speech-intelligibility predictors that exploit explicit reference signals are now widespread, yet they have not consistently surpassed non-intrusive systems. We argue that a primary cause is the limited exploitation of speech foundation models (SFMs). This work revisits intrusive prediction by combining reference conditioning with multi-layer SFM representations. Our final system achieves RMSE 22.36 on the development set and 24.98 on the evaluation set, ranking 1st on CPC3. These findings provide practical guidance for constructing SFM-based intrusive intelligibility predictors.
#### RADE for Land Mobile Radio: A Neural Codec for Transmission of Speech over Baseband FM Radio Channels
 - **Authors:** David Rowe, Tibor Bece
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2509.17286

 - **Pdf link:** https://arxiv.org/pdf/2509.17286

 - **Abstract**
 In the 1990s Land Mobile Radio (LMR) systems evolved from analog frequency modulation (FM) to standardised digital systems. Both digital and analog FM systems now co-exist in various services and exhibit similar speech quality. The architecture of many digital radios retains the analog FM modulator and demodulator from legacy analog radios, but driven by a multi-level digital pulse train rather than an analog voice signal. We denote this architecture baseband FM (BBFM). In this paper we describe a modern machine learning approach that uses an autoencoder to send high quality, 8 kHz bandwidth speech over the BBFM channel. The speech quality is shown to be superior to analog FM over simulated LMR channels in the presence of fading, and a demonstration of the system running over commodity UHF radios is presented.
#### FUN-SSL: Full-band Layer Followed by U-Net with Narrow-band Layers for Multiple Moving Sound Source Localization
 - **Authors:** Yuseon Choi, Hyeonseung Kim, Jewoo Jun, Jong Won Shin
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Signal Processing (eess.SP)
 - **Arxiv link:** https://arxiv.org/abs/2509.17490

 - **Pdf link:** https://arxiv.org/pdf/2509.17490

 - **Abstract**
 Dual-path processing along the temporal and spectral dimensions has shown to be effective in various speech processing applications. While the sound source localization (SSL) models utilizing dual-path processing such as the FN-SSL and IPDnet demonstrated impressive performances in localizing multiple moving sources, they require significant amount of computation. In this paper, we propose an architecture for SSL which introduces a U-Net to perform narrow-band processing in multiple resolutions to reduce computational complexity. The proposed model replaces the full-narrow network block in the IPDnet consisting of one full-band LSTM layer along the spectral dimension followed by one narrow-band LSTM layer along the temporal dimension with the FUN block composed of one Full-band layer followed by a U-net with Narrow-band layers in multiple scales. On top of the skip connections within each U-Net, we also introduce the skip connections between FUN blocks to enrich information. Experimental results showed that the proposed FUN-SSL outperformed previously proposed approaches with computational complexity much lower than that of the IPDnet.
#### Audiobook-CC: Controllable Long-context Speech Generation for Multicast Audiobook
 - **Authors:** Min Liu, JingJing Yin, Xiang Zhang, Siyu Hao, Yanni Hu, Bin Lin, Yuan Feng, Hongbin Zhou, Jianhao Ye
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2509.17516

 - **Pdf link:** https://arxiv.org/pdf/2509.17516

 - **Abstract**
 Existing text-to-speech systems predominantly focus on single-sentence synthesis and lack adequate contextual modeling as well as fine-grained performance control capabilities for generating coherent multicast audiobooks. To address these limitations, we propose a context-aware and emotion controllable speech synthesis framework specifically engineered for multicast audiobooks with three key innovations: a context mechanism for contextual consistency, a disentanglement paradigm to decouple style control from speech prompts for semantic consistency, and self-distillation to boost emotional expressiveness and instruction controllability. Experimental results show superior performance across the generation of narration, dialogue, and the whole chapter, significantly outperforming existing baselines. Ablation studies are conducted to validate the effectiveness of our proposed methods. Demo samples can be found in this https URL.
#### Comparator Loss: An Ordinal Contrastive Loss to Derive a Severity Score for Speech-based Health Monitoring
 - **Authors:** Jacob J Webber, Oliver Watts, Lovisa Wihlborg, David Wheatley, Johnny Tam, Christine Weaver, Suvankar Pal, Siddharthan Chandran, Cassia Valentini-Botinhao
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2509.17661

 - **Pdf link:** https://arxiv.org/pdf/2509.17661

 - **Abstract**
 Monitoring the progression of neurodegenerative disease has important applications in the planning of treatment and the evaluation of future medications. Whereas much of the state-of-the-art in health monitoring from speech has been focused on classifying patients versus healthy controls, or predicting real-world health metrics, we propose here a novel measure of disease progression: the severity score. This score is derived from a model trained to minimize what we call the comparator loss. The comparator loss ensures scores follow an ordering relation, which can be based on diagnosis, clinically annotated scores, or simply the chronological order of the recordings. In addition to giving a more detailed picture than a simple discrete classification, the proposed comparator loss-based system has the potential to incorporate information from disparate health metrics, which is critical for making full use of small health-related datasets. We evaluated our proposed models based on their ability to affirmatively track the progression of patients with motor neuron disease (MND), the correlation of their output with clinical annotations such as ALSFRS-R, as well as their ability to distinguish between subjects with MND and healthy controls.
#### GAN-Based Multi-Microphone Spatial Target Speaker Extraction
 - **Authors:** Shrishti Saha Shetu, Emanuël A. P. Habets, Andreas Brendel
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2509.17741

 - **Pdf link:** https://arxiv.org/pdf/2509.17741

 - **Abstract**
 Spatial target speaker extraction isolates a desired speaker's voice in multi-speaker environments using spatial information, such as the direction of arrival (DoA). Although recent deep neural network (DNN)-based discriminative methods have shown significant performance improvements, the potential of generative approaches, such as generative adversarial networks (GANs), remains largely unexplored for this problem. In this work, we demonstrate that a GAN can effectively leverage both noisy mixtures and spatial information to extract and generate the target speaker's speech. By conditioning the GAN on intermediate features of a discriminative spatial filtering model in addition to DoA, we enable steerable target extraction with high spatial resolution of 5 degrees, outperforming state-of-the-art discriminative methods in perceptual quality-based objective metrics.
#### Benchmarking Humans and Machines on Complex Multilingual Speech Understanding Tasks
 - **Authors:** Sai Samrat Kankanala, Ram Chandra, Sriram Ganapathy
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2509.17965

 - **Pdf link:** https://arxiv.org/pdf/2509.17965

 - **Abstract**
 Auditory attention and selective phase-locking are central to human speech understanding in complex acoustic scenes and cocktail party settings, yet these capabilities in multilingual subjects remain poorly understood. While machine understanding of natural speech has advanced in recent years, questions persist about comprehension of overlapped and mixed-channel speech. We propose a systematic paradigm for studying humans and machines in speech question-answering tasks in multilingual settings with clean and mixed-channel speech. For human listeners, selective attention to a target speaker was significantly better in their native language (L1) than in their second language (L2). For machine listening, speech-based large language models (LLMs) match or exceed human performance in clean, single-speaker conditions but often struggle to selectively attend in two-speaker settings. These results reveal a key divergence: humans rely on attentional cues that are more streamlined in their native language, whereas LLMs default to parallel information extraction which exceed human skills.
#### Nord-Parl-TTS: Finnish and Swedish TTS Dataset from Parliament Speech
 - **Authors:** Zirui Li, Jens Edlund, Yicheng Gu, Nhan Phan, Lauri Juvela, Mikko Kurimo
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2509.17988

 - **Pdf link:** https://arxiv.org/pdf/2509.17988

 - **Abstract**
 Text-to-speech (TTS) development is limited by scarcity of high-quality, publicly available speech data for most languages outside a few high-resource languages. We present Nord-Parl-TTS, an open TTS dataset for Finnish and Swedish based on speech found in the wild. Using recordings of Nordic parliamentary proceedings, we extract 900 hours of Finnish and 5090 hours of Swedish speech suitable for TTS training. The dataset is built using an adapted version of the Emilia data processing pipeline and includes unified evaluation sets to support model development and benchmarking. By offering open, large-scale data for Finnish and Swedish, Nord-Parl-TTS narrows the resource gap in TTS between high- and lower-resourced languages.
#### Speech-to-See: End-to-End Speech-Driven Open-Set Object Detection
 - **Authors:** Wenhuan Lu, Xinyue Song, Wenjun Ke, Zhizhi Yu, Wenhao Yang, Jianguo Wei
 - **Subjects:** Subjects:
Sound (cs.SD); Multimedia (cs.MM); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2509.16670

 - **Pdf link:** https://arxiv.org/pdf/2509.16670

 - **Abstract**
 Audio grounding, or speech-driven open-set object detection, aims to localize and identify objects directly from speech, enabling generalization beyond predefined categories. This task is crucial for applications like human-robot interaction where textual input is impractical. However, progress in this domain faces a fundamental bottleneck from the scarcity of large-scale, paired audio-image data, and is further constrained by previous methods that rely on indirect, text-mediated pipelines. In this paper, we introduce Speech-to-See (Speech2See), an end-to-end approach built on a pre-training and fine-tuning paradigm. Specifically, in the pre-training stage, we design a Query-Guided Semantic Aggregation module that employs learnable queries to condense redundant speech embeddings into compact semantic representations. During fine-tuning, we incorporate a parameter-efficient Mixture-of-LoRA-Experts (MoLE) architecture to achieve deeper and more nuanced cross-modal adaptation. Extensive experiments show that Speech2See achieves robust and adaptable performance across multiple benchmarks, demonstrating its strong generalization ability and broad applicability.
#### Idiosyncratic Versus Normative Modeling of Atypical Speech Recognition: Dysarthric Case Studies
 - **Authors:** Vishnu Raja, Adithya V Ganesan, Anand Syamkumar, Ritwik Banerjee, H Andrew Schwartz
 - **Subjects:** Subjects:
Sound (cs.SD); Computation and Language (cs.CL); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2509.16718

 - **Pdf link:** https://arxiv.org/pdf/2509.16718

 - **Abstract**
 State-of-the-art automatic speech recognition (ASR) models like Whisper, perform poorly on atypical speech, such as that produced by individuals with dysarthria. Past works for atypical speech have mostly investigated fully personalized (or idiosyncratic) models, but modeling strategies that can both generalize and handle idiosyncracy could be more effective for capturing atypical speech. To investigate this, we compare four strategies: (a) $\textit{normative}$ models trained on typical speech (no personalization), (b) $\textit{idiosyncratic}$ models completely personalized to individuals, (c) $\textit{dysarthric-normative}$ models trained on other dysarthric speakers, and (d) $\textit{dysarthric-idiosyncratic}$ models which combine strategies by first modeling normative patterns before adapting to individual speech. In this case study, we find the dysarthric-idiosyncratic model performs better than idiosyncratic approach while requiring less than half as much personalized data (36.43 WER with 128 train size vs 36.99 with 256). Further, we found that tuning the speech encoder alone (as opposed to the LM decoder) yielded the best results reducing word error rate from 71% to 32% on average. Our findings highlight the value of leveraging both normative (cross-speaker) and idiosyncratic (speaker-specific) patterns to improve ASR for underrepresented speech populations.
#### The Sound of Syntax: Finetuning and Comprehensive Evaluation of Language Models for Speech Pathology
 - **Authors:** Fagun Patel, Duc Q. Nguyen, Sang T. Truong, Jody Vaynshtok, Sanmi Koyejo, Nick Haber
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Artificial Intelligence (cs.AI); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2509.16765

 - **Pdf link:** https://arxiv.org/pdf/2509.16765

 - **Abstract**
 According to the U.S. National Institutes of Health, more than 3.4 million children experience speech disorders that require clinical intervention. The number of speech-language pathologists (SLPs) is roughly 20 times fewer than the number of affected children, highlighting a significant gap in children's care and a pressing need for technological support that improves the productivity of SLPs. State-of-the-art multimodal language models (MLMs) show promise for supporting SLPs, but their use remains underexplored largely due to a limited understanding of their performance in high-stakes clinical settings. To address this gap, we collaborate with domain experts to develop a taxonomy of real-world use cases of MLMs in speech-language pathologies. Building on this taxonomy, we introduce the first comprehensive benchmark for evaluating MLM across five core use cases, each containing 1,000 manually annotated data points. This benchmark includes robustness and sensitivity tests under various settings, including background noise, speaker gender, and accent. Our evaluation of 15 state-of-the-art MLMs reveals that no single model consistently outperforms others across all tasks. Notably, we find systematic disparities, with models performing better on male speakers, and observe that chain-of-thought prompting can degrade performance on classification tasks with large label spaces and narrow decision boundaries. Furthermore, we study fine-tuning MLMs on domain-specific data, achieving improvements of over 30% compared to base models. These findings highlight both the potential and limitations of current MLMs for speech-language pathology applications, underscoring the need for further research and targeted development.
#### Drum-to-Vocal Percussion Sound Conversion and Its Evaluation Methodology
 - **Authors:** Rinka Nobukawa, Makito Kitamura, Tomohiko Nakamura, Shinnosuke Takamichi, Hiroshi Saruwatari
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2509.16862

 - **Pdf link:** https://arxiv.org/pdf/2509.16862

 - **Abstract**
 This paper defines the novel task of drum-to-vocal percussion (VP) sound conversion. VP imitates percussion instruments through human vocalization and is frequently employed in contemporary a cappella music. It exhibits acoustic properties distinct from speech and singing (e.g., aperiodicity, noisy transients, and the absence of linguistic structure), making conventional speech or singing synthesis methods unsuitable. We thus formulate VP synthesis as a timbre transfer problem from drum sounds, leveraging their rhythmic and timbral correspondence. To support this formulation, we define three requirements for successful conversion: rhythmic fidelity, timbral consistency, and naturalness as VP. We also propose corresponding subjective evaluation criteria. We implement two baseline conversion methods using a neural audio synthesizer, the real-time audio variational autoencoder (RAVE), with and without vector quantization (VQ). Subjective experiments show that both methods produce plausible VP outputs, with the VQ-based RAVE model yielding more consistent conversion.
#### Leveraging Multiple Speech Enhancers for Non-Intrusive Intelligibility Prediction for Hearing-Impaired Listeners
 - **Authors:** Boxuan Cao, Linkai Li, Hanlin Yu, Changgeng Mo, Haoshuai Zhou, Shan Xiang Wang
 - **Subjects:** Subjects:
Sound (cs.SD); Artificial Intelligence (cs.AI); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2509.16979

 - **Pdf link:** https://arxiv.org/pdf/2509.16979

 - **Abstract**
 Speech intelligibility evaluation for hearing-impaired (HI) listeners is essential for assessing hearing aid performance, traditionally relying on listening tests or intrusive methods like HASPI. However, these methods require clean reference signals, which are often unavailable in real-world conditions, creating a gap between lab-based and real-world assessments. To address this, we propose a non-intrusive intelligibility prediction framework that leverages speech enhancers to provide a parallel enhanced-signal pathway, enabling robust predictions without reference signals. We evaluate three state-of-the-art enhancers and demonstrate that prediction performance depends on the choice of enhancer, with ensembles of strong enhancers yielding the best results. To improve cross-dataset generalization, we introduce a 2-clips augmentation strategy that enhances listener-specific variability, boosting robustness on unseen datasets. Our approach consistently outperforms the non-intrusive baseline, CPC2 Champion across multiple datasets, highlighting the potential of enhancer-guided non-intrusive intelligibility prediction for real-world applications.
#### Advancing Speech Understanding in Speech-Aware Language Models with GRPO
 - **Authors:** Avishai Elmakies, Hagai Aronowitz, Nimrod Shabtay, Eli Schwartz, Ron Hoory, Avihu Dekel
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Artificial Intelligence (cs.AI); Machine Learning (cs.LG); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2509.16990

 - **Pdf link:** https://arxiv.org/pdf/2509.16990

 - **Abstract**
 In this paper, we introduce a Group Relative Policy Optimization (GRPO)-based method for training Speech-Aware Large Language Models (SALLMs) on open-format speech understanding tasks, such as Spoken Question Answering and Automatic Speech Translation. SALLMs have proven highly effective for speech understanding tasks. GRPO has recently gained traction for its efficiency in training LLMs, and prior work has explored its application to SALLMs, primarily in multiple-choice tasks. Building on this, we focus on open-format tasks that better reflect the generative abilities of the models. Our approach leverages GRPO with BLEU as the reward signal to optimize SALLMs, and we demonstrate empirically that it surpasses standard SFT across several key metrics. Finally, we explore the potential of incorporating off-policy samples within GRPO for these tasks, highlighting avenues for further improvement and further research.
#### MBCodec:Thorough disentangle for high-fidelity audio compression
 - **Authors:** Ruonan Zhang, Xiaoyang Hao, Yichen Han, Junjie Cao, Yue Liu, Kai Zhang
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2509.17006

 - **Pdf link:** https://arxiv.org/pdf/2509.17006

 - **Abstract**
 High-fidelity neural audio codecs in Text-to-speech (TTS) aim to compress speech signals into discrete representations for faithful reconstruction. However, prior approaches faced challenges in effectively disentangling acoustic and semantic information within tokens, leading to a lack of fine-grained details in synthesized speech. In this study, we propose MBCodec, a novel multi-codebook audio codec based on Residual Vector Quantization (RVQ) that learns a hierarchically structured representation. MBCodec leverages self-supervised semantic tokenization and audio subband features from the raw signals to construct a functionally-disentangled latent space. In order to encourage comprehensive learning across various layers of the codec embedding space, we introduce adaptive dropout depths to differentially train codebooks across layers, and employ a multi-channel pseudo-quadrature mirror filter (PQMF) during training. By thoroughly decoupling semantic and acoustic features, our method not only achieves near-lossless speech reconstruction but also enables a remarkable 170x compression of 24 kHz audio, resulting in a low bit rate of just 2.2 kbps. Experimental evaluations confirm its consistent and substantial outperformance of baselines across all evaluations.
#### Bridging the gap between training and inference in LM-based TTS models
 - **Authors:** Ruonan Zhang, Lingzhou Mu, Xixin Wu, Kai Zhang
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2509.17021

 - **Pdf link:** https://arxiv.org/pdf/2509.17021

 - **Abstract**
 Recent advancements in text-to-speech (TTS) have shown that language model (LM) based systems offer competitive performance compared to traditional approaches. However, in training, TTS models use ground-truth (GT) tokens as prefixes to predict the next token, while in inference these tokens are not available, a gap between training and inference that is often neglected. In this study, we propose a prompt-guided hybrid training scheme to mitigate exposure bias in popular LM-based TTS systems. Our core idea is to adopt a hybrid training paradigm that combines teacher forcing with free running, thereby introducing self-generated tokens into the training process. This makes the training mode more consistent with inference, reducing the training-inference gap. In addition, we incorporate an EOS prediction mechanism during training to detect incorrect sequence termination and adaptively control the free running process. Experimental results provide a comprehensive evaluation of the impact of exposure bias on LM-based TTS, and demonstrate that our method effectively narrows the training-inference gap, thereby improving the quality of synthesized long-form speech.
#### Sidon: Fast and Robust Open-Source Multilingual Speech Restoration for Large-scale Dataset Cleansing
 - **Authors:** Wataru Nakata, Yuki Saito, Yota Ueda, Hiroshi Saruwatari
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2509.17052

 - **Pdf link:** https://arxiv.org/pdf/2509.17052

 - **Abstract**
 Large-scale text-to-speech (TTS) systems are limited by the scarcity of clean, multilingual recordings. We introduce Sidon, a fast, open-source speech restoration model that converts noisy in-the-wild speech into studio-quality speech and scales to dozens of languages. Sidon consists of two models: w2v-BERT 2.0 finetuned feature predictor to cleanse features from noisy speech and vocoder trained to synthesize restored speech from the cleansed features. Sidon achieves restoration performance comparable to Miipher: Google's internal speech restoration model with the aim of dataset cleansing for speech synthesis. Sidon is also computationally efficient, running up to 3,390 times faster than real time on a single GPU. We further show that training a TTS model using a Sidon-cleansed automatic speech recognition corpus improves the quality of synthetic speech in a zero-shot setting. Code and model are released to facilitate reproducible dataset cleansing for the research community.
#### STAR: Speech-to-Audio Generation via Representation Learning
 - **Authors:** Zeyu Xie, Xuenan Xu, Yixuan Li, Mengyue Wu, Yuexian Zou
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2509.17164

 - **Pdf link:** https://arxiv.org/pdf/2509.17164

 - **Abstract**
 This work presents STAR, the first end-to-end speech-to-audio generation framework, designed to enhance efficiency and address error propagation inherent in cascaded systems. Unlike prior approaches relying on text or vision, STAR leverages speech as it constitutes a natural modality for interaction. As an initial step to validate the feasibility of the system, we demonstrate through representation learning experiments that spoken sound event semantics can be effectively extracted from raw speech, capturing both auditory events and scene cues. Leveraging the semantic representations, STAR incorporates a bridge network for representation mapping and a two-stage training strategy to achieve end-to-end synthesis. With a 76.9% reduction in speech processing latency, STAR demonstrates superior generation performance over the cascaded systems. Overall, STAR establishes speech as a direct interaction signal for audio generation, thereby bridging representation learning and multimodal synthesis. Generated samples are available at this https URL.
#### Leveraging Audio-Visual Data to Reduce the Multilingual Gap in Self-Supervised Speech Models
 - **Authors:** María Andrea Cruz Blandón, Zakaria Aldeneh, Jie Chi, Maureen de Seyssel
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2509.17523

 - **Pdf link:** https://arxiv.org/pdf/2509.17523

 - **Abstract**
 Self-supervised learning (SSL) has made significant advances in speech representation learning. Models like wav2vec 2.0 and HuBERT have achieved state-of-the-art results in tasks such as speech recognition, particularly in monolingual settings. However, multilingual SSL models tend to underperform their monolingual counterparts on each individual language, especially in multilingual scenarios with few languages such as the bilingual setting. In this work, we investigate a novel approach to reduce this performance gap by introducing limited visual grounding into bilingual speech SSL models. Our results show that visual grounding benefits both monolingual and bilingual models, with especially pronounced gains for the latter, reducing the multilingual performance gap on zero-shot phonetic discrimination from 31.5% for audio-only models to 8.04% with grounding.
#### Enhancing the NAO: Extending Capabilities of Legacy Robots for Long-Term Research
 - **Authors:** Austin Wilson, Sahar Kapasi, Zane Greene, Alexis E. Block
 - **Subjects:** Subjects:
Robotics (cs.RO); Human-Computer Interaction (cs.HC); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2509.17760

 - **Pdf link:** https://arxiv.org/pdf/2509.17760

 - **Abstract**
 Many research groups face challenges when legacy (unsupported) robotic platforms lose manufacturer support and cannot accommodate modern sensing, speech, and interaction capabilities. We present the Enhanced NAO, a revitalized version of Aldebaran's NAO robot that uses upgraded microphones, RGB-D and thermal cameras, and additional compute resources in a fully self-contained package. This system combines cloud and local models for perception and dialogue, while preserving the NAO's expressive body and behaviors. In a pilot validation study, the Enhanced NAO delivered significantly higher conversational quality and stronger user preference compared to the NAO AI Edition, without increasing response latency. Key upgrades, such as beamforming microphones and low-latency audio processing, reduced artifacts like self-hearing and improved multi-party separation. Expanded visual and thermal sensing established a foundation for future interaction capabilities. Beyond the NAO, our framework provides a platform-agnostic strategy for extending the lifespan and research utility of legacy robots, ensuring they remain valuable tools for human-robot interaction.
#### Qwen3-Omni Technical Report
 - **Authors:** Jin Xu, Zhifang Guo, Hangrui Hu, Yunfei Chu, Xiong Wang, Jinzheng He, Yuxuan Wang, Xian Shi, Ting He, Xinfa Zhu, Yuanjun Lv, Yongqi Wang, Dake Guo, He Wang, Linhan Ma, Pei Zhang, Xinyu Zhang, Hongkun Hao, Zishan Guo, Baosong Yang, Bin Zhang, Ziyang Ma, Xipin Wei, Shuai Bai, Keqin Chen, Xuejing Liu, Peng Wang, Mingkun Yang, Dayiheng Liu, Xingzhang Ren, Bo Zheng, Rui Men, Fan Zhou, Bowen Yu, Jianxin Yang, Le Yu, Jingren Zhou, Junyang Lin
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Artificial Intelligence (cs.AI); Computer Vision and Pattern Recognition (cs.CV); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2509.17765

 - **Pdf link:** https://arxiv.org/pdf/2509.17765

 - **Abstract**
 We present Qwen3-Omni, a single multimodal model that, for the first time, maintains state-of-the-art performance across text, image, audio, and video without any degradation relative to single-modal counterparts. Qwen3-Omni matches the performance of same-sized single-modal models within the Qwen series and excels particularly on audio tasks. Across 36 audio and audio-visual benchmarks, Qwen3-Omni achieves open-source SOTA on 32 benchmarks and overall SOTA on 22, outperforming strong closed-source models such as Gemini-2.5-Pro, Seed-ASR, and GPT-4o-Transcribe. Qwen3-Omni adopts a Thinker-Talker MoE architecture that unifies perception and generation across text, images, audio, and video, yielding fluent text and natural real-time speech. It supports text interaction in 119 languages, speech understanding in 19 languages, and speech generation in 10 languages. To reduce first-packet latency in streaming synthesis, Talker autoregressively predicts discrete speech codecs using a multi-codebook scheme. Leveraging the representational capacity of these codebooks, we replace computationally intensive block-wise diffusion with a lightweight causal ConvNet, enabling streaming from the first codec frame. In cold-start settings, Qwen3-Omni achieves a theoretical end-to-end first-packet latency of 234 ms. To further strengthen multimodal reasoning, we introduce a Thinking model that explicitly reasons over inputs from any modality. Since the research community currently lacks a general-purpose audio captioning model, we fine-tuned Qwen3-Omni-30B-A3B to obtain Qwen3-Omni-30B-A3B-Captioner, which produces detailed, low-hallucination captions for arbitrary audio inputs. Qwen3-Omni-30B-A3B, Qwen3-Omni-30B-A3B-Thinking, and Qwen3-Omni-30B-A3B-Captioner are publicly released under the Apache 2.0 license.


by Zyzzyva0381 (Windy). 


2025-09-23
