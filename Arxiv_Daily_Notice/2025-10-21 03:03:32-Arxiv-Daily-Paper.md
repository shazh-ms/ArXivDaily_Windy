# Showing new listings for Tuesday, 21 October 2025
Auto update papers at about 2:30am UTC (10:30am Beijing time) every weekday.


阅读 `Usage.md`了解如何使用此repo实现个性化的Arxiv论文推送

See `Usage.md` for instructions on how to personalize the repo. 


Keyword list: ['text-to-speech', 'text to speech', 'tts', 'LLM-based', 'speech', 'voice']


Excluded: []


### Today: 13papers 
#### AsyncVoice Agent: Real-Time Explanation for LLM Planning and Reasoning
 - **Authors:** Yueqian Lin, Zhengmian Hu, Jayakumar Subramanian, Qinsi Wang, Nikos Vlassis, Hai "Helen" Li, Yiran Chen
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Artificial Intelligence (cs.AI); Multimedia (cs.MM)
 - **Arxiv link:** https://arxiv.org/abs/2510.16156

 - **Pdf link:** https://arxiv.org/pdf/2510.16156

 - **Abstract**
 Effective human-AI collaboration on complex reasoning tasks requires that users understand and interact with the model's process, not just receive an output. However, the monolithic text from methods like Chain-of-Thought (CoT) prevents this, as current interfaces lack real-time verbalization and robust user barge-in. We present AsyncVoice Agent, a system whose asynchronous architecture decouples a streaming LLM backend from a conversational voice frontend. This design allows narration and inference to run in parallel, empowering users to interrupt, query, and steer the model's reasoning process at any time. Objective benchmarks show this approach reduces interaction latency by more than 600x compared to monolithic baselines while ensuring high fidelity and competitive task accuracy. By enabling a two-way dialogue with a model's thought process, AsyncVoice Agent offers a new paradigm for building more effective, steerable, and trustworthy human-AI systems for high-stakes tasks.
#### Audio-Visual Speech Enhancement for Spatial Audio - Spatial-VisualVoice and the MAVE Database
 - **Authors:** Danielle Yaffe (1), Ferdinand Campe (2), Prachi Sharma (2), Dorothea Kolossa (2), Boaz Rafaely (1) ((1) School of Electrical and Computer Engineering, Ben-Gurion University of the Negev (2) Technische Universität Berlin)
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2510.16437

 - **Pdf link:** https://arxiv.org/pdf/2510.16437

 - **Abstract**
 Audio-visual speech enhancement (AVSE) has been found to be particularly useful at low signal-to-noise (SNR) ratios due to the immunity of the visual features to acoustic noise. However, a significant gap exists in AVSE methods tailored to enhance spatial audio under low-SNR conditions. The latter is of growing interest with augmented reality applications. To address this gap, we present a multi-channel AVSE framework based on VisualVoice that leverages spatial cues from microphone arrays and visual information for enhancing the target speaker in noisy environments. We also introduce MAVe, a novel database containing multi-channel audio-visual signals in controlled, reproducible room conditions across a wide range of SNR levels. Experiments demonstrate that the proposed method consistently achieves significant gains in SI-SDR, STOI, and PESQ, particularly in low SNRs. Binaural signal analysis further confirms the preservation of spatial cues and intelligibility.
#### SAC: Neural Speech Codec with Semantic-Acoustic Dual-Stream Quantization
 - **Authors:** Wenxi Chen, Xinsheng Wang, Ruiqi Yan, Yushen Chen, Zhikang Niu, Ziyang Ma, Xiquan Li, Yuzhe Liang, Hanlin Wen, Shunshun Yin, Ming Tao, Xie Chen
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2510.16841

 - **Pdf link:** https://arxiv.org/pdf/2510.16841

 - **Abstract**
 Speech codecs that convert continuous speech signals into discrete tokens have become essential for speech language models (SLMs). However, existing codecs struggle to balance high-quality reconstruction with semantically rich representations, limiting their effectiveness in both generative and understanding tasks. In this work, we propose SAC, a neural speech codec with semantic-acoustic dual-stream quantization. By disentangling semantic and acoustic modeling into two dedicated streams, SAC enables each to be optimized for its respective role. Comprehensive evaluations show that SAC achieves strong reconstruction performance across diverse bitrates under both clean and noisy conditions, with particularly high scores on UTMOS and WER, demonstrating superior perceptual quality and intelligibility. Moreover, SAC substantially outperforms state-of-the-art codecs in semantic representation, achieving a level comparable to that of self-supervised learning (SSL) continuous embeddings. Finally, our analysis of speech disentanglement highlights the effectiveness of the dual-stream design, offering new potential for controllable speech applications.
#### Adaptive Deterministic Flow Matching for Target Speaker Extraction
 - **Authors:** Tsun-An Hsieh, Minje Kim
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2510.16995

 - **Pdf link:** https://arxiv.org/pdf/2510.16995

 - **Abstract**
 Generative target speaker extraction (TSE) methods often produce more natural outputs than predictive models. Recent work based on diffusion or flow matching (FM) typically relies on a small, fixed number of reverse steps with a fixed step size. We introduce Adaptive Discriminative Flow Matching TSE (AD-FlowTSE), which extracts the target speech using an adaptive step size. We formulate TSE within the FM paradigm but, unlike prior FM-based speech enhancement and TSE approaches that transport between the mixture (or a normal prior) and the clean-speech distribution, we define the flow between the background and the source, governed by the mixing ratio (MR) of the source and background that creates the mixture. This design enables MR-aware initialization, where the model starts at an adaptive point along the background-source trajectory rather than applying the same reverse schedule across all noise levels. Experiments show that AD-FlowTSE achieves strong TSE with as few as a single step, and that incorporating auxiliary MR estimation further improves target speech accuracy. Together, these results highlight that aligning the transport path with the mixture composition and adapting the step size to noise conditions yields efficient and accurate TSE.
#### Towards Real-Time Generative Speech Restoration with Flow-Matching
 - **Authors:** Tsun-An Hsieh, Sebastian Braun
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2510.16997

 - **Pdf link:** https://arxiv.org/pdf/2510.16997

 - **Abstract**
 Generative models have shown robust performance on speech enhancement and restoration tasks, but most prior approaches operate offline with high latency, making them unsuitable for streaming applications. In this work, we investigate the feasibility of a low-latency, real-time generative speech restoration system based on flow-matching (FM). Our method tackles diverse real-world tasks, including denoising, dereverberation, and generative restoration. The proposed causal architecture without time-downsampling achieves introduces an total latency of only 20 ms, suitable for real-time communication. In addition, we explore a broad set of architectural variations and sampling strategies to ensure effective training and efficient inference. Notably, our flow-matching model maintains high enhancement quality with only 5 number of function evaluations (NFEs) during sampling, achieving similar performance as when using ~20 NFEs under the same conditions. Experimental results indicate that causal FM-based models favor few-step reverse sampling, and smaller backbones degrade with longer reverse trajectories. We further show a side-by-side comparison of FM to typical adversarial-loss-based training for the same model architecture.
#### AnyRIR: Robust Non-intrusive Room Impulse Response Estimation in the Wild
 - **Authors:** Kyung Yun Lee, Nils Meyer-Kahlen, Karolina Prawda, Vesa Välimäki, Sebastian J. Schlecht
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2510.17788

 - **Pdf link:** https://arxiv.org/pdf/2510.17788

 - **Abstract**
 We address the problem of estimating room impulse responses (RIRs) in noisy, uncontrolled environments where non-stationary sounds such as speech or footsteps corrupt conventional deconvolution. We propose AnyRIR, a non-intrusive method that uses music as the excitation signal instead of a dedicated test signal, and formulate RIR estimation as an L1-norm regression in the time-frequency domain. Solved efficiently with Iterative Reweighted Least Squares (IRLS) and Least-Squares Minimal Residual (LSMR) methods, this approach exploits the sparsity of non-stationary noise to suppress its influence. Experiments on simulated and measured data show that AnyRIR outperforms L2-based and frequency-domain deconvolution, under in-the-wild noisy scenarios and codec mismatch, enabling robust RIR estimation for AR/VR and related applications.
#### MuseTok: Symbolic Music Tokenization for Generation and Semantic Understanding
 - **Authors:** Jingyue Huang, Zachary Novack, Phillip Long, Yupeng Hou, Ke Chen, Taylor Berg-Kirkpatrick, Julian McAuley
 - **Subjects:** Subjects:
Sound (cs.SD); Artificial Intelligence (cs.AI); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2510.16273

 - **Pdf link:** https://arxiv.org/pdf/2510.16273

 - **Abstract**
 Discrete representation learning has shown promising results across various domains, including generation and understanding in image, speech and language. Inspired by these advances, we propose MuseTok, a tokenization method for symbolic music, and investigate its effectiveness in both music generation and understanding tasks. MuseTok employs the residual vector quantized-variational autoencoder (RQ-VAE) on bar-wise music segments within a Transformer-based encoder-decoder framework, producing music codes that achieve high-fidelity music reconstruction and accurate understanding of music theory. For comprehensive evaluation, we apply MuseTok to music generation and semantic understanding tasks, including melody extraction, chord recognition, and emotion recognition. Models incorporating MuseTok outperform previous representation learning baselines in semantic understanding while maintaining comparable performance in content generation. Furthermore, qualitative analyses on MuseTok codes, using ground-truth categories and synthetic datasets, reveal that MuseTok effectively captures underlying musical concepts from large music collections.
#### Probing the Hidden Talent of ASR Foundation Models for L2 English Oral Assessment
 - **Authors:** Fu-An Chao, Bi-Cheng Yan, Berlin Chen
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Artificial Intelligence (cs.AI); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2510.16387

 - **Pdf link:** https://arxiv.org/pdf/2510.16387

 - **Abstract**
 In this paper, we explore the untapped potential of Whisper, a well-established automatic speech recognition (ASR) foundation model, in the context of L2 spoken language assessment (SLA). Unlike prior studies that extrinsically analyze transcriptions produced by Whisper, our approach goes a step further to probe its latent capabilities by extracting acoustic and linguistic features from hidden representations. With only a lightweight classifier being trained on top of Whisper's intermediate and final outputs, our method achieves strong performance on the GEPT picture-description dataset, outperforming existing cutting-edge baselines, including a multimodal approach. Furthermore, by incorporating image and text-prompt information as auxiliary relevance cues, we demonstrate additional performance gains. Finally, we conduct an in-depth analysis of Whisper's embeddings, which reveals that, even without task-specific fine-tuning, the model intrinsically encodes both ordinal proficiency patterns and semantic aspects of speech, highlighting its potential as a powerful foundation for SLA and other spoken language understanding tasks.
#### Interpreting the Dimensions of Speaker Embedding Space
 - **Authors:** Mark Huckvale
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2510.16489

 - **Pdf link:** https://arxiv.org/pdf/2510.16489

 - **Abstract**
 Speaker embeddings are widely used in speaker verification systems and other applications where it is useful to characterise the voice of a speaker with a fixed-length vector. These embeddings tend to be treated as "black box" encodings, and how they relate to conventional acoustic and phonetic dimensions of voices has not been widely studied. In this paper we investigate how state-of-the-art speaker embedding systems represent the acoustic characteristics of speakers as described by conventional acoustic descriptors, age, and gender. Using a large corpus of 10,000 speakers and three embedding systems we show that a small set of 9 acoustic parameters chosen to be "interpretable" predict embeddings about the same as 7 principal components, corresponding to over 50% of variance in the data. We show that some principal dimensions operate differently for male and female speakers, suggesting there is implicit gender recognition within the embedding systems. However we show that speaker age is not well captured by embeddings, suggesting opportunities exist for improvements in their calculation.
#### Hallucination Benchmark for Speech Foundation Models
 - **Authors:** Alkis Koudounas, Moreno La Quatra, Manuel Giollo, Sabato Marco Siniscalchi, Elena Baralis
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2510.16567

 - **Pdf link:** https://arxiv.org/pdf/2510.16567

 - **Abstract**
 Hallucinations in automatic speech recognition (ASR) systems refer to fluent and coherent transcriptions produced by neural ASR models that are completely unrelated to the underlying acoustic input (i.e., the speech signal). While similar to conventional decoding errors in potentially compromising the usability of transcriptions for downstream applications, hallucinations can be more detrimental due to their preservation of syntactically and semantically plausible structure. This apparent coherence can mislead subsequent processing stages and introduce serious risks, particularly in critical domains such as healthcare and law. Conventional evaluation metrics are primarily centered on error-based metrics and fail to distinguish between phonetic inaccuracies and hallucinations. Consequently, there is a critical need for new evaluation frameworks that can effectively identify and assess models with a heightened propensity for generating hallucinated content. To this end, we introduce SHALLOW, the first benchmark framework that systematically categorizes and quantifies hallucination phenomena in ASR along four complementary axes: lexical, phonetic, morphological, and semantic. We define targeted metrics within each category to produce interpretable profiles of model behavior. Through evaluation across various architectures and speech domains, we have found that SHALLOW metrics correlate strongly with word error rate (WER) when recognition quality is high (i.e., low WER). Still, this correlation weakens substantially as WER increases. SHALLOW, therefore, captures fine-grained error patterns that WER fails to distinguish under degraded and challenging conditions. Our framework supports specific diagnosis of model weaknesses and provides feedback for model improvement beyond what aggregate error rates can offer.
#### End-to-end Listen, Look, Speak and Act
 - **Authors:** Siyin Wang, Wenyi Yu, Xianzhao Chen, Xiaohai Tian, Jun Zhang, Lu Lu, Chao Zhang
 - **Subjects:** Subjects:
Artificial Intelligence (cs.AI); Computation and Language (cs.CL); Computer Vision and Pattern Recognition (cs.CV); Robotics (cs.RO); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2510.16756

 - **Pdf link:** https://arxiv.org/pdf/2510.16756

 - **Abstract**
 Human interaction is inherently multimodal and full-duplex: we listen while watching, speak while acting, and fluidly adapt to turn-taking and interruptions. Realizing these capabilities is essential for building models simulating humans. We present ELLSA (End-to-end Listen, Look, Speak and Act), which, to our knowledge, is the first full-duplex, end-to-end model that simultaneously perceives and generates across vision, text, speech, and action within a single architecture, enabling interaction patterns previously out of reach, yielding more natural, human-like behaviors. At its core is a novel SA-MoE architecture (Self-Attention Mixture-of-Experts) that routes each modality to specialized experts and fuses them through a unified attention backbone. This provides a generalizable solution for joint multimodal perception and concurrent generation, leveraging strong pre-trained components while enabling efficient modality integration and mitigating modality interference. On speech-interaction and robot-manipulation benchmarks, ELLSA matches modality-specific baselines, while uniquely supporting advanced multimodal and full-duplex behaviors such as dialogue and action turn-taking, defective instruction rejection, speaking-while-acting, context-grounded visual question answering, and action barge-ins. We contend that ELLSA represents a step toward more natural and general interactive intelligence, contributing to the broader pursuit of artificial general intelligence. All data, code and model checkpoints will be released upon acceptance.
#### Schrödinger Bridge Mamba for One-Step Speech Enhancement
 - **Authors:** Jing Yang, Sirui Wang, Chao Wu, Fan Fan
 - **Subjects:** Subjects:
Sound (cs.SD); Artificial Intelligence (cs.AI); Machine Learning (cs.LG); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2510.16834

 - **Pdf link:** https://arxiv.org/pdf/2510.16834

 - **Abstract**
 We propose Schrödinger Bridge Mamba (SBM), a new concept of training-inference framework motivated by the inherent compatibility between Schrödinger Bridge (SB) training paradigm and selective state-space model Mamba. We exemplify the concept of SBM with an implementation for generative speech enhancement. Experiments on a joint denoising and dereverberation task using four benchmark datasets demonstrate that SBM, with only 1-step inference, outperforms strong baselines with 1-step or iterative inference and achieves the best real-time factor (RTF). Beyond speech enhancement, we discuss the integration of SB paradigm and selective state-space model architecture based on their underlying alignment, which indicates a promising direction for exploring new deep generative models potentially applicable to a broad range of generative tasks. Demo page: this https URL
#### Investigating Safety Vulnerabilities of Large Audio-Language Models Under Speaker Emotional Variations
 - **Authors:** Bo-Han Feng, Chien-Feng Liu, Yu-Hsuan Li Liang, Chih-Kai Yang, Szu-Wei Fu, Zhehuai Chen, Ke-Han Lu, Sung-Feng Huang, Chao-Han Huck Yang, Yu-Chiang Frank Wang, Yun-Nung Chen, Hung-yi Lee
 - **Subjects:** Subjects:
Sound (cs.SD); Artificial Intelligence (cs.AI); Computation and Language (cs.CL); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2510.16893

 - **Pdf link:** https://arxiv.org/pdf/2510.16893

 - **Abstract**
 Large audio-language models (LALMs) extend text-based LLMs with auditory understanding, offering new opportunities for multimodal applications. While their perception, reasoning, and task performance have been widely studied, their safety alignment under paralinguistic variation remains underexplored. This work systematically investigates the role of speaker emotion. We construct a dataset of malicious speech instructions expressed across multiple emotions and intensities, and evaluate several state-of-the-art LALMs. Our results reveal substantial safety inconsistencies: different emotions elicit varying levels of unsafe responses, and the effect of intensity is non-monotonic, with medium expressions often posing the greatest risk. These findings highlight an overlooked vulnerability in LALMs and call for alignment strategies explicitly designed to ensure robustness under emotional variation, a prerequisite for trustworthy deployment in real-world settings.


by Zyzzyva0381 (Windy). 


2025-10-21
