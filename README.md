# Showing new listings for Friday, 16 October 2026
Auto update papers at about 2:30am UTC (10:30am Beijing time) every weekday.


阅读 `Usage.md`了解如何使用此repo实现个性化的Arxiv论文推送

See `Usage.md` for instructions on how to personalize the repo. 


Keyword list: ['text-to-speech', 'text to speech', 'tts', 'LLM-based', 'speech', 'voice']


Excluded: []


### Today: 9papers 
#### SNOAR: Self-training with Noise filtering, Observation adding, and Remixing for Speech Denoising Without Clean References
 - **Authors:** Samuele Cornell, Matthew Maciejewski
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Machine Learning (cs.LG); Sound (cs.SD); Signal Processing (eess.SP)
 - **Arxiv link:** https://arxiv.org/abs/2610.20863

 - **Pdf link:** https://arxiv.org/pdf/2610.20863

 - **Abstract**
 Unsupervised speech denoising methods such as MixIT and noisy-target training can be trained directly on noisy recordings, avoiding the sim-to-real gap of synthetic training data, but remain well below supervised performance. Teacher--student self-training, as in RemixIT, narrows this gap by remixing a teacher's estimates into new mixtures that supervise a student; however, since the student is trained only toward the teacher's estimates, teacher errors, in particular over-suppressed speech, are never recovered and propagate as the teacher is updated from the student. We propose SNOAR, a self-training framework that corrects the teacher's estimates before use: the noise leaked into its speech estimate is removed by differential noise filtering (DNF), and part of the noisy observation is added back in to restore suppressed speech and attenuate artifacts. We show that the two corrections depend on each other, as without DNF the loop collapses to the identity mapping. Compared to RemixIT, SNOAR is more stable along training and outperforms it in signal-based, perceptual, and word error rate metrics, reaching 97\% of the SI-SDR improvement of a fully-supervised topline on WHAM!; on real CHiME-3 recordings, it improves every metric of three different teachers, while RemixIT degrades their perceptual quality.
#### Probing Neural Audio Codecs with Speaker Traits and Speech Characteristics
 - **Authors:** Jihwan Lee, Tiantian Feng, Minu Kim, Junhyeok Lee, Sean Foley, Thanathai Lertpetchpun, Najim Dehak, Dani Byrd, Shrikanth Narayanan
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2610.20955

 - **Pdf link:** https://arxiv.org/pdf/2610.20955

 - **Abstract**
 Neural audio codecs are increasingly used for low-bitrate speech coding and serve as tokenizers for speech language models, yet it remains unclear how their codebook hierarchies encode speaker traits and speech characteristics. We systematically study 17 multi-codebook neural audio codecs across seven traits: sex, age, accent, voice quality, emotion, speech flow, and non-verbal sounds. For each codec, we reconstruct speech using partial codebooks and use machine listeners to compare trait predictions on each reconstruction. We measure prediction agreement with reweighted Cohen's \k{appa}, and define trait depth as a proxy for the codebook depth by which each trait is encoded and recoverable. Overall, sex has the shallowest trait depth, followed by speech flow, whereas voice quality and age group require deeper stacks; emotion, accent, and non-verbal sounds fall in between. Within each trait, speech-flow classes are at relatively similar trait depths, whereas emotion, accent, and non-verbal sounds differ more. Additionally, we explore how certain codec design choices relate to trait depths. These findings show that codebook hierarchies are trait-dependent and may not exhibit uniform coarse-to-fine structures, motivating trait-specific codec evaluation and design for future applications.
#### VoxTurn: Answering Several Turn-Taking Questions in One Pass with One Model
 - **Authors:** Yang Xiao
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2610.21150

 - **Pdf link:** https://arxiv.org/pdf/2610.21150

 - **Abstract**
 A cascaded voice agent must decide, throughout each silence of its user, whether the user has finished their turn: answering too early cuts the user off, answering too late makes the agent slow. It often needs related decisions too: will the user soon go on, is speech during its own turn an interruption or a backchannel, is a sound speech at all. With open detectors, each extra decision costs another model or another call, and on the 14-language eot-bench they cut users off in more than a quarter of mid-turn pauses. Because they are trained on segmented utterances, this mismatch is the largest single source of error. We present VoxTurn, a 0.8B speech LLM that casts these decisions as multiple-choice questions, scores the hidden state of each option, and answers any set of them in one forward pass with a block-isolating attention mask, exactly as if asked one at a time. Trained on real conversational pauses with labels derived from their timing, and re-queried while a silence lasts, VoxTurn reaches 11.5% false cut-offs in English at a 300 ms latency budget, comparable to two commercial detectors, and a 14-language mean of 19.9%, the lowest among open models, at 8.3 ms per decision. In Pipecat and LiveKit it cuts off 17% and 9% of turns, and it ranks fifth of 24 in TurnBench end-of-turn detection without any TurnBench training data. The model, code and framework plugins are released.
#### Full-Duplex Interaction with Speech, Expressions and Gestures
 - **Authors:** Siyin Wang, Wenyi Yu, Terumi Chiba, Xianzhao Chen, Xiaohai Tian, Jun Zhang, Lu Lu, Chao Zhang
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Artificial Intelligence (cs.AI); Computer Vision and Pattern Recognition (cs.CV); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2610.21452

 - **Pdf link:** https://arxiv.org/pdf/2610.21452

 - **Abstract**
 Human face-to-face conversation requires coordinating speech, gestures and expressions across both speaking and listening. We present ELLSA 2, an end-to-end full-duplex dialogue model that jointly optimizes speech interaction and streaming gesture-expression generation. An autoregressive, chunk-wise synthesizer combines pretrained acoustic features with distinct listening and speaking hidden states from the dialogue backbone and speech synthesizer, enabling continuous non-verbal generation even when the model is verbally silent. To translate dialogue understanding into vivid semantic gestures, we introduce contextual semantic calls and template guidance, separating whether to invoke an available semantic gesture from how to realize it. We also establish a curated data and evaluation pipeline for dialog-aware co-speech gesture generation and investigate reference-based multimodal large language models (MLLMs) as automatic judges. Experiments show that ELLSA 2 remains highly competitive with open-source full-duplex speech interaction models and outperforms cascaded baselines across dialog-aware gesture and expression metrics. Semantic template guidance further significantly improves semantic gesture quality. Strong system-level correlations with human judgments support these reference-based scores as useful preliminary metrics for full-duplex and semantic gesture assessment. Together, these results advance dialogue modeling from spoken responses toward coordinated multimodal interaction.
#### Rethinking Batch Composition: Promoting Feature Diversity for Speech Enhancement
 - **Authors:** Mingchi Hou, Ina Kodrasi
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2610.21803

 - **Pdf link:** https://arxiv.org/pdf/2610.21803

 - **Abstract**
 Speech enhancement systems are commonly trained using randomly sampled mini-batches, which may contain limited diversity in intrinsic acoustic characteristics of the clean speech signals. Motivated by previous findings linking clean speech characteristics to enhancement performance, we propose a feature-stratified mini-batch sampling strategy that divides training examples into quantiles according to a clean-speech characteristic and enforces equal representation within each mini-batch. Experiments across two large-scale English and Spanish speech datasets and a smaller dataset containing neurotypical and pathological speech demonstrate improvements over standard random mini-batch sampling across multiple objective enhancement metrics. Ablation studies demonstrate that the gains are robust across training runs, while their magnitude depends on the acoustic characteristic used for stratification. The proposed strategy requires no modification to the enhancement model or training objective, providing a simple and complementary approach to speech enhancement training.
#### Bridging the Gap: Converting Read Text to Conversational Dialogue
 - **Authors:** Parshav Singla, Agnik Banerjee, Aaditya Arora, Shruti Aggarwal, Anil Kumar Verma, Vikram C M, Raj Prakash Gohil, Gopal Kumar Agarwal
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Artificial Intelligence (cs.AI); Machine Learning (cs.LG); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2605.18001

 - **Pdf link:** https://arxiv.org/pdf/2605.18001

 - **Abstract**
 In recent advancements within speech processing, converting read speech to conversational speech has gained significant attention. The primary challenge in this domain is maintaining naturalness and intelligibility while minimizing computational overhead for real-time applications. Traditional read speech often lacks the nuanced prosodic variation essential for natural conversational interactions, posing challenges for applications in virtual assistants, customer service, and language learning tools. This paper introduces a novel approach, Prosodic Adjustment with Conversational Context (PACC), aimed at converting read speech into natural conversational speech used in various modern applications. PACC utilizes advanced deep neural networks to analyze and modify prosodic features such as intonation, stress, and rhythm. Unlike conventional methods, our approach uses High-Fidelity Generative Adversarial Networks (HiFi-GAN) for speech synthesis. Our experimental results demonstrate significant improvements in speech conversion, enhancing naturalness and achieving better model accuracy with additional training on speech datasets. This research establishes new benchmarks in speech conversion tasks and Mean Opinion Score (MOS) evaluation for testing model accuracy, and we show that our approach can be successfully extended to other speech conversion applications.
#### Zero-Shot Spoken Language Understanding with SLAM-ASR: What Do Monolingual ASR-Trained Linear Projectors Enable?
 - **Authors:** Edwin Arkel Rios, Sneh Sourabh, Jocelyn Zaruma, Girish Ewoorkar, Julian Mack, Bo-Cheng Lai
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2610.20844

 - **Pdf link:** https://arxiv.org/pdf/2610.20844

 - **Abstract**
 The SLAM-ASR framework connects the features of a frozen speech encoder as input to a frozen LLM decoder to enable speech transcription. In this work we ask whether this training-on speech transcription-provides the LLM with the capability to also perform spoken language understanding (SLU) tasks, by training monolingual projectors on 11 languages. When prompted to answer a yes/no question about the input speech, the mean instruction following rate (IFR) is 49.8%. When not following the instruction, these models default back to transcription. The transcription quality-measured by the character error rate (CER)-is not sufficient to predict IFR. Remarkably, when the model is instructed to answer the topic question as a single digit option, the mean IFR rises to 81.3%. Our models score a mean 23.2% topic balanced accuracy (BAcc) against 14.3% chance. However, the same decoder scores a mean 71.5% BAcc when using reference transcripts as text input. This indicates a failure of the projector to transfer information for the SLU tasks.
#### High-Order Sinusoidal Modeling for Precise Parameter Refinement
 - **Authors:** Shaowen Chen, Tomoki Toda
 - **Subjects:** Subjects:
Signal Processing (eess.SP); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2610.21643

 - **Pdf link:** https://arxiv.org/pdf/2610.21643

 - **Abstract**
 Although the amplitude estimation of sinusoids is a linear and tractable problem when their frequencies are known, the accuracy of signal modeling is fundamentally limited by the inevitable frequency mismatches produced by existing methods. Quasi-harmonic modeling (QHM) and its extensions (aQHM and eaQHM) offer the ability to refine frequency estimates while extracting amplitudes; however, their extraction and correction capabilities remain limited. In this paper, we first revisit QHM methods from the perspective of frequency mismatches, and reveal that their performances are essentially constrained by the order of the model. Specifically, QHM methods are linear models and thus incapable of capturing higher-order frequency variations, leading to suboptimal modeling accuracy. Second, to address this limitation, we propose a higher-order modeling framework that enables more precise approximation of signal components. Eventually, we qualitatively characterize the optimization problem and employ the Levenberg--Marquardt (LM) algorithm for parameter estimation. As a result, the proposed method allows for significantly improved estimation of amplitudes, frequencies, and phases of sinusoidal components, enabling more accurate signal modeling. Experiments on both simulation study and speech signals demonstrate the superiority of the high-order model in frequency refinement and parameter extraction and its robustness under additive noise conditions.
#### Frame-Wise Streaming for Efficient Learned Speech Compression
 - **Authors:** Yuhan Liu, Jun Xu, Fengxi Zhang, Zhengxue Cheng, Li Song
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2610.21998

 - **Pdf link:** https://arxiv.org/pdf/2610.21998

 - **Abstract**
 Speech compression is increasingly important in scenarios such as real-time and satellite communication. Neural codecs have shown superior Rate-Distortion (RD) performance over traditional signal-processing codecs. Models with long-range dependency encoders/decoders and entropy models outperform convolution-based structures, but they still lack streaming support since they rely on sequence-level correlations. In this paper, we propose StreamLSC, a frame-level streaming learned speech compression model built on three fundamental components of streaming networks, i.e. Boundary-Cached Causal Convolution, Intra-Frame Layer Normalization, and RNN-mode RWKV. Our model incorporates a streamable entropy quantizer module, ensuring efficient context model while preserving causal constraints. Experiments show that the streaming model achieves performance comparable to its offline counterpart, attaining state-of-the-art rate-distortion results, significant gains in subjective evaluations, -51.37% (ViSQOL) and -53.99% (PESQ) BD-RATE savings on average, and improvements of 0.2154 and 0.5156 in BD-ViSQOL and BD-PESQ.


by Zyzzyva0381 (Windy). 


2026-10-16
