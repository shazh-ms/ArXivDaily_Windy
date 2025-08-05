# Showing new listings for Tuesday, 5 August 2025
Auto update papers at about 2:30am UTC (10:30am Beijing time) every weekday.


阅读 `Usage.md`了解如何使用此repo实现个性化的Arxiv论文推送

See `Usage.md` for instructions on how to personalize the repo. 


Keyword list: ['text-to-speech', 'text to speech', 'tts', 'LLM-based', 'speech', 'voice']


Excluded: []


### Today: 14papers 
#### Fusion of Modulation Spectrogram and SSL with Multi-head Attention for Fake Speech Detection
 - **Authors:** Rishith Sadashiv T N, Abhishek Bedge, Saisha Suresh Bore, Jagabandhu Mishra, Mrinmoy Bhattacharjee, S R Mahadeva Prasanna
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2508.01034

 - **Pdf link:** https://arxiv.org/pdf/2508.01034

 - **Abstract**
 Fake speech detection systems have become a necessity to combat against speech deepfakes. Current systems exhibit poor generalizability on out-of-domain speech samples due to lack to diverse training data. In this paper, we attempt to address domain generalization issue by proposing a novel speech representation using self-supervised (SSL) speech embeddings and the Modulation Spectrogram (MS) feature. A fusion strategy is used to combine both speech representations to introduce a new front-end for the classification task. The proposed SSL+MS fusion representation is passed to the AASIST back-end network. Experiments are conducted on monolingual and multilingual fake speech datasets to evaluate the efficacy of the proposed model architecture in cross-dataset and multilingual cases. The proposed model achieves a relative performance improvement of 37% and 20% on the ASVspoof 2019 and MLAAD datasets, respectively, in in-domain settings compared to the baseline. In the out-of-domain scenario, the model trained on ASVspoof 2019 shows a 36% relative improvement when evaluated on the MLAAD dataset. Across all evaluated languages, the proposed model consistently outperforms the baseline, indicating enhanced domain generalization.
#### Multi-Granularity Adaptive Time-Frequency Attention Framework for Audio Deepfake Detection under Real-World Communication Degradations
 - **Authors:** Haohan Shi, Xiyu Shi, Safak Dogan, Tianjin Huang, Yunxiao Zhang
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2508.01467

 - **Pdf link:** https://arxiv.org/pdf/2508.01467

 - **Abstract**
 The rise of highly convincing synthetic speech poses a growing threat to audio communications. Although existing Audio Deepfake Detection (ADD) methods have demonstrated good performance under clean conditions, their effectiveness drops significantly under degradations such as packet losses and speech codec compression in real-world communication environments. In this work, we propose the first unified framework for robust ADD under such degradations, which is designed to effectively accommodate multiple types of Time-Frequency (TF) representations. The core of our framework is a novel Multi-Granularity Adaptive Attention (MGAA) architecture, which employs a set of customizable multi-scale attention heads to capture both global and local receptive fields across varying TF granularities. A novel adaptive fusion mechanism subsequently adjusts and fuses these attention branches based on the saliency of TF regions, allowing the model to dynamically reallocate its focus according to the characteristics of the degradation. This enables the effective localization and amplification of subtle forgery traces. Extensive experiments demonstrate that the proposed framework consistently outperforms state-of-the-art baselines across various real-world communication degradation scenarios, including six speech codecs and five levels of packet losses. In addition, comparative analysis reveals that the MGAA-enhanced features significantly improve separability between real and fake audio classes and sharpen decision boundaries. These results highlight the robustness and practical deployment potential of our framework in real-world communication environments.
#### An Age-Agnostic System for Robust Speaker Verification
 - **Authors:** Jiusi Zheng, Vishwas Shetty, Natarajan Balaji Shankar, Abeer Alwan
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2508.01637

 - **Pdf link:** https://arxiv.org/pdf/2508.01637

 - **Abstract**
 In speaker verification (SV), the acoustic mismatch between children's and adults' speech leads to suboptimal performance when adult-trained SV systems are applied to children's speaker verification (C-SV). While domain adaptation techniques can enhance performance on C-SV tasks, they often do so at the expense of significant degradation in performance on adults' SV (A-SV) tasks. In this study, we propose an Age Agnostic Speaker Verification (AASV) system that achieves robust performance across both C-SV and A-SV tasks. Our approach employs a domain classifier to disentangle age-related attributes from speech and subsequently expands the embedding space using the extracted domain information, forming a unified speaker representation that is robust and highly discriminative across age groups. Experiments on the OGI and VoxCeleb datasets demonstrate the effectiveness of our approach in bridging SV performance disparities, laying the foundation for inclusive and age-adaptive SV systems.
#### Test-Time Training for Speech Enhancement
 - **Authors:** Avishkar Behera, Riya Ann Easow, Venkatesh Parvathala, K. Sri Rama Murty
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Machine Learning (cs.LG); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2508.01847

 - **Pdf link:** https://arxiv.org/pdf/2508.01847

 - **Abstract**
 This paper introduces a novel application of Test-Time Training (TTT) for Speech Enhancement, addressing the challenges posed by unpredictable noise conditions and domain shifts. This method combines a main speech enhancement task with a self-supervised auxiliary task in a Y-shaped architecture. The model dynamically adapts to new domains during inference time by optimizing the proposed self-supervised tasks like noise-augmented signal reconstruction or masked spectrogram prediction, bypassing the need for labeled data. We further introduce various TTT strategies offering a trade-off between adaptation and efficiency. Evaluations across synthetic and real-world datasets show consistent improvements across speech quality metrics, outperforming the baseline model. This work highlights the effectiveness of TTT in speech enhancement, providing insights for future research in adaptive and robust speech processing.
#### Word Error Rate Definitions and Algorithms for Long-Form Multi-talker Speech Recognition
 - **Authors:** Thilo von Neumann, Christoph Boeddeker, Marc Delcroix, Reinhold Haeb-Umbach
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2508.02112

 - **Pdf link:** https://arxiv.org/pdf/2508.02112

 - **Abstract**
 The predominant metric for evaluating speech recognizers, the Word Error Rate (WER) has been extended in different ways to handle transcripts produced by long-form multi-talker speech recognizers. These systems process long transcripts containing multiple speakers and complex speaking patterns so that the classical WER cannot be applied. There are speaker-attributed approaches that count speaker confusion errors, such as the concatenated minimum-permutation WER cpWER and the time-constrained cpWER (tcpWER), and speaker-agnostic approaches, which aim to ignore speaker confusion errors, such as the Optimal Reference Combination WER (ORC-WER) and the MIMO-WER. These WERs evaluate different aspects and error types (e.g., temporal misalignment). A detailed comparison has not been made. We therefore present a unified description of the existing WERs and highlight when to use which metric. To further analyze how many errors are caused by speaker confusion, we propose the Diarization-invariant cpWER (DI-cpWER). It ignores speaker attribution errors and its difference to cpWER reflects the impact of speaker confusions on the WER. Since error types cannot reliably be classified automatically, we discuss ways to visualize sequence alignments between the reference and hypothesis transcripts to facilitate the spotting of errors by a human judge. Since some WER definitions have high computational complexity, we introduce a greedy algorithm to approximate the ORC-WER and DI-cpWER with high precision ($<0.1\%$ deviation in our experiments) and polynomial complexity instead of exponential. To improve the plausibility of the metrics, we also incorporate the time constraint from the tcpWER into ORC-WER and MIMO-WER, also significantly reducing the computational complexity.
#### Guiding an Automatic Speech Recognition Decoder Using Large Language Models
 - **Authors:** Eyal Cohen (1), Bhiksha Raj (2), Joseph Keshet (1) ((1) Technion - Israel Institute of Technology, (2) Carnegie Mellon University)
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2508.02228

 - **Pdf link:** https://arxiv.org/pdf/2508.02228

 - **Abstract**
 Automatic Speech Recognition (ASR) consists of an acoustic model (AM) and a language model (LM). The AM estimates the probability of an acoustic signal based on a sequence of linguistic units, typically phones, characters, or tokens, while the LM assesses the likelihood of a specific sequence of words or tokens. Although Large Language Models (LLMs) have demonstrated significant potential across various tasks, integrating them into ASR remains an open challenge. By decomposing the maximum a posteriori (MAP) estimator of words (or tokens) given the acoustic signal, we derive an iterative procedure that facilitates a novel integration of the AM and LLM, while maintaining their separability. This approach enables each component to be independently trained and improved using its own data, thereby maximizing the system's performance by leveraging the strengths of both models without requiring joint optimization. We illustrate the effectiveness of our method in comparison to three language models: N-gram, GCNN, and TransformerLM across multiple datasets spanning various speech styles, including ALLSSTAR, WSJ0, and TED-LIUM 3. Our experiments involved two acoustic models (wav2vec 2.0 and HuBERT) and three LLMs (GPT-2, LLaMA 2, and Falcon). Notably, our method demonstrates particular efficacy in addressing complex speech sentences, acronyms, and domain-specific vocabulary.
#### Reference-free Adversarial Sex Obfuscation in Speech
 - **Authors:** Yangyang Qu, Michele Panariello, Massimiliano Todisco, Nicholas Evans
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2508.02295

 - **Pdf link:** https://arxiv.org/pdf/2508.02295

 - **Abstract**
 Sex conversion in speech involves privacy risks from data collection and often leaves residual sex-specific cues in outputs, even when target speaker references are unavailable. We introduce RASO for Reference-free Adversarial Sex Obfuscation. Innovations include a sex-conditional adversarial learning framework to disentangle linguistic content from sex-related acoustic markers and explicit regularisation to align fundamental frequency distributions and formant trajectories with sex-neutral characteristics learned from sex-balanced training data. RASO preserves linguistic content and, even when assessed under a semi-informed attack model, it significantly outperforms a competing approach to sex obfuscation.
#### Revisiting the Privacy of Low-Frequency Speech Signals: Exploring Resampling Methods, Evaluation Scenarios, and Speaker Characteristics
 - **Authors:** Jule Pohlhausen, Jörg Bitzer
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2508.02483

 - **Pdf link:** https://arxiv.org/pdf/2508.02483

 - **Abstract**
 While audio recordings in real life provide insights into social dynamics and conversational behavior, they also raise concerns about the privacy of personal, sensitive data. This article explores the effectiveness of restricting recordings to low-frequency audio to protect spoken content. For resampling the audio signals to different sampling rates, we compare the effect of employing anti-aliasing filtering. Privacy enhancement is measured by an increased word error rate of automatic speech recognition models. The impact on utility performance is measured with voice activity detection models. Our experimental results show that for clean recordings, models trained with a sampling rate of up to 800 Hz transcribe the majority of words correctly. For both models, we analyzed the impact of the speaker's sex and pitch, and we demonstrated that missing anti-aliasing filters more strongly compromise speech privacy.
#### Enhancing Spectrogram Realism in Singing Voice Synthesis via Explicit Bandwidth Extension Prior to Vocoder
 - **Authors:** Runxuan Yang, Kai Li, Guo Chen, Xiaolin Hu
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2508.01796

 - **Pdf link:** https://arxiv.org/pdf/2508.01796

 - **Abstract**
 This paper addresses the challenge of enhancing the realism of vocoder-generated singing voice audio by mitigating the distinguishable disparities between synthetic and real-life recordings, particularly in high-frequency spectrogram components. Our proposed approach combines two innovations: an explicit linear spectrogram estimation step using denoising diffusion process with DiT-based neural network architecture optimized for time-frequency data, and a redesigned vocoder based on Vocos specialized in handling large linear spectrograms with increased frequency bins. This integrated method can produce audio with high-fidelity spectrograms that are challenging for both human listeners and machine classifiers to differentiate from authentic recordings. Objective and subjective evaluations demonstrate that our streamlined approach maintains high audio quality while achieving this realism. This work presents a substantial advancement in overcoming the limitations of current vocoding techniques, particularly in the context of adversarial attacks on fake spectrogram detection.
#### Marco-Voice Technical Report
 - **Authors:** Fengping Tian, Chenyang Lyu, Xuanfan Ni, Haoqin Sun, Qingjuan Li, Zhiqiang Qian, Haijun Li, Longyue Wang, Zhao Xu, Weihua Luo, Kaifu Zhang
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2508.02038

 - **Pdf link:** https://arxiv.org/pdf/2508.02038

 - **Abstract**
 This paper presents a multifunctional speech synthesis system that integrates voice cloning and emotion control speech synthesis within a unified framework. The goal of this work is to address longstanding challenges in achieving highly expressive, controllable, and natural speech generation that faithfully preserves speaker identity across diverse linguistic and emotional contexts. Our approach introduces an effective speaker-emotion disentanglement mechanism with in-batch contrastive learning, enabling independent manipulation of speaker identity and eemotional style, as well as rotational emotional embedding integration method for smooth emotion control. To support comprehensive training and evaluation, we construct CSEMOTIONS, a high-quality emotional speech dataset containing 10 hours of Mandarin speech from six professional speakers across seven emotional categories. Extensive experiments demonstrate that our system, Marco-Voice, achieves substantial improvements in both objective and subjective metrics. Comprehensive evaluations and analysis were conducted, results show that MarcoVoice delivers competitive performance in terms of speech clarity and emotional richness, representing a substantial advance in the field of expressive neural speech synthesis.
#### Unsupervised Multi-channel Speech Dereverberation via Diffusion
 - **Authors:** Yulun Wu, Zhongweiyang Xu, Jianchong Chen, Zhong-Qiu Wang, Romit Roy Choudhury
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2508.02071

 - **Pdf link:** https://arxiv.org/pdf/2508.02071

 - **Abstract**
 We consider the problem of multi-channel single-speaker blind dereverberation, where multi-channel mixtures are used to recover the clean anechoic speech. To solve this problem, we propose USD-DPS, {U}nsupervised {S}peech {D}ereverberation via {D}iffusion {P}osterior {S}ampling. USD-DPS uses an unconditional clean speech diffusion model as a strong prior to solve the problem by posterior sampling. At each diffusion sampling step, we estimate all microphone channels' room impulse responses (RIRs), which are further used to enforce a multi-channel mixture consistency constraint for diffusion guidance. For multi-channel RIR estimation, we estimate reference-channel RIR by optimizing RIR parameters of a sub-band RIR signal model, with the Adam optimizer. We estimate non-reference channels' RIRs analytically using forward convolutive prediction (FCP). We found that this combination provides a good balance between sampling efficiency and RIR prior modeling, which shows superior performance among unsupervised dereverberation approaches. An audio demo page is provided in this https URL.
#### WhiSQA: Non-Intrusive Speech Quality Prediction Using Whisper Encoder Features
 - **Authors:** George Close, Kris Hong, Thomas Hain, Stefan Goetze
 - **Subjects:** Subjects:
Sound (cs.SD); Machine Learning (cs.LG); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2508.02210

 - **Pdf link:** https://arxiv.org/pdf/2508.02210

 - **Abstract**
 There has been significant research effort developing neural-network-based predictors of SQ in recent years. While a primary objective has been to develop non-intrusive, i.e.~reference-free, metrics to assess the performance of SE systems, recent work has also investigated the direct inference of neural SQ predictors within the loss function of downstream speech tasks. To aid in the training of SQ predictors, several large datasets of audio with corresponding human labels of quality have been created. Recent work in this area has shown that speech representations derived from large unsupervised or semi-supervised foundational speech models are useful input feature representations for neural SQ prediction. In this work, a novel and robust SQ predictor is proposed based on feature representations extracted from an ASR model, found to be a powerful input feature for the SQ prediction task. The proposed system achieves higher correlation with human MOS ratings than recent approaches on all NISQA test sets and shows significantly better domain adaption compared to the commonly used DNSMOS metric.
#### StutterCut: Uncertainty-Guided Normalised Cut for Dysfluency Segmentation
 - **Authors:** Suhita Ghosh, Melanie Jouaiti, Jan-Ole Perschewski, Sebastian Stober
 - **Subjects:** Subjects:
Sound (cs.SD); Artificial Intelligence (cs.AI); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2508.02255

 - **Pdf link:** https://arxiv.org/pdf/2508.02255

 - **Abstract**
 Detecting and segmenting dysfluencies is crucial for effective speech therapy and real-time feedback. However, most methods only classify dysfluencies at the utterance level. We introduce StutterCut, a semi-supervised framework that formulates dysfluency segmentation as a graph partitioning problem, where speech embeddings from overlapping windows are represented as graph nodes. We refine the connections between nodes using a pseudo-oracle classifier trained on weak (utterance-level) labels, with its influence controlled by an uncertainty measure from Monte Carlo dropout. Additionally, we extend the weakly labelled FluencyBank dataset by incorporating frame-level dysfluency boundaries for four dysfluency types. This provides a more realistic benchmark compared to synthetic datasets. Experiments on real and synthetic datasets show that StutterCut outperforms existing methods, achieving higher F1 scores and more precise stuttering onset detection.
#### Perception of dynamic multi-speaker auditory scenes under different modes of attention
 - **Authors:** Stephanie Graceffo, David F Little, Emine Merve Kaya, Mounya Elhilali
 - **Subjects:** Subjects:
Neurons and Cognition (q-bio.NC); Audio and Speech Processing (eess.AS); Signal Processing (eess.SP)
 - **Arxiv link:** https://arxiv.org/abs/2508.02620

 - **Pdf link:** https://arxiv.org/pdf/2508.02620

 - **Abstract**
 Attention is not monolithic; rather, it operates in multiple forms to facilitate efficient cognitive processing. In the auditory domain, attention enables the prioritization of relevant sounds in an auditory scene and can be either attracted by elements in the scene in a bottom-up fashion or directed towards features, objects, or the entire scene in a top-down fashion. How these modes of attention interact and whether their neural underpinnings are distinct remains unclear. In this work, we investigate the perceptual and neural correlates of different attentional modes in a controlled "cocktail party" paradigm, where listeners listen to the same stimuli and attend to either a spatial location (feature-based), a speaker (object-based), or the entire scene (global or free-listening) while detecting deviations in pitch of a voice in the scene. Our findings indicate that object-based attention is more perceptually effective than feature-based or global attention. Furthermore, object-based and spatial-based attention engage distinct neural mechanisms and are differentially modulated by bottom-up salience. Notably, while bottom-up salience aids in the initial segregation of auditory objects, it plays a reduced role in object tracking once attention has been voluntarily allocated. In addition, decoding the stimulus envelope from the EEG data revealed a source-sampling scheme in the global attention mode that is not present in the object or spatial modes. Overall, the study shows that the perception of the same acoustic scene differs according to the listening task, guided by an interaction between top-down and bottom-up processes.


by Zyzzyva0381 (Windy). 


2025-08-05
