# Showing new listings for Tuesday, 24 June 2025
Auto update papers at about 2:30am UTC (10:30am Beijing time) every weekday.


阅读 `Usage.md`了解如何使用此repo实现个性化的Arxiv论文推送

See `Usage.md` for instructions on how to personalize the repo. 


Keyword list: ['text-to-speech', 'text to speech', 'tts', 'LLM-based', 'speech', 'voice']


Excluded: []


### Today: 13papers 
#### Enhancing Few-shot Keyword Spotting Performance through Pre-Trained Self-supervised Speech Models
 - **Authors:** Alican Gok, Oguzhan Buyuksolak, Osman Erman Okman, Murat Saraclar
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Computation and Language (cs.CL); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2506.17686

 - **Pdf link:** https://arxiv.org/pdf/2506.17686

 - **Abstract**
 Keyword Spotting plays a critical role in enabling hands-free interaction for battery-powered edge devices. Few-Shot Keyword Spotting (FS-KWS) addresses the scalability and adaptability challenges of traditional systems by enabling recognition of custom keywords with only a few examples. However, existing FS-KWS systems achieve subpar accuracy at desirable false acceptance rates, particularly in resource-constrained edge environments. To address these issues, we propose a training scheme that leverages self-supervised learning models for robust feature extraction, dimensionality reduction, and knowledge distillation. The teacher model, based on Wav2Vec 2.0 is trained using Sub-center ArcFace loss, which enhances inter-class separability and intra-class compactness. To enable efficient deployment on edge devices, we introduce attention-based dimensionality reduction and train a standard lightweight ResNet15 student model. We evaluate the proposed approach on the English portion of the Multilingual Spoken Words Corpus (MSWC) and the Google Speech Commands (GSC) datasets. Notably, the proposed training method improves the 10-shot classification accuracy from 33.4% to 74.1% on 11 classes at 1% false alarm accuracy on the GSC dataset, thus making it significantly better-suited for a real use case scenario.
#### Zero-Shot Cognitive Impairment Detection from Speech Using AudioLLM
 - **Authors:** Mostafa Shahin, Beena Ahmed, Julien Epps
 - **Subjects:** Subjects:
Sound (cs.SD); Artificial Intelligence (cs.AI); Computation and Language (cs.CL); Multimedia (cs.MM); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2506.17351

 - **Pdf link:** https://arxiv.org/pdf/2506.17351

 - **Abstract**
 Cognitive impairment (CI) is of growing public health concern, and early detection is vital for effective intervention. Speech has gained attention as a non-invasive and easily collectible biomarker for assessing cognitive decline. Traditional CI detection methods typically rely on supervised models trained on acoustic and linguistic features extracted from speech, which often require manual annotation and may not generalise well across datasets and languages. In this work, we propose the first zero-shot speech-based CI detection method using the Qwen2- Audio AudioLLM, a model capable of processing both audio and text inputs. By designing prompt-based instructions, we guide the model in classifying speech samples as indicative of normal cognition or cognitive impairment. We evaluate our approach on two datasets: one in English and another multilingual, spanning different cognitive assessment tasks. Our results show that the zero-shot AudioLLM approach achieves performance comparable to supervised methods and exhibits promising generalizability and consistency across languages, tasks, and datasets.
#### Splitformer: An improved early-exit architecture for automatic speech recognition on edge devices
 - **Authors:** Maxence Lasbordes, Daniele Falavigna, Alessio Brutti
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2506.18035

 - **Pdf link:** https://arxiv.org/pdf/2506.18035

 - **Abstract**
 The ability to dynamically adjust the computational load of neural models during inference in a resource aware manner is crucial for on-device processing scenarios, characterised by limited and time-varying computational resources. Early-exit architectures represent an elegant and effective solution, since they can process the input with a subset of their layers, exiting at intermediate branches (the upmost layers are hence removed from the model). From a different perspective, for automatic speech recognition applications there are memory-efficient neural architectures that apply variable frame rate analysis, through downsampling/upsampling operations in the middle layers, reducing the overall number of operations and improving significantly the performance on well established benchmarks. One example is the Zipformer. However, these architectures lack the modularity necessary to inject early-exit branches. With the aim of improving the performance in early-exit models, we propose introducing parallel layers in the architecture that process downsampled versions of their inputs. % in conjunction with standard processing layers. We show that in this way the speech recognition performance on standard benchmarks significantly improve, at the cost of a small increase in the overall number of model parameters but without affecting the inference time.
#### Face-Voice Association for Audiovisual Active Speaker Detection in Egocentric Recordings
 - **Authors:** Jason Clarke, Yoshihiko Gotoh, Stefan Goetze
 - **Subjects:** Subjects:
Multimedia (cs.MM); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2506.18055

 - **Pdf link:** https://arxiv.org/pdf/2506.18055

 - **Abstract**
 Audiovisual active speaker detection (ASD) is conventionally performed by modelling the temporal synchronisation of acoustic and visual speech cues. In egocentric recordings, however, the efficacy of synchronisation-based methods is compromised by occlusions, motion blur, and adverse acoustic conditions. In this work, a novel framework is proposed that exclusively leverages cross-modal face-voice associations to determine speaker activity. An existing face-voice association model is integrated with a transformer-based encoder that aggregates facial identity information by dynamically weighting each frame based on its visual quality. This system is then coupled with a front-end utterance segmentation method, producing a complete ASD system. This work demonstrates that the proposed system, Self-Lifting for audiovisual active speaker detection(SL-ASD), achieves performance comparable to, and in certain cases exceeding, that of parameter-intensive synchronisation-based approaches with significantly fewer learnable parameters, thereby validating the feasibility of substituting strict audiovisual synchronisation modelling with flexible biometric associations in challenging egocentric scenarios.
#### AI Harmonizer: Expanding Vocal Expression with a Generative Neurosymbolic Music AI System
 - **Authors:** Lancelot Blanchard, Cameron Holt, Joseph A. Paradiso
 - **Subjects:** Subjects:
Human-Computer Interaction (cs.HC); Artificial Intelligence (cs.AI); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2506.18143

 - **Pdf link:** https://arxiv.org/pdf/2506.18143

 - **Abstract**
 Vocals harmonizers are powerful tools to help solo vocalists enrich their melodies with harmonically supportive voices. These tools exist in various forms, from commercially available pedals and software to custom-built systems, each employing different methods to generate harmonies. Traditional harmonizers often require users to manually specify a key or tonal center, while others allow pitch selection via an external keyboard-both approaches demanding some degree of musical expertise. The AI Harmonizer introduces a novel approach by autonomously generating musically coherent four-part harmonies without requiring prior harmonic input from the user. By integrating state-of-the-art generative AI techniques for pitch detection and voice modeling with custom-trained symbolic music models, our system arranges any vocal melody into rich choral textures. In this paper, we present our methods, explore potential applications in performance and composition, and discuss future directions for real-time implementations. While our system currently operates offline, we believe it represents a significant step toward AI-assisted vocal performance and expressive musical augmentation. We release our implementation on GitHub.
#### Human Voice is Unique
 - **Authors:** Rita Singh, Bhiksha Raj
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2506.18182

 - **Pdf link:** https://arxiv.org/pdf/2506.18182

 - **Abstract**
 Voice is increasingly being used as a biometric entity in many applications. These range from speaker identification and verification systems to human profiling technologies that attempt to estimate myriad aspects of the speaker's persona from their voice. However, for an entity to be a true biometric identifier, it must be unique. This paper establishes a first framework for calculating the uniqueness of human voice objectively. The approach in this paper is based on statistical considerations that take into account a set of measurable characteristics of the voice signal that bear a causal relationship to the vocal production process, but are not inter-dependent or derivable from each other. Depending on how we quantize these variables, we show that the chances of two people having the same voice in a world populated by 10 billion people range from one in a few thousand, to one in a septillion or less. The paper also discusses the implications of these calculations on the choices made in voice processing applications.
#### JIS: A Speech Corpus of Japanese Idol Speakers with Various Speaking Styles
 - **Authors:** Yuto Kondo, Hirokazu Kameoka, Kou Tanaka, Takuhiro Kaneko
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2506.18296

 - **Pdf link:** https://arxiv.org/pdf/2506.18296

 - **Abstract**
 We construct Japanese Idol Speech Corpus (JIS) to advance research in speech generation AI, including text-to-speech synthesis (TTS) and voice conversion (VC). JIS will facilitate more rigorous evaluations of speaker similarity in TTS and VC systems since all speakers in JIS belong to a highly specific category: "young female live idols" in Japan, and each speaker is identified by a stage name, enabling researchers to recruit listeners familiar with these idols for listening experiments. With its unique speaker attributes, JIS will foster compelling research, including generating voices tailored to listener preferences-an area not yet widely studied. JIS will be distributed free of charge to promote research in speech generation AI, with usage restricted to non-commercial, basic research. We describe the construction of JIS, provide an overview of Japanese live idol culture to support effective and ethical use of JIS, and offer a basic analysis to guide application of JIS.
#### Rethinking Mean Opinion Scores in Speech Quality Assessment: Aggregation through Quantized Distribution Fitting
 - **Authors:** Yuto Kondo, Hirokazu Kameoka, Kou Tanaka, Takuhiro Kaneko
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2506.18307

 - **Pdf link:** https://arxiv.org/pdf/2506.18307

 - **Abstract**
 Speech quality assessment (SQA) aims to evaluate the quality of speech samples without relying on time-consuming listener questionnaires. Recent efforts have focused on training neural-based SQA models to predict the mean opinion score (MOS) of speech samples produced by text-to-speech or voice conversion systems. This paper targets the enhancement of MOS prediction models' performance. We propose a novel score aggregation method to address the limitations of conventional annotations for MOS, which typically involve ratings on a scale from 1 to 5. Our method is based on the hypothesis that annotators internally consider continuous scores and then choose the nearest discrete rating. By modeling this process, we approximate the generative distribution of ratings by quantizing the latent continuous distribution. We then use the peak of this latent distribution, estimated through the loss between the quantized distribution and annotated ratings, as a new representative value instead of MOS. Experimental results demonstrate that substituting MOSNet's predicted target with this proposed value improves prediction performance.
#### Selecting N-lowest scores for training MOS prediction models
 - **Authors:** Yuto Kondo, Hirokazu Kameoka, Kou Tanaka, Takuhiro Kaneko
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2506.18326

 - **Pdf link:** https://arxiv.org/pdf/2506.18326

 - **Abstract**
 The automatic speech quality assessment (SQA) has been extensively studied to predict the speech quality without time-consuming questionnaires. Recently, neural-based SQA models have been actively developed for speech samples produced by text-to-speech or voice conversion, with a primary focus on training mean opinion score (MOS) prediction models. The quality of each speech sample may not be consistent across the entire duration, and it remains unclear which segments of the speech receive the primary focus from humans when assigning subjective evaluation for MOS calculation. We hypothesize that when humans rate speech, they tend to assign more weight to low-quality speech segments, and the variance in ratings for each sample is mainly due to accidental assignment of higher scores when overlooking the poor quality speech segments. Motivated by the hypothesis, we analyze the VCC2018 and BVCC datasets. Based on the hypothesis, we propose the more reliable representative value N_low-MOS, the mean of the $N$-lowest opinion scores. Our experiments show that LCC and SRCC improve compared to regular MOS when employing N_low-MOS to MOSNet training. This result suggests that N_low-MOS is a more intrinsic representative value of subjective speech quality and makes MOSNet a better comparator of VC models.
#### Smooth Operators: LLMs Translating Imperfect Hints into Disfluency-Rich Transcripts
 - **Authors:** Duygu Altinok
 - **Subjects:** Subjects:
Sound (cs.SD); Artificial Intelligence (cs.AI); Computation and Language (cs.CL); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2506.18510

 - **Pdf link:** https://arxiv.org/pdf/2506.18510

 - **Abstract**
 Accurate detection of disfluencies in spoken language is crucial for enhancing the performance of automatic speech and language processing systems, as well as fostering the development of more inclusive speech and language technologies. Leveraging the growing trend of large language models (LLMs) as versatile learners capable of processing both lexical and non-lexical inputs (e.g., audio and video), we propose a novel approach to transcribing disfluencies as explicit tokens with timestamps, enabling the generation of fully annotated disfluency-rich transcripts. Our method integrates acoustic representations extracted from an audio encoder with textual inputs of varying quality: clean transcriptions without disfluencies, time-aligned transcriptions from aligners, or outputs from phoneme-based ASR models -- all of which may contain imperfections. Importantly, our experiments demonstrate that textual inputs do not need to be flawless. As long as they include timestamp-related cues, LLMs can effectively smooth the input and produce fully disfluency-annotated transcripts, underscoring their robustness in handling imperfect hints.
#### Evaluating Multichannel Speech Enhancement Algorithms at the Phoneme Scale Across Genders
 - **Authors:** Nasser-Eddine Monir, Paul Magron, Romain Serizel
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2506.18691

 - **Pdf link:** https://arxiv.org/pdf/2506.18691

 - **Abstract**
 Multichannel speech enhancement algorithms are essential for improving the intelligibility of speech signals in noisy environments. These algorithms are usually evaluated at the utterance level, but this approach overlooks the disparities in acoustic characteristics that are observed in different phoneme categories and between male and female speakers. In this paper, we investigate the impact of gender and phonetic content on speech enhancement algorithms. We motivate this approach by outlining phoneme- and gender-specific spectral features. Our experiments reveal that while utterance-level differences between genders are minimal, significant variations emerge at the phoneme level. Results show that the tested algorithms better reduce interference with fewer artifacts on female speech, particularly in plosives, fricatives, and vowels. Additionally, they demonstrate greater performance for female speech in terms of perceptual and speech recognition metrics.
#### Frequency-Weighted Training Losses for Phoneme-Level DNN-based Speech Enhancement
 - **Authors:** Nasser-Eddine Monir, Paul Magron, Romain Serizel
 - **Subjects:** Subjects:
Sound (cs.SD); Artificial Intelligence (cs.AI); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2506.18714

 - **Pdf link:** https://arxiv.org/pdf/2506.18714

 - **Abstract**
 Recent advances in deep learning have significantly improved multichannel speech enhancement algorithms, yet conventional training loss functions such as the scale-invariant signal-to-distortion ratio (SDR) may fail to preserve fine-grained spectral cues essential for phoneme intelligibility. In this work, we propose perceptually-informed variants of the SDR loss, formulated in the time-frequency domain and modulated by frequency-dependent weighting schemes. These weights are designed to emphasize time-frequency regions where speech is prominent or where the interfering noise is particularly strong. We investigate both fixed and adaptive strategies, including ANSI band-importance weights, spectral magnitude-based weighting, and dynamic weighting based on the relative amount of speech and noise. We train the FaSNet multichannel speech enhancement model using these various losses. Experimental results show that while standard metrics such as the SDR are only marginally improved, their perceptual frequency-weighted counterparts exhibit a more substantial improvement. Besides, spectral and phoneme-level analysis indicates better consonant reconstruction, which points to a better preservation of certain acoustic cues.
#### USAD: Universal Speech and Audio Representation via Distillation
 - **Authors:** Heng-Jui Chang, Saurabhchand Bhati, James Glass, Alexander H. Liu
 - **Subjects:** Subjects:
Sound (cs.SD); Computation and Language (cs.CL); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2506.18843

 - **Pdf link:** https://arxiv.org/pdf/2506.18843

 - **Abstract**
 Self-supervised learning (SSL) has revolutionized audio representations, yet models often remain domain-specific, focusing on either speech or non-speech tasks. In this work, we present Universal Speech and Audio Distillation (USAD), a unified approach to audio representation learning that integrates diverse audio types - speech, sound, and music - into a single model. USAD employs efficient layer-to-layer distillation from domain-specific SSL models to train a student on a comprehensive audio dataset. USAD offers competitive performance across various benchmarks and datasets, including frame and instance-level speech processing tasks, audio tagging, and sound classification, achieving near state-of-the-art results with a single encoder on SUPERB and HEAR benchmarks.


by Zyzzyva0381 (Windy). 


2025-06-24
