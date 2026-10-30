# Showing new listings for Friday, 30 October 2026
Auto update papers at about 2:30am UTC (10:30am Beijing time) every weekday.


阅读 `Usage.md`了解如何使用此repo实现个性化的Arxiv论文推送

See `Usage.md` for instructions on how to personalize the repo. 


Keyword list: ['text-to-speech', 'text to speech', 'tts', 'LLM-based', 'speech', 'voice']


Excluded: []


### Today: 17papers 
#### Beyond Utterance-Level Speaker Similarity: Perception-Aligned Segmental Evaluation of Accent in Speech Generation
 - **Authors:** Jinzuomu Zhong, Alice Ross, Catherine Lai, Korin Richmond
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Computation and Language (cs.CL); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2610.37912

 - **Pdf link:** https://arxiv.org/pdf/2610.37912

 - **Abstract**
 Utterance-level similarity ratings do not reveal where perceived accent differences occur in generated speech. We introduce a segmental evaluation protocol combining speaker and accent similarity ratings with subword annotations of perceived accent differences provided by listeners from target accent regions. Using newly collected recordings, 47 retained listeners evaluate 200 content-matched reference--candidate pairs spanning four Scottish and Australian voices, two zero-shot TTS systems, and an articulation-based voice conversion system. Subword annotations locate salient pronunciation differences and sometimes yield different system rankings from similarity ratings. Speaker and accent ratings remain strongly associated ($\rho = 0.71$), highlighting the value of examining local perceptual evidence beyond these scores. Evaluation of seven objective distances reveals weak correspondence with subword selections, with correlations reaching at most $\rho = 0.22$. We release code, data, and an interactive demo to support the development of perceptually grounded segmental accent evaluation.
#### ODAQ-Collection: Expanding the Open Dataset of Audio Quality with a Focus on Spatial Distortions
 - **Authors:** Chih-Wei Wu, Matteo Torcoli, Mhd Modar Halimeh, Pablo M. Delgado, Sascha Dick, Christoph Thompson, Phillip A. Williams, Emanuël A. P. Habets
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2610.37959

 - **Pdf link:** https://arxiv.org/pdf/2610.37959

 - **Abstract**
 The Open Dataset of Audio Quality (ODAQ) is an expanding resource that addresses the scarcity of openly available data in audio quality assessment research. While previous contributions focused mainly on timbral distortions, with spatial processing limited to basic mid-side and left-right stereo conditions, we present a new contribution featuring high-quality music and speech recordings with diverse stereo images, controlled spatial distortions, and subjective listening test results from trained listeners in a laboratory setting. With 148 pairs of stereo audio samples processed by three different method classes, two applying controlled binaural cue distortions and one applying dialogue enhancement, this addition expands the collection to 688 pairs of processed and unprocessed audio signals spanning 17 processing method classes, along with 18,984 raw subjective ratings from listeners. All associated data and code are released as open resources to facilitate future expansions and research on objective audio quality metrics.
#### Improving the Dependency on Fine-Tuning Data in Transfer Learning for Speech Emotion Recognition
 - **Authors:** Ricardo García, Rodrigo Mahu, Carlos Busso, Néstor Becerra Yoma
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2610.38050

 - **Pdf link:** https://arxiv.org/pdf/2610.38050

 - **Abstract**
 Compensating for train-test mismatches in acoustic environments is important for deployable speech solutions. This paper proposes to reduce the amount of fine-tuning data needed in transfer learning to mitigate the train-test mismatch by using a compensation mechanism based on the feature maximum likelihood linear regression (fMLLR) algorithm. Our scheme incorporates an autoencoder to reduce the number of trainable parameters. Operating on the encoder latent space, our compensation strategy employs an affine linear transform that accounts for rotations and shifts in the feature space. We evaluate the proposed solution in a speech emotion recognition (SER) task within a realistic human-robot interaction scenario. The results show that the proposed architecture outperforms the original scheme without the compensation strategy. We observe a performance gain in the mean concordance correlation coefficient (CCC) of 16.8% using 12.9% of the training set, and 14.4% using 50% of the training set.
#### A Tutorial on Generative Models for Music Audio
 - **Authors:** Mateusz Modrzejewski, Joanna Kurkowska, Dorien Herremans
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Artificial Intelligence (cs.AI); Emerging Technologies (cs.ET); Multimedia (cs.MM); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2610.38252

 - **Pdf link:** https://arxiv.org/pdf/2610.38252

 - **Abstract**
 Music audio generation spans instrumental clips, full songs with vocals, edits of existing recordings, and systems that play alongside a musician in real time. This monograph explains the methods behind these systems for readers familiar with neural-network training who are new to music generation. It introduces neural audio codecs and autoencoders, autoregressive and masked token models, diffusion and flow matching, sampling acceleration, and preference optimization in a shared notation. Worked examples connect training objectives and sampling procedures to published systems. The application sections relate the musical requirements of instrumental generation, songs, editing, and live performance to model design. Comparison tables record representations, methods, inputs, tasks, and release information documented in papers and official sources. Symbolic composition, speech, and sound effects are included where they connect to music audio. We explain how to interpret automatic metrics and learned judges, design and report listening tests, and assess evidence about data provenance, memorization, attribution, and output detection. Selected legal decisions and disputes illustrate the implications for training and release. The reviewed systems combine established generative paradigms with new representations, planning stages, and post-training procedures. Differences in data, scale, and evaluation limit what cross-system comparisons can establish about the sources of reported gains.
#### DRC-TTS: A Dual-Resolution Codec-Based Architecture for Lexicon-Free Malay Speech Synthesis
 - **Authors:** Tianhui Su, Tien-Ping Tan
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2610.38316

 - **Pdf link:** https://arxiv.org/pdf/2610.38316

 - **Abstract**
 Modern discrete neural audio codecs facilitate speech synthesis by replacing continuous acoustic representations with low-frame-rate discrete token sequences. For Malay speech synthesis, character-level input reduces reliance on pronunciation lexicons. However, combining character-level inputs with low-frame-rate discrete codecs introduces a temporal alignment challenge. When predicted character durations are quantized into frame counts at 12.5 Hz, durations shorter than one codec frame may be assigned zero frames. Consequently, the corresponding character representations may be lost during length regulation, potentially leading to pronunciation omissions. This study proposes DRC-TTS, a dual-resolution codec-based architecture for lexicon-free Malay speech synthesis. The model expands character representations at a temporal resolution of 50 Hz and subsequently downsamples them through strided convolution to 12.5 Hz for discrete codec-token prediction. Objective evaluations show that the proposed model reduces the character error rate (CER) from 27.11% for the baseline to 21.86%, an absolute reduction of 5.25 percentage points. The proposed model also achieves a mean opinion score (MOS) of 4.30, compared with 4.12 for the baseline. These results suggest that the dual-resolution design may improve character-level alignment and perceived speech naturalness under the evaluated conditions.
#### MiDiff: MIDI-Conditioned Diffusion Models for Drum Audio Enhancement
 - **Authors:** Elad Aspis, Sharon Gannot
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Signal Processing (eess.SP)
 - **Arxiv link:** https://arxiv.org/abs/2610.38619

 - **Pdf link:** https://arxiv.org/pdf/2610.38619

 - **Abstract**
 Bridging the semantic gap between perfect symbolic musical structure and severely degraded acoustic signals offers a powerful paradigm for high-fidelity audio restoration. We introduce MiDiff, a diffusion-based generative model for drum audio enhancement that conditions the reverse process on temporally aligned Musical Instrument Digital Interface (MIDI) piano rolls. We extend the Score-Based Generative Models for Speech Enhancement (SGMSE) framework in the complex Short-Time Fourier Transform (STFT) domain by integrating a pretrained MIDI encoder, multi-scale temporal alignment, and three complementary conditioning strategies: early injection, Feature-wise Linear Modulation (FiLM), and cross-attention. We further show that providing the same piano roll only through classifier guidance, driven by an Automatic Drum Transcription (ADT) model on the audio-only score, yields no measurable gain: the symbolic prior is effective when it conditions the score network. To relax the requirement for ground-truth MIDI at inference, we further propose a hybrid reverse schedule that transcribes the diffusion state online with the same noise-aware ADT network and hands over from the audio-only score to MiDiff at a tunable switch time, cutting test FAD by 47% relative to the audio-only baseline without any symbolic input (switch time selected on the test set). We train our model on the StemGMD dataset using a non-additive degradation pipeline that simulates room reverberation with echo, electret capsule frequency response, and soft-clipping saturation with noise. MiDiff substantially outperforms the SGMSE baseline in terms of Fréchet Audio Distance (FAD) on both the validation and held-out test sets. Ablation studies demonstrate that combining all three conditioning strategies yields superior performance and that authentic MIDI velocity dynamics are critical for effective symbolic conditioning.
#### Wav2Aug: A Drop-In Speech Augmentation Policy
 - **Authors:** Gianfranco Dumoulin Bertucci, Mirco Ravanelli
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2610.38656

 - **Pdf link:** https://arxiv.org/pdf/2610.38656

 - **Abstract**
 SpecAugment is the dominant speech augmentation policy, but researchers have studied it primarily for automatic speech recognition (ASR), and it operates on spectrograms, limiting its applicability to other speech tasks and waveform models. We propose Wav2Aug, a drop-in waveform augmentation policy that randomly samples two augmentations from a calibrated pool of waveform transformations. We calibrate the augmentation hyperparameters once and reuse them across tasks and architectures, eliminating task-specific tuning. Experiments on ten speech benchmarks show that Wav2Aug achieves ASR performance comparable to SpecAugment while improving over it on average across non-ASR tasks, yielding an average relative error reduction of 12.8% over the strongest compatible SpecAugment baseline. We further show that multi-view training provides additional gains, with two augmented views offering the best tradeoff between performance and computational cost.
#### Streaming Speech Recognition on a Microcontroller: a 65.7M-Parameter Transducer in 2 and 4 Bits on the ESP32-P4
 - **Authors:** Billal Boumedine
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Hardware Architecture (cs.AR); Machine Learning (cs.LG)
 - **Arxiv link:** https://arxiv.org/abs/2610.38739

 - **Pdf link:** https://arxiv.org/pdf/2610.38739

 - **Abstract**
 We run Kroko, a 65.7M-parameter streaming Zipformer2 transducer for English, on the ESP32-P4, a microcontroller with two 400 MHz RISC-V cores, a 128-bit SIMD extension (PIE) and 32 MB of PSRAM, but no neural accelerator. Without any training, the encoder's 290 matrices are stored with 2 or 4 bits per weight in Hadamard-rotated groups: the width of each matrix comes from its measured sensitivity, and each matrix is first corrected toward the original model's output, then rounded with GPTQ and a search over the group norms. The released 71 MB of int8 weights become a 26.65 MB file. Activations are int16 between the matrix products and int8 into them. A C engine with hand-written PIE kernels processes each 1.28 s chunk of audio in 0.90 s on both cores, beam search included, and its PC build computes the same bits as the chip, so accuracy is measured on whole test sets. LibriSpeech WER is 4.34% on test-clean and 10.26% on test-other, against 3.94% and 9.17% for the released model; on FLEURS and Common Voice the gap grows to 2.3 and 1.1 points. A GPU simulator shows that the integer arithmetic costs nothing and that the whole gap comes from the weights, and an ablation on all of test-clean shows that GPTQ does most of the work. We also report the port's speed history, from 51.3 s to 0.62 s of encoder time per chunk, each design step measured again in one build, an attempt to reach 16 MB that stopped at 19.63 MB, and what did not work. We did not measure energy. The inference engine, the firmware and the 2/4-bit weights are available at this https URL.
#### The Importance of Speech Genre for End-of-Turn Detector Training
 - **Authors:** Rini Sharon, Kadri Hacioglu, Jeena Prakash, Andreas Stolcke
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Artificial Intelligence (cs.AI)
 - **Arxiv link:** https://arxiv.org/abs/2610.39742

 - **Pdf link:** https://arxiv.org/pdf/2610.39742

 - **Abstract**
 End-of-turn (EoT) detection has been studied mostly through features, models, and modalities, while the speech genre of the training data has drawn far less attention. We isolate this factor. Using one reproducible annotation pipeline, we build a matched pair of corpora, multi-turn human-human conversational and single-turn human-machine, and train the same streaming ConvLSTM EoT detector on each. Across four test sets the conversational model reaches up to +0.45 higher F1, driven by fewer false alarms on the mid-turn pauses and hesitations that are rare in single-turn corpora. This conversational-training effect is not specific to frame-level decoding. Under a matched complete-or-not test it also appears at the utterance level, across two encoders, but only on conversational test data. Testing only on single-turn speech, as most prior work does, therefore hides the benefit of conversational training data. Against open-source baselines, our 1.8M-parameter model attains the highest F1 on every test set.
#### Towards Multilingual End-of-Turn Detection using Frozen Speech Encoders
 - **Authors:** Rini Sharon, Kadri Hacioglu, Andreas Stolcke
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2610.39755

 - **Pdf link:** https://arxiv.org/pdf/2610.39755

 - **Abstract**
 A spoken dialogue system must judge when the user has yielded the floor. This end-of-turn (EoT) detector is normally trained and tuned afresh for each language, which is costly to scale to new languages. We instead build one on a frozen multilingual encoder, training only a small fixed-size head. Using five CALLHOME languages, we investigate three questions. First, how well a head trained on one language transfers zero-shot to the others. Second, how much target-language adaptation is required to recover full task performance. Third, whether a single fixed decision threshold holds across languages without per-language tuning. We find that the head transfers well, even between tonal and pitch-accent languages, and that a few hours of audio recover nearly all the loss, matching in-language training. A frozen multilingual encoder with one lightweight head is a practical, language-agnostic baseline for EoT detection.
#### SpeakerTurn: Target-Speaker Semantic Turn Detection in Multi-Speaker Environments
 - **Authors:** Nan Li, Hetao Shan, Jintao Liang, Jiahao Yan, Longbiao Wang, Tubin Huang, Min Xu
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2610.39828

 - **Pdf link:** https://arxiv.org/pdf/2610.39828

 - **Abstract**
 Semantic completeness alone cannot determine when an assistant should respond in a multi-speaker environment: a completed utterance may belong to an interfering speaker, while the designated user is still speaking, acknowledging, or requesting time. We therefore study semantic turn inference conditioned on speaker identity. We propose SpeakerTurn, a framework that jointly predicts the target transcript and turn state from a speech mixture and an independent target-speaker reference. Built on CocktailASR-1, it links semantic decisions to the corresponding target content through joint supervision, distinguishing complete, incomplete, backchannel, wait, and target-absent states, with a separate interruption task. We also examine whether structured chain-of-thought (CoT) supervision improves decisions by representing speaker presence and semantic cues. On a synthetic multi-speaker test set, SpeakerTurn achieves five-state macro F1 of 0.811 and target-transcription character error rate (CER) of 4.20%, correctly resolving all ten reference-swapped interruption pairs. Joint-format transcription and native-interface recognition nevertheless behave differently, and CoT improves validation transcription without improving turn macro F1. These results show why target attribution, semantic turn discrimination, and recognition retention require separate assessment when adapting speech language models for speaker-conditioned interaction.
#### AKDQ: Adaptive Key-Delta Quantization for Low-Bitrate RVQ-based Neural Speech Codecs
 - **Authors:** Xiao-Hang Jiang, Yang Ai, Zhen-Hua Ling, Ji Wu
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2610.39885

 - **Pdf link:** https://arxiv.org/pdf/2610.39885

 - **Abstract**
 Low-bitrate speech coding is essential for bandwidth-constrained channels and on-device applications, but conventional residual vector quantization (RVQ)-based codecs employ a fixed number of codebooks per frame regardless of content. Adjacent speech frames often exhibit strong similarity, and quantizing them independently with the same scheme results in redundant encoding, thereby wasting bitrate. Therefore, we propose Adaptive Key-Delta Quantization (AKDQ), which replaces the fixed per-frame VQ allocation of RVQ with an adaptive frame-wise key-or-delta policy. AKDQ is a plug-in strategy that requires no modification to the codec's encoder-decoder backbone. Specifically, in AKDQ, each frame is classified as either a key-frame (K-frame) quantized with all VQs, or a delta-frame (D-frame) quantized with only a few VQs on a delta correction from the previous reconstruction, thereby reducing bitrate. The K/D decision is made by a budget-aware causal scheduler through a bit-bank mechanism, keeping the system streamable with no additional lookahead. Experiments on both causal and non-causal codec families demonstrate that AKDQ effectively reduces bitrate redundancy. For instance, on DAC, using AKDQ at 1.5 kbps achieves reconstruction quality comparable to using the baseline RVQ at 2.0 kbps, saving 25% of the bitrate.
#### Discovering Interpretable Acoustic Features in Autoregressive Voice-Cloning TTS Models
 - **Authors:** Anuj Diwan, David Harwath, Eunsol Choi
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Computation and Language (cs.CL); Machine Learning (cs.LG); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2610.40505

 - **Pdf link:** https://arxiv.org/pdf/2610.40505

 - **Abstract**
 Voice-cloning TTS models synthesize an unseen speaker's voice from brief speech prompts, but it remains unclear which acoustic properties they represent and use. We introduce a feature-based interpretability framework for autoregressive voice-cloning TTS models, analyzing features from sparse autoencoders (SAEs), PCA, and random directions via sparse probing, steering, and quantitative auto-interpretability. We focus on Qwen3-TTS and extend probing and steering to $5$ additional models to test generalization across architectures. These models encode concepts like pitch, energy, noise, and accent, but sparse prediction and steering are effective only for some concepts. Strong probing does not always translate to steering or vice-versa. SAEs are useful for sparse probing of human-labelled concepts and novel-concept discovery, while PCA is competitive for dense probing and steering. Auto-interpretability finds $684$ automatically validated SAE features in Qwen3-TTS covering speaker clusters, speaking style, and recording conditions beyond known concepts. Overall, we perform a comprehensive interpretability study of autoregressive voice-cloning TTS models.
#### MixCards: Evaluating Structured Descriptions of Multi-Source Sound Scenes
 - **Authors:** Sripathi Sridhar, Yujin Kim, Giovana Morais, Liang-Yuan Wu, Mehmet Atilay Kucukoglu, Prem Seetharaman, Lloyd May, Mark Cartwright, Magdalena Fuentes
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2610.38276

 - **Pdf link:** https://arxiv.org/pdf/2610.38276

 - **Abstract**
 Audio captioning benchmarks typically evaluate a single unstructured free-text description of an audio scene, providing limited insight into how a model fails to understand polyphonic audio. A plausible caption may omit a source, or conflate multiple sources into one, or assign an attribute of one source to another, and conventional captioning evaluation cannot locate these errors. We introduce MixCardBench, a benchmark for structured captioning of polyphonic audio, which evaluates models in two stages: first, identifying and temporally localizing the sources in a scene, and second, generating a structured MixCard describing the semantic, temporal, and acoustic properties of each source. The benchmark contains 4884 ecologically plausible synthetic soundscapes with up to nine overlapping speech, music and sound effects sources built from high-quality recordings and source-level annotations, where each source has a reference MixCard. We evaluate six state-of-the-art large audio language models and find substantial limitations in current models, which struggle considerably with speech and sound effects timing and detection, with the strongest model reaching only around 60% mean card-field score. Performance varies significantly across attributes, exposing capability differences that are hidden by aggregate caption scores. We assess the quality and ecological validity of MixCardBench through human validation, distribution matching and a listening study, and show that human corrections have little effect on model rankings, and our synthesized soundscapes are closer to in-the-wild YouTube videos than source guided but otherwise random synthesis. The benchmark dataset and code are publicly available, and demo examples are at this https URL
#### Reading-Level Verification of Kanji Words in Japanese Financial Speech Synthesis
 - **Authors:** Cheng Zhang
 - **Subjects:** Subjects:
Sound (cs.SD); Computation and Language (cs.CL); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2610.39168

 - **Pdf link:** https://arxiv.org/pdf/2610.39168

 - **Abstract**
 Few-shot voice cloning makes it cheap to produce daily Japanese financial commentary with several synthetic speakers, but kanji words with more than one reading, such as the word for "market" (shijo vs. ichiba) or "overall trend" (taisei vs. oze), are still misread. A common quality gate transcribes each synthesized sentence with automatic speech recognition (ASR), converts both the input and the transcript to kana, and regenerates on mismatch. We show that this gate is unreliable in both directions: when the audio contains a misreading, the ASR language model frequently writes the expected kanji back, hiding the error, and when the audio is correct, ordinary recognition errors trigger needless regeneration. We propose a reading-level check that scores the expected reading and its confusable alternatives with a connectionist temporal classification (CTC) hiragana recognizer and flags a word when an alternative explains the audio better. On 540 utterances from three voice-cloned production voices covering 30 financial terms, with misreadings injected under controlled conditions, the transcript-based gate detected 78.9% of misreadings at a 44.4% false-alarm rate, whereas the proposed check detected 91.7% at 5.0% (AUC 0.990). On natural kanji input, it caught a front-end error that the transcript-based gate missed for one voice, and reduced regeneration requests from 38.9% to 6.1% of sentences.
#### EmpDPO: Enabling Emphasis Control in TTS via Direct Preference Optimization
 - **Authors:** Haonan Chen, Yunyun Wang, Megan Wei, Zeyu Jin
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2610.39427

 - **Pdf link:** https://arxiv.org/pdf/2610.39427

 - **Abstract**
 Word-level emphasis conveys meaning that text alone cannot express and is therefore essential for a practical text-to-speech (TTS) system. In this work, we study how to add emphasis control post hoc to a pretrained TTS model that has never been trained with emphasis annotations. We present EmpDPO, a simple, efficient, and effective two-stage recipe that adds emphasis control to a multilingual zero-shot diffusion TTS model through supervised fine-tuning followed by direct preference optimization (DPO). The training signals for both stages come from an off-the-shelf multilingual emphasis detector, so that no manual emphasis annotation or task-specific reward design is needed. We find that DPO can substantially strengthen emphasis control with only a few thousand preference pairs and training steps. Objective and subjective evaluations against strong public systems show that our model produces perceptible emphasis while preserving naturalness and overall quality.
#### Toward Complete Speaker Anonymization: a Streaming Cascaded System for Voice Privacy and PII Removal
 - **Authors:** Stanisław Kacprzak, Thomas Thebaud, Najim Dehak
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2610.39754

 - **Pdf link:** https://arxiv.org/pdf/2610.39754

 - **Abstract**
 We present a cascade streaming voice privacy system, consisting of ASR, NLP, and TTS modules. By leveraging recent improvements in all three aspects of speech and language processing, we show that such a system is capable of achieving a result comparable to state-of-the-art offline voice privacy systems, while providing controllable capabilities of content protection beyond voice. With an estimated utterance-onset end-to-end latency of 717 ms, the presented system preserves speech content with 11% WER on Librispeech, while detecting PII with a 69% SLU F1 score and achieving up to 47.66% EER against a semi-informed attacker.


by Zyzzyva0381 (Windy). 


2026-10-30
