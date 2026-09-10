# Showing new listings for Thursday, 10 September 2026
Auto update papers at about 2:30am UTC (10:30am Beijing time) every weekday.


阅读 `Usage.md`了解如何使用此repo实现个性化的Arxiv论文推送

See `Usage.md` for instructions on how to personalize the repo. 


Keyword list: ['text-to-speech', 'text to speech', 'tts', 'LLM-based', 'speech', 'voice']


Excluded: []


### Today: 15papers 
#### Language Orthogonalization of Self-Supervised Speech Representations for Cross-lingual Parkinson's Detection
 - **Authors:** Minu Kim, Eunjung Yeo, Kwanghee Choi, June-Woo Kim
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2609.09499

 - **Pdf link:** https://arxiv.org/pdf/2609.09499

 - **Abstract**
 Self-supervised speech models (S3Ms) provide powerful representations for Parkinson's disease (PD) detection, making cross-lingual transfer attractive for languages lacking labeled patient speech. However, these representations also encode language identity, which can confound this transfer: without target-language PD speech, classifiers may separate languages rather than pathology, yielding high specificity but low sensitivity on target patients. We propose \emph{language orthogonalization}, a closed-form ridge residualization of S3M features against external VoxLingua107 language embeddings, fitted using only healthy-control (HC) speech. By removing language-predictable components while retaining pathology-related variation, it produces a less language-dependent geometry in which HC representations concentrate while PD representations disperse. Across five S3M backbones, three speech tasks, and three target languages, our method consistently improves cross-lingual PD-detection performance while correcting the high-specificity/low-sensitivity failure.
#### UniStream: Multi-Expert Residual Vector Quantization for 48 kHz Causal Streaming Audio Coding
 - **Authors:** Mingyu Zhao, Zhiyong Wu
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2609.09866

 - **Pdf link:** https://arxiv.org/pdf/2609.09866

 - **Abstract**
 We present UniStream, a fully causal 48 kHz neural audio codec for streaming speech, music, and environmental sounds. At its core is Multi-Expert Residual Vector Quantization (ME-RVQ), which replaces the single shared codebook in each residual quantization layer with four expert codebooks controlled by a deterministic Top-K router. Because routing decisions are derived solely from previously decoded quantized states, the decoder can reproduce the selected experts without transmitting expert identifiers, thereby expanding quantization capacity while adding 5.5M parameters. We further introduce an auxiliary Optimal Transport Conditional Flow Matching (OT-CFM) objective to regularize the quantized latent space during training. The flow module is removed entirely at inference and therefore incurs no runtime overhead. UniStream supports a 12 kbps Top-1 mode and a 22.5 kbps Top-2 mode within a causal 48 kHz encoder-decoder framework, while achieving real-time GPU inference. To complement narrow-band speech metrics, we report 48 kHz ViSQOL audio mode, ViSQOL speech mode, standard VGGish-FAD, DNSMOS P.835, and higher-rate reference comparisons with Opus and EnCodec. At 12 kbps, UniStream-Top1 achieves PESQ and UTMOS scores comparable to EnCodec while reducing speech Mel-D from 13.07 to 8.21. At 22.5 kbps, UniStream-Top2 achieves a ViSQOL speech-mode score of 4.67 and an environmental audio-mode score of 3.96, exceeding all evaluated systems operating at 12 kbps or below in the latter setting. It also comes within 0.03 MOS-LQO of Opus at 24 kbps on speech in ViSQOL audio mode. Ablation studies confirm that ME-RVQ is the primary source of quality improvement, whereas OT-CFM provides perceptual gains on speech with a mild trade-off in spectral distortion.
#### SphereVAE: Hyperspherical Latent Autoencoders for Robust Autoregressive Speech Representation Modeling
 - **Authors:** Haoyu Zhang, Jingbin Hu, Hanke Xie, Qirui Zhan, Wenhao Li, Ziyu Zhang, Xiaming Ren, Yue Li, Xunyu Zhu, Zhipeng Chen, Lei Xie
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2609.09903

 - **Pdf link:** https://arxiv.org/pdf/2609.09903

 - **Abstract**
 With the rapid development of speech generation technology, discrete codec representations have been widely used because they provide a stable prediction paradigm. In expressive speech generation, however, the quantization bottleneck of discrete codecs results in information gaps in fine-grained prosody, timbre, pronunciation, and frame-to-frame continuity. Continuous representations (e.g., VAE latents), by eliminating this constraint, have emerged as a more effective alternative for autoregressive modeling. Yet when continuous representations are used as autoregressive prediction targets, prediction errors can accumulate along the generation chain, causing latent drift and degrading long-form stability. To mitigate this problem, we propose SphereVAE, which constrains the VAE latent space to the unit hypersphere. SphereVAE defines a Power Spherical posterior on the hypersphere and regularizes the latent distribution toward a uniform prior, so that information is encoded mainly by directional variation, providing a bounded geometric target for autoregressive prediction and reducing the risk of norm drift. SphereVAE underperforms the standard VAE on reconstruction metrics due to reduced latent freedom. However, when integrated into VoxCPM for zero-shot TTS and long-text generation, it yields lower content error rates with comparable speaker similarity, and shows more stable long-range speaker consistency. These results indicate that an appropriate latent geometric constraint can effectively mitigate autoregressive error accumulation and drift in speech generation.
#### Source-Adaptive Data Curation for Bilingual NVV-Aware ASR
 - **Authors:** Yuang Cao, Qirui Zhan, Jingbin Hu, Ziyu Zhang, Yunxiang Chen, Houdun Liu, Shuo Feng, Bengu Wu, Lei Xie, Liumeng Xue
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2609.09929

 - **Pdf link:** https://arxiv.org/pdf/2609.09929

 - **Abstract**
 Nonverbal vocalizations (NVVs), such as laughter, sighs, breaths, and coughs, convey affective and interactional information that conventional automatic speech recognition (ASR) systems often discard. We present a bilingual Mandarin-English system for Track 1 of the NVVSpeech Challenge at ISCSLP 2026, which requires joint transcription of lexical content and 16 NVV categories at their transcript-relative positions. Our NVV-Aware Whisper adapts Whisper-medium through checkpoint-compatible vocabulary remapping, enabling lexical tokens and inline NVV tags to be decoded within a unified autoregressive sequence without expanding the vocabulary. To provide reliable and diverse supervision, we further introduce a source-adaptive data curation strategy that refines public NVV corpora through acoustic augmentation and multimodal LLM filtering, while mining spontaneous NVVs from in-the-wild media through automated preprocessing and annotation. Under the official bilingual evaluation protocol, the proposed system improves final score from 33.32 to 53.61, with ablations confirming the complementary benefits of the proposed data-curation components.
#### NVV-Locator: From Transcript Tags to Acoustic Boundaries for Fine-Grained Nonverbal Vocalization Grounding
 - **Authors:** Yuang Cao, Bingshen Mu, Zhennan Lin, Guojian Li, Haoyue Zhan, Jie Liu, Chuan Xie, Qiang Zhang, Liumeng Xue, Lei Xie
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2609.09940

 - **Pdf link:** https://arxiv.org/pdf/2609.09940

 - **Abstract**
 Human speech includes nonverbal vocalizations (NVVs), such as laughter, sighs, breaths, and coughs, which convey affective and interactional information. Existing approaches typically represent NVVs as transcript-level tags, providing limited supervision for their waveform-time boundaries. We present NVV-Locator for fine-grained NVV temporal grounding. We first unify 26 NVV categories across public resources and construct large-scale timestamp-supervised training data through dual-LLM verification, transcript-guided forced alignment, and energy-based boundary refinement. We further introduce NVV-TimeBench, an expert-refined benchmark with 667 utterances and 1,094 events. NVV-Locator uses a non-autoregressive slot-filling architecture to jointly predict lexical timestamps, NVV categories, and event boundaries. On NVV-TimeBench, it achieves 71.0% Micro F1, 70.2% Macro F1, 80.4% Macro mIoU, and 59.6 ms Macro mMAE, outperforming the evaluated large audio model counterparts. Evaluation on an external corpus further demonstrates the cross-corpus generalization of NVV-Locator.
#### SpeechAnnotator: A Context-Aware Multi-Agent Framework and Benchmark for Multidimensional Speech Annotation
 - **Authors:** Qirui Zhan, Shuiyuan Wang, Jingbin Hu, Haoyu Zhang, Xiaming Ren, Jinrui Liang, Chaoren Yu, Bengu Wu, Yunxiang Chen, Houdun Liu, Su Feng, Liumeng Xue, Lei Xie
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2609.09947

 - **Pdf link:** https://arxiv.org/pdf/2609.09947

 - **Abstract**
 Recent controllable speech generation requires training data with fine-grained annotations of speaker traits, prosody, emotion, paralinguistic cues, acoustic scenes, and context. Existing workflows often rely on manual correction, paid hosted multimodal services, or fixed processing chains, which limits large-scale data processing through annotation cost, external-service dependence, or weak cross-stage recovery. We introduce SpeechAnnotator, a locally deployable, context-aware multi-agent framework built entirely from open-source models and tools. Supporting frontend modules first obtain speaker-aware segments and final segment transcripts, while prior evidence extractors attach heterogeneous segment-level cues. Three specialist agents then collaborate through shared state: the Planning Agent converts local audio evidence, speaker history, neighboring segments, and recording-level context into field-specific contracts; the Labeling Agent performs contract-guided multimodal prediction for directly observable attributes; and the Review Agent runs a bounded review loop that checks evidence support and cross-segment consistency, triggering relabeling only for unsupported or inconsistent fields. To address the fragmentation of existing evaluation resources across isolated tasks and narrow-domain test sets, we introduce SpeechAnnotator-Bench (SA-Bench), containing 8.87 hours of human-annotated audio across nine source formats, together with SpeechAnnotator-Eval (SA-Eval), which separates Timeline-Eval for speaker-aware timeline recovery, Closed-Eval for finite-set attributes, and Open-Eval for open-ended attributes. Experiments and ablations show that SpeechAnnotator provides a locally deployable alternative to commercial audio-capable systems, while the bounded review loop improves multidimensional annotation through evidence- and context-aware field-level recovery.
#### Over-Tightening-Aware Pseudo-Labeling for Tight-Boundary Speaker Diarization
 - **Authors:** Shota Horiguchi, Takanori Ashihara, Marc Delcroix, Naohiro Tawara, Alexis Plaquet
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2609.09965

 - **Pdf link:** https://arxiv.org/pdf/2609.09965

 - **Abstract**
 Training speaker diarization models on loose labels, such as speech segments with padded boundaries or filled pauses, often results in similarly loose model outputs. To obtain tighter boundaries, pseudo-labeling based on the averaged outputs of causal and anticausal models has been proposed. However, since the pseudo-labels are estimation-based, they can suffer from over-tightening, which increases missed detections that can propagate as unrecoverable errors to downstream tasks. This paper carefully analyzes the causes of over-tightening and proposes three approaches to address them: (i) removing pause filling rather than padding, (ii) introducing a burn-in phase to mitigate missed detections near the beginning of causal and anticausal predictions, and (iii) making pseudo-label-based co-training aware of the non-causal model used for final inference. Experimental results show that the proposed method reduces missed detections caused by over-tightening and improves both diarization accuracy and downstream multi-talker ASR performance.
#### SCNet: Enhancing GAN-based Speech Generation with Subband Condition Network and Magnitude-aware Phase Loss
 - **Authors:** Nan Xu, Mingxue Yang
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2609.10025

 - **Pdf link:** https://arxiv.org/pdf/2609.10025

 - **Abstract**
 Recent speech generation has been predominantly driven by GAN-based networks aimed at high-quality waveform synthesis from mel-spectrograms. However, these methods often operate as black-box models, leading to the loss of inherent spectral information. In this work, we propose SCNet, a GAN-based vocoder augmented with a Subband Condition Network to address this issue. Specifically, SCNet leverages a subband signal predicted by a lightweight condition network as prior knowledge. This subband signal is then transformed via STFT to obtain Fourier coefficients, which are integrated into the backbone for the enhanced reconstruction. Additionally, to mitigate the phase wrapping, we introduce a magnitude-aware phase loss that computes instantaneous phase errors weighted by the corresponding magnitude, emphasizing regions with higher energy. Experimental results demonstrate that SCNet achieves superior performance in both objective and subjective evaluations for high-quality speech generation.
#### Pushing the Boundaries of Streaming Multi-Speaker ASR: A Systematic Study of Architectural Trade-offs
 - **Authors:** Taejin Park, Ivan Medennikov, Kunal Dhawan, Weiqing Wang, Jagadeesh Balam, Boris Ginsburg
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2609.10265

 - **Pdf link:** https://arxiv.org/pdf/2609.10265

 - **Abstract**
 Streaming multi-speaker ASR is a challenging task that must balance accuracy, latency, and efficiency while handling overlapping speech and maintaining coherent long-context modeling over extended conversations in an online fashion. We present a unified framework that categorizes streaming multi-speaker ASR into four architectural strategies based on how diarization and ASR are integrated. Using a shared pair of open-source streaming ASR and diarization models as a common foundation, we derive four multi-speaker ASR systems that differ in whether they employ multiple model instances, fine-tuning, or both. We evaluate these systems across multi-speaker accuracy, single-speaker accuracy degradation, memory footprint, and training complexity. Through this systematic architectural analysis, we clarify the design space for streaming multi-speaker ASR and provide practical guidance for selecting the most suitable approach under diverse deployment constraints.
#### AVSRBench: A Multi-Condition AVSR Benchmark
 - **Authors:** Rishabh Jain, Naomi Harte
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Computer Vision and Pattern Recognition (cs.CV); Multimedia (cs.MM)
 - **Arxiv link:** https://arxiv.org/abs/2609.10366

 - **Pdf link:** https://arxiv.org/pdf/2609.10366

 - **Abstract**
 While AVSR has achieved sub-1% word error rates on the standard LRS3 benchmark, its reliance on broadcast speech obscures whether this reflects true generalization or just domain adaptation. To investigate this gap, we evaluate three AVSR architectures across six conditions: controlled broadcast speech, fixed-grammar utterances, hyper-articulated Lombard speech, read speech from professional lipspeakers and non-professional speakers, and spontaneous multi-party video conversations. We find that visual-only performance deteriorates rapidly beyond broadcast domains, and audio-video fusion mainly benefits Lombard speech environments. Visual understanding degrades sharply at 90° profile views, with multimodal systems relying largely on acoustic fallback. Additionally, speaker articulation proves more critical than minor camera shifts, and LLM-based architectures suffer from poor out-of-domain generalization. Our work highlights a significant generalization gap in current AVSR research. To address this, we also introduce RoomReader-AV as a new benchmark for AVSR and release a unified data preprocessing pipeline to make comprehensive multi-condition evaluation accessible.
#### Teacher-Free Self-Distilled Consistency Trajectory Learning for Fast Speech Enhancement
 - **Authors:** Shuubham Ojha, Carol Espy-Wilson
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2609.10392

 - **Pdf link:** https://arxiv.org/pdf/2609.10392

 - **Abstract**
 Consistency trajectory models offer a route to fast, high-quality speech enhancement, collapsing the many reverse steps of diffusion-based enhancers into a handful. When instantiated on a Schrödinger bridge (SB), which pins the generative process to fixed clean and noisy endpoints, existing consistency-trajectory enhancers (SBCTMs) still require a pretrained teacher to supply trajectory supervision, which raises training cost and ties the final quality to that of the teacher. We propose a teacher-free, self-distilled consistency-trajectory framework that removes the external teacher: trajectory targets are generated by an exponential-moving-average (EMA) copy of the student, and the model is trained with a three-stage curriculum of $\x_0$ prediction, a self-distilled shortcut objective, and perceptual fine-tuning with a multi-resolution short-time Fourier transform (MR-STFT) loss. Using the same NCSN++ backbone as SBCTM, our model attains a wide-band PESQ of $3.01$, ESTOI $0.87$, and SI-SDR $19.07$\,dB on VoiceBank+DEMAND without a teacher. Varying step count and inference schedule we find that a geometric schedule at low reverse step count maximizes perceptual quality, while a higher-step uniform schedule favors signal fidelity, with the geometric advantage narrowing with reverse step count.
#### Candor-LR: A Dyadic Conversational Dataset for Audio-Visual Speech Recognition
 - **Authors:** Rishabh Jain, Aristeidis Papadopoulos, Zhaofeng Lin, Naomi Harte
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Computer Vision and Pattern Recognition (cs.CV); Multimedia (cs.MM)
 - **Arxiv link:** https://arxiv.org/abs/2609.10394

 - **Pdf link:** https://arxiv.org/pdf/2609.10394

 - **Abstract**
 Current audio-visual speech recognition (AVSR) benchmarks, like LRS3, rely heavily on clean, scripted and rehearsed speech. They fail to reflect the complexity of natural conversation, which involves overlapping speech, spontaneous turn-taking, unscripted vocabulary and variable acoustic conditions. To shift the field toward realistic dialogue, we introduce Candor-LR, a conversational benchmark derived from the CANDOR corpus of 1,656 natural dyadic videoconferences. Our custom data preparation pipeline yields 713.5, 10.1, and 60.1 hours of training, validation, and test data, respectively. Evaluating pretrained AVSR models on Candor-LR reveals that audio-only accuracy drops sharply compared to LRS3, but visual cues compensate effectively, driving much larger performance gains on Candor-LR than on LRS3. Furthermore, training on this corpus significantly improves cross-domain robustness under both clean and noisy conditions, as its realistic conversational data captures broader audio-video features. We open-source our pipeline to ensure reproducibility, establishing Candor-LR as a challenging benchmark for conversational AVSR.
#### Phoneme-Aware Pronunciation Representations for L2-English L1-Background Accent Identification
 - **Authors:** Yangyang Qu, Massimiliano Todisco, Nicholas Evans
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2609.10466

 - **Pdf link:** https://arxiv.org/pdf/2609.10466

 - **Abstract**
 We study speaker-disjoint accent identification for L2 English, where the goal is to predict a speaker's first-language (L1) background from English pronunciation. Most existing systems classify accents using a single utterance-level representation, but such global representations can obscure pronunciation cues that depend on specific English phonemes. We propose a transcript-assisted model that makes phoneme information explicit during accent identification. Instead of representing an utterance only as a global speech embedding, we represent it as a sequence of pronunciation units, each combining acoustic evidence from a spoken segment with the aligned English phoneme for that segment. A frozen speech encoder provides the acoustic features, while the transcript is used only to obtain phoneme-level forced alignments. No word-level or sentence-level text representation is passed to the accent classifier. Under a four-fold speaker-disjoint protocol on L2-ARCTIC, our model achieves 81.41% accuracy and 81.21% macro-F1, the highest mean performance among the evaluated systems. Diagnostic ablations support the importance of phoneme-aligned token construction, while a Whisper-based ablation shows an additional gain from phoneme information.
#### StreamAlign: Streaming Text-Aligned Speech Tokenization
 - **Authors:** Kang-wook Kim, Jinyoung Park, Jinsoo Kim, Sehun Lee, Sang Hoon Woo, Gunhee Kim
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2609.09719

 - **Pdf link:** https://arxiv.org/pdf/2609.09719

 - **Abstract**
 Text-aligned speech tokenization methods have emerged to better align speech tokens with LLM token spaces, enabling more effective utilization of pretrained LLMs. However, they rely on offline automatic speech recognition (ASR), leading to two key limitations: (i) the need for complete utterances before tokenization, precluding real-time streaming, and (ii) vocabulary mismatch between ASR and LLMs, which reduces acoustic granularity from the subword to the word level. We introduce StreamAlign, a text-aligned speech tokenization framework that enables streaming tokenization for real-time speech-text joint modeling. StreamAlign performs online speech-text alignment by combining character-level RNN-Transducer alignment with word-level ASR guidance, mitigating ASR-LLM vocabulary mismatch while preserving recognition accuracy. A proactive word boundary classifier anticipates word completion at chunk boundaries, reducing tokenization latency from 560 ms to 270 ms. On LibriSpeech, StreamAlign achieves the lowest WER and highest UTMOS among evaluated tokenizers. Furthermore, StreamAlign-SLM, a spoken language model trained on StreamAlign units, outperforms other end-to-end spoken language models in speech continuation while achieving the strongest overall consistency on SALMon and spoken StoryCloze.
#### Orukeet: Multilingual ASR with Frozen Gabor Kernels
 - **Authors:** Nathan Roll (1 and 2), Irene Yi (1 and 2), Büşra Marşan (1 and 2), Vianney Grenez (1), Gabriel Stein (4), Momcilo Mrkaic (5), Pavle Padjin (5), Vladimir Zeljkovic (5), Calbert Graham (1 and 3) ((1) Oruk AI, (2) Stanford University, (3) University of Cambridge, (4) OpenWhispr, (5) Hoid)
 - **Subjects:** Subjects:
Sound (cs.SD); Machine Learning (cs.LG); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2609.10054

 - **Pdf link:** https://arxiv.org/pdf/2609.10054

 - **Abstract**
 Orukeet replaces half of an adapted Parakeet encoder's temporal filters with 12,288 fitted Gabor kernels, freezes these replacements, and trains the remaining parameters on multilingual and multi-accent data. Final adaptation and checkpoint selection use LibriSpeech test-other. Across 20,146 FLEURS recordings in 25 languages, pooled word error rate (WER) falls from Parakeet's 11.01% to Orukeet's 9.85%, a 10.6% relative reduction. Orukeet has lower WER on 23 of the 25 languages. Orukeet outperforms Parakeet on 61 out of 74 tested splits, including LibriSpeech test-clean (1.46% vs. 1.53% WER), test-other (2.86% vs. 3.14%), and FLEURS English (3.82% vs. 4.28%). All comparisons decode the same audio with matched NeMo settings. The fitted kernels are stored as ordinary convolution weights, retaining Parakeet's architecture and inference operators.


by Zyzzyva0381 (Windy). 


2026-09-10
