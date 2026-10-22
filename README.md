# Showing new listings for Thursday, 22 October 2026
Auto update papers at about 2:30am UTC (10:30am Beijing time) every weekday.


阅读 `Usage.md`了解如何使用此repo实现个性化的Arxiv论文推送

See `Usage.md` for instructions on how to personalize the repo. 


Keyword list: ['text-to-speech', 'text to speech', 'tts', 'LLM-based', 'speech', 'voice']


Excluded: []


### Today: 8papers 
#### XingChen-UniFE: A Unified Generative Speech Front-End via Complementary Acoustic-Semantic Modeling
 - **Authors:** Kelan Kuang, Qihan Hu, Jian Kang, Jie Li, Yongxiang Li, Yi Liu
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2610.29164

 - **Pdf link:** https://arxiv.org/pdf/2610.29164

 - **Abstract**
 This paper presents XingChen-UniFE, a unified large audio language model (LALM) for general speech restoration (GSR) and target speaker extraction (TSE). The proposed model employs complementary semantic and acoustic representations through a dual-stream encoder, where a self-supervised learning model provides hierarchical semantic features and a learnable acoustic branch captures fine-grained structures from complex spectra. A bidirectional cross-attention module further facilitates information exchange between the two representations. The fused representation conditions the LALM for discrete speech token generation with task and speaker information. Experiments demonstrate strong performance on both GSR and TSE and robust generalization to unseen degradation conditions.
#### Optimized Compressor Parameterization for Produced Music in Hearing Aids
 - **Authors:** Robin Holland, Christoph Hoog Antink, Henning Puder
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2610.29613

 - **Pdf link:** https://arxiv.org/pdf/2610.29613

 - **Abstract**
 Hearing aid compression is typically designed for speech, yet music demands different trade-offs due to its narrower dynamic range (DR) and impulsive transients. This paper investigates whether music fitting benefits from being signal-aware, both in mapping compression demand across signal-audiogram conditions and in tuning its parameters to these combinations rather than inheriting them from speech-derived prescriptions. Compression demand is characterized across eight audiometric profiles, three residual DR severities, and three produced music signals under linear gain. Signal dynamics emerge as a first-order factor: production-compressed music fits the residual DR of the mild-to-moderate profiles evaluated, while dynamically rich recordings challenge even moderate hearing loss. Per-signal optimization of single- and multi-band wide dynamic range compression is then applied to two profiles with the highest compression demand. This optimization improves the Hearing Aid Audio Quality Index over gain-optimized NAL-NL2 baselines while also improving audibility, indicating substantial headroom for music-specific fitting.
#### NeuroCodec: EEG-Guided Latent-Domain Generative Model for High-Fidelity Target Speaker Extraction
 - **Authors:** Julian Silva, Jaliya Kodithuwakku, Navindu Sirimanna, Shaveen Herath, Wageesha N. Manamperi, Joshua P. Kulasingham
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Machine Learning (cs.LG)
 - **Arxiv link:** https://arxiv.org/abs/2610.29815

 - **Pdf link:** https://arxiv.org/pdf/2610.29815

 - **Abstract**
 Driven by recent advancements in hearing assistive devices, target speaker extraction (TSE) in multi-talker environments has drawn significant attention. To address this, hearing technologies identify attention using neural signals recorded via electroencephalography (EEG) to guide TSE, enabling neuro-steered auditory attention decoding. However, most existing techniques rely on time-frequency masking, which suffers from low fidelity in spectro-temporally overlapping regions and is often restricted to narrowband audio. In contrast, we introduce NeuroCodec, which, to the best of our knowledge, is the first generative EEG-guided TSE framework operating in the latent space of a neural audio codec. Specifically, NeuroCodec models spatial EEG dependencies via graph convolutions and fuses cross-modal EEG and speech representations through attention mechanisms and Mamba-based temporal modeling. We further introduce a six-phase training curriculum that progressively incorporates latent reconstruction, multi-scale spectral, and adversarial objectives to ensure stable, high-fidelity speech generation. Experiments on the public Cocktail Party and KUL datasets show NeuroCodec outperforms baselines, across several perceptual metrics. A listening experiment with 20 participants validated that NeuroCodec offers greater perceptual accuracy, achieving a mean score of 82.24 compared with 49.63 for M3ANet. Additionally, the proposed method is seen to enable 44.1 kHz reconstruction with spectral content extending up to 20 kHz, further supporting the perceptual results. Finally, we extend NeuroCodec with a causal streaming formulation suited for real-time processing, providing a basis for hearing assistive devices.
#### How Far Can the Gemma 4 Speech Encoder Go Without Changing Its Weights? Saudi and Gulf Arabic Speech Recognition with a Frozen and an Adapter-Augmented Encoder
 - **Authors:** Amr Abdelkhalik
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2610.28673

 - **Pdf link:** https://arxiv.org/pdf/2610.28673

 - **Abstract**
 We study the 305M-parameter audio encoder shared by Gemma 4 and EmbeddingGemma 2 (identical except for its final projection) for Saudi and Gulf Arabic speech recognition, without changing any of its weights. A frozen system trains only a 22.6M-parameter CTC reader over a mix of encoder layers; an adapted system adds 3.1M low-rank adapter parameters, a 41.5M-parameter hybrid CTC/attention reader and joint decoding with a sub-word language model. Against Cohere Transcribe Arabic (2.07B parameters), re-run on identical audio, the frozen system trails in word error rate (WER) but is statistically indistinguishable in character error rate (CER) on five of six human-labelled conditions, while running 3.4 times faster on CPU with greedy decoding. The adapted system is statistically indistinguishable in WER on a public Saudi customer-service call (19.20% against 19.27%) and on clean Emirati speech, is significantly better on telephone-coded Emirati speech, has a significantly lower CER on all six conditions of these sets, and beats a 0.6B recogniser fine-tuned on SADA in WER on all six; it still trails Cohere in WER on long-form Saudi speech. On a blind, human-corrected public call-centre sample it ranks 4 of 15 systems, behind three commercial APIs (significantly only one) and ahead of the other self-hostable systems by point estimate. An optimised joint decoder runs it at 139 times real time on one GPU. We also show that references made by correcting machine drafts favour the drafting system. Code and public-set results are released.
#### Decodability Is Not Controllability: Local and Anisotropic Speaker Control in Continuous-Latent TTS
 - **Authors:** Yuanzhong Zheng, Wei Huang, Ming Ma, Kailun Jin, Youhai Jiang
 - **Subjects:** Subjects:
Sound (cs.SD); Computation and Language (cs.CL); Machine Learning (cs.LG); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2610.29345

 - **Pdf link:** https://arxiv.org/pdf/2610.29345

 - **Abstract**
 High linear decodability does not entail causal controllability in continuous-latent text-to-speech (TTS). In VoxCPM2, position-matched probes and equal-norm interventions across 30 speakers and 60 directed dyads expose this separation: reference-audio states at the final residual acoustic language model layer (RALM L7) retain 0.970 probe accuracy, yet aligned edits yield 0/60 target-dominant dyads. Edits at early reference-audio positions produce the opposite pattern. At L0, pair-aligned updates increase target similarity by 0.326 and achieve target dominance in all 60 dyads, against at most 1/60 under any equal-norm direction or position control. Independent speaker encoders, retrieval against held-out speaker prototypes, and double-blind listening confirm that the shifted voice is heard as the intended target. Cross-architecture interventions in this http URL distinguish backbone leverage from direct acoustic control: backbone edits displace the source without target attraction, while x-vector ablation reduces source similarity from 0.849 to 0.006. These findings identify two requirements for training-free speaker control: geometric specificity and a computational route that carries the intended speaker condition into acoustic synthesis.
#### Closed-Loop Dual-Reference Guidance for Accent Control in Cross-Lingual Voice Cloning
 - **Authors:** Yuanzhong Zheng, Wei Huang, Ming Ma, Kailun Jin, Youhai Jiang
 - **Subjects:** Subjects:
Sound (cs.SD); Computation and Language (cs.CL); Machine Learning (cs.LG); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2610.29351

 - **Pdf link:** https://arxiv.org/pdf/2610.29351

 - **Abstract**
 Zero-shot cross-lingual voice cloning often carries the reference speaker's accent into the target language. We propose closed-loop dual-reference guidance, which pairs the original reference with a reusable native anchor constructed by voice conversion (VC). The references initialize separate language-model (LM) states that guide a shared acoustic trajectory; each completed patch then updates both states, keeping their generated histories aligned with the emitted speech. Anchor construction runs once per speaker, and online synthesis uses a frozen TTS model without per-utterance VC. On 20 held-out French speakers, the fixed setting $\lambda^{\!*}=0.33$ reduces residual French LID mass by 73\% versus direct cloning, at an ECAPA similarity cost of only $0.042$; relative to anchor-only cloning, it recovers $0.155$. The method achieves the lowest mean WER ($0.009$) and highest mean GOP ($4.750$) among the held-out test arms. In blinded preference tests, native-English listeners favor it over direct cloning on pronunciation naturalness, and over both anchor-only cloning and a TTS$\to$VC cascade on speaker identity. Online RTF is approximately 26\% lower than the cascade's ($0.59$ vs.\ $0.80$), and the same setting reduces Spanish LID mass by 56\% without re-tuning.
#### RubatoCodec: Budgeted Adaptive Speech Tokenization with Streaming Decoding
 - **Authors:** Chihiro Arata, Shohei Taniguchi, Gouki Minegishi, Kan Hatakeyama-Sato
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2610.29434

 - **Pdf link:** https://arxiv.org/pdf/2610.29434

 - **Abstract**
 Neural speech codecs provide discrete tokens for speech generation. Lower frame rates shorten token sequences, but each token must then carry more linguistic and acoustic information. Variable-frame-rate codecs address this trade-off by adapting token allocation to speech content, but threshold-based allocation lets content determine the token count, so the sequence length a speech generator must produce varies across utterances and datasets at a fixed setting. We present RubatoCodec, a variable-frame-rate speech codec built on X-Codec-2 with an exact utterance-level event budget and streaming decoding. Its Adaptive Firing mechanism uses changes in codec features and waveform spectra to divide speech into variable-duration events without a learned boundary predictor. Discrete codes and an explicit duration represent each event, allowing a speech generator to predict and decode one event at a time. Controlled experiments show that adaptive allocation outperforms fixed-rate coding in reconstruction fidelity and intelligibility at low event rates. A model trained on large-scale multilingual speech achieves competitive reconstruction quality across English, Japanese, and Mandarin, with the best intelligibility preservation among the compared codecs on the evaluated English and Japanese benchmarks at comparable bitrates. In a matched text-to-speech comparison, the adaptive representation yields lower WER and higher UTMOS and UTMOSv2 with comparable speaker similarity. The speech generator also learns the codec's utterance-level event budget without an explicit generation-time rate constraint. These results demonstrate that budgeted adaptive speech representations support both high-quality reconstruction and streaming text-to-speech generation. Code and weights are available at this https URL and this https URL.
#### Bio-Inspired Auditory Processing: Coupling Parallel Resonate-and-Fire Dynamics with Ternary Recurrent Networks for Speech Recognition
 - **Authors:** Pavlos Poulos, Katelyn Sadorf, Themos Stafylakis, Costas A. Anastassiou
 - **Subjects:** Subjects:
Sound (cs.SD); Artificial Intelligence (cs.AI); Neural and Evolutionary Computing (cs.NE); Audio and Speech Processing (eess.AS); Neurons and Cognition (q-bio.NC)
 - **Arxiv link:** https://arxiv.org/abs/2610.29720

 - **Pdf link:** https://arxiv.org/pdf/2610.29720

 - **Abstract**
 Spiking Neural Networks (SNNs) promise energy-efficient inference through sparse, event-driven computation, yet remain constrained on continuous, context-dependent tasks such as automatic speech recognition (ASR). A key bottleneck is the Leaky Integrate-and-Fire (LIF) neuron, whose membrane decay erases information across pauses and phonetic boundaries while it must both extract features and retain long-range context. Inspired by cortical separation of nonlinear filtering and somatic integration, we introduce the Spiking Ternary GRU (stGRU): a ternary-weight gated recurrent unit performs temporal filtering, while downstream Resonate-and-Fire (RF) neurons generate the binary spike stream driving the next block's event pathway. This makes binary event transmission a native training constraint rather than a separate deployment-time activation quantization step; together with ternary weights, learned projections avoid dense floating-point matrix MACs. Trained end-to-end on LibriSpeech-960, Small, Medium, and Large stGRUs reach 8.09%, 7.04%, and $6.44 \pm 0.17\%$ WER on test-clean without an external language model. To our knowledge, these are the first matrix-MAC-free spiking networks without a learned ANN sequence decoder to achieve single-digit WER on LibriSpeech. The architecture also achieves 10.88% test CER on Aishell-1 (9.28% on dev) when trained from scratch. Internally, trained networks develop a depth-graded hierarchy: shallow layers act as fast acoustic filters, while deep layers form slower semantic integrators whose latent representations segregate into part-of-speech manifolds and word-level spike events. Deep-layer firing rates follow lognormal distributions with a sparsity gradient, mirroring hierarchical processing observed in human cortex in vivo. Our biologically grounded architecture combines competitive ASR with favorable modeled compute and memory cost.


by Zyzzyva0381 (Windy). 


2026-10-22
