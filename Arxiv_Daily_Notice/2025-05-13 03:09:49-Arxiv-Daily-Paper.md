# Showing new listings for Tuesday, 13 May 2025
Auto update papers at about 2:30am UTC (10:30am Beijing time) every weekday.


阅读 `Usage.md`了解如何使用此repo实现个性化的Arxiv论文推送

See `Usage.md` for instructions on how to personalize the repo. 


Keyword list: ['text-to-speech', 'text to speech', 'tts', 'LLM-based', 'speech', 'voice']


Excluded: []


### Today: 6papers 
#### RADE: A Neural Codec for Transmitting Speech over HF Radio Channels
 - **Authors:** David Rowe, Jean-Marc Valin
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2505.06671

 - **Pdf link:** https://arxiv.org/pdf/2505.06671

 - **Abstract**
 Speech compression is commonly used to send voice over radio channels in applications such as mobile telephony and two-way push-to-talk (PTT) radio. In classical systems, the speech codec is combined with forward error correction, modulation and radio hardware. In this paper we describe an autoencoder that replaces many of the traditional signal processing elements with a neural network. The encoder takes a vocoder feature set (short term spectrum, pitch, voicing), and produces discrete time, but continuously valued quadrature amplitude modulation (QAM) symbols. We use orthogonal frequency domain multiplexing (OFDM) to send and receive these symbols over high frequency (HF) radio channels. The decoder converts received QAM symbols to vocoder features suitable for synthesis. The autoencoder has been trained to be robust to additive Gaussian noise and multipath channel impairments while simultaneously maintaining a Peak To Average Power Ratio (PAPR) of less than 1~dB. Over simulated and real world HF radio channels we have achieved output speech intelligibility that clearly surpasses existing analog and digital radio systems over a range of SNRs.
#### TACOS: Temporally-aligned Audio CaptiOnS for Language-Audio Pretraining
 - **Authors:** Paul Primus, Florian Schmid, Gerhard Widmer
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Machine Learning (cs.LG); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2505.07609

 - **Pdf link:** https://arxiv.org/pdf/2505.07609

 - **Abstract**
 Learning to associate audio with textual descriptions is valuable for a range of tasks, including pretraining, zero-shot classification, audio retrieval, audio captioning, and text-conditioned audio generation. Existing contrastive language-audio pretrained models are typically trained using global, clip-level descriptions, which provide only weak temporal supervision. We hypothesize that CLAP-like language-audio models - particularly, if they are expected to produce frame-level embeddings - can benefit from a stronger temporal supervision. To confirm our hypothesis, we curate a novel dataset of approximately 12,000 audio recordings from Freesound, each annotated with single-sentence free-text descriptions linked to a specific temporal segment in an audio recording. We use large language models to clean these annotations by removing references to non-audible events, transcribed speech, typos, and annotator language bias. We further propose a frame-wise contrastive training strategy that learns to align text descriptions with temporal regions in an audio recording and demonstrate that our model has better temporal text-audio alignment abilities compared to models trained only on global captions when evaluated on the AudioSet Strong benchmark. The dataset and our source code are available on Zenodo and GitHub, respectively.
#### TS-SUPERB: A Target Speech Processing Benchmark for Speech Self-Supervised Learning Models
 - **Authors:** Junyi Peng, Takanori Ashihara, Marc Delcroix, Tsubasa Ochiai, Oldrich Plchot, Shoko Araki, Jan Černocký
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2505.06660

 - **Pdf link:** https://arxiv.org/pdf/2505.06660

 - **Abstract**
 Self-supervised learning (SSL) models have significantly advanced speech processing tasks, and several benchmarks have been proposed to validate their effectiveness. However, previous benchmarks have primarily focused on single-speaker scenarios, with less exploration of target-speaker tasks in noisy, multi-talker conditions -- a more challenging yet practical case. In this paper, we introduce the Target-Speaker Speech Processing Universal Performance Benchmark (TS-SUPERB), which includes four widely recognized target-speaker processing tasks that require identifying the target speaker and extracting information from the speech mixture. In our benchmark, the speaker embedding extracted from enrollment speech is used as a clue to condition downstream models. The benchmark result reveals the importance of evaluating SSL models in target speaker scenarios, demonstrating that performance cannot be easily inferred from related single-speaker tasks. Moreover, by using a unified SSL-based target speech encoder, consisting of a speaker encoder and an extractor module, we also investigate joint optimization across TS tasks to leverage mutual information and demonstrate its effectiveness.
#### On the Cost and Benefits of Training Context with Utterance or Full Conversation Training: A Comparative Stud
 - **Authors:** Hyouin Liu, Zhikuan Zhang
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2505.07202

 - **Pdf link:** https://arxiv.org/pdf/2505.07202

 - **Abstract**
 Modern TTS systems designed for conversations achieve high-quality utterances but often remain inaccessible publicly. Are existing open-source architectures inadequate, or are current training techniques insufficient? This paper investigates prominent models and their underlying behaviors regarding conversational context. Using 20 GPU-hours on an NVIDIA H100, we empirically examine two approaches: context-based utterance-level training versus full conversation training. Results demonstrate that context-based utterance training achieves superior MOS scores (4.3/5.0 vs 3.7/5.0) and reduces training time by 37%, while full conversation approaches suffer from speaker similarity hallucination issues. These findings provide practical guidelines for conversational TTS development, favoring utterance-level training with contextual conditioning for both resource efficiency and output quality.
#### Lightweight End-to-end Text-to-speech Synthesis for low resource on-device applications
 - **Authors:** Biel Tura Vecino, Adam Gabryś, Daniel Mątwicki, Andrzej Pomirski, Tom Iddon, Marius Cotescu, Jaime Lorenzo-Trueba
 - **Subjects:** Subjects:
Sound (cs.SD); Artificial Intelligence (cs.AI); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2505.07701

 - **Pdf link:** https://arxiv.org/pdf/2505.07701

 - **Abstract**
 Recent works have shown that modelling raw waveform directly from text in an end-to-end (E2E) fashion produces more natural-sounding speech than traditional neural text-to-speech (TTS) systems based on a cascade or two-stage approach. However, current E2E state-of-the-art models are computationally complex and memory-consuming, making them unsuitable for real-time offline on-device applications in low-resource scenarios. To address this issue, we propose a Lightweight E2E-TTS (LE2E) model that generates high-quality speech requiring minimal computational resources. We evaluate the proposed model on the LJSpeech dataset and show that it achieves state-of-the-art performance while being up to $90\%$ smaller in terms of model parameters and $10\times$ faster in real-time-factor. Furthermore, we demonstrate that the proposed E2E training paradigm achieves better quality compared to an equivalent architecture trained in a two-stage approach. Our results suggest that LE2E is a promising approach for developing real-time, high quality, low-resource TTS applications for on-device applications.
#### Spoken Language Understanding on Unseen Tasks With In-Context Learning
 - **Authors:** Neeraj Agrawal, Sriram Ganapathy
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Machine Learning (cs.LG); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2505.07731

 - **Pdf link:** https://arxiv.org/pdf/2505.07731

 - **Abstract**
 Spoken language understanding (SLU) tasks involve diverse skills that probe the information extraction, classification and/or generation capabilities of models. In this setting, task-specific training data may not always be available. While traditional task-specific SLU models are unable to cater to such requirements, the speech-text large language models (LLMs) offer a promising alternative with emergent abilities. However, out of-the-box, our evaluations indicate that the zero/few-shot performance of prominent open-source speech-text LLMs on SLU tasks are not up to the mark. In this paper, we introduce a novel approach to robust task-agnostic fine-tuning using randomized class labels. With this proposed fine-tuning, we illustrate that the performance of the speech-text LLMs on an unseen task is significantly improved over standard approaches. Critically, the proposed approach avoids the requirement of task-specific data annotations for enabling new tasks in speech-text LLMs.


by Zyzzyva0381 (Windy). 


2025-05-13
