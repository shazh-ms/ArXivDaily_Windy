# Showing new listings for Monday, 11 May 2026
Auto update papers at about 2:30am UTC (10:30am Beijing time) every weekday.


阅读 `Usage.md`了解如何使用此repo实现个性化的Arxiv论文推送

See `Usage.md` for instructions on how to personalize the repo. 


Keyword list: ['text-to-speech', 'text to speech', 'tts', 'LLM-based', 'speech', 'voice']


Excluded: []


### Today: 4papers 
#### Evaluating voice anonymisation using similarity rank disclosure
 - **Authors:** Shilpa Chandra, Matteo Pettenò, Nicholas Evans, Michele Panariello, Massimiliano Todisco, Tom Bäckström, Dorothea Kolossa, Rainer Martin, Themos Stafylakis, Nicolas Gengembre
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2605.07291

 - **Pdf link:** https://arxiv.org/pdf/2605.07291

 - **Abstract**
 The evaluation of voice anonymisation remains challenging. Current practice relies on automatic speaker verification metrics such as the equal error rate (EER). Performance estimates dependent on the classifier and operating point provide an incomplete or even misleading characterisation of privacy risk. We investigate the use of similarity rank disclosure (SRD), an information-theoretic metric, which operates on feature representations rather than classifier decisions, providing a threshold-independent assessment of privacy and analysis of both average and worst-case disclosure. We report its application to speaker embeddings, fundamental frequency, and phone embeddings using 2024 VoicePrivacy Challenge systems. The SRD reveals privacy leaks and system-specific weaknesses missed by EER-based evaluation. Findings highlight the merit of representation-level metrics and demonstrate the potential of SRD as a flexible and interpretable tool for the evaluation of voice anonymisation.
#### MIST: Multimodal Interactive Speech-based Tool-calling Conversational Assistants for Smart Homes
 - **Authors:** Maximillian Chen, Xuanming Zhang, Michael Peng, Zhou Yu, Alexandros Papangelis, Yohan Jo
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Artificial Intelligence (cs.AI); Human-Computer Interaction (cs.HC); Multimedia (cs.MM); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2605.06897

 - **Pdf link:** https://arxiv.org/pdf/2605.06897

 - **Abstract**
 The rise of Internet of Things (IoT) devices in the physical world necessitates voice-based interfaces capable of handling complex user experiences. While modern Large Language Models (LLMs) already demonstrate strong tool-usage capabilities, modeling real-world IoT devices presents a difficult, understudied challenge which combines modeling spatiotemporal constraints with speech inputs, dynamic state tracking, and mixed-initiative interaction patterns. We introduce MIST (the Multimodal Interactive Speech-based Tool-calling Dataset), a synthetic multi-turn, voice-driven code generation task that operates over IoT devices. We find that there is a significant gap between open- and closed-weight multimodal LLMs on MIST, and that even frontier closed-weight LLMs have substantial headroom. We release MIST and an extensible data generation framework to build related datasets in order to facilitate research on mixed-initiative voice assistants which reason about physical world constraints.
#### Asymmetric Phase Coding Audio Watermarking
 - **Authors:** Guang Yang, Amir Ghasemian, Ninareh Mehrabi, Homa Hosseinmardi
 - **Subjects:** Subjects:
Cryptography and Security (cs.CR); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2605.07241

 - **Pdf link:** https://arxiv.org/pdf/2605.07241

 - **Abstract**
 The proliferation of deepfake audio challenges voice-based authentication systems; passive forensic detectors are sensitive to evolving generative models and to real-world channel distortions. We propose Asymmetric Phase Coding (APC), a training-free cryptographic signing layer for audio, designed as a compact and auditable provenance primitive that can stand alone or be stacked with learned watermarks. APC combines Ed25519 digital signatures (EdDSA, FIPS 186-5; 64-byte signatures) with Reed-Solomon error correction, pseudo-random STFT phase-bin selection, and a redundant quantization-index-modulation (QIM) code on log-magnitude differences of adjacent bin pairs, yielding a compact, non-repudiable, blind-extractable watermark. We evaluate APC on 1,000 LibriSpeech test-clean clips (10 s each, 44.1 kHz) under eight attack configurations -- identity, 10% end-cropping, 20% end-cropping, 8 kHz low-pass, 16 kHz round-trip resampling, FLAC re-encoding, MP3 at 128 kbps, and OGG-Vorbis at 128 kbps -- and achieve cryptographic verification rates between 97.5% and 98.3% on every condition at mean PESQ=3.02 and tens-of-milliseconds CPU latency. We explicitly compare APC against recent neural baselines (AudioSeal, WavMark, SilentCipher), detail the threat model (forgery resistance vs. erasure), characterize the dataset, define all metrics, quantify an adaptive white-box erasure attack, and release code, keys, and metadata for reproducibility.
#### Zero-Shot Imagined Speech Decoding via Imagined-to-Listened MEG Mapping
 - **Authors:** Maryam Maghsoudi, Shihab Shamma
 - **Subjects:** Subjects:
Machine Learning (cs.LG); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2605.08075

 - **Pdf link:** https://arxiv.org/pdf/2605.08075

 - **Abstract**
 Decoding imagined speech from non-invasive brain recordings is challenging because imagined datasets are scarce and difficult to align temporally across subjects and sessions In this work, we propose a new approach to the decoding of imagined speech that leverages the richer and more reliably labeled recordings during listening to speech. We collected paired listened and imagined MEG recordings to rhythmic melodic and spoken stimuli from trained musicians. Using trained musicians helped improve temporal alignment across conditions. We then developed a three-stage decoding pipeline that revealed consistent and meaningful relationships between neural activity evoked by imagining and listening to the same stimuli. First, we trained six linear and neural models to map imagined MEG responses to listened responses. We evaluated these models against a null baseline from unseen subjects to validate that the predicted-listening responses preserve stimulus-specific information. In the second stage, we trained a contrastive word decoder exclusively on the listened MEG responses, and evaluated it using four embedding strategies including semantic, acoustic, and phonetic representations. In the third stage, we process the imagined MEG responses from held-out subjects through the mapping pipeline to compute the corresponding listening responses that are then decoded by the listened decoder. Using rank-based analysis, we show that the imagined words are decodable significantly above chance. We shall report here the results of a proof-of-concept implementation to decode imagined speech, where all evaluations are performed on held-out subjects. We also demonstrate that performance improves with training data size, suggesting that this approach is scalable and can directly be made applicable to realistic brain-computer interface scenarios.


by Zyzzyva0381 (Windy). 


2026-05-11
