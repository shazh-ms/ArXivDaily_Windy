# Showing new listings for Wednesday, 28 October 2026
Auto update papers at about 2:30am UTC (10:30am Beijing time) every weekday.


阅读 `Usage.md`了解如何使用此repo实现个性化的Arxiv论文推送

See `Usage.md` for instructions on how to personalize the repo. 


Keyword list: ['text-to-speech', 'text to speech', 'tts', 'LLM-based', 'speech', 'voice']


Excluded: []


### Today: 5papers 
#### A Lightweight Dual-Branch Visual Front-End for Efficient Audio-Visual Target Speaker Extraction
 - **Authors:** Zhan Jin, Peijun Yang, Juan Liu, Ming Li
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2610.36834

 - **Pdf link:** https://arxiv.org/pdf/2610.36834

 - **Abstract**
 Audio-Visual Target Speaker Extraction (AVTSE) uses face-related visual cues to extract a target speaker from a speech mixture. In many AVTSE systems, visual features are obtained from convolutional image encoders, and their computation may become a non-negligible part of the overall system when lightweight separation backbones are used. This work proposes a dual-branch visual front-end that reorganizes visual information according to its spatial characteristics. A compact lip region of interest is used to model local articulation cues through convolutional processing, while facial landmark coordinates are used to represent global geometric motion with a lightweight temporal convolutional network. The two representations are integrated into IIANet and SpatialNet with limited fusion overhead. Experiments on LRS3-2mix under a unified reduced training protocol show that the proposed method achieves a performance-computation trade-off close to the lightweight image-based baseline. Compared with the ResNet18-based visual encoder, it reduces the module-level visual encoding computation by more than 90% across both evaluated separation backbones, with a moderate performance gap. This comparison excludes face detection and landmark estimation and therefore applies to systems where landmark coordinates are already available from preprocessing. Demos are available at this https URL.
#### Perceptual Differences and Preferences in Speech Enhancement
 - **Authors:** Simon Dahl Jepsen, Jesper Rindom Jensen
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2610.36892

 - **Pdf link:** https://arxiv.org/pdf/2610.36892

 - **Abstract**
 Speech enhancement (SE) systems are commonly evaluated with objective metrics, with and without references, and with categorical subjective listening tests, all measuring speech quality and intelligibility. So far, it is unclear from such evaluations whether the processed speech is perceptually different or is preferred by everyday listeners. We investigate discrimination and preference as complementary dimensions of SE evaluation through crowdsourced triangle and pairwise preference tests across four SE systems, with comparisons to clean references. Results show that both listeners' ability to distinguish and whether they exhibit systematic preference vary across systems and substantially across acoustic conditions. Pairwise evaluations of objective metrics showed variable correlation with both discriminability and preference, motivating further studies into these novel aspects in speech enhancement evaluation.
#### StreamSE: Neural Audio Codec-based Streaming Speech Enhancement for On-device Inference
 - **Authors:** Hyunjin Choi, Zinu Kim, Junwon Moon, Jeongyun Chae, Seungbeom Kim, Kyuhong Shim
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2610.37024

 - **Pdf link:** https://arxiv.org/pdf/2610.37024

 - **Abstract**
 Recent generative speech enhancement (SE) models achieve high perceptual quality, but their non-causal processing and expensive computation make them difficult to use for streaming, on-device inference. We propose StreamSE, a 20M-parameter streaming SE model that operates in the latent space of a frozen pretrained neural audio codec. At each frame, StreamSE directly regresses a clean continuous latent from the noisy continuous latent while conditioning on its own predictions from previous frames. Enhancement therefore requires a single forward pass per frame and no iterative sampling or look-ahead. We further truncate the codec's residual vector quantizer (RVQ) to its coarse levels, halving the number of tokens per frame while improving robustness to reverberation. StreamSE is the only streaming generative model in our comparison, yet it matches or exceeds the perceptual quality of the offline generative baselines. Furthermore, the complete StreamSE pipeline runs in real time on a single smartphone CPU core. Our results show that continuous latent regression provides a practical route to generative SE under fully causal, on-device streaming constraints.
#### From Voiceprints to User Entities: Online Speaker-Memory Construction for Multi-User Spoken Agents
 - **Authors:** Yucong Zhang, Cancan Li, Bo Xiao, Ke Zeng, Ming Li
 - **Subjects:** Subjects:
Sound (cs.SD); Artificial Intelligence (cs.AI); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2610.36713

 - **Pdf link:** https://arxiv.org/pdf/2610.36713

 - **Abstract**
 Personalized spoken agents must identify speakers before accessing user-specific memory, yet memory benchmarks typically assume correctly attributed histories. We argue that voiceprints are observations of identity, not identities themselves, and formulate online speaker-memory construction to maintain persistent user entities as acoustic evidence evolves. We introduce EvoSpeakerMem, a training-free layer for causal speaker routing, selective acoustic-state updates, and identity-partitioned memory writes and retrieval through a replaceable text-memory backend. We also construct LoCoMo-ID, a spoken-memory benchmark with hidden speaker identities, speaker-sensitive queries, simulated recording-condition shifts, and four-speaker identity-confusion tests. On longitudinal SMIIP-TV, genuine-probe rejection under a frozen near-enrollment threshold rises from 29.8% to 55.6% over approximately one year. On LoCoMo-ID speaker-sensitive queries, gated updating improves identity-gated QA over fixed representations by 3.86 and 11.62 percentage points under clean and shifted conditions, respectively, with no wrong-user authorizations. The advantage persists in the four-speaker test. Yet reducing fragmentation and improving QA can coexist with incorrect cross-user links, revealing a fragmentation--contamination trade-off that QA alone cannot capture.
#### Developing a Speech Language Model for Basque
 - **Authors:** Andoni Sudupe, Inma Hernaez, Ander Arriandiaga, Christoforos Souganidis, Ibon Saratxaga, Eva Navas
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Artificial Intelligence (cs.AI); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2610.36911

 - **Pdf link:** https://arxiv.org/pdf/2610.36911

 - **Abstract**
 Although the recent development of LLMs has paved the way for multi-modal interaction, minority languages face difficulties keeping pace. This work presents the first open Speech Language Model for Basque, designed for low-latency, end-to-end spoken interaction. Based on the LLaMA-Omni architecture, the system integrates a Latxa model as its backbone, transferring its Basque capabilities to the spoken modality by fine-tuning several modules. To address data scarcity, the English VoiceAssistant-400K corpus was translated into Basque, and the audio was synthesized using a Basque TTS. Evaluation results confirm the system's viability, achieving 3.86/5 in content quality, a 9.94% WER for spoken intelligibility, and a 3.07 UTMOS for audio quality. This work demonstrates that advanced spoken interaction systems can be developed using modular architectures and synthetic data even in low-resource languages.


by Zyzzyva0381 (Windy). 


2026-10-28
