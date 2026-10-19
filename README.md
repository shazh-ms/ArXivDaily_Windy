# Showing new listings for Monday, 19 October 2026
Auto update papers at about 2:30am UTC (10:30am Beijing time) every weekday.


阅读 `Usage.md`了解如何使用此repo实现个性化的Arxiv论文推送

See `Usage.md` for instructions on how to personalize the repo. 


Keyword list: ['text-to-speech', 'text to speech', 'tts', 'LLM-based', 'speech', 'voice']


Excluded: []


### Today: 4papers 
#### Bridge, Distill, Refine: High-Fidelity One-Step Generation for Multi-Bandwidth Speech Extension
 - **Authors:** Zhiqi Ai, Xu Shen, Mou Wang, Yongjin Zhou, Shugong Xu
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2610.22910

 - **Pdf link:** https://arxiv.org/pdf/2610.22910

 - **Abstract**
 Speech bandwidth extension (BWE) aims to recover missing spectral content from band-limited speech. However, most generative BWE methods are designed for fixed input bandwidths and rely on iterative sampling, limiting their flexibility and efficiency. We propose BDR-BWE, a three-stage generative framework for multi-bandwidth speech extension. First, to enable conditional generation from the observed band-limited spectrum, we design a cutoff-aware conditional flow bridge in the compressed complex STFT domain, focusing generation on the missing high-frequency band while preserving observed spectral information. Second, progressive consistency distillation compresses the multi-step flow transport into a one-step generator. Third, adversarial refinement further improves high-frequency detail and harmonic coherence. BDR-BWE achieves state-of-the-art log-spectral distance (LSD) on VCTK across all evaluated bandwidths with one-step inference, while maintaining competitive perceptual quality and generalizing well to EARS. Audio samples are available at this https URL.
#### AcousticAgent: A Floorplan-Centric Simulation Platform for Distributed Multi-Room Speech Processing
 - **Authors:** Zhiqi Ai, Han Cheng, Shiyi Mu, Yongjin Zhou, Shugong Xu
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2610.22936

 - **Pdf link:** https://arxiv.org/pdf/2610.22936

 - **Abstract**
 Existing acoustic simulation tools either generate room impulse responses (RIRs) efficiently in parameterized rooms or rely on preconstructed three-dimensional scenes, making it difficult to build large numbers of editable residential multi-room environments. We present AcousticAgent, a floorplan-centric acoustic simulation platform that converts structured floorplans into editable, acoustically connected multi-room scenes. AcousticAgent jointly models room geometry and connectivity, furniture, materials, sound sources, and microphone configurations. A scene-oriented Acoustic Materials Database maps semantic elements to acoustic parameters, while a floorplan-aware hybrid geometrical-acoustics solver generates multichannel RIRs and reverberant audio. The platform also supports directional and moving sources, distributed microphones, and VLM-assisted scene construction from floorplan images and text. Using AcousticAgent, we construct FP-RIR, a dataset covering 15,376 residential floorplans with diverse room topologies, propagation conditions, and source--receiver configurations. We validate its acoustic fidelity against real-world measurements and evaluate whole-home source localization, speech enhancement, and far-field speech processing under single-device and distributed microphone settings. The results demonstrate the effectiveness of floorplan-aware modeling for large-scale residential acoustic simulation and the benefits of distributed spatial processing in multi-room environments. AcousticAgent and FP-RIR provide scalable and configurable resources for developing and evaluating residential speech systems.
#### JAVIS: Joint Audio-Visual Interaction and Streaming for Real-Time Conversational Agents
 - **Authors:** Junseok Ahn, Jaehyun Park, Jeongsoo Choi, Joon Son Chung, Ji-Hoon Kim
 - **Subjects:** Subjects:
Computer Vision and Pattern Recognition (cs.CV); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2610.22499

 - **Pdf link:** https://arxiv.org/pdf/2610.22499

 - **Abstract**
 Human communication is inherently multimodal, conveying intent not only through linguistic and vocal cues but also through visual signals. However, existing conversational systems have mainly focused on text or speech, leaving out the visual feedback that is essential for natural interaction. In this paper, we propose JAVIS, an end-to-end streaming framework for real-time audio-visual conversational agents. Our key idea is to let the voice and the face speak from a single representation. On top of a shared LLM, a Joint Refined Head produces a unified audio-visual representation, which is fed into dedicated streaming branches for speech and face generation. This end-to-end design avoids intermediate bottlenecks, enabling real-time generation while keeping speech and face tightly synchronized. In addition, we present JAVIS-Syn, a 334-hour synthetic corpus designed to address the lack of expressive audio-visual dialogue data, covering emotionally rich multi-turn conversations and assistant-style exchanges. Experiments show that JAVIS achieves approximately 200$\times$ lower initial latency than a strong baseline, while outperforming baselines in lip synchronization and facial expression appropriateness.
#### ConvexLLM: Scaling Convex-Hull Speech Interfaces to General-Purpose Audio Intelligence
 - **Authors:** Ming-Hao Hsu, Shujie Liu, Yuxuan Hu, Yuxiang Wang, Yingda Shen, Jinyu Li, Zhizheng Wu
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2610.23539

 - **Pdf link:** https://arxiv.org/pdf/2610.23539

 - **Abstract**
 Speech and audio language models connect continuous acoustic representations to decoders pretrained primarily on text data. Most such models use MLP or Q-Former interfaces to map acoustic features into the decoder's input space, requiring the decoder to learn how to interpret the resulting representations. We investigate whether grounding these representations in the decoder's existing text embeddings can help it use acoustic information while better preserving its pretrained text capabilities. We present ConvexLLM, a general-purpose speech and audio language model whose C-Gate interface represents audio as continuous convex combinations of frozen text embeddings. ConvexLLM achieves an eleven-task VoiceBench aggregate of 75.36, the highest among the open-weight models we compare, and an MMAU Test-mini accuracy of 62.50%. Its largest margins over the baselines are on spoken knowledge and reasoning tasks, while the strongest baselines are more accurate on audio understanding and transcription. In a comparison with a 2B-parameter decoder, C-Gate achieves higher aggregate spoken-language performance than MLP and Q-Former interfaces, with text-task performance close to MLP and task-dependent tradeoffs in transcription and audio understanding. These results support convex combinations of text embeddings as a viable interface for general-purpose speech and audio language models.


by Zyzzyva0381 (Windy). 


2026-10-19
