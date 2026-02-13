# Showing new listings for Friday, 13 February 2026
Auto update papers at about 2:30am UTC (10:30am Beijing time) every weekday.


阅读 `Usage.md`了解如何使用此repo实现个性化的Arxiv论文推送

See `Usage.md` for instructions on how to personalize the repo. 


Keyword list: ['text-to-speech', 'text to speech', 'tts', 'LLM-based', 'speech', 'voice']


Excluded: []


### Today: 3papers 
#### SLD-L2S: Hierarchical Subspace Latent Diffusion for High-Fidelity Lip to Speech Synthesis
 - **Authors:** Yifan Liang, Andong Li, Kang Yang, Guochen Yu, Fangkun Liu, Lingling Dai, Xiaodong Li, Chengshi Zheng
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Computational Engineering, Finance, and Science (cs.CE)
 - **Arxiv link:** https://arxiv.org/abs/2602.11477

 - **Pdf link:** https://arxiv.org/pdf/2602.11477

 - **Abstract**
 Although lip-to-speech synthesis (L2S) has achieved significant progress in recent years, current state-of-the-art methods typically rely on intermediate representations such as mel-spectrograms or discrete self-supervised learning (SSL) tokens. The potential of latent diffusion models (LDMs) in this task remains largely unexplored. In this paper, we introduce SLD-L2S, a novel L2S framework built upon a hierarchical subspace latent diffusion model. Our method aims to directly map visual lip movements to the continuous latent space of a pre-trained neural audio codec, thereby avoiding the information loss inherent in traditional intermediate representations. The core of our method is a hierarchical architecture that processes visual representations through multiple parallel subspaces, initiated by a subspace decomposition module. To efficiently enhance interactions within and between these subspaces, we design the diffusion convolution block (DiCB) as our network backbone. Furthermore, we employ a reparameterized flow matching technique to directly generate the target latent vectors. This enables a principled inclusion of speech language model (SLM) and semantic losses during training, moving beyond conventional flow matching objectives and improving synthesized speech quality. Our experiments show that SLD-L2S achieves state-of-the-art generation quality on multiple benchmark datasets, surpassing existing methods in both objective and subjective evaluations.
#### TC-BiMamba: Trans-Chunk bidirectionally within BiMamba for unified streaming and non-streaming ASR
 - **Authors:** Qingshun She, Jing Peng, Yangui Fang, Yu Xi, Kai Yu
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2602.11546

 - **Pdf link:** https://arxiv.org/pdf/2602.11546

 - **Abstract**
 This work investigates bidirectional Mamba (BiMamba) for unified streaming and non-streaming automatic speech recognition (ASR). Dynamic chunk size training enables a single model for offline decoding and streaming decoding with various latency settings. In contrast, existing BiMamba based streaming method is limited to fixed chunk size decoding. When dynamic chunk size training is applied, training overhead increases substantially. To tackle this issue, we propose the Trans-Chunk BiMamba (TC-BiMamba) for dynamic chunk size training. Trans-Chunk mechanism trains both bidirectional sequences in an offline style with dynamic chunk size. On the one hand, compared to traditional chunk-wise processing, TC-BiMamba simultaneously achieves 1.3 times training speedup, reduces training memory by 50%, and improves model performance since it can capture bidirectional context. On the other hand, experimental results show that TC-BiMamba outperforms U2++ and matches LC-BiMmaba with smaller model size.
#### When Audio-LLMs Don't Listen: A Cross-Linguistic Study of Modality Arbitration
 - **Authors:** Jayadev Billa
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2602.11488

 - **Pdf link:** https://arxiv.org/pdf/2602.11488

 - **Abstract**
 When audio and text conflict, speech-enabled language models follow the text 10 times more often than when arbitrating between two text sources, even when explicitly instructed to trust the audio. Using ALME, a benchmark of 57,602 controlled audio-text conflict stimuli across 8 languages, we find that Gemini 2.0 Flash exhibits 16.6\% text dominance under audio-text conflict versus 1.6\% under text-text conflict with identical reliability cues. This gap is not explained by audio quality: audio-only accuracy (97.2\%) exceeds cascade accuracy (93.9\%), indicating audio embeddings preserve more information than text transcripts. We propose that text dominance reflects an asymmetry not in information content but in arbitration accessibility: how easily the model can reason over competing representations. This framework explains otherwise puzzling findings. Forcing transcription before answering increases text dominance (19\% to 33\%), sacrificing audio's information advantage without improving accessibility. Framing text as ``deliberately corrupted'' reduces text dominance by 80\%. A fine-tuning ablation provides interventional evidence: training only the audio projection layer increases text dominance (+26.5\%), while LoRA on the language model halves it ($-$23.9\%), localizing text dominance to the LLM's reasoning rather than the audio encoder. Experiments across four state-of-the-art audio-LLMs and 8 languages show consistent trends with substantial cross-linguistic and cross-model variation, establishing modality arbitration as a distinct reliability dimension not captured by standard speech benchmarks.


by Zyzzyva0381 (Windy). 


2026-02-13
