# Showing new listings for Thursday, 30 July 2026
Auto update papers at about 2:30am UTC (10:30am Beijing time) every weekday.


阅读 `Usage.md`了解如何使用此repo实现个性化的Arxiv论文推送

See `Usage.md` for instructions on how to personalize the repo. 


Keyword list: ['text-to-speech', 'text to speech', 'tts', 'LLM-based', 'speech', 'voice']


Excluded: []


### Today: 5papers 
#### A Study on Online Mask-based Beamforming Using Per-channel Masking for Spatially Distributed Microphones
 - **Authors:** Wiebke Middelberg, Svantje Void, Simon Doclo, Ryan Corey
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2607.26623

 - **Pdf link:** https://arxiv.org/pdf/2607.26623

 - **Abstract**
 Mask-based beamforming is a popular geometry-agnostic approach for speech enhancement, typically applying a single mask across all microphones to estimate the required covariance matrices. While effective for compact arrays, this strategy may be suboptimal for spatially distributed microphones, where signal characteristics may vary strongly across microphones. To effectively capture the spatial diversity across microphones, we extend the mask-based beamformer to a multi-channel formulation, where each microphone is pre-filtered by a separate mask before covariance estimation. To address time-varying acoustic scenes, caused by spectro-temporal nonstationarity, we adopt a frame-causal online implementation with a sliding window. Experiments with simulated compact arrays and distributed microphones show that multi-channel masking yields a benefit over using a single mask when microphone signals differ substantially, while retaining similar performance in compact arrays. We further demonstrate the robustness of the multi-channel masking approach by comparing oracle ideal ratio masks to blind DNN-based mask estimation.
#### Zero-Shot Face-to-Speech Synthesis via Latent Space Adaptation of a Style-Diffusion TTS Model
 - **Authors:** Carlos MuÃ±oz-Romero, Jose A. Gonzalez-Lopez
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Artificial Intelligence (cs.AI); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2607.26742

 - **Pdf link:** https://arxiv.org/pdf/2607.26742

 - **Abstract**
 Zero-shot text-to-speech (TTS) clones a voice from a short audio prompt, but this reliance on reference audio is a barrier when only visual information is available, e.g. for historical figures or video-game characters. In this work, we propose a Face-to-Speech (F2S) framework that predicts a plausible voice from a static facial image. A lightweight Face Adapter, together with soft-tuning of the face encoder's upper blocks, aligns face-recognition features with the style space of a frozen StyleTTS 2 model, kept frozen during training. We evaluate on held-out identities from LRS3, a large-scale audiovisual corpus of English TED-talk videos. The synthesized speech is highly natural (UTMOS 3.7-4.0, matching or exceeding the 3.61 of ground truth), face-to-voice retrieval is consistently above chance, and the generated voice is consistent with the target speaker. Without any retraining, an English-trained adapter also produces fluent Spanish speech, indicating that the face-to-style mapping is largely language-agnostic.
#### Qwen-Audio-3.0-Gen-Preview Technical Report
 - **Authors:** Junyu Dai, Xiaoyue Duan, Xinyue Fan, Yihan Feng, Xiangang Li, Yunjia Li, Lejun Min, Yufei Shi, Xingchen Song, Yiran Wang, Cheng Wen, Menglin Wu, Bajian Xiang, Huaicheng Zhang, Han Zhao, Ruichen Zheng
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2607.27011

 - **Pdf link:** https://arxiv.org/pdf/2607.27011

 - **Abstract**
 Existing single-domain and multi-task audio systems remain limited in directly organizing speech, music, sound effects, ambience, and multiple roles into long-form temporal scenes. We present Qwen-Audio-3.0-Gen-Preview, a unified non-autoregressive framework that uses a Diffusion Transformer (DiT) and a shared variational autoencoder (VAE) to generate the complete mixed waveform. Prompt enhancement converts free-form requests into structured temporal records that are rendered as textual conditions, while a two-stage data curriculum and semantic conditional views train the proposed model to use these conditions across domains. A shared continuous VAE compresses 48kHz stereo waveforms into 25Hz latent sequences and incorporates semantic supervision, providing one representation for speech, music, sound effects, and their mixtures. On Seed-TTS-Eval, speaker similarity is the proposed model's clearest strength across all three subsets, and on the multi-speaker benchmark, the proposed model shows higher cross-turn consistency than Seed-Audio-1.0 in both languages. On AudioCaps, its advantages are concentrated in evaluations using large audio-language models and AudioBox. Relative to Seed-Audio-1.0, it achieves stronger temporal localization. Using approximately 10% music data of a dedicated in-house model, the proposed model remains close across all seven SongBench components and leads in three while retaining speech and general-audio capabilities. These results demonstrate the potential of unified generation for temporally structured, multi-domain audio.
#### Voice Memory for Agentic Speech Recognition
 - **Authors:** Chao-Han Huck Yang, Zih-Ching Chen, Piotr Zelasko, Zhehuai Chen, Jagadeesh Balam, Boris Ginsburg
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Artificial Intelligence (cs.AI); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2607.26410

 - **Pdf link:** https://arxiv.org/pdf/2607.26410

 - **Abstract**
 We present Voice Memory, a inference-only scheme for agentic speech recognition: at stream time, a frozen corrector reads a single per-domain this http URL and decides per utterance whether to act on the hypothesis or abstain and keep the 1-best. Asynchronously, a score-gated optimizer revises that file through bounded edits, accepting an edit only when it strictly improves a held-out score. Extended from classical ASR-LM framework, we refer this split the listener-thinker architecture; the two roles are coupled only through the memory, so no weights change and the learned skill stays auditable and portable. Restraint turns out to be the operative skill this loop discovers: unconstrained generative error correction (GER) over-corrects, breaking correct tokens on up to 64% of its edits on financial news, and Voice Memory, reduces this rate to 35%. Across ten HyPoradise domains with an open corrector, Voice Memory, lowers weighted word error rate from 8.36% to 7.52% (7.47% with three added in-context examples) without regressing any dataset below its 1-best baseline; gains concentrate where recoverable headroom is largest, including air-travel commands (8.40% to 3.40%) and noisy far-field speech (CHiME-4, 12.69% to 10.46%). The memory transfers across corrector families and adds zero parameters to the inference path. A demo and example code are provided for future studies.
#### MPEcho: A Melody and Phoneme-Aware Generative Framework for Controllable Cover Song Generation
 - **Authors:** Wei-Jaw Lee, Hsuan-Yu Yeh, Ting-Yi Hu, Chih-Pin Tan, Fang-Duo Tsai, Yi-Hsuan Yang
 - **Subjects:** Subjects:
Sound (cs.SD); Artificial Intelligence (cs.AI); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2607.26698

 - **Pdf link:** https://arxiv.org/pdf/2607.26698

 - **Abstract**
 Cover song generation (CSG) should preserve the melodic and linguistic content of a reference song while recreating the remaining musical components. The state-of-the-art model SongEcho utilizes $F_0$ sequences and voiced/unvoiced (V/UV) tags for conditioning; however, implicit linguistic information from V/UV tags cannot guarantee lyric accuracy, leading to a high phoneme error rate (PER). Inspired by singing voice synthesis (SVS), we propose MPEcho, which integrates a phoneme encoder and a length regulator (LR) into the SongEcho framework. By providing explicit phoneme-level conditioning and precise temporal boundaries, MPEcho significantly reduces PER. To enable this, we developed Phonsa, a Whisper-based automatic transcription model that provides high-precision phoneme-level annotations for singing voices, overcoming the scarcity of high-quality audio-phoneme pairs. Experimental results validate the effectiveness of Phonsa for alignment and MPEcho for end-to-end CSG. The audio samples, code and weights can be accessed from this https URL.


by Zyzzyva0381 (Windy). 


2026-07-30
