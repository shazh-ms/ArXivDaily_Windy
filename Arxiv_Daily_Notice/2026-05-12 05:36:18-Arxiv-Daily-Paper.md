# Showing new listings for Tuesday, 12 May 2026
Auto update papers at about 2:30am UTC (10:30am Beijing time) every weekday.


阅读 `Usage.md`了解如何使用此repo实现个性化的Arxiv论文推送

See `Usage.md` for instructions on how to personalize the repo. 


Keyword list: ['text-to-speech', 'text to speech', 'tts', 'LLM-based', 'speech', 'voice']


Excluded: []


### Today: 9papers 
#### Low-Cost Detection of Degraded Voice Clones via Source-Output Acoustic Consistency
 - **Authors:** Jana Shokr, Minos Papadopoulos, Jeremy Cooperstock, Pavo Orepic
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2605.08165

 - **Pdf link:** https://arxiv.org/pdf/2605.08165

 - **Abstract**
 Recent advances in generative speech have increased the need for automatic detection of obviously failed synthetic outputs. This is particularly important in clinical settings such as AVATAR therapy, in which schizophrenia patients engage with a computer-generated representation of their hallucinated voices and degraded synthesis may disrupt immersion and therapeutic engagement. We investigate whether low-dimensional, interpretable source-output acoustic features can provide a lightweight first-pass detector of degraded voice-cloning outputs. Motivated by source-filter models of speech, we first test median fundamental frequency (f0) as a source-related consistency measure, and compare it with vocal tract length (VTL) as a filter-related measure and Harmonics-to-Noise Ratio (HNR) as a noise-related descriptor. Human-labeled voice-cloning samples generated with two vocoder families, WaveRNN (n=54) and HiFi-GAN (n=40), were evaluated using an asymmetric thresholding procedure in the input-output feature space. For WaveRNN, f0 and HNR both achieved 85.2% accuracy, outperforming VTL (64.8%). For HiFi-GAN, HNR achieved 80.0% accuracy, followed by f0 at 77.5% and VTL at 67.5%. Sample-level overlap and spectrographic inspection showed that f0 and HNR capture partly distinct failure patterns, rather than providing redundant rankings of the same samples. These results show that simple source-output acoustic consistency measures can provide useful first-pass detection of degraded voice clones, and support the use of interpretable threshold-based screening in applications where failed synthetic speech must be rejected quickly.
#### DiffVQE: Hybrid Diffusion Voice Quality Enhancement Under Acoustic Echo and Noise
 - **Authors:** Haljan Lugo Girao, Ernst Seidel, Pejman Mowlaee, Ziyue Zhao, Tim Fingscheidt
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2605.08189

 - **Pdf link:** https://arxiv.org/pdf/2605.08189

 - **Abstract**
 Acoustic echo and background noise pose challenges on speech enhancement in hands-free systems and speakerphones. Discriminatively trained end-to-end methods represent a powerful solution for joint acoustic echo control (AEC) and denoising. However, with the advent of generative methods, diffusion-based approaches have seen remarkable performance in speech enhancement tasks. In this work, to the best of our knowledge, we provide the first (still non-causal) diffusion-based AEC model (DiffVQE) that is reproducible in terms of topology, training data, and training framework. So far, without employing diffusion, Microsoft's discriminative DeepVQE model has been shown to excel any of the ICASSP 2023 AEC Challenge entries achieving remarkable performance. Using data from the Interspeech 2025 URGENT Challenge for a diverse, high-quality training dataset, our DiffVQE excels DeepVQE both in echo and noise control performance, as well as in computational complexity and model size.
#### Latent Secret Spin: Keyed Orthogonal Rotations for Blind Speech Watermarking in Anisotropic Latent Spaces
 - **Authors:** Emma Coletta, Massimiliano Todisco, Michele Panariello, Antonio Faonio, Nicholas Evans
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2605.08431

 - **Pdf link:** https://arxiv.org/pdf/2605.08431

 - **Abstract**
 We introduce Latent Secret Spin (LSS), a blind speech watermarking method based on geometric operations in codec latent space. Based upon orthogonal rotations to principal components, LSS induces imperceptible but detectable covariance signatures according to a pseudo-random watermarking schedule. The scheme generalises across datasets, preserves perceptual quality and, unlike some learned, neural watermarking schemes, it does not require neural network training, is resistant to common signal manipulations and is flexible to payload size. Analyses show that structured latent-space watermarking is a promising and interpretable alternative to existing approaches.
#### Reducing Linguistic Hallucination in LM-Based Speech Enhancement via Noise-Invariant Acoustic-Semantic Distillation
 - **Authors:** Zheng Wang, Xiaobin Rong, Hang Su, Tianyi Tan, Junnan Wu, Lichun Fan, Zhenbo Luo, Jian Luan, Jing Lu
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2605.08608

 - **Pdf link:** https://arxiv.org/pdf/2605.08608

 - **Abstract**
 Language model (LM)-based speech enhancement (SE) can generate natural-sounding speech, but under severe noise it often suffers from unreliable conditioning, leading to perceptually plausible yet linguistically incorrect outputs. To address this issue, we propose L3-SE, a noise-invariant acoustic-semantic distillation framework for reducing linguistic hallucination in LM-based SE. The proposed method learns a noise-invariant conditioning encoder from noisy speech by jointly distilling two complementary clean-speech targets: an acoustic target for reconstruction fidelity and a semantic target for linguistic consistency. The resulting noise-invariant acoustic-semantic representations are used to condition a decoder-only autoregressive language model, which predicts clean acoustic tokens that are decoded into enhanced speech. To support high-quality generation, we further employ a high-fidelity codec built on learnable weighted WavLM layer representations as the discrete acoustic interface. By improving the reliability of conditioning under adverse conditions, the proposed framework substantially reduces hallucination and improves content faithfulness. Experiments show that the proposed method consistently outperforms prior LM-based speech enhancement baselines on linguistic consistency metrics, with especially clear gains under low-SNR and reverberant conditions, while maintaining competitive perceptual quality. Audio samples are available at this https URL. The complete source code will be released after the manuscript is accepted.
#### Kinetic-Optimal Scheduling with Moment Correction for Metric-Induced Discrete Flow Matching in Zero-Shot Text-to-Speech
 - **Authors:** Dong Yang, Yiyi Cai, Haoyu Zhang, Yuki Saito, Hiroshi Saruwatari
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Artificial Intelligence (cs.AI); Machine Learning (cs.LG)
 - **Arxiv link:** https://arxiv.org/abs/2605.09386

 - **Pdf link:** https://arxiv.org/pdf/2605.09386

 - **Abstract**
 Metric-induced discrete flow matching (MI-DFM) exploits token-latent geometry for discrete generation, but its practical use is limited by two issues: heuristic schedulers requiring hyperparameter search, and finite-step path-tracking error from its first-order continuous-time Markov chain (CTMC) solver. We address both issues. First, we derive a kinetic-optimal scheduler for prescribed scalar-parameterized probability paths, and instantiate it for MI-DFM as a training-free numerical schedule that traverses the path at constant Fisher-Rao speed. Second, we introduce a finite-step moment correction that adjusts the jump probability while preserving the CTMC jump destination distribution. We validate the resulting method, GibbsTTS, on codec-based zero-shot text-to-speech (TTS). Under controlled comparisons with a unified architecture and large-scale dataset, GibbsTTS achieves the best objective naturalness and is preferred in subjective evaluations over masked discrete generative baselines. Additionally, in comparison with the evaluated state-of-the-art TTS systems, GibbsTTS shows strong speaker similarity, achieving the highest similarity on three of four test sets and ranking second on the fourth. Project page: this https URL
#### Evaluating the Expressive Appropriateness of Speech in Rich Contexts
 - **Authors:** Tianrui Wang, Ziyang Ma, Yizhou Peng, Haoyu Wang, Zhikang Niu, Zikang Huang, Yihao Wu, Yi-Wen Chao, Yu Jiang, Yuheng Lu, Guanrou Yang, Xuanchen Li, Hexin Liu, Chunyu Qiang, Cheng Gong, Yifan Yang, Tianchi Liu, Junyu Wang, Nana Hou, Meng Ge, Fuming You, Wei Yang, Zhongqian Sun, Haifeng Hu, Xiaobao Wang, Eng Siong Chng, Xie Chen, Longbiao Wang, Jianwu Dang
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2605.09413

 - **Pdf link:** https://arxiv.org/pdf/2605.09413

 - **Abstract**
 Evaluating expressive speech remains challenging, as existing methods mainly assess emotional intensity and overlook whether a speech sample is expressively appropriate for its contextual setting. This limitation hinders reliable evaluation of speech systems used in narrative-driven and interactive applications, such as audiobooks and conversational agents. We introduce CEAEval, a Context-rich framework for Evaluating Expressive Appropriateness in speech, which assesses whether a speech sample expressively aligns with the underlying communicative intent implied by its discourse-level narrative context. To support this task, we construct CEAEval-D, the first context-rich speech dataset with real human performances in Mandarin conversational speech, providing narrative descriptions together with fifteen dimensions of human annotations covering expressive attributes and expressive appropriateness. We further develop CEAEval-M, a model that integrates knowledge distillation, planner-based multi-model collaboration, adaptive audio attention bias, and reinforcement learning to perform context-rich expressive appropriateness evaluation. Experiments on a human-annotated test set demonstrate that CEAEval-M substantially outperforms existing speech evaluation and analysis systems.
#### PoDAR: Power-Disentangled Audio Representation for Generative Modeling
 - **Authors:** Alejandro Luebs, Mithilesh Vaidya, Ishaan Kumar, Sumukh Badam, Stephen W. Bailey, Matthew Bendel, Jose Sotelo, Xingzhe He
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Artificial Intelligence (cs.AI); Machine Learning (cs.LG); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2605.10084

 - **Pdf link:** https://arxiv.org/pdf/2605.10084

 - **Abstract**
 The performance of audio latent diffusion models is primarily governed by generator expressivity and the modelability of the underlying latent space. While recent research has focused primarily on the former, as well as improving the reconstruction fidelity of audio codecs, we demonstrate that latent modelability can be significantly improved through explicit factor disentanglement. We present PoDAR (Power-Disentangled Audio Representation), a framework that utilizes a randomized power augmentation and latent consistency objective to decouple signal power from invariant semantic content. This factorization makes the latent space easier to model, which both accelerates the convergence of downstream generative models and improves final overall performance. When applied to a Stable Audio 1.0 VAE with an F5-TTS generator, PoDAR achieves about a $2\times$ acceleration in convergence to match baseline performance, while increasing final speaker similarity by 0.055 and UTMOS by 0.22 on the LibriSpeech-PC dataset. Furthermore, isolating power into dedicated channels enables the application of CFG exclusively to power-invariant content, effectively extending the stable guidance regime to higher scales.
#### SF-Flow: Sound field magnitude estimation via flow matching guided by sparse measurements
 - **Authors:** Ege Erdem, Shoichi Koyama, Tomohiko Nakamura, Orchisama Das, Zoran Cvetković
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2605.10398

 - **Pdf link:** https://arxiv.org/pdf/2605.10398

 - **Abstract**
 Reconstructing a 3D sound field from sparse microphone measurements is a fundamental yet ill-posed problem, which we address through Acoustic Transfer Function (ATF) magnitude estimation. ATF magnitude encapsulates key perceptual and acoustic properties of a physical space with applications in room characterization and correction. Although recent generative paradigms such as Flow Matching (FM) have achieved state-of-the-art performance in speech and music generation, their potential in spatial audio remains underexplored. We propose a novel framework for 3D ATF magnitude reconstruction as a guided generation task, with a 3D U-Net conditioned by a permutation-invariant set encoder. This architecture enables reconstruction from an arbitrary number of sparse inputs while leveraging the stable and efficient training properties of FM. Experimental results demonstrate that SF-Flow achieves accurate reconstruction up to \SI{1}{kHz}, trains substantially faster than the autoencoder baseline, and improves significantly with dataset size.
#### Bangla-WhisperDiar: Fine-Tuning Whisper and PyAnnote for Bangla Long-Form Speech Recognition and Speaker Diarization
 - **Authors:** Mohammed Aman Bhuiyan, Md Sazzad Hossain Adib, Samiul Basir Bhuiyan, Amit Chakraborty, Aritra Islam Saswato, Ahmed Faizul Haque Dhrubo, Mohammad Ashrafuzzaman Khan
 - **Subjects:** Subjects:
Sound (cs.SD); Artificial Intelligence (cs.AI); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2605.08214

 - **Pdf link:** https://arxiv.org/pdf/2605.08214

 - **Abstract**
 Automatic Speech Recognition (ASR) and speaker diarization in Bangla remain challenging due to long form recordings, diverse acoustic conditions, and significant speaker variability. This work addresses these two core tasks in Bangla spoken language understanding by developing robust systems for long form ASR and speaker diarization. For ASR (Problem 1), we fine tune the tugstugi bengaliai regional asr whisper medium model on a custom-curated dataset of approximately 15,000 chunked and aligned Bangla audio segments, employing full weight training with extensive data augmentation including noise injection, reverb simulation, echo, clipping distortion, and pitch/time perturbation. For speaker diarization (Problem 2), we fine-tune the pyannote/segmentation-3.0 model using PyTorch Lightning on the competition annotated diarization dataset, swapping the fine-tuned segmentation backbone into the pyannote/speaker-diarization-community-1 pipeline while retaining the pretrained speaker embedding and clustering components. Our ASR system achieves a Word Error Rate (WER) of 0.2441, while our diarization system achieves a Diarization Error Rate (DER) of 0.2392, both evaluated on the test set, demonstrating notable improvements over the respective pretrained baselines. We describe our complete pipeline, including data preprocessing, text normalization, audio augmentation, training strategies, inference optimization, and post-processing for both tasks.


by Zyzzyva0381 (Windy). 


2026-05-12
