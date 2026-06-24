# Showing new listings for Wednesday, 24 June 2026
Auto update papers at about 2:30am UTC (10:30am Beijing time) every weekday.


阅读 `Usage.md`了解如何使用此repo实现个性化的Arxiv论文推送

See `Usage.md` for instructions on how to personalize the repo. 


Keyword list: ['text-to-speech', 'text to speech', 'tts', 'LLM-based', 'speech', 'voice']


Excluded: []


### Today: 12papers 
#### A Variational-Flow Analysis of StoRM under Noise-Power Mismatch
 - **Authors:** Shuubham Ojha
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2606.24035

 - **Pdf link:** https://arxiv.org/pdf/2606.24035

 - **Abstract**
 Diffusion-based speech enhancement architectures that pair a deterministic predictor with a learned score network, exhibit a sharp non-smooth transition (``kink'') in the SI-SDR degradation curve at the training-time noise amplitude. We give a pathwise variational-flow analysis that localizes this non-smoothness to the predictor stage. The central identity is an exact factorization of the parametric sensitivity, $\partial \sig^{(M)} / \partial M = K(M) \cdot \partial C_M / \partial M$, where $K(M)$ is a continuous matrix-valued functional of the score Jacobian along the reverse trajectory and $C_M = \Pi(y^{(M)})$ is the predictor output. Under three hypotheses on the reverse-process flow (score-Jacobian continuity, conditioning-Jacobian continuity, non-degeneracy of $K$), failure of $M \mapsto \sig^{(M)}$ to be $C^1$ at $M^\ast$ holds if and only if $M \mapsto \Pi(y^{(M)})$ fails to be $C^1$ at $M^\ast$. We extend the localization to the finite-step Euler--Maruyama sampler actually run at inference. The hypotheses translate into a concrete experimental program; this paper specifies the program and presents the variational structure. The empirical validation is deferred to a companion experimental report.
#### Audio--Image Alignment as a Continued-Pretraining Stage Improves Low-Resource ASR
 - **Authors:** Sujith Pulikodan, Nihar Desai, Prasanta Kumar Ghosh
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2606.24080

 - **Pdf link:** https://arxiv.org/pdf/2606.24080

 - **Abstract**
 Thousands of languages are spoken worldwide, yet many remain under-resourced for Automatic Speech Recognition (ASR) due to the limited availability of high-quality transcribed speech data. Collecting accurate transcriptions is often costly and labor-intensive, particularly for low-resource languages. In this work, we investigate the use of aligned audio-image pairs to adapt pretrained audio encoders without requiring transcription data before supervised fine-tuning. Our proposed representation alignment stage is introduced between large-scale pretraining and supervised ASR fine-tuning. Specifically, image representations extracted from pretrained vision encoders are aligned with audio representations to further adapt a pretrained audio encoder. For this alignment process, we utilize the Vaani dataset, in which images serve as prompts for speech collection, naturally providing paired audio-image data. We evaluate the proposed approach using multiple vision encoders and a pretrained FastConformer audio encoder. Experimental results demonstrate that models fine-tuned after representation alignment consistently achieve improved ASR performance compared to direct fine-tuning. These findings highlight the potential of audio-image representation alignment as an effective transcription-free adaptation strategy for enhancing ASR systems in low-resource language settings.
#### Comparative Reasoning: Making an Audio Language Model Better at Comparing Emotions
 - **Authors:** Abinay Reddy Naini, Jaeyeon Kim, Chao-Han Huck Yang, Shinji Watanabe, Carlos Busso
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2606.24082

 - **Pdf link:** https://arxiv.org/pdf/2606.24082

 - **Abstract**
 Large audio-language models (LALMs) can reason about audio, yet it remains unclear whether they can perform comparative judgments between two speech signals along emotional, environmental, linguistic, prosodic, and interpersonal dimensions. We study this question in the context of speech emotion recognition (SER), where the model determines which utterance exhibits higher arousal, valence, or dominance. We introduce a reasoning-guided ordinal SER framework that conditions an LALM on paired speech inputs. The model is trained using reasoning traces generated from both semantic audio descriptions and acoustic evidence derived from GeMAPS features, enabling interpretable comparative decisions. Beyond direct supervision, we also employ direct preference optimization to encourage stronger separation for emotional differences. Experiments show that the proposed framework improves preference prediction while requiring only 5% of the training data used by conventional ordinal SER systems.
#### Autoencoder based optimized SSL representations: Complexity Minimization and improved Dysarthric ASR
 - **Authors:** Paban Sapkota, Hemant Kumar Kathania, Mikko Kurimo, Shrikanth Narayanan, Sudarsana Reddy Kadiri
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2606.24088

 - **Pdf link:** https://arxiv.org/pdf/2606.24088

 - **Abstract**
 Self-supervised learning (SSL) models extract rich speech representations but often come with high-dimensional features, increasing computational complexity. This work explores an SSL-AutoEncoder (SSL-AE) bottlenecking approach to efficiently reduce feature dimensions while maintaining dysarthric Automatic Speech Recognition (ASR) performance. By leveraging an autoencoder, we transform high-dimensional SSL features into a compact space, reducing model complexity and training time. Our method preserves essential speech information, achieving reduced Word Error Rates (WER) while significantly lowering computational costs. Experiments show SSL-AE bottlenecking reduces training time by 8x compared to the SSL baseline, demonstrating efficiency without sacrificing recognition performance. These results highlight AE as an effective solution for SSL feature compression in resource-constrained environments.
#### Joint Learning of Covariance Estimation and White Noise Gain for Robust MVDR Beamforming
 - **Authors:** Yongyi Deng, Hanchen Pei, Jianbo Ma, Gongping Huang, Jingdong Chen, Jacob Benesty
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2606.24137

 - **Pdf link:** https://arxiv.org/pdf/2606.24137

 - **Abstract**
 The minimum variance distortionless response (MVDR) beamformer is widely used for multichannel speech enhancement due to strong noise suppression while preserving target signals. In practice, its performance is sensitive to microphone self-noise and array mismatches. Existing approaches typically rely on fixed, manually tuned WNG thresholds or diagonal loading, leading to suboptimal performance under unknown or time-varying acoustic conditions. This paper proposes a data-driven MVDR framework that adaptively estimates the WNG constraint using a deep neural network. The network jointly predicts a time-frequency noise mask for covariance estimation and a frequency-dependent WNG threshold, enabling dynamic robustness-directivity control. A differentiable robust MVDR layer is integrated into the framework, allowing end-to-end optimization. Experiments demonstrate consistent improvements in speech quality and intelligibility over conventional fixed-WNG MVDR methods.
#### Progressive Alignment Objectives for Aligner-Encoder based ASR
 - **Authors:** Jaeyong Lee, Masato Mimura, Takafumi Moriya
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Computation and Language (cs.CL); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2606.24147

 - **Pdf link:** https://arxiv.org/pdf/2606.24147

 - **Abstract**
 Aligner-Encoders are recently proposed seq2seq end-to-end ASR models that replace decoder attention by predicting the uth token directly from the u-th encoder position, so the encoder must learn the alignment internally without cross-attention or a transducer lattice. In practice, this alignment often forms abruptly in the upper layers, making training sensitive and brittle on long utterances. We propose InterAligner, which adds an intermediate Aligner objective so alignment can form progressively across depth, together with an intermediate CTC loss (InterCTC) to stabilize optimization. On LibriSpeech with a 17-layer Conformer, a final-only Aligner reaches 5.0/7.8 WER (test-clean/other). InterCTC improves to 3.4/6.0, and InterAligner further reduces WER to 3.1/5.6 with the largest gains on long utterances.
#### Breaking Shortcut Learning for Cross-Trial EEG-Guided Target Speech Extraction via Two-Stage Training
 - **Authors:** Wonchul Shin, Inyong Choi, Kyogu Lee
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Artificial Intelligence (cs.AI); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2606.24164

 - **Pdf link:** https://arxiv.org/pdf/2606.24164

 - **Abstract**
 Recent end-to-end models for EEG-guided target speech extraction report impressive results, underscoring potential for neuro-steered hearing technologies. However, our analysis reveals that high within-trial performance can be driven by trial-specific EEG structure that acts as shortcuts for target selection, leading to poor generalization on unseen trials. To overcome this gap, we propose TRUST-TSE, a two-stage framework to mitigate shortcut learning. By introducing contrastive pretraining with attended-speaker negative sampling, we encourage the EEG encoder to capture fine-grained EEG--speech alignment while suppressing trial-identity cues. We also employ a confidence-weighted extraction objective based on EEG--source similarity to guide extraction using the learned representations. Experiments on KUL and DTU datasets show that TRUST-TSE outperforms end-to-end baselines under strict cross-trial protocols, addressing a key reliability bottleneck of existing approaches.
#### Neuromorphic Speech Enhancement with Dual-Branch Spiking Neural Networks
 - **Authors:** Taiyu Meng, Wenbin Jiang, Haoyi Zhang, Yuhan Zhou, Haibing Yin
 - **Subjects:** Subjects:
Sound (cs.SD); Artificial Intelligence (cs.AI); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2606.23761

 - **Pdf link:** https://arxiv.org/pdf/2606.23761

 - **Abstract**
 Spiking neural network (SNN)-based neuromorphic speech enhancement has emerged as a promising paradigm due to its energy efficiency, yet it still underperforms classical artificial neural network (ANN)-based approaches owing to binary activations and the lack of well-designed network architectures. To overcome this limitation, we propose a novel dual-branch spiking neural network architecture equipped with a gated spiking unit (GSU), termed GSU-DBNet. Specifically, GSU-DBNet simultaneously models the speech magnitude spectrum and complex spectrum, predicting the corresponding magnitude and complex spectral masks. Meanwhile, a dual-path GSU module is adopted to exploit temporal and frequency information for enhanced spatiotemporal feature representation. Experiments on a popular benchmark dataset show that GSU-DBNet achieves a PESQ score of 3.04 with only 394K parameters, outperforming existing SNN-based methods while using only 4.5%--10.6% of the parameters of representative ANN-based models.
#### VieSpeaker: A Large-Scale Vietnamese Speaker Recognition Dataset Beyond Visual Dependency
 - **Authors:** Viet Hoang Pham, Tran Trung Nguyen, Bao Thu Ho, Phuong Tuan Dat, Thi Thu Trang Nguyen
 - **Subjects:** Subjects:
Sound (cs.SD); Computation and Language (cs.CL); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2606.24066

 - **Pdf link:** https://arxiv.org/pdf/2606.24066

 - **Abstract**
 Speaker recognition has advanced rapidly with large-scale training datasets, yet Vietnamese remains under-resourced, with existing corpora limited in scale and acoustic diversity. Most large-scale datasets rely on facial cues to link speech with speaker identities, restricting data collection to recordings where speakers appear on camera. We propose a face-independent dataset construction pipeline and introduce VieSpeaker, a large-scale Vietnamese speaker recognition dataset. Our approach leverages textual metadata and large language model reasoning to infer speaker identities from transcripts and contextual information. VieSpeaker contains approximately 902 hours of speech from 4,715 speakers. Experiments show that models trained on VieSpeaker achieve improved robustness and generalization compared to existing Vietnamese datasets. This work demonstrates the feasibility of face-independent dataset construction and provides a new direction for building large-scale speech resources.
#### ParaPairAudioBench: Paralinguistic Pairwise Audio Benchmark for LALM-as-a-Judge
 - **Authors:** Jisu Jeon, Seungyeon Jwa, Joosung Lee, Jinhyeon Kim, Woojin Chung, Hwiyeol Jo, Jeonghoon Kim, Jonghyun Choi, Soyoon Kim
 - **Subjects:** Subjects:
Sound (cs.SD); Computation and Language (cs.CL); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2606.24648

 - **Pdf link:** https://arxiv.org/pdf/2606.24648

 - **Abstract**
 Large Audio-Language Models (LALMs) have been widely used as judge models for the automatic evaluation of generated speech. However, prior approaches predominantly focus on holistic naturalness, leaving fine-grained paralinguistic distinctions underexplored. We introduce ParaPairAudioBench, a pairwise benchmark of 5,175 audio pairs across five paralinguistic dimensions: Style, Rate, Emphasis, Age, and Gender. Our experiments show that current LALM judges still lag behind human judgments by 32%p on average and exhibit severe calibration failures, particularly in Tie cases where the correct decision is to abstain. To further analyze lexical versus acoustic reliance, the benchmark includes both same-transcript and cross-transcript conditions. ParaPairAudioBench enables multi-dimensional, calibration-aware assessment of the reliability of LALM-as-a-Judge for paralinguistic speech evaluation.
#### CN-NewsTTS Bench: a target-level automatic benchmark for raw-input Chinese news TTS pronunciation
 - **Authors:** Shijun Luo
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2606.24714

 - **Pdf link:** https://arxiv.org/pdf/2606.24714

 - **Abstract**
 Chinese news text contains dense written forms such as scores, hyphenated model names, ranges, unit symbols, percentages, English abbreviations, and mixed Chinese-Latin-digit names. These forms are frequent in real listening workflows, and a text-to-speech (TTS) system can preserve the written string while changing the spoken meaning. We introduce CN-NewsTTS Bench v0.1, an open target-level benchmark for evaluating whether Chinese news TTS products pronounce such targets correctly from raw text, without user-side rules, LLM rewriting, SSML hints, or manual edits. The release contains a 200-record development set, an 800-record public test set, 992 public auto-evaluable targets, fixed transcripts from a three-ASR ensemble, an automatic target scorer, and initial results for seven product TTS systems. We additionally report ASR-route diagnostics, ASR-subset ablations, category-level results, confidence intervals, and provider configuration metadata. The best system reaches 0.879 strict accuracy, while several systems remain below 0.60.
#### Beyond U-Net: A Latent-Representation-Aligned Skip-Free Backbone for Flow-Matching Speech Enhancement
 - **Authors:** Wangyi Pu, Michele Scarpiniti
 - **Subjects:** Subjects:
Sound (cs.SD); Artificial Intelligence (cs.AI); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2606.24745

 - **Pdf link:** https://arxiv.org/pdf/2606.24745

 - **Abstract**
 Generative models, particularly diffusion and score-based approaches, have recently achieved strong performance in speech enhancement, but their iterative sampling process limits real-time deployment. Flow Matching offers an efficient alternative by transporting noisy speech toward clean speech through an ordinary differential equation with few function evaluations. In this work, we propose a skip-free encoder-decoder backbone for flow-matching speech enhancement, guided by Latent Representation Alignment (LRA). Instead of relying on U-Net skip connections, which may transfer noise-correlated low-level features to the decoder, the proposed model aligns its bottleneck and decoder representations with clean latent features extracted from a frozen Descript Audio Codec encoder-decoder without quantization. This codec-aligned supervision promotes compact clean-speech representations while preserving efficient few-step inference. Experiments on WSJ0-CHiME3 and VoiceBank-DEMAND show improved PESQ and perceptual quality, especially on VoiceBank-DEMAND, using only five function evaluations.


by Zyzzyva0381 (Windy). 


2026-06-24
