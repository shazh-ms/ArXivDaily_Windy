# Showing new listings for Wednesday, 15 October 2025
Auto update papers at about 2:30am UTC (10:30am Beijing time) every weekday.


阅读 `Usage.md`了解如何使用此repo实现个性化的Arxiv论文推送

See `Usage.md` for instructions on how to personalize the repo. 


Keyword list: ['text-to-speech', 'text to speech', 'tts', 'LLM-based', 'speech', 'voice']


Excluded: []


### Today: 5papers 
#### FakeMark: Deepfake Speech Attribution With Watermarked Artifacts
 - **Authors:** Wanying Ge, Xin Wang, Junichi Yamagishi
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2510.12042

 - **Pdf link:** https://arxiv.org/pdf/2510.12042

 - **Abstract**
 Deepfake speech attribution remains challenging for existing solutions. Classifier-based solutions often fail to generalize to domain-shifted samples, and watermarking-based solutions are easily compromised by distortions like codec compression or malicious removal attacks. To address these issues, we propose FakeMark, a novel watermarking framework that injects artifact-correlated watermarks associated with deepfake systems rather than pre-assigned bitstring messages. This design allows a detector to attribute the source system by leveraging both injected watermark and intrinsic deepfake artifacts, remaining effective even if one of these cues is elusive or removed. Experimental results show that FakeMark improves generalization to cross-dataset samples where classifier-based solutions struggle and maintains high accuracy under various distortions where conventional watermarking-based solutions fail.
#### DiSTAR: Diffusion over a Scalable Token Autoregressive Representation for Speech Generation
 - **Authors:** Yakun Song, Xiaobin Zhuang, Jiawei Chen, Zhikang Niu, Guanrou Yang, Chenpeng Du, Zhuo Chen, Yuping Wang, Yuxuan Wang, Xie Chen
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Computation and Language (cs.CL); Machine Learning (cs.LG)
 - **Arxiv link:** https://arxiv.org/abs/2510.12210

 - **Pdf link:** https://arxiv.org/pdf/2510.12210

 - **Abstract**
 Recent attempts to interleave autoregressive (AR) sketchers with diffusion-based refiners over continuous speech representations have shown promise, but they remain brittle under distribution shift and offer limited levers for controllability. We introduce DISTAR, a zero-shot text-to-speech framework that operates entirely in a discrete residual vector quantization (RVQ) code space and tightly couples an AR language model with a masked diffusion model, without forced alignment or a duration predictor. Concretely, DISTAR drafts block-level RVQ tokens with an AR language model and then performs parallel masked-diffusion infilling conditioned on the draft to complete the next block, yielding long-form synthesis with blockwise parallelism while mitigating classic AR exposure bias. The discrete code space affords explicit control at inference: DISTAR produces high-quality audio under both greedy and sample-based decoding using classifier-free guidance, supports trade-offs between robustness and diversity, and enables variable bit-rate and controllable computation via RVQ layer pruning at test time. Extensive experiments and ablations demonstrate that DISTAR surpasses state-of-the-art zero-shot TTS systems in robustness, naturalness, and speaker/style consistency, while maintaining rich output diversity. Audio samples are provided on this https URL.
#### A Phase Synthesizer for Decorrelation to Improve Acoustic Feedback Cancellation
 - **Authors:** Klaus Linhard, Philipp Bulling
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2510.12377

 - **Pdf link:** https://arxiv.org/pdf/2510.12377

 - **Abstract**
 Undesired acoustic feedback is a known issue in communication systems, such as speech in-car communication, public address systems, or hearing aids. Without additional precautions, there is a high risk that the adaptive filter - intended to cancel the feedback path - also suppresses parts of the desired signal. One solution is to decorrelate the loudspeaker and microphone signals. In this work, we combine the two decorrelation approaches frequency shifting and phase modulation in a unified framework: a so-called \textit{phase synthesizer}, implemented in a discrete Fourier transform (DFT) filter bank. Furthermore, we extend the phase modulation technique using variable delay lines, as known from vibrato and chorus effects. We demonstrate the benefits of the proposed phase synthesizer using an example from speech in-car communication, employing an adaptive frequency-domain Kalman filter. Improvements in system stability, speech quality measured by perceptual evaluation of speech quality (PESQ) are presented.
#### I-DCCRN-VAE: An Improved Deep Representation Learning Framework for Complex VAE-based Single-channel Speech Enhancement
 - **Authors:** Jiatong Li, Simon Doclo
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2510.12485

 - **Pdf link:** https://arxiv.org/pdf/2510.12485

 - **Abstract**
 Recently, a complex variational autoencoder (VAE)-based single-channel speech enhancement system based on the DCCRN architecture has been proposed. In this system, a noise suppression VAE (NSVAE) learns to extract clean speech representations from noisy speech using pretrained clean speech and noise VAEs with skip connections. In this paper, we improve DCCRN-VAE by incorporating three key modifications: 1) removing the skip connections in the pretrained VAEs to encourage more informative speech and noise latent representations; 2) using $\beta$-VAE in pretraining to better balance reconstruction and latent space regularization; and 3) a NSVAE generating both speech and noise latent representations. Experiments show that the proposed system achieves comparable performance as the DCCRN and DCCRN-VAE baselines on the matched DNS3 dataset but outperforms the baselines on mismatched datasets (WSJ0-QUT, Voicebank-DEMEND), demonstrating improved generalization ability. In addition, an ablation study shows that a similar performance can be achieved with classical fine-tuning instead of adversarial training, resulting in a simpler training pipeline.
#### Serial-Parallel Dual-Path Architecture for Speaking Style Recognition
 - **Authors:** Guojian Li, Qijie Shao, Zhixian Zhao, Shuiyuan Wang, Zhonghua Fu, Lei Xie
 - **Subjects:** Subjects:
Sound (cs.SD); Artificial Intelligence (cs.AI); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2510.11732

 - **Pdf link:** https://arxiv.org/pdf/2510.11732

 - **Abstract**
 Speaking Style Recognition (SSR) identifies a speaker's speaking style characteristics from speech. Existing style recognition approaches primarily rely on linguistic information, with limited integration of acoustic information, which restricts recognition accuracy improvements. The fusion of acoustic and linguistic modalities offers significant potential to enhance recognition performance. In this paper, we propose a novel serial-parallel dual-path architecture for SSR that leverages acoustic-linguistic bimodal information. The serial path follows the ASR+STYLE serial paradigm, reflecting a sequential temporal dependency, while the parallel path integrates our designed Acoustic-Linguistic Similarity Module (ALSM) to facilitate cross-modal interaction with temporal simultaneity. Compared to the existing SSR baseline -- the OSUM model, our approach reduces parameter size by 88.4% and achieves a 30.3% improvement in SSR accuracy for eight styles on the test set.


by Zyzzyva0381 (Windy). 


2025-10-15
