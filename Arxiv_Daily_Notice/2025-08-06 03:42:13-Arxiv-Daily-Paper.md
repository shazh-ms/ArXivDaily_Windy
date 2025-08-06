# Showing new listings for Wednesday, 6 August 2025
Auto update papers at about 2:30am UTC (10:30am Beijing time) every weekday.


阅读 `Usage.md`了解如何使用此repo实现个性化的Arxiv论文推送

See `Usage.md` for instructions on how to personalize the repo. 


Keyword list: ['text-to-speech', 'text to speech', 'tts', 'LLM-based', 'speech', 'voice']


Excluded: []


### Today: 9papers 
#### SecoustiCodec: Cross-Modal Aligned Streaming Single-Codecbook Speech Codec
 - **Authors:** Chunyu Qiang, Haoyu Wang, Cheng Gong, Tianrui Wang, Ruibo Fu, Tao Wang, Ruilong Chen, Jiangyan Yi, Zhengqi Wen, Chen Zhang, Longbiao Wang, Jianwu Dang, Jianhua Tao
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Artificial Intelligence (cs.AI); Computation and Language (cs.CL); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2508.02849

 - **Pdf link:** https://arxiv.org/pdf/2508.02849

 - **Abstract**
 Speech codecs serve as a crucial bridge in unifying speech and text language models. Existing codec methods face several challenges in semantic encoding, such as residual paralinguistic information (e.g., timbre, emotion), insufficient semantic completeness, limited reconstruction capability, and lack of support for streaming. To address these challenges, we propose SecoustiCodec, a cross-modal aligned low-bitrate streaming speech codec that disentangles semantic and paralinguistic information in a single-codebook space. To ensure semantic completeness and reconstruction fidelity, paralinguistic encoding is introduced to bridge the information gap between semantic and acoustic encoding. A semantic-only efficient quantization method based on VAE (Variational Autoencoder) and FSQ (Finite Scalar Quantization) is proposed. This approach alleviates the long-tail distribution problem of tokens while maintaining high codebook utilization. A semantic disentanglement method based on contrastive learning is proposed, which aligns text and speech in a joint multimodal frame-level space, effectively removing paralinguistic information from semantic encoding. An acoustic-constrained multi-stage optimization strategy is proposed to ensure robust and stable convergence. Figure~\ref{fig:pesq_kbps_below_2kbps} shows SecoustiCodec achieves SOTA (state-of-the-art) reconstruction quality (PESQ) of 1.77/2.58 at 0.27/1 kbps. The code and model weights for SecoustiCodec will be open-sourced upon the completion of the peer-review process. We've open-sourced SecoustiCodec's demo, code, and model weights.
#### Real-time speech enhancement in noise for throat microphone using neural audio codec as foundation model
 - **Authors:** Julien Hauret, Thomas Joubaud, Éric Bavu
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2508.02974

 - **Pdf link:** https://arxiv.org/pdf/2508.02974

 - **Abstract**
 We present a real-time speech enhancement demo using speech captured with a throat microphone. This demo aims to showcase the complete pipeline, from recording to deep learning-based post-processing, for speech captured in noisy environments with a body-conducted microphone. The throat microphone records skin vibrations, which naturally attenuate external noise, but this robustness comes at the cost of reduced audio bandwidth. To address this challenge, we fine-tune Kyutai's Mimi--a neural audio codec supporting real-time inference--on Vibravox, a dataset containing paired air-conducted and throat microphone recordings. We compare this enhancement strategy against state-of-the-art models and demonstrate its superior performance. The inference runs in an interactive interface that allows users to toggle enhancement, visualize spectrograms, and monitor processing latency.
#### Fast Algorithm for Moving Sound Source
 - **Authors:** Dong Yang
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2508.03065

 - **Pdf link:** https://arxiv.org/pdf/2508.03065

 - **Abstract**
 Modern neural network-based speech processing systems need reverberation resistance, relying on large amounts of reverberation data for training. Existing methods simulate dynamic scenarios by sampling static systems or supplement with measured data, but struggle to simulate motion data conforming to physical laws. To address insufficient training data for speech enhancement models in moving scenarios, this paper proposes Yang's motion spatio-temporal sampling reconstruction theory, enabling efficient simulation of motion-induced continuous time-varying reverberation. It breaks through the limitations of traditional static Image-Source Method (ISM) in time-varying systems by decomposing the moving image source's impulse response into linear time-invariant modulation and discrete time-varying fractional delay, establishing a physics-compliant moving sound field model. Based on the band-limited nature of motion displacement, a hierarchical sampling strategy is adopted: high sampling rates for low-order images to retain details, and low rates for high-order ones to reduce complexity, combined with a fast synthesis architecture for real-time simulation. Experiments show that compared to open-source model GSound, the theory more accurately restores amplitude and phase changes in moving scenarios, solving the industry challenge of motion sound source data simulation. It provides high-quality dynamic training data for speech enhancement models and improves the robustness of multi-channel end-to-end voice tracking algorithms.
#### PatchDSU: Uncertainty Modeling for Out of Distribution Generalization in Keyword Spotting
 - **Authors:** Bronya Roni Chernyak, Yael Segal, Yosi Shrem, Joseph Keshet
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Machine Learning (cs.LG)
 - **Arxiv link:** https://arxiv.org/abs/2508.03190

 - **Pdf link:** https://arxiv.org/pdf/2508.03190

 - **Abstract**
 Deep learning models excel at many tasks but rely on the assumption that training and test data follow the same distribution. This assumption often does not hold in real-world speech systems, where distribution shifts are common due to varying environments, recording conditions, and speaker diversity. The method of Domain Shifts with Uncertainty (DSU) augments the input of each neural network layer based on the input feature statistics. It addresses the problem of out-of-domain generalization by assuming feature statistics follow a multivariate Gaussian distribution and substitutes the input with sampled features from this distribution. While effective for computer vision, applying DSU to speech presents challenges due to the nature of the data. Unlike static visual data, speech is a temporal signal commonly represented by a spectrogram - the change of frequency over time. This representation cannot be treated as a simple image, and the resulting sparsity can lead to skewed feature statistics when applied to the entire input. To tackle out-of-distribution issues in keyword spotting, we propose PatchDSU, which extends DSU by splitting the input into patches and independently augmenting each patch. We evaluated PatchDSU and DSU alongside other methods on the Google Speech Commands, Librispeech, and TED-LIUM. Additionally, we evaluated performance under white Gaussian and MUSAN music noise conditions. We also explored out-of-domain generalization by analyzing model performance on datasets they were not trained on. Overall, in most cases, both PatchDSU and DSU outperform other methods. Notably, PatchDSU demonstrates more consistent improvements across the evaluated scenarios compared to other approaches.
#### Adaptive Knowledge Distillation for Device-Directed Speech Detection
 - **Authors:** Hyung Gun Chi, Florian Pesce, Wonil Chang, Oggi Rudovic, Arturo Argueta, Stefan Braun, Vineet Garg, Ahmed Hussen Abdelaziz
 - **Subjects:** Subjects:
Sound (cs.SD); Artificial Intelligence (cs.AI); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2508.02801

 - **Pdf link:** https://arxiv.org/pdf/2508.02801

 - **Abstract**
 Device-directed speech detection (DDSD) is a binary classification task that separates the user's queries to a voice assistant (VA) from background speech or side conversations. This is important for achieving naturalistic user experience. To this end, we propose knowledge distillation (KD) to enhance DDSD accuracy while ensuring efficient deployment. Specifically, we introduce a novel adaptive KD method that transfers knowledge from general representations of an ASR large pre-trained acoustic encoder (teacher). We apply task-specific adapters, on top of the (frozen) teacher encoder, trained jointly with the student model on DDSD. We demonstrate that the proposed adaptive KD outperforms the student model without distillation in the keyword and keyword-free (follow-up) invocations, with an improvement of +26% and +19% in terms of Equal Error Rate, respectively. We also show that this approach generalizes across the transformer and conformer-based model architectures.
#### Neural Speech Extraction with Human Feedback
 - **Authors:** Malek Itani, Ashton Graves, Sefik Emre Eskimez, Shyamnath Gollakota
 - **Subjects:** Subjects:
Sound (cs.SD); Machine Learning (cs.LG); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2508.03041

 - **Pdf link:** https://arxiv.org/pdf/2508.03041

 - **Abstract**
 We present the first neural target speech extraction (TSE) system that uses human feedback for iterative refinement. Our approach allows users to mark specific segments of the TSE output, generating an edit mask. The refinement system then improves the marked sections while preserving unmarked regions. Since large-scale datasets of human-marked errors are difficult to collect, we generate synthetic datasets using various automated masking functions and train models on each. Evaluations show that models trained with noise power-based masking (in dBFS) and probabilistic thresholding perform best, aligning with human annotations. In a study with 22 participants, users showed a preference for refined outputs over baseline TSE. Our findings demonstrate that human-in-the-loop refinement is a promising approach for improving the performance of neural speech extraction.
#### TF-MLPNet: Tiny Real-Time Neural Speech Separation
 - **Authors:** Malek Itani, Tuochao Chen, Shyamnath Gollakota
 - **Subjects:** Subjects:
Sound (cs.SD); Machine Learning (cs.LG); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2508.03047

 - **Pdf link:** https://arxiv.org/pdf/2508.03047

 - **Abstract**
 Speech separation on hearable devices can enable transformative augmented and enhanced hearing capabilities. However, state-of-the-art speech separation networks cannot run in real-time on tiny, low-power neural accelerators designed for hearables, due to their limited compute capabilities. We present TF-MLPNet, the first speech separation network capable of running in real-time on such low-power accelerators while outperforming existing streaming models for blind speech separation and target speech extraction. Our network operates in the time-frequency domain, processing frequency sequences with stacks of fully connected layers that alternate along the channel and frequency dimensions, and independently processing the time sequence at each frequency bin using convolutional layers. Results show that our mixed-precision quantization-aware trained (QAT) model can process 6 ms audio chunks in real-time on the GAP9 processor, achieving a 3.5-4x runtime reduction compared to prior speech separation models.
#### Fine-Tuning Text-to-Speech Diffusion Models Using Reinforcement Learning with Human Feedback
 - **Authors:** Jingyi Chen, Ju Seung Byun, Micha Elsner, Pichao Wang, Andrew Perrault
 - **Subjects:** Subjects:
Sound (cs.SD); Artificial Intelligence (cs.AI); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2508.03123

 - **Pdf link:** https://arxiv.org/pdf/2508.03123

 - **Abstract**
 Diffusion models produce high-fidelity speech but are inefficient for real-time use due to long denoising steps and challenges in modeling intonation and rhythm. To improve this, we propose Diffusion Loss-Guided Policy Optimization (DLPO), an RLHF framework for TTS diffusion models. DLPO integrates the original training loss into the reward function, preserving generative capabilities while reducing inefficiencies. Using naturalness scores as feedback, DLPO aligns reward optimization with the diffusion model's structure, improving speech quality. We evaluate DLPO on WaveGrad 2, a non-autoregressive diffusion-based TTS model. Results show significant improvements in objective metrics (UTMOS 3.65, NISQA 4.02) and subjective evaluations, with DLPO audio preferred 67\% of the time. These findings demonstrate DLPO's potential for efficient, high-quality diffusion TTS in real-time, resource-limited settings.
#### MiSTR: Multi-Modal iEEG-to-Speech Synthesis with Transformer-Based Prosody Prediction and Neural Phase Reconstruction
 - **Authors:** Mohammed Salah Al-Radhi, Géza Németh, Branislav Gerazov
 - **Subjects:** Subjects:
Sound (cs.SD); Machine Learning (cs.LG); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2508.03166

 - **Pdf link:** https://arxiv.org/pdf/2508.03166

 - **Abstract**
 Speech synthesis from intracranial EEG (iEEG) signals offers a promising avenue for restoring communication in individuals with severe speech impairments. However, achieving intelligible and natural speech remains challenging due to limitations in feature representation, prosody modeling, and phase reconstruction. We introduce MiSTR, a deep-learning framework that integrates: 1) Wavelet-based feature extraction to capture fine-grained temporal, spectral, and neurophysiological representations of iEEG signals, 2) A Transformer-based decoder for prosody-aware spectrogram prediction, and 3) A neural phase vocoder enforcing harmonic consistency via adaptive spectral correction. Evaluated on a public iEEG dataset, MiSTR achieves state-of-the-art speech intelligibility, with a mean Pearson correlation of 0.91 between reconstructed and original Mel spectrograms, improving over existing neural speech synthesis baselines.


by Zyzzyva0381 (Windy). 


2025-08-06
