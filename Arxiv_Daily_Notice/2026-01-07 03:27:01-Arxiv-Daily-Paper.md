# Showing new listings for Wednesday, 7 January 2026
Auto update papers at about 2:30am UTC (10:30am Beijing time) every weekday.


阅读 `Usage.md`了解如何使用此repo实现个性化的Arxiv论文推送

See `Usage.md` for instructions on how to personalize the repo. 


Keyword list: ['text-to-speech', 'text to speech', 'tts', 'LLM-based', 'speech', 'voice']


Excluded: []


### Today: 8papers 
#### Vclip: Face-based Speaker Generation by Face-voice Association Learning
 - **Authors:** Yao Shi, Yunfei Xu, Hongbin Suo, Yulong Wan, Haifeng Liu
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2601.02753

 - **Pdf link:** https://arxiv.org/pdf/2601.02753

 - **Abstract**
 This paper discusses the task of face-based speech synthesis, a kind of personalized speech synthesis where the synthesized voices are con- strained to perceptually match with a reference face image. Due to the lack of TTS-quality audio-visual corpora, previous approaches suffer from either low synthesis quality or domain mismatch induced by a knowledge transfer scheme. This paper proposes a new approach called Vclip that utilizes the facial-semantic knowledge of the CLIP encoder on noisy audio-visual data to learn the association between face and voice efficiently, achieving 89.63% cross-modal verification AUC score on Voxceleb testset. The proposed method then uses a retrieval-based strategy, combined with GMM-based speaker generation module for a downstream TTS system, to produce probable target speakers given reference images. Experimental results demonstrate that the proposed Vclip system in conjunction with the retrieval step can bridge the gap between face and voice features for face-based speech synthesis. And using the feedback information distilled from downstream TTS helps to synthesize voices that match closely with reference faces. Demos available at this http URL.
#### XLSR-MamBo: Scaling the Hybrid Mamba-Attention Backbone for Audio Deepfake Detection
 - **Authors:** Kwok-Ho Ng, Tingting Song, Yongdong WU, Zhihua Xia
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2601.02944

 - **Pdf link:** https://arxiv.org/pdf/2601.02944

 - **Abstract**
 Advanced speech synthesis technologies have enabled highly realistic speech generation, posing security risks that motivate research into audio deepfake detection (ADD). While state space models (SSMs) offer linear complexity, pure causal SSMs architectures often struggle with the content-based retrieval required to capture global frequency-domain artifacts. To address this, we explore the scaling properties of hybrid architectures by proposing XLSR-MamBo, a modular framework integrating an XLSR front-end with synergistic Mamba-Attention backbones. We systematically evaluate four topological designs using advanced SSM variants, Mamba, Mamba2, Hydra, and Gated DeltaNet. Experimental results demonstrate that the MamBo-3-Hydra-N3 configuration achieves competitive performance compared to other state-of-the-art systems on the ASVspoof 2021 LA, DF, and In-the-Wild benchmarks. This performance benefits from Hydra's native bidirectional modeling, which captures holistic temporal dependencies more efficiently than the heuristic dual-branch strategies employed in prior works. Furthermore, evaluations on the DFADD dataset demonstrate robust generalization to unseen diffusion- and flow-matching-based synthesis methods. Crucially, our analysis reveals that scaling backbone depth effectively mitigates the performance variance and instability observed in shallower models. These results demonstrate the hybrid framework's ability to capture artifacts in spoofed speech signals, providing an effective method for ADD.
#### Towards Fine-Grained and Multi-Granular Contrastive Language-Speech Pre-training
 - **Authors:** Yifan Yang, Bing Han, Hui Wang, Wei Wang, Ziyang Ma, Long Zhou, Zengrui Jin, Guanrou Yang, Tianrui Wang, Xu Tan, Xie Chen
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2601.03065

 - **Pdf link:** https://arxiv.org/pdf/2601.03065

 - **Abstract**
 Modeling fine-grained speaking styles remains challenging for language-speech representation pre-training, as existing speech-text models are typically trained with coarse captions or task-specific supervision, and scalable fine-grained style annotations are unavailable. We present FCaps, a large-scale dataset with fine-grained free-text style descriptions, encompassing 47k hours of speech and 19M fine-grained captions annotated via a novel end-to-end pipeline that directly grounds detailed captions in audio, thereby avoiding the error propagation caused by LLM-based rewriting in existing cascaded pipelines. Evaluations using LLM-as-a-judge demonstrate that our annotations surpass existing cascaded annotations in terms of correctness, coverage, and naturalness. Building on FCaps, we propose CLSP, a contrastive language-speech pre-trained model that integrates global and fine-grained supervision, enabling unified representations across multiple granularities. Extensive experiments demonstrate that CLSP learns fine-grained and multi-granular speech-text representations that perform reliably across global and fine-grained speech-text retrieval, zero-shot paralinguistic classification, and speech style similarity scoring, with strong alignment to human judgments. All resources will be made publicly available.
#### WearVox: An Egocentric Multichannel Voice Assistant Benchmark for Wearables
 - **Authors:** Zhaojiang Lin, Yong Xu, Kai Sun, Jing Zheng, Yin Huang, Surya Teja Appini, Krish Narang, Renjie Tao, Ishan Kapil Jain, Siddhant Arora, Ruizhi Li, Yiteng Huang, Kaushik Patnaik, Wenfang Xu, Suwon Shon, Yue Liu, Ahmed A Aly, Anuj Kumar, Florian Metze, Xin Luna Dong
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2601.02391

 - **Pdf link:** https://arxiv.org/pdf/2601.02391

 - **Abstract**
 Wearable devices such as AI glasses are transforming voice assistants into always-available, hands-free collaborators that integrate seamlessly with daily life, but they also introduce challenges like egocentric audio affected by motion and noise, rapid micro-interactions, and the need to distinguish device-directed speech from background conversations. Existing benchmarks largely overlook these complexities, focusing instead on clean or generic conversational audio. To bridge this gap, we present WearVox, the first benchmark designed to rigorously evaluate voice assistants in realistic wearable scenarios. WearVox comprises 3,842 multi-channel, egocentric audio recordings collected via AI glasses across five diverse tasks including Search-Grounded QA, Closed-Book QA, Side-Talk Rejection, Tool Calling, and Speech Translation, spanning a wide range of indoor and outdoor environments and acoustic conditions. Each recording is accompanied by rich metadata, enabling nuanced analysis of model performance under real-world constraints. We benchmark leading proprietary and open-source speech Large Language Models (SLLMs) and find that most real-time SLLMs achieve accuracies on WearVox ranging from 29% to 59%, with substantial performance degradation on noisy outdoor audio, underscoring the difficulty and realism of the benchmark. Additionally, we conduct a case study with two new SLLMs that perform inference with single-channel and multi-channel audio, demonstrating that multi-channel audio inputs significantly enhance model robustness to environmental noise and improve discrimination between device-directed and background speech. Our results highlight the critical importance of spatial audio cues for context-aware voice assistants and establish WearVox as a comprehensive testbed for advancing wearable voice AI research.
#### Quantifying Quanvolutional Neural Networks Robustness for Speech in Healthcare Applications
 - **Authors:** Ha Tran, Bipasha Kashyap, Pubudu N. Pathirana
 - **Subjects:** Subjects:
Sound (cs.SD); Machine Learning (cs.LG); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2601.02432

 - **Pdf link:** https://arxiv.org/pdf/2601.02432

 - **Abstract**
 Speech-based machine learning systems are sensitive to noise, complicating reliable deployment in emotion recognition and voice pathology detection. We evaluate the robustness of a hybrid quantum machine learning model, quanvolutional neural networks (QNNs) against classical convolutional neural networks (CNNs) under four acoustic corruptions (Gaussian noise, pitch shift, temporal shift, and speed variation) in a clean-train/corrupted-test regime. Using AVFAD (voice pathology) and TESS (speech emotion), we compare three QNN models (Random, Basic, Strongly) to a simple CNN baseline (CNN-Base), ResNet-18 and VGG-16 using accuracy and corruption metrics (CE, mCE, RCE, RmCE), and analyze architectural factors (circuit complexity or depth, convergence) alongside per-emotion robustness. QNNs generally outperform the CNN-Base under pitch shift, temporal shift, and speed variation (up to 22% lower CE/RCE at severe temporal shift), while the CNN-Base remains more resilient to Gaussian noise. Among quantum circuits, QNN-Basic achieves the best overall robustness on AVFAD, and QNN-Random performs strongest on TESS. Emotion-wise, fear is most robust (80-90% accuracy under severe corruptions), neutral can collapse under strong Gaussian noise (5.5% accuracy), and happy is most vulnerable to pitch, temporal, and speed distortions. QNNs also converge up to six times faster than the CNN-Base. To our knowledge, this is a systematic study of QNN robustness for speech under common non-adversarial acoustic corruptions, indicating that shallow entangling quantum front-ends can improve noise resilience while sensitivity to additive noise remains a challenge.
#### VocalBridge: Latent Diffusion-Bridge Purification for Defeating Perturbation-Based Voiceprint Defenses
 - **Authors:** Maryam Abbasihafshejani, AHM Nazmus Sakib, Murtuza Jadliwala
 - **Subjects:** Subjects:
Sound (cs.SD); Artificial Intelligence (cs.AI); Cryptography and Security (cs.CR); Machine Learning (cs.LG); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2601.02444

 - **Pdf link:** https://arxiv.org/pdf/2601.02444

 - **Abstract**
 The rapid advancement of speech synthesis technologies, including text-to-speech (TTS) and voice conversion (VC), has intensified security and privacy concerns related to voice cloning. Recent defenses attempt to prevent unauthorized cloning by embedding protective perturbations into speech to obscure speaker identity while maintaining intelligibility. However, adversaries can apply advanced purification techniques to remove these perturbations, recover authentic acoustic characteristics, and regenerate cloneable voices. Despite the growing realism of such attacks, the robustness of existing defenses under adaptive purification remains insufficiently studied. Most existing purification methods are designed to counter adversarial noise in automatic speech recognition (ASR) systems rather than speaker verification or voice cloning pipelines. As a result, they fail to suppress the fine-grained acoustic cues that define speaker identity and are often ineffective against speaker verification attacks (SVA). To address these limitations, we propose Diffusion-Bridge (VocalBridge), a purification framework that learns a latent mapping from perturbed to clean speech in the EnCodec latent space. Using a time-conditioned 1D U-Net with a cosine noise schedule, the model enables efficient, transcript-free purification while preserving speaker-discriminative structure. We further introduce a Whisper-guided phoneme variant that incorporates lightweight temporal guidance without requiring ground-truth transcripts. Experimental results show that our approach consistently outperforms existing purification methods in recovering cloneable voices from protected speech. Our findings demonstrate the fragility of current perturbation-based defenses and highlight the need for more robust protection mechanisms against evolving voice-cloning and speaker verification threats.
#### Dynamic Quantization Error Propagation in Encoder-Decoder ASR Quantization
 - **Authors:** Xinyu Wang, Yajie Luo, Yihong Wu, Liheng Ma, Ziyu Zhao, Jingrui Tian, Lei Ding, Yufei Cui, Xiao-Wen Chang
 - **Subjects:** Subjects:
Sound (cs.SD); Computation and Language (cs.CL); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2601.02455

 - **Pdf link:** https://arxiv.org/pdf/2601.02455

 - **Abstract**
 Running Automatic Speech Recognition (ASR) models on memory-constrained edge devices requires efficient compression. While layer-wise post-training quantization is effective, it suffers from error accumulation, especially in encoder-decoder architectures. Existing solutions like Quantization Error Propagation (QEP) are suboptimal for ASR due to the model's heterogeneity, processing acoustic features in the encoder while generating text in the decoder. To address this, we propose Fine-grained Alpha for Dynamic Quantization Error Propagation (FADE), which adaptively controls the trade-off between cross-layer error correction and local quantization. Experiments show that FADE significantly improves stability by reducing performance variance across runs, while simultaneously surpassing baselines in mean WER.
#### MoE Adapter for Large Audio Language Models: Sparsity, Disentanglement, and Gradient-Conflict-Free
 - **Authors:** Yishu Lei, Shuwei He, Jing Hu, Dan Zhang, Xianlong Luo, Danxiang Zhu, Shikun Feng, Rui Liu, Jingzhou He, Yu Sun, Hua Wu, Haifeng Wang
 - **Subjects:** Subjects:
Sound (cs.SD); Artificial Intelligence (cs.AI); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2601.02967

 - **Pdf link:** https://arxiv.org/pdf/2601.02967

 - **Abstract**
 Extending the input modality of Large Language Models~(LLMs) to the audio domain is essential for achieving comprehensive multimodal perception. However, it is well-known that acoustic information is intrinsically \textit{heterogeneous}, entangling attributes such as speech, music, and environmental context. Existing research is limited to a dense, parameter-shared adapter to model these diverse patterns, which induces \textit{gradient conflict} during optimization, as parameter updates required for distinct attributes contradict each other. To address this limitation, we introduce the \textit{\textbf{MoE-Adapter}}, a sparse Mixture-of-Experts~(MoE) architecture designed to decouple acoustic information. Specifically, it employs a dynamic gating mechanism that routes audio tokens to specialized experts capturing complementary feature subspaces while retaining shared experts for global context, thereby mitigating gradient conflicts and enabling fine-grained feature learning. Comprehensive experiments show that the MoE-Adapter achieves superior performance on both audio semantic and paralinguistic tasks, consistently outperforming dense linear baselines with comparable computational costs. Furthermore, we will release the related code and models to facilitate future research.


by Zyzzyva0381 (Windy). 


2026-01-07
