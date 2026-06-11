# Showing new listings for Thursday, 11 June 2026
Auto update papers at about 2:30am UTC (10:30am Beijing time) every weekday.


阅读 `Usage.md`了解如何使用此repo实现个性化的Arxiv论文推送

See `Usage.md` for instructions on how to personalize the repo. 


Keyword list: ['text-to-speech', 'text to speech', 'tts', 'LLM-based', 'speech', 'voice']


Excluded: []


### Today: 11papers 
#### MA-DLE: Speech-based Automatic Depression Level Estimation via Memory Augmentation
 - **Authors:** Xuzhi Wang, Xinran Wu, Ziping Zhao, Jianhua Tao, Björn W. Schuller
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Artificial Intelligence (cs.AI); Computation and Language (cs.CL); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2606.11197

 - **Pdf link:** https://arxiv.org/pdf/2606.11197

 - **Abstract**
 Speech-based automatic estimation of depression levels is essential for enabling early detection and timely intervention, particularly in resource-constrained mental health settings. In recent years, deep learning has demonstrated impressive success across various domains, including affective computing and mental health assessment. Most existing approaches rely on RNN-based architectures (such as LSTM and GRU) to model temporal information for depression estimation. However, the extracted features often emphasize only a few adjacent speech segments, limiting their ability to capture long-range dependencies. To overcome this limitation, we introduce a memory-based feature augmentation method that enhances the representational capacity of GRU-extracted features. Rather than indiscriminately incorporating historical data, our memory bank is designed to selectively integrate two types of components in order to reduce redundancy and irrelevance: (1) historical temporal features that closely resemble the current GRU output, offering complementary contextual information; and (2) dynamic memory features identified based on feature variability, which capture behavioral and emotional fluctuations indicative of depressive symptoms. To effectively fuse the memory-augmented features with GRU outputs, we further design a Hierarchical Attention Fusion (HAF) module. Our method is evaluated on the widely used DAIC-WOZ and E-DAIC datasets, achieving state-of-the-art performance.
#### Massive Open-Vocabulary Keyword Spotting
 - **Authors:** Leonor Barreiros, Raul Monteiro, Afonso Mendes, Gonçalo M. Correia
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Computation and Language (cs.CL); Machine Learning (cs.LG); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2606.11279

 - **Pdf link:** https://arxiv.org/pdf/2606.11279

 - **Abstract**
 Automatic speech recognition systems have been shown to under-perform when it comes to transcribing words rarely seen in the training data, namely specialized terminology. Open-vocabulary keyword spotting, combined with contextual biasing, has been shown to mitigate this issue. However, existing systems can only handle glossaries of a few hundred terms without becoming an infeasible bottleneck. We propose a system that stores features with a memory footprint up to 128 times smaller than a comparable baseline and allows users to process massive databases while remaining open-vocabulary. Without fine-tuning the speech recognition model, our system achieves a comparable entity recall as uncompressed solutions, even in languages not seen during training.
#### Gumbel-BEARD: Automatic Layer Selection for Self-Supervised Adaptation of Whisper in Low-Resource Domains
 - **Authors:** Zilai Wang, Natarajan Balaji Shankar, Mohan Shi, Kaiyuan Zhang, Abeer Alwan
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Computation and Language (cs.CL); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2606.11429

 - **Pdf link:** https://arxiv.org/pdf/2606.11429

 - **Abstract**
 Speech foundation models often struggle in low-resource domains due to domain mismatch and data scarcity. We propose Gumbel-BEARD, a domain adaptation framework that automates Whisper encoder layer selection via an end-to-end trainable hard Gumbel-Softmax selector. It enables self-supervised adaptation with a BEST-RQ objective that dynamically adapts to target acoustic characteristics without manual tuning. Experiments on the MyST child speech corpus demonstrate efficiency and scalability: with 10 h of labeled data for fine-tuning, our method matches a fully supervised baseline trained on the complete 133 h labeled set. We establish new state-of-the-art word error rates (WERs) of 8.21% using Whisper-medium on MyST and 11.06% using Whisper-small on the OGI Spontaneous dataset. Evaluation on CORAAL further confirms robustness to adult dialectal domain shifts, with up to 6% relative WER reduction, highlighting the generalizability of our approach to diverse low-resource conditions.
#### Benchmarking Neural Speech Compression from a Rate-Distortion Perspective
 - **Authors:** Jun Xu, Zhengxue Cheng, Fengxi Zhang, Yuhan Liu, Li Song, Wenjun Zhang
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2606.11631

 - **Pdf link:** https://arxiv.org/pdf/2606.11631

 - **Abstract**
 Learning-based speech compression has achieved promising low-bitrate performance, but many neural speech codecs still describe quantized latents with preset-rate discrete symbols or apply entropy coding only after symbol generation. Such designs decouple representation learning from probability modeling, limiting their ability to exploit the non-uniform usage and temporal dependencies of learned speech latents. In this paper, we benchmark neural speech compression from a rate--distortion perspective and further investigate entropy-constrained coding for low-bitrate speech compression. We first formulate a unified learning-based speech coding pipeline and provide a benchmark-style analysis of recent neural speech codecs, showing that explicit probability modeling remains underexplored in learned speech compression. We then propose ECC, an Entropy-Constrained Codec that combines scalar quantization with a learned entropy model. ECC integrates hyperprior-based side information, channel-wise context modeling, latent residual prediction, and lightweight temporal modeling to estimate latent likelihoods for rate estimation during training and arithmetic coding during inference. To further improve low-bitrate efficiency, ECC introduces entropy skip, which omits highly predictable residual symbols using decoder-available scale estimates without transmitting additional skip masks. Extensive experiments show that ECC achieves a favorable low-bitrate rate--distortion trade-off over conventional and neural codec baselines, reducing BD-rate by 39.9% on ViSQOL and 76.3% on PESQ on average over two widely-used test sets. Ablation and diagnostic studies further validate the effectiveness of entropy modeling. Project Page: this https URL
#### Fast Speech Foundation Model Distillation Using Interleaved Stacking
 - **Authors:** Eungbeom Kim, Kyogu Lee
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Artificial Intelligence (cs.AI); Computation and Language (cs.CL); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2606.11766

 - **Pdf link:** https://arxiv.org/pdf/2606.11766

 - **Abstract**
 Distilling a large speech foundation model (SFM) into an efficient student model has been successfully applied to low-resource environments. Although distillation reduces inference latency, it requires an additional student model training. However, the training efficiency of SFM distillation remains underexplored. In this work, we explore training acceleration of SFM distillation to speed up model deployment. We examine the potential of stacking, in which the model depth is progressively increased through training until the target model depth is reached. While existing stacking methods improve training speed, they suffer from performance degradation. To handle this limitation, we propose interleaved stacking, a novel stacking method that consistently preserves layer position throughout the stacking process. This property is particularly critical in SFMs, in which each layer encodes distinct layer-specific knowledge. We validate the effectiveness of the proposed method on SUPERB.
#### Tight Boundary Prediction in Speaker Diarization Using Causal-Anticausal Consistency
 - **Authors:** Shota Horiguchi, Marc Delcroix, Naohiro Tawara, Takanori Ashihara, Atsushi Ando
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2606.11795

 - **Pdf link:** https://arxiv.org/pdf/2606.11795

 - **Abstract**
 Multi-talker conversational automatic speech recognition data are often used to train speaker diarization models. Because such data prioritize semantic continuity, pauses and boundary margins are included within speech segments, resulting in loose annotations. Models trained on such data tend to internalize mechanisms that reproduce this looseness, although tight speech intervals are sometimes preferable for downstream applications. In this paper, we address the novel task of enabling models to produce tight predictions using loose labels. Our method generates tighter pseudo labels using causal and anticausal models, which are inherently incapable of learning loosening behavior. We further propose a co-training scheme that iteratively tightens labels and updates both models for more progressive refinement. Experimental results show that the proposed method recovers about 70 % of the tightening effect achieved by ideal tight-label training and improves downstream performance.
#### Which Speech Representation Better Matches Text-Native Reasoning? A Study of Speech-Text Alignment on Frame Rate and Representation
 - **Authors:** Zhen Ye, Xu Tan, Yiming Li, Guangyan Zhang, Chimin Chan, Haohe Liu, Zhengxi Liu, Hongzhan Lin, Zheqi Dai, Xinshen Zhang, Peiwen Sun, Qiuqiang Kong, Wei Xue
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Computation and Language (cs.CL); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2606.12199

 - **Pdf link:** https://arxiv.org/pdf/2606.12199

 - **Abstract**
 Spoken dialogue models typically start from text LLM backbones, yet reasoning often degrades when conditioning on speech instead of text. We attribute part of this modality gap to a temporal-granularity mismatch: speech tokens are temporally redundant and far longer than text under matched semantics, diluting per-token semantic density and weakening text-native reasoning dynamics. We study speech token design as a representation selection problem and sweep frame rates under a frozen LLM backbone with a fixed information rate. To make low frame rates feasible, we introduce factorized FSQ and a lightweight non-autoregressive audio LM head, scaling capacity to nearly 300\,bits/frame without sacrificing efficient prediction. With the bottleneck removed, we sweep frame rates (50$\rightarrow$2.08\,Hz) and alignment depth, and observe a consistent best regime for speech QA at 4.17\,Hz with intermediate-layer representation alignment.
#### HALO: Half-Frame-Rate Adaptive Learnable Operator for Lightweight STFT-Based Speech Enhancement
 - **Authors:** Jiadong Zhao, Dahan Wang, Yu Sun, Leyan Yang, Xiaobin Rong, Shiruo Sun, Yuxiang Hu, Jing Lu
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2606.12328

 - **Pdf link:** https://arxiv.org/pdf/2606.12328

 - **Abstract**
 STFT-based speech enhancement typically adopts overlapping analysis frames. While overlap is essential for stable STFT processing, it makes adjacent frames highly correlated, causing redundant computation in lightweight models. We propose Half-frame-rate Adaptive Learnable Operator (HALO), a causal plug-in module that halves the internal frame rate without altering the STFT procedure. Broadly applicable to many lightweight models, HALO applies adaptive rate reduction before the backbone and restoration afterward, reconstructing the full-rate spectrum on the original STFT grid. Both reduction and restoration are implemented with lightweight dynamic convolutions. By halving the processed frame rate, HALO reduces backbone compute cost with no added algorithmic latency, freeing budget for channel widening. Experiments on the DNS3 dataset show consistent gains across diverse lightweight models under matched complexity, demonstrating the effectiveness of reducing overlap-induced redundancy.
#### The Dynamics of Human and AI-Generated Language: How Semantics Fluctuates across Different Timescales
 - **Authors:** Han-Jen Chang, Yasir Çatal, Angelika Wolman, Agustín Ibáñez, David Smith, I-Wen Su, Kai-Yuan Cheng, Georg Northoff
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Artificial Intelligence (cs.AI); Audio and Speech Processing (eess.AS); Signal Processing (eess.SP)
 - **Arxiv link:** https://arxiv.org/abs/2606.11371

 - **Pdf link:** https://arxiv.org/pdf/2606.11371

 - **Abstract**
 Spoken language, whether produced by humans or large language models (LLM), unfolds over time with varying semantic content. However, we still lack simple, interpretable time-series features that capture how generic versus specific content is distributed over time, and that can be used to compare human and AI-generated speech. We introduce a semantic-timescale analysis pipeline that turns word-level transcripts with timestamps into semantic time-series. For each spoken narrative, we compute (i) semantic specificity using WordNet-based word depth and (ii) contextual similarity using SBERT embeddings and quantify their temporal dependence using autocorrelation-window measures (ACW-0 and related metrics). We then compare original speech to multiple shuffled controls that selectively disrupt lexical identity, temporal order, and word duration. Across human-read autobiographical narratives, TTS readings, and LLM-generated texts rendered with TTS, we find that segments with longer ACW-0 in the semantic time-series tend to contain more generic vocabulary, whereas segments with shorter ACW-0 are enriched in more specific words. These associations are strongly attenuated or abolished when word order and timing are randomized, indicating that ACW-based measures capture non-trivial temporal organization of semantic content beyond static lexical distributions. Our results suggest that ACW-based semantic timescales are a useful family of features for analyzing and comparing the temporal structure of human and AI-generated speech.
#### Overcoming State Inertia in Full-Duplex Spoken Language Models via Activation Steering
 - **Authors:** Cheng-Kuang Chang, Kai-Wei Chang, Alexander H. Liu, James Glass
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Artificial Intelligence (cs.AI); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2606.11386

 - **Pdf link:** https://arxiv.org/pdf/2606.11386

 - **Abstract**
 Full-duplex spoken language models (FD-SLMs) enable seamless speech interaction by allowing models to listen and speak simultaneously, yet the internal mechanism by which they coordinate listening and speaking remains underexplored. We analyze the predictive behavior encoded in FD-SLM hidden representations and find that they exhibit stream-specific predictive patterns: during listening, they preferentially predict the incoming user stream, whereas during speaking, they preferentially predict the model output stream. Building on this observation, we show that FD-SLMs dynamically modulate their internal predictive focus between two states: a generative state aligned with model output generation and a perceptive state aligned with incoming user input. However, this modulation can lag behind abrupt changes in conversational context. During user interruptions, the model remains transiently biased toward the generative state before transitioning into the perceptive state, causing it to miss the beginning of the incoming input. We term this delayed internal transition state inertia. To quantify its downstream impact, we introduce the Zero-Buffer Benchmark (ZBB), a diagnostic benchmark for evaluating immediate interruption comprehension when user speech begins abruptly. We evaluate this setting using response correctness and initial-word occurrence rate (IWOR). Finally, we mitigate state inertia through activation steering with a perception vector, a training-free intervention with little additional computational overhead. Across multiple state-of-the-art FD-SLMs, activation steering substantially improves interruption handling; for example, on PersonaPlex, it improves correctness from 28% to 45% and IWOR from 40% to 72% without any fine-tuning.
#### Towards Data-free and Training-free Compression for Speech Foundation Models Using Parameter Clustering
 - **Authors:** Haoning Xu, Zhaoqing Li, Huimeng Wang, Youjun Chen, Chengxi Deng, Mengzhe Geng, Xunying Liu
 - **Subjects:** Subjects:
Sound (cs.SD); Artificial Intelligence (cs.AI); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2606.11836

 - **Pdf link:** https://arxiv.org/pdf/2606.11836

 - **Abstract**
 This paper presents a novel data-free and training-free compression approach for speech foundation models using channelwise clustering via k-means. More fine-grained, mixed sparsity pruning by layer-level varying number of parameter clusters is also explored. Experiments conducted on the LibriSpeech dataset suggest that when operating with pruning sparsity of 50% on HuBERT-large, consistent WER reductions of 27.73%/18.61% absolute (34.37%/21.91% relative) over the magnitude-based pruning were obtained on the test-clean and test-other subsets before fine-tuning and 0.19%/0.79% absolute (3.36%/4.62% relative) after fine-tuning with only 3 epochs. Similar WER reductions of 2.86%/5.02% absolute (59.21%/55.29% relative) were observed against magnitudebased pruning on Whisper-large-v3 at 10% sparsity, all with no significant WER increase relative to the uncompressed baseline.


by Zyzzyva0381 (Windy). 


2026-06-11
