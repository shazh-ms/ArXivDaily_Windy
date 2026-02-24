# Showing new listings for Tuesday, 24 February 2026
Auto update papers at about 2:30am UTC (10:30am Beijing time) every weekday.


阅读 `Usage.md`了解如何使用此repo实现个性化的Arxiv论文推送

See `Usage.md` for instructions on how to personalize the repo. 


Keyword list: ['text-to-speech', 'text to speech', 'tts', 'LLM-based', 'speech', 'voice']


Excluded: []


### Today: 5papers 
#### [b]=[d]-[t]+[p]: Self-supervised Speech Models Discover Phonological Vector Arithmetic
 - **Authors:** Kwanghee Choi, Eunjung Yeo, Cheol Jun Cho, David Harwath, David R. Mortensen
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Computation and Language (cs.CL); Machine Learning (cs.LG); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2602.18899

 - **Pdf link:** https://arxiv.org/pdf/2602.18899

 - **Abstract**
 Self-supervised speech models (S3Ms) are known to encode rich phonetic information, yet how this information is structured remains underexplored. We conduct a comprehensive study across 96 languages to analyze the underlying structure of S3M representations, with particular attention to phonological vectors. We first show that there exist linear directions within the model's representation space that correspond to phonological features. We further demonstrate that the scale of these phonological vectors correlate to the degree of acoustic realization of their corresponding phonological features in a continuous manner. For example, the difference between [d] and [t] yields a voicing vector: adding this vector to [p] produces [b], while scaling it results in a continuum of voicing. Together, these findings indicate that S3Ms encode speech using phonologically interpretable and compositional vectors, demonstrating phonological vector arithmetic. All code and interactive demos are available at this https URL .
#### MDM-ASR: Bridging Accuracy and Efficiency in ASR with Diffusion-Based Non-Autoregressive Decoding
 - **Authors:** Hao Yen, Pin-Jui Ku, Ante JukiÄ‡, Sabato Marco Siniscalchi
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2602.18952

 - **Pdf link:** https://arxiv.org/pdf/2602.18952

 - **Abstract**
 In sequence-to-sequence Transformer ASR, autoregressive (AR) models achieve strong accuracy but suffer from slow decoding, while non-autoregressive (NAR) models enable parallel decoding at the cost of degraded performance. We propose a principled NAR ASR framework based on Masked Diffusion Models to reduce this gap. A pre-trained speech encoder is coupled with a Transformer diffusion decoder conditioned on acoustic features and partially masked transcripts for parallel token prediction. To mitigate the training-inference mismatch, we introduce Iterative Self-Correction Training that exposes the model to its own intermediate predictions. We also design a Position-Biased Entropy-Bounded Confidence-based sampler with positional bias to further boost results. Experiments across multiple benchmarks demonstrate consistent gains over prior NAR models and competitive performance with strong AR baselines, while retaining parallel decoding efficiency.
#### CosyAccent: Duration-Controllable Accent Normalization Using Source-Synthesis Training Data
 - **Authors:** Qibing Bai, Shuhao Shi, Shuai Wang, Yukai Ju, Yannan Wang, Haizhou Li
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Artificial Intelligence (cs.AI); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2602.19166

 - **Pdf link:** https://arxiv.org/pdf/2602.19166

 - **Abstract**
 Accent normalization (AN) systems often struggle with unnatural outputs and undesired content distortion, stemming from both suboptimal training data and rigid duration modeling. In this paper, we propose a "source-synthesis" methodology for training data construction. By generating source L2 speech and using authentic native speech as the training target, our approach avoids learning from TTS artifacts and, crucially, requires no real L2 data in training. Alongside this data strategy, we introduce CosyAccent, a non-autoregressive model that resolves the trade-off between prosodic naturalness and duration control. CosyAccent implicitly models rhythm for flexibility yet offers explicit control over total output duration. Experiments show that, despite being trained without any real L2 speech, CosyAccent achieves significantly improved content preservation and superior naturalness compared to strong baselines trained on real-world data.
#### CTC-TTS: LLM-based dual-streaming text-to-speech with CTC alignment
 - **Authors:** Hanwen Liu, Saierdaer Yusuyin, Hao Huang, Zhijian Ou
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Artificial Intelligence (cs.AI); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2602.19574

 - **Pdf link:** https://arxiv.org/pdf/2602.19574

 - **Abstract**
 Large-language-model (LLM)-based text-to-speech (TTS) systems can generate natural speech, but most are not designed for low-latency dual-streaming synthesis. High-quality dual-streaming TTS depends on accurate text--speech alignment and well-designed training sequences that balance synthesis quality and latency. Prior work often relies on GMM-HMM based forced-alignment toolkits (e.g., MFA), which are pipeline-heavy and less flexible than neural aligners; fixed-ratio interleaving of text and speech tokens struggles to capture text--speech alignment regularities. We propose CTC-TTS, which replaces MFA with a CTC based aligner and introduces a bi-word based interleaving strategy. Two variants are designed: CTC-TTS-L (token concatenation along the sequence length) for higher quality and CTC-TTS-F (embedding stacking along the feature dimension) for lower latency. Experiments show that CTC-TTS outperforms fixed-ratio interleaving and MFA-based baselines on streaming synthesis and zero-shot tasks. Speech samples are available at this https URL.
#### ReHear: Iterative Pseudo-Label Refinement for Semi-Supervised Speech Recognition via Audio Large Language Models
 - **Authors:** Zefang Liu, Chenyang Zhu, Sangwoo Cho, Shi-Xiong Zhang
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2602.18721

 - **Pdf link:** https://arxiv.org/pdf/2602.18721

 - **Abstract**
 Semi-supervised learning in automatic speech recognition (ASR) typically relies on pseudo-labeling, which often suffers from confirmation bias and error accumulation due to noisy supervision. To address this limitation, we propose ReHear, a framework for iterative pseudo-label refinement that integrates an instruction-tuned, audio-aware large language model (LLM) into the self-training loop. Unlike conventional text-based correctors, our approach conditions the LLM on both the ASR hypothesis and the source audio, allowing it to recover phonetically accurate transcripts even from severe recognition errors. These refined pseudo-labels serve as high-fidelity targets for fine-tuning the ASR model in an iterative cycle. Experimental results across diverse benchmarks demonstrate that ReHear effectively mitigates error propagation, consistently outperforming both supervised and pseudo-labeling baselines.


by Zyzzyva0381 (Windy). 


2026-02-24
