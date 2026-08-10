# Showing new listings for Monday, 10 August 2026
Auto update papers at about 2:30am UTC (10:30am Beijing time) every weekday.


阅读 `Usage.md`了解如何使用此repo实现个性化的Arxiv论文推送

See `Usage.md` for instructions on how to personalize the repo. 


Keyword list: ['text-to-speech', 'text to speech', 'tts', 'LLM-based', 'speech', 'voice']


Excluded: []


### Today: 4papers 
#### LSEAD: A Privacy-Preserving LLM-Based Speech Analysis Framework for Early Alzheimer's Disease Screening
 - **Authors:** Xin Wang, Yingchao Huang, Yuhan Su, Shanshan Yao, Wei Peng
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Artificial Intelligence (cs.AI); Machine Learning (cs.LG)
 - **Arxiv link:** https://arxiv.org/abs/2608.07378

 - **Pdf link:** https://arxiv.org/pdf/2608.07378

 - **Abstract**
 Early diagnosis of Alzheimer's disease (AD) is critical for enabling timely interventions that may slow disease progression and improve patient outcomes. There is a growing need for AD detection methods that are non-invasive and cost-effective, especially in real-world clinical settings with diverse patient populations and recording conditions. Speech-based screening addresses these needs by using natural speech collected without specialized equipment. Recent advances in large language models (LLMs) have improved speech analysis by providing rich linguistic representations and strong generalization. In this study, we propose LSEAD, a speech-based AD detection framework using pretrained open-source LLMs. Speech recordings are automatically transcribed, and text embeddings are extracted using locally deployed LLMs. Principal component analysis (PCA) is applied to reduce dimensionality before classification. Because the framework relies only on speech transcripts and locally deployed models, it supports privacy-preserving AD risk assessment without external data exchange. We evaluate LSEAD on the ADReSS20 and ADReSSo2021 benchmark datasets. Experimental results show that LLM-based embeddings generalize well across datasets and improve AD classification accuracy by up to 5 percent over existing methods, especially for early-stage detection. These results demonstrate that LSEAD provides a practical, secure, and scalable approach for early AD screening.
#### SemBridge: Semantic Token Anchoring for Continuous-Latent Autoregressive Speech Generation
 - **Authors:** Hanke Xie, Haopeng Lin, Jiale Qian, Dake Guo, Yuepeng Jiang, Zhichao Wang, Wenxiao Cao, Jingbin Hu, Guobin Ma, Wenhao Li, Huakang Chen, Chengyou Wang, Ming Tao, Zhonghua Fu, Lei Xie, Xinsheng Wang
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2608.07462

 - **Pdf link:** https://arxiv.org/pdf/2608.07462

 - **Abstract**
 Continuous-latent autoregressive speech generation has emerged as a promising alternative to discrete-token modeling by avoiding quantization loss and preserving richer acoustic information. However, continuous acoustic targets do not ex- pose linguistic structure as explicit token-level prediction tar- gets. Consequently, the autoregressive language model (LM) must acquire linguistic structure indirectly through acous- tic prediction, which can compromise the content fidelity of generated speech. We propose SemBridge, a training-only semantic-token anchoring framework for continuous-latent autoregressive speech generation. SemBridge uses discrete se- mantic tokens to directly supervise autoregressive LM states and employs a Semantic-Aligned Acoustic VAE to organize the continuous target space under the same semantic refer- ence. The semantic supervision is used only during train- ing, while inference remains entirely continuous. We evalu- ate SemBridge on zero-shot text-to-speech (TTS) and score- conditioned singing voice synthesis (SVS). Across multi- ple benchmarks, SemBridge improves content accuracy, as measured by word and character error rates (WER/CER), while maintaining competitive speaker similarity and percep- tual quality. Experimental results demonstrate that explicit semantic-token supervision for autoregressive state learning is an effective and general direction for continuous speech generation. Speech samples are available.1 The model code and checkpoints will be available at this https URL lab/SemBridge
#### LILAC: An Idempotent Neural Speech Codec
 - **Authors:** June Young Yi, Dongwook Lee, Jiheum Yeom, Sungroh Yoon
 - **Subjects:** Subjects:
Sound (cs.SD); Machine Learning (cs.LG); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2608.05727

 - **Pdf link:** https://arxiv.org/pdf/2608.05727

 - **Abstract**
 Neural Audio Codecs are widely adopted in speech generation and editing. However, existing neural audio codecs are not idempotent: across the paper's twelve baseline systems, every configuration tested rewrites, on average, at least 15% of its tokens in a single decode-re-encode pass. This poses a problem for utilizing Neural Audio Codecs as token interfaces in pipelines where re-encoding decoded outputs can occur. We present LILAC, a fully convolutional 24 kHz speech codec at 9.375 Hz and 0.75 kbit/s that is codec idempotent by construction; re-encoding the decoded audio of any valid token stream returns the identical stream. LILAC achieves idempotency while maintaining competitive quality, reaching UTMOS 4.14 and 4.24 on LibriSpeech and LibriTTS-R test sets, comparable to SOTA sub-1 kbit/s Neural Audio Codecs.
#### MMAG: A Multi-Control Mixed Audio Generation Benchmark
 - **Authors:** Zihao Zheng, Xuenan Xu, Jiahao Mei, Yixuan Li, Minghao Lv, Wen Wu, Chao Zhang, Mengyue Wu
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2608.06900

 - **Pdf link:** https://arxiv.org/pdf/2608.06900

 - **Abstract**
 Recent audio generation systems have progressed from single-modality synthesis to generating complex acoustic scenes containing speech, music, and sound effects. Therefore, evaluating these models requires assessing multiple interacting capabilities, including semantic fidelity, speaker consistency, and temporal control, yet existing benchmarks focus on isolated domains or coarse-grained descriptions. To address this gap, we introduce the Multi-control Mixed Audio Generation (MMAG) benchmark. MMAG contains approximately 4,000 manually verified audio clips with rich annotations covering speech content, speaker identity, music attributes, sound events, and temporal relationships, together with dedicated subsets for voice cloning and timestamp-conditioned generation. We further propose a systematic evaluation protocol that measures acoustic fidelity, speech quality, semantic alignment, and temporal accuracy. Benchmarking representative agentic orchestrators, unified audio-visual generation models, and native mixed-audio generators reveals substantial performance trade-offs across these capabilities, with no existing model performing consistently well. Our results highlight the remaining challenges of controllable mixed audio generation and establish MMAG as a comprehensive benchmark for future research.


by Zyzzyva0381 (Windy). 


2026-08-10
