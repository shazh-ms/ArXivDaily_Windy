# Showing new listings for Wednesday, 15 April 2026
Auto update papers at about 2:30am UTC (10:30am Beijing time) every weekday.


阅读 `Usage.md`了解如何使用此repo实现个性化的Arxiv论文推送

See `Usage.md` for instructions on how to personalize the repo. 


Keyword list: ['text-to-speech', 'text to speech', 'tts', 'LLM-based', 'speech', 'voice']


Excluded: []


### Today: 8papers 
#### StreamMark: A Deep Learning-Based Semi-Fragile Audio Watermarking for Proactive Deepfake Detection
 - **Authors:** Zhentao Liu, Milos Cernak
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2604.11917

 - **Pdf link:** https://arxiv.org/pdf/2604.11917

 - **Abstract**
 The rapid advancement of generative AI has made it increasingly challenging to distinguish between deepfake audio and authentic human speech. To overcome the limitations of passive detection methods, we propose StreamMark, a novel deep learning-based, semi-fragile audio watermarking system. StreamMark is designed to be robust against benign audio conversions that preserve semantic meaning (e.g., compression, noise) while remaining fragile to malicious, semantics-altering manipulations (e.g., voice conversion, speech editing). Our method introduces a complex-domain embedding technique within a unique Encoder-Distortion-Decoder architecture, trained explicitly to differentiate between these two classes of transformations. Comprehensive benchmarks demonstrate that StreamMark achieves high imperceptibility (SNR 24.16 dB, PESQ 4.20), is resilient to real-world distortions like Opus encoding, and exhibits principled fragility against a suite of deepfake attacks, with message recovery accuracy dropping to chance levels (~50%), while remaining robust to benign AI-based style transfers (ACC >98%).
#### TokenSE: a Mamba-based discrete token speech enhancement framework for cochlear implants
 - **Authors:** Hsin-Tien Chiang, John H. L. Hansen
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2604.12246

 - **Pdf link:** https://arxiv.org/pdf/2604.12246

 - **Abstract**
 Speech enhancement (SE) is critical for improving speech intelligibility and quality in real-world environments, particularly for cochlear implant (CI) users who experience severe degradations in speech understanding under noisy and reverberant conditions. In this study, we propose TokenSE, a discrete token-based SE framework operating in the neural audio codec space, which predicts clean codec token indices from degraded speech using a Mamba-based model. Unlike the earlier Transformer architecture, whose self-attention mechanism has a computational complexity that grows quadratically with sequence length, the input-dependent selection mechanism of Mamba achieves linear complexity, making it a compelling alternative to Transformers, especially for CI and hearing-aid (HA) applications. Objective evaluations show that TokenSE consistently outperforms baseline methods on both in-domain and out-of-domain datasets. Moreover, subjective listening experiments with CI users indicate clear benefit in speech intelligibility under adverse noisy and reverberant environments.
#### VoxEffects: A Speech-Oriented Audio Effects Dataset and Benchmark
 - **Authors:** Zhe Zhang, Yigitcan Özer, Junichi Yamagishi
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2604.12389

 - **Pdf link:** https://arxiv.org/pdf/2604.12389

 - **Abstract**
 Speech audio in the wild is often processed by post-production effects, but existing speech datasets rarely provide precise annotations of effects and parameters, limiting systematic study. We introduce VoxEffects, a speech audio effects dataset that pairs produced speech with exact effect-chain supervision at multiple granularities. VoxEffects supports speech-oriented audio effect identification: given a produced waveform, infer which effects are present and how they are applied. Built from minimally edited clean speech, it provides an extensible rendering pipeline for both offline synthesis and on-the-fly rendering for efficient training and evaluation. The audio effect identification benchmark includes effect presence detection, preset classification, and intensity prediction, with a robustness protocol covering capture-side and platform-side degradations. We provide an AudioMAE-based multi-task baseline and analyses of domain shift, robustness, input duration, and gender fairness.
#### Contextual Biasing for ASR in Speech LLM with Common Word Cues and Bias Word Position Prediction
 - **Authors:** Sashi Novitasari, Takashi Fukuda, Kurata Gakuto, George Saon
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2604.12398

 - **Pdf link:** https://arxiv.org/pdf/2604.12398

 - **Abstract**
 Speech-aware LLMs (SLLMs) have recently achieved state-of-the-art ASR performance; however, they still fail to accurately transcribe bias words that appear rarely or never in the training data. Contextual biasing mechanisms are commonly implemented by introducing a predefined bias word list into the model via a text prompt or additional module. For further improvement, predefined bias words can be paired with their phoneme representations as pronunciation cues. Typically, phoneme sequences are generated through a G2P system that covers the target languages and domains of the bias words. Therefore, when a compatible G2P system is unavailable, phoneme-assisted contextual biasing becomes difficult to perform. Moreover, manually adding accurate phoneme sequences requires advanced phonetic knowledge. In this paper, we explore contextual biasing in SLLM based on acoustic cues associated with a set of common words whose pronunciations are partially similar to those of the target bias words. We assume ASR applications in which end users do not require special knowledge of phonetics or utilize G2P tools for inference. For enhanced robustness, we also introduce bias word positional prediction implemented in a multi-output learning fashion. Our method reduces bias word recognition errors by 16.3% compared to baseline systems, including on out-of-domain data.
#### An Ultra-Low Latency, End-to-End Streaming Speech Synthesis Architecture via Block-Wise Generation and Depth-Wise Codec Decoding
 - **Authors:** Tianhui Su, Tien-Ping Tan, Salima Mdhaffar, Yannick Estève, Aghilas Sini
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2604.12438

 - **Pdf link:** https://arxiv.org/pdf/2604.12438

 - **Abstract**
 Real-time speech synthesis requires balancing inference latency and acoustic fidelity for interactive applications. Conventional continuous text-to-speech pipelines require computationally intensive neural vocoders to reconstruct phase information, creating a significant streaming bottleneck. Furthermore, regression-based acoustic modeling frequently induces spectral over-smoothing artifacts. To address these limitations, this paper proposes a novel end-to-end non-autoregressive architecture optimized for ultra-low latency block-wise generation, directly modeling the highly compressed discrete latent space of the Mimi neural audio codec. Integrating a modified FastSpeech 2 backbone with a progressive depth-wise sequential decoding strategy, the architecture dynamically conditions 32 layers of residual vector quantization codes. This mechanism resolves phonetic alignment degradation and manages the complexity of high-fidelity discrete representations without temporal autoregressive overhead. Experimental evaluations on English and Malay datasets validate its language-independent deployment capability. Compared to conventional continuous regression models, the proposed architecture demonstrates quantitative improvements in fundamental voicing accuracy and mitigates high-frequency spectral degradation. It achieves ultra-low latency inference, translating to a 10.6-fold absolute acceleration over conventional cascaded pipelines. Crucially, the system achieves an average time-to-first-byte latency of 48.99 milliseconds, falling significantly below the human perception threshold for real-time interactive streaming. These results firmly establish the proposed architecture as a highly optimized solution for deploying real-time streaming speech interfaces.
#### X-VC: Zero-shot Streaming Voice Conversion in Codec Space
 - **Authors:** Qixi Zheng, Yuxiang Zhao, Tianrui Wang, Wenxi Chen, Kele Xu, Yikang Li, Qinyuan Chen, Xipeng Qiu, Kai Yu, Xie Chen
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Artificial Intelligence (cs.AI)
 - **Arxiv link:** https://arxiv.org/abs/2604.12456

 - **Pdf link:** https://arxiv.org/pdf/2604.12456

 - **Abstract**
 Zero-shot voice conversion (VC) aims to convert a source utterance into the voice of an unseen target speaker while preserving its linguistic content. Although recent systems have improved conversion quality, building zero-shot VC systems for interactive scenarios remains challenging because high-fidelity speaker transfer and low-latency streaming inference are difficult to achieve simultaneously. In this work, we present X-VC, a zero-shot streaming VC system that performs one-step conversion in the latent space of a pretrained neural codec. X-VC uses a dual-conditioning acoustic converter that jointly models source codec latents and frame-level acoustic conditions derived from target reference speech, while injecting utterance-level target speaker information through adaptive normalization. To reduce the mismatch between training and inference, we train the model with generated paired data and a role-assignment strategy that combines standard, reconstruction, and reversed modes. For streaming inference, we further adopt a chunkwise inference scheme with overlap smoothing that is aligned with the segment-based training paradigm of the codec. Experiments on Seed-TTS-Eval show that X-VC achieves the best streaming WER in both English and Chinese, strong speaker similarity in same-language and cross-lingual settings, and substantially lower offline real-time factor than the compared baselines. These results suggest that codec-space one-step conversion is a practical approach for building high-quality low-latency zero-shot VC systems. Audio samples are available at this https URL. Our code and checkpoints will also be released.
#### Audio-Cogito: Towards Deep Audio Reasoning in Large Audio Language Models
 - **Authors:** Longhao Li, Hongjie Chen, Zehan Li, Qihan Hu, Jian Kang, Jie Li, Lei Xie, Yongxiang Li
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2604.12527

 - **Pdf link:** https://arxiv.org/pdf/2604.12527

 - **Abstract**
 Recent advances in reasoning models have driven significant progress in text and multimodal domains, yet audio reasoning remains relatively limited. Only a few Large Audio Language Models (LALMs) incorporate explicit Chain-of-Thought (CoT) reasoning, and their capabilities are often inconsistent and insufficient for complex tasks. To bridge this gap, we introduce Audio-Cogito, a fully open-source solution for deep audio reasoning. We develop Cogito-pipe for high-quality audio reasoning data curation, producing 545k reasoning samples that will be released after review. Based on this dataset, we adopt a self-distillation strategy for model fine-tuning. Experiments on the MMAR benchmark, the only audio benchmark evaluating the CoT process, show that our model achieves the best performance among open-source models and matches or surpasses certain closed-source models in specific metrics. Our approach also ranks among the top-tier systems in the Interspeech 2026 Audio Reasoning Challenge.
#### MoshiRAG: Asynchronous Knowledge Retrieval for Full-Duplex Speech Language Models
 - **Authors:** Chung-Ming Chien, Manu Orsini, Eugene Kharitonov, Neil Zeghidour, Karen Livescu, Alexandre Défossez
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2604.12928

 - **Pdf link:** https://arxiv.org/pdf/2604.12928

 - **Abstract**
 Speech-to-speech language models have recently emerged to enhance the naturalness of conversational AI. In particular, full-duplex models are distinguished by their real-time interactivity, including handling of pauses, interruptions, and backchannels. However, improving their factuality remains an open challenge. While scaling the model size could address this gap, it would make real-time inference prohibitively expensive. In this work, we propose MoshiRAG, a modular approach that combines a compact full-duplex interface with selective retrieval to access more powerful knowledge sources. Our asynchronous framework enables the model to identify knowledge-demanding queries and ground its responses in external information. By leveraging the natural temporal gap between response onset and the delivery of core information, the retrieval process can be completed while maintaining a natural conversation flow. With this approach, MoshiRAG achieves factuality comparable to the best publicly released non-duplex speech language models while preserving the interactivity inherent to full-duplex systems. Moreover, our flexible design supports plug-and-play retrieval methods without retraining and demonstrates strong performance on out-of-domain mathematical reasoning tasks.


by Zyzzyva0381 (Windy). 


2026-04-15
