# Showing new listings for Friday, 29 May 2026
Auto update papers at about 2:30am UTC (10:30am Beijing time) every weekday.


阅读 `Usage.md`了解如何使用此repo实现个性化的Arxiv论文推送

See `Usage.md` for instructions on how to personalize the repo. 


Keyword list: ['text-to-speech', 'text to speech', 'tts', 'LLM-based', 'speech', 'voice']


Excluded: []


### Today: 4papers 
#### The WER Trap: Shattering the Illusion of Unified Tokens in Speech Language Models
 - **Authors:** Xiangyu Zhang, Yuxin Li, Haoyang Zhang, Shiqi Han, Hexin Liu, Qiquan Zhang, Beena Ahmed, Julien Epps
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2605.29209

 - **Pdf link:** https://arxiv.org/pdf/2605.29209

 - **Abstract**
 The pursuit of a "unified" discrete token for both speech understanding and generation has led the Speech Language Model (SLM) community to heavily rely on Word Error Rate (WER) -- the core metric for Whisper-style tokenizers -- as the definitive proxy for representation quality. This fosters the assumption that low-WER tokens inherently preserve the information necessary for intelligible acoustic synthesis. We argue this is fundamentally deceptive. While high-frequency tokens succeed in generation tasks due to implicit information leakage, isolating pure semantic information at ultra-low frame rates strips away the finegrained articulation and micro-dynamics essential for ODE-based generation. Empirically validating this requires extreme compression without sacrificing WER -- a methodological bottleneck, as standard fixed-stride downsampling arbitrarily truncates phonetic boundaries. To overcome this, we develop a dynamic compression tokenizer that intelligently aligns representations with semantic boundaries, achieving ultra-low frame rates with exceptionally low WER. Using these isolated "pure" semantic tokens, we expose the WER trap: when conditioning generative models -- even with oracle duration alignments -- the reconstructed speech suffers from severe articulation blur and is rendered acoustically unintelligible. Our findings demonstrate that semantic categorization rewarded by low WER is inherently orthogonal to the continuous phonetic trajectories required for synthesis, shattering the illusion of the unified token and advocating for explicitly decoupled speech representations.
#### Decoding Strategies for Diffusion-Based ASR: A Systematic Evaluation of Confidence-Based Thresholding
 - **Authors:** Jeong Hun Yeo, Minsu Kim, Hyeongseop Rha, Yong Man Ro
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2605.29613

 - **Pdf link:** https://arxiv.org/pdf/2605.29613

 - **Abstract**
 While LLM-based Automatic Speech Recognition (ASR) achieves high accuracy, its speed is limited by sequential autoregressive decoding. Diffusion Language Models (DLMs) offer a parallel alternative, yet their decoding strategies remain under-explored in ASR contexts. This paper analyzes three decoding schemes for DLM-based ASR: fixed-number, static confidence threshold, and dynamic confidence threshold. We propose measuring round-wise accuracy using Negative Log-Likelihood-based uncertainty as a proxy for decoding progress. Our results show that both threshold-based strategies significantly outperform fixed-number schemes in accuracy and speed. We attribute this to a property unique to ASR: most tokens reach high confidence early, allowing reliable ones to be harvested aggressively while leaving only difficult tokens for later rounds. Notably, the static-threshold strategy matches the accuracy of autoregressive decoding while offering superior efficiency.
#### MELD: Mel-Spectrogram-Based Speech Language Modeling with Discrete Latent Variables
 - **Authors:** Sung-Lin Yeh, Wei Zhou, Gil Keren, Duc Le, Zhong Meng, Hao Tang, Jay Mahadeokar, Ozlem Kalinli, Alexandre Mourachko
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Computation and Language (cs.CL)
 - **Arxiv link:** https://arxiv.org/abs/2605.29859

 - **Pdf link:** https://arxiv.org/pdf/2605.29859

 - **Abstract**
 Recent speech language models rely on encoders that are optimized separately from autoregressive models. Since these encoders are unaware of the downstream objectives, the extracted representations may not be optimal for downstream tasks. To address this limitation, we introduce a discrete latent variable model on mel spectrograms that jointly optimizes the encoder and the speech language model. Joint optimization not only brings improvements over codec-based and other mel-spectrogram-based baselines on zero-shot Text-to-Speech (TTS) and Speech-to-Text (STT) tasks, but also effectively alleviates common issues in autoregressive mel-spectrogram modeling, such as prolonged silence generation and word omissions.
#### HoliTok:A Coutinuous Holistic Tokenization with Robust Dual Capabilities of Speech Generation and Understanding
 - **Authors:** Bohan Li, Shi Lian, Hankun Wang, Yiwei Guo, Yu Xi, Zhihan Li, Da Zheng, Colin Zhang, Kai Yu
 - **Subjects:** Subjects:
Sound (cs.SD); Artificial Intelligence (cs.AI); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2605.29948

 - **Pdf link:** https://arxiv.org/pdf/2605.29948

 - **Abstract**
 Unified speech foundation models require a holistic tokenization space that is both learnable by language models and decodable into high-quality waveforms. Existing speech tokenizers, however, often fail to satisfy these requirements simultaneously, leading to increased architectural complexity and more involved training designs. We propose HoliTok, a continuous Holistic speech Tokenization model designed for unified generation-understanding modeling. HoliTok encodes 48~kHz speech into a compact 25~Hz sequence of 128-dimensional latents. It is trained with a progressive strategy that jointly preserves signal-level fidelity, incorporates semantic information, and maintains strong latent learnability. Based on this tokenization, we build a unified AR+DiT model for speech synthesis and recognition, where the same latent sequence supports both generation-specific and unified generation-understanding tasks. Experiments show that HoliTok achieves competitive reconstruction fidelity, improves generative learnability for high-quality and controllable synthesis, and, among the evaluated representations, is the only one that operates robustly in our unified generation-understanding architecture without additional optimization tricks. These results suggest that HoliTok serves as an effective speech tokenizer and a foundational representation interface for unified spoken language modeling. The code is available at: this https URL.


by Zyzzyva0381 (Windy). 


2026-05-29
