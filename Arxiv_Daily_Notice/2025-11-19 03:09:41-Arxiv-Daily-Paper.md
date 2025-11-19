# Showing new listings for Wednesday, 19 November 2025
Auto update papers at about 2:30am UTC (10:30am Beijing time) every weekday.


阅读 `Usage.md`了解如何使用此repo实现个性化的Arxiv论文推送

See `Usage.md` for instructions on how to personalize the repo. 


Keyword list: ['text-to-speech', 'text to speech', 'tts', 'LLM-based', 'speech', 'voice']


Excluded: []


### Today: 2papers 
#### Principled Coarse-Grained Acceptance for Speculative Decoding in Speech
 - **Authors:** Moran Yanuka, Paul Dixon, Eyal Finkelshtein, Daniel Rotman, Raja Giryes
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Machine Learning (cs.LG)
 - **Arxiv link:** https://arxiv.org/abs/2511.13732

 - **Pdf link:** https://arxiv.org/pdf/2511.13732

 - **Abstract**
 Speculative decoding accelerates autoregressive speech generation by letting a fast draft model propose tokens that a larger target model verifies. However, for speech LLMs that generate acoustic tokens, exact token matching is overly restrictive: many discrete tokens are acoustically or semantically interchangeable, reducing acceptance rates and limiting speedups. We introduce Principled Coarse-Graining (PCG), which verifies proposals at the level of Acoustic Similarity Groups (ASGs) derived from the target model's embedding space. By splitting each token's probability mass across the overlapping groups that contain it, we define an overlap-aware coarse-grained distribution and perform rejection sampling on the resulting group variable. This yields an exactness guarantee at the group level while allowing the accepted draft token to stand in for any member of the group in practice. On LibriTTS, PCG increases acceptance and throughput relative to standard speculative decoding and prior speech-specific relaxations while maintaining intelligibility and speaker similarity. These results suggest acoustically aware, group-level acceptance as a simple and general way to accelerate speech token generation while maintaining speech quality.
#### TTA: Transcribe, Translate and Alignment for Cross-lingual Speech Representation
 - **Authors:** Wei Liu, Jiahong Li, Yiwen Shao, Dong Yu
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2511.14410

 - **Pdf link:** https://arxiv.org/pdf/2511.14410

 - **Abstract**
 Speech-LLM models have demonstrated great performance in multi-modal and multi-task speech understanding. A typical speech-LLM paradigm is integrating speech modality with a large language model (LLM). While the Whisper encoder was frequently adopted in previous studies for speech input, it shows limitations regarding input format, model scale, and semantic performance. To this end, we propose a lightweight TTA model specialized in speech semantics for more effective LLM integration. With large-scale training of 358k hours of speech data on multilingual speech recognition (ASR), speech translation (ST) and speech-text alignment tasks, TTA is capable of producing robust cross-lingual speech representations. Extensive evaluations across diverse benchmarks, including ASR/ST, speech retrieval, and ASR-LLM performance assessments, demonstrate TTA's superiority over Whisper. Furthermore, we rigorously validate the interplay between cross-lingual capabilities and ASR/ST performance. The model weights and training recipes of TTA will be released as part of an audio understanding toolkit Auden.


by Zyzzyva0381 (Windy). 


2025-11-19
