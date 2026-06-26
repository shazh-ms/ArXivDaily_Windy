# Showing new listings for Friday, 26 June 2026
Auto update papers at about 2:30am UTC (10:30am Beijing time) every weekday.


阅读 `Usage.md`了解如何使用此repo实现个性化的Arxiv论文推送

See `Usage.md` for instructions on how to personalize the repo. 


Keyword list: ['text-to-speech', 'text to speech', 'tts', 'LLM-based', 'speech', 'voice']


Excluded: []


### Today: 6papers 
#### A Large-Scale Database and Predictive Model of Listener-Rated Ease of Speech Understanding in Commercial Hearing Aids
 - **Authors:** Andrew Sabin, Steve Taddei, Abram Bailey
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2606.26342

 - **Pdf link:** https://arxiv.org/pdf/2606.26342

 - **Abstract**
 HearAdvisor aims to provide hearing-aid consumers with audio-performance metrics and recordings that reflect real listening experience. For speech-related metrics, HearAdvisor has historically used HASPIv2, a metric designed to predict objective intelligibility and validated primarily under simulated distortions. Its relationship to consumer-rated ease of understanding for commercial hearing aids is uncertain. Here we introduce a large-scale perceptual dataset and learned metric for listener-rated perceived benefit for speech understanding. Website visitors with self-reported hearing loss completed a blind, MUSHRA-inspired listening test in which they rated recordings of commercial hearing aids on a five-point "Ease of Understanding" scale. The dataset contains 151,608 ratings, 104,298 after quality screening, spanning 10,394 binaural acoustic-manikin recordings from 83 commercial products across 72 realistic acoustic scenes. To predict these ratings, we pass aided audio and a matched clean-speech reference through a frozen Whisper encoder, subtract their internal representations, and train a small MLP head on the resulting difference embedding. On devices held out of training, the learned metric substantially outperforms HASPIv2 at the scene level (overall r = 0.92 vs. 0.83; loud = 0.89 vs. 0.75; quiet = 0.79 vs. 0.58). In loud scenes, performance reaches the split-half reliability of the listener ratings; in quiet scenes, it approaches that ceiling. The model also responds sensibly to controlled gain and SNR manipulations. Together, the dataset and model provide a new way to predict listener-rated ease of speech understanding for real commercial hearing-aid recordings.
#### DNSMOS-C: Improving End-to-end Speech Quality Models via Contrastive Learning
 - **Authors:** Xinyu Liang, Fredrik Cumlin, Victor Ungureanu, Chandan K.A. Reddy, Christian Schuldt, Saikat Chatterjee
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2606.26903

 - **Pdf link:** https://arxiv.org/pdf/2606.26903

 - **Abstract**
 We introduce DNSMOS-C, a compact end-to-end speech quality assessment model that extends the DNSMOS Pro framework by integrating a MOS-guided triplet-based contrastive loss. Applied directly to the intermediate embeddings, this contrastive supervision encourages the latent space to be better organized with respect to perceptual quality while preserving the simplicity and efficiency of DNSMOS Pro. Unlike prior methods that depend on large pre-trained self-supervised learning (SSL) encoders and multi-stage training, DNSMOS-C jointly learns speech representations and MOS regression within a single, unified framework. Experiments on multiple datasets show that DNSMOS-C consistently improves correlation metrics over DNSMOS Pro and achieves better generalization on challenging out-of-domain test sets. Furthermore, latent space analyses indicate that our approach learns representations that exhibit an emergent low-dimensional quality ordering, which enhances interpretability and improves training stability. These findings demonstrate that MOS-guided contrastive learning enables more robust and accurate quality predictions without incurring additional computational overhead.
#### PairAlign: A Framework for Sequence Tokenization via Self-Alignment with Applications to Audio Tokenization
 - **Authors:** Adhiraj Banerjee, Vipul Arora
 - **Subjects:** Subjects:
Machine Learning (cs.LG); Computation and Language (cs.CL); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2605.06582

 - **Pdf link:** https://arxiv.org/pdf/2605.06582

 - **Abstract**
 Modern learning systems represent perceptual signals with continuous vectors, but comparison, retrieval, memory, alignment, and reasoning are often naturally symbolic. In language, this interface is given by tokens; for speech and audio, it must be learned. Existing audio tokenizers use local quantization, clustering, or reconstruction, leaving sequence consistency, compactness, length control, termination, and edit geometry indirectly optimized. We introduce PairAlign, a framework for compact audio tokenization through sequence-level self-alignment. PairAlign treats tokenization as conditional sequence generation: an encoder maps speech to a condition, and an autoregressive decoder emits tokens from BOS to EOS, learning identity, order, length, and termination. Given two content-preserving views, each token string is trained to be likely under the other's representation, while unrelated examples provide competing sequences. This yields a surrogate for edit-distance preservation while discouraging collapse. Starting from a VQ tokenizer, PairAlign extends a frame-synchronous prior into an autoregressive tokenizer using VQ-derived and EMA-teacher targets, cross-paired teacher forcing, anti-bypass regularization, likelihood contrast, length control, and timing recovery. On 3 s speech, PairAlign learns compact token strings with strong cross-view consistency. In retrieval, it operates at 12.71 tokens/s and reduces archive tokens by 55% versus VQ while preserving edit-distance search. The results expose a compactness--locality trade-off: PairAlign does not aim to dominate dense geometric or SSL tokenizers on every local metric, but provides a lower-rate symbolic interface for comparison, retrieval, and analysis. More broadly, PairAlign is a sequence-symbolic analogue of JEPA-style predictive learning, predicting a learned variable-length symbolic sequence rather than a continuous latent.
#### Sarashina2.2-TTS: Tackling Kanji Polyphony in Japanese Speech Generation via Data Scaling and Targeted Data Synthesis
 - **Authors:** Lianbo Liu, Shiao Zhu, Kai Washizaki, Reo Yoneyama, Haesung Jeon, Mengjie Zhao, Yusuke Fujita, Hao Shi, Nao Yoshida, Yuan Gao, Roman Koshkin, Yukiya Hono, Yui Sudo
 - **Subjects:** Subjects:
Sound (cs.SD); Artificial Intelligence (cs.AI); Computation and Language (cs.CL); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2606.25369

 - **Pdf link:** https://arxiv.org/pdf/2606.25369

 - **Abstract**
 While large language model (LLM)-based text-to-speech (TTS) systems have achieved high-quality speech synthesis, most existing systems focus on English and Chinese. Japanese, however, remains under-explored, and its unique linguistic challenges, such as widespread context-dependent kanji polyphony, have yet to be adequately tackled. Here we introduce Sarashina2.2-TTS (this https URL), a Japanese-centric LLM-TTS system that tackles these challenges through a dual approach: data strategy and evaluation methodology. First, we scale training to approximately 361k hours of speech, incorporating a balanced mix of Japanese and English data. Furthermore, we design a targeted data augmentation pipeline covering all 2,136 Joyo (regular-use) kanji designated by Japan's Agency for Cultural Affairs to efficiently address kanji polyphony disambiguation. Second, we introduce the Joyo Kanji Yomi Benchmark (this https URL), covering all 2,136 Joyo kanji and their 4,378 readings. Alongside this benchmark, we propose Kana-CER, a metric that compares synthesized speech against reference readings in the kana space, eliminating orthographic variations to directly measure pronunciation correctness. Experiments demonstrate that our targeted data augmentation significantly improves reading accuracy. Overall, Sarashina2.2-TTS achieves state-of-the-art kanji-level reading accuracy and matches top baselines on general sentence-level pronunciation, while delivering the highest speaker similarity in zero-shot Japanese speech synthesis. Furthermore, cross-lingual evaluation reveals that Sarashina2.2-TTS is the only system that maintains stable Japanese pronunciation regardless of the prompt language, confirming that our balanced training approach improves cross-lingual robustness.
#### WQ-Fusion: Dynamic Gated Attention for Cross-Domain Audio Representation
 - **Authors:** Mingda Lin, Lei Ding, Xinyue Zhou, Tiantian Xiong, Hanchen Pei, Gongping Huang, Hao Zhang, Jingdong Chen, Jacob Benesty
 - **Subjects:** Subjects:
Sound (cs.SD); Multimedia (cs.MM); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2606.26556

 - **Pdf link:** https://arxiv.org/pdf/2606.26556

 - **Abstract**
 While pre-trained models excel in specialized tasks, learning universal representations across diverse acoustic domains remains challenging. To address this, we propose WQ-Fusion, a robust dual-encoder framework for cross-domain audio representation learning. Overcoming the limitations of static concatenation, WQ-Fusion integrates whisper and qwen via an Adaptive Feature Modulation module and a novel element-wise gated attention mechanism. This design enables dynamic feature selection, allowing the model to selectively emphasize relevant acoustic and semantic dimensions. Extensive experiments on the Interspeech 2026 Audio Encoder Capability Challenge (Track A) benchmark demonstrate that by effectively routing heterogeneous information, WQ-Fusion achieves a superior overall score of 0.836, significantly outperforming the strongest single-encoder baseline.
#### wav2tok 2.0: Scalable Audio Tokenization Maintaining Explicit Pairwise Token Alignment for Efficient Audio Retrieval
 - **Authors:** Adhiraj Banerjee, Vipul Arora
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2606.26824

 - **Pdf link:** https://arxiv.org/pdf/2606.26824

 - **Abstract**
 Learning discrete speech representations that preserve similarity across variable-length utterances is central to query-by-example spoken term detection (QbE-STD). While wav2tok introduced CTC-based sequence alignment to enforce token consistency, its tightly coupled clustering and alignment training recipe limits scalability. We propose wav2tok 2.0, a scalable alignment-aware speech tokenizer built on the BEST-STD backbone. wav2tok 2.0 employs staged training, first learning discriminative, speaker-invariant representations via contrastive learning and vector quantization, and then enforcing pairwise token consistency using a CTC alignment loss and a novel DTW-aligned framewise prediction objective with adaptive weighting. Experiments show that wav2tok 2.0 consistently outperforms BEST-STD and general-purpose tokenizers on QbE-STD while remaining efficient and scalable.


by Zyzzyva0381 (Windy). 


2026-06-26
