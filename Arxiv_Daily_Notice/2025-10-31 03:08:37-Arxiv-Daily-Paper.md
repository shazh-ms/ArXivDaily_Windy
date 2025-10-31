# Showing new listings for Friday, 31 October 2025
Auto update papers at about 2:30am UTC (10:30am Beijing time) every weekday.


阅读 `Usage.md`了解如何使用此repo实现个性化的Arxiv论文推送

See `Usage.md` for instructions on how to personalize the repo. 


Keyword list: ['text-to-speech', 'text to speech', 'tts', 'LLM-based', 'speech', 'voice']


Excluded: []


### Today: 4papers 
#### SPEAR: A Unified SSL Framework for Learning Speech and Audio Representations
 - **Authors:** Xiaoyu Yang, Yifan Yang, Zengrui Jin, Ziyun Cui, Wen Wu, Baoxiang Li, Chao Zhang, Phil Woodland
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2510.25955

 - **Pdf link:** https://arxiv.org/pdf/2510.25955

 - **Abstract**
 Self-Supervised Learning (SSL) excels at learning generic representations of acoustic signals, yet prevailing methods remain domain-specific, tailored to either speech or general audio, hindering the development of a unified representation model with a comprehensive capability over both domains. To address this, we present SPEAR (SPEech and Audio Representations), the first SSL framework to successfully learn unified speech and audio representations from a mixture of speech and audio data. SPEAR proposes a unified pre-training objective based on masked prediction of fine-grained discrete tokens for both speech and general audio. These tokens are derived from continuous speech and audio representations using a Multi-codebook Vector Quantisation (MVQ) method, retaining rich acoustic detail essential for modelling both speech and complex audio events. SPEAR is applied to pre-train both single-domain and unified speech-and-audio SSL models. Our speech-domain model establishes a new state-of-the-art on the SUPERB benchmark, a speech processing benchmark for SSL models, matching or surpassing the highly competitive WavLM Large on 12 out of 15 tasks with the same pre-training corpora and a similar model size. Crucially, our unified model learns complementary features and demonstrates comprehensive capabilities across two major benchmarks, SUPERB and HEAR, for evaluating audio representations. By further scaling up the model size and pre-training data, we present a unified model with 600M parameters that excels in both domains, establishing it as one of the most powerful and versatile open-source SSL models for auditory understanding. The inference code and pre-trained models will be made publicly available.
#### POWSM: A Phonetic Open Whisper-Style Speech Foundation Model
 - **Authors:** Chin-Jou Li, Kalvin Chang, Shikhar Bharadwaj, Eunjung Yeo, Kwanghee Choi, Jian Zhu, David Mortensen, Shinji Watanabe
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2510.24992

 - **Pdf link:** https://arxiv.org/pdf/2510.24992

 - **Abstract**
 Recent advances in spoken language processing have led to substantial progress in phonetic tasks such as automatic speech recognition (ASR), phone recognition (PR), grapheme-to-phoneme conversion (G2P), and phoneme-to-grapheme conversion (P2G). Despite their conceptual similarity, these tasks have largely been studied in isolation, each relying on task-specific architectures and datasets. In this paper, we introduce POWSM (Phonetic Open Whisper-style Speech Model), the first unified framework capable of jointly performing multiple phone-related tasks. POWSM enables seamless conversion between audio, text (graphemes), and phones, opening up new possibilities for universal and low-resource speech processing. Our model outperforms or matches specialized PR models of similar size (Wav2Vec2Phoneme and ZIPA) while jointly supporting G2P, P2G, and ASR. Our training data, code and models are released to foster open science.
#### SP-MCQA: Evaluating Intelligibility of TTS Beyond the Word Level
 - **Authors:** Hitomi Jin Ling Tee, Chaoren Wang, Zijie Zhang, Zhizheng Wu
 - **Subjects:** Subjects:
Sound (cs.SD); Computation and Language (cs.CL); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2510.26190

 - **Pdf link:** https://arxiv.org/pdf/2510.26190

 - **Abstract**
 The evaluation of intelligibility for TTS has reached a bottleneck, as existing assessments heavily rely on word-by-word accuracy metrics such as WER, which fail to capture the complexity of real-world speech or reflect human comprehension needs. To address this, we propose Spoken-Passage Multiple-Choice Question Answering, a novel subjective approach evaluating the accuracy of key information in synthesized speech, and release SP-MCQA-Eval, an 8.76-hour news-style benchmark dataset for SP-MCQA evaluation. Our experiments reveal that low WER does not necessarily guarantee high key-information accuracy, exposing a gap between traditional metrics and practical intelligibility. SP-MCQA shows that even state-of-the-art (SOTA) models still lack robust text normalization and phonetic accuracy. This work underscores the urgent need for high-level, more life-like evaluation criteria now that many systems already excel at WER yet may fall short on real-world intelligibility.
#### Modeling strategies for speech enhancement in the latent space of a neural audio codec
 - **Authors:** Sofiene Kammoun, Xavier Alameda-Pineda, Simon Leglaive
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2510.26299

 - **Pdf link:** https://arxiv.org/pdf/2510.26299

 - **Abstract**
 Neural audio codecs (NACs) provide compact latent speech representations in the form of sequences of continuous vectors or discrete tokens. In this work, we investigate how these two types of speech representations compare when used as training targets for supervised speech enhancement. We consider both autoregressive and non-autoregressive speech enhancement models based on the Conformer architecture, as well as a simple baseline where the NAC encoder is simply fine-tuned for speech enhancement. Our experiments reveal three key findings: predicting continuous latent representations consistently outperforms discrete token prediction; autoregressive models achieve higher quality but at the expense of intelligibility and efficiency, making non-autoregressive models more attractive in practice; and encoder fine-tuning yields the strongest enhancement metrics overall, though at the cost of degraded codec reconstruction. The code and audio samples are available online.


by Zyzzyva0381 (Windy). 


2025-10-31
