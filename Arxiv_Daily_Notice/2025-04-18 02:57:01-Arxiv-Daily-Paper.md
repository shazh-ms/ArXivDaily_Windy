# Showing new listings for Friday, 18 April 2025
Auto update papers at about 2:30am UTC (10:30am Beijing time) every weekday.


阅读 `Usage.md`了解如何使用此repo实现个性化的Arxiv论文推送

See `Usage.md` for instructions on how to personalize the repo. 


Keyword list: ['acoustic', 'reinforcement learning', 'reverb', 'meta', 'separate', 'reconstruction', 'noise', 'enhance', 'localization', 'speech']


Excluded: []


### Today: 6papers 
#### Temporal Attention Pooling for Frequency Dynamic Convolution in Sound Event Detection
 - **Authors:** Hyeonuk Nam, Yong-Hwa Park
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2504.12670

 - **Pdf link:** https://arxiv.org/pdf/2504.12670

 - **Abstract**
 Recent advances in deep learning, particularly frequency dynamic convolution (FDY conv), have significantly improved sound event detection (SED) by enabling frequency-adaptive feature extraction. However, FDY conv relies on temporal average pooling, which treats all temporal frames equally, limiting its ability to capture transient sound events such as alarm bells, door knocks, and speech plosives. To address this limitation, we propose temporal attention pooling frequency dynamic convolution (TFD conv) to replace temporal average pooling with temporal attention pooling (TAP). TAP adaptively weights temporal features through three complementary mechanisms: time attention pooling (TA) for emphasizing salient features, velocity attention pooling (VA) for capturing transient changes, and conventional average pooling for robustness to stationary signals. Ablation studies show that TFD conv improves average PSDS1 by 3.02% over FDY conv with only a 14.8% increase in parameter count. Classwise ANOVA and Tukey HSD analysis further demonstrate that TFD conv significantly enhances detection performance for transient-heavy events, outperforming existing FDY conv models. Notably, TFD conv achieves a maximum PSDS1 score of 0.456, surpassing previous state-of-the-art SED systems. We also explore the compatibility of TAP with other FDY conv variants, including dilated FDY conv (DFD conv), partial FDY conv (PFD conv), and multi-dilated FDY conv (MDFD conv). Among these, the integration of TAP with MDFD conv achieves the best result with a PSDS1 score of 0.459, validating the complementary strengths of temporal attention and multi-scale frequency adaptation. These findings establish TFD conv as a powerful and generalizable framework for enhancing both transient sensitivity and overall feature robustness in SED.
#### EmoVoice: LLM-based Emotional Text-To-Speech Model with Freestyle Text Prompting
 - **Authors:** Guanrou Yang, Chen Yang, Qian Chen, Ziyang Ma, Wenxi Chen, Wen Wang, Tianrui Wang, Yifan Yang, Zhikang Niu, Wenrui Liu, Fan Yu, Zhihao Du, Zhifu Gao, ShiLiang Zhang, Xie Chen
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Artificial Intelligence (cs.AI); Computation and Language (cs.CL)
 - **Arxiv link:** https://arxiv.org/abs/2504.12867

 - **Pdf link:** https://arxiv.org/pdf/2504.12867

 - **Abstract**
 Human speech goes beyond the mere transfer of information; it is a profound exchange of emotions and a connection between individuals. While Text-to-Speech (TTS) models have made huge progress, they still face challenges in controlling the emotional expression in the generated speech. In this work, we propose EmoVoice, a novel emotion-controllable TTS model that exploits large language models (LLMs) to enable fine-grained freestyle natural language emotion control, and a phoneme boost variant design that makes the model output phoneme tokens and audio tokens in parallel to enhance content consistency, inspired by chain-of-thought (CoT) and modality-of-thought (CoM) techniques. Besides, we introduce EmoVoice-DB, a high-quality 40-hour English emotion dataset featuring expressive speech and fine-grained emotion labels with natural language descriptions. EmoVoice achieves state-of-the-art performance on the English EmoVoice-DB test set using only synthetic training data, and on the Chinese Secap test set using our in-house data. We further investigate the reliability of existing emotion evaluation metrics and their alignment with human perceptual preferences, and explore using SOTA multimodal LLMs GPT-4o-audio and Gemini to assess emotional speech. Demo samples are available at this https URL. Dataset, code, and checkpoints will be released.
#### CST-former: Multidimensional Attention-based Transformer for Sound Event Localization and Detection in Real Scenes
 - **Authors:** Yusun Shul, Dayun Choi, Jung-Woo Choi
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2504.12870

 - **Pdf link:** https://arxiv.org/pdf/2504.12870

 - **Abstract**
 Sound event localization and detection (SELD) is a task for the classification of sound events and the identification of direction of arrival (DoA) utilizing multichannel acoustic signals. For effective classification and localization, a channel-spectro-temporal transformer (CST-former) was suggested. CST-former employs multidimensional attention mechanisms across the spatial, spectral, and temporal domains to enlarge the model's capacity to learn the domain information essential for event detection and DoA estimation over time. In this work, we present an enhanced version of CST-former with multiscale unfolded local embedding (MSULE) developed to capture and aggregate domain information over multiple time-frequency scales. Also, we propose finetuning and post-processing techniques beneficial for conducting the SELD task over limited training datasets. In-depth ablation studies of the proposed architecture and detailed analysis on the proposed modules are carried out to validate the efficacy of multidimensional attentions on the SELD task. Empirical validation through experimentation on STARSS22 and STARSS23 datasets demonstrates the remarkable performance of CST-former and post-processing techniques without using external data.
#### GOAT-TTS: LLM-based Text-To-Speech Generation Optimized via A Dual-Branch Architecture
 - **Authors:** Yaodong Song, Hongjie Chen, Jie Lian, Yuxin Zhang, Guangmin Xia, Zehan Li, Genliang Zhao, Jian Kang, Yongxiang Li, Jie Li
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2504.12339

 - **Pdf link:** https://arxiv.org/pdf/2504.12339

 - **Abstract**
 While large language models (LLMs) have revolutionized text-to-speech (TTS) synthesis through discrete tokenization paradigms, current architectures exhibit fundamental tensions between three critical dimensions: 1) irreversible loss of acoustic characteristics caused by quantization of speech prompts; 2) stringent dependence on precisely aligned prompt speech-text pairs that limit real-world deployment; and 3) catastrophic forgetting of the LLM's native text comprehension during optimization for speech token generation. To address these challenges, we propose an LLM-based text-to-speech Generation approach Optimized via a novel dual-branch ArchiTecture (GOAT-TTS). Our framework introduces two key innovations: (1) The modality-alignment branch combines a speech encoder and projector to capture continuous acoustic embeddings, enabling bidirectional correlation between paralinguistic features (language, timbre, emotion) and semantic text representations without transcript dependency; (2) The speech-generation branch employs modular fine-tuning on top-k layers of an LLM for speech token prediction while freezing the bottom-k layers to preserve foundational linguistic knowledge. Moreover, multi-token prediction is introduced to support real-time streaming TTS synthesis. Experimental results demonstrate that our GOAT-TTS achieves performance comparable to state-of-the-art TTS models while validating the efficacy of synthesized dialect speech data.
#### Can Masked Autoencoders Also Listen to Birds?
 - **Authors:** Lukas Rauch, Ilyass Moummad, RenÃ© Heinrich, Alexis Joly, Bernhard Sick, Christoph Scholz
 - **Subjects:** Subjects:
Machine Learning (cs.LG); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2504.12880

 - **Pdf link:** https://arxiv.org/pdf/2504.12880

 - **Abstract**
 Masked Autoencoders (MAEs) pretrained on AudioSet fail to capture the fine-grained acoustic characteristics of specialized domains such as bioacoustic monitoring. Bird sound classification is critical for assessing environmental health, yet general-purpose models inadequately address its unique acoustic challenges. To address this, we introduce Bird-MAE, a domain-specialized MAE pretrained on the large-scale BirdSet dataset. We explore adjustments to pretraining, fine-tuning and utilizing frozen representations. Bird-MAE achieves state-of-the-art results across all BirdSet downstream tasks, substantially improving multi-label classification performance compared to the general-purpose Audio-MAE baseline. Additionally, we propose prototypical probing, a parameter-efficient method for leveraging MAEs' frozen representations. Bird-MAE's prototypical probes outperform linear probing by up to 37\% in MAP and narrow the gap to fine-tuning to approximately 3\% on average on BirdSet.
#### A Multi-task Learning Balanced Attention Convolutional Neural Network Model for Few-shot Underwater Acoustic Target Recognition
 - **Authors:** Wei Huang, Shumeng Sun, Junpeng Lu, Zhenpeng Xu, Zhengyang Xiu, Hao Zhang
 - **Subjects:** Subjects:
Sound (cs.SD); Artificial Intelligence (cs.AI); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2504.13102

 - **Pdf link:** https://arxiv.org/pdf/2504.13102

 - **Abstract**
 Underwater acoustic target recognition (UATR) is of great significance for the protection of marine diversity and national defense security. The development of deep learning provides new opportunities for UATR, but faces challenges brought by the scarcity of reference samples and complex environmental interference. To address these issues, we proposes a multi-task balanced channel attention convolutional neural network (MT-BCA-CNN). The method integrates a channel attention mechanism with a multi-task learning strategy, constructing a shared feature extractor and multi-task classifiers to jointly optimize target classification and feature reconstruction tasks. The channel attention mechanism dynamically enhances discriminative acoustic features such as harmonic structures while suppressing noise. Experiments on the Watkins Marine Life Dataset demonstrate that MT-BCA-CNN achieves 97\% classification accuracy and 95\% $F1$-score in 27-class few-shot scenarios, significantly outperforming traditional CNN and ACNN models, as well as popular state-of-the-art UATR methods. Ablation studies confirm the synergistic benefits of multi-task learning and attention mechanisms, while a dynamic weighting adjustment strategy effectively balances task contributions. This work provides an efficient solution for few-shot underwater acoustic recognition, advancing research in marine bioacoustics and sonar signal processing.


by Zyzzyva0381 (Windy). 


2025-04-18
