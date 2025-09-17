# Showing new listings for Wednesday, 17 September 2025
Auto update papers at about 2:30am UTC (10:30am Beijing time) every weekday.


阅读 `Usage.md`了解如何使用此repo实现个性化的Arxiv论文推送

See `Usage.md` for instructions on how to personalize the repo. 


Keyword list: ['text-to-speech', 'text to speech', 'tts', 'LLM-based', 'speech', 'voice']


Excluded: []


### Today: 7papers 
#### Multi-Modal Embedding-based Target Speaker Enhancement
 - **Authors:** Zhan Jin
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2509.12583

 - **Pdf link:** https://arxiv.org/pdf/2509.12583

 - **Abstract**
 Target Speaker Extraction (TSE) is a critical challenge in cocktail party scenarios. While leveraging multiple modalities, such as voice, lip, face, and expression embeddings, can enhance performance, real-world applications often suffer from intermittent modality dropout. This paper presents a comprehensive study on the interactions and robustness of various multimodal fusion strategies under varying degrees of modality dropout. We build upon a state-of-the-art audio-visual speech enhancement system and integrate four distinct speaker identity cues: lip embeddings for synchronized contextual information, a voice speaker embedding extracted via cross-attention for acoustic consistency, a static face embedding for speaker identity, and a novel dynamic expression embedding for frame-wise emotional features. We systematically evaluate different combinations of these modalities under two key training regimes: zero dropout and 80% modality dropout. Extensive experiments demonstrate that while a full multimodal ensemble achieves optimal performance under ideal (zero dropout) conditions, its effectiveness diminishes significantly when test-time dropout occurs without prior exposure during training. Crucially, we show that training with a high (80%) modality dropout rate dramatically enhances model robustness, enabling the system to maintain superior performance even under severe test-time missing modalities. Our findings highlight that voice embeddings exhibit consistent robustness, while the proposed expression embedding provides valuable complementary information. This work underscores the importance of training strategies that account for real-world imperfection, moving beyond pure performance maximization to achieve practical reliability in multimodal speech enhancement systems.
#### MSR-Codec: A Low-Bitrate Multi-Stream Residual Codec for High-Fidelity Speech Generation with Information Disentanglement
 - **Authors:** Jingyu Li, Guangyan Zhang, Zhen Ye, Yiwen Guo
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2509.13068

 - **Pdf link:** https://arxiv.org/pdf/2509.13068

 - **Abstract**
 Audio codecs are a critical component of modern speech generation systems. This paper introduces a low-bitrate, multi-scale residual codec that encodes speech into four distinct streams: semantic, timbre, prosody, and residual. This architecture achieves high-fidelity speech reconstruction at competitive low bitrates while demonstrating an inherent ability for information disentanglement. We construct a two-stage language model for text-to-speech (TTS) synthesis using this codec, which, despite its lightweight design and minimal data requirements, achieves a state-of-the-art Word Error Rate (WER) and superior speaker similarity compared to several larger models. Furthermore, the codec's design proves highly effective for voice conversion, enabling independent manipulation of speaker timbre and prosody.
#### Token-based Attractors and Cross-attention in Spoof Diarization
 - **Authors:** Kyo-Won Koo, Chan-yeong Lim, Jee-weon Jung, Hye-jin Shim, Ha-Jin Yu
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2509.13085

 - **Pdf link:** https://arxiv.org/pdf/2509.13085

 - **Abstract**
 Spoof diarization identifies ``what spoofed when" in a given speech by temporally locating spoofed regions and determining their manipulation techniques. As a first step toward this task, prior work proposed a two-branch model for localization and spoof type clustering, which laid the foundation for spoof diarization. However, its simple structure limits the ability to capture complex spoofing patterns and lacks explicit reference points for distinguishing between bona fide and various spoofing types. To address these limitations, our approach introduces learnable tokens where each token represents acoustic features of bona fide and spoofed speech. These attractors interact with frame-level embeddings to extract discriminative representations, improving separation between genuine and generated speech. Vast experiments on PartialSpoof dataset consistently demonstrate that our approach outperforms existing methods in bona fide detection and spoofing method clustering.
#### More Similar than Dissimilar: Modeling Annotators for Cross-Corpus Speech Emotion Recognition
 - **Authors:** James Tavernor, Emily Mower Provost
 - **Subjects:** Subjects:
Sound (cs.SD); Machine Learning (cs.LG); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2509.12295

 - **Pdf link:** https://arxiv.org/pdf/2509.12295

 - **Abstract**
 Speech emotion recognition systems often predict a consensus value generated from the ratings of multiple annotators. However, these models have limited ability to predict the annotation of any one person. Alternatively, models can learn to predict the annotations of all annotators. Adapting such models to new annotators is difficult as new annotators must individually provide sufficient labeled training data. We propose to leverage inter-annotator similarity by using a model pre-trained on a large annotator population to identify a similar, previously seen annotator. Given a new, previously unseen, annotator and limited enrollment data, we can make predictions for a similar annotator, enabling off-the-shelf annotation of unseen data in target datasets, providing a mechanism for extremely low-cost personalization. We demonstrate our approach significantly outperforms other off-the-shelf approaches, paving the way for lightweight emotion adaptation, practical for real-world deployment.
#### FunAudio-ASR Technical Report
 - **Authors:** Keyu An, Yanni Chen, Chong Deng, Changfeng Gao, Zhifu Gao, Bo Gong, Xiangang Li, Yabin Li, Xiang Lv, Yunjie Ji, Yiheng Jiang, Bin Ma, Haoneng Luo, Chongjia Ni, Zexu Pan, Yiping Peng, Zhendong Peng, Peiyao Wang, Hao Wang, Wen Wang, Wupeng Wang, Biao Tian, Zhentao Tan, Nan Yang, Bin Yuan, Jieping Ye, Jixing Yu, Qinglin Zhang, Kun Zou, Han Zhao, Shengkui Zhao, Jingren Zhou
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Artificial Intelligence (cs.AI); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2509.12508

 - **Pdf link:** https://arxiv.org/pdf/2509.12508

 - **Abstract**
 In recent years, automatic speech recognition (ASR) has witnessed transformative advancements driven by three complementary paradigms: data scaling, model size scaling, and deep integration with large language models (LLMs). However, LLMs are prone to hallucination, which can significantly degrade user experience in real-world ASR applications. In this paper, we present FunAudio-ASR, a large-scale, LLM-based ASR system that synergistically combines massive data, large model capacity, LLM integration, and reinforcement learning to achieve state-of-the-art performance across diverse and complex speech recognition scenarios. Moreover, FunAudio-ASR is specifically optimized for practical deployment, with enhancements in streaming capability, noise robustness, code-switching, hotword customization, and satisfying other real-world application requirements. Experimental results show that while most LLM-based ASR systems achieve strong performance on open-source benchmarks, they often underperform on real industry evaluation sets. Thanks to production-oriented optimizations, FunAudio-ASR achieves SOTA performance on real application datasets, demonstrating its effectiveness and robustness in practical settings.
#### PAC: Pronunciation-Aware Contextualized Large Language Model-based Automatic Speech Recognition
 - **Authors:** Li Fu, Yu Xin, Sunlu Zeng, Lu Fan, Youzheng Wu, Xiaodong He
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2509.12647

 - **Pdf link:** https://arxiv.org/pdf/2509.12647

 - **Abstract**
 This paper presents a Pronunciation-Aware Contextualized (PAC) framework to address two key challenges in Large Language Model (LLM)-based Automatic Speech Recognition (ASR) systems: effective pronunciation modeling and robust homophone discrimination. Both are essential for raw or long-tail word recognition. The proposed approach adopts a two-stage learning paradigm. First, we introduce a pronunciation-guided context learning method. It employs an interleaved grapheme-phoneme context modeling strategy that incorporates grapheme-only distractors, encouraging the model to leverage phonemic cues for accurate recognition. Then, we propose a pronunciation-discriminative reinforcement learning method with perturbed label sampling to further enhance the modelś ability to distinguish contextualized homophones. Experimental results on the public English Librispeech and Mandarin AISHELL-1 datasets indicate that PAC: (1) reduces relative Word Error Rate (WER) by 30.2% and 53.8% compared to pre-trained LLM-based ASR models, and (2) achieves 31.8% and 60.5% relative reductions in biased WER for long-tail words compared to strong baselines, respectively.
#### The CCF AATC 2025: Speech Restoration Challenge
 - **Authors:** Junan Zhang, Mengyao Zhu, Xin Xu, Hui Bu, Zhenhua Ling, Zhizheng Wu
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2509.12974

 - **Pdf link:** https://arxiv.org/pdf/2509.12974

 - **Abstract**
 Real-world speech communication is often hampered by a variety of distortions that degrade quality and intelligibility. While many speech enhancement algorithms target specific degradations like noise or reverberation, they often fall short in realistic scenarios where multiple distortions co-exist and interact. To spur research in this area, we introduce the Speech Restoration Challenge as part of the China Computer Federation (CCF) Advanced Audio Technology Competition (AATC) 2025. This challenge focuses on restoring speech signals affected by a composite of three degradation types: (1) complex acoustic degradations including non-stationary noise and reverberation; (2) signal-chain artifacts such as those from MP3 compression; and (3) secondary artifacts introduced by other pre-processing enhancement models. We describe the challenge's background, the design of the task, the comprehensive dataset creation methodology, and the detailed evaluation protocol, which assesses both objective performance and model complexity. Homepage: this https URL.


by Zyzzyva0381 (Windy). 


2025-09-17
