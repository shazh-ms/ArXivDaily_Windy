# Showing new listings for Monday, 17 February 2025
Auto update papers at about 2:30am UTC (10:30am Beijing time) every weekday.


阅读 `Usage.md`了解如何使用此repo实现个性化的Arxiv论文推送

See `Usage.md` for instructions on how to personalize the repo. 


Keyword list: ['acoustic', 'reinforcement learning', 'reverb', 'meta', 'separate', 'reconstruction', 'noise', 'enhance', 'localization', 'speech']


Excluded: []


### Today: 8papers 
#### Microphone Array Geometry Independent Multi-Talker Distant ASR: NTT System for the DASR Task of the CHiME-8 Challenge
 - **Authors:** Naoyuki Kamo, Naohiro Tawara, Atsushi Ando, Takatomo Kano, Hiroshi Sato, Rintaro Ikeshita, Takafumi Moriya, Shota Horiguch, Kohei Matsuura, Atsunori Ogawa, Alexis Plaquet, Takanori Ashihara, Tsubasa Ochiai, Masato Mimura, Marc Delcroix, Tomohiro Nakatani, Taichi Asami, Shoko Araki
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Signal Processing (eess.SP)
 - **Arxiv link:** https://arxiv.org/abs/2502.09859

 - **Pdf link:** https://arxiv.org/pdf/2502.09859

 - **Abstract**
 In this paper, we introduce a multi-talker distant automatic speech recognition (DASR) system we designed for the DASR task 1 of the CHiME-8 challenge. Our system performs speaker counting, diarization, and ASR. It handles various recording conditions, from diner parties to professional meetings and from two to eight speakers. We perform diarization first, followed by speech enhancement, and then ASR as the challenge baseline. However, we introduced several key refinements. First, we derived a powerful speaker diarization relying on end-to-end speaker diarization with vector clustering (EEND-VC), multi-channel speaker counting using enhanced embeddings from EEND-VC, and target-speaker voice activity detection (TS-VAD). For speech enhancement, we introduced a novel microphone selection rule to better select the most relevant microphones among the distributed microphones and investigated improvements to beamforming. Finally, for ASR, we developed several models exploiting Whisper and WavLM speech foundation models. We present the results we submitted to the challenge and updated results we obtained afterward. Our strongest system achieves a 63% relative macro tcpWER improvement over the baseline and outperforms the challenge best results on the NOTSOFAR-1 meeting evaluation data among geometry-independent systems.
#### SIToBI - A Speech Prosody Annotation Tool for Indian Languages
 - **Authors:** Preethi Thinakaran, Malarvizhi Muthuramalingam, Sooriya S, Anushiya Rachel Gladston, P. Vijayalakshmi, Hema A Murthy, T. Nagarajan
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2502.09661

 - **Pdf link:** https://arxiv.org/pdf/2502.09661

 - **Abstract**
 The availability of prosodic information from speech signals is useful in a wide range of applications. However, deriving this information from speech signals can be a laborious task involving manual intervention. Therefore, the current work focuses on developing a tool that can provide prosodic annotations corresponding to a given speech signal, particularly for Indian languages. The proposed Segmentation with Intensity, Tones and Break Indices (SIToBI) tool provides time-aligned phoneme, syllable, and word transcriptions, syllable-level pitch contour annotations, break indices, and syllable-level relative intensity indices. The tool focuses more on syllable-level annotations since Indian languages are syllable-timed. Indians, regardless of the language they speak, may exhibit influences from other languages. As a result, other languages spoken in India may also exhibit syllable-timed characteristics. The accuracy of the annotations derived from the tool is analyzed by comparing them against manual annotations and the tool is observed to perform well. While the current work focuses on three languages, namely, Tamil, Hindi, and Indian English, the tool can easily be extended to other Indian languages and possibly other syllable-timed languages as well.
#### Improving Acoustic Side-Channel Attacks on Keyboards Using Transformers and Large Language Models
 - **Authors:** Jin Hyun Park, Seyyed Ali Ayati, Yichen Cai
 - **Subjects:** Subjects:
Machine Learning (cs.LG); Artificial Intelligence (cs.AI); Computation and Language (cs.CL); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2502.09782

 - **Pdf link:** https://arxiv.org/pdf/2502.09782

 - **Abstract**
 The increasing prevalence of microphones in everyday devices and the growing reliance on online services have amplified the risk of acoustic side-channel attacks (ASCAs) targeting keyboards. This study explores deep learning techniques, specifically vision transformers (VTs) and large language models (LLMs), to enhance the effectiveness and applicability of such attacks. We present substantial improvements over prior research, with the CoAtNet model achieving state-of-the-art performance. Our CoAtNet shows a 5.0% improvement for keystrokes recorded via smartphone (Phone) and 5.9% for those recorded via Zoom compared to previous benchmarks. We also evaluate transformer architectures and language models, with the best VT model matching CoAtNet's performance. A key advancement is the introduction of a noise mitigation method for real-world scenarios. By using LLMs for contextual understanding, we detect and correct erroneous keystrokes in noisy environments, enhancing ASCA performance. Additionally, fine-tuned lightweight language models with Low-Rank Adaptation (LoRA) deliver comparable performance to heavyweight models with 67X more parameters. This integration of VTs and LLMs improves the practical applicability of ASCA mitigation, marking the first use of these technologies to address ASCAs and error correction in real-world scenarios.
#### A Preliminary Exploration with GPT-4o Voice Mode
 - **Authors:** Yu-Xiang Lin, Chih-Kai Yang, Wei-Chih Chen, Chen-An Li, Chien-yu Huang, Xuanjun Chen, Hung-yi Lee
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2502.09940

 - **Pdf link:** https://arxiv.org/pdf/2502.09940

 - **Abstract**
 With the rise of multimodal large language models, GPT-4o stands out as a pioneering model, driving us to evaluate its capabilities. This report assesses GPT-4o across various tasks to analyze its audio processing and reasoning abilities. We find that GPT-4o exhibits strong knowledge in audio, speech, and music understanding, performing well in tasks like intent classification, spoken command classification, semantic and grammatical reasoning., multilingual speech recognition, and singing analysis. It also shows greater robustness against hallucinations than other large audio-language models (LALMs). However, it struggles with tasks such as audio duration prediction and instrument classification. Additionally, GPT-4o's safety mechanisms cause it to decline tasks like speaker identification, age classification, MOS prediction, and audio deepfake detection. Notably, the model exhibits a significantly different refusal rate when responding to speaker verification tasks on different datasets. This is likely due to variations in the accompanying instructions or the quality of the input audio, suggesting the sensitivity of its built-in safeguards. Finally, we acknowledge that model performance varies with evaluation protocols. This report only serves as a preliminary exploration of the current state of LALMs.
#### MTLM: an Innovative Language Model Training Paradigm for ASR
 - **Authors:** Qingliang Meng, Pengju Ren, Tian Li, Changsong Dai
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2502.10058

 - **Pdf link:** https://arxiv.org/pdf/2502.10058

 - **Abstract**
 Pre-training Transformer-based language models (LMs) on a large amount of text has proven crucial for improving automatic speech recognition (ASR) performance. Generally, traditional LMs are unidirectional and unable to access the context on the right. This paper proposes a method for training LMs that enable traditional unidirectional LMs to fully utilize left and right contexts. Compared with the unidirectional LMs, our LM facilitates ASR to transcribe hypotheses more consistently and in a more semantically unambiguous way, as it incorporates richer contextual representations. Finally, our experimental results on the LibriSpeech corpus demonstrate that our model outperforms traditional unidirectional LMs, whether n-best rescoring or shallow fusion is used as the decoding algorithm.
#### VocalCrypt: Novel Active Defense Against Deepfake Voice Based on Masking Effect
 - **Authors:** Qingyuan Fei, Wenjie Hou, Xuan Hai, Xin Liu
 - **Subjects:** Subjects:
Sound (cs.SD); Cryptography and Security (cs.CR); Multimedia (cs.MM); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2502.10329

 - **Pdf link:** https://arxiv.org/pdf/2502.10329

 - **Abstract**
 The rapid advancements in AI voice cloning, fueled by machine learning, have significantly impacted text-to-speech (TTS) and voice conversion (VC) fields. While these developments have led to notable progress, they have also raised concerns about the misuse of AI VC technology, causing economic losses and negative public perceptions. To address this challenge, this study focuses on creating active defense mechanisms against AI VC systems. We propose a novel active defense method, VocalCrypt, which embeds pseudo-timbre (jamming information) based on SFS into audio segments that are imperceptible to the human ear, thereby forming systematic fragments to prevent voice cloning. This approach protects the voice without compromising its quality. In comparison to existing methods, such as adversarial noise incorporation, VocalCrypt significantly enhances robustness and real-time performance, achieving a 500\% increase in generation speed while maintaining interference effectiveness. Unlike audio watermarking techniques, which focus on post-detection, our method offers preemptive defense, reducing implementation costs and enhancing feasibility. Extensive experiments using the Zhvoice and VCTK Corpus datasets show that our AI-cloned speech defense system performs excellently in automatic speaker verification (ASV) tests while preserving the integrity of the protected audio.
#### CLaMP 3: Universal Music Information Retrieval Across Unaligned Modalities and Unseen Languages
 - **Authors:** Shangda Wu, Zhancheng Guo, Ruibin Yuan, Junyan Jiang, Seungheon Doh, Gus Xia, Juhan Nam, Xiaobing Li, Feng Yu, Maosong Sun
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2502.10362

 - **Pdf link:** https://arxiv.org/pdf/2502.10362

 - **Abstract**
 CLaMP 3 is a unified framework developed to address challenges of cross-modal and cross-lingual generalization in music information retrieval. Using contrastive learning, it aligns all major music modalities--including sheet music, performance signals, and audio recordings--with multilingual text in a shared representation space, enabling retrieval across unaligned modalities with text as a bridge. It features a multilingual text encoder adaptable to unseen languages, exhibiting strong cross-lingual generalization. Leveraging retrieval-augmented generation, we curated M4-RAG, a web-scale dataset consisting of 2.31 million music-text pairs. This dataset is enriched with detailed metadata that represents a wide array of global musical traditions. To advance future research, we release WikiMT-X, a benchmark comprising 1,000 triplets of sheet music, audio, and richly varied text descriptions. Experiments show that CLaMP 3 achieves state-of-the-art performance on multiple MIR tasks, significantly surpassing previous strong baselines and demonstrating excellent generalization in multimodal and multilingual music contexts.
#### OWLS: Scaling Laws for Multilingual Speech Recognition and Translation Models
 - **Authors:** William Chen, Jinchuan Tian, Yifan Peng, Brian Yan, Chao-Han Huck Yang, Shinji Watanabe
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Artificial Intelligence (cs.AI); Machine Learning (cs.LG); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2502.10373

 - **Pdf link:** https://arxiv.org/pdf/2502.10373

 - **Abstract**
 Neural scaling laws offer valuable insights for designing robust sequence processing architectures. While these laws have been extensively characterized in other modalities, their behavior in speech remains comparatively underexplored. In this work, we introduce OWLS, an open-access, reproducible suite of multilingual speech recognition and translation models spanning 0.25B to 18B parameters, with the 18B version being the largest speech model, to the best of our knowledge. OWLS leverages up to 360K hours of public speech data across 150 languages, enabling a systematic investigation into how data, model, and compute scaling each influence performance in multilingual speech tasks. We use OWLS to derive neural scaling laws, showing how final performance can be reliably predicted when scaling. One of our key findings is that scaling enhances performance on low-resource languages/dialects, helping to mitigate bias and improve the accessibility of speech technologies. Finally, we show how OWLS can be used to power new research directions by discovering emergent abilities in large-scale speech models. Model checkpoints will be released on this https URL for future studies.


by Zyzzyva0381 (Windy). 


2025-02-18
