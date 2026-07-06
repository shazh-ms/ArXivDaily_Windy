# Showing new listings for Friday, 3 July 2026
Auto update papers at about 2:30am UTC (10:30am Beijing time) every weekday.


阅读 `Usage.md`了解如何使用此repo实现个性化的Arxiv论文推送

See `Usage.md` for instructions on how to personalize the repo. 


Keyword list: ['text-to-speech', 'text to speech', 'tts', 'LLM-based', 'speech', 'voice']


Excluded: []


### Today: 8papers 
#### Beyond Words: Towards Effective Modeling of Non-Verbal Vocalizations in ASR
 - **Authors:** Gene Yang, Haibin Wu, Peng Su, Ruizhe Huang, Suwon Shon, Bach Do, Minxue Niu, Zhaoheng Ni, Shang-Wen Li, Florian Metze, Yossi Adi, Ming Sun, Yuzong Liu
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2607.01563

 - **Pdf link:** https://arxiv.org/pdf/2607.01563

 - **Abstract**
 Modern automatic speech recognition (ASR) systems excel at transcribing lexical content but often omit nonverbal vocalizations (NVs), such as laughter, breaths, coughs, and cries, that carry conversational and affective information. Modeling NVs in ASR is challenging because NV annotations are sparse and highly long-tailed, with frequent categories such as breaths and laughter dominating rarer events such as cries and coughs. We study three data-centric strategies for improving low-resource NV recognition: (1) a two-stage curriculum that first maps all NV events to a generic token and then fine-tunes on target categories; (2) inter-token transfer from high-resource events, such as laughter and breath, to rare events, such as crying; and (3) voice-conversion augmentation with class balancing. Experiments show that shared acoustic structure across vocal events can be exploited to improve rare-category detection while preserving lexical ASR quality.
#### Enhancing Acoustic-to-Articulatory Inversion with Multi-Target Pretraining for Low-Resource Settings
 - **Authors:** Jesuraj Bandekar, Prasanta Kumar Ghosh
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2607.01594

 - **Pdf link:** https://arxiv.org/pdf/2607.01594

 - **Abstract**
 Acoustic-to-Articulatory Inversion (AAI) estimates vocal tract articulator movements from speech, benefiting tasks like ASR, speech synthesis, and speaker verification. While deep learning-based methods (CNNs, RNNs, Transformers) have advanced AAI, recent studies show that Self-Supervised Learning (SSL) features further enhance performance, particularly in low-resource settings. However, SSL feature extractors introduce inference latency and computational overhead. To address this, we propose a novel pretraining method leveraging three target representations-Phoneme Labels, Articulatory Feature Labels, and Critical-articulator Labels-eliminating the need for an SSL extractor during inference. We evaluate our approach against both baseline and SSL-based models across various data conditions. Results demonstrate that our method consistently improves AAI performance, particularly in low-resource scenarios, while significantly reducing inference costs without sacrificing accuracy.
#### Self-Supervised Test-Time Tuning for Packet Loss Concealment
 - **Authors:** Yehoshua Dissen, Joseph Keshet
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Computation and Language (cs.CL)
 - **Arxiv link:** https://arxiv.org/abs/2607.01823

 - **Pdf link:** https://arxiv.org/pdf/2607.01823

 - **Abstract**
 Packet loss concealment (PLC) reconstructs audio packets that are missing at the receiver, usually with a trained model whose parameters remain fixed at deployment time. This treats the PLC model as static, even though each call or recording exposes signal-specific information through the packets that did arrive. We present TTT-PLC, a self-supervised test-time tuning framework that adapts existing PLC models using only those received packets. The method creates supervision by synthetically masking portions of the available signal, training the model to conceal them with its native PLC objective, and then using the adapted model to reconstruct the true packet losses. No clean reference signal, external adaptation data, or architectural modification is required. We study TTT-PLC in two deployment settings. In the non-causal setting, the received file is available before reconstruction, allowing repeated self-supervised adaptation passes and providing a per-file adaptation ceiling. In the causal setting, audio is streamed without revising emitted samples; adaptation is performed only on completed past blocks, and updated parameters affect only future audio. We instantiate the framework on two public PLC backbones, FRN, a recurrent full-band speech PLC model, and PARCnet, a hybrid autoregressive-neural model for networked music. Across these settings, the results show that pretrained PLC systems do not need to be treated as fixed at inference time, the still-observed portions of a lossy signal can provide an effective training signal for improving concealment on that same signal.
#### An Efficient vLLM-Based Inference Pipeline for Unified Audio Understanding and Generation
 - **Authors:** Haoran Wang, Jinchuan Tian, Siddhant Arora, Shinji Watanabe
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Artificial Intelligence (cs.AI)
 - **Arxiv link:** https://arxiv.org/abs/2607.02119

 - **Pdf link:** https://arxiv.org/pdf/2607.02119

 - **Abstract**
 While Large Multimodal Models excel in comprehension, high-throughput inference engines lack native support for multimodal generation. This is severe in Speech Language Models, where generating multi-layered audio tokens via decoupled AR+NAR or synchronous Multi-Token Prediction (MTP) with delay-pattern interleaving conflicts with standard single-stream loops. We present a vLLM-based inference pipeline for unified speech understanding and generation. We extend autoregressive decoding to natively execute delay-pattern de-interleaving and coordinated multi-stream sampling, integrating an on-GPU acoustic decoder for end-to-end waveform synthesis. Crucially, we overcome the shared intuition that Classifier-Free Guidance (CFG) halves throughput. By co-scheduling paired conditional and unconditional requests within a continuous batch, our CFG implementation sustains 80% of non-CFG throughput, absorbing dual-request and logit merging overheads. We open-source our framework.
#### Spatial Speech Perception Systems: A Survey of Sound Source Localization, Directional Enhancement, and Speech Recognition
 - **Authors:** Pengyuan Shao, Dimitrios Kanoulas
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2607.02296

 - **Pdf link:** https://arxiv.org/pdf/2607.02296

 - **Abstract**
 Robust speech understanding in real-world acoustic environments remains a fundamental challenge for intelligent auditory systems such as robot audition, hearing aids, teleconferencing systems, smart speakers, and voice-controlled assistants. These systems must operate under background noise, reverberation, competing speakers, and dynamic acoustic conditions. Spatial speech perception addresses this challenge by exploiting microphone-array information to localize, enhance, and interpret target speech in complex acoustic scenes. This paper surveys spatial speech perception systems with emphasis on the roles of sound source localization (SSL), directional speech enhancement (DSE), and automatic speech recognition (ASR), both individually and within integrated processing pipelines. We review classical signal-processing approaches and recent learning-based methods for microphone-array localization, beamforming, neural enhancement, speech separation, and modern recognition architectures. Beyond component-level analysis, we discuss robustness to noise and reverberation, multi-speaker operation, real-time constraints, and computational efficiency. We also examine representative applications in robot audition, hearing assistance, smart speakers, and teleconferencing, and identify open challenges and future directions toward robust, low-latency, and perception-aware speech systems for complex acoustic environments.
#### SPARCLE: SPeaker-aware Aligned Representations via Contrastive Language Embeddings
 - **Authors:** Priyam Mazumdar, Yurii Halychanskyi, Steven Guo, Mark Hasegawa-Johnson, Volodymyr Kindratenko
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Artificial Intelligence (cs.AI); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2607.01238

 - **Pdf link:** https://arxiv.org/pdf/2607.01238

 - **Abstract**
 Recent advances in speech synthesis have shifted from phoneme representations to direct grapheme modeling. While phonemes address the one-to-many mapping between text and acoustics, they rely on grapheme-to-phoneme (G2P) systems that fail to capture speaker-specific acoustic variation. Prior work demonstrates that grapheme-based models outperform phoneme-based systems at scale, but not in low-resource settings. In this paper, we propose SPARCLE, a speaker-aware grapheme representation model that enriches characters with their precise acoustic realizations. SPARCLE is trained with a contrastive objective to align graphemes with corresponding Wav2Vec2 acoustic representations while conditioned on speaker identity. The resulting model serves as a replacement to G2P systems for downstream text-to-speech (TTS) tasks. We demonstrate that SPARCLE improves generation quality, reducing word error rates by half in extreme low-resource settings compared to standard grapheme-based models.
#### Rethinking Speech-LLM Integration for ASR: Effective Joint Speech-Text Training by Interleaving
 - **Authors:** Ruchao Fan, Yiming Wang, Rui Zhao, Liliang Ren, Keqi Deng, Xiaoyang Chen, Ali Zare, Bo Ren, Yuxuan Hu, Junkun Chen, Yan Huang, Yelong Shen, Jinyu Li
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2607.01733

 - **Pdf link:** https://arxiv.org/pdf/2607.01733

 - **Abstract**
 Speech-LLM integration has shown promising results by leveraging extensive textual pretraining, yet its specific benefits for automatic speech recognition (ASR) remain unclear. We observe that as supervised ASR training data increases, the contribution of LLM priors becomes less evident, and simple speech-text joint training under-utilizes textual knowledge. We therefore propose Joint Speech-Text Interleaved Pretraining (JSTIP), an ASR-oriented pretraining strategy that constructs word-level and segment-level interleaved speech-text sequences within aligned pairs for speech-LLM architectures that accept continuous inputs. Experiments on 38k hours of ASR data show consistent entity accuracy improvement compared to ASR-only and joint speech-text training baselines. JSTIP achieves on-par entity recognition performance using domain transcription text compared to synthetic speech-text pairs, simplifying domain adaptation. Benefiting from textual pretraining and domain text data, JSTIP is competitive with open-source ASR and Speech-LLM systems in medical entity recognition. The zero-shot speech question answering behaviors further suggest that interleaving reduces the speech-text modality gap and preserves the LLM generative prior, which is likely the reason for the entity improvements on the ASR task.
#### Unlocking Speech-Text Compositional Powers: Instruction-Following Speech Language Models without Instruction Tuning
 - **Authors:** Congrui Du, Yang Zhang, Kaizhi Qian, Shiyu Chang
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2607.02214

 - **Pdf link:** https://arxiv.org/pdf/2607.02214

 - **Abstract**
 Instruction tuning for speech language models (SLMs) is substantially more challenging than for text-based large language models (LLMs), as it requires learning a new modality and a wide range of speech-specific instructions in addition to those supported by text LLMs. Existing SLM training approaches largely replicate the text LLM training paradigm by synthesizing large-scale speech pre-training and instruction-tuning datasets. However, this strategy is difficult to scale, since speech sequences are significantly longer than text sequences. In this paper, we propose SpeechCombine, an instruction-following speech language model trained without any instruction tuning, using only a single round of speech pre-training on 30k hours of data. Starting from a text LLM base model, we perform continuous pre-training on speech utterances to obtain a speech-adapted model, and then directly combine its weights with the weight difference between the instruction-tuned and base versions of the text LLM. Our results show that this simple combination strategy not only preserves the knowledge and capabilities of the original text LLM, but also effectively transfers them to the speech domain. These findings suggest a new direction for SLM training that avoids reliance on massive speech data.


by Zyzzyva0381 (Windy). 


2026-07-06
