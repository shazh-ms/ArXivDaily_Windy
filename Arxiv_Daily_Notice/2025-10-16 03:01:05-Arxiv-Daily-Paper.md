# Showing new listings for Thursday, 16 October 2025
Auto update papers at about 2:30am UTC (10:30am Beijing time) every weekday.


阅读 `Usage.md`了解如何使用此repo实现个性化的Arxiv论文推送

See `Usage.md` for instructions on how to personalize the repo. 


Keyword list: ['text-to-speech', 'text to speech', 'tts', 'LLM-based', 'speech', 'voice']


Excluded: []


### Today: 8papers 
#### Automatic Speech Recognition in the Modern Era: Architectures, Training, and Evaluation
 - **Authors:** Md. Nayeem, Md Shamse Tabrej, Kabbojit Jit Deb, Shaonti Goswami, Md. Azizul Hakim
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Artificial Intelligence (cs.AI); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2510.12827

 - **Pdf link:** https://arxiv.org/pdf/2510.12827

 - **Abstract**
 Automatic Speech Recognition (ASR) has undergone a profound transformation over the past decade, driven by advances in deep learning. This survey provides a comprehensive overview of the modern era of ASR, charting its evolution from traditional hybrid systems, such as Gaussian Mixture Model-Hidden Markov Models (GMM-HMMs) and Deep Neural Network-HMMs (DNN-HMMs), to the now-dominant end-to-end neural architectures. We systematically review the foundational end-to-end paradigms: Connectionist Temporal Classification (CTC), attention-based encoder-decoder models, and the Recurrent Neural Network Transducer (RNN-T), which established the groundwork for fully integrated speech-to-text systems. We then detail the subsequent architectural shift towards Transformer and Conformer models, which leverage self-attention to capture long-range dependencies with high computational efficiency. A central theme of this survey is the parallel revolution in training paradigms. We examine the progression from fully supervised learning, augmented by techniques like SpecAugment, to the rise of self-supervised learning (SSL) with foundation models such as wav2vec 2.0, which drastically reduce the reliance on transcribed data. Furthermore, we analyze the impact of largescale, weakly supervised models like Whisper, which achieve unprecedented robustness through massive data diversity. The paper also covers essential ecosystem components, including key datasets and benchmarks (e.g., LibriSpeech, Switchboard, CHiME), standard evaluation metrics (e.g., Word Error Rate), and critical considerations for real-world deployment, such as streaming inference, on-device efficiency, and the ethical imperatives of fairness and robustness. We conclude by outlining open challenges and future research directions.
#### HyWA: Hypernetwork Weight Adapting Personalized Voice Activity Detection
 - **Authors:** Mahsa Ghazvini Nejad, Hamed Jafarzadeh Asl, Amin Edraki, Mohammadreza Sadeghi, Masoud Asgharian, Yuanhao Yu, Vahid Partovi Nia
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Artificial Intelligence (cs.AI); Machine Learning (cs.LG); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2510.12947

 - **Pdf link:** https://arxiv.org/pdf/2510.12947

 - **Abstract**
 Personalized Voice Activity Detection (PVAD) systems activate only in response to a specific target speaker by incorporating speaker embeddings from enrollment utterances. Unlike existing methods that require architectural changes, such as FiLM layers, our approach employs a hypernetwork to modify the weights of a few selected layers within a standard voice activity detection (VAD) model. This enables speaker conditioning without changing the VAD architecture, allowing the same VAD model to adapt to different speakers by updating only a small subset of the layers. We propose HyWA-PVAD, a hypernetwork weight adaptation method, and evaluate it against multiple baseline conditioning techniques. Our comparison shows consistent improvements in PVAD performance. HyWA also offers practical advantages for deployment by preserving the core VAD architecture. Our new approach improves the current conditioning techniques in two ways: i) increases the mean average precision, ii) simplifies deployment by reusing the same VAD architecture.
#### Continuous-Token Diffusion for Speaker-Referenced TTS in Multimodal LLMs
 - **Authors:** Xinlu He, Swayambhu Nath Ray, Harish Mallidi, Jia-Hong Huang, Ashwin Bellur, Chander Chandak, M. Maruf, Venkatesh Ravichandran
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2510.12995

 - **Pdf link:** https://arxiv.org/pdf/2510.12995

 - **Abstract**
 Unified architectures in multimodal large language models (MLLM) have shown promise in handling diverse tasks within a single framework. In the text-to-speech (TTS) task, current MLLM-based approaches rely on discrete token representations, which disregard the inherently continuous nature of speech and can lead to loss of fine-grained acoustic this http URL this work, we investigate the TTS within the MLLM paradigm using continuous speech representations. We design a dual-head architecture and implement two complementary training strategies for a robust model. (1) A diffusion head generating continuous speech representations is added on the MLLM, which is on frame-level and strictly autoregressive. (2) The original language model head is retained to preserve multitask capability and to control the start and end of speech synthesis. (3) Masked training is employed to address exposure bias in autoregressive decoding. (4) To stabilize optimization, we propose a two-stage scheme where the LM is frozen in the second stage, ensuring the diffusion head learns from a fixed input distribution. Evaluations on LibriSpeech(PC) test-clean show that our approach achieves state-of-the-art autoregressive performance, with a WER of 1.95%, speaker similarity of 0.54, and UTMOS of 4.00. The two-stage training yields a 46% relative WER reduction over the one-stage training baseline. These results highlight the effectiveness of combining autoregressive modeling with continuous-token diffusion, supported by a two-stage training procedure.
#### Acoustic Teleportation via Disentangled Neural Audio Codec Representations
 - **Authors:** Philipp Grundhuber, Mhd Modar Halimeh, Emanuël A. P. Habets
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2510.13221

 - **Pdf link:** https://arxiv.org/pdf/2510.13221

 - **Abstract**
 This paper presents an approach for acoustic teleportation by disentangling speech content from acoustic environment characteristics in neural audio codec representations. Acoustic teleportation transfers room characteristics between speech recordings while preserving content and speaker identity. We build upon previous work using the EnCodec architecture, achieving substantial objective quality improvements with non-intrusive ScoreQ scores of 3.03, compared to 2.44 for prior methods. Our training strategy incorporates five tasks: clean reconstruction, reverberated reconstruction, dereverberation, and two variants of acoustic teleportation. We demonstrate that temporal downsampling of the acoustic embedding significantly degrades performance, with even 2x downsampling resulting in a statistically significant reduction in quality. The learned acoustic embeddings exhibit strong correlations with RT60. Effective disentanglement is demonstrated using t-SNE clustering analysis, where acoustic embeddings cluster by room while speech embeddings cluster by speaker.
#### Two Heads Are Better Than One: Audio-Visual Speech Error Correction with Dual Hypotheses
 - **Authors:** Sungnyun Kim, Kangwook Jang, Sungwoo Cho, Joon Son Chung, Hoirin Kim, Se-Young Yun
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Computation and Language (cs.CL); Machine Learning (cs.LG)
 - **Arxiv link:** https://arxiv.org/abs/2510.13281

 - **Pdf link:** https://arxiv.org/pdf/2510.13281

 - **Abstract**
 This paper introduces a new paradigm for generative error correction (GER) framework in audio-visual speech recognition (AVSR) that reasons over modality-specific evidences directly in the language space. Our framework, DualHyp, empowers a large language model (LLM) to compose independent N-best hypotheses from separate automatic speech recognition (ASR) and visual speech recognition (VSR) models. To maximize the effectiveness of DualHyp, we further introduce RelPrompt, a noise-aware guidance mechanism that provides modality-grounded prompts to the LLM. RelPrompt offers the temporal reliability of each modality stream, guiding the model to dynamically switch its focus between ASR and VSR hypotheses for an accurate correction. Under various corruption scenarios, our framework attains up to 57.7% error rate gain on the LRS2 benchmark over standard ASR baseline, contrary to single-stream GER approaches that achieve only 10% gain. To facilitate research within our DualHyp framework, we release the code and the dataset comprising ASR and VSR hypotheses at this https URL.
#### Gelina: Unified Speech and Gesture Synthesis via Interleaved Token Prediction
 - **Authors:** Téo Guichoux, Théodor Lemerle, Shivam Mehta, Jonas Beskow, Gustave Eje Henter, Laure Soulier, Catherine Pelachaud, Nicolas Obin
 - **Subjects:** Subjects:
Sound (cs.SD); Artificial Intelligence (cs.AI); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2510.12834

 - **Pdf link:** https://arxiv.org/pdf/2510.12834

 - **Abstract**
 Human communication is multimodal, with speech and gestures tightly coupled, yet most computational methods for generating speech and gestures synthesize them sequentially, weakening synchrony and prosody alignment. We introduce Gelina, a unified framework that jointly synthesizes speech and co-speech gestures from text using interleaved token sequences in a discrete autoregressive backbone, with modality-specific decoders. Gelina supports multi-speaker and multi-style cloning and enables gesture-only synthesis from speech inputs. Subjective and objective evaluations demonstrate competitive speech quality and improved gesture generation over unimodal baselines.
#### Adaptive vector steering: A training-free, layer-wise intervention for hallucination mitigation in large audio and multimodal models
 - **Authors:** Tsung-En Lin, Kuan-Yi Lee, Hung-Yi Lee
 - **Subjects:** Subjects:
Sound (cs.SD); Machine Learning (cs.LG); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2510.12851

 - **Pdf link:** https://arxiv.org/pdf/2510.12851

 - **Abstract**
 Large Audio-Language Models and Multi-Modal Large Language Models have demonstrated strong capabilities in tasks such as Audio Question Answering (AQA), Audio Captioning, and Automatic Speech Recognition (ASR). However, there is growing evidence that these models can hallucinate about the content of the audio. To address this issue, we probe the models' internal states and propose Adaptive Vector Steering (AVS), a method that better grounds generation in audio content. We also identify a strong correlation between output correctness and internal representations. Experiments show consistent performance gains across two models and two benchmarks. On the Audio Hallucination QA dataset, our method boosts the F1-score of Gemma from 0.550 to 0.619 and Qwen from 0.626 to 0.632. Furthermore, our method increases the accuracy of Qwen on MMAU from 0.548 to 0.592, marking an 8% relative increase. To the best of our knowledge, this is the first work to apply vector steering to mitigate hallucination in audio.
#### Closing the Gap Between Text and Speech Understanding in LLMs
 - **Authors:** Santiago Cuervo, Skyler Seto, Maureen de Seyssel, Richard He Bai, Zijin Gu, Tatiana Likhomanenko, Navdeep Jaitly, Zakaria Aldeneh
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Artificial Intelligence (cs.AI); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2510.13632

 - **Pdf link:** https://arxiv.org/pdf/2510.13632

 - **Abstract**
 Large Language Models (LLMs) can be adapted to extend their text capabilities to speech inputs. However, these speech-adapted LLMs consistently underperform their text-based counterparts--and even cascaded pipelines--on language understanding tasks. We term this shortfall the text-speech understanding gap: the performance drop observed when a speech-adapted LLM processes spoken inputs relative to when the original text-based LLM processes the equivalent text. Recent approaches to narrowing this gap either rely on large-scale speech synthesis of text corpora, which is costly and heavily dependent on synthetic data, or on large-scale proprietary speech datasets, which are not reproducible. As a result, there remains a need for more data-efficient alternatives for closing the text-speech understanding gap. In this work, we analyze the gap as driven by two factors: (i) forgetting of text capabilities during adaptation, and (ii) cross-modal misalignment between speech and text. Based on this analysis, we introduce SALAD--Sample-efficient Alignment with Learning through Active selection and cross-modal Distillation--which combines cross-modal distillation with targeted synthetic data to improve alignment while mitigating forgetting. Applied to 3B and 7B LLMs, SALAD achieves competitive performance with a strong open-weight model across broad-domain benchmarks in knowledge, language understanding, and reasoning, while training on over an order of magnitude less speech data from public corpora.


by Zyzzyva0381 (Windy). 


2025-10-16
