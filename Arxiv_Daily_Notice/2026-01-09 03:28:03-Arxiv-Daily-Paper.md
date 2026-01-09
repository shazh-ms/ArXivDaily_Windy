# Showing new listings for Friday, 9 January 2026
Auto update papers at about 2:30am UTC (10:30am Beijing time) every weekday.


阅读 `Usage.md`了解如何使用此repo实现个性化的Arxiv论文推送

See `Usage.md` for instructions on how to personalize the repo. 


Keyword list: ['text-to-speech', 'text to speech', 'tts', 'LLM-based', 'speech', 'voice']


Excluded: []


### Today: 5papers 
#### Latent-Level Enhancement with Flow Matching for Robust Automatic Speech Recognition
 - **Authors:** Da-Hee Yang, Joon-Hyuk Chang
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2601.04459

 - **Pdf link:** https://arxiv.org/pdf/2601.04459

 - **Abstract**
 Noise-robust automatic speech recognition (ASR) has been commonly addressed by applying speech enhancement (SE) at the waveform level before recognition. However, speech-level enhancement does not always translate into consistent recognition improvements due to residual distortions and mismatches with the latent space of the ASR encoder. In this letter, we introduce a complementary strategy termed latent-level enhancement, where distorted representations are refined during ASR inference. Specifically, we propose a plug-and-play Flow Matching Refinement module (FM-Refiner) that operates on the output latents of a pretrained CTC-based ASR encoder. Trained to map imperfect latents-either directly from noisy inputs or from enhanced-but-imperfect speech-toward their clean counterparts, the FM-Refiner is applied only at inference, without fine-tuning ASR parameters. Experiments show that FM-Refiner consistently reduces word error rate, both when directly applied to noisy inputs and when combined with conventional SE front-ends. These results demonstrate that latent-level refinement via flow matching provides a lightweight and effective complement to existing SE approaches for robust ASR.
#### LLMs-Integrated Automatic Hate Speech Recognition Using Controllable Text Generation Models
 - **Authors:** Ryutaro Oshima, Yuya Hosoda, Youji Iiguni
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Artificial Intelligence (cs.AI); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2601.04654

 - **Pdf link:** https://arxiv.org/pdf/2601.04654

 - **Abstract**
 This paper proposes an automatic speech recognition (ASR) model for hate speech using large language models (LLMs). The proposed method integrates the encoder of the ASR model with the decoder of the LLMs, enabling simultaneous transcription and censorship tasks to prevent the exposure of harmful content. Instruction tuning of the LLM to mask hate-related words with specific tokens requires an annotated hate speech dataset, which is limited. We generate text samples using an LLM with the Chain-of-Thought (CoT) prompting technique guided by cultural context and examples and then convert them into speech samples using a text-to-speech (TTS) system. However, some of them contain non-hate speech samples with hate-related words, which degrades the censorship performance. This paper filters the samples which text classification models correctly label as hate content. By adjusting the threshold for the number of correct answer models, we can control the level of hate in the generated dataset, allowing us to train the LLMs through curriculum learning in a gradual manner. Experimental results show that the proposed method achieves a masking accuracy of 58.6\% for hate-related words, surpassing previous baselines. We also confirm that the curriculum training contributes to the efficiency of both transcription and censorship tasks.
#### Defense Against Synthetic Speech: Real-Time Detection of RVC Voice Conversion Attacks
 - **Authors:** Prajwal Chinchmalatpure, Suyash Chinchmalatpure, Siddharth Chavan
 - **Subjects:** Subjects:
Sound (cs.SD); Artificial Intelligence (cs.AI); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2601.04227

 - **Pdf link:** https://arxiv.org/pdf/2601.04227

 - **Abstract**
 Generative audio technologies now enable highly realistic voice cloning and real-time voice conversion, increasing the risk of impersonation, fraud, and misinformation in communication channels such as phone and video calls. This study investigates real-time detection of AI-generated speech produced using Retrieval-based Voice Conversion (RVC), evaluated on the DEEP-VOICE dataset, which includes authentic and voice-converted speech samples from multiple well-known speakers. To simulate realistic conditions, deepfake generation is applied to isolated vocal components, followed by the reintroduction of background ambiance to suppress trivial artifacts and emphasize conversion-specific cues. We frame detection as a streaming classification task by dividing audio into one-second segments, extracting time-frequency and cepstral features, and training supervised machine learning models to classify each segment as real or voice-converted. The proposed system enables low-latency inference, supporting both segment-level decisions and call-level aggregation. Experimental results show that short-window acoustic features can reliably capture discriminative patterns associated with RVC speech, even in noisy backgrounds. These findings demonstrate the feasibility of practical, real-time deepfake speech detection and underscore the importance of evaluating under realistic audio mixing conditions for robust deployment.
#### LEMAS: Large A 150K-Hour Large-scale Extensible Multilingual Audio Suite with Generative Speech Models
 - **Authors:** Zhiyuan Zhao, Lijian Lin, Ye Zhu, Kai Xie, Yunfei Liu, Yu Li
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2601.04233

 - **Pdf link:** https://arxiv.org/pdf/2601.04233

 - **Abstract**
 We present the LEMAS-Dataset, which, to our knowledge, is currently the largest open-source multilingual speech corpus with word-level timestamps. Covering over 150,000 hours across 10 major languages, LEMAS-Dataset is constructed via a efficient data processing pipeline that ensures high-quality data and annotations. To validate the effectiveness of LEMAS-Dataset across diverse generative paradigms, we train two benchmark models with distinct architectures and task specializations on this dataset. LEMAS-TTS, built upon a non-autoregressive flow-matching framework, leverages the dataset's massive scale and linguistic diversity to achieve robust zero-shot multilingual synthesis. Our proposed accent-adversarial training and CTC loss mitigate cross-lingual accent issues, enhancing synthesis stability. Complementarily, LEMAS-Edit employs an autoregressive decoder-only architecture that formulates speech editing as a masked token infilling task. By exploiting precise word-level alignments to construct training masks and adopting adaptive decoding strategies, it achieves seamless, smooth-boundary speech editing with natural transitions. Experimental results demonstrate that models trained on LEMAS-Dataset deliver high-quality synthesis and editing performance, confirming the dataset's quality. We envision that this richly timestamp-annotated, fine-grained multilingual corpus will drive future advances in prompt-based speech generation systems.
#### SmoothSync: Dual-Stream Diffusion Transformers for Jitter-Robust Beat-Synchronized Gesture Generation from Quantized Audio
 - **Authors:** Yujiao Jiang, Qingmin Liao, Zongqing Lu
 - **Subjects:** Subjects:
Sound (cs.SD); Artificial Intelligence (cs.AI); Robotics (cs.RO); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2601.04236

 - **Pdf link:** https://arxiv.org/pdf/2601.04236

 - **Abstract**
 Co-speech gesture generation is a critical area of research aimed at synthesizing speech-synchronized human-like gestures. Existing methods often suffer from issues such as rhythmic inconsistency, motion jitter, foot sliding and limited multi-sampling diversity. In this paper, we present SmoothSync, a novel framework that leverages quantized audio tokens in a novel dual-stream Diffusion Transformer (DiT) architecture to synthesis holistic gestures and enhance sampling variation. Specifically, we (1) fuse audio-motion features via complementary transformer streams to achieve superior synchronization, (2) introduce a jitter-suppression loss to improve temporal smoothness, (3) implement probabilistic audio quantization to generate distinct gesture sequences from identical inputs. To reliably evaluate beat synchronization under jitter, we introduce Smooth-BC, a robust variant of the beat consistency metric less sensitive to motion noise. Comprehensive experiments on the BEAT2 and SHOW datasets demonstrate SmoothSync's superiority, outperforming state-of-the-art methods by -30.6% FGD, 10.3% Smooth-BC, and 8.4% Diversity on BEAT2, while reducing jitter and foot sliding by -62.9% and -17.1% respectively. The code will be released to facilitate future research.


by Zyzzyva0381 (Windy). 


2026-01-09
