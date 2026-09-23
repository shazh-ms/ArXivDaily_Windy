# Showing new listings for Wednesday, 23 September 2026
Auto update papers at about 2:30am UTC (10:30am Beijing time) every weekday.


阅读 `Usage.md`了解如何使用此repo实现个性化的Arxiv论文推送

See `Usage.md` for instructions on how to personalize the repo. 


Keyword list: ['text-to-speech', 'text to speech', 'tts', 'LLM-based', 'speech', 'voice']


Excluded: []


### Today: 16papers 
#### Beyond Short Segments : Expanding Speaker Embeddings with Vector Archives
 - **Authors:** Hyunku Kang, Minkyu Cho, Chanwoo Kim
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Computation and Language (cs.CL); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2609.25007

 - **Pdf link:** https://arxiv.org/pdf/2609.25007

 - **Abstract**
 The performance of state-of-the-art speaker verification (SV) systems severely degrades on short utterances due to insufficient speaker-specific information. To address this critical challenge, we propose the Vector Archive Mapping ECAPA (VAM-ECAPA), a novel system designed to enhance feature extraction from short-duration speech. The core of our system is the Transformer-based Vector Archive Mapping with Statistical Pooling (TVAMSP) module, which enriches information-scarce features by mapping them against a learnable Vector Archive of canonical speaker traits. By integrating the TVAMSP module into a strong WavLM+ECAPA-TDNN baseline, our system learns to map sparse features from short segments into robust, discriminative speaker representations. Experiments on the VoxCeleb1 benchmark show that our proposed VAM-ECAPA achieves a highly competitive EER of 8.334% on 1-second test segments, a 54.8% relative error reduction compared to a conventionally-trained baseline.
#### Qwen-Audio-3.1-Realtime: Towards Reliable Agentic Voice Interaction
 - **Authors:** Lujia Bao, Qian Chen, Luyao Cheng, Chong Deng, Yuxiang Kong, Xiangang Li, Xu Li, Jiaqing Liu, Chao-Hong Tan, Haoyu Wang, Wen Wang, Xilou Wang, Junhao Xu, Liang Yi, Binbin Zhang, Qinglin Zhang, Qiquan Zhang
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Artificial Intelligence (cs.AI); Computation and Language (cs.CL); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2609.25176

 - **Pdf link:** https://arxiv.org/pdf/2609.25176

 - **Abstract**
 Real-time voice assistants must reason over evolving requests, execute actions, and follow conversational rules. Qwen-Audio-3.1-Realtime brings these requirements together through Think, Act, and Speak and Coordinate. Think combines Core-Cocktail supervised fine-tuning with Multimodality and Multi-Teacher On-Policy Distillation (M$^{2}$-OPD) to transfer language capabilities and develop native audio skills. Act uses self-evolving executable environments and multi-granularity rollouts for Group Relative Policy Optimization (GRPO), teaching the model to use tools, interpret feedback, and complete tasks. Speak and Coordinate aligns how, when, and whether the assistant speaks or acts. We evaluate audio reasoning, multilingual understanding, tool use, conversational behavior, full-duplex interaction, and safety. Compared with Qwen-Audio-3.0-Realtime, 3.1 raises overall task success from 78.4% to 82.0% on our half-duplex speech-to-text adaptation of $\tau$-Voice. On speech-to-speech Full-Duplex-Bench v1.5, the response rate to background speech falls from 73.0% to 13.0%. We also present a separate Voice Harness prototype, using Qwen-Audio-3.0-Realtime as its foreground, that extends spoken interaction to persistent tasks through foreground--background coordination and memory.
#### Qwen-Audio-Agent Technical Report
 - **Authors:** Chong Deng, Yunjie Ji, Yuxiang Kong, Xiangang Li, Xu Li, Binbin Zhang, Haina Zhu, Jianheng Zhuo
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Multiagent Systems (cs.MA)
 - **Arxiv link:** https://arxiv.org/abs/2609.25195

 - **Pdf link:** https://arxiv.org/pdf/2609.25195

 - **Abstract**
 We present Qwen-Audio-Agent, a harness that combines full-duplex voice interaction with asynchronous task execution through a foreground-background architecture. A Frontend Agent manages dialogue and selects between direct tool use and delegation, while a Backend Agent carries out delegated tasks in a separate context. An Orchestration Runtime maintains task state, coordinates requests for user input and authorization, and schedules the return of results to the conversation. The runtime separates speech interruption from task cancellation and execution completion from result delivery, allowing conversation to continue while delegated work proceeds. Environmental events and persistent memory provide context within and across sessions. Independent adapters support integration with different frontend models, backend agents, and clients. We instantiate the architecture in desktop assistance, intelligent cockpits, and voice customer service. On an in-house cockpit benchmark of 134 cases, mixed execution achieves a task success rate of 91.04%, compared with 72.39% and 80.60% for the direct and all delegated configurations, respectively. In a separate latency evaluation on matched successful turns, mixed execution reduces mean task execution latency by 26.73% and 30.91% relative to these baselines, respectively. These results support the complementary use of direct tool calls for immediate operations and backend delegation for multi-step tasks.
#### SPADE: A Multilingual Dataset for Speech Partial Deepfake Detection and Localization
 - **Authors:** Yuan Tseng, Aishwarya Fursule, Andrew Zijun Ma, Vamshi Nallaguntla, Anderson Avila, Shruti Kshirsagar, David Harwath
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2609.25197

 - **Pdf link:** https://arxiv.org/pdf/2609.25197

 - **Abstract**
 Recent improvements in voice-cloning speech generation systems raise concerns about misuse by malicious actors to impersonate others and spread misinformation. Detecting such tampering is difficult, since deepfakes in the wild may be created by different generative models in a wide range of languages. Furthermore, the speech audio may also only be partially modified, presenting a different and potentially more challenging task than detecting fully-synthetic speech waveforms. To enable further research in this direction, we propose a multilingual dataset for detection and localization of partially edited speech samples. Our dataset includes speech in 12 languages, generated by up to five systems per language, and includes both a training set as well as an evaluation benchmark. To showcase the utility of our proposed dataset, we train localization models of existing architectures and study generalization across three axes: across different languages, across different speech synthesis and editing systems, and across different acoustic environments. Our results show that localization models almost always generalize poorly to speech edited by systems not seen during training. On the other hand, generalization to edited speech in unseen languages still degrades performance but to a lesser extent. We also augment our testing sets with noise to evaluate generalization across acoustic environments, and find that performance of localization models degrade significantly when tested on different acoustic conditions. All together, our results imply that existing deepfake speech detection methods are insufficient for reliably detecting edit-based speech deepfakes in various scenarios unseen during training. SPADE is publicly available on HuggingFace.
#### Learnable Classifier-Free Guidance Null Embeddings for Enhanced Controllable Speech Synthesis
 - **Authors:** Biel Tura Vecino, Yoach Lacombe, Julian Weber, Zbigniew Łatka, Haitong Zhang, Logan Hart, Eren Gölge
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Machine Learning (cs.LG)
 - **Arxiv link:** https://arxiv.org/abs/2609.25411

 - **Pdf link:** https://arxiv.org/pdf/2609.25411

 - **Abstract**
 Classifier-free Guidance (CFG) is widely adopted in text-to-speech (TTS) systems to enhance generation quality and conditioning fidelity by interpolating between conditioned and unconditioned predictions. A common unconditional technique is to use an empty representation, in the form of a fixed null vector. In this work, we propose replacing this representation with a learnable unconditional embedding, optimized to represent a meaningful unconditional state. Objective and subjective evaluations demonstrate that learnable null embeddings consistently outperform fixed null embeddings across speaker similarity, speech stability, and expressiveness, while exhibiting greater robustness to larger guidance scales. We further show that learning a distinct unconditional embedding for each of the TTS conditioning modalities allows fine-grained control over speaker and text guidance, showcasing the trade-off between similarity and quality, and stability and expressiveness in the generated speech.
#### SRF-SVB: Style-Consistent Singing Voice Beautifying via Rectified Flow
 - **Authors:** Wenhui Li, Biao Dong, Liwei Hu, Jiqing Han, Yongjun He
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2609.25610

 - **Pdf link:** https://arxiv.org/pdf/2609.25610

 - **Abstract**
 Singing voice beautifying (SVB) aims to correct pitch and rhythm of amateur singing while enhancing vocal quality, preserving lyrics and the singer's timbre. Existing methods, however, suffer from limited generation quality and efficiency, and tend to neglect the preservation of the singer's style. We propose SRF-SVB, a style-consistent model for SVB via rectified flow, which achieves high-fidelity and efficient beautification covering pitch and rhythm correction. Furthermore, we design a context-guided masked mel-spectrogram inpainting mechanism that effectively preserves the amateur singer's style, including unique timbre and expressive patterns. Experiments on both English and Chinese test sets show that SRF-SVB outperforms baseline models in most objective and subjective metrics.
#### Interactive TTS: Dynamic Speaking Style Adaptation for Expressive Speech Synthesis
 - **Authors:** Wenjie Tian, Kangxiang Xia, Jingbin Hu, Xinfa Zhu, HangRui Hu, Ziyue Jiang, Kexin Huang, Ting He, Lei Xie, Jin Xu
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD); Image and Video Processing (eess.IV)
 - **Arxiv link:** https://arxiv.org/abs/2609.25707

 - **Pdf link:** https://arxiv.org/pdf/2609.25707

 - **Abstract**
 Dynamic speaking style adaptation in multi-turn multimodal interaction remains a major challenge for text-to-speech (TTS) systems. Existing context-aware TTS (CTTS) methods typically map dialogue context to speech in an end-to-end manner. Such implicit modeling makes contextual style decisions difficult to supervise, while the entanglement of style, timbre, and content often leads to weak instruction-following and severe timbre drift across turns. To overcome these limitations, we propose Interactive TTS, a dynamic, style-adaptive framework for contextually appropriate and speaker-consistent speech generation. Interactive TTS decouples the process by explicitly modeling contextual style decisions as executable instructions. To bridge the gap between style decisions and speech generation, we introduce Iterative Rejection Sampling Fine-Tuning (Iterative RSFT) and Context-Aware Direct Preference Optimization (CADPO), which significantly enhance instruction-following and align the generated speech with conversational contexts. Extensive experiments demonstrate that Interactive TTS outperforms state-of-the-art models on VStyle and SpeechParaling-Bench. Demo is available at this https URL
#### SE-MSB: End-to-End Unpaired Speech Enhancement using Mamba Schrödinger Bridges
 - **Authors:** Andreas Bagge, Andreas Nymand, Michael Riis Andersen, Bjørn Sand Jensen
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Artificial Intelligence (cs.AI); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2609.26000

 - **Pdf link:** https://arxiv.org/pdf/2609.26000

 - **Abstract**
 Speech enhancement (SE) models typically rely on supervised learning with paired data examples where clean speech is synthetically degraded. This paradigm limits performance in real-world scenarios where the target environment's specific acoustic characteristics are unknown. We propose a fully unpaired SE framework that uses principled Diffusion Schrödinger Bridges (DSB) to learn a stochastic transport process between a clean and a degraded speech distribution. Algorithms for learning transport maps are computationally heavy since they require simulating differential equations during training, usually at each training step. Therefore, we propose using a high-efficiency Mamba Diffusion Model designed for end-to-end waveform processing. We compare against state-of-the-art methods for speech enhancement, both paired and unpaired, as well as a classical signal processing algorithm. Experimental results show that we are on par or better than the baselines while being orders of magnitude faster during inference. Furthermore, we show that the flexibility of the DSB formulation allows our model to generalize across SE tasks, offering a robust and efficient solution for real-world speech restoration.
#### A Hybrid Classical-Learning Framework for Adaptive Decision Directed Speech Enhancement
 - **Authors:** Ali Rajabi, Xiangwei Zhou
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2609.26183

 - **Pdf link:** https://arxiv.org/pdf/2609.26183

 - **Abstract**
 Speech enhancement aims to recover clean speech signals from noisy observations while preserving speech quality and intelligibility. Classical methods such as Spectral Subtraction and Decision-Directed (DD) enhancement remain widely used because of their interpretability and low computational complexity, but they may suffer from musical-noise artifacts or excessive attenuation of weak speech components under low signal-to-noise ratio (SNR) conditions. This paper proposes an Adaptive Beta-Constrained Decision-Directed (ABCDD) speech enhancement framework that extends the conventional DD method through a frame-dependent lower gain bound. The introduced beta parameter controls the tradeoff between noise suppression and speech preservation. To automate parameter selection for large and diverse datasets, a lightweight multilayer perceptron (MLP) model is further developed to predict frame-level beta values directly from noisy-speech features. The proposed framework is evaluated using both a representative speech example and large-scale testing on the VoiceBank-DEMAND dataset. In the representative example, ABCDD outperformed conventional Spectral Subtraction and classical DD across multiple objective metrics, including SNR, Log-Spectral Distance (LSD), Root-Mean-Square Error (RMSE), correlation, and Scale-Invariant Signal-to-Distortion Ratio (SI-SDR). On 100 unseen VoiceBank-DEMAND test files, the proposed MLP-beta ABCDD method improved average scale-aligned SNR from 9.41 dB to 13.82 dB, corresponding to an average gain of 4.41 dB. The results indicate that combining interpretable classical enhancement structure with lightweight machine-learning-based parameter adaptation provides an effective and practical direction for robust speech enhancement.
#### Persistent Delivery Optimization for Streaming Speech-to-Text Translation with Revisions
 - **Authors:** Zixiang Wan, Delin Chen, Wei Shi, Haihua Xu, Youxi Xie, Yuexian Zou
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2609.26427

 - **Pdf link:** https://arxiv.org/pdf/2609.26427

 - **Abstract**
 Revision-capable streaming speech-to-text translation (S2TT) can correct earlier drafts, but process rewards based on visible text may credit content later withdrawn. Persistent Delivery Optimization (PDO) assigns intermediate reward only to content that survives revisions while scoring final quality separately. With 7.49 h of task-specific FLEURS adaptation, PDO achieves the best BLEU on four of five directions and higher COMET than every external streaming baseline in all five directions. Relative to its History-SFT initialization, PDO reduces mean/P90 finalization-aware latency by 10.8\%/11.3\% and normalized erasure by 15.8\%, while emitting at the first permitted 2-s update and improving macro BLEU. Zero-shot evaluation on Europarl-ST and CoVoST 2 confirms that these gains are not confined to the FLEURS training domain.
#### Not Quite My Tempo: Voice Activity-aware Speech Synthesis for Lip-Synchronous Dubbing
 - **Authors:** Alejandro Pérez-González-de-Martos, Florian Lux, Angelina Elizarova, Milana Shkhanukova, Andreas Kellner, Mattia Antonino Di Gangi
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Artificial Intelligence (cs.AI)
 - **Arxiv link:** https://arxiv.org/abs/2609.26486

 - **Pdf link:** https://arxiv.org/pdf/2609.26486

 - **Abstract**
 Automatic lip-synchronous dubbing requires a speech synthesis model to generate alternating voice and silence patterns in the target language that match the timing of the source clip precisely to ensure an optimal viewing experience. Prior works address this problem by conditioning the speech synthesis process on lip movements extracted from the video signal. In this work, we condition the speech generation on a binary voice-activity signal, which has a lightweight representation and can be produced in multiple ways. We show that the model follows the voice-activity signal with high accuracy while maintaining natural prosody and semantically appropriate pause placement within sentences, as demonstrated through extensive objective and subjective evaluations. By randomly masking this condition during training, we make the feature entirely optional during inference, allowing editors to enforce or relax lip-sync constraints when desired.
#### Boundary and Intra-Segment Learning for Partial Audio Deepfake Localization
 - **Authors:** Zhe Ye, Xiangui Kang, Minhua Huang, Kai Wu, Kong Aik Lee, Chng Eng Siong
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2609.25822

 - **Pdf link:** https://arxiv.org/pdf/2609.25822

 - **Abstract**
 Partial audio deepfakes manipulate only selected speech regions, making them difficult to be localized. Existing methods exploit boundary cues for partial deepfake localization, but primarily focus on identifying boundary positions rather than modeling the feature changes that characterize authenticity transitions. Meanwhile, the internal characteristics of continuous bona fide and spoofed segments remain underexplored. In this paper, we propose Boundary and Intra-Segment Learning (BISL), which introduces boundary learning to model feature differences between adjacent frames and distinguish authenticity transitions from general acoustic variations. In addition, intra-segment learning captures the overall characteristics of continuous bona fide and spoofed segments while enhancing feature consistency within each segment. By jointly learning frame, boundary, and segment information, BISL enables more effective fine-grained partial audio deepfake localization. Experiments on multiple localization benchmarks show that BISL achieves an EER of 2.52\% and an F1-score of 97.40\% on PartialSpoof, outperforming the compared methods, while maintaining competitive performance on HAD and improved cross-dataset performance on LPS. The code will be made publicly available upon acceptance.
#### Enriching Speech Emotion Representations with Conversational Context
 - **Authors:** Arthur Peuvot, Romaric Besançon, Gaël de Chalendar, Bianca Vieru, Ioana Vasilescu
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2609.26422

 - **Pdf link:** https://arxiv.org/pdf/2609.26422

 - **Abstract**
 Detecting emotions is necessary for building systems that can accurately and adaptively interact with humans. Speech Emotion Recognition (SER) has become an important research focus to develop intelligent spoken interfaces. However, most studies predict emotions at the utterance level, ignoring the conversational context, along with the emotional flow and speaker interactions it carries. In this paper, we introduce ACERT (Averaged Contextual Emotion Representation through Time), a module that integrates a flexible-length window of conversational context to better capture emotional evolution in spoken interactions. To evaluate the robustness of this method, we conducted experiments on datasets spanning diverse emotionally expressive styles and contexts. ACERT outperforms current state-of-the-art (SOTA) approaches on IEMOCAP, establishes the first context-aware benchmark on SAFE, and obtains strong results on MELD for unweighted, class-balanced metrics. Ablation studies show that ACERT's gains come from emotional and conversational continuity, rather than from speaker identity or acoustic conditions.
#### Spoken Language Models that Think Aloud
 - **Authors:** Junyi Ao, Kainan Peng, Mingbo Ma, Shun Zhang, Zhenyu Tang, Xutai Ma, Xiang Li, Yinghao Li, Yuancheng Wang, Zhizheng Wu, Haizhou Li, Qing He, Xubo Liu
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2609.26488

 - **Pdf link:** https://arxiv.org/pdf/2609.26488

 - **Abstract**
 While Chain-of-Thought (CoT) reasoning has improved the capability of language models, directly applying it to Spoken Language Models (SLMs) may introduce long silent intervals under the serial "think-then-speak" paradigm, disrupting real-time spoken interaction. To address this issue, we propose an asynchronous think-aloud framework for reasoning-based SLMs within the Thinker-Talker architecture. The framework maintains a primary reasoning stream for logical deduction and a lightweight think-aloud stream that generates short, task-grounded progress utterances conditioned on the user input and the evolving reasoning state. A dynamic balance strategy coordinates the two streams at runtime, triggering additional think-aloud speech to avoid silent gaps and canceling pending utterances when the final response becomes ready. Experiments on spoken reasoning and question-answering benchmarks show that our approach substantially reduces user-audible silence during reasoning while maintaining answer accuracy comparable to that of a serial "think-then-speak" baseline, demonstrating the potential of asynchronous think-aloud for responsive interaction in SLMs.
#### Transcribe, Translate, and Optimize: Joint Reward Learning for Speech Translation
 - **Authors:** Yanghe Dong, Wanting Huang, Weiran Wang
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2609.26536

 - **Pdf link:** https://arxiv.org/pdf/2609.26536

 - **Abstract**
 In LLM-based speech translation, transcription-based chain-of-thought (CoT) suffers from a mismatch between reference transcripts used in supervised fine-tuning (SFT) and model-generated transcripts at inference. To address this, we propose joint recognition and translation fine-tuning via group relative policy optimization (GRPO). We score both transcripts and translations, with translation conditioned on model-generated transcripts, and compare three token advantage strategies. Using Qwen2.5-Omni-3B across four languages, we evaluate CoT against direct speech translation (Direct ST) under SFT and GRPO, training on CoVoST 2 and testing on CoVoST 2 and FLEURS. CoT GRPO outperforms Direct ST GRPO by 1.77 and 0.83 average BLEU points on CoVoST 2 and FLEURS. Compared to CoT SFT, GRPO boosts BLEU by 0.82 and 0.67 points and reduces word error rate (WER) by 8.8% and 7.2% relatively. These results highlight reinforcement fine-tuning as an effective method to mitigate the training-inference mismatch, jointly improving recognition and translation.
#### ROAM-ASD: Robust Open-World Active Speaker Detection with Flexible Multimodal Fusion
 - **Authors:** Pu Wang, Yujun Wang, Hugo Van hamme
 - **Subjects:** Subjects:
Multimedia (cs.MM); Computer Vision and Pattern Recognition (cs.CV); Sound (cs.SD); Audio and Speech Processing (eess.AS); Image and Video Processing (eess.IV)
 - **Arxiv link:** https://arxiv.org/abs/2609.26648

 - **Pdf link:** https://arxiv.org/pdf/2609.26648

 - **Abstract**
 Active speaker detection (ASD) requires reliable association between visible faces and acoustic speech, yet existing systems often degrade under challenging domains or incomplete observations. We introduce ROAM-ASD, a robust audiovisual framework that jointly models audio, full-face, and fine-grained mouth representations. A unified joint self-attention mechanism processes all input streams together with modality-agnostic query tokens, enabling direct interaction among available modality inputs. Modality dropout further improves robustness when input streams are unavailable. ROAM-ASD achieves state-of-the-art performance across five ASD benchmarks: 98.8% mAP on WASD, 87.9% on UniTalk, 96.5% on AVA, 99.3% on ASW, and 98.2% on Talkies, improving over previous best systems by 5.1, 4.7, 0.9, 1.0, and 2.1 mAP points, respectively. ROAM-ASD also substantially improves zero-shot cross-dataset generalization and remains robust to missing observations.


by Zyzzyva0381 (Windy). 


2026-09-23
