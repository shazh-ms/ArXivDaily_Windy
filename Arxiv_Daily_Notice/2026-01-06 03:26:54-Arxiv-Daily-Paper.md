# Showing new listings for Tuesday, 6 January 2026
Auto update papers at about 2:30am UTC (10:30am Beijing time) every weekday.


阅读 `Usage.md`了解如何使用此repo实现个性化的Arxiv论文推送

See `Usage.md` for instructions on how to personalize the repo. 


Keyword list: ['text-to-speech', 'text to speech', 'tts', 'LLM-based', 'speech', 'voice']


Excluded: []


### Today: 12papers 
#### Speak the Art: A Direct Speech to Image Generation Framework
 - **Authors:** Mariam Saeed, Manar Amr, Farida Adel, Nada Hassan, Nour Walid, Eman Mohamed, Mohamed Hussein, Marwan Torki
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Artificial Intelligence (cs.AI); Multimedia (cs.MM)
 - **Arxiv link:** https://arxiv.org/abs/2601.00827

 - **Pdf link:** https://arxiv.org/pdf/2601.00827

 - **Abstract**
 Direct speech-to-image generation has recently shown promising results. However, compared to text-to-image generation, there is still a large gap to enclose. Current approaches use two stages to tackle this task: speech encoding network and image generative adversarial network (GAN). The speech encoding networks in these approaches produce embeddings that do not capture sufficient linguistic information to semantically represent the input speech. GANs suffer from issues such as non-convergence, mode collapse, and diminished gradient, which result in unstable model parameters, limited sample diversity, and ineffective generator learning, respectively. To address these weaknesses, we introduce a framework called \textbf{Speak the Art (STA)} which consists of a speech encoding network and a VQ-Diffusion network conditioned on speech embeddings. To improve speech embeddings, the speech encoding network is supervised by a large pre-trained image-text model during training. Replacing GANs with diffusion leads to more stable training and the generation of diverse images. Additionally, we investigate the feasibility of extending our framework to be multilingual. As a proof of concept, we trained our framework with two languages: English and Arabic. Finally, we show that our results surpass state-of-the-art models by a large margin.
#### Improving Code-Switching Speech Recognition with TTS Data Augmentation
 - **Authors:** Yue Heng Yeo, Yuchen Hu, Shreyas Gopal, Yizhou Peng, Hexin Liu, Eng Siong Chng
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Artificial Intelligence (cs.AI)
 - **Arxiv link:** https://arxiv.org/abs/2601.00935

 - **Pdf link:** https://arxiv.org/pdf/2601.00935

 - **Abstract**
 Automatic speech recognition (ASR) for conversational code-switching speech remains challenging due to the scarcity of realistic, high-quality labeled speech data. This paper explores multilingual text-to-speech (TTS) models as an effective data augmentation technique to address this shortage. Specifically, we fine-tune the multilingual CosyVoice2 TTS model on the SEAME dataset to generate synthetic conversational Chinese-English code-switching speech, significantly increasing the quantity and speaker diversity of available training data. Our experiments demonstrate that augmenting real speech with synthetic speech reduces the mixed error rate (MER) from 12.1 percent to 10.1 percent on DevMan and from 17.8 percent to 16.0 percent on DevSGE, indicating consistent performance gains. These results confirm that multilingual TTS is an effective and practical tool for enhancing ASR robustness in low-resource conversational code-switching scenarios.
#### MORE: Multi-Objective Adversarial Attacks on Speech Recognition
 - **Authors:** Xiaoxue Gao, Zexin Li, Yiming Chen, Nancy F. Chen
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Artificial Intelligence (cs.AI); Machine Learning (cs.LG)
 - **Arxiv link:** https://arxiv.org/abs/2601.01852

 - **Pdf link:** https://arxiv.org/pdf/2601.01852

 - **Abstract**
 The emergence of large-scale automatic speech recognition (ASR) models such as Whisper has greatly expanded their adoption across diverse real-world applications. Ensuring robustness against even minor input perturbations is therefore critical for maintaining reliable performance in real-time environments. While prior work has mainly examined accuracy degradation under adversarial attacks, robustness with respect to efficiency remains largely unexplored. This narrow focus provides only a partial understanding of ASR model vulnerabilities. To address this gap, we conduct a comprehensive study of ASR robustness under multiple attack scenarios. We introduce MORE, a multi-objective repetitive doubling encouragement attack, which jointly degrades recognition accuracy and inference efficiency through a hierarchical staged repulsion-anchoring mechanism. Specifically, we reformulate multi-objective adversarial optimization into a hierarchical framework that sequentially achieves the dual objectives. To further amplify effectiveness, we propose a novel repetitive encouragement doubling objective (REDO) that induces duplicative text generation by maintaining accuracy degradation and periodically doubling the predicted sequence length. Overall, MORE compels ASR models to produce incorrect transcriptions at a substantially higher computational cost, triggered by a single adversarial input. Experiments show that MORE consistently yields significantly longer transcriptions while maintaining high word error rates compared to existing baselines, underscoring its effectiveness in multi-objective adversarial attack.
#### Towards Prosodically Informed Mizo TTS without Explicit Tone Markings
 - **Authors:** Abhijit Mohanta, Remruatpuii, Priyankoo Sarmah, Rohit Sinha, Wendy Lalhminghlui
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2601.02073

 - **Pdf link:** https://arxiv.org/pdf/2601.02073

 - **Abstract**
 This paper reports on the development of a text-to-speech (TTS) system for Mizo, a low-resource, tonal, and Tibeto-Burman language spoken primarily in the Indian state of Mizoram. The TTS was built with only 5.18 hours of data; however, in terms of subjective and objective evaluations, the outputs were considered perceptually acceptable and intelligible. A baseline model using Tacotron2 was built, and then, with the same data, another TTS model was built with VITS. In both subjective and objective evaluations, the VITS model outperformed the Tacotron2 model. In terms of tone synthesis, the VITS model showed significantly lower tone errors than the Tacotron2 model. The paper demonstrates that a non-autoregressive, end-to-end framework can achieve synthesis of acceptable perceptual quality and intelligibility.
#### On the Role of Spatial Features in Foundation-Model-Based Speaker Diarization
 - **Authors:** Marc Deegen, Tobias Gburrek, Tobias Cord-Landwehr, Thilo von Neumann, Jiangyu Han, Lukáš Burget, Reinhold Haeb-Umbach
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2601.02231

 - **Pdf link:** https://arxiv.org/pdf/2601.02231

 - **Abstract**
 Recent advances in speaker diarization exploit large pretrained foundation models, such as WavLM, to achieve state-of-the-art performance on multiple datasets. Systems like DiariZen leverage these rich single-channel representations, but are limited to single-channel audio, preventing the use of spatial cues available in multi-channel recordings. This work analyzes the impact of incorporating spatial information into a state-of-the-art single-channel diarization system by evaluating several strategies for conditioning the model on multi-channel spatial features. Experiments on meeting-style datasets indicate that spatial information can improve diarization performance, but the overall improvement is smaller than expected for the proposed system, suggesting that the features aggregated over all WavLM layers already capture much of the information needed for accurate speaker discrimination, also in overlapping speech regions. These findings provide insight into the potential and limitations of using spatial cues to enhance foundation model-based diarization.
#### Index-ASR Technical Report
 - **Authors:** Zheshu Song, Lu Wang, Wei Deng, Zhuo Yang, Yong Wu, Bin Xia
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2601.00890

 - **Pdf link:** https://arxiv.org/pdf/2601.00890

 - **Abstract**
 Automatic speech recognition (ASR) has witnessed remarkable progress in recent years, largely driven by the emergence of LLM-based ASR paradigm. Despite their strong performance on a variety of open-source benchmarks, existing LLM-based ASR systems still suffer from two critical limitations. First, they are prone to hallucination errors, often generating excessively long and repetitive outputs that are not well grounded in the acoustic input. Second, they provide limited support for flexible and fine-grained contextual customization. To address these challenges, we propose Index-ASR, a large-scale LLM-based ASR system designed to simultaneously enhance robustness and support customizable hotword recognition. The core idea of Index-ASR lies in the integration of LLM and large-scale training data enriched with background noise and contextual information. Experimental results show that our Index-ASR achieves strong performance on both open-source benchmarks and in-house test sets, highlighting its robustness and practicality for real-world ASR applications.
#### IO-RAE: Information-Obfuscation Reversible Adversarial Example for Audio Privacy Protection
 - **Authors:** Jiajie Zhu, Xia Du, Xiaoyuan Liu, Jizhe Zhou, Qizhen Xu, Zheng Lin, Chi-Man Pun
 - **Subjects:** Subjects:
Sound (cs.SD); Cryptography and Security (cs.CR); Multimedia (cs.MM); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2601.01239

 - **Pdf link:** https://arxiv.org/pdf/2601.01239

 - **Abstract**
 The rapid advancements in artificial intelligence have significantly accelerated the adoption of speech recognition technology, leading to its widespread integration across various applications. However, this surge in usage also highlights a critical issue: audio data is highly vulnerable to unauthorized exposure and analysis, posing significant privacy risks for businesses and individuals. This paper introduces an Information-Obfuscation Reversible Adversarial Example (IO-RAE) framework, the pioneering method designed to safeguard audio privacy using reversible adversarial examples. IO-RAE leverages large language models to generate misleading yet contextually coherent content, effectively preventing unauthorized eavesdropping by humans and Automatic Speech Recognition (ASR) systems. Additionally, we propose the Cumulative Signal Attack technique, which mitigates high-frequency noise and enhances attack efficacy by targeting low-frequency signals. Our approach ensures the protection of audio data without degrading its quality or our ability. Experimental evaluations demonstrate the superiority of our method, achieving a targeted misguidance rate of 96.5% and a remarkable 100% untargeted misguidance rate in obfuscating target keywords across multiple ASR models, including a commercial black-box system from Google. Furthermore, the quality of the recovered audio, measured by the Perceptual Evaluation of Speech Quality score, reached 4.45, comparable to high-quality original recordings. Notably, the recovered audio processed by ASR systems exhibited an error rate of 0%, indicating nearly lossless recovery. These results highlight the practical applicability and effectiveness of our IO-RAE framework in protecting sensitive audio privacy.
#### UltraEval-Audio: A Unified Framework for Comprehensive Evaluation of Audio Foundation Models
 - **Authors:** Qundong Shi, Jie Zhou, Biyuan Lin, Junbo Cui, Guoyang Zeng, Yixuan Zhou, Ziyang Wang, Xin Liu, Zhen Luo, Yudong Wang, Zhiyuan Liu
 - **Subjects:** Subjects:
Sound (cs.SD); Artificial Intelligence (cs.AI); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2601.01373

 - **Pdf link:** https://arxiv.org/pdf/2601.01373

 - **Abstract**
 The development of audio foundation models has accelerated rapidly since the emergence of GPT-4o. However, the lack of comprehensive evaluation has become a critical bottleneck for further progress in the field, particularly in audio generation. Current audio evaluation faces three major challenges: (1) audio evaluation lacks a unified framework, with datasets and code scattered across various sources, hindering fair and efficient cross-model comparison;(2) audio codecs, as a key component of audio foundation models, lack a widely accepted and holistic evaluation methodology; (3) existing speech benchmarks are heavily reliant on English, making it challenging to objectively assess models' performance on Chinese. To address the first issue, we introduce UltraEval-Audio, a unified evaluation framework for audio foundation models, specifically designed for both audio understanding and generation tasks. UltraEval-Audio features a modular architecture, supporting 10 languages and 14 core task categories, while seamlessly integrating 24 mainstream models and 36 authoritative benchmarks. To enhance research efficiency, the framework provides a one-command evaluation feature, accompanied by real-time public leaderboards. For the second challenge, UltraEval-Audio adopts a novel comprehensive evaluation scheme for audio codecs, evaluating performance across three key dimensions: semantic accuracy, timbre fidelity, and acoustic quality. To address the third issue, we propose two new Chinese benchmarks, SpeechCMMLU and SpeechHSK, designed to assess Chinese knowledge proficiency and language fluency. We wish that UltraEval-Audio will provide both academia and industry with a transparent, efficient, and fair platform for comparison of audio models. Our code, benchmarks, and leaderboards are available at this https URL.
#### OV-InstructTTS: Towards Open-Vocabulary Instruct Text-to-Speech
 - **Authors:** Yong Ren, Jiangyan Yi, Jianhua Tao, Haiyang Sun, Zhengqi Wen, Hao Gu, Le Xu, Ye Bai
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2601.01459

 - **Pdf link:** https://arxiv.org/pdf/2601.01459

 - **Abstract**
 Instruct Text-to-Speech (InstructTTS) leverages natural language descriptions as style prompts to guide speech synthesis. However, existing InstructTTS methods mainly rely on a direct combination of audio-related labels or their diverse rephrasings, making it difficult to handle flexible, high-level instructions. Such rigid control is insufficient for users such as content creators who wish to steer generation with descriptive instructions. To address these constraints, we introduce OV-InstructTTS, a new paradigm for open-vocabulary InstructTTS. We propose a comprehensive solution comprising a newly curated dataset, OV-Speech, and a novel reasoning-driven framework. The OV-Speech dataset pairs speech with open-vocabulary instructions, each augmented with a reasoning process that connects high-level instructions to acoustic features. The reasoning-driven framework infers emotional, acoustic, and paralinguistic information from open-vocabulary instructions before synthesizing speech. Evaluations show that this reasoning-driven approach significantly improves instruction-following fidelity and speech expressiveness. We believe this work can inspire the next user-friendly InstructTTS systems with stronger generalization and real-world applicability. The dataset and demos are publicly available on our project page.
#### Bridging the gap: A comparative exploration of Speech-LLM and end-to-end architecture for multilingual conversational ASR
 - **Authors:** Yuxiang Mei, Dongxing Xu, Jiaen Liang, Yanhua Long
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2601.01461

 - **Pdf link:** https://arxiv.org/pdf/2601.01461

 - **Abstract**
 The INTERSPEECH 2025 Challenge on Multilingual Conversational Speech Language Models (MLC-SLM) promotes multilingual conversational ASR with large language models (LLMs). Our previous SHNU-mASR system adopted a competitive parallel-speech-encoder architecture that integrated Whisper and mHuBERT with an LLM. However, it faced two challenges: simple feature concatenation may not fully exploit complementary information, and the performance gap between LLM-based ASR and end-to-end(E2E) encoder-decoder ASR remained unexplored. In this work, we present an enhanced LLM-based ASR framework that combines fine-tuned Whisper and mHuBERT encoders with an LLM to enrich speech representations. We first evaluate E2E Whisper models with LoRA and full fine-tuning on the MLC-SLM ASR task, and then propose cross-attention-based fusion mechanisms for the parallel-speech-encoder. On the official evaluation set of the MLC-SLM Challenge, our system achieves a CER/WER of 10.69%, ranking on par with the top-ranked Track 1 systems, even though it uses only 1,500 hours of baseline training data compared with their large-scale training sets. Nonetheless, we find that our final LLM-based ASR still does not match the performance of a fine-tuned E2E Whisper model, providing valuable empirical guidance for future Speech-LLM design. Our code is publicly available at this https URL.
#### MM-Sonate: Multimodal Controllable Audio-Video Generation with Zero-Shot Voice Cloning
 - **Authors:** Chunyu Qiang, Jun Wang, Xiaopeng Wang, Kang Yin, Yuxin Guo, Xijuan Zeng, Nan Li, Zihan Li, Yuzhe Liang, Ziyu Zhang, Teng Ma, Yushen Chen, Zhongliang Liu, Feng Deng, Chen Zhang, Pengfei Wan
 - **Subjects:** Subjects:
Sound (cs.SD); Artificial Intelligence (cs.AI); Computer Vision and Pattern Recognition (cs.CV); Multimedia (cs.MM); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2601.01568

 - **Pdf link:** https://arxiv.org/pdf/2601.01568

 - **Abstract**
 Joint audio-video generation aims to synthesize synchronized multisensory content, yet current unified models struggle with fine-grained acoustic control, particularly for identity-preserving speech. Existing approaches either suffer from temporal misalignment due to cascaded generation or lack the capability to perform zero-shot voice cloning within a joint synthesis framework. In this work, we present MM-Sonate, a multimodal flow-matching framework that unifies controllable audio-video joint generation with zero-shot voice cloning capabilities. Unlike prior works that rely on coarse semantic descriptions, MM-Sonate utilizes a unified instruction-phoneme input to enforce strict linguistic and temporal alignment. To enable zero-shot voice cloning, we introduce a timbre injection mechanism that effectively decouples speaker identity from linguistic content. Furthermore, addressing the limitations of standard classifier-free guidance in multimodal settings, we propose a noise-based negative conditioning strategy that utilizes natural noise priors to significantly enhance acoustic fidelity. Empirical evaluations demonstrate that MM-Sonate establishes new state-of-the-art performance in joint generation benchmarks, significantly outperforming baselines in lip synchronization and speech intelligibility, while achieving voice cloning fidelity comparable to specialized Text-to-Speech systems.
#### Towards Multi-Level Transcript Segmentation: LoRA Fine-Tuning for Table-of-Contents Generation
 - **Authors:** Steffen Freisinger, Philipp Seeberger, Thomas Ranzenberger, Tobias Bocklet, Korbinian Riedhammer
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2601.02128

 - **Pdf link:** https://arxiv.org/pdf/2601.02128

 - **Abstract**
 Segmenting speech transcripts into thematic sections benefits both downstream processing and users who depend on written text for accessibility. We introduce a novel approach to hierarchical topic segmentation in transcripts, generating multi-level tables of contents that capture both topic and subtopic boundaries. We compare zero-shot prompting and LoRA fine-tuning on large language models, while also exploring the integration of high-level speech pause features. Evaluations on English meeting recordings and multilingual lecture transcripts (Portuguese, German) show significant improvements over established topic segmentation baselines. Additionally, we adapt a common evaluation measure for multi-level segmentation, taking into account all hierarchical levels within one metric.


by Zyzzyva0381 (Windy). 


2026-01-06
