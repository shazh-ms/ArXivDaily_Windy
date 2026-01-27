# Showing new listings for Tuesday, 27 January 2026
Auto update papers at about 2:30am UTC (10:30am Beijing time) every weekday.


阅读 `Usage.md`了解如何使用此repo实现个性化的Arxiv论文推送

See `Usage.md` for instructions on how to personalize the repo. 


Keyword list: ['text-to-speech', 'text to speech', 'tts', 'LLM-based', 'speech', 'voice']


Excluded: []


### Today: 22papers 
#### The Voice of Equity: A Systematic Evaluation of Bias Mitigation Techniques for Speech-Based Cognitive Impairment Detection Across Architectures and Demographics
 - **Authors:** Yasaman Haghbin, Sina Rashidi, Ali Zolnour, Maryam Zolnoori
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Computation and Language (cs.CL); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2601.16989

 - **Pdf link:** https://arxiv.org/pdf/2601.16989

 - **Abstract**
 Speech-based detection of cognitive impairment offers a scalable, non-invasive screening, yet algorithmic bias across demographic and linguistic subgroups remains critically underexplored. We present the first comprehensive fairness analysis framework for speech-based multi-class cognitive impairment detection, systematically evaluating bias mitigation across architectures, and demographic subgroups. We developed two transformer-based architectures, SpeechCARE-AGF and Whisper-LWF-LoRA, on the multilingual NIA PREPARE Challenge dataset. Unlike prior work that typically examines single mitigation techniques, we compared pre-processing, in-processing, and post-processing approaches, assessing fairness via Equality of Opportunity and Equalized Odds across gender, age, education, and language. Both models achieved strong performance (F1: SpeechCARE-AGF 70.87, Whisper-LWF-LoRA 71.46) but exhibited substantial fairness disparities. Adults >=80 showed lower sensitivity versus younger groups; Spanish speakers demonstrated reduced TPR versus English speakers. Mitigation effectiveness varied by architecture: oversampling improved SpeechCARE-AGF for older adults (80+ TPR: 46.19%=>49.97%) but minimally affected Whisper-LWF-LoRA. This study addresses a critical healthcare AI gap by demonstrating that architectural design fundamentally shapes bias patterns and mitigation effectiveness. Adaptive fusion mechanisms enable flexible responses to data interventions, while frequency reweighting offers robust improvements across architectures. Our findings establish that fairness interventions must be tailored to both model architecture and demographic characteristics, providing a systematic framework for developing equitable speech-based screening tools essential for reducing diagnostic disparities in cognitive healthcare.
#### Recovering Performance in Speech Emotion Recognition from Discrete Tokens via Multi-Layer Fusion and Paralinguistic Feature Integration
 - **Authors:** Esther Sun, Abinay Reddy Naini, Carlos Busso
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Machine Learning (cs.LG); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2601.17085

 - **Pdf link:** https://arxiv.org/pdf/2601.17085

 - **Abstract**
 Discrete speech tokens offer significant advantages for storage and language model integration, but their application in speech emotion recognition (SER) is limited by paralinguistic information loss during quantization. This paper presents a comprehensive investigation of discrete tokens for SER. Using a fine-tuned WavLM-Large model, we systematically quantify performance degradation across different layer configurations and k-means quantization granularities. To recover the information loss, we propose two key strategies: (1) attention-based multi-layer fusion to recapture complementary information from different layers, and (2) integration of openSMILE features to explicitly reintroduce paralinguistic cues. We also compare mainstream neural codec tokenizers (SpeechTokenizer, DAC, EnCodec) and analyze their behaviors when fused with acoustic features. Our findings demonstrate that through multi-layer fusion and acoustic feature integration, discrete tokens can close the performance gap with continuous representations in SER tasks.
#### End-to-End Joint ASR and Speaker Role Diarization with Child-Adult Interactions
 - **Authors:** Anfeng Xu, Tiantian Feng, Somer Bishop, Catherine Lord, Shrikanth Narayanan
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2601.17640

 - **Pdf link:** https://arxiv.org/pdf/2601.17640

 - **Abstract**
 Accurate transcription and speaker diarization of child-adult spoken interactions are crucial for developmental and clinical research. However, manual annotation is time-consuming and challenging to scale. Existing automated systems typically rely on cascaded speaker diarization and speech recognition pipelines, which can lead to error propagation. This paper presents a unified end-to-end framework that extends the Whisper encoder-decoder architecture to jointly model ASR and child-adult speaker role diarization. The proposed approach integrates: (i) a serialized output training scheme that emits speaker tags and start/end timestamps, (ii) a lightweight frame-level diarization head that enhances speaker-discriminative encoder representations, (iii) diarization-guided silence suppression for improved temporal precision, and (iv) a state-machine-based forced decoding procedure that guarantees structurally valid outputs. Comprehensive evaluations on two datasets demonstrate consistent and substantial improvements over two cascaded baselines, achieving lower multi-talker word error rates and demonstrating competitive diarization accuracy across both Whisper-small and Whisper-large models. These findings highlight the effectiveness and practical utility of the proposed joint modeling framework for generating reliable, speaker-attributed transcripts of child-adult interactions at scale. The code and model weights are publicly available
#### Speech Emotion Recognition with ASR Integration
 - **Authors:** Yuanchao Li
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2601.17901

 - **Pdf link:** https://arxiv.org/pdf/2601.17901

 - **Abstract**
 Speech Emotion Recognition (SER) plays a pivotal role in understanding human communication, enabling emotionally intelligent systems, and serving as a fundamental component in the development of Artificial General Intelligence (AGI). However, deploying SER in real-world, spontaneous, and low-resource scenarios remains a significant challenge due to the complexity of emotional expression and the limitations of current speech and language technologies. This thesis investigates the integration of Automatic Speech Recognition (ASR) into SER, with the goal of enhancing the robustness, scalability, and practical applicability of emotion recognition from spoken language.
#### AmbER$^2$: Dual Ambiguity-Aware Emotion Recognition Applied to Speech and Text
 - **Authors:** Jingyao Wu, Grace Lin, Yinuo Song, Rosalind Picard
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2601.18010

 - **Pdf link:** https://arxiv.org/pdf/2601.18010

 - **Abstract**
 Emotion recognition is inherently ambiguous, with uncertainty arising both from rater disagreement and from discrepancies across modalities such as speech and text. There is growing interest in modeling rater ambiguity using label distributions. However, modality ambiguity remains underexplored, and multimodal approaches often rely on simple feature fusion without explicitly addressing conflicts between modalities. In this work, we propose AmbER$^2$, a dual ambiguity-aware framework that simultaneously models rater-level and modality-level ambiguity through a teacher-student architecture with a distribution-wise training objective. Evaluations on IEMOCAP and MSP-Podcast show that AmbER$^2$ consistently improves distributional fidelity over conventional cross-entropy baselines and achieves performance competitive with, or superior to, recent state-of-the-art systems. For example, on IEMOCAP, AmbER$^2$ achieves relative improvements of 20.3% on Bhattacharyya coefficient (0.83 vs. 0.69), 13.6% on R$^2$ (0.67 vs. 0.59), 3.8% on accuracy (0.683 vs. 0.658), and 4.5% on F1 (0.675 vs. 0.646). Further analysis across ambiguity levels shows that explicitly modeling ambiguity is particularly beneficial for highly uncertain samples. These findings highlight the importance of jointly addressing rater and modality ambiguity when building robust emotion recognition systems.
#### SpatialEmb: Extract and Encode Spatial Information for 1-Stage Multi-channel Multi-speaker ASR on Arbitrary Microphone Arrays
 - **Authors:** Yiwen Shao, Yong Xu, Sanjeev Khudanpur, Dong Yu
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Artificial Intelligence (cs.AI); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2601.18037

 - **Pdf link:** https://arxiv.org/pdf/2601.18037

 - **Abstract**
 Spatial information is a critical clue for multi-channel multi-speaker target speech recognition. Most state-of-the-art multi-channel Automatic Speech Recognition (ASR) systems extract spatial features only during the speech separation stage, followed by standard single-channel ASR on the separated speech. This approach results in an inefficient, lengthy pipeline and sub-optimal ASR performance due to the accumulated errors from preprocessing modules. Furthermore, most spatial feature extraction methods depend on the knowledge of speaker positions and microphone topology, making the systems reliant on specific settings and challenging to adapt to new equipment. In this work, we propose a solution to these issues with a lightweight embedding module named SpatialEmb, which extracts and encodes spatial information directly for the ASR model, supporting both fixed and arbitrary microphone topology. We conduct comprehensive experiments on AliMeeting, a real meeting corpus, to determine the optimal model design for SpatialEmb in terms of both performance and efficiency. Our best model trained with 105 hours Train-Ali-far achieves 17.04% and 20.32% character error rates (CER) on the Eval and Test sets, establishing a new state-of-the-art result with the same training data.
#### OneVoice: One Model, Triple Scenarios-Towards Unified Zero-shot Voice Conversion
 - **Authors:** Zhichao Wang, Tao Li, Wenshuo Ge, Zihao Cui, Shilei Zhang, Junlan Feng
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2601.18094

 - **Pdf link:** https://arxiv.org/pdf/2601.18094

 - **Abstract**
 Recent progress of voice conversion~(VC) has achieved a new milestone in speaker cloning and linguistic preservation. But the field remains fragmented, relying on specialized models for linguistic-preserving, expressive, and singing scenarios. We propose OneVoice, a unified zero-shot framework capable of handling all three scenarios within a single model. OneVoice is built upon a continuous language model trained with VAE-free next-patch diffusion, ensuring high fidelity and efficient sequence modeling. Its core design for unification lies in a Mixture-of-Experts (MoE) designed to explicitly model shared conversion knowledge and scenario-specific expressivity. Expert selection is coordinated by a dual-path routing mechanism, including shared expert isolation and scenario-aware domain expert assignment with global-local cues. For precise conditioning, scenario-specific prosodic features are fused into each layer via a gated mechanism, allowing adaptive usage of prosody information. Furthermore, to enable the core idea and alleviate the imbalanced issue (abundant speech vs. scarce singing), we adopt a two-stage progressive training that includes foundational pre-training and scenario enhancement with LoRA-based domain experts. Experiments show that OneVoice matches or surpasses specialized models across all three scenarios, while verifying flexible control over scenarios and offering a fast decoding version as few as 2 steps. Code and model will be released soon.
#### Efficient Rehearsal for Continual Learning in ASR via Singular Value Tuning
 - **Authors:** Steven Vander Eeckt, Hugo Van hamme
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2601.18266

 - **Pdf link:** https://arxiv.org/pdf/2601.18266

 - **Abstract**
 Continual Learning (CL) in Automatic Speech Recognition (ASR) suffers from catastrophic forgetting when adapting to new tasks, domains, or speakers. A common strategy to mitigate this is to store a subset of past data in memory for rehearsal. However, rehearsal-based methods face key limitations: storing data is often costly, infeasible with pre-trained models, or restricted by privacy regulations. Running existing rehearsal-based methods with smaller memory sizes to alleviate these issues usually leads to degraded performance. We propose a rehearsal-based CL method that remains effective even with minimal memory. It operates in two stages: first, fine-tuning on the new task; second, applying Singular Value Decomposition (SVD) to the changes in linear layers and, in a parameter-efficient manner, retraining only gating vectors on the singular values, which control to extent to which updates from the first stage are accepted, using rehearsal. We extensively test and analyze our method on two monolingual and two multilingual benchmarks. Our method reduces forgetting and outperforms state-of-the-art CL approaches for ASR, even when limited to a single utterance per previous task.
#### Noise-Robust AV-ASR Using Visual Features Both in the Whisper Encoder and Decoder
 - **Authors:** Zhengyang Li, Thomas Graave, Björn Möller, Zehang Wu, Matthias Franz, Tim Fingscheidt
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Computation and Language (cs.CL); Computer Vision and Pattern Recognition (cs.CV); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2601.18396

 - **Pdf link:** https://arxiv.org/pdf/2601.18396

 - **Abstract**
 In audiovisual automatic speech recognition (AV-ASR) systems, information fusion of visual features in a pre-trained ASR has been proven as a promising method to improve noise robustness. In this work, based on the prominent Whisper ASR, first, we propose a simple and effective visual fusion method -- use of visual features both in encoder and decoder (dual-use) -- to learn the audiovisual interactions in the encoder and to weigh modalities in the decoder. Second, we compare visual fusion methods in Whisper models of various sizes. Our proposed dual-use method shows consistent noise robustness improvement, e.g., a 35% relative improvement (WER: 4.41% vs. 6.83%) based on Whisper small, and a 57% relative improvement (WER: 4.07% vs. 9.53%) based on Whisper medium, compared to typical reference middle fusion in babble noise with a signal-to-noise ratio (SNR) of 0dB. Third, we conduct ablation studies examining the impact of various module designs and fusion options. Fine-tuned on 1929 hours of audiovisual data, our dual-use method using Whisper medium achieves 4.08% (MUSAN babble noise) and 4.43% (NoiseX babble noise) average WER across various SNRs, thereby establishing a new state-of-the-art in noisy conditions on the LRS3 AV-ASR benchmark. Our code is at this https URL
#### SonoEdit: Null-Space Constrained Knowledge Editing for Pronunciation Correction in LLM-Based TTS
 - **Authors:** Ayush Pratap Singh, Harshit Singh, Nityanand Mathur, Akshat Mandloi, Sudarshan Kamath
 - **Subjects:** Subjects:
Sound (cs.SD); Artificial Intelligence (cs.AI); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2601.17086

 - **Pdf link:** https://arxiv.org/pdf/2601.17086

 - **Abstract**
 Neural text-to-speech (TTS) systems systematically mispronounce low-resource proper nouns, particularly non-English names, brands, and geographic locations, due to their underrepresentation in predominantly English training corpora. Existing solutions typically rely on expensive multilingual data collection, supervised finetuning, or manual phonetic annotation, which limits the deployment of TTS systems in linguistically diverse settings. We introduce SonoEdit, a model editing technique that surgically corrects pronunciation errors in pre-trained TTS models without retraining. Instead of costly finetuning or explicit phoneme injection, we propose a parsimonious alternative based on Null-Space Pronunciation Editing, which performs a single-shot parameter update to modify the pronunciation of specific words while provably preserving all other model behavior. We first adapt Acoustic Causal Tracing to identify the Transformer layers responsible for text-to-pronunciation mapping. We then apply Null-Space Constrained Editing to compute a closed-form weight update that corrects the target pronunciation while remaining mathematically orthogonal to the subspace governing general speech generation. This constrained update steers the model's acoustic output toward a desired pronunciation exemplar while guaranteeing zero first-order change on a preserved speech corpus.
#### Sink or SWIM: Tackling Real-Time ASR at Scale
 - **Authors:** Federico Bruzzone, Walter Cazzola, Matteo Brancaleoni, Dario Pellegrino
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2601.17097

 - **Pdf link:** https://arxiv.org/pdf/2601.17097

 - **Abstract**
 Real-time automatic speech recognition systems are increasingly integrated into interactive applications, from voice assistants to live transcription services. However, scaling these systems to support multiple concurrent clients while maintaining low latency and high accuracy remains a major challenge. In this work, we present SWIM, a novel real-time ASR system built on top of OpenAI's Whisper model that enables true model-level parallelization for scalable, multilingual transcription. SWIM supports multiple concurrent audio streams without modifying the underlying model. It introduces a buffer merging strategy that maintains transcription fidelity while ensuring efficient resource usage. We evaluate SWIM in multi-client settings -- scaling up to 20 concurrent users -- and show that it delivers accurate real-time transcriptions in English, Italian, and Spanish, while maintaining low latency and high throughput. While Whisper-Streaming achieves a word error rate of approximately 8.2% with an average delay of approximately 3.4 s in a single-client, English-only setting, SWIM extends this capability to multilingual, multi-client environments. It maintains comparable accuracy with significantly lower delay -- around 2.4 s with 5 clients -- and continues to scale effectively up to 20 concurrent clients without degrading transcription quality and increasing overall throughput. Our approach advances scalable ASR by improving robustness and efficiency in dynamic, multi-user environments.
#### Window Size Versus Accuracy Experiments in Voice Activity Detectors
 - **Authors:** Max McKinnon, Samir Khaki, Chandan KA Reddy, William Huang
 - **Subjects:** Subjects:
Sound (cs.SD); Computation and Language (cs.CL); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2601.17270

 - **Pdf link:** https://arxiv.org/pdf/2601.17270

 - **Abstract**
 Voice activity detection (VAD) plays a vital role in enabling applications such as speech recognition. We analyze the impact of window size on the accuracy of three VAD algorithms: Silero, WebRTC, and Root Mean Square (RMS) across a set of diverse real-world digital audio streams. We additionally explore the use of hysteresis on top of each VAD output. Our results offer practical references for optimizing VAD systems. Silero significantly outperforms WebRTC and RMS, and hysteresis provides a benefit for WebRTC.
#### AVMeme Exam: A Multimodal Multilingual Multicultural Benchmark for LLMs' Contextual and Cultural Knowledge and Thinking
 - **Authors:** Xilin Jiang, Qiaolin Wang, Junkai Wu, Xiaomin He, Zhongweiyang Xu, Yinghao Ma, Minshuo Piao, Kaiyi Yang, Xiuwen Zheng, Riki Shimizu, Yicong Chen, Arsalan Firoozi, Gavin Mischler, Sukru Samet Dindar, Richard Antonello, Linyang He, Tsun-An Hsieh, Xulin Fan, Yulun Wu, Yuesheng Ma, Chaitanya Amballa, Weixiong Chen, Jiarui Hai, Ruisi Li, Vishal Choudhari, Cong Han, Yinghao Aaron Li, Adeen Flinker, Mounya Elhilali, Emmanouil Benetos, Mark Hasegawa-Johnson, Romit Roy Choudhury, Nima Mesgarani
 - **Subjects:** Subjects:
Sound (cs.SD); Computation and Language (cs.CL); Computer Vision and Pattern Recognition (cs.CV); Multimedia (cs.MM); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2601.17645

 - **Pdf link:** https://arxiv.org/pdf/2601.17645

 - **Abstract**
 Internet audio-visual clips convey meaning through time-varying sound and motion, which extend beyond what text alone can represent. To examine whether AI models can understand such signals in human cultural contexts, we introduce AVMeme Exam, a human-curated benchmark of over one thousand iconic Internet sounds and videos spanning speech, songs, music, and sound effects. Each meme is paired with a unique Q&A assessing levels of understanding from surface content to context and emotion to usage and world knowledge, along with metadata such as original year, transcript, summary, and sensitivity. We systematically evaluate state-of-the-art multimodal large language models (MLLMs) alongside human participants using this benchmark. Our results reveal a consistent limitation: current models perform poorly on textless music and sound effects, and struggle to think in context and in culture compared to surface content. These findings highlight a key gap in human-aligned multimodal intelligence and call for models that can perceive contextually and culturally beyond the surface of what they hear and see. Project page: this http URL
#### BanglaRobustNet: A Hybrid Denoising-Attention Architecture for Robust Bangla Speech Recognition
 - **Authors:** Md Sazzadul Islam Ridoy, Mubaswira Ibnat Zidney, Sumi Akter, Md. Aminur Rahman
 - **Subjects:** Subjects:
Sound (cs.SD); Computation and Language (cs.CL); Computer Vision and Pattern Recognition (cs.CV); Machine Learning (cs.LG); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2601.17679

 - **Pdf link:** https://arxiv.org/pdf/2601.17679

 - **Abstract**
 Bangla, one of the most widely spoken languages, remains underrepresented in state-of-the-art automatic speech recognition (ASR) research, particularly under noisy and speaker-diverse conditions. This paper presents BanglaRobustNet, a hybrid denoising-attention framework built on Wav2Vec-BERT, designed to address these challenges. The architecture integrates a diffusion-based denoising module to suppress environmental noise while preserving Bangla-specific phonetic cues, and a contextual cross-attention module that conditions recognition on speaker embeddings for robustness across gender, age, and dialects. Trained end-to-end with a composite objective combining CTC loss, phonetic consistency, and speaker alignment, BanglaRobustNet achieves substantial reductions in word error rate (WER) and character error rate (CER) compared to Wav2Vec-BERT and Whisper baselines. Evaluations on Mozilla Common Voice Bangla and augmented noisy speech confirm the effectiveness of our approach, establishing BanglaRobustNet as a robust ASR system tailored to low-resource, noise-prone linguistic settings.
#### CaSNet: Compress-and-Send Network Based Multi-Device Speech Enhancement Model for Distributed Microphone Arrays
 - **Authors:** Chengqian Jiang, Jie Zhang, Haoyin Yan
 - **Subjects:** Subjects:
Sound (cs.SD); Artificial Intelligence (cs.AI); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2601.17711

 - **Pdf link:** https://arxiv.org/pdf/2601.17711

 - **Abstract**
 Distributed microphone array (DMA) is a promising next-generation platform for speech interaction, where speech enhancement (SE) is still required to improve the speech quality in noisy cases. Existing SE methods usually first gather raw waveforms at a fusion center (FC) from all devices and then design a multi-microphone model, causing high bandwidth and energy costs. In this work, we propose a \emph{Compress-and-Send Network (CaSNet)} for resource-constrained DMAs, where one microphone serves as the FC and reference. Each of other devices encodes the measured raw data into a feature matrix, which is then compressed by singular value decomposition (SVD) to produce a more compact representation. The received features at the FC are aligned via cross window query with respect to the reference, followed by neural decoding to yield spatially coherent enhanced speech. Experiments on multiple datasets show that the proposed CaSNet can save the data amount with a negligible impact on the performance compared to the uncompressed case. The reproducible code is available at this https URL.
#### dLLM-ASR: A Faster Diffusion LLM-based Framework for Speech Recognition
 - **Authors:** Wenjie Tian, Bingshen Mu, Guobin Ma, Xuelong Geng, Zhixian Zhao, Lei Xie
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2601.17902

 - **Pdf link:** https://arxiv.org/pdf/2601.17902

 - **Abstract**
 Automatic speech recognition (ASR) systems based on large language models (LLMs) achieve superior performance by leveraging pretrained LLMs as decoders, but their token-by-token generation mechanism leads to inference latency that grows linearly with sequence length. Meanwhile, discrete diffusion large language models (dLLMs) offer a promising alternative, enabling high-quality parallel sequence generation with pretrained decoders. However, directly applying native text-oriented dLLMs to ASR leads to a fundamental mismatch between open-ended text generation and the acoustically conditioned transcription paradigm required by ASR. As a result, it introduces unnecessary difficulty and computational redundancy, such as denoising from pure noise, inflexible generation lengths, and fixed denoising steps. We propose dLLM-ASR, an efficient dLLM-based ASR framework that formulates dLLM's decoding as a prior-guided and adaptive denoising process. It leverages an ASR prior to initialize the denoising process and provide an anchor for sequence length. Building upon this prior, length-adaptive pruning dynamically removes redundant tokens, while confidence-based denoising allows converged tokens to exit the denoising loop early, enabling token-level adaptive computation. Experiments demonstrate that dLLM-ASR achieves recognition accuracy comparable to autoregressive LLM-based ASR systems and delivers a 4.44$\times$ inference speedup, establishing a practical and efficient paradigm for ASR.
#### From Human Speech to Ocean Signals: Transferring Speech Large Models for Underwater Acoustic Target Recognition
 - **Authors:** Mengcheng Huang, Xue Zhou, Chen Xu, Dapeng Man
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2601.18086

 - **Pdf link:** https://arxiv.org/pdf/2601.18086

 - **Abstract**
 Underwater acoustic target recognition (UATR) plays a vital role in marine applications but remains challenging due to limited labeled data and the complexity of ocean environments. This paper explores a central question: can speech large models (SLMs), trained on massive human speech corpora, be effectively transferred to underwater acoustics? To investigate this, we propose UATR-SLM, a simple framework that reuses the speech feature pipeline, adapts the SLM as an acoustic encoder, and adds a lightweight this http URL on the DeepShip and ShipsEar benchmarks show that UATR-SLM achieves over 99% in-domain accuracy, maintains strong robustness across variable signal lengths, and reaches up to 96.67% accuracy in cross-domain evaluation. These results highlight the strong transferability of SLMs to UATR, establishing a promising paradigm for leveraging speech foundation models in underwater acoustics.
#### VIBEVOICE-ASR Technical Report
 - **Authors:** Zhiliang Peng, Jianwei Yu, Yaoyao Chang, Zilong Wang, Li Dong, Yingbo Hao, Yujie Tu, Chenyu Yang, Wenhui Wang, Songchen Xu, Yutao Sun, Hangbo Bao, Weijiang Xu, Yi Zhu, Zehua Wang, Ting Song, Yan Xia, Zewen Chi, Shaohan Huang, Liang Wang, Chuang Ding, Shuai Wang, Xie Chen, Furu Wei
 - **Subjects:** Subjects:
Sound (cs.SD); Artificial Intelligence (cs.AI); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2601.18184

 - **Pdf link:** https://arxiv.org/pdf/2601.18184

 - **Abstract**
 This report presents VibeVoice-ASR, a general-purpose speech understanding framework built upon VibeVoice, designed to address the persistent challenges of context fragmentation and multi-speaker complexity in long-form audio (e.g., meetings, podcasts) that remain despite recent advancements in short-form speech recognition. Unlike traditional pipelined approaches that rely on audio chunking, VibeVoice-ASRsupports single-pass processing for up to 60 minutes of audio. It unifies Automatic Speech Recognition, Speaker Diarization, and Timestamping into a single end-to-end generation task. In addition, VibeVoice-ASR supports over 50 languages, requires no explicit language setting, and natively handles code-switching within and across utterances. Furthermore, we introduce a prompt-based context injection mechanism that allows users to supply customized conetxt, significantly improving accuracy on domain-specific terminology and polyphonic character disambiguation.
#### LLM-ForcedAligner: A Non-Autoregressive and Accurate LLM-Based Forced Aligner for Multilingual and Long-Form Speech
 - **Authors:** Bingshen Mu, Xian Shi, Xiong Wang, Hexin Liu, Jin Xu, Lei Xie
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2601.18220

 - **Pdf link:** https://arxiv.org/pdf/2601.18220

 - **Abstract**
 Forced alignment (FA) predicts start and end timestamps for words or characters in speech, but existing methods are language-specific and prone to cumulative temporal shifts. The multilingual speech understanding and long-sequence processing abilities of speech large language models (SLLMs) make them promising for FA in multilingual, crosslingual, and long-form speech settings. However, directly applying the next-token prediction paradigm of SLLMs to FA results in hallucinations and slow inference. To bridge the gap, we propose LLM-ForcedAligner, reformulating FA as a slot-filling paradigm: timestamps are treated as discrete indices, and special timestamp tokens are inserted as slots into the transcript. Conditioned on the speech embeddings and the transcript with slots, the SLLM directly predicts the time indices at slots. During training, causal attention masking with non-shifted input and label sequences allows each slot to predict its own timestamp index based on itself and preceding context, with loss computed only at slot positions. Dynamic slot insertion enables FA at arbitrary positions. Moreover, non-autoregressive inference is supported, avoiding hallucinations and improving speed. Experiments across multilingual, crosslingual, and long-form speech scenarios show that LLM-ForcedAligner achieves a 69%~78% relative reduction in accumulated averaging shift compared with prior methods. The checkpoint and inference code will be released later.
#### OCR-Enhanced Multimodal ASR Can Read While Listening
 - **Authors:** Junli Chen, Changli Tang, Yixuan Li, Guangzhi Sun, Chao Zhang
 - **Subjects:** Subjects:
Sound (cs.SD); Computation and Language (cs.CL); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2601.18393

 - **Pdf link:** https://arxiv.org/pdf/2601.18393

 - **Abstract**
 Visual information, such as subtitles in a movie, often helps automatic speech recognition. In this paper, we propose Donut-Whisper, an audio-visual ASR model with dual encoder to leverage visual information to improve speech recognition performance in both English and Chinese. Donut-Whisper combines the advantage of the linear and the Q-Former-based modality alignment structures via a cross-attention module, generating more powerful audio-visual features. Meanwhile, we propose a lightweight knowledge distillation scheme showcasing the potential of using audio-visual models to teach audio-only models to achieve better performance. Moreover, we propose a new multilingual audio-visual speech recognition dataset based on movie clips containing both Chinese and English partitions. As a result, Donut-Whisper achieved significantly better performance on both English and Chinese partition of the dataset compared to both Donut and Whisper large V3 baselines. In particular, an absolute 5.75% WER reduction and a 16.5% absolute CER reduction were achieved on the English and Chinese sets respectively compared to the Whisper ASR baseline.
#### Pisets: A Robust Speech Recognition System for Lectures and Interviews
 - **Authors:** Ivan Bondarenko, Daniil Grebenkin, Oleg Sedukhin, Mikhail Klementev, Roman Derunets, Lyudmila Budneva
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2601.18415

 - **Pdf link:** https://arxiv.org/pdf/2601.18415

 - **Abstract**
 This work presents a speech-to-text system "Pisets" for scientists and journalists which is based on a three-component architecture aimed at improving speech recognition accuracy while minimizing errors and hallucinations associated with the Whisper model. The architecture comprises primary recognition using Wav2Vec2, false positive filtering via the Audio Spectrogram Transformer (AST), and final speech recognition through Whisper. The implementation of curriculum learning methods and the utilization of diverse Russian-language speech corpora significantly enhanced the system's effectiveness. Additionally, advanced uncertainty modeling techniques were introduced, contributing to further improvements in transcription quality. The proposed approaches ensure robust transcribing of long audio data across various acoustic conditions compared to WhisperX and the usual Whisper model. The source code of "Pisets" system is publicly available at GitHub: this https URL.
#### Geneses: Unified Generative Speech Enhancement and Separation
 - **Authors:** Kohei Asai, Wataru Nakata, Yuki Saito, Hiroshi Saruwatari
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2601.18456

 - **Pdf link:** https://arxiv.org/pdf/2601.18456

 - **Abstract**
 Real-world audio recordings often contain multiple speakers and various degradations, which limit both the quantity and quality of speech data available for building state-of-the-art speech processing models. Although end-to-end approaches that concatenate speech enhancement (SE) and speech separation (SS) to obtain a clean speech signal for each speaker are promising, conventional SE-SS methods suffer from complex degradations beyond additive noise. To this end, we propose \textbf{Geneses}, a generative framework to achieve unified, high-quality SE--SS. Our Geneses leverages latent flow matching to estimate each speaker's clean speech features using multi-modal diffusion Transformer conditioned on self-supervised learning representation from noisy mixture. We conduct experimental evaluation using two-speaker mixtures from LibriTTS-R under two conditions: additive-noise-only and complex degradations. The results demonstrate that Geneses significantly outperforms a conventional mask-based SE--SS method across various objective metrics with high robustness against complex degradations. Audio samples are available in our demo page.


by Zyzzyva0381 (Windy). 


2026-01-27
