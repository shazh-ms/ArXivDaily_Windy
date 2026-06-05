# Showing new listings for Friday, 5 June 2026
Auto update papers at about 2:30am UTC (10:30am Beijing time) every weekday.


阅读 `Usage.md`了解如何使用此repo实现个性化的Arxiv论文推送

See `Usage.md` for instructions on how to personalize the repo. 


Keyword list: ['text-to-speech', 'text to speech', 'tts', 'LLM-based', 'speech', 'voice']


Excluded: []


### Today: 22papers 
#### Age-Aware Adapter Tuning for Children's Speech Recognition
 - **Authors:** Jialu Li
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2606.05440

 - **Pdf link:** https://arxiv.org/pdf/2606.05440

 - **Abstract**
 Children's automatic speech recognition (ASR) remains challenging because child speech differs from adult speech and varies substantially across developmental stages. While adapter tuning provides a promising way to adapt large pretrained ASR models to children's speech, a single shared child adapter may not fully capture age-dependent variation. In this work, we present one of the first systematic studies of age-aware adapter tuning for child ASR, focusing on speech from children aged 3--12 and older years. We propose age-specialized adapters trained separately for different age groups and compare them with a unified age-conditioned FiLM adapter. With ground-truth age routing, age-specialized adapters improve over the standard shared child adapter baseline from 12.6% to 12.3% overall word error rate (WER) and from 18.4% to 17.6% macro WER, while consistently improving WER for all age groups. We further show that predicted-age routing remains close to ground-truth routing, achieving 12.3% overall WER and 17.8% macro WER without ground-truth age labels at inference. In contrast, unified FiLM conditioning provides smaller gains, indicating that a single unified adapter may be insufficient to capture developmental variation in child speech.
#### Enhancing Audio Captioning with Auxiliary AudioSet Semantics
 - **Authors:** Shubham Gupta, Adarsh Arigala, Sri Rama Murty Kodukula
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2606.05717

 - **Pdf link:** https://arxiv.org/pdf/2606.05717

 - **Abstract**
 Automatic Audio Captioning (AAC) seeks to generate natural language descriptions of complex acoustic scenes, bridging auditory perception and language understanding. However, word-selection indeterminacy and increasing reliance on large-scale sequence-to-sequence or LLM-based models limit practical deployment. We propose a resource-efficient AAC framework that explicitly grounds caption generation in auxiliary AudioSet semantics. Frame-level acoustic representations extracted using a ConvNeXt encoder are augmented with top-$K$ predicted AudioSet keywords, providing structured contextual cues for decoding. A compact six-layer BART-style decoder conditions on this joint acoustic-semantic representation, enabling caption generation without LLM-scale decoding. The proposed design balances semantic grounding and computational efficiency within a compact architecture. Evaluations on Clotho V2 and AudioCaps confirm competitive caption quality under practical deployment constraints.
#### M2S-AVSR: Modality-aware Multi-view Self-supervised Representation for Robust Audio-Visual Speech Recognition
 - **Authors:** Fei Su, Cancan Li, Juan Liu, Ming Li
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2606.05763

 - **Pdf link:** https://arxiv.org/pdf/2606.05763

 - **Abstract**
 Audio-Visual Speech Recognition (AVSR) enhances speech recognition robustness by leveraging visual cues, while real-world scenarios remain challenging due to viewpoint variation, audio distortion, and visual occlusion, which degrade modality quality and increase audio-visual asynchrony. In this paper, we propose a novel Modality-aware Multi-view Self-supervised representation framework for robust Audio-Visual Speech Recognition (M2S-AVSR). First, we introduce a multi-view representation learning encoder to learn view-invariant visual speech representations. Next, we employ a modality-aware module that explicitly models modality quality and cross-modal synchrony to perform fine-grained modality-aware fusion, enabling fine-grained visual information injection during decoding. In addition, we present AISHELL8-RealScene, a public multi-scenario, multi-view conversational audio-visual dataset recorded in real-world environments, and establish a speech recognition benchmark on it. Experiments on English and Mandarin benchmarks demonstrate the effectiveness of the proposed method under challenging conditions. On LRS3, M2S-AVSR achieves up to 29.4% relative improvement under viewpoint perturbation and visual degradation settings. Our method also achieves new state-of-the-art performance on the MISP2021-AVSR test set. On AISHELL8-RealScene, it achieves the best result in outdoor scenes. The proposed method and dataset provide useful support for future research on robust speech and multimodal tasks under realistic conditions.
#### An Ultra-Low-Bitrate Neural Speech Codec with Plain-to-Pseudo Synergistic Vector Quantization
 - **Authors:** Xiao-Hang Jiang, Yang Ai, Fei Liu, Rui-Chen Zheng, Jian-Qing Gao, Zhen-Hua Ling, Ji Wu
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2606.05876

 - **Pdf link:** https://arxiv.org/pdf/2606.05876

 - **Abstract**
 Most neural speech codecs use residual vector quantization (RVQ), in which later VQs contribute less but consume the same bitrate, leading to inefficiency. We propose P2PSynCodec, an ultra-low-bitrate neural speech codec with a plain-to-pseudo synergistic vector quantizer (P2PSVQ). P2PSVQ consists of one plain VQ and multiple pseudo VQs. The plain VQ produces basic tokens by quantization, while the pseudo VQs generate auxiliary tokens by neural prediction and incur zero transmitted bitrate. Thus, speech is decoded from the plain-VQ tokens together with predicted pseudo-VQ tokens, greatly reducing bitrate. Experiments show that P2PSynCodec achieves speech reconstruction quality comparable to competing codecs at 2.0 kbps while operating at only 0.5 kbps, demonstrating high efficiency for ultra-low-bitrate speech coding.
#### VoCodec: A Low-bitrate Streamable Neural Speech Codec with Voicing-driven Quantization
 - **Authors:** Xiao-Hang Jiang, Yang Ai, Rui-Chen Zheng, Li-Rong Dai, Zhen-Hua Ling, Ji Wu
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2606.05892

 - **Pdf link:** https://arxiv.org/pdf/2606.05892

 - **Abstract**
 Neural speech codecs are key to speech transmission and storage, but most use uniform quantization across frames, allocating the same bitrate regardless of content and wasting bits. We propose VoCodec, a low-bitrate streamable neural speech codec with voicing-driven quantization that assigns higher bitrate to voiced frames and lower bitrate to unvoiced frames according to perceptual sensitivity. VoCodec embeds a voicing detector in a fully causal encoder-quantizer-decoder neural coding framework, using residual scalar-vector quantization for voiced frames and simple scalar quantization for unvoiced ones. Experiments show that on the LibriTTS dataset at a 16 kHz sampling rate, VoCodec outperforms baseline neural speech codecs even at a bitrate as low as 1.1 kbps. Our further experiments also confirm that introducing voicing-driven quantization can effectively reduce the bitrate by approximately 27% compared with uniform quantization strategy.
#### CoSTA: Cognitive-State-Conditioned TTS Data Augmentation Using ASR Transcripts for Alzheimer's Disease Detection
 - **Authors:** Yin-Long Liu, Yuanchao Li, Yiming Wang, Yue Li, Rui Feng, Jiaxin Chen, Shaobo Liu, Liu He, Yuang Chen, Jiahong Yuan, Zhen-Hua Ling
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2606.06170

 - **Pdf link:** https://arxiv.org/pdf/2606.06170

 - **Abstract**
 Speech-based Alzheimer's Disease (AD) detection is constrained by scarce pathological speech data. To address this, we propose CoSTA, a Text-to-Speech (TTS)-based data augmentation framework. Specifically, we first develop two Cognitive-State-Conditioned (CS-Cond) TTS models by adapting CosyVoice2 and F5-TTS to synthesize speech with distinct AD and Healthy Control characteristics. Furthermore, by constructing a transcript pool comprising Manual Transcripts (MT) and 36 Automatic Speech Recognition (ASR) transcripts, we investigate the impact of text sources on TTS-based augmentation. We also perform augmentation-factor analysis and test-time augmentation. Experiments on the ADReSS dataset show that CS-Cond TTS significantly improves synthetic speech utility, and ASR-driven augmentation frequently outperforms MT-driven augmentation. Finally, CoSTA yields a 4.16% gain over the baseline, achieving an audio-only accuracy of 85.83% on the ADReSS test set and outperforming prior methods.
#### Revisiting Lexicon Evaluation in Unsupervised Word Discovery
 - **Authors:** Simon Malan, Danel Slabbert, Herman Kamper
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Computation and Language (cs.CL)
 - **Arxiv link:** https://arxiv.org/abs/2606.06183

 - **Pdf link:** https://arxiv.org/pdf/2606.06183

 - **Abstract**
 Building a lexicon from discovered word-like units is a central goal in zero-resource speech processing. But do our evaluations provide a trustworthy indication of lexicon quality? A common metric, normalized edit distance, averages the phoneme edit distances between discovered units in each cluster. We show that this metric has an inherent bias toward the quality of large clusters, inhibiting fair evaluation. Moreover, it ignores how well true classes are distributed across clusters. Based on established theory in clustering literature, we propose two metrics that address these shortcomings: a modified metric that weighs cluster size when assessing within-cluster consistency, and an inverse metric that assesses how true words are spread across clusters. Through experiments on synthetic and real-world lexicons, we demonstrate that combined, these metrics are: (1) more closely correlated with how similar a lexicon is to the ground-truth distribution, and (2) more robust to biases that skew lexicon evaluations.
#### USAD 2.0: Scaling Representation Distillation for Universal Audio Understanding
 - **Authors:** Heng-Jui Chang, Alexander H. Liu, Saurabhchand Bhati, Mrudula Athi, Anton Ratnarajah, Amit Chhetri, James Glass
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Computation and Language (cs.CL); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2606.06444

 - **Pdf link:** https://arxiv.org/pdf/2606.06444

 - **Abstract**
 Audio encoders are critical to modern audio applications as large language models (LLMs) increasingly rely on a single encoder for diverse inputs. While self-supervised learning (SSL) has yielded strong domain-specific encoders like speech or music experts, multi-domain approaches like USAD and SPEAR remain limited in coverage and evaluation. Recent studies also suggest supervised encoders align better with audio LLMs. We present USAD 2.0, a universal encoder integrating knowledge from both SSL and supervised foundation models. USAD 2.0 introduces domain-aware distillation to address teacher mismatch, extends coverage to the music domain, and adds second-stage supervised distillation for downstream use. We further scale the model to one billion parameters via depth scaling. Experiments show USAD 2.0 achieves strong or state-of-the-art performance across probing and LLM-based evaluations.
#### Task-Vector Arithmetic for Emotional Expressivity Control in Language-Model-Based Text-to-Speech
 - **Authors:** Daniel Oliveira de Brito, Arnaldo Candido Junior
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2606.05367

 - **Pdf link:** https://arxiv.org/pdf/2606.05367

 - **Abstract**
 We investigate whether task-vector arithmetic, successful for cross-speaker emotional intensity control in modular text-to-speech (TTS), transfers to large-scale TTS systems built on language-model backbones with in-context learning (LM-TTS). Through a systematic elimination study over four progressively narrower operands on Qwen3-TTS-12Hz-1.7B - model weights via LoRA fine-tuning, continuous codec embeddings, discrete codec tokens, and the speaker embedding (x-vector) produced by an ECAPA-TDNN encoder jointly trained with the synthesis backbone - we localize the dominant carrier of emotional prosody to the x-vector. Building on this finding, we propose a training-free method based on centroid arithmetic in x-vector space: an emotion direction $\tau = \mathbb{E}_i[x(s_i,\text{emo})] -\mathbb{E}_i[x(s_i,\text{neutral})]$ applied to an unseen target speaker as $x_{\text{new}} = x(\text{target},\text{neutral}) + \alpha\cdot\tau$. Using ESD (English) as the $\tau$ source and emoUERJ (Brazilian Portuguese) as a cross-lingual ground-truth target, we observe average gains of $+0.29$ in emotion2vec cosine over the ICL baseline on English held-out speakers and $+0.09$ on Brazilian Portuguese held-out speakers, while largely preserving identity (WavLM SECS $\gtrsim 0.88$ for the multi-speaker $\tau$ variant) and intelligibility (WER $\approx 0$ in PT-BR). These results offer initial evidence that the reported incompatibility of centroid-arithmetic style control with token-based TTS architectures may be circumvented when the arithmetic operates on the speaker embedding.
#### Domain-Aware Mispronunciation Detection and Diagnosis Using Language-Specific Statistical Graphs
 - **Authors:** Huu Tuong Tu, Hanh Nguyen, Thien Van Luong, Nguyen Tien Cuong, Vu Huan, Nguyen Thi Thu Trang
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2606.05569

 - **Pdf link:** https://arxiv.org/pdf/2606.05569

 - **Abstract**
 Mispronunciation Detection and Diagnosis (MDD) has gained increasing importance in computer-assisted language learning and speech technology in recent years. In this paper, we propose a method for constructing statistical graphs that enable models to learn phoneme confusion patterns represented as directed graphs. Furthermore, we introduce a language-specific strategy to capture systematic pronunciation differences across various native language (L1) backgrounds. The effectiveness of our approach is demonstrated through extensive experiments on the L2-ARCTIC benchmark, where it achieves an F1-score of 59.52%, outperforming several competitive baselines.
#### SB-RF: Schrödinger Bridge Rectified Flow for One-Step Robust Speech Enhancement
 - **Authors:** Caixia Lu, Xueyang Lv, Penglong Hu, Jiaming Xu
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2606.05575

 - **Pdf link:** https://arxiv.org/pdf/2606.05575

 - **Abstract**
 Generative models have shown impressive results in speech enhancement but often suffer from multi-step inference. We propose SB-RF, a one-step generative framework integrating Rectified Flow (RF) with Schrödinger Bridge (SB) theory. SB-RF constructs a conditional bridge between clean and noisy speech distributions via entropy-regularized optimal transport. By aligning SB trajectories with the optimal transport geodesic through the velocity-matching objective of RF, SB-RF enables high-quality enhancement with one-step generation. Experiments demonstrate that SB-RF achieves leading performance among generative methods on the VoiceBank-DEMAND benchmark. Furthermore, to fully assess performance in challenging real-world scenarios, we evaluate SB-RF on a simulated low signal-to-noise ratio test set using an expanded training dataset. Under these conditions, SB-RF exhibits strong and competitive robustness with high efficiency, validating its potential for real-world applications.
#### Do speech foundation models perceive speaker similarity as humans do?
 - **Authors:** Minoru Kishi, Hayato Yagi, Shinnosuke Takamichi, Yuki Saito
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2606.05739

 - **Pdf link:** https://arxiv.org/pdf/2606.05739

 - **Abstract**
 This study presents a comparative analysis between the speaker embeddings of speech foundation models and human subjective perception of speaker similarity. Human listeners have the ability to judge speaker similarity on a continuous scale discerning how similar two voices are. In contrast, speech foundation models embed speaker characteristics into numerical representation. However, a question remains: does the numerical distance between speaker embeddings in these models truly align with the similarity perceived by humans? To address this, we conduct a comprehensive investigation using more than 40 models to compare model-derived distances with human-perceived similarity scores. Furthermore, we identify which factors in model configuration contribute most to a speaker embedding that mirrors human perception. Our findings provide insights for the development of more perceptually grounded speech foundation models.
#### Towards Truly Multilingual ASR: Generalizing Code-Switching ASR to Unseen Language Pairs
 - **Authors:** Gio Paik, Hyunseo Shin, Soungmin Lee
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2606.05846

 - **Pdf link:** https://arxiv.org/pdf/2606.05846

 - **Abstract**
 Automatic Speech Recognition (ASR) has become a key technology for human--AI interaction. However, code-switching ASR (CS-ASR) remains particularly challenging due to the severe scarcity of multilingual CS speech resources across diverse language pairs. Existing approaches primarily improve CS-ASR performance through synthetic CS speech generation or pair-specific fine-tuning on limited bilingual datasets. Nevertheless, these approaches face an inherent scalability limitation, as support for CS must be developed separately for language pairs whose number grows combinatorially with the number of supported languages. In this work, we investigate whether CS capabilities learned from a limited set of seen language pairs can generalize to unseen language pairs through model merging and domain generalization methods. Our experiments show that merged bilingual CS-ASR models modestly generalize to unseen language pairs, suggesting limited transfer of bilingual CS capabilities across language pairs.
#### UniVoice: A Unified Model for Speech and Singing Voice Generation
 - **Authors:** Junjie Zheng, Huixin Xue, Shihong Ren, Chaofan Ding, Hao Liu, Zihao Chen
 - **Subjects:** Subjects:
Sound (cs.SD); Artificial Intelligence (cs.AI); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2606.05852

 - **Pdf link:** https://arxiv.org/pdf/2606.05852

 - **Abstract**
 Text-to-speech (TTS) and singing voice synthesis (SVS) both aim to generate human vocal audio from symbolic inputs, but they impose different requirements on the generation process. Speech generation relies on flexible, language-driven prosody, whereas singing generation requires explicit melody control and accurate rhythmic alignment. This mismatch makes it challenging to train a single model that can generate both natural speech and controllable singing, since melody-related conditions should strongly constrain singing but should not restrict speech prosody. We present UniVoice, a unified speech and singing voice generation framework based on conditional flow matching. Instead of using a single undifferentiated conditioning representation, UniVoice factorizes the condition into content, melody, and timbre, which are encoded by modality-appropriate encoders and consumed by a shared Diffusion Transformer (DiT) backbone. For singing, the melody condition is represented by MIDI note sequences; for speech, it is replaced with a learned null melody token, allowing the model to infer prosody from linguistic and acoustic context. This design preserves explicit melody control for singing while avoiding the need to impose melody constraints on speech. We further analyze the null melody token as an approximation to melody marginalization in the conditional flow. Trained on 30k hours of speech and 35k hours of singing data, UniVoice achieves a speech PER of 5.26\%, comparable to dedicated TTS systems such as F5-TTS (5.21\%) and CosyVoice3 (5.30\%). On singing generation, UniVoice achieves a PER of 16.22\%, outperforming the unified baseline Vevo1.5 (24.72\%).
#### GLASS: GRPO-Trained LoRA for Acoustic Style Steering in Zero-Shot Text-to-Speech
 - **Authors:** Jaehoon Kang, Yejin Lee, Kyuhong Shim
 - **Subjects:** Subjects:
Sound (cs.SD); Computation and Language (cs.CL); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2606.05889

 - **Pdf link:** https://arxiv.org/pdf/2606.05889

 - **Abstract**
 We propose GLASS, a framework for composable acoustic style control in zero-shot autoregressive text-to-speech (TTS) that learns controls from post-generation rewards rather than style labels. In zero-shot TTS, a speaker prompt often entangles speaker identity with prosodic attributes such as speaking rate and pitch, making it difficult to change style without changing the prompt itself. GLASS instead treats each acoustic attribute as a reward-defined control direction. For each control axis, GLASS freezes the TTS backbone and trains one lightweight LoRA adapter with Group Relative Policy Optimization (GRPO), using speech-token length and mean F0 as style rewards and WER as an intelligibility anchor. Because each control is represented as a LoRA weight update, independently trained adapters can be swapped, interpolated, and composed through linear LoRA arithmetic without retraining the backbone. Experiments on speaking rate and pitch control show targeted style shifts while preserving naturalness, speaker similarity, and intelligibility, and demonstrate smooth interpolation and multi-axis composition across independently trained adapters.
#### Beyond WER: A Paired Acoustic Stress Test for Ambient Clinical Scribes
 - **Authors:** Xiao-Hang Jiang, Han-Jie Guo, Ying-Si Liang, Yang Ai, Zhen-Hua Ling, Lei Jiang, Zhi-Yang He
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2606.05909

 - **Pdf link:** https://arxiv.org/pdf/2606.05909

 - **Abstract**
 Ambient clinical scribes increasingly combine Automatic Speech Recognition with Large Language Models to automate documentation. However, traditional metrics like Word Error Rate mask systemic safety degradation. We present a paired acoustic stress test to isolate the causal impact of noise on clinical reasoning. For the same dialogues, we inject diverse noise types while keeping the downstream model configuration frozen. Crucially, we uncover a dangerous disconnect between signal fidelity and clinical safety. Stationary ambient noise increased the Word Error Rate by a negligible 0.71 percentage points yet nearly doubled the rate of unsafe outputs. Our analysis reveals that minor acoustic perturbations can invert clinical meaning without substantially inflating error rates. Furthermore, we demonstrate a lightweight mitigation strategy that mitigates safety degradation under noisy conditions without requiring model fine tuning.
#### DBHN-Net: Dual-Branch Hybrid Neural Network For Low-Complexity Monaural Speech Enhancement
 - **Authors:** Cunhang Fan, Enrui Liu, Jing Zhou, Jian Kang, Jie Li, Andong Li, Jian Zhou, Zhao Lv, Xuelong Li
 - **Subjects:** Subjects:
Sound (cs.SD); Machine Learning (cs.LG); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2606.05911

 - **Pdf link:** https://arxiv.org/pdf/2606.05911

 - **Abstract**
 Although artificial neural network (ANN) based speech enhancement (SE) methods demonstrate excellent performance, the high computational complexity and high energy consumption hinder their deployment in practical front-end processing tasks.} Currently, the spiking neural networks (SNNs) have shown potential in reducing power consumption. However, the discrete binary activation and complex spatio-temporal dynamics of SNNs often result in information loss. The current challenge therefore focuses on how to maintain performance and reduce computational complexity. To address this issue, this work propose a Dual-Branch Hybrid Neural (DBHN) Network. 1) In terms of network architecture: A dual-branch network integrating ANN and SNN was designed, where the SNN branch reduces power consumption while the ANN branch addresses information loss; The BandSplit and Time-Frequency (TF) -Mamba modules were developed to simultaneously compress energy consumption and enhance model performance; Spiking Feature Extraction Group (SFEG) and Information Transformation Block (ITB) components were implemented with residual connections to mitigate information loss while further refining feature representations. 2) To facilitate inter-branch information fusion: An Interaction module was designed to promote information exchange at various stages of the dual-branch network; A TF-Cross Attention-Fusion module was designed to perform time-frequency domain fusion of dual-branch information while data-adaptively guiding the SNN branch to retain more critical information. Results show that the proposed model maintains superior performance across three public datasets while achieving an average 7.5 fold reduction in computational complexity compared to baseline models.
#### To Be Multimodal or Not to Be: Query-Adaptive Audio-Visual Person Retrieval via Active Modality Detection
 - **Authors:** Erfan Loweimi, Mengjie Qian, Kate Knill, Guanfeng Wu, Chi-Ho Chan, Abbas Haider, Muhammad Awan, Josef Kittler, Hui Wang, Mark Gales
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Artificial Intelligence (cs.AI); Computer Vision and Pattern Recognition (cs.CV); Information Retrieval (cs.IR); Machine Learning (cs.LG); Multimedia (cs.MM); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2606.05931

 - **Pdf link:** https://arxiv.org/pdf/2606.05931

 - **Abstract**
 When retrieving a person from a video archive by voice and face, should the system be multimodal or not? In real-world broadcast archives, unlike curated benchmarks, a target may be heard but unseen, seen but unheard, or both. Fusing scores from an absent modality injects noise, degrading precision below the best unimodal system. We propose a query-adaptive framework that detects active modalities via cross-modal score consistency: when both modalities are active, files retrieved by one also score highly on the other; this agreement breaks down when a modality is absent. Classifiers driven by these cross-modal features achieve 89% detection accuracy. On the BBC Rewind corpus (with over 12,000 broadcast videos) the adaptive system attains 94.2% P@1, outperforming speaker-only (82.9%), face-only (93.4%), and fixed fusion (90.0%), recovering 64% of the gap to an oracle with ground-truth modality labels (96.6%).
#### SpeechJBB: Probing Safety Alignment and Comprehension in Large Audio Language Models under Code-Switched Speech
 - **Authors:** Virginia Ceccatelli, Yejin Jeon, David Ifeoluwa Adelani
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2606.06037

 - **Pdf link:** https://arxiv.org/pdf/2606.06037

 - **Abstract**
 Large audio language models (LALMs) are increasingly deployed in real-world applications, yet their safety alignment is still primarily evaluated on monolingual, text-based harmful prompts. This leaves their generalizability under multilingual and spoken settings, particularly code-switched speech, largely underexplored. To address this gap, we introduce SpeechJBB, an audio jailbreak dataset for benchmarking across multiple state-of-the-art LALMs. The extent of safety weaknesses is further probed by introducing an augmented setting where phonologically plausible pseudo-words are inserted around safety-critical terms to simulate localized obfuscation. Across models, code-switched harmful audio yields substantially high jailbreak success rates (JSR), with non-English monolingual and non-English code-switched pairs exhibiting the highest attack success. Pseudo-word insertion further reduces refusal rates, which demonstrates that natural-sounding obfuscation can effectively bypass safety policies.
#### Multi-task Learning is Not Enough: Representational Entanglement in Dual-output Second Language Speech Recognition
 - **Authors:** Seung Hwan Cho, Young-Min Kim
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2606.06065

 - **Pdf link:** https://arxiv.org/pdf/2606.06065

 - **Abstract**
 Second-language (L2) speech recognition often requires transcriptions of pronunciations and intended meanings. Multi-task learning (MTL) is a natural approach because it assumes that shared representations benefit both outputs. However, this paper shows that this assumption does not hold across Korean and English. MTL improves meaning but degrades surface transcription, especially in English, where the degradation scales with surface-meaning divergence measured by Levenshtein edit this http URL analysis links these patterns to encoder-level entanglement, with Korean preserving distinct task representations while English produces nearly identical ones. Cross-task decoder analysis shows that the meaning dual-output decoder adapts with a unique representation, while the surface dual-output decoder remains constrained by the encoder. These findings motivate the design of MTL frameworks that mitigate encoder-level entanglement to reduce surface degradation in dual-output L2 automatic speech recognition.
#### Learning Emotion-discriminative Representations for Zero-Shot Cross-lingual Speech Emotion Recognition
 - **Authors:** Jinyi Mi, Ding Ma, Tomoki Toda
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2606.06200

 - **Pdf link:** https://arxiv.org/pdf/2606.06200

 - **Abstract**
 Zero-shot cross-lingual speech emotion recognition (SER) remains challenging due to distribution mismatches across languages and the lack of emotion annotations in target language. Under such conditions, models trained solely on source-language data frequently suffer from degraded generalization when evaluated on unseen target languages. To address this limitation, we propose an emotion-discriminative representation learning method that integrates supervised contrastive learning and speaker adversarial learning. The contrastive learning promotes cross-lingual emotion alignment, while speaker adversarial learning suppresses speaker-related cues to encourage speaker-invariant representations. Experimental results under a zero-shot cross-lingual SER setting demonstrate that the proposed method significantly improves SER performance over conventional training strategies.
#### FiLM-Based Speaker Conditioning of a SpeechLLM for Pathological Speech Recognition
 - **Authors:** Fernando López, Santosh Kesiraju, Jordi Luque
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2606.06211

 - **Pdf link:** https://arxiv.org/pdf/2606.06211

 - **Abstract**
 Automatic speech recognition (ASR) has advanced remarkably for standard speech; however, pathological speech from neurological conditions remains a significant challenge. We investigate speaker conditioning via Feature-wise Linear Modulation (FiLM), injecting x-vector-derived information into each transformer layer of a frozen ASR encoder to adapt internal representations to individual pathological speakers without modifying base model weights. We benchmark this for the ASR task against standard and parameter-efficient fine-tuning baselines, complemented by post-processing, on Spanish and English pathological speech. Additionally, we evaluate if the adapted model preserves the ability to answer speech-related questions. Results show that speaker-conditioned ASR is competitive with established adaptation strategies while retaining performance on non-conditioned speech.


by Zyzzyva0381 (Windy). 


2026-06-05
