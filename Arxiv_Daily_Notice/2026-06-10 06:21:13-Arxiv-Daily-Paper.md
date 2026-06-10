# Showing new listings for Wednesday, 10 June 2026
Auto update papers at about 2:30am UTC (10:30am Beijing time) every weekday.


阅读 `Usage.md`了解如何使用此repo实现个性化的Arxiv论文推送

See `Usage.md` for instructions on how to personalize the repo. 


Keyword list: ['text-to-speech', 'text to speech', 'tts', 'LLM-based', 'speech', 'voice']


Excluded: []


### Today: 15papers 
#### LLM can Read Spectrogram: Encoder-free Speech-Language Modeling
 - **Authors:** Ruchao Fan, Yiming Wang, Yuxuan Hu, Bo Ren, Yufei Xia, Xiaofei Wang, Yao Qian, Jinyu Li
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2606.10231

 - **Pdf link:** https://arxiv.org/pdf/2606.10231

 - **Abstract**
 Recent speech-aware large language models (Speech-LLMs) rely on a pre-trained speech encoder to convert audio into semantic-rich representations consumable by LLM. In this work, instead, we explore: can an LLM learn to read Mel spectrogram directly without a dedicated speech encoder? We propose Mel-LLM, an encoder-free Speech-LLM that feeds lightly pre-processed Mel spectrogram patches directly into the LLM through a linear projection, allowing the LLM to learn speech-text alignment purely through its own parameters. We conduct extensive experiments on both automatic speech recognition (ASR) and text-to-speech (TTS) tasks. For ASR, we evaluate on the OpenASR leaderboard public sets and production-level scaling experiments, demonstrating that the encoder-free solution achieves competitive performance with only limited degradation compared to encoder-initialized counterparts. We find that when data is limited, initialization from a multimodal checkpoint (Phi-4-MM) is crucial for maintaining performance. We also present ablation studies revealing which LLM layers are less relevant to speech encoding. For TTS, we show preliminary results with a next-token VAE approach. While TTS performance is not yet optimal, these results establish the feasibility of a fully unified encoder-free architecture for autoregressive speech-text modeling.
#### ANCHOR: Autoregressive Non-intrusive Chunk-Ordered Refinement for Joint Multi-Resolution Speech Quality Modeling
 - **Authors:** Zhuoyan Tao, Jiatong Shi, Hye-jin Shim, Shinji Watanabe
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Machine Learning (cs.LG); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2606.10233

 - **Pdf link:** https://arxiv.org/pdf/2606.10233

 - **Abstract**
 While speech quality is typically assessed on complete utterances, streaming and generative systems require incremental estimation from partial audio. Existing predictors assume full context, degrading on prefix-constrained inputs. Extending ARECHO, we propose ANCHOR, reformulating incremental assessment as a multi-resolution autoregressive task. It models chunk- and utterance-level quality within a single decoder using dual-resolution tokens and a resolution-aware hierarchy for coarse-to-fine refinement. Experiments show substantial robustness under partial input, including a 48% PLCMOS error reduction on 2-second prefixes. Convergence analysis reveals a 4-6 s effective perceptual context horizon. A stress test further isolates structured extrapolation biases under localized corruption. Results demonstrate that hierarchical supervision improves incremental prediction and elucidates how perceptual quality accumulates over time.
#### SSL-GMMVC: Interpretable Voice Conversion via Locally Linear GMM Transforms in Self-Supervised Representation Space
 - **Authors:** Tomoya Tanabu, Hiroshi Nishijima, Daisuke Saito, Nobuaki Minematsu
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2606.10317

 - **Pdf link:** https://arxiv.org/pdf/2606.10317

 - **Abstract**
 We introduce SSL-GMMVC, an interpretable voice conversion method in self-supervised speech space. The method models paired source-target features with a Gaussian mixture model and performs conversion as a posterior-weighted sum of affine transforms. This yields locally linear transformations that adapt to heterogeneous feature-space structure while remaining analytically tractable. Through objective and subjective evaluations, we show that SSL-GMMVC improves speaker similarity with comparable intelligibility and naturalness, and that even a constrained covariance variant surpasses a deep learning baseline as the number of mixture components increases. Further analyses link component selection to phonetic structure and reveal interpretable scaling and rotation in the learned transforms. These findings highlight SSL-GMMVC as an effective, analyzable framework for voice conversion.
#### Entropy-Aware Domain-Routed Mixture-of-Experts Speech-LLM Framework: A Case Study of Multi-Domain Child-Adult ASR
 - **Authors:** Mohan Shi, Kaiyuan Zhang, Zilai Wang, Natarajan Balaji Shankar, Eray Eren, Abeer Alwan
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2606.10454

 - **Pdf link:** https://arxiv.org/pdf/2606.10454

 - **Abstract**
 While Speech Large Language Models (Speech-LLMs) have achieved strong performance on adult Automatic Speech Recognition (ASR), their effectiveness on child speech remains under-explored, and single models often struggle to handle diverse adult and child age groups simultaneously. This paper proposes a Mixture-of-Experts (MoE) Speech-LLM for unified ASR across adult and child speech spanning diverse environments and age groups. The framework employs a Classifier-based Domain Router (C-DR) with a coarse-to-fine strategy and integrates both a Mixture-of-Projectors (MoP) and a Mixture-of-LoRAs (MoL) to model domain-specific variations. To address routing uncertainty near domain boundaries, an Entropy-Aware Routing (EAR) mechanism is introduced to dynamically incorporate a shared expert. Experiments on public child corpora demonstrate consistent improvements over baselines while preserving adult ASR performance. To our knowledge, this is the first work leveraging Speech-LLMs for unified, multi-domain ASR encompassing both children and adults.
#### GC-LoRA: Gated Convolutional LoRA for Parameter-Efficient Acoustic Adaptation
 - **Authors:** Natarajan Balaji Shankar, Zilai Wang, Kaiyuan Zhang, Mohan Shi, Abeer Alwan
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2606.10464

 - **Pdf link:** https://arxiv.org/pdf/2606.10464

 - **Abstract**
 Transformer-based Speech Foundation Models excel in most Automatic Speech Recognition tasks but often suffer performance degradation when applied to domains with mismatched acoustic characteristics. While Parameter Efficient Fine-Tuning (PEFT) methods, such as Low-Rank Adaptation (LoRA), adjust global attention, they lack the local context modeling crucial for capturing domain-specific variations. We propose GC-LoRA, a novel adapter architecture that injects Conformer-style local convolutional processing into pretrained Transformer encoders. By integrating a lightweight adapter to encoder attention output projections, our method efficiently captures local acoustic dependencies without disrupting pretrained global representations. Experiments across diverse datasets (acoustically-degraded, bandlimited, dialectal, child) demonstrate the efficacy of our approach, achieving Word Error Rate (WER) reductions of up to 10.9% compared to baselines while adding minimal trainable parameters.
#### Anchoring the Unknown: Open-Set Model Attribution via Proxy-Anchor Learning
 - **Authors:** Cristian-Teodor Neamtu, Serban Mihalache, Stefan Smeu, Dan Oneata, Horia Cucu, Dragos Burileanu
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2606.10758

 - **Pdf link:** https://arxiv.org/pdf/2606.10758

 - **Abstract**
 The proliferation of text-to-speech (TTS) systems capable of generating realistic synthetic speech poses growing challenges for audio forensics. While binary deepfake detection has received considerable attention, source tracing (i.e., identifying which TTS system produced a given audio sample) remains underexplored, particularly in open-set scenarios where unknown systems may be encountered. We propose a metric learning framework based on the Proxy-Anchor loss function that operates on Wav2Vec2-BERT embeddings to learn a discriminative embedding space for TTS source attribution and out-of-distribution (OOD) detection of unseen systems. We evaluate it on the MLAAD v9 dataset spanning 140 TTS systems across 51 languages, and introduce an architecture merging strategy that groups TTS system versions into unified classes, reducing inter-class confusion. Our system achieves 99.76% accuracy on 110 in-distribution classes and a False Positive Rate (FPR@95) as low as 2.04% for OOD detection. Also, for a fair comparison against the current state of the art, we further evaluate it on the MLAAD v5 official dataset splits, improving the OOD accuracy by almost doubling it. These results demonstrate that Proxy-Anchor metric learning, combined with architecture-aware class design and post-hoc OOD scoring, provides an effective framework for forensic TTS source tracing in both closed-set and open-set settings.
#### Recovering the Zipfian Distribution in Unsupervised Term Discovery
 - **Authors:** Danel Slabbert, Simon Malan, Herman Kamper
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Computation and Language (cs.CL)
 - **Arxiv link:** https://arxiv.org/abs/2606.10781

 - **Pdf link:** https://arxiv.org/pdf/2606.10781

 - **Abstract**
 Unsupervised term discovery involves segmenting unlabelled speech into word- or syllable-like units and clustering these into a lexicon of candidate types. True lexicons follow a Zipfian distribution, yet the dominant centre-based clustering approach -- K-means -- produces a more uniform distribution due to an inductive bias toward spherical clusters. In this paper we revisit graph-based clustering as a bottom-up alternative, where segment embeddings are connected by pairwise similarity and partitioned using the Leiden algorithm. We show that graph clustering substantially outperforms centre-based approaches (K-means, GMM, BIRCH) in both word- and syllable-level lexicon discovery across three languages, producing more Zipf-like distributions. Another bottom-up approach, agglomerative clustering with average linkage, also performs well, although it is computationally less efficient and allows for less control over the resulting distribution. Our work calls into question the dominance of centre-based clustering for term discovery, and promotes graph clustering as an attractive alternative.
#### Towards Deep Contextual Reasoning from Broad Descriptions for ASR with Speech-LLM via Metadata-Driven Reasoning Chains
 - **Authors:** Jakob Poncelet, Hugo Van hamme
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2606.10838

 - **Pdf link:** https://arxiv.org/pdf/2606.10838

 - **Abstract**
 Speech recognition often fails on rare, domain-specific terms and context-related named entities. Existing contextualization techniques typically bias decoding with keywords or phrase lists, which does not scale well or exploit deeper knowledge. We propose a training method that teaches a speech-LLM to use broad descriptions (e.g. from videos) as weak semantic priors to perform contextual reasoning grounded in the audio. We build 400 hours of reasoning-augmented speech data by pairing erroneous hypotheses with video metadata and LLM-generated reasoning explanations that justify context-driven corrections. We finetune the speech-LLM to perform chain-of-thought reasoning: generate an initial transcript, then reason over the context, and finally return a corrected transcript. On held-out YouTube-derived test sets, our approach reduces errors, with specific improvements on rare words and named entities, and lays groundwork for deeper contextual reasoning in speech recognition.
#### Speech Encoder Fusion for LLM-based Automatic Speech Recognition
 - **Authors:** Jakob Poncelet, Hugo Van hamme
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2606.10853

 - **Pdf link:** https://arxiv.org/pdf/2606.10853

 - **Abstract**
 Speech-aware large language models (LLMs) can incorporate speech through pre-trained acoustic encoders that project speech features into the LLM embedding space. While the choice of the speech encoder critically influences performance, different encoders often exhibit complementary strengths, motivating their combination. In this work, we investigate whether fusing multiple pre-trained speech encoders can enhance speech-aware LLMs for automatic speech recognition (ASR). We explore several fusion strategies beyond simple feature concatenation, including learned combinations and Transformer-based fusion architectures, and evaluate them across mono- and multilingual ASR settings as well as diarized speech recognition. Our results indicate that carefully fusing multiple parallel speech encoders improves downstream performance in all scenarios with limited computational overhead.
#### Phoneme-First Prediction for LLM-Based Speech Recognition
 - **Authors:** Jakob Poncelet, Hugo Van hamme
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2606.10864

 - **Pdf link:** https://arxiv.org/pdf/2606.10864

 - **Abstract**
 Recent research has explored integrating Large Language Models (LLMs) with speech encoders to create speech-augmented LLMs capable of contextualized speech recognition. The main challenge lies in aligning the semantic embeddings of LLMs with the acoustic representations of speech encoders. We propose a novel approach that teaches the LLM to first predict phonemes from the speech features before generating the final transcript. By integrating a phoneme prediction step directly into the LLM, the model develops a fine-grained knowledge of pronunciation, reducing acoustic confusion and improving transcription accuracy and explainability. Our method is cheap and simple, as phoneme targets can be automatically derived from existing transcripts. Through comprehensive experiments, we show that intermediate phoneme prediction can improve speech recognition, particularly in low-resource settings, and yields outputs that are acoustically more faithful to the speech.
#### Enhancing Multilingual LLM-based ASR with Mixture of Experts and Dynamic Downsampling
 - **Authors:** Guodong Lin, Ziqi Chen, Yuxiang Fu, Ke Li, Wei-Qiang Zhang
 - **Subjects:** Subjects:
Sound (cs.SD); Computation and Language (cs.CL); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2606.10439

 - **Pdf link:** https://arxiv.org/pdf/2606.10439

 - **Abstract**
 The rapid progress of large language models (LLMs) has opened up a new frontier for automatic speech recognition (ASR), making their effective integration a critical and challenging research direction. To this end, this work proposes a projector-based LLM-ASR framework targeting the key challenges of multilingual generalization and modality alignment. Our approach incorporates a Mixture of Experts (MoE) architecture to improve cross-lingual adaptability, and a Continuous Integrate-and-Fire (CIF) mechanism for dynamic downsampling and modality alignment. Experimental results show that the combination of these components yields substantial performance improvements, surpassing strong baseline models. The proposed method represents a step toward building more accurate, robust, and generalizable LLM-based ASR systems.
#### A Lightweight Dual-Factor Acoustic Authentication System via Cascaded GMM-DTW Architecture for Edge Computing
 - **Authors:** Yutong Zhang
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2606.10565

 - **Pdf link:** https://arxiv.org/pdf/2606.10565

 - **Abstract**
 This paper presents a lightweight, cascaded GMM-DTW dual-factor voice lock system for resource-constrained edge environments. By utilizing a shared MFCC feature space, the framework implements a sequential defense mechanism combining GMM speaker screening and DTW passphrase verification. To counter presentation threats without extra hardware, a dynamic joint absolute-relative margin constraint is integrated into the GMM classification space, limiting the physical imposter and high-fidelity replay attack False Acceptance Rates (FAR) to 2.73% and 6.67%, respectively, with a legitimate False Rejection Rate (FRR) of 16.67%. Due to Sakoe-Chiba window optimization, the global end-to-end processing latency under temporal stress is rigidly bounded at 9.82ms on a single-core CPU, comprising 1.51ms for feature extraction, 0.54ms for GMM scoring, and 7.77ms for worst-case DTW matching. These empirical benchmarks demonstrate the viability of white-box acoustic cascades for secure, deterministic real-time deployment on low-power edge nodes.
#### ParaBridge: Bridging Paralinguistic Perception and Dialogue Behavior in Speech Language Models
 - **Authors:** Yuxiang Wang, Qinke Ni, Shengbo Cai, Wan Lin, Liqiang Zhang, Zhizheng Wu
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2606.10581

 - **Pdf link:** https://arxiv.org/pdf/2606.10581

 - **Abstract**
 Speech carries more information than just words: a child's voice, a fearful tone, or a noisy background should all lead a sufficiently competent spoken-dialogue assistant to different replies. Current Speech Language Models (SLMs) can recognize such paralinguistic cues but often ignore them in open-ended dialogue. We observe that a simple paralinguistic instruction scaffold at the inference stage narrows this perception-behavior gap, suggesting that the relevant cues are already latent in the model. Such scaffolds, however, remain brittle under multi-turn context and competing instructions. Therefore, we propose \textbf{ParaBridge}, an on-policy self-distillation method that turns a brittle inference-time scaffold into stable model behavior. During training, the scaffold serves only as a temporary privileged view; the scaffold-free model rolls out its own response, while the scaffolded view supplies dense, full-vocabulary next-token targets along its trajectory. This supervision teaches when non-lexical cues should affect the reply without the need for curated dialogues, human labels, or external reward models. On Qwen3-Omni-thinking, ParaBridge raises scaffold-free VoxSafeBench SAR from $14.6\%$ to $40.3\%$ and improves EchoMind average rating from $3.27$ to $3.92$. It also preserves general ability, with MMAU-Pro, VoiceBench, and GPQA all within $0.4$ points of the original model. Beyond the training distribution, ParaBridge generalizes to unseen paralinguistic cues, transfers from safety-oriented training to empathy-oriented dialogue, and works on a different SLM backbone.
#### Multilingual Word-Level Forced Alignment with Self-Supervised Representations and Learned Dynamic Programming
 - **Authors:** Roy Weber, Meidan Zehavi, Rotem Rousso, Joseph Keshet
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2606.10675

 - **Pdf link:** https://arxiv.org/pdf/2606.10675

 - **Abstract**
 We present a method for accurate multilingual word-level forced alignment, consisting of an alignment encoder and a learned alignment decoder. The encoder integrates two representations: one from the Massively Multilingual Speech (MMS) model and another from a self-supervised phoneme boundary detector (UnSupSeg). It learns to fuse them and to estimate word-boundary probabilities over long temporal contexts. The alignment decoder is a learned dynamic programming that combines encoder outputs with segmental features over the MMS and UnSupSeg representations to infer final word boundaries. Trained iteratively on TIMIT and Buckeye, the proposed approach outperforms Montreal Forced Aligner (MFA) and MMS-based alignment on both datasets. On unseen languages (Dutch, German, and Hebrew), the proposed model achieves performance consistently better than or on par with existing alignment approaches, indicating its potential to scale to 1100+ languages supported by MMS without further training.
#### Multi-Faceted Interactivity Alignment in Full-Duplex Speech Models
 - **Authors:** Atsumoto Ohashi, Neil Zeghidour, Alexandre DÃ©fossez, Eugene Kharitonov
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2606.11167

 - **Pdf link:** https://arxiv.org/pdf/2606.11167

 - **Abstract**
 Full-duplex spoken dialogue models can listen and speak simultaneously, making them a promising architecture for natural conversation. However, current models are trained solely with supervised learning through token-level likelihood maximization, which does not directly optimize interaction-level behaviors, causing interactivity issues such as excessive silence and ill-timed turn-taking. Recent work has applied reinforcement learning (RL) to improve interactivity, but existing methods address only a limited set of interactive behaviors in their rewards. In this work, we propose a post-training alignment method that comprehensively improves the interactivity of full-duplex spoken dialogue models through RL. We address the four canonical axes of interactivity: pause handling, turn-taking, backchanneling, and user interruption. For each axis, we extract short audio segments from human conversation corpora and optimize the model with axis-specific reward functions. An extra LLM-based reward for response quality prevents semantic degradation. We apply our method to two open-source models, Moshi and PersonaPlex, demonstrating consistent improvements in interactivity on both offline evaluation with pre-recorded audio and real-time multi-turn dialogue evaluation.


by Zyzzyva0381 (Windy). 


2026-06-10
