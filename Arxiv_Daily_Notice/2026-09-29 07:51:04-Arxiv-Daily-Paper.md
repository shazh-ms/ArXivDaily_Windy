# Showing new listings for Tuesday, 29 September 2026
Auto update papers at about 2:30am UTC (10:30am Beijing time) every weekday.


阅读 `Usage.md`了解如何使用此repo实现个性化的Arxiv论文推送

See `Usage.md` for instructions on how to personalize the repo. 


Keyword list: ['text-to-speech', 'text to speech', 'tts', 'LLM-based', 'speech', 'voice']


Excluded: []


### Today: 50papers 
#### OneVoice: An Intermediate Representation for Agentic Speech Pipelines
 - **Authors:** Vipul Charugundla, Dancheng Liu, Jinjun Xiong
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Multiagent Systems (cs.MA); Multimedia (cs.MM); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2609.31673

 - **Pdf link:** https://arxiv.org/pdf/2609.31673

 - **Abstract**
 Agentic speech systems must exchange more than text, including speaker identity, timing, phonetic information, and behavioral annotations. Yet these signals are often produced in incompatible tool-specific formats, making agent-to-agent handoff fragile. We present OneVoice, a lightweight, JSON-native intermediate representation that provides a shared semantic structure for speech pipelines. OneVoice organizes heterogeneous speech evidence into validated session records with stable identifiers, layered transcripts, explicit timing relationships, and provenance information. We evaluate OneVoice in two complementary multi-agent workflows covering event aggregation and acoustic-phonetic temporal linking across three language models. Compared with implicit agent-defined handoffs, OneVoice substantially improves aggregation reliability and the preservation of temporal relationships, demonstrating the value of an explicit speech-specific representation for agent communication.
#### DiffVQE2: An Efficient Low-delay Diffusion Model for Acoustic Echo and Noise Control
 - **Authors:** Haljan Lugo, Ernst Seidel, Pejman Mowlaee, Ziyue Zhao, Tim Fingscheidt
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2609.31703

 - **Pdf link:** https://arxiv.org/pdf/2609.31703

 - **Abstract**
 Hands-free communication devices and speakerphones are inherently affected by acoustic echo and background noise. To mitigate these impairments, end-to-end discriminatively trained neural networks have emerged as the best-performing approach in research and deployment. While recent advancements in generative methods have provided remarkable results for various speech enhancement tasks, diffusion-based acoustic echo control (AEC) research is still restricted to non-causal, utterance-level processing, thereby not widely applicable in practice. With this work, we are the first to propose low-delay (i.e., causal) diffusion-based joint AEC and noise control models DiffVQE2 / DiffVQE2-S, excelling the so-far state of the art DeepVQE / DeepVQE-S models in multiple objective metrics, and, most importantly, in subjective MOS, respectively. In addition, our models are less complex. Furthermore, we show that a limited lookahead applied to the efficient DiffVQE2-S model allows for an even higher performance. These results have been obtained on the ICASSP 2023 AEC Challenge blind test set. We claim the first streaming-capable, diffusion-based acoustic echo and noise control that excels state-of-the-art discriminative approaches.
#### RadarVox: Radar-Audio Multimodal Cocktail-Party Speech Separation with Speaker-Aware Cross-Modal Matching
 - **Authors:** Yanlin Xu, Yiwei Ru, Mupei Li, Yongji Liu, Jie Wang, Zhenan Sun
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Multimedia (cs.MM); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2609.31708

 - **Pdf link:** https://arxiv.org/pdf/2609.31708

 - **Abstract**
 In embodied voice interaction, cocktail-party speech perception requires both speech separation and speaker attribution across machine-generated and human speech sources. However, conventional audio-only blind source separation remains permutation ambiguous, making the correspondence between separated streams and physical speakers unclear. This paper presents RadarVox, a radar-audio multimodal benchmark for identity-aware cocktail-party speech separation. RadarVox provides acoustic mixtures and source-level radar displacement signals from loudspeaker-emitted and human speech, enabling source-aware speaker assignment. FMCW radar captures laryngeal mechanical motion, providing speaker-specific spatial-motion cues unavailable to a single-channel microphone. We inject radar-derived priors into a DPRNN separator via gated fusion and learn a speaker-aware cross-modal matcher to associate unordered speech streams with radar-observed speakers. Experiments on multi-speaker mixtures show that radar cues provide complementary benefits, achieving scale-invariant signal-to-distortion ratio (SI-SDR) values of 9.75 dB and 7.13 dB in two- and three-speaker scenarios, respectively. More importantly, the proposed method improves ordered SI-SDR by more than 13 dB over audio-only methods while achieving over 80\% speaker assignment accuracy.
#### Distributional Metrics for Evaluating Spoken Conversational Systems
 - **Authors:** Shree Harsha Bokkahalli Satish, Erica Cooper, Patrícia Schmidtová, Maike Züfle, Éva Székely, Nicholas Sanders, Ondřej Klejch
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Computation and Language (cs.CL); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2609.31719

 - **Pdf link:** https://arxiv.org/pdf/2609.31719

 - **Abstract**
 Evaluating conversational systems is a difficult and unresolved problem. We introduce the Conversational Distribution Score (CDS), which compares distributions of conversational behaviour using human conversations as a reference. CDS describes speech rate, syllabic rhythm, and turn interaction through eight interpretable features plus a separate two-feature semantic baseline. We compare conversations with two reference scales: one based on conversational success within human dialogue and another contrasting human and synthetic dialogue. Using listener judgments from out-of-domain goal--oriented dialogues, we examine system ranking, preferences between conversations, and ranking stability. Composite CDS recovers five of six listener system comparisons while individual features show strong correlation with listener preferences between conversations. We examine how many minutes and conversations are required before rankings stabilize. These findings support distributional comparisons as a complement to specific interactional metrics to evaluate conversations and conversational models while showing their interpretable value.
#### Acoustic domain shift in spoken language identification from systematic domain generalization evaluation to real-world application
 - **Authors:** Francois Derrida (X), Raphaël Duroselle (X), Thomas Courtat, Jean-François Bonastre (X)
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Machine Learning (cs.LG); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2609.31759

 - **Pdf link:** https://arxiv.org/pdf/2609.31759

 - **Abstract**
 Domain Generalization (DG) aims to develop models that remain robust to conditions unseen during training. While DG has been systematically studied in computer vision through controlled benchmarks and diverse distribution shifts, its evaluation in spoken language recognition remains less structured. Existing speech datasets provide valuable benchmarks for robustness to real-world acoustic conditions, but are primarily designed around specific scenarios and large scale rather than as general-purpose tools for systematically constructing and evaluating domain shifts. In this work, we introduce a smallscale speech dataset and evaluation protocol for controlled studies of acoustic domain shifts. It enables the evaluation of spoken language recognition models under a variety of acousitc domain shifts. We introduce the speech modality into the DomainBed Domain Generalization framework and evaluate three domain generalization algorithms. We show that in-domain performance is not a reliable predictor of cross-domain robustness and verify that explicit domain invariant algorithms such as MMD or DANN algorithms do not outperform ERM. We further validate the generality of these findings on MMS-LID-126, a state-of-the-art spoken language identification system. We release the code.
#### Optimal transport meets speech: a tutorial review
 - **Authors:** Xugang Lu, Yu Tsao
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Artificial Intelligence (cs.AI); Computation and Language (cs.CL); Machine Learning (cs.LG)
 - **Arxiv link:** https://arxiv.org/abs/2609.31787

 - **Pdf link:** https://arxiv.org/pdf/2609.31787

 - **Abstract**
 Optimal Transport (OT) provides a principled framework for comparing and transforming probability distributions while preserving geometric structure. Recently, OT has gained significant attention in machine learning due to its ability to measure discrepancies between distributions, even when their supports do not overlap, making it effective for tasks such as generative modeling, domain adaptation, and transfer learning. Despite its success in fields such as computer vision and natural language processing, OT remains relatively underexplored in speech research. Speech signals present unique challenges, including temporal dynamics, speaker variability, noise, reverberation, and heterogeneous multimodal representations involving audio, text, and visual information. These factors often lead to distribution mismatches, where OT offers a natural framework for alignment and interpretation. This work aims to promote broader adoption of OT in speech processing by: (1) reviewing OT foundations through intuitive physical interpretations and highlighting connections to modern generative models; (2) presenting computational algorithms suitable for deep learning frameworks; and (3) demonstrating OT applications in cross-domain and cross-modal speech tasks, including speech enhancement, automatic speech recognition, language and speaker recognition, and audio spoof detection. We highlight OT's strong potential for addressing distributional variations in real-world speech applications.
#### MAESTRO: a Multimodal Auditory-attention Egocentric Speech-TRacking Open corpus
 - **Authors:** K M Naimul Hassan, Ali Alavi, Donald S. Williamson
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Human-Computer Interaction (cs.HC); Machine Learning (cs.LG); Sound (cs.SD); Neurons and Cognition (q-bio.NC)
 - **Arxiv link:** https://arxiv.org/abs/2609.31898

 - **Pdf link:** https://arxiv.org/pdf/2609.31898

 - **Abstract**
 Humans rely on gaze, head movements, and visual cues to attend to speakers in noisy environments, yet auditory attention decoding (AAD) has been studied primarily using electroencephalography (EEG). We introduce the Multimodal Auditory-attention Egocentric Speech-TRacking Open (MAESTRO) corpus, the first AAD dataset to simultaneously record EEG, eye gaze, pupillometry, egocentric video, and head inertial measurement unit (IMU) data. MAESTRO includes four competing speakers and background noise across multiple signal-to-noise ratio (SNR) conditions, enabling attention decoding under realistic listening scenarios. Through a four-speaker attention decoding benchmark, we show that combining behavioral and physiological signals improves decoding performance over EEG-only approaches, enabling future advances in multimodal auditory attention decoding. These findings open the door to new applications, analyses, and methodological advances in multimodal AAD. The complete dataset is publicly available at this https URL . The official code repository is available at this https URL .
#### Improving Audiovisual Speech Recognition through Synthetic Visual Data Augmentation
 - **Authors:** Pol Buitrago, Pol Gàlvez, Javier Hernando
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Computation and Language (cs.CL); Sound (cs.SD); Image and Video Processing (eess.IV)
 - **Arxiv link:** https://arxiv.org/abs/2609.31961

 - **Pdf link:** https://arxiv.org/pdf/2609.31961

 - **Abstract**
 Audiovisual Speech Recognition (AVSR) is a multimodal approach to speech recognition that incorporates visual information from lip movements to enhance model performance. Despite its advantages, its development remains constrained by the limited availability of labeled audiovisual (AV) datasets. This work explores the use of synthetic visual data as a solution, using an audio-driven talking-head pipeline to generate lip-synchronized visual content from existing audio data. We evaluate the effectiveness of synthetic visual data both as an augmentation strategy and as a standalone training resource, applying our approach to Spanish and Catalan. Our results show that augmenting real AV data with synthetic samples yields relative Word Error Rate (WER) reductions of up to 16.2%, demonstrating the potential of this approach. Moreover, we demonstrate that synthetic data alone can serve as a baseline for AVSR training in languages lacking AV datasets. These findings provide evidence that synthetic visual data can serve as a scalable solution to AVSR data scarcity, enabling broader language coverage.
#### Audio Preprocessing Effects on Stuttering Detection: A Class-Specific Analysis
 - **Authors:** Anisha Pattanayak, Hanie Kang, Huang-Cheng Chou, Sudarsana Reddy Kadiri
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2609.32285

 - **Pdf link:** https://arxiv.org/pdf/2609.32285

 - **Abstract**
 Audio preprocessing can affect how well a system detects stuttering. We study a simulated chain of denoising,loudness normalisation, Opus coding, and voice activity detection on SEP-28k. We use frozen WavLM Base+ features and report pointwise confidence intervals from episode-level bootstrap resampling. At a fixed threshold of 0.5, the chain reduces block F1 from 0.638 to 0.465, with smaller decreases for the other four classes. ROC-AUC decreases for all five classes. Blocks show the largest F1 and ROC-AUC losses, while sound repetitions show the largest average precision loss. Tuning the threshold on processed validation audio raises block F1 to 0.630. Retraining on processed audio with threshold tuning gives 0.628. Threshold adjustment therefore accounts for most of the observed block F1 recovery. It does not change ROC-AUC, which retraining raises only from 0.620 to 0.633, compared with 0.724 on clean audio.
#### Toward Human-Aligned Judgement of Speech Emotion Similarity
 - **Authors:** Yun-Shao Tsai, Yi-Cheng Lin, Chih-Kai Yang, Ho-Jung Cheng, Tsun-Yi Chang, Sheng-Wei Wu, Yi-Shan Chen, Hsiang-Chun Chang, Liang-Chieh Lee, Hung-yi Lee
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2609.32504

 - **Pdf link:** https://arxiv.org/pdf/2609.32504

 - **Abstract**
 Evaluating emotion preservation in expressive speech generation involves assessing how closely generated speech matches a reference in emotion. Human listening tests assess this similarity, but their cost motivates automatic measures aligned with human judgments. To support the development and evaluation of such measures, we introduce SES-Bench, a speech emotion similarity benchmark built from human comparisons of two candidate utterances against a shared reference. These comparisons record which candidate listeners find emotionally closer to the reference and the strength of their preference. Using these annotations, we train SES-Judge to score emotion similarity between two utterances. SES-Judge significantly outperforms embedding cosine similarity and prompted large audio-language models in preference accuracy and correlation with human ratings that capture both preference direction and strength.
#### VoxMem: Benchmarking Multimodal Memory in Large Audio Language Models
 - **Authors:** Yang Xiao, Vidhyasaharan Sethu, Eun-Jung Holden, Ting Dang
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2609.32607

 - **Pdf link:** https://arxiv.org/pdf/2609.32607

 - **Abstract**
 Spoken conversational systems must recover information from prior interactions (i.e., memory), yet relevant information in speech extends beyond what was said to who said it, how it was spoken, and what was audible, information that exists only in the audio signal and cannot be recovered from a transcript. Beyond what to remember, memory also demands diverse operations: retrieving a single fact, integrating evidence across turns, tracking an evolving state. Real interactions further unfold across sessions, meaning information accumulates across distinct episodes rather than a single continuous recording. Existing benchmarks fall short on all three dimensions: they focus primarily on lexical content, adopt limited and ad hoc memory operations, and treat memory as a single-session problem. We argue that principled memory evaluation requires jointly characterizing the acoustic evidence to be retained and the operations applied to it, and introduce a taxonomy along these two axes. Building on this taxonomy, we present VoxMem: 3,196 evaluation instances over 34,743 spoken sessions (177 hours) crossing four acoustic evidence types (speech semantics, speaker identity, paralinguistic cues, environmental sound) with four memory operations (information extraction, multi-session reasoning, temporal tracking, and answer refusal), grounded in multi-session histories and stratified across context budgets from 8K to 64K tokens. Evaluating 15 LALMs, no model exceeds 40% at 32K. Models retain what was said far better than who said it, how, or what was audible, a gap that widens for complex operations, grows with history length, and manifests as qualitatively distinct failure modes across evidence types. VoxMem aims to provide a foundation to measure and drive progress on the full scope of spoken conversational memory.
#### WhisperVC-AV: Audio-Visual Content Restoration for Noise-Robust Whisper-to-Normal Voice Conversion
 - **Authors:** Ziyue Yin, Dong Liu, Ming Li
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2609.32843

 - **Pdf link:** https://arxiv.org/pdf/2609.32843

 - **Abstract**
 Environmental noise obscures the content cues needed for whisper-to-normal conversion. We propose WhisperVC-AV, which restores acoustic content features while leaving the original WhisperVC conversion module unchanged. Its context-guided restoration module combines temporal acoustic context with synchronized lip features through attention and gated residual correction. Experiments on AISHELL6-Whisper show lower character error rates (CERs) than WhisperVC across three ASR systems, on clean speech and under all six signal-to-noise ratio (SNR) conditions with MUSAN noise. The largest gains occur at 0 dB SNR, where Qwen3-ASR CER falls from 36.29% to 28.88%. WhisperVC-AV also improves predicted speech quality while maintaining speaker similarity. The CER gains extend to unseen background noise without retraining, while visual controls support the use of utterance-specific lip cues. Audio examples are available on our demo page.
#### Acoustic Progress Propagation for Long-Horizon Speculative Decoding in ASR
 - **Authors:** Yuanyuan Jia, Qianqian Yang
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Computation and Language (cs.CL)
 - **Arxiv link:** https://arxiv.org/abs/2609.33245

 - **Pdf link:** https://arxiv.org/pdf/2609.33245

 - **Abstract**
 Speculative decoding accelerates autoregressive automatic speech recognition (ASR), but the acceptance length of alignment-aware drafters can saturate as the draft horizon increases. We propose a progress-aware speculative drafter that recurrently propagates an acoustic progress state across draft steps and feeds it back into audio cross-attention to guide token generation. We jointly train the drafter and progress predictor over variable draft horizons. On five ASR test sets, our method achieves lossless, macro-averaged end-to-end speedups of 1.657x and 1.227x over target-only autoregressive decoding with Qwen3-ASR-0.6B and Qwen3-ASR-1.7B, respectively. Relative to AnchorDraft, our method improves the macro-averaged speedup by 34.3% and 9.0%, respectively. Horizon sweeps show continued growth in acceptance length beyond the baselines' saturation. Code is available at this https URL.
#### From Script to Drama: An Agentic Framework for Controllable Multi-Speaker Dialogue TTS
 - **Authors:** Kangxiang Xia, Xinfa Zhu, HangRui Hu, Kexin Huang, Wenjie Tian, Ziyue Jiang, Bingshen Mu, Jingbin Hu, Ting He, Lei Xie, Jin Xu
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Computation and Language (cs.CL); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2609.33362

 - **Pdf link:** https://arxiv.org/pdf/2609.33362

 - **Abstract**
 Multi-speaker dialogue TTS requires natural speech generation, consistent speaker identity, coherent cross-turn transitions, and fine-grained control of expressive attributes such as emotion, speaking rate, and loudness. These requirements are difficult to satisfy reliably with one-shot generation, especially in long-form dialogue. We propose a controllable multi-speaker dialogue TTS framework that formulates synthesis as critique-driven iterative refinement. Its speech backbone, ControlEdit-TTS, unifies instruction-following synthesis and natural-language-guided attribute editing, enabling correction of expressive errors without full regeneration. The framework further performs hierarchical utterance-level and scene-level critique, routing detected issues to editing, resynthesis, or timing adjustment. Experiments on a bilingual Chinese--English dialogue benchmark show improved utterance-level instruction following, better dialogue-level preference than direct dialogue models and agentic baselines, and more effective refinement than regeneration-only alternatives while preserving speaker identity. Ablations further confirm the benefits of scene-level critique and edit-based correction.
#### Pruned CTC for Memory-Efficient Large-Vocabulary ASR Training
 - **Authors:** Yifan Yang, Xiaoyu Yang, Zengrui Jin, Xian Shi, Yuxuan Wang, Yu Xi, Ziyang Ma, Qi Chen, Ruiyang Xu, Hui Wang, Dongchao Yang, Jin Xu, Xie Chen
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2609.33645

 - **Pdf link:** https://arxiv.org/pdf/2609.33645

 - **Abstract**
 Connectionist temporal classification (CTC) naturally supports offline and streaming speech recognition with utterance-level supervision, but conventional implementations materialize frame-by-vocabulary activations in memory, making CTC training with native LLM vocabularies prohibitively memory-intensive. A key observation is that every valid CTC alignment uses only target tokens and blank, and their union across a batch typically forms a small subset of the full vocabulary. We introduce Pruned CTC, which restricts alignment computation to this subset while retaining full-vocabulary normalization. We prove that this vocabulary reduction is exactly equivalent to full-vocabulary CTC in loss and gradients. Head-and-loss activation memory no longer scales linearly with vocabulary size. We further apply finite-beam alignment pruning. Building on Pruned CTC, we develop LLM-CTC, which adapts pretrained LLMs for non-autoregressive ASR while retaining causal attention and native vocabularies, and extend it to bounded-history streaming, avoiding chunk-level speech--text alignments. Experiments show that, with Zipformer-M encoder and 180K vocabulary, Pruned CTC reduces full-step memory by 5.1$\times$ with only 17% step-time overhead. Across three corpora, it matches standard CTC accuracy. On GigaSpeech, across six Qwen3 model sizes from 0.6B to 32B, LLM-CTC remains within 7% relative WER of LLM-CE with 7 to 10$\times$ faster recognition; when fine-tuning Qwen3-ASR for bounded-history streaming, LLM-CTC remains within 3% relative WER of matched offline models on the test set. Together, these results establish Pruned CTC as a scalable sequence objective for native-vocabulary LLM ASR across offline and streaming settings.
#### DGS-MLDG: Domain Gradient Surgery Guided Meta-Learning for Domain Generalization in Speech Deepfake Detection
 - **Authors:** Siqing Qin, Kong Aik Lee, Youzhi Tu, Eng Siong Chng, Man-Wai Mak
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2609.33706

 - **Pdf link:** https://arxiv.org/pdf/2609.33706

 - **Abstract**
 Speech deepfake detection faces significant challenges due to domain shifts. Domain generalization (DG), particularly meta-learning for domain generalization (MLDG), offers a promising solution by simulating and mitigating domain shifts. However, MLDG is often hindered by conflicting gradients between its meta-train and meta-test objectives, leading to suboptimal performance. To address this problem, we propose domain gradient surgery (DGS), a meta-learning method that resolves conflicts through an asymmetric projection strategy. DGS removes the destructive component from the meta-test gradient, ensuring a conflict-free optimization trajectory versus the meta-train gradient. Furthermore, we introduce layer-wise DGS (LW-DGS), an efficient variant of DGS that dynamically identifies and intervenes only conflict-prone layers. Extensive experiments on challenging benchmarks demonstrate that DGS-MLDG and LW-DGS-MLDG achieve an average relative EER reduction of 5.29% and 4.04%, respectively.
#### Domain-Adaptive Dual-Gating Mixture of Experts for Generalizable Speech Deepfake Detection
 - **Authors:** Siqing Qin, Zhe Li, Kong Aik Lee, Man-Wai Mak
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2609.33709

 - **Pdf link:** https://arxiv.org/pdf/2609.33709

 - **Abstract**
 Recent advances in speech deepfake detection (SDD) have leveraged the Mixture of Experts (MoE) to enhance generalization capacity. However, existing gating networks often overlook the acoustic and temporal cues of deepfakes. In this work, we propose a novel domain-adaptive dual-gating MoE (DADGMoE) framework for SDD under unseen attack types and acoustic conditions. Our innovative dual-gating mechanism leverages Sinc-layer-based filters to process both low-level acoustic signals (raw waveforms) and high-level speech representations from a large self-supervised learning (SSL) model. It further incorporates domain prototypes to guide expert routing based on implicit deepfake patterns. The lightweight affine experts process the routed inputs. Experiments show that our DADGMoE significantly outperforms the baseline, achieving up to a 40.8% relative EER reduction on challenging out-of-dataset benchmarks. This framework demonstrates superior generalization capabilities and efficient design.
#### Rethinking Automated Voice Similarity by Shifting from EER to Embedding Geometry
 - **Authors:** Szu-Chi Chen, Jia-Kai Dong, Yi-Cheng Lin, Sung-Feng Huang, Hung-yi Lee
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2609.33999

 - **Pdf link:** https://arxiv.org/pdf/2609.33999

 - **Abstract**
 Speaker verification (SV) models are commonly assumed to better capture nuances among speaker characteristics as verification accuracy improves, leading to their widespread use as automated proxies for human voice similarity in speech generation tasks. However, by establishing a human perceptual alignment metric and conducting systematic analysis, we demonstrate that perceptual alignment is governed far more by how a model is trained (its learning objective) than by how well it performs (EER). Notably, standard margin-based classification losses (e.g., AAM-Softmax) yield substantially lower perceptual alignment than prototypical metric losses, while EER itself fails to track human judgment, directly challenging the community's implicit assumption. We trace this divergence to embedding geometry, where a model's effective dimensionality ($d_{\mathrm{eff}}$) tracks perceptual alignment with a $-0.95$ rank correlation, revealing that the dimensional spread favored by classification losses fundamentally clashes with the low-dimensional nature of human voice perception. Imposing a dimensionality bottleneck compresses $d_{\mathrm{eff}}$ and raises perceptual alignment ($\rho_{\mathrm{align}}$) from 0.08 to 0.74, establishing a principled geometric criterion for evaluating voice similarity.
#### SPEAR-Gen: Generation-Aware Pre-training for Unified Speech Representations
 - **Authors:** Xiaoyu Yang, Arthur Hinsvark, Antonios Alexos, Osama Hanna, Philip C. Woodland, Yiting Lu
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2609.34147

 - **Pdf link:** https://arxiv.org/pdf/2609.34147

 - **Abstract**
 Speech understanding and generation place different demands on speech representations, and existing models are typically optimised towards one capability or the other. To reduce this gap, we introduce SPEAR-Gen, a speech representation model that learns a single representation for both capabilities. Task-aligned feature aggregation consolidates complementary linguistic and paralinguistic information across a frozen encoder into discrete targets for masked prediction, while a coarse-to-fine objective combines log-Mel reconstruction with residual flow matching to preserve spectral structure and fine-grained acoustic variation. Experiments on SUPERB and speech resynthesis show that SPEAR-Gen maintains strong understanding performance while substantially improving resynthesis quality and speaker preservation. These results demonstrate that a single speech representation can effectively support both understanding and generation.
#### Explainable and Generalisable LLM-based Cognitive Decline Detection with Spontaneous Speech
 - **Authors:** Ziyun Cui, Wen Wu, Chuan Shi, Shuguang Yang, Xueying Gui, Yan Zheng, Qiong Yang, Haiyan Zhao, Wei-Qiang Zhang, Ji Wu, Yelei Li, Nan Li, Chao Zhang
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Computation and Language (cs.CL)
 - **Arxiv link:** https://arxiv.org/abs/2609.34217

 - **Pdf link:** https://arxiv.org/pdf/2609.34217

 - **Abstract**
 Alzheimer's disease (AD) and mild cognitive impairment (MCI), which may precede AD, manifest early through subtle linguistic and acoustic alterations. Traditional diagnostics, however, are often resource-intensive and lack scalability for mass screening. To address these challenges, we introduce a novel bilingual speech large language model framework for automated, explainable cognitive screening. Unlike conventional pipelines that rely on error-prone automatic speech recognition, our system directly processes raw speech to learn joint acoustic-semantic representations, preserving critical prosodic cues often lost in transcription. Utilising our newly collected PUTH-AD dataset alongside multiple open-source corpora, we implemented a multi-task learning objective that simultaneously performs cognitive status classification and generates clinician-understandable natural language explanations. Our system achieved the highest average accuracy and AUROC across six dataset/task conditions, comparing three representative baselines. The system demonstrated cross-task transfer to held-out PUTH-AD task subsets, maintaining classification accuracy on an entirely unseen cognitive task without task-specific fine-tuning. Furthermore, clinician evaluation confirms that the generated explanations are both clinically relevant and largely consistent with the underlying speech evidence, supporting their potential utility in clinical interpretation. This study provides a scalable, objective, and explainable framework for speech-based cognitive screening, combining cognitive status classification with natural language explanations that clinicians can assess and verify, bridging the gap between advanced AI and clinical utility.
#### Audio Tokens as a Budgeted Resource: Marginal-Utility Allocation for Scalable Audio Representations
 - **Authors:** Mingyu Zhao, Jinchao Zhang, Zhiyong Wu
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2609.34337

 - **Pdf link:** https://arxiv.org/pdf/2609.34337

 - **Abstract**
 Discrete audio tokens are widely used as a representation interface, yet fixed-depth RVQ tokenizers allocate equal capacity to every frame despite varying refinement value. We introduce UniAdapt, which learns marginal utility of RVQ refinements on a frozen codec and allocates them under exact serialized-bit budgets. A rate-independent causal controller predicts acoustic utility, while an optional semantic head supports speech-only utterance-level allocation; measured acoustic and semantic marginal gains on speech have a correlation of 0.42. For causal allocation, a primal-dual allocator selects prefix-valid depths, while an exact guard constrains each sequence prefix to its matched fixed-depth serialized budget. Under utterance-level allocation, UniAdapt reduces Log-STFT distortion by 1.07-4.39 percent across speech, music, and environmental audio without larger budgets. Causally, it improves three of four speech rates with zero violations across 800 utterance-rate evaluations and runs faster than real time. A 20-listener utterance-level MUSHRA study shows a significant 3.52-point speech improvement, with no significant differences on music or environmental audio. These results support separating utility prediction from budget enforcement for scalable, budget-conditioned audio representations.
#### CharDuplex: Building Character-Consistent Full-Duplex Spoken Dialogue Models
 - **Authors:** Donghang Wu, Yisi Liu, Chen Chen, Hexin Liu, Eng Siong Chng
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2609.34461

 - **Pdf link:** https://arxiv.org/pdf/2609.34461

 - **Abstract**
 Full-duplex speech models are moving voice interaction beyond conventional turn-taking, yet natural conversation is shaped not only by when an agent speaks, but also by how it behaves as a conversational character. We present CharDuplex, a character-driven full-duplex speech model that combines real-time spoken interaction with persona-conditioned behavior. We first adapt GLM-4-Voice to an always-on dual-stream architecture and train the model for full-duplex conversation. Then a fully automated pipeline constructs character-conditioned dialogue data from open-source character descriptions for character-conditioned supervised fine-tuning. The model is further refined with the proposed FDGym, where an LLM-simulated user dynamically interacts with the model, enabling reinforcement learning over evolving multi-turn interactions. On SpeechRole-Eval, CharDuplex achieves the highest average score among the evaluated open-source models, while remaining competitive with closed-source systems. It also demonstrates competitive general speech intelligence and strong full-duplex interaction capabilities. CharDuplex demonstrates a practical training recipe for building full-duplex voice assistants that are not only interactive, but also character-consistent.
#### Measurement-Based Bitrate-Energy-Quality Analysis of Neural Audio Codec Decoders on Laptop and Phone Platforms
 - **Authors:** Seunghyeon Shin, Seokjin Lee
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Signal Processing (eess.SP)
 - **Arxiv link:** https://arxiv.org/abs/2609.34524

 - **Pdf link:** https://arxiv.org/pdf/2609.34524

 - **Abstract**
 Neural audio codecs can achieve similar objective quality at lower bitrates than conventional codecs, but their decoder-side computational cost may offset this bitrate advantage on battery-powered client devices. This paper presents a measurement-based rate--energy--quality analysis of four neural audio codecs--EnCodec, DAC, HILCodec, and SNAC--and two conventional baselines, AAC-LC and Opus, on laptop and phone platforms. Speech and music are evaluated separately using the original-reference ViSQOL protocol. For the main comparison, operating points are matched by selecting the measured point nearest to the midpoint of the common overlap in treatment-level mean ViSQOL. A secondary analysis includes only explicitly evaluated bitrate settings. Decoder energy is measured as idle-subtracted energy per second of audio (J/s), and results are summarized as the median of three repeated runs for execution-valid runtime and device paths. Pairwise break-even transmission-energy thresholds are then derived analytically from the measured decoder energy and bitrate. At the matched-quality operating points, the evaluated neural codecs achieved similar ViSQOL scores at lower bitrates but generally required more decoder-side energy than the applicable conventional codecs. EnCodec produced the lowest neural break-even thresholds in both matched-quality cohorts on both platforms. By contrast, several DAC and SNAC comparisons on the Phone XNNPACK CPU path exceeded 200 mJ/kbit, and their full-band Phone configurations required more than 1 s to decode 1 s of audio. These results show that lower bitrate alone does not guarantee an energy benefit: deployment efficiency also depends on decoder complexity, the effective runtime and device mapping, execution validity, accelerator availability, and the transmission-energy coefficient.
#### Domain-Incremental Learning for Generative Speech Enhancement
 - **Authors:** Manjunath Mulimani, Annamaria Mesaros, Minje Kim, Jesper Rindom Jensen
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2609.34901

 - **Pdf link:** https://arxiv.org/pdf/2609.34901

 - **Abstract**
 We propose a domain-incremental learning framework for generative speech enhancement (SE) that learns from a sequence of datasets or domains recorded under diverse acoustic conditions. Fine-tuning a pretrained model on continuously evolving domains leads to catastrophic forgetting of previously acquired knowledge, while zero-shot generalization often fails to adequately adapt to unseen domains. To address these challenges, we first develop a novel language model-based generative SE model that we then use as a pretrained backbone and incrementally adapt it to acoustically mismatched domains using lightweight domain-specific Low-Rank Adaptation. The proposed framework enables the model to acquire enhancement capabilities for new domains while preserving performance on previously learned domains. Evaluated on four heterogeneous speech datasets, our approach effectively adapts to new domains without forgetting previously learned domains.
#### Perceptual Quality Loss or Loss of Perceptual Quality?
 - **Authors:** Danilo de Oliveira, Tal Peer, Maurício do V. M. da Costa, Timo Gerkmann
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Machine Learning (cs.LG)
 - **Arxiv link:** https://arxiv.org/abs/2609.35054

 - **Pdf link:** https://arxiv.org/pdf/2609.35054

 - **Abstract**
 Contemporary deep speech enhancement (SE) models are often trained with specific auxiliary terms in the loss function as a way to improve their performance in terms of perceptual metrics. Nevertheless, a higher score on a perceptual metric does not necessarily correlate with an improved listening experience. Through objective and subjective experiments, we assess the performance of SE models trained with two different types of auxiliary PESQ loss terms. The numerical evaluation on a suite of standard metrics suggests that, while models optimized for PESQ naturally obtain higher PESQ scores in the test set, for most other metrics the scores do not significantly change. In some cases, the PESQ loss even results in worse PESQ scores on mismatched data. A formal listening experiment reveals that the models without a PESQ loss were generally preferred over models that include it, across all settings. Finally, we analyze the relative importance of PESQ in the composite metrics CSIG, CBAK and COVL, and find that PESQ dominates all of them. Our study highlights the perils of over-reliance on PESQ and stresses the importance of a complete evaluation procedure for SE.
#### Open-Qwen-Music: An Auditable Framework for LLM-Based Music Composition and Diffusion Rendering
 - **Authors:** Yangbin Yu, Mingyu Yang
 - **Subjects:** Subjects:
Sound (cs.SD); Artificial Intelligence (cs.AI); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2609.31652

 - **Pdf link:** https://arxiv.org/pdf/2609.31652

 - **Abstract**
 We present Open-Qwen-Music, an open reconstruction of Qwen-Music and a fully specified research system for text-to-music generation that couples LLM-based semantic composition with diffusion-based acoustic rendering. The system comprises a 25 Hz single-codebook music tokenizer, a 3B-parameter autoregressive Music LLM, and a diffusion renderer producing 48 kHz stereo audio, following the cross-module interfaces reported by Qwen-Music. The strongest systems of this design remain closed, and prominent open music-generation projects release weights and inference code without their training corpora or end-to-end training implementations. This limits independent and controlled study of how information loss and prediction errors propagate from semantic representation through autoregressive planning to acoustic rendering. To our knowledge, Open-Qwen-Music is the first fully open release of an LLM-composition-plus-diffusion-rendering text-to-music system. Beyond model weights and inference code, the release includes the training datasets and provenance manifests, complete data-processing, annotation, training, inference, and evaluation pipelines, configurations, and pretrained weights for every learned module. Artifact manifests bind the identities of these artifacts across the complete workflow. Together, these artifacts establish a reproducible implementation of the modular architecture and provide an empirical basis for component-level analysis and future evaluation. We present the system as a transparent, executable research baseline and a starting point for the community, not as evidence of quality parity with Qwen-Music. Open-Qwen-Music is an ongoing effort, and we will continue to improve its generation quality, controllability, and robustness. All release artifacts are available at this https URL.
#### Normalise or condition? Noise-floor front-ends for on-board keyword spotting under UAV rotor ego-noise
 - **Authors:** Yida Lin, Bing Xue, Mengjie Zhang, Sam Schofield, Richard Green
 - **Subjects:** Subjects:
Sound (cs.SD); Computer Vision and Pattern Recognition (cs.CV); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2609.31699

 - **Pdf link:** https://arxiv.org/pdf/2609.31699

 - **Abstract**
 A microphone on the airframe of a small multi-rotor UAV is dominated by rotor ego-noise, so spoken flight commands arrive at negative signal-to-noise ratio (SNR). We study small-footprint keyword spotting (KWS) for a ten-word command vocabulary under real ego-noise, training on one quadrotor and testing on another. Besides per-clip accuracy we measure the streaming false-alarm rate on 4.4 h of continuous rotor noise. We compare classical noise-robust front-ends (CMN, PCEN, spectral subtraction), test-time adaptation, and two front-ends that track the per-band ego-noise floor over the two seconds preceding the decision window and either subtract it (normalisation) or feed it to the network as a second input channel (conditioning). Per clip, all front-ends look alike: +2-3 points on average, up to +14 at -15 dB. On continuous rotor noise they differ sharply. Normalising front-ends fire about ten times more often than plain log-mel at the same threshold and end up below it at a budget of one false alarm per hour (-9 points at 0 dB). Conditioning keeps the baseline's false-alarm rate and turns its gain into detections (+8 points at -10 dB over three seeds); on top of PCEN it gives the best per-clip accuracy and false-alarm rate, and a level-anchored variant is also invariant to the microphone gain. Real drone+interferer recordings expose the remaining failure mode, environmental sounds and bystander speech, which training negatives halve. A single script reproduces all on-device numbers on the NVIDIA Jetson Orin NX flight computer, where the complete pipeline costs 3 ms per 100 ms hop on the CPU.
#### NVAlign: Direct-Gradient Optimization for Non-Verbal Control in Continuous Autoregressive Flow Matching Text-to-Speech
 - **Authors:** Qiaolin Wang, Pedro Sandoval-Segura, Anunaya Joshi, Edvardas Jurkonis, Jake Downie
 - **Subjects:** Subjects:
Sound (cs.SD); Artificial Intelligence (cs.AI); Computation and Language (cs.CL); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2609.31892

 - **Pdf link:** https://arxiv.org/pdf/2609.31892

 - **Abstract**
 While modern text-to-speech (TTS) systems generate highly natural speech and support inline non-verbal vocalization (NVV) tags, accurate control over these events remains challenging. A key gap is the lack of established post-training methods for non-verbal control in continuous autoregressive flow-matching TTS. To this end, we present NVAlign, a direct-gradient post-training framework for NVV tag-following in this architecture. We first perform supervised fine-tuning (SFT) of TTS models and an NVV-aware automatic speech recognition (NV-ASR) model on NVV-annotated speech, then freeze the NV-ASR model to serve as the reward model for post-training. A two-step gradient surrogate enables efficient reward backpropagation through the flow-matching sampler to jointly update the autoregressive backbone and acoustic flow head. Fidelity penalties and reference-velocity regularization help preserve speaker similarity and speech quality. Results from NVV-SuperBench and human listening evaluations show that NVAlign improves tag-following accuracy over SFT and Flow-GRPO baselines. These findings demonstrate that direct reward-gradient optimization can improve non-verbal control in continuous autoregressive flow-matching TTS. Audio samples are available at this https URL.
#### Duplex-MPE: Benchmarking Multi-Party Interaction in Full-Duplex Dialogue
 - **Authors:** Chengqian Ma, Wenhao Feng, Weixuan Jin, Gaole Dai, Tianyu Xie, Yuexiao Ma, Zhaolu Kang, Xiangyu Zhao, Xiawu Zheng, Fei Chao
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2609.31948

 - **Pdf link:** https://arxiv.org/pdf/2609.31948

 - **Abstract**
 Real-time full-duplex speech models can listen while speaking, enabling natural interaction without rigid turn boundaries. Existing benchmarks evaluate turn-taking, interruption handling and multi-round dialogue, but largely centre on a designated user rather than an assistant participating in a shared conversation among several people. We introduce Duplex-MPE to evaluate when such an assistant should answer, remain silent or stop speaking. The benchmark contains 2,000 scenarios with three or four human speakers and one assistant, each paired across explicit and implicit addressing of the same request. Models receive continuous conversation audio without transcripts or supplied turn boundaries. Four scores measure fresh response initiation, answer accuracy, silence preservation and stopping when a human resolves a request. We evaluate five open-weight speech systems: MiniCPM-o 4.5, Moshi, FLM-Audio, Voila and Freeze-Omni. MiniCPM-o 4.5 leads on three scored capabilities, while frequent speech from other systems can coexist with inaccurate answers or failures to remain silent. A transcript-based Gemini 3.1 Pro reference responds 64.3 percentage points more often to explicit than implicit requests; paired tests detect no significant response-rate difference for the speech systems.
#### VoiceNet: Fine-Grained Voice Understanding Beyond Emotion at Scale
 - **Authors:** Christoph Schuhmann, Robert Kaczmarczyk, Gollam Rabby, Felix Friedrich, Maurice Kraus, Gijs Wijngaard, Kourosh Nadi, Huu Nguyen, Kristian Kersting, Sören Auer
 - **Subjects:** Subjects:
Sound (cs.SD); Artificial Intelligence (cs.AI); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2609.32016

 - **Pdf link:** https://arxiv.org/pdf/2609.32016

 - **Abstract**
 Expressive speech synthesis has outpaced expressive speech perception: systems now render fine-grained vocal performances that no public benchmark can score. Most benchmarks for this inverse problem stop at six to nine basic emotion categories, largely on acted speech. This paper introduces VoiceNet, a human-annotated representation-level benchmark for voice performance understanding on permissively-licensed in-the-wild speech. VoiceNet has two subsets: VoiceNet-Emo applies a 40-emotion taxonomy with three expert ratings per item, and VoiceNet-Ext, a preliminary subset, scores 57 talking-style attributes including speaking rate, vocal tension, breathiness, and register. The paper also releases Emolia, an emotion-annotated version of the Emilia corpus, with a curated rebalanced subset enriched by dense MOSS-Audio Thinking annotations. Two voice-text contrastive baselines train on this data: a 110M-parameter VoiceCLAP-Small for fast large-scale data filtering and a 7B VoiceCLAP-Large for state-of-the-art performance. Both outperform existing CLAP baselines, which sit near chance on VoiceNet-Emo. On VoiceNet-Emo, VoiceCLAP-Large aligns more closely with the aggregate expert consensus than individual experts agree with one another: a comparison against the majority label rather than evidence of surpassing human emotion perception. All systems evaluated here are voice-text embedding models: VoiceNet scores representation-level attribute recognition and retrieval, not end-to-end spoken-dialogue behaviour. Clustering and filtering uncurated speech corpora into subsets that span diverse talking styles and emotions remains an open challenge; VoiceCLAP embeddings offer a promising tool for this task. VoiceNet, Emolia, and VoiceCLAP are publicly available for research use.
#### Tracing Decoder Artifacts for Compact Synthetic Speech Screening
 - **Authors:** Yi Chen Liu, Jian Liu
 - **Subjects:** Subjects:
Sound (cs.SD); Artificial Intelligence (cs.AI); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2609.32050

 - **Pdf link:** https://arxiv.org/pdf/2609.32050

 - **Abstract**
 Recent advances in speech synthesis and voice cloning have increased the need for reliable synthetic-speech detection, yet high-accuracy detectors increasingly rely on large pretrained models that are costly to invoke on every recording. Rather than replacing such detectors, we investigate a compact front-end screen that processes all inputs cheaply and forwards only suspicious recordings for more expensive analysis. To enable lightweight screening without a large learned encoder, we exploit spectral traces introduced by speech-generation operations. We analyze how learned upsampling and inverse short-time Fourier transform synthesis can produce predictable spectral artifacts and measure their presence directly in generated waveforms. Because the strength of these artifacts varies across generators, we combine decoder-guided spectral measurements with complementary descriptors of short-time spectral shape and temporal variation in a compact gradient-boosted tree. Across seven speech generators and two human-speech sources, the proposed screen achieves an equal error rate of 0.021\% with an estimated model storage of 151 KiB. When used as the first stage of a simulated cascade with a 1.15-billion-parameter detector, it reduces estimated detection energy by 84.4\% while operating at a 0.050\% synthetic-speech miss rate, demonstrating the potential of decoder-guided acoustic evidence for low-cost front-end screening.
#### Do Audio LLMs Listen Before They Act? Diagnosing Acoustic-Context Gating in Voice Agents
 - **Authors:** Yanjie Zhang, Nanchen Hu, Yushi Sun
 - **Subjects:** Subjects:
Sound (cs.SD); Artificial Intelligence (cs.AI); Computation and Language (cs.CL); Multimedia (cs.MM); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2609.32536

 - **Pdf link:** https://arxiv.org/pdf/2609.32536

 - **Abstract**
 Audio language models can recognize spoken commands and invoke tools, but an agent must first decide whether the acoustic and conversational context warrants action. We introduce VGBench, a 1,018-item diagnostic benchmark for action-level addressedness across side-talk, self-talk, and speaker-switch scenarios. Each item uses a shared action space comprising silence, a tool call, and a natural-language answer. Speaker-switch pairs hold the specified words fixed while source, distance rendering, and a temporal boundary define a controlled wearer-to-bystander shift. Six raw Audio LLMs and three training-free adaptations often identify the target tool yet rarely withhold action under this shift; the highest raw switch mute rate is 14%. We then use VoxGate as a post-training case study. Supervised training mutes 91.3% of switched commands while choosing the correct tool for all nearby wearer commands and text-only controls. An exploratory GRPO stage has similar switch performance; side-talk accuracy rises from 68.4% to 70.9%, and self-talk muting from 52.0% to 60.0%. Factorized controls identify an independent source-change effect, while sensitivity to the far-field manipulation varies across acoustic renderings. The benchmark therefore measures multi-cue acoustic-context gating rather than isolated speaker identity.
#### DEFINE: Exemplar-Guided Accent Control for Zero-Shot TTS
 - **Authors:** Ambuj Mehrish, Abhinaba Roy, Alex Ivanov, Tawsif Ahmed, Dorien Herremans
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2609.32777

 - **Pdf link:** https://arxiv.org/pdf/2609.32777

 - **Abstract**
 Zero-shot text-to-speech (TTS) can reproduce an unseen speaker from a short reference recording, but typically entangles speaker identity and accent within the same reference. We introduce DEFINE, an end-to-end framework that decouples these factors by conditioning speaker identity and target accent on separate audio exemplars. A single inference-time guidance weight continuously controls accent strength without retraining. Built on F5-TTS with parameter-efficient LoRA adaptation, DEFINE maps short accent exemplars into a conditioning space using an exemplar encoder supervised through learned accent prototypes, requiring neither accent labels at inference time nor post-synthesis waveform conversion. On seen accents, increasing accent guidance improves accent-probe accuracy from 6.5% to 19.6%. More importantly, a single DEFINE model generalizes accent control beyond its training accent set: on seen and out-of-domain accents, though not on held-out accents, it matches the accent transfer performance of a two-model TTS-voice-conversion cascade while achieving higher speaker similarity and comparable predicted speech quality. These results demonstrate that speaker identity and accent can be independently controlled from audio exemplars within a single zero-shot TTS model, including for accents unseen during training.
#### Finding Emotions Where They Belong: Rethinking Audio Emotion Recognition through Masked Temporal Affective Grounding
 - **Authors:** Abdelrahman Mohamed, Lars Kai Hansen, Zheng-Hua Tan
 - **Subjects:** Subjects:
Sound (cs.SD); Artificial Intelligence (cs.AI); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2609.32804

 - **Pdf link:** https://arxiv.org/pdf/2609.32804

 - **Abstract**
 Audio emotion recognition (AER) typically assigns a single label to an entire recording, leaving the temporal scope of that label ambiguous when multiple speakers and affective events are present. We address this limitation by reformulating AER as a Temporal Affective Grounding (TAG) task that associates emotions with temporally bounded speech spans and vocal tone descriptions. To support this formulation, we curate temporally annotated versions of existing emotion recognition datasets and construct recordings containing two to four affective speech spans, including overlapping speech. Training in this longer format with a standard language-modeling objective can degrade both emotion recognition and temporal grounding performance, while tone descriptions can provide shortcuts for emotion prediction. To address these challenges, we introduce Masked Temporal Affective Grounding (M-TAG), a supervised training objective that combines full-sequence language modeling with emotion and timestamp cross-entropy losses under attention masking. The masking varies the context visible to emotion-label tokens to reduce reliance on shortcuts and improve generalization, while the timestamp loss incorporates a distance-aware weight to penalize larger temporal errors. We evaluate EMO-TAG, a model fine-tuned using our dataset and objective, on emotion recognition and affective temporal-grounding against three AER and audio-language baselines: Flamingo-Next, Audio-Reasoner, and AffectGPT. Our results show that existing models achieve limited affective temporal-grounding despite competitive emotion recognition performance.
#### Whisper-Flash: Acoustically Conditioned Parallel Drafting for Faster Whisper Decoding
 - **Authors:** Huapeng Zhou, Huayu Wang, Junkai Wu, Kangqi Wang, Xinyu Wang
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2609.32869

 - **Pdf link:** https://arxiv.org/pdf/2609.32869

 - **Abstract**
 Whisper is a widely used encoder-decoder model for speech recognition. Its encoder reads an utterance in one parallel pass, but its decoder writes the transcript one token at a time, which dominates inference time. Speculative decoding shortens such loops without changing their output: a small drafter guesses several upcoming tokens, and the original model verifies them all in one forward pass. We present Whisper-Flash, a two-layer drafter built on a property of speech recognition: the words still to be written have already been spoken. It reads Whisper's encoded audio and accepted decoder states and proposes eight tokens in a single forward pass. On the complete LibriSpeech test sets, Whisper-Flash processes $3.16\times/2.85\times$ as much audio per second as greedy decoding with identical outputs, and it remains faster at batch sizes up to 96 and under temperature sampling. Ablations show that direct access to the audio matters most.
#### CoLMbo-SV: A Grounded Language Model for Explainable Speaker Verification
 - **Authors:** Massa Baali, Sarthak Bisht, Ziyue Qiu, Joseph Konan, Rita Singh, Bhiksha Raj
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2609.33212

 - **Pdf link:** https://arxiv.org/pdf/2609.33212

 - **Abstract**
 Speaker verification systems achieve high accuracy but provide little account of the acoustic evidence behind their judgments. Making these systems inspectable requires exposing interpretable evidence while retaining the richer information on which their decisions depend. We present \textbf{CoLMbo-SV}, a speaker language model that combines strong speaker discrimination with structured, acoustically grounded comparison reports. By connecting a pretrained speaker encoder to a language model and supplying explicit acoustic measurements, CoLMbo-SV makes voice comparisons inspectable without restricting verification to the evidence verbalized in its reports. We additionally introduce \textbf{VoxReason}, paired recordings with measured acoustic properties and comparison reports filtered through numerical and qualitative checks, providing supervision for this combined capability. We also develop an evaluation framework that separates what acoustic information a speaker representation encodes, what influences the verification score, and what the generated report discusses. On VoxCeleb1-O, CoLMbo-SV achieves 0.99\% EER, reducing verification error by approximately 80\% relative to the strongest audio-language baseline fine-tuned on VoxReason, while attaining a numerical-grounding score of 0.82. Our analysis further demonstrates that acoustic correctness and decision relevance are distinct properties of an explanation, exposing a gap that numerical-grounding metrics miss. Together, these contributions substantially advance audio-language speaker verification, bring its accuracy toward that of dedicated speaker encoders while adding checkable acoustic reporting, and establish an empirical framework for connecting natural-language explanations to the decisions they explain.
#### Language Discrimination Improves Linguistic Learning in Multilingual Speech Models
 - **Authors:** Maureen de Seyssel, Jie Chi, Zakaria Aldeneh
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2609.33345

 - **Pdf link:** https://arxiv.org/pdf/2609.33345

 - **Abstract**
 Multilingual self-supervised speech models can benefit from sharing information across languages, but under a matched total pretraining data budget they still fall short of monolingual models. We show that strengthening the model's ability to discriminate languages during pretraining reduces and, on some measures, closes this multilingual gap on continuous phonetic and higher-level linguistic measures, while preserving substantial cross-language sharing. Using a controlled English/French HuBERT setting, we test two interventions which strengthen language discrimination: an auxiliary language classifier and per-language k-means targets. Across interventions, continuous-feature phone discrimination error (phone-ABX, lower is better) decreases from 11.6% in the bilingual baseline to 10.4% (monolingual: 10.8%), while lexical performance (sWUGGY, higher is better) increases from 52.1% to 56.7% (monolingual: 58.5%) and prosodic performance (ProsAudit, lexical subtask, higher is better) from 68.9% to 72.9% (monolingual: 72.6%). Across HuBERT training stages, the strongest gains on most linguistic measures occur when language discrimination is introduced in the first iteration, whereas later or repeated interventions yield smaller improvements and are accompanied by increased language-wise segregation. These results support a causal role for language discrimination in reducing the additional cost of multilingual learning.
#### Identity-Assisted Association of Unordered DOA Estimates for Neural Speech Source Tracking
 - **Authors:** Bing Yang, Di Liang, Xiaofei Li
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2609.33373

 - **Pdf link:** https://arxiv.org/pdf/2609.33373

 - **Abstract**
 Tracking speech sources remains a challenge due to ambiguous data association arising from intermittent speech, close spatial proximity, and complex acoustic conditions. To address these issues, we propose an identity-assisted association that maps unordered direction-of-arrival (DOA) estimates to speaker-consistent source trajectories for reliable speech source tracking. Specifically, speaker identity embeddings are directly integrated into the model input as a complementary cue to spatial features. This enables maintaining identity consistency by combining long-term time-invariant vocal identity characteristics with the short-term continuity of spatial cues. To effectively process these heterogeneous inputs while accommodating their distinct characteristics, we design a unified neural tracker. Within this model, time self-attention modules capture the temporal evolution of each source, while source self-attention modules distinguish between competing source tracks. Experimental results demonstrate the superiority of the proposed neural tracker in mitigating association confusion for speech source tracking.
#### What Survives the Codec Shift: Pooled No-Vocals Residuals for Speech Deepfake Detection
 - **Authors:** Jiajun Xu, Menglu Li, Xiao-Ping Zhang
 - **Subjects:** Subjects:
Sound (cs.SD); Artificial Intelligence (cs.AI); Machine Learning (cs.LG); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2609.33375

 - **Pdf link:** https://arxiv.org/pdf/2609.33375

 - **Abstract**
 The transition from vocoder-based to neural-codec speech synthesis makes generalization more difficult for speech deepfake detectors, particularly those relying on speech-oriented representations. It remains unclear which acoustic representations retain discriminative information when the generation mechanism changes. We therefore compare 12 acoustic representations using a shared low-capacity linear classifier to identify effective evidence under codec shift. The analysis shows that hierarchical XLS-R leads on the pooled test set, while pooled no-vocals residual statistics perform best on the unseen-codec condition, revealing complementary behavior across generation conditions. Building on this finding, we propose MN-P, a dual-view detector that integrates an utterance-level pooled no-vocals representation with token-level XLS-R features through adaptive gating. The proposed MN-P reduces EER by 54.2% overall and by 60.9% on the codec-unseen condition relative to the best-performing retrained state-of-the-art system, with consistent gains across different detector backends. These results indicate that pooled no-vocals residual statistics provide effective complementary evidence for cross-generation speech deepfake detection.
#### Context Spanning: A Communication Framework for Full-Duplex Speech Models and External LLM Backends
 - **Authors:** Seonghyeon Go, Yongwoo Kim, Hyeonjin Cha, Jaeho Shin
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2609.33443

 - **Pdf link:** https://arxiv.org/pdf/2609.33443

 - **Abstract**
 Full-duplex spoken dialogue models can listen and speak simultaneously like the real-time dynamics of human conversation. For natural dialogue, the ability to search for external information in real-time is also an important capability. Many models remain trapped in parametric knowledge, leaving them unable to access real-time information and tool execution. Furthermore, even when Large Language Models (LLM) retrieve information, many duplex speech models process it within a compressed latent space rather than in its raw text form, which can lead to information loss from compression. To address this issue, we propose Context Spanning, a framework for information injection between a full-duplex speech model and an external LLM backend via real-time chunked prefill. The injected frame is encoded in a single forward pass inside the real-time frame budget. It feeds the retrieved information to the speech model as-is, enabling it to reason over the information independently and generate responses. With this approach, our model achieves high performance on Full-Duplex benchmarks and strong results on Question Answering tasks, demonstrating its conversation potential. Context Spanning shows that external information can be injected directly into a duplex speech model, introducing a new simple and powerful mechanism for duplex systems.
#### Jev Matches 7B Language Models for Speech-Neuroprosthesis Rescoring
 - **Authors:** Gabriele Cinà
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2609.33538

 - **Pdf link:** https://arxiv.org/pdf/2609.33538

 - **Abstract**
 A speech neuroprosthesis decodes attempted speech from brain activity and ends by rescoring the decoder's candidate sentences with a language model of several billion parameters, the only component that needs a GPU. Replacing that model with a cheaper one is hard: general language models asked to pick one sentence from a list answer from where a label sits in the list rather than from the sentence itself. We pose rescoring as a single typed decision, one call that returns a probability for every candidate, served by Jev, a hosted model trained for calibrated decisions, and combine it with the decoder's own score. On 978 held-out sentences from a participant with ALS, where the published decoder alone reaches 8.1% word error, Jev reaches 7.5% against 7.8% for both OPT-6.7b and Qwen2.5-7B; with the decoder's weight re-tuned, 6.9% against 7.2% and 7.4%. Jev is ahead in all four comparisons and at most 0.2 points behind at the 95% bound. It costs 0.07 USD per thousand sentences and needs no GPU; a dedicated GPU running a 7B model is cheaper per sentence only above 43% utilisation, far beyond what one user generates. End-to-end latency over the internet is 262 ms, of which 62 ms is spent at the provider, the same order as a 7B model on a local GPU (27 ms) but not faster.
#### DuraS2ST: Chain-of-Thought and Reinforcement Learning for Duration-Aligned Speech-to-Speech Translation
 - **Authors:** Yayue Deng, Dingdong Wang, Yuxuan Hu, Jinyu Li, Yanqing Liu, Yuanyuan Wang, Weidong Chen, Helen M. Meng, Shujie Liu, Xixin Wu
 - **Subjects:** Subjects:
Sound (cs.SD); Computation and Language (cs.CL); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2609.33742

 - **Pdf link:** https://arxiv.org/pdf/2609.33742

 - **Abstract**
 Speech-to-speech translation (S2ST) in time-sensitive applications such as video dubbing requires not only semantic fidelity and speaker preservation, but also strict duration consistency to avoid audio-visual misalignment. However, existing S2ST systems largely generate target speech without explicit temporal planning, making duration control an unresolved challenge. We introduce DuraS2ST, a duration-aligned reasoning framework that enables a single speech language model to first generate an explicit chain-of-thought (CoT) for planning target wording and phonetic length, and then synthesize the corresponding speech tokens. To support this paradigm, we construct DuraSet-440K, a high-quality duration-aligned CoT corpus for supervised initialization. We further optimize the model with multi-modal multi-dimensional reinforcement learning, using a Duration Margin Reward to balance translation quality and duration consistency, and Modality-Aware Reward Attribution to assign rewards to appropriate token spans. Experiments on CVSS-T show that DuraS2ST achieves a strong balance between translation quality and duration consistency, outperforming competitive open-source and commercial baselines. Project page: this https URL.
#### Transformer-based Neural Beamforming for Real-Time Speech Enhancement on Smart Low-Power Hearable Devices
 - **Authors:** Luca Bompani, Marco Fariselli, Giovanni Oltrecolli, Francesco Conti
 - **Subjects:** Subjects:
Sound (cs.SD); Artificial Intelligence (cs.AI); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2609.33755

 - **Pdf link:** https://arxiv.org/pdf/2609.33755

 - **Abstract**
 Accurate, efficient, and low-latency spatial beamforming is a key component in emerging smart hearable devices, enhancing speech while suppressing noise and interference. However, handling multiple input sources under strict real-time constraints poses significant challenges for the low-power, resource-constrained microcontroller units (MCUs) used in hearables. We present an optimized methodology for the real-time execution of a neural-network-based minimum variance distortionless response (MVDR) beamformer on MCUs. Using six microphones and a three-stage mixed-precision scheme (float32 MVDR, int8 CNN, float16 Transformer), the pipeline pairs a CNN that estimates the MVDR weights with a lightweight Transformer that applies a per-frame correction. By time-slicing weight estimation with beamforming, it achieves a 15~ms per-frame latency while refreshing a complete set of CNN-derived weights every 564~ms. The deployed mixed-precision pipeline attains a short-time objective intelligibility (STOI) of 97.65\%, a scale-invariant signal-to-noise ratio (SI-SNR) of 20.26~dB, and a wideband PESQ of 3.676 at an average power of 45.9~mW. A speech activity detection (SAD) module (98.5\% accuracy, 0.62~mJ per inference) bypasses the pipeline during silence; under realistic deployment conditions, the system exceeds the 16~h all-day target on a 100~mAh battery, with an estimated lifetime of up to $\sim$20~h. To our knowledge, this is the first real-time multi-channel Transformer-based neural beamforming pipeline deployed on an MCU-class device.
#### Tokens Change, Structure Endures: Spectral Watermarking for Generated Speech
 - **Authors:** Kanghwi Lee, Kyeongseok Jeong, Jeongmin Liu
 - **Subjects:** Subjects:
Sound (cs.SD); Computation and Language (cs.CL); Cryptography and Security (cs.CR); Machine Learning (cs.LG); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2609.33774

 - **Pdf link:** https://arxiv.org/pdf/2609.33774

 - **Abstract**
 Watermarking is a promising tool for establishing the provenance of AI-generated speech. While many neural audio watermarking methods rely on a separately trained watermark generator, token-level watermarking is a training-free alternative that operates directly during generation. Its main weakness is retokenization: decoding generated speech to a waveform and encoding it again can change token identities and erode the watermark. To make the watermark robust to these changes, we propose Redwing, REtokenization-Durable Watermarking IN Generation. It builds a graph from the token substitutions observed under retokenization, whose Laplacian yields a basis that assigns similar values to tokens likely to substitute for one another. Over this basis, embedding and detection functions are jointly optimized to preserve watermark signal through retokenization while limiting embedding distortion and detector variability on unwatermarked speech. On the Moshi full-duplex system, after eight consecutive passes of Mimi resynthesis, Redwing achieves 80.7% TPR at a calibrated 1% FPR, compared with 8.3% for KGW and at most 7.3% for WMAR. It also has the highest TPR after eight passes through three other neural codecs (77.5-93.0%), and the gains generalize to TTS models at a speech-quality cost close to that of KGW. These results show that retokenization is not merely a source of noise: its transition structure can be exploited as a design principle for robust token-level watermarking.
#### Controlling Speaking Rate in Autoregressive TTS via Activation Steering
 - **Authors:** Francesco Verdini, Antonis Asonitis, Aref Farhadipour, Marzieh Razavi, Pierre-Edouard Honnet, Vijeta Avijeet, Juan Pablo Zuluaga Gomez
 - **Subjects:** Subjects:
Sound (cs.SD); Computation and Language (cs.CL); Machine Learning (cs.LG); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2609.33810

 - **Pdf link:** https://arxiv.org/pdf/2609.33810

 - **Abstract**
 Autoregressive text-to-speech (TTS) systems synthesize natural speech but, once trained, offer little control over speaking rate. We show that speaking rate can be steered at inference time, without retraining, by clamping a single decoder block's activation along a discovered speed axis. A decoder-block analysis recovers the rate axis, a neutral operating point, and a per-step intensity scale; at inference, the activation's projection onto this axis is set to a fixed scalar. Learning this direction from synthetically time-stretched and time-compressed speech yields rate control that largely preserves speaker identity, generalizes across model architectures, and maintains high naturalness in objective and human evaluations. Unlike standard additive steering, which breaks at the slow extreme, clamping remains stable on all three systems tested; at moderate targets, the better rule depends on the model. Finally, we show that rate information is decodable across layers but causally steerable only within a mid-depth window, and demonstrate the effectiveness of our approach on the public Seed-TTS-Eval benchmark.
#### Unified Target-Speaker ASR with Text and Enrollment Speech Cues
 - **Authors:** Yuxiang Mei, Yuchen Yan, Dongxing Xu, Jiaen Liang, Yanhua Long
 - **Subjects:** Subjects:
Signal Processing (eess.SP); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2609.33853

 - **Pdf link:** https://arxiv.org/pdf/2609.33853

 - **Abstract**
 Target-speaker automatic speech recognition (TS-ASR) aims to recognize a designated speaker while suppressing interfering speech in multi-talker environments. Conventional TS-ASR typically relies on an enrollment utterance, whereas text-guided methods use known lexical content, such as a wake word, to identify the target speaker from the observed mixture. These two cues provide complementary information but are usually studied separately. We propose a Unified Dual-Cue TS-ASR framework that supports text cues, enrollment speech, or both within a single model. Text cues interact with the mixture representation to extract target-speaker information conditioned on known lexical content, while an independent enrollment utterance provides complementary speaker information. Cross-attention cue-conditioning modules are integrated into shared Conformer blocks, and negative-cue sampling provides cue-validity supervision during dual-cue training. Experiments on 30,000 two-speaker mixtures across five recording/domain conditions and four oracle text-cue lengths show that, with five-character text cues, the concatenated dual-cue method achieves 8.80% CER, compared with 17.32% for text-only and 29.06% for enrollment-only inference. It also outperforms parallel dual-cue fusion (9.49% CER) and yields lower dual-cue CER across all five evaluation subsets. These results demonstrate the benefit of jointly exploiting complementary lexical and speaker information for target-speaker ASR.
#### In-Context Adaptation of Encoder-Decoder Models in Speech Recognition
 - **Authors:** Yen Meng, Sharon Goldwater, Hao Tang
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2609.33865

 - **Pdf link:** https://arxiv.org/pdf/2609.33865

 - **Abstract**
 In-context learning offers an appealing approach to adapt automatic speech recognition (ASR) models to new speakers, accents, and domains by providing speech-text pairs as demonstrations at inference time. Recent work shows that some LLM-based speech models are capable of ASR in-context adaptation, when providing interleaved speech-text demonstrations. In this work, we ask whether in-context adaptation is an inherent ability for all encoder-decoder models. We study two forms of demonstration, collated and interleaved demonstration, across six encoder-decoder models, spanning conventional cross-attention-based and LLM-based architectures. We find that all tested models are able to perform in-context adaptation out of the box, achieving up to 30% relative improvement in the oracle experiments and up to 23% using first-pass hypotheses. Through controlled experiments on three English datasets, we show that lexical and speaker information both contribute to successful adaptation. While interleaved demonstration is effective in certain cases, collated demonstration brings consistent adaptation across the board. Our results suggest that in-context adaptation for ASR is not unique to specific architectures, training, or demonstration approaches.
#### SALMONN-duo: Adaptive Dual-System Coordination for Full-Duplex Voice Agents
 - **Authors:** Wenyi Yu, Siyin Wang, Terumi Chiba, Xianzhao Chen, Xiaohai Tian, Jun Zhang, Lu Lu, Chao Zhang
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2609.34247

 - **Pdf link:** https://arxiv.org/pdf/2609.34247

 - **Abstract**
 Full-duplex speech large language models (LLMs) enable low-latency, natural voice interaction. However, real-world agents must also use tools and perform deliberative reasoning-operations whose variable latency and computational cost conflict with the stringent timing requirements of real-time conversation. To reconcile these demands, we propose SALMONN-duo, an adaptive dual-system voice agent inspired by dual-process theories of cognition. SALMONN-duo separates real-time interaction from deliberative computation by pairing an always-on, fast-thinking full-duplex speech LLM (system 1) with a powerful asynchronous slow-thinking LLM agent (system 2). Beyond handling real-time interaction, system 1 learns when to answer directly and when to delegate, remaining responsive during backend execution and seamlessly integrating returned information into the ongoing dialogue without exposing tool traces or losing conversational context. Evaluations on single-turn spoken question answering (QA) and multi-turn conversations demonstrate that adaptive delegation substantially improves accuracy on knowledge-intensive and multi-hop reasoning questions, while knowledge-boundary-aware training avoids unnecessary system 2 invocations. On a customized version of $\tau$-Voice, SALMONN-duo further demonstrates its ability to complete environment-grounded, policy-constrained tasks through multi-turn interactions in realistic business scenarios. Finally, cost-aware reinforcement learning further enhances the trade-off between task performance and backend usage across the QA and conversation tasks, while improving task success and response safety on $\tau$-Voice with an acceptable increase in the delegation rate.
#### Unsupervised Speech Enhancement via Drifting
 - **Authors:** Diego Caviedes-Nozal, Liang Xu, Rasmus Kongsgaard Olsson, W. Bastiaan Kleijn
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2609.34662

 - **Pdf link:** https://arxiv.org/pdf/2609.34662

 - **Abstract**
 This paper addresses unsupervised speech enhancement in the unpaired setting using drifting methods, where training relies on separate collections of degraded and clean audio without corresponding pairs. While recent drifting approaches enable unpaired training, they do so at a heavy cost: because the objective optimizes only a marginal prior over clean speech, the enhancer gradually loses the input's linguistic content and speaker identity. To fix this, we introduce input-conditioned drifting. We preserve the pull of the clean corpus while re-tethering the output to the degraded input via two mechanisms: an anchor encoder supplies the missing likelihood by pulling toward the input's features, and a key encoder conditions the prior by re-weighting retrieved frames. Neither requires labels or paired data. Using a training-free encoder selection criterion, Word Error Rate on VoiceBank-DEMAND falls to 10.1% (unprocessed: 11.7%), speaker similarity recovers from 0.490 to 0.879, and the recipe transfers in part to dereverberation on WSJ0-REVERB: content improves, rendering quality does not.
#### CoSE-E: A Benchmark for Code-switched Speech Evaluation in Enterprise Settings
 - **Authors:** Shama Gupta, Hoang H Nguyen, Chelsea Huang, Lindsay Devon Brin, Fanny Riols
 - **Subjects:** Subjects:
Sound (cs.SD); Computation and Language (cs.CL); Machine Learning (cs.LG); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2609.35645

 - **Pdf link:** https://arxiv.org/pdf/2609.35645

 - **Abstract**
 Code-switching (CS), a seamless alternation between languages within a single utterance, remains a critical challenge in automatic speech recognition (ASR). While prior works focus on conversational CS-ASR, enterprise settings demand evaluation of operational impact beyond edit-distance errors: how code-switching transcription errors propagate to downstream voice agent task failures. In this work, we propose (1) a CS-ASR synthetic benchmark and multidimensional evaluation framework tailored to enterprise domains, (2) systematic evaluation of frontier ASR systems across 5 language pairs, (3) diagnostic analysis of the additional transcription errors that code-switching introduces across language pairs and models. We release COSE-E to support enterprise-focused CSASR evaluation for multilingual voice agents in enterprise deployment.


by Zyzzyva0381 (Windy). 


2026-09-29
