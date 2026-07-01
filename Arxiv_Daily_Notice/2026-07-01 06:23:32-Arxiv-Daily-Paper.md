# Showing new listings for Wednesday, 1 July 2026
Auto update papers at about 2:30am UTC (10:30am Beijing time) every weekday.


阅读 `Usage.md`了解如何使用此repo实现个性化的Arxiv论文推送

See `Usage.md` for instructions on how to personalize the repo. 


Keyword list: ['text-to-speech', 'text to speech', 'tts', 'LLM-based', 'speech', 'voice']


Excluded: []


### Today: 16papers 
#### Listening Between the Lines: Joint Learning of ASR Embeddings and LLM-Augmented Linguistics for Dementia Detection
 - **Authors:** Olivier Jiyoun Jung, Jonghyeon Park, Myungwoo Oh
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Artificial Intelligence (cs.AI); Machine Learning (cs.LG); Quantitative Methods (q-bio.QM)
 - **Arxiv link:** https://arxiv.org/abs/2606.30675

 - **Pdf link:** https://arxiv.org/pdf/2606.30675

 - **Abstract**
 Early detection of dementia through speech analysis offers a non-invasive screening alternative, but capturing both acoustic and linguistic biomarkers remains challenging. We propose a multimodal framework leveraging Whisper for dual-purpose extraction: acoustic representations from encoder outputs and transcripts via automatic speech recognition (ASR). For the acoustic pathway, temporal networks with attention pooling aggregate variable-length sequences into fixed-dimensional embeddings. For the linguistic pathway, we prompt a large language model (LLM) to extract interpretable features spanning lexical diversity, syntactic complexity, semantic coherence, and discourse patterns. A gated fusion network integrates both modalities. On ADReSS and ADReSSo, our method achieves F1-scores of 89.47% and 90.14%, demonstrating effective integration of acoustic and LLM-augmented linguistic features. Ablation shows that multimodal fusion consistently outperforms either modality alone.
#### Preserving Speech-to-Text LLM Capabilities in Speech-to-Speech Generation
 - **Authors:** Yuxuan Hu, Heng Lu, Ruchao Fan, Yao Qian, Xiaofei Wang, Jian Xue, Heming Wang, Shuohang Wang, Young Jin Kim, Yelong Shen, Jinyu Li
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2606.30944

 - **Pdf link:** https://arxiv.org/pdf/2606.30944

 - **Abstract**
 Strong speech-to-text (S2T) LLMs already provide robust speech perception and text reasoning, but adding speech-to-speech (S2S) output is challenging: fine-tuning the backbone can degrade the original S2T performance, while attaching a downstream talker reintroduces a serial text-to-speech bottleneck. We present PRIME-Speech, a frozen-backbone S2S conversion framework that trains only speech-generation modules. PRIME-Speech synchronizes a causal audio post-decoder with intermediate hidden states of the frozen backbone, so codec tokens are generated from the model's evolving reasoning trajectory rather than from completed text chunks. The post-decoder uses mixed hidden-state, text, and audio-history conditioning, and a training-time packing strategy with turn-level audio KV-cache and position reset stabilizes multi-turn spoken interaction without additional multi-turn S2S training data. Multi-token prediction further reduces the effective codec prediction rate and improves first-audio latency without modifying the reasoning path. Across speech translation, spoken QA, speech understanding, and multi-turn dialogue, PRIME-Speech preserves the S2T behavior of the frozen backbone while producing accurate, low-WER spoken responses.
#### Beyond Cross-Reconstruction: Probing-Based Disentanglement Evaluation for Acoustic Teleportation Codecs
 - **Authors:** Philipp Grundhuber, Emanuël A. P. Habets
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2606.31365

 - **Pdf link:** https://arxiv.org/pdf/2606.31365

 - **Abstract**
 Some neural audio codecs disentangle speech into latent subspaces encoding content, speaker identity, and acoustics, enabling acoustic teleportation and voice conversion. Existing evaluations rely on cross-reconstruction quality, which cannot reliably detect leakage across partitions. We extend a probing based framework to assess disentanglement by regressing room-acoustic parameters (reverberation time, clarity, and direct-to-reverberant ratio) and classifying speaker identity, using the gap between intended and unintended partitions as the disentanglement measure. Applied to an acoustic teleportation codec, we find speaker identity is largely confined to its partition, while acoustics leak into the speech embeddings due to the training objective. Acoustic embeddings blindly estimate room parameters within 0.02 s of supervised baselines, indicating physically meaningful structure emerges without explicit supervision.
#### How Bilingual Are SSL Speech Models? Cross-Lingual Probing of Articulatory Encoding with Finnish and Russian EMA
 - **Authors:** Ailín Pollio San Pedro, Tomi Kinnunen, Alexandre Nikolaev, Ruchi Pandey
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2606.31527

 - **Pdf link:** https://arxiv.org/pdf/2606.31527

 - **Abstract**
 SSL speech models capture rich phonetic, prosodic, and acoustic patterns from raw audio, yet how they encode articulatory information across diverse languages remains unclear. Using EMA data from bilingual Finnish-Russian speakers, we evaluate cross-lingual correlations between SSL latent representations and articulatory movements. Models achieve strong prediction performance (Pearson r up to 0.68) even with approximately 5 minutes of training data, with multilingual models outperforming monolingual ones. Intermediate layers encode articulatory features most effectively, and tongue movements are more predictable than lip movements. We also assess the impact of task type (read versus spontaneous speech) and language proficiency, finding higher accuracy for structured tasks and strong generalization across proficiency levels. These results enhance the interpretability of SSL models and show their potential for speech-technology applications.
#### Improving multichannel speech enhancement through accurate room-acoustic simulations
 - **Authors:** Georg Götz, Alessia Milo, Steinar Guðjónsson, Daniel Gert Nielsen, Jesper Pedersen, Finnur Pind
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Artificial Intelligence (cs.AI); Machine Learning (cs.LG); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2606.31552

 - **Pdf link:** https://arxiv.org/pdf/2606.31552

 - **Abstract**
 Room-acoustic simulations are widely used to augment training data for deep-learning-based speech enhancement. While most pipelines rely on simplified geometrical acoustics, wave-based approaches offer greater physical accuracy. In this work, we examine how simulation fidelity affects multichannel speech enhancement performance. To this end, we train SpatialNet on datasets augmented with different room-acoustic simulation methods and evaluate the resulting models on measured data. We compare lower-fidelity datasets based on geometrical acoustics with a high-fidelity dataset using advanced acoustic modelling and a hybrid combination of wave-based and geometrical acoustics simulations. Training on the high-fidelity dataset results in an up to 38 % relative reduction in median word error rate compared to the lower-fidelity alternatives. These results show that augmentation with high-fidelity room-acoustic simulations directly translates into improved multichannel speech enhancement performance.
#### Is Natural Always Appropriate? Investigating Naturalness and Appropriateness Across Different Domains for TTS Evaluation
 - **Authors:** Dominika Woszczyk, Andreas Triantafyllopoulos, Jura Miniota, Éva Székely, Bjoern Schuller
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Machine Learning (cs.LG)
 - **Arxiv link:** https://arxiv.org/abs/2606.31729

 - **Pdf link:** https://arxiv.org/pdf/2606.31729

 - **Abstract**
 Text-to-speech (TTS) evaluation is an open challenge. While the primary target was "naturalness," recent fidelity gains shifted focus toward "appropriateness" and whether speech is correct for its context. In this work, we examine how perception changes when the expected downstream use varies. We measure the appropriateness and human-likeness of five SOTA TTS systems across five domains: AI assistant, reader, actor, animated character, and spontaneous speaker. Results show appropriateness varies across domains independently of naturalness. While systems shine at reading, expressive domains remain challenging, and optimizing for one can degrade others. Furthermore, naturalness scores tend to penalize stylized speech while rewarding spontaneity. Finally, our study also highlights blind spots in one-size-fits-all evaluation metrics across more expressive domains. We demonstrate that TTS performance is not "solved" but depends on the target domain, requiring context-aware evaluation.
#### A Fair and Transparent Framework for Speech-Based Depression Detection: Balancing Interpretability and Performance
 - **Authors:** Mariel Estevez, Alfonso Ortega, Antonio Miguel, Eduardo Lleida
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2606.31730

 - **Pdf link:** https://arxiv.org/pdf/2606.31730

 - **Abstract**
 While speech provides rich, non-invasive biomarkers for mental-health assessment, clinical adoption is limited by opaque models and potential demographic bias. In this work we propose a methodological framework to evaluate robustness and interpretability for automated depression detection on the extended DAIC-WOZ dataset using low-complexity machine learning baselines (RF, SVM, and MLP) chosen to mitigate overfitting and enhance generalization in combination with human-understandable acoustic features (MFCCs, eGeMAPS). To balance accuracy with clinical trust, we leverage explainability methods (LIME and SHAP) for feature selection, validating our findings with statistical significance tests and demographic fairness analyses to mitigate spurious, artifact-driven correlations. Empirical results demonstrate that an optimized subset of explainable AI (XAI)-selected features combined with an MLP architecture achieves a state-of-the-art test accuracy of 82\%. Ultimately, this work provides a transparent framework for robust and ethical assistive technologies that can be applied to any other binary task.
#### ASR-Agnostic Multimodal Spectrotemporal Modeling for Early Dementia Detection
 - **Authors:** Chukwuemeka Ugwu, Oluwafemi Richard Oyeleke
 - **Subjects:** Subjects:
Sound (cs.SD); Artificial Intelligence (cs.AI); Computation and Language (cs.CL); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2606.30646

 - **Pdf link:** https://arxiv.org/pdf/2606.30646

 - **Abstract**
 Speech recruits the same executive, attentional, and working memory processes underlying instrumental activities of daily living, or IADLs, providing a non-invasive proxy for cognitive assessment. Yet most speech-based dementia detection systems depend on transcription, discard within-recording temporal structure, and are validated on a single English corpus with known recording artifacts. We propose an ASR-agnostic framework operating directly on Mel spectrograms. Our key contribution is extracting spectrotemporal displacement fields from consecutive spectrogram frames, capturing shifting spectral energy patterns as digital biomarkers of cognitive decline. These features are fused with CNN-ConvGRU acoustic embeddings via a learned cross-attention mechanism and aggregated using a Transformer encoder with learnable query pooling. A composite temporal loss enforces smoothness and contrastive coherence across segments. We train independent models on English DementiaBank, Slovak EWA-DB, and Spanish Ivanova corpora, using clinical elicitation protocols taxing IADL-relevant cognitive domains. The Slovak model achieves 83.9% accuracy, and Spanish achieves, while the English baseline yields 53.2%, confirming known artifacts. Cross-lingual ablation studies reveal distinct fusion regimes: removing cross-attention collapses Spanish performance to 53.7%, below unimodal models, while the Slovak audio encoder alone outperforms the full model, 93.7% vs. 83.9%, and all English configurations remain near chance. Thus, multimodal fusion's value is corpus-dependent: essential when signal is distributed across modalities, counterproductive when one dominates, and irrelevant when no signal exists. Auxiliary temporal losses converge to language-invariant values, indicating cross-lingual architectural stability.
#### Enhancing BEST-RQ Pseudo-Label Quality through Online Refinement for Automatic Speech Recognition
 - **Authors:** Jingjing Xu, Zijian Yang, Mohammad Zeineldeen, Eugen Beck, Ralf Schlueter, Hermann Ney
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2606.30671

 - **Pdf link:** https://arxiv.org/pdf/2606.30671

 - **Abstract**
 BEST-RQ is a simple and effective self-supervised training method for speech representation learning that performs well on automatic speech recognition (ASR) tasks. It generates pseudolabels using a fixed online quantization scheme, which simplifies training but provides weaker supervision than HuBERT-style models that iteratively refine pseudo-labels. In this work, we improve online pseudo-label generation while preserving simplicity. We propose three modifications: replacing the quantizer's linear projection with Principal Component Analysis (PCA), updating the codebook via iterative codebook refinement, and introducing an additional codebook updated via codebook distillation. We pre-train on the LibriSpeech 960-hour dataset and fine-tune using 100 hours of supervised LibriSpeech data. With all three modifications enabled, we achieve a 12% relative reduction in word error rate (WER) on the LibriSpeech test-other set, improving from 10.1% to 8.8%.
#### ALM2Vec: Learning Audio Embeddings for Universal Audio Retrieval with Large Audio-Language Models
 - **Authors:** Fengjie Lu, Chenang Jiang, Jiarui Hai, Helin Wang, Aaron Yee
 - **Subjects:** Subjects:
Sound (cs.SD); Artificial Intelligence (cs.AI); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2606.30682

 - **Pdf link:** https://arxiv.org/pdf/2606.30682

 - **Abstract**
 Recent advances in language--audio retrieval have been largely driven by contrastive dual-encoder architectures that align audio and text in a shared embedding space. While effective, existing retrieval embeddings are primarily optimized for audio--caption matching, limiting their ability to support diverse retrieval objectives and controllable retrieval behaviors. We present ALM2Vec, a universal audio embedding framework derived from pretrained large audio--language models (LALMs). By transferring the audio understanding, instruction-following, and reasoning capabilities acquired through large-scale multimodal training, ALM2Vec learns a unified embedding space for retrieval across audio domains and task types. Beyond conventional text--audio retrieval, ALM2Vec incorporates natural-language instructions into the embedding process, enabling instruction-aware retrieval for scenarios such as audio question answering and aspect-conditioned retrieval. Experimental results show that ALM2Vec achieves competitive performance on standard audio and speech retrieval benchmarks while exhibiting promising compositional and controllable retrieval capabilities, highlighting its potential as a unified audio embedding model for retrieval across domains, tasks, and user intents.
#### BEST-RQ-2: Contextualize-Then-Predict, a Two-Step Approach for Self-Supervised Audio Representations
 - **Authors:** Ludovic K. Tuncay (IRIT-SAMoVA), Etienne Labbé (IRIT-SAMoVA), Thomas Pellegrini (IRIT-SAMoVA)
 - **Subjects:** Subjects:
Sound (cs.SD); Artificial Intelligence (cs.AI); Machine Learning (cs.LG); Audio and Speech Processing (eess.AS); Signal Processing (eess.SP)
 - **Arxiv link:** https://arxiv.org/abs/2606.30700

 - **Pdf link:** https://arxiv.org/pdf/2606.30700

 - **Abstract**
 Self-supervised learning enables audio representations that transfer across domains and tasks. We present BEST-RQ-2, an evolution of BEST-RQ that retains frozen randomprojection-based discrete targets while introducing a two-step contextualize-then-predict pretraining scheme. A ViT context encoder processes only the unmasked spectrogram regions, and a lightweight predictor infers targets for the masked regions; the predictor is discarded after pretraining. Replacing the original Conformer encoder with a ViT shifts performance across domains, slightly reducing speech performance while improving music and environmental sounds, with comparable average scores. The main improvement comes from decomposing masked prediction into separate contextualization and prediction stages. On the X-ARES and XARES-LLM benchmarks, BEST-RQ-2 consistently outperforms one-stage baselines in overall transfer while keeping inference compute unchanged. Code and model checkpoints are publicly available.
#### Probing-Guided Layer Selection from Self-Supervised Speech Models for Generalizable Audio Deepfake Detection
 - **Authors:** Marjan Beheshti, Majid Rostami, Bo Chen
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2606.30791

 - **Pdf link:** https://arxiv.org/pdf/2606.30791

 - **Abstract**
 Audio deepfake detection systems often fail to generalize across domains because they rely on features tied to specific attacks or recording conditions. Self-supervised speech models offer rich multi-layer representations, yet existing approaches either use a single layer or fuse all layers indiscriminately, and only reveal layer importance after training. We propose a model-agnostic, two-stage methodology that identifies informative depth zones before any task-specific model is trained. In the first stage, lightweight XGBoost probes evaluate each transformer layer's cross-domain discriminative power, producing a layer ranking. In the second stage, a compact neural classifier fuses only the selected layers through per-layer attention pooling and a shared bottleneck projection, while the backbone remains frozen. Applied across three backbones, the probing reveals two key findings. First, informative layers cluster in depth zones rather than at uniquely optimal positions: within-zone substitutions fall within multi-seed noise, while zone violations degrade performance by up to 5x. Second, the probing produces backbone-specific selections rather than a fixed layer recipe. On XLS-R-300M, four probing-selected layers with 1.34M trainable parameters achieve 4.94 +/- 0.32% equal error rate on In-The-Wild and 5.07% cross-domain average over four shared datasets, a 28% relative improvement over the best prior frozen-backbone result (Xiao and Vu, 2025) using all 25 layers with identical training data.
#### Reference-Based Prosody and Rhythm Evaluation for Spoken Dialogue Systems
 - **Authors:** Ashish Hallur, Thomas Thebaud, Georgi Tinchev, Venkatesh Ravichandran, Laureano Moro-Velazquez
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2606.31055

 - **Pdf link:** https://arxiv.org/pdf/2606.31055

 - **Abstract**
 Speech-to-speech (S2S) AI agents are advancing rapidly, yet evaluation lacks interpretable speech-native measures for conversational prosody and rhythm. Because $F_0$, speaking rate, articulation rate, and pausing shift with model-predicted speaker traits and interaction state, pooled human statistics can be poorly calibrated for evaluating a particular output. Using 4000+ hours of dyadic English conversation from the Seamless Interaction dataset, we construct matched reference regimes for $F_0$ mean, $F_0$ expressivity, speech rate, articulation rate, pause ratio, and mean pause duration. We then define a percentile-based evaluation protocol: extract the same metrics from an S2S output waveform, compare them to the closest matched human reference stratum, and report percentile deviations or 5th-95th percentile out-of-regime flags. On held-out human rows, pooled references over-flag state-conditioned $F_0$ expressivity and rhythm, while matched references return flag rates closer to the nominal 10% and make deviation direction interpretable. These outputs serve as behavioral plausibility checks that complement, rather than replace, perceptual and user-centered evaluation.
#### Attacking UTMOS: Probing the Robustness of a Speech Quality Assessment Model
 - **Authors:** Wen-Chin Huang, Tomoki Toda
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2606.31105

 - **Pdf link:** https://arxiv.org/pdf/2606.31105

 - **Abstract**
 UTMOS has become one of the most commonly used deep neural network-based speech quality assessment (SQA) metrics in speech processing research. In this paper, we attack UTMOS to probe its robustness. Starting from high-quality speech samples, we optimize the input in two directions: a score-preserving attack, which degrades perceived quality while maintaining the predicted score, and a quality-preserving attack, which lowers the predicted score while maintaining perceived quality. We consider three input spaces: raw waveform, mel spectrogram with a HiFi-GAN vocoder, and the latent space of EnCodec, a neural audio codec. Experimental results show that score-preserving attacks are effective against UTMOS. Although perfect quality-preserving attacks are more difficult, optimization in the EnCodec latent space provides the best chance of success. These results reveal failure modes of UTMOS and highlight the importance of robustness analysis for DNN-based SQA metrics.
#### UniSAE: Unified Speech Attribute Editing on Speaker, Emotion and Low-Level Content via Discrete Phonetic Posteriorgram Modelling
 - **Authors:** Chuanbo Zhu, Wuyou Zhou, Rongxiu Zhong, Shilei Zhang, Kun Qian, Yike Guo, Wei Xue
 - **Subjects:** Subjects:
Sound (cs.SD); Artificial Intelligence (cs.AI); Computation and Language (cs.CL); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2606.31128

 - **Pdf link:** https://arxiv.org/pdf/2606.31128

 - **Abstract**
 Speech editing aims to modify specific portions of an utterance while preserving the remaining speech. Existing approaches primarily focus on word-level content modification and typically treat content, speaker, and emotion editing as separate tasks, limiting both editing granularity and flexibility. We propose UniSAE, a unified speech attribute editing framework which supports composable speaker, emotion and content editing from sub-phoneme to word level within a single architecture. UniSAE introduces a Discrete Phonetic PosteriorGram (DPPG) representation that factorizes speech content into discrete tokens encoding phoneme identity, pronunciation variants, and duration, enabling direct phoneme- and sub-phoneme-level editing. For higher-level modifications, an autoregressive content transformer predicts edited DPPG sequences for word-level content editing. The edited sequences are rendered into speech by a diffusion-based acoustic decoder, conditioned on disentangled speaker and emotion representations. Experimental results demonstrate that the proposed unified framework supports precise speaker and emotion control, content editing at multiple granularities, and joint modification of all three attributes within a single framework.
#### FlexiSLM: A Dynamic and Controllable Frame Rate Spoken Language Model
 - **Authors:** Jiaqi Li, Chaoren Wang, Xiaohai Tian, Mingjie Chen, Xinyu Liang, Xu Li, Yufan Lin, Junwen Qiu, Jun Zhang, Lu Lu, Haizhou Li, Zhizheng Wu
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2606.31247

 - **Pdf link:** https://arxiv.org/pdf/2606.31247

 - **Abstract**
 Spoken language models (SLMs) extend LLMs to speech input and output. Existing SLMs represent speech at fixed frame rates (e.g., 25 or 12.5 Hz), ignoring the time-varying information density of speech and offering no flexibility to trade off quality for speed at inference time. Recent audio tokenizer research has proposed dynamic frame rate speech coding, which exploits this non-uniformity and enables two new capabilities: very low average frame rates and frame rate controllability. However, this technique has not yet been applied to SLMs. We introduce Flexible Spoken Language Model (FlexiSLM), the first SLM that supports dynamic and controllable frame rates on both speech input and output. Using dynamic frame rate representations, FlexiSLM outperforms fixed-frame-rate 7B models including Qwen2.5-Omni and Kimi-Audio at its high-quality operating points. We further verify that FlexiSLM can be accurately steered down to 4.0 Hz; at 6.25 Hz, it roughly halves inference time relative to 12.5 Hz while retaining strong speech-to-speech quality. Audio samples are available at this https URL .


by Zyzzyva0381 (Windy). 


2026-07-01
