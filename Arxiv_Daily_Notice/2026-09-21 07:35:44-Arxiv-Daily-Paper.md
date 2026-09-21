# Showing new listings for Monday, 21 September 2026
Auto update papers at about 2:30am UTC (10:30am Beijing time) every weekday.


阅读 `Usage.md`了解如何使用此repo实现个性化的Arxiv论文推送

See `Usage.md` for instructions on how to personalize the repo. 


Keyword list: ['text-to-speech', 'text to speech', 'tts', 'LLM-based', 'speech', 'voice']


Excluded: []


### Today: 13papers 
#### Cross-Lingual Parkinson's Disease Severity Assessment Using Pre-trained Speech Embeddings: A Multi-Class Evaluation
 - **Authors:** Simon Pals, Cristian Tejedor-Garcia
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Computation and Language (cs.CL); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2609.20875

 - **Pdf link:** https://arxiv.org/pdf/2609.20875

 - **Abstract**
 Parkinson's disease (PD) often manifests through speech impairments, facilitating accessible, non-invasive, and cost-effective severity assessment for early diagnosis and progression tracking. Despite advances in speech foundation models (SFMs), their cross-lingual generalization for PD severity multi-class classification remains underexplored due to limited labeled data, a lack of explainable methods and variability across languages and datasets. In this work, we evaluate pre-trained embeddings from four state-of-the-art open-source SFMs across three datasets in zero-shot and k-shot cross-lingual settings for multi-class PD severity assessment. Our results show that pre-trained speech embeddings enable meaningful cross-lingual transfer, although performance is sensitive to dataset properties, preprocessing, and adaptation strategy. Misclassifications under these conditions related to inter-speaker variability and atypical speech patterns highlight the need for more robust feature extraction and modeling for PD severity assessment while emphasizing the importance of explainability for reliable clinical insights.
#### All I Hear is Noise: Investigating Clever Hans Effects in Clinical Speech Datasets
 - **Authors:** Melanie Jouaiti, Ning Ma
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2609.21080

 - **Pdf link:** https://arxiv.org/pdf/2609.21080

 - **Abstract**
 Recent work revealed a striking Clever Hans effect in the Pitt dataset, where Alzheimer's detection achieved nearly 100% accuracy using only silent audio segments. This raises serious concerns about hidden confounding factors in speech-based health datasets. We systematically investigate whether similar biases exist across five widely used clinical speech corpora: DAIC-WoZ (depression), TORGO (dysarthria), Neurovoz (PD), MDVR-KCL (PD), and UCLASS (stuttering). For each dataset, we compare classification using the first second of audio, silent segments, and full recordings, and evaluate both raw and denoised signals. Across all datasets, silence-only classification frequently matched or exceeded full-audio performance, suggesting that classification performance may be influenced by dataset-specific confounds in addition to disorder-related speech characteristics. These findings question the reliability and generalisability of speech-based biomarkers and call for stricter methodological reporting, preprocessing transparency, and bias mitigation.
#### The Hidden Cost of Digits: Number Normalization and WER in ASR Systems
 - **Authors:** Stanisław Kacprzak, Mieszko Fraś
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Computation and Language (cs.CL)
 - **Arxiv link:** https://arxiv.org/abs/2609.21084

 - **Pdf link:** https://arxiv.org/pdf/2609.21084

 - **Abstract**
 Modern automatic speech recognition (ASR) systems trained on extremely large datasets can produce transcripts with numbers written in Arabic numerals. This creates a need for fair comparison with models that output verbatim texts and proper processing of reference transcripts. Popular approaches often reduce text normalization to lowercase and remove punctuation, with no additional normalization applied to languages other than English. In this work, we analyze the impact of normalization of numerical expressions in the evaluation of ASR systems in various languages, using Polish as an example of a highly inflective language. We perform experiments on VoxPopuli and The Polish Parliamentary speech datasets and estimate word error rate (WER) differences for different text normalization approaches. We show that the difference due to the lack of number normalization in WER may be substantial - more than 2 percentage points, and often higher than the differences between systems in popular multilingual benchmarks.
#### HAMMER: Harmonic-Aware Parallel Context Modeling and Discriminator-Free Perceptual Optimization for Speech Enhancement
 - **Authors:** Shang-Fu Chen, Szu-Wei Fu, Sung-Feng Huang, Rong Chao, Wen-Huang Cheng, Yu Tsao
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2609.21171

 - **Pdf link:** https://arxiv.org/pdf/2609.21171

 - **Abstract**
 Recent speech enhancement systems combine self-attention and Mamba to capture global interactions and long-range dependencies. Yet these hybrids usually operate as sequence mixers and do not explicitly exploit harmonic periodicity, a strong cue for preserving voiced speech under noise. Perceptual optimization poses another challenge. PESQ is non-differentiable, so many methods train auxiliary metric discriminators that increase complexity and introduce adversarial instability. We propose \ours, a harmonic-aware and discriminator-free speech enhancer built around two components. (i) The Time-Frequency Harmonic-aware Attention-Mamba (TF-HAM) block runs self-attention and bidirectional Mamba in parallel along both spectrogram axes, then applies a speech-adapted autocorrelation feed-forward network to encode local periodic structure. (ii) Metric-explicit perceptual refinement (MEPR) combines differentiable PESQ and log-likelihood-ratio losses to expose perceptual metric structure without a learned surrogate. On VoiceBank+DEMAND, \ours achieves 3.69 PESQ and 4.41 COVL with only 2.39\,M parameters, outperforming or matching discriminator-based systems. Inference-time perceptual contrast stretching further raises PESQ to 3.79 without retraining. The source code will be available at this https URL.
#### Rethinking Music Tokenization: A Semantic Codec toward High-Fidelity LLM Music Generation
 - **Authors:** Huakang Chen, Guobin Ma, Yuepeng Jiang, Dake Guo, Jingbin Hu, Hanke Xie, Wenhao Li, Lingxin Xiong, Jian Zhao, Zhonglin Jiang, Yong Chen, Lei Xie, Pengcheng Zhu
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2609.21240

 - **Pdf link:** https://arxiv.org/pdf/2609.21240

 - **Abstract**
 Discrete audio tokenization has become the critical interface between raw waveforms and autoregressive modeling in recent music generation. As a result, music tokenizers must simultaneously support high-fidelity reconstruction and produce discrete sequences that remain amenable to language modeling. Existing reconstruction-oriented tokenizers often mix musical structure with fine acoustic details, producing high-entropy tokens that are hard to model. In contrast, semantics-guided alternatives are designed for speech and do not fit music well, often hurting reconstruction quality. We address these trade-offs by rethinking music tokenization around a measurable notion of music semantic content grounded in downstream Music Information Retrieval tasks. Guided by this definition, we propose MuSeC, a music semantic codec that factorizes semantic and acoustic content directly from mixed signals without source separation. MuSeC preserves information required for high-fidelity reconstruction while producing more LM-friendly discrete units. Empirically, it improves reconstruction quality and yields more predictable token sequences, providing a practical foundation toward high-fidelity LLM music generation. Demos are available at this https URL.
#### CGaLore: Curvature-Guided GaLore for Memory-Efficient Continual Adaptation of ASR Foundation Models
 - **Authors:** Steven Vander Eeckt, Hugo Van hamme
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2609.21336

 - **Pdf link:** https://arxiv.org/pdf/2609.21336

 - **Abstract**
 Automatic speech recognition models suffer from catastrophic forgetting when adapted to new domains, accents, or downstream tasks. This problem becomes increasingly important with the growing use of speech foundation models, where adaptation should be both memory-efficient and safe, preserving the broad capabilities learned during pretraining. Gradient Low-Rank Projection (GaLore) has recently been proposed as a memory-efficient fine-tuning method that keeps model parameters full-rank while reducing the optimizer memory through low-rank gradient projection. However, GaLore does not account for catastrophic forgetting. We propose Curvature-Guided GaLore (CGaLore), which incorporates old-task curvature information when selecting the low-rank projection bases. Specifically, CGaLore filters current-task gradients using Kronecker-factored approximate curvature from previous tasks before computing the gradient subspace. Our experiments show that CGaLore enables effective adaptation while alleviating forgetting, outperforming state-of-the-art continual learning baselines. Extensive ablation studies confirm the practical applicability of CGaLore.
#### OmniVChat: Synthesizing, Benchmarking, and Training for Native Audio-Visual Dialogue
 - **Authors:** Haolin He, Yunfei Chu, Qi Chen, Wen Huang, Yuan Feng, Muzhi Zhu, Zheqi Dai, Haoning Xu, Dongchao Yang, Chunyat Wu, Zining Liang, Zhengxi Liu, Xiquan Li, Xie Chen, Xize Cheng, Qize Yang, Jin Xu, Qiuqiang Kong
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Artificial Intelligence (cs.AI); Image and Video Processing (eess.IV)
 - **Arxiv link:** https://arxiv.org/abs/2609.21465

 - **Pdf link:** https://arxiv.org/pdf/2609.21465

 - **Abstract**
 We define OmniVChat (Omni Video Chat) as the task of native audio-visual dialogue between a user and an omni model. In OmniVChat, omni models directly and simultaneously receive audio and video from a user and return text. The user's query is embedded in the audio and video, without a separate text question, external captioning, or speech recognition. Direct audio-visual input reduces external latency and computation while preserving perceptual cues. However, research on OmniVChat faces two constraints: data availability and evaluation. Recordings of people using their own devices are scarce. Furthermore, a good reply often needs to account for the user's surroundings, facial expressions, and nearby objects, and such responses can be expressed in many different ways, making keyword matching unreliable for evaluating reply quality. Recent progress in agent systems and video generation makes generation for comprehension viable, which means using synthesized dialogues for training and evaluation. Therefore, we present OmniVChat-Studio, a multi-agent data engine for synthesizing single- and multi-turn audio-visual dialogues. We use synthesized dialogues to build OmniVChat-Bench, an evaluation benchmark that evaluates omni models' basic dialogue abilities across five ability categories. We also present OmniVChat-RL, a reinforcement learning reward design that jointly targets reply correctness, efficiency, and style in OmniVChat. Training Qwen3-Omni-Instruct with OmniVChat-RL on synthesized dialogues improves its performance on both OmniVChat-Bench and the human-recorded OmniVChat-Bench-Human. These gains validate the reward design and show transfer to real-world dialogues in training and evaluation.
#### Towards Zero-Shot Attribution of Synthetic Speech via Audio-Text Contrastive Retrieval
 - **Authors:** Cristian-Teodor Neamtu, Serban Mihalache, Stefan Smeu, Dan Oneata, Horia Cucu, Dragos Burileanu
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2609.21581

 - **Pdf link:** https://arxiv.org/pdf/2609.21581

 - **Abstract**
 Audio deepfake forensics is moving beyond a simple real-or-fake verdict toward attribution: which system generated the audio clip? Most source-attribution methods cast this as closed-set classification, so they cannot name a generator that was absent from training, a gap that widens with every newly released text-to-speech (TTS) system. We instead frame attribution as cross-modal retrieval: each generator is described in natural language, and a clip is attributed by retrieving the description closest to it in a shared audio-text embedding space. Adding a new system then takes nothing more than writing its description, with no retraining and no new classifier head. Our model couples a frozen Wav2Vec2-BERT audio encoder with a frozen E5 text encoder and aligns them through small trainable projection heads, using a contrastive objective that combines a cross-modal supervised-contrastive loss with intra-modal terms. We evaluate on MLAAD v9 (140 TTS models, 51 languages) under 10-fold leave-models-out cross-validation. For generators it has never encountered before, the model reaches a model-level mean reciprocal rank (MRR) of 58.4%. Even when the correct model is not identified, the audio clip is often matched to systems that share the true generator's vocoder, acoustic model, or architecture. Because the same embedding space also answers natural-language attribute queries, one set of descriptions covers both open-set attribution and attribute-level forensic profiling.
#### The Spoken Wikipedia Presentation Corpus
 - **Authors:** Thomas Ranzenberger, Steffen Freisinger, Tobias Bocklet, Korbinian Riedhammer
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Computation and Language (cs.CL); Multimedia (cs.MM); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2609.21676

 - **Pdf link:** https://arxiv.org/pdf/2609.21676

 - **Abstract**
 We present the Spoken Wikipedia Presentation Corpus, an extension of the Spoken Wikipedia Corpora featuring LLM-generated slide decks for multimodal ASR. Slides are created from LLM-segmented sections using a hybrid pipeline that combines LLM-based content planning with rule-based design decisions. For each section, an LLM generates a slide title, bullet points, a takeaway message, and a visual description that is used to create an illustration. Rule-based matching then selects layouts, themes, and styles to produce the final slides. A vision LLM extracts slide text as Markdown. We evaluate multiple ASR and spoken language models (SLMs). The best model achieves an average micro-WER of 10.23% and an average micro-CER of 6.48% on audio-only inputs. English yields the lowest error rates, followed by German and Dutch, while performance declines across lower-resource languages. Although audio-only baselines are strong, multimodal zero-shot prompting of omni models remains challenging. The aligned slide, text, and audio data show a strong potential to improve recognition through cross-modal context.
#### BLINC: Blind Calibration For Training-Free Speech Enhancement Adaptation
 - **Authors:** Tobias Raichle, Ekaterina Gavrilko, Bin Yang
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2609.21898

 - **Pdf link:** https://arxiv.org/pdf/2609.21898

 - **Abstract**
 Speech enhancement (SE) models degrade under domain shifts and have to adapt to unseen target domains during deployment. Most existing test-time adaptation (TTA) methods for SE do so by adapting a subset of the model weights using a self-supervised loss, which requires backpropagation at test-time and permanently alters the model. We instead recalibrate the prediction itself and propose BLINC, a training-free TTA method that remaps the predicted time-frequency mask onto a bimodal target distribution by histogram matching. At test-time, the target distribution is parameterized from blind features of the noisy recording, so neither a reference distribution from a classical algorithm nor online metric optimization is involved. BLINC improves the overall quality of both evaluated SE models on almost every target condition and matches or exceeds the loss-based TTA baselines at minimal overhead.
#### Partial Accent-Control Editing in Frozen Speech Representations for Accent Conversion
 - **Authors:** Yangyang Qu, Michele Panariello, Massimiliano Todisco, Nicholas Evans
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2609.22031

 - **Pdf link:** https://arxiv.org/pdf/2609.22031

 - **Abstract**
 Accent conversion is the task of modifying a speech recording so that it sounds closer to a target accent while preserving linguistic content and other speaker-related characteristics. Most accent conversion systems use trained, generative models. Although they can induce target-accented speech, the strength of accent modification is not controllable at inference time, making it difficult to analyse how the strength of accent conversion affects source preservation. We propose Partial Accent-Control Editing (PACE), an accent conversion framework based upon the editing of frozen WavLM representations without the training of an accent-conditioned generator. A constrained edit is first applied to source WavLM features, which are then fused with target-accent reference features retrieved from non-parallel accent examples. Fusion weights are used to control the trade-off between accent conversion strength and the degradation of other source attributes. Using a suite of five metrics, and with the cost of accent-entangled speaker similarity, we show that PACE is substantially superior to a pair of competitive baselines in terms of both accent conversion and source preservation.
#### Voice-Light: A Full-Duplex Cascaded Voice Agent with Causal Turn-Taking and Speculative Generation
 - **Authors:** Bertil Braun
 - **Subjects:** Subjects:
Sound (cs.SD); Computation and Language (cs.CL); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2609.20995

 - **Pdf link:** https://arxiv.org/pdf/2609.20995

 - **Abstract**
 Natural spoken interaction requires more than streaming ASR, language generation, and speech synthesis: a system must react to overlap without canceling on every acknowledgment, prepare a response before a turn is certain, and ensure canceled audio cannot enter conversation history. We present Voice-Light, a full-duplex cascaded voice agent that combines immediate acoustic onset, a causal adapter sharing a streaming ASR encoder, reversible playback control, and private speculative response generation. Structured tool calls execute concurrently with audible bridge speech, while browser acknowledgments make rendered audio authoritative for durable history. Locked evaluation on 1,673 real-conversation silence candidates found that an earlier learned completion checkpoint preserved a 2.70% false-cutoff rate but reached only 12.53% end-of-turn recall, compared with 95.60% for a Silero timing policy. The deployed system therefore retains a hybrid controller rather than claiming a learned-policy replacement. Across three unscripted operator-run microphone sessions, 36 measured response turns had a 758 ms median from final VAD endpoint to first server audio; 21 turns were below 800 ms. These sessions are an instrumented case study, not a controlled user evaluation. We release the synthetic data, model artifacts, evaluation code and summaries, source code, and deployment configuration supporting the result.
#### Reusing Latent Speech Representations for Query-Conditioned Topic Localization in Transcripts
 - **Authors:** Steffen Freisinger, Philipp Seeberger, Thomas Ranzenberger, Tobias Bocklet, Korbinian Riedhammer
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2609.21844

 - **Pdf link:** https://arxiv.org/pdf/2609.21844

 - **Abstract**
 Long transcripts are costly inputs for downstream NLP systems and often contain irrelevant context. We study query-conditioned topic localization: predicting the sentence span in a transcript that best addresses a topic-title query. To improve span localization, we reuse ASR encoder states as sentence-level representations and fuse them with textual embeddings. This lets lightweight span locators exploit speech information without running a separate audio encoder. Experiments on two public datasets show consistent gains over text-only baselines, especially under strict boundary-matching criteria. Cross-dataset experiments further indicate that the benefits are strongest for structured or semi-structured speech, while gains on spontaneous speech are limited and mixed.


by Zyzzyva0381 (Windy). 


2026-09-21
