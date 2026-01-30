# Showing new listings for Friday, 30 January 2026
Auto update papers at about 2:30am UTC (10:30am Beijing time) every weekday.


阅读 `Usage.md`了解如何使用此repo实现个性化的Arxiv论文推送

See `Usage.md` for instructions on how to personalize the repo. 


Keyword list: ['text-to-speech', 'text to speech', 'tts', 'LLM-based', 'speech', 'voice']


Excluded: []


### Today: 15papers 
#### Reducing Prompt Sensitivity in LLM-based Speech Recognition Through Learnable Projection
 - **Authors:** Sergio Burdisso, Esaú Villatoro-Tello, Shashi Kumar, Srikanth Madikeri, Andrés Carofilis, Pradeep Rangappa, Manjunath K E, Kadri Hacioglu, Petr Motlicek, Andreas Stolcke
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Computation and Language (cs.CL); Machine Learning (cs.LG)
 - **Arxiv link:** https://arxiv.org/abs/2601.20898

 - **Pdf link:** https://arxiv.org/pdf/2601.20898

 - **Abstract**
 LLM-based automatic speech recognition (ASR), a well-established approach, connects speech foundation models to large language models (LLMs) through a speech-to-LLM projector, yielding promising results. A common design choice in these architectures is the use of a fixed, manually defined prompt during both training and inference. This setup not only enables applicability across a range of practical scenarios, but also helps maximize model performance. However, the impact of prompt design remains underexplored. This paper presents a comprehensive analysis of commonly used prompts across diverse datasets, showing that prompt choice significantly affects ASR performance and introduces instability, with no single prompt performing best across all cases. Inspired by the speech-to-LLM projector, we propose a prompt projector module, a simple, model-agnostic extension that learns to project prompt embeddings to more effective regions of the LLM input space, without modifying the underlying LLM-based ASR model. Experiments on four datasets show that the addition of a prompt projector consistently improves performance, reduces variability, and outperforms the best manually selected prompts.
#### Unseen but not Unknown: Using Dataset Concealment to Robustly Evaluate Speech Quality Estimation Models
 - **Authors:** Jaden Pieper, Stephen D. Voran
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2601.21110

 - **Pdf link:** https://arxiv.org/pdf/2601.21110

 - **Abstract**
 We introduce Dataset Concealment (DSC), a rigorous new procedure for evaluating and interpreting objective speech quality estimation models. DSC quantifies and decomposes the performance gap between research results and real-world application requirements, while offering context and additional insights into model behavior and dataset characteristics. We also show the benefits of addressing the corpus effect by using the dataset Aligner from AlignNet when training models with multiple datasets. We demonstrate DSC and the improvements from the Aligner using nine training datasets and nine unseen datasets with three well-studied models: MOSNet, NISQA, and a Wav2Vec2.0-based model. DSC provides interpretable views of the generalization capabilities and limitations of models, while allowing all available data to be used at training. An additional result is that adding the 1000 parameter dataset Aligner to the 94 million parameter Wav2Vec model during training does significantly improve the resulting model's ability to estimate speech quality for unseen data.
#### DNN-Based Online Source Counting Based on Spatial Generalized Magnitude Squared Coherence
 - **Authors:** Henri Gode, Simon Doclo
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2601.21114

 - **Pdf link:** https://arxiv.org/pdf/2601.21114

 - **Abstract**
 The number of active sound sources is a key parameter in many acoustic signal processing tasks, such as source localization, source separation, and multi-microphone speech enhancement. This paper proposes a novel method for online source counting by detecting changes in the number of active sources based on spatial coherence. The proposed method exploits the fact that a single coherent source in spatially white background noise yields high spatial coherence, whereas only noise results in low spatial coherence. By applying a spatial whitening operation, the source counting problem is reformulated as a change detection task, aiming to identify the time frames when the number of active sources changes. The method leverages the generalized magnitude-squared coherence as a measure to quantify spatial coherence, providing features for a compact neural network trained to detect source count changes framewise. Simulation results with binaural hearing aids in reverberant acoustic scenes with up to 4 speakers and background noise demonstrate the effectiveness of the proposed method for online source counting.
#### Towards Robust Dysarthric Speech Recognition: LLM-Agent Post-ASR Correction Beyond WER
 - **Authors:** Xiuwen Zheng, Sixun Dong, Bornali Phukon, Mark Hasegawa-Johnson, Chang D. Yoo
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2601.21347

 - **Pdf link:** https://arxiv.org/pdf/2601.21347

 - **Abstract**
 While Automatic Speech Recognition (ASR) is typically benchmarked by word error rate (WER), real-world applications ultimately hinge on semantic fidelity. This mismatch is particularly problematic for dysarthric speech, where articulatory imprecision and disfluencies can cause severe semantic distortions. To bridge this gap, we introduce a Large Language Model (LLM)-based agent for post-ASR correction: a Judge-Editor over the top-k ASR hypotheses that keeps high-confidence spans, rewrites uncertain segments, and operates in both zero-shot and fine-tuned modes. In parallel, we release SAP-Hypo5, the largest benchmark for dysarthric speech correction, to enable reproducibility and future exploration. Under multi-perspective evaluation, our agent achieves a 14.51% WER reduction alongside substantial semantic gains, including a +7.59 pp improvement in MENLI and +7.66 pp in Slot Micro F1 on challenging samples. Our analysis further reveals that WER is highly sensitive to domain shift, whereas semantic metrics correlate more closely with downstream task performance.
#### Speech Quality-Based Localization of Low-Quality Speech and Text-to-Speech Synthesis Artefacts
 - **Authors:** Michael Kuhlmann, Alexander Werning, Thilo von Neumann, Reinhold Haeb-Umbach
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2601.21886

 - **Pdf link:** https://arxiv.org/pdf/2601.21886

 - **Abstract**
 A large number of works view the automatic assessment of speech from an utterance- or system-level perspective. While such approaches are good in judging overall quality, they cannot adequately explain why a certain score was assigned to an utterance. frame-level scores can provide better interpretability, but models predicting them are harder to tune and regularize since no strong targets are available during training. In this work, we show that utterance-level speech quality predictors can be regularized with a segment-based consistency constraint which notably reduces frame-level stochasticity. We then demonstrate two applications involving frame-level scores: The partial spoof scenario and the detection of synthesis artefacts in two state-of-the-art text-to-speech systems. For the latter, we perform listening tests and confirm that listeners rate segments to be of poor quality more often in the set defined by low frame-level scores than in a random control set.
#### DisContSE: Single-Step Diffusion Speech Enhancement Based on Joint Discrete and Continuous Embeddings
 - **Authors:** Yihui Fu, Tim Fingscheidt
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2601.21940

 - **Pdf link:** https://arxiv.org/pdf/2601.21940

 - **Abstract**
 Diffusion speech enhancement on discrete audio codec features gain immense attention due to their improved speech component reconstruction capability. However, they usually suffer from high inference computational complexity due to multiple reverse process iterations. Furthermore, they generally achieve promising results on non-intrusive metrics but show poor performance on intrusive metrics, as they may struggle in reconstructing the correct phones. In this paper, we propose DisContSE, an efficient diffusion-based speech enhancement model on joint discrete codec tokens and continuous embeddings. Our contributions are three-fold. First, we formulate both a discrete and a continuous enhancement module operating on discrete audio codec tokens and continuous embeddings, respectively, to achieve improved fidelity and intelligibility simultaneously. Second, a semantic enhancement module is further adopted to achieve optimal phonetic accuracy. Third, we achieve a single-step efficient reverse process in inference with a novel quantization error mask initialization strategy, which, according to our knowledge, is the first successful single-step diffusion speech enhancement based on an audio codec. Trained and evaluated on URGENT 2024 Speech Enhancement Challenge data splits, the proposed DisContSE excels top-reported time- and frequency-domain diffusion baseline methods in PESQ, POLQA, UTMOS, and in a subjective ITU-T P.808 listening test, clearly achieving an overall top rank.
#### TidyVoice 2026 Challenge Evaluation Plan
 - **Authors:** Aref Farhadipour, Jan Marquenie, Srikanth Madikeri, Teodora Vukovic, Volker Dellwo, Kathy Reid, Francis M. Tyers, Ingo Siegert, Eleanor Chodroff
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2601.21960

 - **Pdf link:** https://arxiv.org/pdf/2601.21960

 - **Abstract**
 The performance of speaker verification systems degrades significantly under language mismatch, a critical challenge exacerbated by the field's reliance on English-centric data. To address this, we propose the TidyVoice Challenge for cross-lingual speaker verification. The challenge leverages the TidyVoiceX dataset from the novel TidyVoice benchmark, a large-scale, multilingual corpus derived from Mozilla Common Voice, and specifically curated to isolate the effect of language switching across approximately 40 languages. Participants will be tasked with building systems robust to this mismatch, with performance primarily evaluated using the Equal Error Rate on cross-language trials. By providing standardized data, open-source baselines, and a rigorous evaluation protocol, this challenge aims to drive research towards fairer, more inclusive, and language-independent speaker recognition technologies, directly aligning with the Interspeech 2026 theme, "Speaking Together."
#### VoxMorph: Scalable Zero-shot Voice Identity Morphing via Disentangled Embeddings
 - **Authors:** Bharath Krishnamurthy, Ajita Rattani
 - **Subjects:** Subjects:
Sound (cs.SD); Cryptography and Security (cs.CR); Machine Learning (cs.LG); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2601.20883

 - **Pdf link:** https://arxiv.org/pdf/2601.20883

 - **Abstract**
 Morphing techniques generate artificial biometric samples that combine features from multiple individuals, allowing each contributor to be verified against a single enrolled template. While extensively studied in face recognition, this vulnerability remains largely unexplored in voice biometrics. Prior work on voice morphing is computationally expensive, non-scalable, and limited to acoustically similar identity pairs, constraining practical deployment. Moreover, existing sound-morphing methods target audio textures, music, or environmental sounds and are not transferable to voice identity manipulation. We propose VoxMorph, a zero-shot framework that produces high-fidelity voice morphs from as little as five seconds of audio per subject without model retraining. Our method disentangles vocal traits into prosody and timbre embeddings, enabling fine-grained interpolation of speaking style and identity. These embeddings are fused via Spherical Linear Interpolation (Slerp) and synthesized using an autoregressive language model coupled with a Conditional Flow Matching network. VoxMorph achieves state-of-the-art performance, delivering a 2.6x gain in audio quality, a 73% reduction in intelligibility errors, and a 67.8% morphing attack success rate on automated speaker verification systems under strict security thresholds. This work establishes a practical and scalable paradigm for voice morphing with significant implications for biometric security. The code and dataset are available on our project page: this https URL
#### SW-ASR: A Context-Aware Hybrid ASR Pipeline for Robust Single Word Speech Recognition
 - **Authors:** Manali Sharma (1), Riya Naik (1), Buvaneshwari G (1) ((1) Tetranetics Private Limited)
 - **Subjects:** Subjects:
Sound (cs.SD); Computation and Language (cs.CL); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2601.20890

 - **Pdf link:** https://arxiv.org/pdf/2601.20890

 - **Abstract**
 Single-word Automatic Speech Recognition (ASR) is a challenging task due to the lack of linguistic context and sensitivity to noise, pronunciation variation, and channel artifacts, especially in low-resource, communication-critical domains such as healthcare and emergency response. This paper reviews recent deep learning approaches and proposes a modular framework for robust single-word detection. The system combines denoising and normalization with a hybrid ASR front end (Whisper + Vosk) and a verification layer designed to handle out-of-vocabulary words and degraded audio. The verification layer supports multiple matching strategies, including embedding similarity, edit distance, and LLM-based matching with optional contextual guidance. We evaluate the framework on the Google Speech Commands dataset and a curated real-world dataset collected from telephony and messaging platforms under bandwidth-limited conditions. Results show that while the hybrid ASR front end performs well on clean audio, the verification layer significantly improves accuracy on noisy and compressed channels. Context-guided and LLM-based matching yield the largest gains, demonstrating that lightweight verification and context mechanisms can substantially improve single-word ASR robustness without sacrificing latency required for real-time telephony applications.
#### A Study of Data Selection Strategies for Pre-training Self-Supervised Speech Models
 - **Authors:** Ryan Whetten, Titouan Parcollet, Marco Dinarelli, Yannick Estève
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2601.20896

 - **Pdf link:** https://arxiv.org/pdf/2601.20896

 - **Abstract**
 Self-supervised learning (SSL) has transformed speech processing, yet its reliance on massive pre-training datasets remains a bottleneck. While robustness is often attributed to scale and diversity, the role of the data distribution is less understood. We systematically examine how curated subsets of pre-training data influence Automatic Speech Recognition (ASR) performance. Surprisingly, optimizing for acoustic, speaker, or linguistic diversity yields no clear improvements over random sampling. Instead, we find that prioritizing the longest utterances achieves superior ASR results while using only half the original dataset, reducing pre-training time by 24% on a large corpora. These findings suggest that for pre-training speech SSL models, data length is a more critical factor than either data diversity or overall data quantity for performance and efficiency, offering a new perspective for data selection strategies in SSL speech processing.
#### Text-only adaptation in LLM-based ASR through text denoising
 - **Authors:** Sergio Burdisso, Esaú Villatoro-Tello, Andrés Carofilis, Shashi Kumar, Kadri Hacioglu, Srikanth Madikeri, Pradeep Rangappa, Manjunath K E, Petr Motlicek, Shankar Venkatesan, Andreas Stolcke
 - **Subjects:** Subjects:
Sound (cs.SD); Computation and Language (cs.CL); Machine Learning (cs.LG); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2601.20900

 - **Pdf link:** https://arxiv.org/pdf/2601.20900

 - **Abstract**
 Adapting automatic speech recognition (ASR) systems based on large language models (LLMs) to new domains using text-only data is a significant yet underexplored challenge. Standard fine-tuning of the LLM on target-domain text often disrupts the critical alignment between speech and text modalities learned by the projector, degrading performance. We introduce a novel text-only adaptation method that emulates the audio projection task by treating it as a text denoising task. Our approach thus trains the LLM to recover clean transcripts from noisy inputs. This process effectively adapts the model to a target domain while preserving cross-modal alignment. Our solution is lightweight, requiring no architectural changes or additional parameters. Extensive evaluation on two datasets demonstrates up to 22.1% relative improvement, outperforming recent state-of-the-art text-only adaptation methods.
#### asr_eval: Algorithms and tools for multi-reference and streaming speech recognition evaluation
 - **Authors:** Oleg Sedukhin, Andrey Kostin
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2601.20992

 - **Pdf link:** https://arxiv.org/pdf/2601.20992

 - **Abstract**
 We propose several improvements to the speech recognition evaluation. First, we propose a string alignment algorithm that supports both multi-reference labeling, arbitrary-length insertions and better word alignment. This is especially useful for non-Latin languages, those with rich word formation, to label cluttered or longform speech. Secondly, we collect a novel test set DiverseSpeech-Ru of longform in-the-wild Russian speech with careful multi-reference labeling. We also perform multi-reference relabeling of popular Russian tests set and study fine-tuning dynamics on its corresponding train set. We demonstrate that the model often adopts to dataset-specific labeling, causing an illusion of metric improvement. Based on the improved word alignment, we develop tools to evaluate streaming speech recognition and to align multiple transcriptions to compare them visually. Additionally, we provide uniform wrappers for many offline and streaming speech recognition models. Our code will be made publicly available.
#### Position-invariant Fine-tuning of Speech Enhancement Models with Self-supervised Speech Representations
 - **Authors:** Amit Meghanani, Thomas Hain
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2601.21084

 - **Pdf link:** https://arxiv.org/pdf/2601.21084

 - **Abstract**
 Integrating front-end speech enhancement (SE) models with self-supervised learning (SSL)-based speech models is effective for downstream tasks in noisy conditions. SE models are commonly fine-tuned using SSL representations with mean squared error (MSE) loss between enhanced and clean speech. However, MSE is prone to exploiting positional embeddings in SSL models, allowing the objective to be minimised through positional correlations instead of content-related information. This work frames the problem as a general limitation of self-supervised representation fine-tuning and investigates it through representation-guided SE. Two strategies are considered: (1) zero-padding, previously explored in SSL pre-training but here examined in the fine-tuning setting, and (2) speed perturbations with a soft-DTW loss. Experiments show that the soft-DTW-based approach achieves faster convergence and improved downstream performance, underscoring the importance of position-invariant fine-tuning in SSL-based speech modelling.
#### Multilingual Dysarthric Speech Assessment Using Universal Phone Recognition and Language-Specific Phonemic Contrast Modeling
 - **Authors:** Eunjung Yeo, Julie M. Liss, Visar Berisha, David R. Mortensen
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2601.21205

 - **Pdf link:** https://arxiv.org/pdf/2601.21205

 - **Abstract**
 The growing prevalence of neurological disorders associated with dysarthria motivates the need for automated intelligibility assessment methods that are applicalbe across languages. However, most existing approaches are either limited to a single language or fail to capture language-specific factors shaping intelligibility. We present a multilingual phoneme-production assessment framework that integrates universal phone recognition with language-specific phoneme interpretation using contrastive phonological feature distances for phone-to-phoneme mapping and sequence alignment. The framework yields three metrics: phoneme error rate (PER), phonological feature error rate (PFER), and a newly proposed alignment-free measure, phoneme coverage (PhonCov). Analysis on English, Spanish, Italian, and Tamil show that PER benefits from the combination of mapping and alignment, PFER from alignment alone, and PhonCov from mapping. Further analyses demonstrate that the proposed framework captures clinically meaningful patterns of intelligibility degradation consistent with established observations of dysarthric speech.
#### Qwen3-ASR Technical Report
 - **Authors:** Xian Shi, Xiong Wang, Zhifang Guo, Yongqi Wang, Pei Zhang, Xinyu Zhang, Zishan Guo, Hongkun Hao, Yu Xi, Baosong Yang, Jin Xu, Jingren Zhou, Junyang Lin
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2601.21337

 - **Pdf link:** https://arxiv.org/pdf/2601.21337

 - **Abstract**
 In this report, we introduce Qwen3-ASR family, which includes two powerful all-in-one speech recognition models and a novel non-autoregressive speech forced alignment model. Qwen3-ASR-1.7B and Qwen3-ASR-0.6B are ASR models that support language identification and ASR for 52 languages and dialects. Both of them leverage large-scale speech training data and the strong audio understanding ability of their foundation model Qwen3-Omni. We conduct comprehensive internal evaluation besides the open-sourced benchmarks as ASR models might differ little on open-sourced benchmark scores but exhibit significant quality differences in real-world scenarios. The experiments reveal that the 1.7B version achieves SOTA performance among open-sourced ASR models and is competitive with the strongest proprietary APIs while the 0.6B version offers the best accuracy-efficiency trade-off. Qwen3-ASR-0.6B can achieve an average TTFT as low as 92ms and transcribe 2000 seconds speech in 1 second at a concurrency of 128. Qwen3-ForcedAligner-0.6B is an LLM based NAR timestamp predictor that is able to align text-speech pairs in 11 languages. Timestamp accuracy experiments show that the proposed model outperforms the three strongest force alignment models and takes more advantages in efficiency and versatility. To further accelerate the community research of ASR and audio understanding, we release these models under the Apache 2.0 license.


by Zyzzyva0381 (Windy). 


2026-01-30
