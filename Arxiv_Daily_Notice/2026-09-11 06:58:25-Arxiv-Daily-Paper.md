# Showing new listings for Friday, 11 September 2026
Auto update papers at about 2:30am UTC (10:30am Beijing time) every weekday.


阅读 `Usage.md`了解如何使用此repo实现个性化的Arxiv论文推送

See `Usage.md` for instructions on how to personalize the repo. 


Keyword list: ['text-to-speech', 'text to speech', 'tts', 'LLM-based', 'speech', 'voice']


Excluded: []


### Today: 10papers 
#### From Metrics to Natural Dialogue: French Full-Duplex Benchmark for Spoken Dialogue Models
 - **Authors:** Hamid Soltani, Gilles Boulianne (Luqia Technologies)
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2609.10765

 - **Pdf link:** https://arxiv.org/pdf/2609.10765

 - **Abstract**
 Full-duplex spoken dialogue models aim to make voice agents more natural by allowing them to listen, speak, pause, and respond during ongoing conversation. However, it is not clear whether full-duplex benchmarks behave the same way when models are evaluated in a different language. To investigate this, we introduce a French full-duplex benchmark (FDB) with two variants, CALLFC-FDB for Canadian French and MEDIA-FDB for European French, and compare them with an English FDB. Built from real spoken resources, these benchmarks evaluate key full-duplex skills, including pause handling, turn-taking, backchannels, and interruptions. Beyond introducing French FDBs, we evaluate human--human conversations with FDB metrics to better understand the values these metrics take in real-world dialogue. Our analysis reveals that most timing-based metrics behave similarly across languages, while content-based evaluation degrades under language mismatch. We also find a trade-off between optimizing benchmark metrics and preserving conversational naturalness.
#### Less can be More: What Aspects of Speech Drive End-of-Turn Detection
 - **Authors:** Rini Sharon, Manickavela A, Kadri Hacioglu, Andreas Stolcke
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Artificial Intelligence (cs.AI); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2609.11066

 - **Pdf link:** https://arxiv.org/pdf/2609.11066

 - **Abstract**
 In conversational AI, detecting when a speaker has finished talking is crucial for natural turn taking. While recent work incorporates semantics, the relative contribution of different modalities remains unclear. We present a controlled ablation of acoustic, prosodic, and semantic signals for streaming end of turn detection using a lightweight trimodal classifier. Under identical training conditions, the acoustic prosodic combination achieves the best balance of accuracy and latency, achieving utterance F1 of 0.93 with 7.8% false alarms at 400ms median latency. Adding text increases premature detections without improving performance. Feature space analysis confirms that prosodic features have the strongest class separability, while text representations overlap substantially. These findings suggest that turn-taking is primarily conveyed through intonation and silence patterns rather than semantic completeness, enabling faster and more reliable systems without expensive text inference.
#### Downstream-Task-Aware Unified Source Separation
 - **Authors:** Yoshiki Mitsui, Ryo Aihara, Tatsuhiko Saito, Yoshiki Masuyama, Christoph Boeddeker, Julius Richter, Gordon Wichern, Jonathan Le Roux
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2609.11092

 - **Pdf link:** https://arxiv.org/pdf/2609.11092

 - **Abstract**
 Task-aware unified source separation (TUSS) enables a single model to handle diverse separation tasks by conditioning on input prompts. However, conventional TUSS does not account for downstream task requirements, such as whether the enhanced speech will be used for human listening or automatic speech recognition (ASR). In this paper, we propose a prompt extension framework for TUSS that incorporates downstream task information into the input prompts and switches the loss function according to the given prompt during training, enabling outputs with different signal characteristics at inference time. Specifically, we introduce an ASR-dedicated prompt paired with a regularized loss function that reduces speech artifacts to improve ASR robustness, while the standard prompt is paired with the conventional SNR loss function. Experiments on the LibriSpeech and JNAS corpora demonstrate that the proposed joint-training scheme enables a single model to improve ASR performance over noisy input across a wide range of SNR conditions by selecting the ASR-dedicated prompt, while maintaining general speech enhancement quality when the standard prompt is used.
#### Low-Latency State Space Voice Activity Detection with Robust Onset Time Evaluation
 - **Authors:** Elad Cohen, Arnon Netzer, Hai Victor Habi
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2609.11110

 - **Pdf link:** https://arxiv.org/pdf/2609.11110

 - **Abstract**
 Voice Activity Detection (VAD) systems are commonly evaluated using metrics such as the area under the receiver operating characteristic curve (AUROC), but these metrics do not account for temporal responsiveness. For low-latency applications, however, accurately measuring speech onset delay is essential. This is particularly challenging because onset latency evaluation is affected by noise and systematic misalignment in annotation timestamps. In this work, we introduce a probabilistic framework for evaluating VAD onset time under noisy temporal labels. We model annotated onset times as noisy observations of latent acoustic onsets and estimate the resulting discrepancy distribution from data. We show that this approach provides a more stable and robust estimate of algorithmic latency. In addition, we introduce S4VAD, the first state-space-model-based (SSM-based) VAD architecture. S4VAD supports efficient streaming inference while directly controlling the decay of past acoustic evidence, enabling fast and responsive speech-onset detection. We evaluate VAD onset latency across several low-latency architectures, including CNN, Transformer, and RNN models. Our proposed VAD achieves the lowest latency while maintaining a competitive AUROC.
#### Domain-Incremental Learning for Multi-Channel Replay Speech Detection
 - **Authors:** Michael Neri
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Cryptography and Security (cs.CR); Sound (cs.SD); Signal Processing (eess.SP)
 - **Arxiv link:** https://arxiv.org/abs/2609.11194

 - **Pdf link:** https://arxiv.org/pdf/2609.11194

 - **Abstract**
 Replay attacks are the most accessible threat to voice-controlled systems, and the acoustic cues that expose them are strongly modulated by the environment in which the attack is mounted. A detector deployed in the field therefore has to absorb new acoustic conditions over time, ideally without revisiting past recordings, since retaining speech indefinitely is both expensive and legally constrained. We frame this as Domain-Incremental Learning (DIL) over acoustic environments and present the first continual learning benchmark for multi-channel replay speech detection, evaluating a state-of-the-art beamformer-based detector over all 24 environment orderings of the ReMASC corpus with five seeds. Sequential fine-tuning forgets severely, raising the error rate on previously learned environments by 18.8 points. Elastic weight consolidation (EWC) halves forgetting but loses plasticity, gradient projection memory (GPM) is statistically indistinguishable from naive fine-tuning, and the proposed task-specific beamformer (TSB) that keeps one spatial front-end per environment significantly improves final and incremental accuracy. We further show that the last environment of the sequence dominates final performance. Code, results, and analysis are available at this https URL.
#### Preference Optimization with LALM Feedback for Continuous Autoregressive Non-Verbal Vocalization Generation
 - **Authors:** Jingbin Hu, Qirui Zhan, Yuang Cao, Ziyu Zhang, Yunxiang Chen, Houdun Liu, Shuo Feng, Bengu Wu, Lei Xie, Liumeng Xue
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2609.11260

 - **Pdf link:** https://arxiv.org/pdf/2609.11260

 - **Abstract**
 We propose a preference optimization framework with Large Audio-Language Model (LALM) feedback for controllable non-verbal vocalization (NVV) generation in continuous autoregressive speech models. To construct preference data without human preference annotation, we build a bilingual prompt corpus by combining NVV-injected real transcripts with LLM-generated semantically aligned prompts, perform stochastic model rollouts, and use a LALM to rank candidate utterances and form same-prompt chosen--rejected pairs. We then adopt a two-stage optimization strategy: Rejection Sampling Fine-Tuning (RSFT) first adapts the model to LALM-selected high-scoring samples, followed by Anchored Flow-DPO, which formulates pairwise preference optimization using utterance-level flow-matching loss and retains the chosen-sample flow-matching objective as an SFT anchor. This design enables DPO-style preference learning without explicit sequence likelihoods while preserving direct supervision on preferred realizations. On the official 1,600-utterance NVVSpeech Challenge Track~2 test set, our method achieves a Final Track2Score of \textbf{75.80} (79.39 ZH / 72.21 EN), outperforming the VoxCPM2 baseline by \textbf{+1.84}. The improvements are mainly driven by higher NVV Accuracy and NVV Perceptual Effect, while Overall Quality remains stable.
#### Not All Attacks Are Learned Equally in Speech Deepfake Detection
 - **Authors:** Avantika Singh, Aurosweta Mahapatra, Ismail Rasim Ulgen, Nicholas Andrews, Kong Aik Lee, Berrak Sisman
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2609.11763

 - **Pdf link:** https://arxiv.org/pdf/2609.11763

 - **Abstract**
 Speech deepfake detection (SDD) models are trained on multi-attack datasets containing diverse spoofing systems, such as text-to-speech (TTS) and voice conversion (VC). In standard classifier training on multi-attack datasets, all attacks are treated as one spoofed class, and performance is reported using overall Equal Error Rate (EER). This aggregate view obscures how individual attacks shape learning and generalization. To better understand this attack-level behavior, we first balance TTS and VC exposure using sample and attack omission. We then measure attack-wise EER at inference and analyze attack-wise training loss and predictive entropy to characterize optimization. Results show that attacks contribute unequally: some attacks have high EER sensitivity and concentrated entropy with low loss, indicating strong influence on the decision boundary. We define these as high-impact attacks. To reduce uneven generalization across attacks, we propose a replay-regularized, attack-aware curriculum that steps exposure based on measured attack influence. Experiments on ASVspoof 2019, 2021, ASVspoof 5, and Fake-or-Real show improved overall robustness and reduced attack-level imbalance compared with standard multi-attack training.
#### Whisper-Based Speech Transcription from Videos Across Multiple Languages for Cross-Cultural Understanding
 - **Authors:** Michael Picheny
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Computation and Language (cs.CL)
 - **Arxiv link:** https://arxiv.org/abs/2609.11772

 - **Pdf link:** https://arxiv.org/pdf/2609.11772

 - **Abstract**
 Cross-cultural understanding has become increasingly important in today's highly connected, cross-national world. The success of LLM-based technologies is now driving the development of automated tools to aid understanding for nonnative people trying to succeed in cross-cultural environments. Building such automated tools is often done by leveraging in-thewild text, audio, and video data. This paper presents techniques for improving speech recognition-based transcript creation in multiple languages from videos to better train these automated tools. The focus is on processes and speech tools that can easily be used by cross-cultural tool builders without requiring deep speech processing expertise. Using publicly available videos from YouTube and Whisper-based tools, average transcription error rate across seven languages (Spanish, Japanese, Korean, Mandarin, Turkish, Russian, and Hebrew) of 30% are observed. With a modest amount of fine-tuning data, the average error rate can be reduced to 20% making such output much more usable for downstream processing. Speech and metadata associated with these videos that can be used by the community to further refine these experiments are released as well.
#### RetroThinker: Enabling Retrospective Thinking in Speech LLMs
 - **Authors:** Yi-Jen Shih, Puyuan Peng, Abdelrahman Mohamed, David Harwath
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Artificial Intelligence (cs.AI); Computation and Language (cs.CL)
 - **Arxiv link:** https://arxiv.org/abs/2609.11864

 - **Pdf link:** https://arxiv.org/pdf/2609.11864

 - **Abstract**
 Speech large language models (SpeechLLMs) offer reduced latency and retain paralinguistic nuances that are typically lost in cascaded automatic speech recognition (ASR) and text-based LM architectures. However, they continue to lag behind text-only LLMs on complex reasoning tasks, while real-time spoken interaction imposes strict latency constraints. Although prior works employ Chain-of-Thought (CoT) and concurrent reasoning to enhance reasoning capabilities without inducing prohibitive delays, an inherent accuracy-latency trade-off persists. In this paper, we investigate whether a streaming SpeechLLM can dynamically revise its reasoning traces on the fly. We introduce RetroThinker, a multi-stage post-training framework that equips the Moshi model to self-verify and forward-correct CoT steps during inference. RetroThinker combines supervised fine-tuning (SFT) on curated retrospective thinking data with length-based direct preference optimization (DPO) to optimize retrospective during early reasoning (i.e., reasoning concurrently while the user speaks). Evaluated on the GSM8K benchmark, RetroThinker significantly improves the accuracy-latency trade-off over non-retrospective baselines, achieving an 11% absolute accuracy gain at a comparable latency.
#### Xiaomi-CocktailASR-1 Technical Report
 - **Authors:** Yiru Zhang, Hang Su, Lichun Fan, Ying Zeng, Chang Liu, Yifeng Wang, Yuquan Liang, Tao Li, Lian Li, Wenhao Yang, Jian Luan, Cong Zou, Heng Qu
 - **Subjects:** Subjects:
Sound (cs.SD); Computation and Language (cs.CL); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2609.11274

 - **Pdf link:** https://arxiv.org/pdf/2609.11274

 - **Abstract**
 Recently, large language model (LLM) based ASR models have achieved significant progress, yet they generally lack support for multi-speaker scenarios, where the cocktail party problem remains a critical bottleneck for further advancing ASR. Existing TS-ASR methods, including end-to-end architectures with speaker embeddings and latest LLM-based explorations suffer from degraded single-speaker performance and the inability to reject when the target speaker is absent. In this paper, we propose Xiaomi-CocktailASR-1, an LLM-based end-to-end TS-ASR architecture. By utilizing reference speech as voiceprint prompts, it directly transcribes the target speaker's speech without requiring speech separation. Xiaomi-CocktailASR-1 maintains competitive performance in single-speaker scenarios, comparable to mainstream ASR models. It also features a negative sample rejection capability, outputting empty text when the target speaker is absent from the mixed speech. Additionally, Xiaomi-CocktailASR-1 supports a Chain-of-Thought (CoT) reasoning mode to provide explicit reasoning steps. Extensive experiments on various synthetic and real-world multispeaker benchmarks demonstrate that Xiaomi-CocktailASR-1 achieves state-of-the-art performance, effectively addressing the cocktail party problem through a unified architecture that balances multispeaker and single-speaker recognition accuracy, along with rejection capability.


by Zyzzyva0381 (Windy). 


2026-09-11
