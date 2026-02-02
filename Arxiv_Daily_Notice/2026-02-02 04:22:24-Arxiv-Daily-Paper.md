# Showing new listings for Monday, 2 February 2026
Auto update papers at about 2:30am UTC (10:30am Beijing time) every weekday.


阅读 `Usage.md`了解如何使用此repo实现个性化的Arxiv论文推送

See `Usage.md` for instructions on how to personalize the repo. 


Keyword list: ['text-to-speech', 'text to speech', 'tts', 'LLM-based', 'speech', 'voice']


Excluded: []


### Today: 10papers 
#### Brain-Informed Speech Separation for Cochlear Implants
 - **Authors:** Tom Gajecki, Jonas Althoff, Waldo Nogueira
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Signal Processing (eess.SP)
 - **Arxiv link:** https://arxiv.org/abs/2601.22260

 - **Pdf link:** https://arxiv.org/pdf/2601.22260

 - **Abstract**
 We propose a brain-informed speech separation method for cochlear implants (CIs) that uses electroencephalography (EEG)-derived attention cues to guide enhancement toward the attended speaker. An attention-guided network fuses audio mixtures with EEG features through a lightweight fusion layer, producing attended-source electrodograms for CI stimulation while resolving the label-permutation ambiguity of audio-only separators. Robustness to degraded attention cues is improved with a mixed curriculum that varies cue quality during training, yielding stable gains even when EEG-speech correlation is moderate. In multi-talker conditions, the model achieves higher signal-to-interference ratio improvements than an audio-only electrodogram baseline while remaining slightly smaller (167k vs. 171k parameters). With 2 ms algorithmic latency and comparable cost, the approach highlights the promise of coupling auditory and neural cues for cognitively adaptive CI processing.
#### Sylber 2.0: A Universal Syllable Embedding
 - **Authors:** Cheol Jun Cho, Nicholas Lee, Alan W Black, Gopala K. Anumanchipalli
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Computation and Language (cs.CL)
 - **Arxiv link:** https://arxiv.org/abs/2601.22306

 - **Pdf link:** https://arxiv.org/pdf/2601.22306

 - **Abstract**
 Scaling spoken language modeling requires speech tokens that are both efficient and universal. Recent work has proposed syllables as promising speech tokens at low temporal resolution, but existing models are constrained to English and fail to capture sufficient acoustic detail. To address this gap, we present Sylber 2.0, a self-supervised framework for coding speech at the syllable level that enables efficient temporal compression and high-fidelity reconstruction. Sylber 2.0 achieves a very low token frequency around 5 Hz, while retaining both linguistic and acoustic detail across multiple languages and expressive styles. Experiments show that it performs on par with previous models operating on high-frequency baselines. Furthermore, Sylber 2.0 enables efficient TTS modeling which can generate speech with competitive intelligibility and quality with SOTA models using only 72M parameters. Moreover, the universality of Sylber 2.0 provides more effective features for low resource ASR than previous speech coding frameworks. In sum, we establish an effective syllable-level abstraction for general spoken language.
#### Optimizing Domain-Adaptive Self-Supervised Learning for Clinical Voice-Based Disease Classification
 - **Authors:** Weixin Liu, Bowen Qu, Matthew Pontell, Maria Powell, Bradley Malin, Zhijun Yin
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Signal Processing (eess.SP)
 - **Arxiv link:** https://arxiv.org/abs/2601.22319

 - **Pdf link:** https://arxiv.org/pdf/2601.22319

 - **Abstract**
 The human voice is a promising non-invasive digital biomarker, yet deep learning for voice-based health analysis is hindered by data scarcity and domain mismatch, where models pre-trained on general audio fail to capture the subtle pathological features characteristic of clinical voice data. To address these challenges, we investigate domain-adaptive self-supervised learning (SSL) with Masked Autoencoders (MAE) and demonstrate that standard configurations are suboptimal for health-related audio. Using the Bridge2AI-Voice dataset, a multi-institutional collection of pathological voices, we systematically examine three performance-critical factors: reconstruction loss (Mean Absolute Error vs. Mean Squared Error), normalization (patch-wise vs. global), and masking (random vs. content-aware). Our optimized design, which combines Mean Absolute Error (MA-Error) loss, patch-wise normalization, and content-aware masking, achieves a Macro F1 of $0.688 \pm 0.009$ (over 10 fine-tuning runs), outperforming a strong out-of-domain SSL baseline pre-trained on large-scale general audio, which has a Macro F1 of $0.663 \pm 0.011$. The results show that MA-Error loss improves robustness and content-aware masking boosts performance by emphasizing information-rich regions. These findings highlight the importance of component-level optimization in data-constrained medical applications that rely on audio data.
#### Streaming Speech Recognition with Decoder-Only Large Language Models and Latency Optimization
 - **Authors:** Genshun Wan, Wenhui Zhang, Jing-Xuan Zhang, Shifu Xiong, Jianqing Gao, Zhongfu Ye
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2601.22779

 - **Pdf link:** https://arxiv.org/pdf/2601.22779

 - **Abstract**
 Recent advances have demonstrated the potential of decoderonly large language models (LLMs) for automatic speech recognition (ASR). However, enabling streaming recognition within this framework remains a challenge. In this work, we propose a novel streaming ASR approach that integrates a read/write policy network with monotonic chunkwise attention (MoChA) to dynamically segment speech embeddings. These segments are interleaved with label sequences during training, enabling seamless integration with the LLM. During inference, the audio stream is buffered until the MoChA module triggers a read signal, at which point the buffered segment together with the previous token is fed into the LLM for the next token prediction. We also introduce a minimal-latency training objective to guide the policy network toward accurate segmentation boundaries. Furthermore, we adopt a joint training strategy in which a non-streaming LLM-ASR model and our streaming model share parameters. Experiments on the AISHELL-1 and AISHELL-2 Mandarin benchmarks demonstrate that our method consistently outperforms recent streaming ASR baselines, achieving character error rates of 5.1% and 5.5%, respectively. The latency optimization results in a 62.5% reduction in average token generation delay with negligible impact on recognition accuracy
#### CALM: Joint Contextual Acoustic-Linguistic Modeling for Personalization of Multi-Speaker ASR
 - **Authors:** Muhammad Shakeel, Yosuke Fukumoto, Chikara Maeda, Chyi-Jiunn Lin, Shinji Watanabe
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Computation and Language (cs.CL); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2601.22792

 - **Pdf link:** https://arxiv.org/pdf/2601.22792

 - **Abstract**
 We present CALM, a joint Contextual Acoustic-Linguistic Modeling framework for multi-speaker automatic speech recognition (ASR). In personalized AI scenarios, the joint availability of acoustic and linguistic cues naturally motivates the integration of target-speaker conditioning with contextual biasing in overlapping conversations. CALM implements this integration in an end-to-end framework through speaker embedding-driven target-speaker extraction and dynamic vocabulary-based contextual biasing. We evaluate CALM on simulated English (LibriSpeechMix) and Japanese (Corpus of Spontaneous Japanese mixtures, CSJMix). On two-speaker mixtures, CALM reduces biased word error rate (B-WER) from 12.7 to 4.7 on LibriSpeech2Mix and biased character error rate (B-CER) from 16.6 to 8.4 on CSJMix2 (eval3), demonstrating the effectiveness of joint acoustic-linguistic modeling across languages. We additionally report results on the AMI corpus (IHM-mix condition) to validate performance on standardized speech mixtures.
#### EmoShift: Lightweight Activation Steering for Enhanced Emotion-Aware Speech Synthesis
 - **Authors:** Li Zhou, Hao Jiang, Junjie Li, Tianrui Wang, Haizhou Li
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Artificial Intelligence (cs.AI); Computation and Language (cs.CL); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2601.22873

 - **Pdf link:** https://arxiv.org/pdf/2601.22873

 - **Abstract**
 Achieving precise and controllable emotional expression is crucial for producing natural and context-appropriate speech in text-to-speech (TTS) synthesis. However, many emotion-aware TTS systems, including large language model (LLM)-based designs, rely on scaling fixed emotion embeddings or external guidance, limiting their ability to model emotion-specific latent characteristics. To address this gap, we present EmoShift, a lightweight activation-steering framework incorporating a EmoSteer layer, which learns a steering vector for each target emotion in the output embedding space to capture its latent offset and maintain stable, appropriate expression across utterances and categories. With only 10M trainable parameters,less than 1/30 of full fine-tuning, EmoShift outperforms zero-shot and fully fine-tuned baselines in objective and subjective evaluations, enhancing emotional expressiveness while preserving naturalness and speaker similarity. Further analysis confirms the proposed EmoSteer layer's effectiveness and reveals its potential for controllable emotional intensity in speech synthesis.
#### Layer-Aware Early Fusion of Acoustic and Linguistic Embeddings for Cognitive Status Classification
 - **Authors:** Krystof Novotny, Laureano Moro-Velázquez, Jiri Mekyska
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2601.23004

 - **Pdf link:** https://arxiv.org/pdf/2601.23004

 - **Abstract**
 Speech contains both acoustic and linguistic patterns that reflect cognitive decline, and therefore models describing only one domain cannot fully capture such complexity. This study investigates how early fusion (EF) of speech and its corresponding transcription text embeddings, with attention to encoder layer depth, can improve cognitive status classification. Using a DementiaBank-derived collection of recordings (1,629 speakers; cognitively normal controls$\unicode{x2013}$CN, Mild Cognitive Impairment$\unicode{x2013}$MCI, and Alzheimer's Disease and Related Dementias$\unicode{x2013}$ADRD), we extracted frame-aligned embeddings from different internal layers of wav2vec 2.0 or Whisper combined with DistilBERT or RoBERTa. Unimodal, EF and late fusion (LF) models were trained with a transformer classifier, optimized, and then evaluated across 10 seeds. Performance consistently peaked in mid encoder layers ($\sim$8$\unicode{x2013}$10), with the single best F1 at Whisper + RoBERTa layer 9 and the best log loss at Whisper + DistilBERT layer 10. Acoustic-only models consistently outperformed text-only variants. EF boosts discrimination for genuinely acoustic embeddings, whereas LF improves probability calibration. Layer choice critically shapes clinical multimodal synergy.
#### PersonaCite: VoC-Grounded Interviewable Agentic Synthetic AI Personas for Verifiable User and Design Research
 - **Authors:** Mario Truss
 - **Subjects:** Subjects:
Human-Computer Interaction (cs.HC); Artificial Intelligence (cs.AI); Audio and Speech Processing (eess.AS); Image and Video Processing (eess.IV)
 - **Arxiv link:** https://arxiv.org/abs/2601.22288

 - **Pdf link:** https://arxiv.org/pdf/2601.22288

 - **Abstract**
 LLM-based and agent-based synthetic personas are increasingly used in design and product decision-making, yet prior work shows that prompt-based personas often produce persuasive but unverifiable responses that obscure their evidentiary basis. We present PersonaCite, an agentic system that reframes AI personas as evidence-bounded research instruments through retrieval-augmented interaction. Unlike prior approaches that rely on prompt-based roleplaying, PersonaCite retrieves actual voice-of-customer artifacts during each conversation turn, constrains responses to retrieved evidence, explicitly abstains when evidence is missing, and provides response-level source attribution. Through semi-structured interviews and deployment study with 14 industry experts, we identify preliminary findings on perceived benefits, validity concerns, and design tensions, and propose Persona Provenance Cards as a documentation pattern for responsible AI persona use in human-centered design workflows.
#### An Effective Energy Mask-based Adversarial Evasion Attacks against Misclassification in Speaker Recognition Systems
 - **Authors:** Chanwoo Park, Chanwoo Kim
 - **Subjects:** Subjects:
Sound (cs.SD); Cryptography and Security (cs.CR); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2601.22390

 - **Pdf link:** https://arxiv.org/pdf/2601.22390

 - **Abstract**
 Evasion attacks pose significant threats to AI systems, exploiting vulnerabilities in machine learning models to bypass detection mechanisms. The widespread use of voice data, including deepfakes, in promising future industries is currently hindered by insufficient legal frameworks. Adversarial attack methods have emerged as the most effective countermeasure against the indiscriminate use of such data. This research introduces masked energy perturbation (MEP), a novel approach using power spectrum for energy masking of original voice data. MEP applies masking to small energy regions in the frequency domain before generating adversarial perturbations, targeting areas less noticeable to the human auditory model. The study primarily employs advanced speaker recognition models, including ECAPA-TDNN and ResNet34, which have shown remarkable performance in speaker verification tasks. The proposed MEP method demonstrated strong performance in both audio quality and evasion effectiveness. The energy masking approach effectively minimizes the perceptual evaluation of speech quality (PESQ) degradation, indicating that minimal perceptual distortion occurs to the human listener despite the adversarial perturbations. Specifically, in the PESQ evaluation, the relative performance of the MEP method was 26.68% when compared to the fast gradient sign method (FGSM) and iterative FGSM.
#### Rethinking Speech Representation Aggregation in Speech Enhancement: A Phonetic Mutual Information Perspective
 - **Authors:** Seungu Han, Sungho Lee, Kyogu Lee
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2601.22480

 - **Pdf link:** https://arxiv.org/pdf/2601.22480

 - **Abstract**
 Recent speech enhancement (SE) models increasingly leverage self-supervised learning (SSL) representations for their rich semantic information. Typically, intermediate features are aggregated into a single representation via a lightweight adaptation module. However, most SSL models are not trained for noise robustness, which can lead to corrupted semantic representations. Moreover, the adaptation module is trained jointly with the SE model, potentially prioritizing acoustic details over semantic information, contradicting the original purpose. To address this issue, we first analyze the behavior of SSL models on noisy speech from an information-theoretic perspective. Specifically, we measure the mutual information (MI) between the corrupted SSL representations and the corresponding phoneme labels, focusing on preservation of linguistic contents. Building upon this analysis, we introduce the linguistic aggregation layer, which is pre-trained to maximize MI with phoneme labels (with optional dynamic aggregation) and then frozen during SE training. Experiments show that this decoupled approach improves Word Error Rate (WER) over jointly optimized baselines, demonstrating the benefit of explicitly aligning the adaptation module with linguistic contents.


by Zyzzyva0381 (Windy). 


2026-02-02
