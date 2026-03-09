# Showing new listings for Monday, 9 March 2026
Auto update papers at about 2:30am UTC (10:30am Beijing time) every weekday.


阅读 `Usage.md`了解如何使用此repo实现个性化的Arxiv论文推送

See `Usage.md` for instructions on how to personalize the repo. 


Keyword list: ['text-to-speech', 'text to speech', 'tts', 'LLM-based', 'speech', 'voice']


Excluded: []


### Today: 10papers 
#### Activation Steering for Accent Adaptation in Speech Foundation Models
 - **Authors:** Jinuo Sun, Yang Xiao, Sung Kyun Chung, Qiuchi Hu, Gongping Huang, Eun-Jung Holden, Ting Dang
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2603.05813

 - **Pdf link:** https://arxiv.org/pdf/2603.05813

 - **Abstract**
 Accent variability remains a major errors in automatic speech recognition, yet most adaptation methods rely on parameter fine-tuning without understanding where accent information is encoded. We treat accent variation as an interpretable subspace in hidden representations and investigate whether it can be identified and controlled directly in activation space. We extract layer-wise encoder activations and estimate mean-shift directions capturing accent-induced representation shifts. By injecting these directions into individual layers and measuring how they align accented and standard embeddings, we derive a layer-wise accent sensitivity profile, revealing that accent information concentrates in a narrow band of middle encoder layers. Leveraging this structure, we further introduce parameter-free accent steering that modifies representations during inference without updating model weights. Experiments across eight accents show consistent word error rate reductions.
#### ImKWS: Test-Time Adaptation for Keyword Spotting with Class Imbalance
 - **Authors:** Hanyu Ding, Yang Xiao, Jiaheng Dong, Ting Dang
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2603.05821

 - **Pdf link:** https://arxiv.org/pdf/2603.05821

 - **Abstract**
 Keyword spotting (KWS) identifies words for voice assistants, but environmental noise frequently reduces accuracy. Standard adaptation fixes this issue and strictly requires original or labeled audio. Test time adaptation (TTA) solves this data constraint using only unlabeled test audio. However, current methods fail to handle the severe imbalance between rare keywords and frequent background sounds. Consequently, standard entropy minimization (EM) becomes overconfident and heavily biased toward the frequent background class. To overcome this problem, we propose a TTA method named ImKWS. Our approach splits the entropy process into a reward branch and a penalty branch with separate update strengths. Furthermore, we enforce consistency across multiple audio transformations to ensure stable model updates. Experiments on the Google Speech Commands dataset indicate ImKWS achieves reliable adaptation in realistic imbalanced scenarios. The code is available on GitHub.
#### Reconstruct! Don't Encode: Self-Supervised Representation Reconstruction Loss for High-Intelligibility and Low-Latency Streaming Neural Audio Codec
 - **Authors:** Junhyeok Lee, Xiluo He, Jihwan Lee, Helin Wang, Shrikanth Narayanan, Thomas Thebaud, Laureano Moro-Velazquez, Jesús Villalba, Najim Dehak
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Artificial Intelligence (cs.AI)
 - **Arxiv link:** https://arxiv.org/abs/2603.05887

 - **Pdf link:** https://arxiv.org/pdf/2603.05887

 - **Abstract**
 Neural audio codecs optimized for mel-spectrogram reconstruction often fail to preserve intelligibility. While semantic encoder distillation improves encoded representations, it does not guarantee content preservation in reconstructed speech. In this work, we demonstrate that self-supervised representation reconstruction (SSRR) loss fundamentally improves codec training and performance. First, SSRR significantly accelerates convergence, enabling competitive results using only a single GPU. Second, it enhances intelligibility by reconstructing distilled self-supervised representations from codec outputs. Third, SSRR enables high intelligibility without additional lookahead in streaming Transformer-based codecs, allowing a zero-lookahead architecture for real-time deployment. As a result, our JHCodec achieves state-of-the-art performance while maintaining minimal latency and reduced training cost. We open-source the full implementation, training pipeline, and demo on Github this https URL.
#### Activation Steering for Accent-Neutralized Zero-Shot Text-To-Speech
 - **Authors:** Mu Yang, John H. L. Hansen
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2603.05977

 - **Pdf link:** https://arxiv.org/pdf/2603.05977

 - **Abstract**
 Zero-shot Text-to-Speech (TTS) models can generate speech that captures both the voice timbre and accent of a reference speaker. However, disentangling these attributes remains challenging, as the output often inherits both the accent and timbre from the reference. In this study, we introduce a novel, post-hoc, and training-free approach to neutralize accent while preserving the speaker's original timbre, utilizing inference-time activation steering. We first extract layer-specific "steering vectors" offline, which are derived from the internal activation differences within the TTS model between accented and native speech. During inference, the steering vectors are applied to guide the model to produce accent-neutralized, timbre-preserving speech. Empirical results demonstrate that the proposed steering vectors effectively mitigate the output accent and exhibit strong generalizability to unseen accented speakers, offering a practical solution for accent-free voice cloning.
#### StreamVoiceAnon+: Emotion-Preserving Streaming Speaker Anonymization via Frame-Level Acoustic Distillation
 - **Authors:** Nikita Kuzmin, Kong Aik Lee, Eng Siong Chng
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Artificial Intelligence (cs.AI); Signal Processing (eess.SP)
 - **Arxiv link:** https://arxiv.org/abs/2603.06079

 - **Pdf link:** https://arxiv.org/pdf/2603.06079

 - **Abstract**
 We address the challenge of preserving emotional content in streaming speaker anonymization (SA). Neural audio codec language models trained for audio continuation tend to degrade source emotion: content tokens discard emotional information, and the model defaults to dominant acoustic patterns rather than preserving paralinguistic attributes. We propose supervised finetuning with neutral-emotion utterance pairs from the same speaker, combined with frame-level emotion distillation on acoustic token hidden states. All modifications are confined to finetuning, which takes less than 2 hours on 4 GPUs and adds zero inference latency overhead, while maintaining a competitive 180ms streaming latency. On the VoicePrivacy 2024 protocol, our approach achieves a 49.2% UAR (emotion preservation) with 5.77% WER (intelligibility), a +24% relative UAR improvement over the baseline (39.7%->49.2%) and +10% over the emotion-prompt variant (44.6% UAR), while maintaining strong privacy (EER 49.0%). Demo and code are available: this https URL
#### Continual Adaptation for Pacific Indigenous Speech Recognition
 - **Authors:** Yang Xiao, Aso Mahmudi, Nick Thieberger, Eliathamby Ambikairajah, Eun-Jung Holden, Ting Dang
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Computation and Language (cs.CL); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2603.06310

 - **Pdf link:** https://arxiv.org/pdf/2603.06310

 - **Abstract**
 Speech foundation models struggle with low-resource Pacific Indigenous languages because of severe data scarcity. Furthermore, full fine-tuning risks catastrophic forgetting. To address this gap, we present an empirical study adapting models to real-world Pacific datasets. We investigate how data volume and linguistic features affect adaptation success. Specifically, we evaluate strategies including Full Fine-Tuning and Low-Rank Adaptation (LoRA). Additionally, we analyze a continual learning framework for sequentially acquiring multiple languages. We demonstrate that adapting to these distant languages causes severe internal representational drift. Consequently, these models face a strict plasticity and stability dilemma. While LoRA adapts well initially, it suffers from catastrophic forgetting during sequential learning. Ultimately, this study highlights the urgent need for robust adaptation strategies tailored to underrepresented languages.
#### Classification of Autistic and Non-Autistic Children's Speech: A Cross-Linguistic Study in Finnish, French, and Slovak
 - **Authors:** Sofoklis Kakouros, Ida-Lotta Myllylä
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2603.06327

 - **Pdf link:** https://arxiv.org/pdf/2603.06327

 - **Abstract**
 We present a cross-linguistic study of speech in autistic and non-autistic children speaking Finnish, French, and Slovak. We combine supervised classification with within-language and cross-corpus transfer experiments to evaluate classification performance within and across languages and to probe which acoustic cues are language-specific versus language-general. Using a large set of acoustic-prosodic features, we implement speaker-level classification benchmarks as an analytical tool rather than to seek state-of-the-art performance. Within-language models, evaluated with speaker-level cross-validation, yielded heterogeneous results. The Finnish model performed best (Accuracy 0.84, F1 0.88), followed by Slovak (Accuracy 0.63, F1 0.68) and French (Accuracy 0.68, F1 0.56). We then tested cross-language generalization. A model trained on all pooled corpora reached an overall Accuracy of 0.61 and F1 0.68. Leave-one-corpus-out experiments, which test transfer to an unseen language, showed moderate success when testing on Slovak (F1 0.70) and Finnish (F1 0.78), but poor transfer to French (F1 0.42). Feature-importance analyses across languages highlighted partially shared, but not fully language-invariant, acoustic markers of autism. These findings suggest that some autism-related speech cues generalize across typologically distinct languages, but robust cross-linguistic classifiers will likely require language-aware modeling and more homogeneous recording conditions.
#### Cross-linguistic Prosodic Analysis of Autistic and Non-autistic Child Speech in Finnish, French and Slovak
 - **Authors:** Ida-Lotta Myllylä, Sofoklis Kakouros
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2603.06332

 - **Pdf link:** https://arxiv.org/pdf/2603.06332

 - **Abstract**
 Prosodic differences in autism are well-documented, but cross-linguistic evidence remains limited. This study investigates prosody in autism across a multilingual corpus of Finnish, French, and Slovak speakers. 88 acoustic features from over 5,000 inter-pausal units were extracted, and data were reduced via Principal Component Analysis (PCA) and analyzed using Linear Mixed-Effects Models (LMMs). Cross-linguistically, autistic speakers exhibited increased general intensity variability and a clearer, less breathy voice quality (higher Harmonics-to-Noise Ratio and alpha ratio), alongside reduced temporal intensity dynamics and lower central f0. Monolingual analyses revealed language-specific nuances: Slovak results aligned with cross-linguistic f0 patterns but diverged on voice quality, while Finnish results mirrored the broader voice quality findings. These results emphasize including voice quality and intensity dynamics in the study of possible language-independent markers of autism, alongside traditional pitch measures. The findings challenge deficiency-based models, suggesting instead a complex, acoustically distinct prosodic profile across languages.
#### Doctor or Patient? Synergizing Diarization and ASR for Code-Switched Hinglish Medical Conditions Extraction
 - **Authors:** Séverin Baroudi, Yanis Labrak, Shashi Kumar, Joonas Kalda, Sergio Burdisso, Pawel Cyrta, Juan Ignacio Alvarez-Trejos, Petr Motlicek, Hervé Bredin, Ricard Marxer
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2603.06373

 - **Pdf link:** https://arxiv.org/pdf/2603.06373

 - **Abstract**
 Extracting patient medical conditions from code-switched clinical spoken dialogues is challenging due to rapid turn-taking and highly overlapped speech. We present a robust system evaluated on the DISPLACE-M dataset of real-world Hinglish medical conversations. We propose an End-to-End Neural Diarization with Vector Clustering approach (EEND-VC) to accurately resolve dense and speaker overlaps in Doctor-Patient Conversations (DoPaCo). For transcription, we adapt a Qwen3 ASR model via domain-specific fine-tuning, Devanagari script normalization, and dialogue-level LLM error correction, achieving an 18.59% tcpWER. We benchmark open and proprietary LLMs on medical condition extraction, comparing our text-based cascade system against a multimodal End-to-End (E2E) audio framework. While proprietary E2E models set the performance ceiling, our open cascaded architecture is highly competitive, as it achieved first place out of 25 participants in the DISPLACE-M challenge. All implementations are publicly released.
#### Whisper-CD: Accurate Long-Form Speech Recognition using Multi-Negative Contrastive Decoding
 - **Authors:** Hoseong Ahn, Jeongyun Chae, Yoonji Park, Kyuhong Shim
 - **Subjects:** Subjects:
Sound (cs.SD); Artificial Intelligence (cs.AI); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2603.06193

 - **Pdf link:** https://arxiv.org/pdf/2603.06193

 - **Abstract**
 Long-form speech recognition with large encoder-decoder models such as Whisper often exhibit hallucinations, repetition loops, and content omissions. These errors can accumulate and be further amplified when the previous segment's transcription is used as decoding context. We propose Whisper-CD, a training-free contrastive decoding framework that contrasts clean-audio logits against negative logits computed from three acoustically motivated perturbations: Gaussian noise injection, silence signal, and audio temporal shift. We aggregate these negatives via the log-sum-exp operator, building a unified multi-negative objective for token-by-token decoding. Across five English long-form benchmarks, Whisper-CD reduces WER by up to 24.3pp on CORAAL and shows 48% faster token generation throughput than beam search. Because Whisper-CD operates purely at inference time, it can be applied as a drop-in replacement to already-deployed Whisper systems without retraining.


by Zyzzyva0381 (Windy). 


2026-03-09
