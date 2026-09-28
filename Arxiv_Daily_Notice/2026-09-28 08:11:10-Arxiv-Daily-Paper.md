# Showing new listings for Monday, 28 September 2026
Auto update papers at about 2:30am UTC (10:30am Beijing time) every weekday.


阅读 `Usage.md`了解如何使用此repo实现个性化的Arxiv论文推送

See `Usage.md` for instructions on how to personalize the repo. 


Keyword list: ['text-to-speech', 'text to speech', 'tts', 'LLM-based', 'speech', 'voice']


Excluded: []


### Today: 23papers 
#### Asymmetric Classifier-Free Guidance for Target-Speaker ASR
 - **Authors:** Yiwen Guan, Jacob Whitehill
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Computation and Language (cs.CL); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2609.30476

 - **Pdf link:** https://arxiv.org/pdf/2609.30476

 - **Abstract**
 Target-speaker automatic speech recognition (TS-ASR) must identify and transcribe a desired speaker under varying overlap and noise conditions. These changes alter the acoustic evidence for the target speaker in the speech mixture, motivating inference-time calibration of speaker conditioning. We introduce asymmetric classifier-free guidance (CFG) for TS-ASR using Whisper: the speaker-conditioned branch predicts the target transcript, while the speaker-unconditioned branch predicts serialized multi-speaker transcripts. CFG adjusts the contribution of speaker conditioning during decoding through a single guidance scale. We select a global guidance scale on target-domain development data and train a lightweight encoder-based predictor to adjust it for each utterance, keeping the recognition model fixed. Under domain shifts, our full system achieves relative word error rate (WER) reductions of up to 21.8% over the condition-only baseline, and 5.6% over standard conditional decoding of the same CFG-trained model. Oracle analysis shows that substantially larger WER reductions are possible through utterance-level scale selection and identifies how beneficial adjustments vary with domain shifts.
#### Seeing Speech: Learning Visible Articulatory Dynamics for Speech-Driven 3D Facial Animation
 - **Authors:** Hyung Kyu Kim, Byungchan Hwang, Hak Gu Kim
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Graphics (cs.GR); Machine Learning (cs.LG)
 - **Arxiv link:** https://arxiv.org/abs/2609.30517

 - **Pdf link:** https://arxiv.org/pdf/2609.30517

 - **Abstract**
 Recent progress in speech-driven 3D facial animation has improved vertex-level reconstruction quality, but speech-consistent visible articulation remains difficult. This is because speech production follows structured and constrained articulators' coordination and the mapping from acoustics to motion is inherently one-to-many. Motivated by the structured patterns of visible articulation, we propose a novel articulation-aware framework that models visible speech through directional articulatory motions and composes them into surface-consistent 3D facial motion. To represent visible articulation with three directional articulatory motions, spreading, opening, and protrusion, we propose a Speech--Articulatory Memory (SAM) that captures the correspondence between speech and these motions under phonetic context through retrieval and decoding based on a key-value memory structure. Then, a Topology-aware Articulatory Composition (TAC) integrates the predicted directional articulatory motions under mesh topology to produce surface-consistent 3D facial motion. Experiments on VOCASET and TFHP show that our method achieves state-of-the-art performance on standard reconstruction metrics and improves visible articulatory distance and velocity errors for lip articulation, while a user study confirms clear preference in lip sync and realism.
#### Adapting Personalized Speech Enhancement for Low-Latency Audio-Visual Target-Speaker Extraction
 - **Authors:** Rayhan Rashed, Senja Filipi, Ross Cutler
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Computer Vision and Pattern Recognition (cs.CV); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2609.30631

 - **Pdf link:** https://arxiv.org/pdf/2609.30631

 - **Abstract**
 Online audio-visual target-speaker extraction aims to remove competing voices while preserving speech quality and bounding lookahead. Existing extractors are built and evaluated for separation on synthetic mixtures, leaving listening quality and meeting behavior largely untested. We introduce Audio-Visual Personalized Voice Quality Enhancement (AV-PVQE), which approaches these requirements from the other direction. We start from a personalized speech enhancement model that reconstructs a requested voice at high quality but confuses the target in 46% of two-speaker mixtures despite clean enrollment. Adding mouth features at its speaker-conditioning input and jointly fine-tuning the visual and reconstruction networks reduces this rate to 1.6%, with no future frames and 20 ms of algorithmic delay. Compared with an online autoregressive audio-visual extractor, AV-PVQE yields separation gains on two synthetic benchmarks and larger gains on recorded meetings, and keeps its advantage on excerpts with more speakers than the fine-tuning mixtures. In personalized P.835 listening tests on two meeting corpora, it improves overall quality over this extractor by 0.57 and 0.63 MOS, with similar mean rating relative to the starting model. Preservation and rejection tests show that it keeps the target intact when no competing voice is present and suppresses competing speech when the target is absent.
#### A Comprehensive Study of Content Representations for Speech Synthesis
 - **Authors:** Diego Torres, Axel Roebel, Nicolas Obin
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Machine Learning (cs.LG); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2609.30975

 - **Pdf link:** https://arxiv.org/pdf/2609.30975

 - **Abstract**
 Speech content representations are central to voice conversion, speech-to-speech translation, and multimodal language models, yet they are rarely compared under a common generative framework that directly measures what each representation contains. We address this by training a generative model conditioned solely on each representation and evaluating the generated audio along the content, speaker identity, and prosody axes. Across SSL features, supervised tokens, posteriorgrams, and neural audio codecs, we find two distinct regimes: representations that nearly reconstruct the original audio, and representations that effectively disentangle speaker identity. These results show that disentanglement depends not on supervision alone, but on the interaction between the training objective and the representation's information capacity: supervised representations only disentangle speaker identity when their capacity is sufficiently constrained.
#### Room Impulse Response Embeddings for Speech Enhancement in Noisy and Reverberant Environments
 - **Authors:** Adrian Meise, Reinhold Haeb-Umbach
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2609.31041

 - **Pdf link:** https://arxiv.org/pdf/2609.31041

 - **Abstract**
 We propose a self-supervised approach for learning room impulse response (RIR) representations from single-channel noisy-reverberant speech. It consists of first training on reverberant data, then on noisy-reverberant data, and finally with a teacher-student approach, where the student learns to replicate the teacher's embeddings when given a noisy version of the reverberant input. We assess their representational capabilities by estimating acoustic room parameters from them. Conditioning a discriminative speech enhancement model on the derived embeddings yields consistent gains across all evaluated metrics, including downstream word error rate, for both reverberant and noisy-reverberant speech.
#### Why Alzheimer's Speech Screening Fails to Generalize: Bridging the Deployment Gap via Cross-Corpus Evidence Anchoring
 - **Authors:** Zijian Lu, Sizhe Liu, Yin Zhang, Jixuan Deng, Xinrong Lin, Xinchen Yuan, Chicheng Jin, Yiping Zuo, Yuanchao Li
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Computation and Language (cs.CL); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2609.31293

 - **Pdf link:** https://arxiv.org/pdf/2609.31293

 - **Abstract**
 Speech-based screening is a promising, non-invasive approach for detecting Alzheimer's disease and related cognitive risks. However, models trained on a single domain often generalize poorly to unseen languages, tasks, or recording protocols. This paper investigates this deployment gap using a leave-one-corpus-out evaluation across four distinct datasets. Among 70 interpretable speech and language features, 59 exhibit direction conflicts between healthy control and cognitive risk groups across corpora, with pause, silence, and speech rate showing high protocol sensitivity. Furthermore, while the XLM-R text baseline achieves strong average performance, its Area Under the ROC Curve (AUC) drops to 0.520 on the weakest held-out domain. A standard GroupDRO baseline reaches a 0.766 mean speaker AUC and a 0.504 worst-domain AUC under the same protocol. To address this, we propose a fusion method that integrates XLM-R text baseline scores with evidence anchors selected during training. Balanced fusion achieves a 0.785 mean speaker AUC, while anchor-heavy fusion raises the worst-case speaker AUC to 0.615. This work highlights the need to audit feature transferability and report worst-case domain robustness in cognitive speech screening.
#### SAID: Semantic Acoustic Imaging Detector for Sound Event Localization and Detection
 - **Authors:** Runbang Wang, Zining Liang, Yin Cao, Qiuqiang Kong
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2609.31492

 - **Pdf link:** https://arxiv.org/pdf/2609.31492

 - **Abstract**
 In daily life, people hear speech, footsteps, and music around them. We can often recognize these sounds and judge where they come from. Each sound source can be shown on a separate acoustic map, a rectangular image covering $360^{\circ}$ horizontally and $180^{\circ}$ vertically. The map shows the directions occupied by the source as a region and the sound energy within that region. A class label identifies the sound. Predicting these labeled acoustic maps from audio is called semantic acoustic imaging. Such maps could help robots perceive their surroundings and allow augmented reality displays to show sound regions and classes over the real world. Existing models can recognize sound classes and estimate a direction for each source. However, a direction alone does not describe the source region or its energy. Acoustic imaging must also distinguish sound sources in nearby directions, while the number of active sources and the regions they occupy can change over time. We therefore propose the Semantic Acoustic Imaging Detector (SAID), which predicts a separate labeled acoustic map for each active source from audio. First, we pretrain Audio2Sph, SAID's audio encoder, through sound energy estimation across directions without class labels. Then, we train the complete SAID model to predict source regions, energy, and classes together. We also develop a pipeline that generates simulated recordings for pretraining and supports fine-tuning on real recordings. On the official DCASE2026 Task 3 Track A evaluation set, our submitted system ranks first with 0.1080 macro-averaged mean average precision (Macro mAP) and 0.3962 Macro Pearson $r$. Demos and code are provided at this https URL.
#### RePlay: Retrieval-Based Voice Playback for Multi-Turn spoken dialogue
 - **Authors:** Sathvik Udupa, Naveen Kumar, Ryan Folmsbee
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2609.31588

 - **Pdf link:** https://arxiv.org/pdf/2609.31588

 - **Abstract**
 Many voice interaction applications require exact control over both the content and delivery of responses, typically using pre-recorded lines. Recent full-duplex models respond with low latency but cannot guarantee exact content or reproduce a specific recorded performance, while cascaded systems can be constrained to predefined responses at the cost of additional latency. We propose RePlay, a spoken dialogue system adapted from PersonaPlex that handles multi-turn conversations by retrieving and playing pre-recorded lines. Using probing, we identify the layer and frame at which the upcoming response becomes recoverable, and use this hidden state as the retrieval query. RePlay retains only the layers up to that point and replaces text and speech generation with lightweight turn-taking and retrieval heads. In simulated multi-turn interviews, RePlay reaches a median latency of 383 ms, 3 to 7 times lower than ASR-LLM cascades of comparable dialogue quality, at the cost of lower exact-line accuracy. In a user study, participants preferred RePlay in 63% of ratings versus 12% for a fast cascade with a small LLM (p = 0.008), and showed a non-significant preference (46% vs. 21%) over a slower cascade with a stronger LLM.
#### CLEAR: Online Speech Content Leakage Estimation through Cross-ASR Disagreement
 - **Authors:** Bhawana Chhaglani, Tanvi Kandepuneni, Jeremy Gummeson, Prashant Shenoy
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2609.30415

 - **Pdf link:** https://arxiv.org/pdf/2609.30415

 - **Abstract**
 Signal-level speech privacy mechanisms suppress linguistic content while preserving acoustic information needed by downstream sensing applications. However, their privacy settings are typically evaluated/selected offline and remain fixed during deployment, even though speech-content leakage can vary substantially across utterances and speakers. Adapting privacy protection at runtime requires estimating how much speech remains recoverable, but conventional measures such as WER or PER require ground-truth transcripts and therefore cannot be computed online. We present CLEAR, a reference-free approach for estimating speech-content leakage at runtime using disagreement among heterogeneous ASR systems. Our key insight is that independently trained ASRs exhibit consistent behavior when linguistic content remains recoverable and increasingly disagree as privacy transformations obscure speech. Using configurable speech-suppression mechanism, we show that cross-ASR disagreement closely tracks transcript-grounded leakage across privacy operating points, achieving a correlation of 0.8. We further characterize the latency-accuracy trade-off of heterogeneous ASR subsets and use their hypotheses to identify potentially exposed words. These capabilities enable privacy to be treated as a runtime property rather than a fixed configuration: CLEAR can communicate residual speech exposure to users and provide feedback for dynamically adjusting privacy aggressiveness while retaining acoustic utility for downstream sensing tasks.
#### Inference-Time Target Speaker Unlearning in LLM-Based Automatic Speech Recognition
 - **Authors:** Bo Su, Yueru Yan, Thai Le
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2609.30439

 - **Pdf link:** https://arxiv.org/pdf/2609.30439

 - **Abstract**
 We introduce target-speaker unlearning ASR (TSU-ASR) task in a fully end-to-end framework for multi-speaker ASR and diarization. Given a multi-speaker utterance and a set of opt-out speakers who do not wish to have their speech transcribed, the task requires an ASR system to transcribe all speakers except the opt-out ones, while still indicating when those speakers are active. As a first step towards tackling this task, we introduce a novel, light-weight Enrollment-Conditioned Gating (ECG) module attachable to a frozen dual-stream speech LLM that enables ASR for new opt-out speakers dynamically during inference, even those who were not seen during initial ECG training phase. Our experiments on both AMI (English) and AliMeeting (Mandarin) datasets show that speech transcription accuracy for corresponding opt-out words or characters falls from 72.3% to 48.2% and from 73.6% to 27.3%, respectively, while retained speakers' transcription error rates maintain more or less the same. Our approach provides a practical solution for modern video conferencing platforms, allowing speakers to dynamically opt-out from automated AI transcriptions without forcefully leaving the meeting sessions, enabling a privacy-preserving interface for potentially millions of online meetings daily.
#### AcoustiClaim: A Numeric Claim Benchmark with Instrument Ground Truth
 - **Authors:** Sheng-Tse Lin, Siyuan Zhai, Chien-Liang Kuo, Massa Baali, Bhiksha Raj
 - **Subjects:** Subjects:
Sound (cs.SD); Computation and Language (cs.CL); Machine Learning (cs.LG); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2609.30483

 - **Pdf link:** https://arxiv.org/pdf/2609.30483

 - **Abstract**
 Audio language models state numbers for acoustic quantities, and neither human opinion nor a judge model says whether such a number is true of the signal. AcoustiClaim extracts each numeric claim from free text, scores it against the instrument that defines the quantity, and classes each quantity by where its reference can be read. Four open-weight systems and one closed model, asked for ten quantities five ways on two corpora, fill 207 cells. Of these, 49 emit fewer than five distinct values, and eight of the 158 cells that can be ranked exceed a rank correlation of 0.3, the bar we set, three with an interval clear of it, five of them one closed model reading pitch. Error sits at or above a constant-predictor floor in every ranked cell but three. The reference decoder we train declines the five voice quantities in prose on 95% of mixtures, with nothing withheld, and states them on the clean twins, reproducing its targets' rule from audio alone. With a calibrated threshold, withholding lowers error on all ten quantities on the mixtures in the mean and on eight at every split, against at most 0.6% from a random selector. A linear baseline orders errors at least as well as ours. F0 s.d. and shimmer stay above the constant floor.
#### MuseTimbre: Zero-Shot Timbre Transfer by Controlling a Frozen Music Generator
 - **Authors:** Yuan-Chiao Cheng, Zhiyao Duan
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2609.30548

 - **Pdf link:** https://arxiv.org/pdf/2609.30548

 - **Abstract**
 Instrument timbre transfer re-voices a performance using the timbre of another instrument. Extracting the target timbre from an audio reference capture more nuances than inferring it from a text prompt. Systems that read timbre from such a clip train a dedicated model for the task, which captures the timbre cleanly but stays a narrow, single-purpose system. More versatile approaches add control to a pretrained music generator, yet a reference clip entangles timbre with genre and melody, so these systems fall back on text to name the timbre. We present MuseTimbre, the first system, to our best knowledge, that transfers timbre from an audio reference to a polyphonic source through conditioning a pretrained music generator. This system employs a multi-pitch estimator to extract pitch information from the source and finetune a CLAP encoder to extract timbre information from the reference audio. Experiments show that across four datasets of real polyphonic recordings, MuseTimbre achieves pitch alignment on par with the baselines while matching the reference timbre far more closely. Results also show that the finetuned CLAP-based timbre extractor is robust to pitch variations, making it useful in timbre similarity measures.
#### Training-Free Contextual ASR via SpeechLLM-Based Error-Aware Selective Retrieval
 - **Authors:** Natsuo Yamashita, Ai Nemoto, Ryosuke Koichi, Masaaki Yamamoto
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2609.30694

 - **Pdf link:** https://arxiv.org/pdf/2609.30694

 - **Abstract**
 Recognition of domain-specific and low-frequency terms remains challenging for automatic speech recognition (ASR). Although contextual biasing can improve their recognition, directly providing a large terminology dictionary introduces many irrelevant biasing terms. Retrieval-based contextual biasing addresses this issue by selecting candidate terms from an external dictionary, but querying many recognized words requires numerous dictionary lookups and may yield poorly targeted candidates. We propose a training-free contextual ASR framework in which a pretrained speech large language model (SpeechLLM) jointly generates an ASR hypothesis and localizes error spans likely to involve domain-specific terms. Only the localized spans are used to retrieve phonologically similar terms from an external terminology dictionary. The same SpeechLLM then re-recognizes the audio conditioned on the first-pass hypothesis and the retrieved terms, without task-specific model training. To assess applicability across domains, we evaluate the framework on medical, air traffic control, and financial speech. The proposed method substantially reduces dictionary queries while improving the recall and ranking of relevant terminology candidates and second-pass ASR performance across all three domains.
#### Learning Natural Conversational Behavior in Tandem Speech-to-Speech Models with Randomized Guidance
 - **Authors:** Manato Yaguchi, Yotaro Kubo, Hikaru Asano, So Kuroki
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2609.30773

 - **Pdf link:** https://arxiv.org/pdf/2609.30773

 - **Abstract**
 Tandem speech-to-speech architectures couple a responsive speech frontend with an asynchronous text backend. In KAME, a large language model (LLM) serves as the backend, supplying candidate responses as guidance to the speech frontend while the user is still speaking. Ordinary conversation recordings capture the eventual response but not the guidance the backend would supply during the user's utterance. Generating the missing guidance with a simulator LLM adds substantial data-preparation overhead when training on real conversations. We propose randomized intermediate guidance, which derives guidance directly from the conversation corpus rather than simulating backend LLM behavior. During training, target responses provide informative guidance, while randomly sampled responses provide potentially irrelevant updates during the utterance. This combination aims to teach the frontend to use backend information selectively. On synthetic dialogues, KAME trained with this recipe achieves response quality comparable to that of the LLM-generated and similarity-based baselines. Training on 3.8k hours of real conversations improves smooth turn-taking and audio-judge naturalness over synthetic-data KAME while retaining a response-quality advantage over Moshi. These results show that randomized guidance offers a practical route to combining the response-quality benefits of tandem models with natural conversational behavior learned from real speech.
#### Dialogue-Based Streaming Audio-Visual Target Speaker Extraction with Predictive Dialogue Information
 - **Authors:** Shuhan Zhang, Wenxuan Wu, Haizhou Li
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2609.30774

 - **Pdf link:** https://arxiv.org/pdf/2609.30774

 - **Abstract**
 In face-to-face, real-time communication, a talking agent must track the target speaker through natural pauses, turn-taking, and backchannels, often amid background cross-talk. Most target speaker extraction (TSE) studies, however, rely on simulated mixtures with full or sparse overlap and ignore the turn-taking of real conversations. We therefore introduce, to our knowledge, the first benchmark for online audio-visual TSE (AV-TSE), built from intact dyadic interactions with independent third-party interference. Observing that anticipating upcoming activity from semantic, acoustic, and facial cues benefits online AV-TSE, we propose an LLM-based target-speaker voice activity projection (TS-VAP) module. Unlike conventional VAP with separated speaker channels, it forecasts the future activity of the target and conversational partner directly from the overlapping mixture, drawing on the linguistic and conversational knowledge of a speech-LLM, and uses this prediction to guide a low-latency separator. We further combine this predictive context with historical and synchronous speaker context. Experiments show that TS-VAP consistently improves streaming extraction across multiple AV-TSE backbones, and that further combining historical, synchronous, and predictive context yields nearly 1 dB gain on real AV conversations. Project page: this https URL.
#### Symbiotic Architecture for Post-Hoc Audio Extension of Frozen Language Models
 - **Authors:** Yotaro Kubo, Qi Sun, Yujin Tang
 - **Subjects:** Subjects:
Sound (cs.SD); Computation and Language (cs.CL); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2609.30784

 - **Pdf link:** https://arxiv.org/pdf/2609.30784

 - **Abstract**
 This paper proposes an architecture for equipping large language models (LLMs) with audio-understanding capabilities without fine-tuning their weights. The proposed symbiotic architecture employs an injector module that writes audio-conditioned vectors directly into the target LLM's short-term memory, i.e., the key-value (KV) cache, enabling the LLM to behave as an audio language model (ALM). The architectural advantages are twofold. First, it improves the scalability of ALMs: because the proposed method bypasses the LLM during audio injection, the injection cost is governed by the injector width rather than the backbone width, and can therefore scale more slowly than the cost of full-backbone prefilling. Second, since the training scheme does not update the LLM weights, the original capabilities of the LLM are preserved without the risk of degradation from fine-tuning. The effectiveness of the proposed method is evaluated on both audio-understanding tasks (automatic speech recognition, audio question answering, and acoustic scene classification) and text-only tasks. We confirm that, while activating fewer parameters during audio prefilling, our architecture outperforms the conventional method with a frozen LLM and approaches the performance of a fine-tuned ALM, all while preserving the backbone LLM's original text-only task performance by construction.
#### Subject-Invariant Cross-Modal Decoding of Perceived Speech from Brain Recordings
 - **Authors:** Aoke Zhang, Jing Chen
 - **Subjects:** Subjects:
Sound (cs.SD); Artificial Intelligence (cs.AI); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2609.30832

 - **Pdf link:** https://arxiv.org/pdf/2609.30832

 - **Abstract**
 Perceived speech decoding based on non-invasive brain-computer interface (BCI) signals has been extensively studied in recent years. Research in this field primarily faces two challenges: extracting neural representations with rich spatiotemporal information and achieving cross-subject generalization. Although separate studies have proposed methods to cope with these issues, a unified approach that simultaneously tackles both challenges remains lacking. To fill this gap, we propose the Subject-Invariant Cross-Modal Perceived Speech Decoding (SICMD) method, which integrates functional magnetic resonance imaging (fMRI) and magnetoencephalography (MEG). We conduct comprehensive analyses of the fusion method, fusion position, encoder architecture, and model inputs. Our results demonstrate that the proposed method improves Top-1, Top-10, and Rankacc by more than 10.6%, 10.1%, and 1.7%, respectively, compared to baseline methods in cross-subject perceived speech decoding tasks, while reducing training costs by 88.8% and 60.5% compared to multi-subject and intra-subject decoding settings. Further visualization experiments also confirm the effectiveness of our approach.
#### I-Parakeet: Integer-Only Conformer ASR on Mobile NPU
 - **Authors:** Taichi Nishimura
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2609.30846

 - **Pdf link:** https://arxiv.org/pdf/2609.30846

 - **Abstract**
 In this paper, we propose I-Parakeet, an integer-only implementation of NVIDIA's Parakeet-CTC (0.6B parameters) that runs on a smartphone NPU without any floating-point operator or CPU fallback. Modern Conformer ASR models are hard to deploy on edge devices because of their size, and quantized models still fall back to floating point for numerically sensitive operations. This prevents them from fully exploiting integer accelerators such as mobile NPUs. To achieve this, our contributions are threefold. First, we derive an integer formulation of the relative-positional self-attention at the core of the Conformer. We fuse its two score branches with different quantization scales and the relative shift into integer-only operations. Second, we introduce a minimax-optimized Swish approximation that minimizes the maximum error of the Swish output. Third, a layer-wise range analysis of activations yields two targeted remedies: an INT16 grid for the BatchNorm output and percentile calibration for the heavy-tailed pre-encoder activations. I-Parakeet achieves 4.97% WER on LibriSpeech test-other, running on a Qualcomm NPU at a real-time factor of 0.048, 7.5x faster than a CPU baseline.
#### Training-Free Pronunciation Transcription via Text-Constrained Acoustic Rescoring
 - **Authors:** Hikaru Asano, Yotaro Kubo, So Kuroki
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2609.30924

 - **Pdf link:** https://arxiv.org/pdf/2609.30924

 - **Abstract**
 Accurate and efficient pronunciation transcription is essential for preparing text-to-speech training data at scale. Existing approaches have different limitations: grapheme-to-pronunciation (G2P) and speech-to-pronunciation (S2P) methods each capture only partial information, using only text or only speech, while speech-and-text-to-pronunciation (ST2P) methods use both but require costly pronunciation-annotated data. To address this problem, we propose a training-free ST2P pipeline that integrates both lexical and acoustic information at inference time. Lexical resources and G2P tools generate text-constrained candidates, and a left-to-right greedy search selects the best one using whole-sequence negative log-likelihoods from frozen pretrained S2P models. On three Japanese corpora, our method reduces Character Error Rate (CER) from 0.60--1.40\% (text-only baseline) to 0.04--0.17\% with reference transcripts, and 0.64--1.58\% with ASR transcripts. It outperforms all baselines, including a trained ST2P model and commercial multimodal LLMs. Our greedy search method is 3--3.5$\times$ faster than beam search at similar CER, and the cascade is 2$\times$ faster than direct decoding ensuring the efficiency and accuracy. In Spanish, French, and preliminary English, it also surpasses four open multimodal LLMs and the best traditional methods.
#### Tracing and Relearning Detection Evidence in Text-to-Speech Systems
 - **Authors:** Eunji Shin, Kyudan Jung, Jihwan Kim, Minwoo Lee, Jaegul Choo
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2609.30983

 - **Pdf link:** https://arxiv.org/pdf/2609.30983

 - **Abstract**
 Recent audio deepfake detectors separate bona fide speech from synthetic speech, yet it remains unclear which stage of a text-to-speech system supplies the detection evidence. We address this with controlled resynthesis and detector adaptation in an F5-TTS-BigVGAN pipeline. Since vocoder reconstruction of a real mel can itself be separable from the source utterance, we fix the vocoder and trace the larger change in detector separation to acoustic generation. Adversarially fine-tuning the acoustic model, with no detector in its objective, raises EER against fixed detectors at comparable quality. However, adapting a detector only on the tuned model's VCTK outputs lowers its LibriSpeech EER from 19.42% to 7.46% and improves detection of unseen base F5-TTS outputs. These results suggest that acoustic-model updates can reduce the detection evidence available to fixed detectors, while detector adaptation keeps the updated outputs detectable in this pipeline.
#### BreathGRU: A Novel Semi-Supervised Bidirectional Gated Recurrent Unit Framework for Speech and Breath Segmentation for Respiratory Audio
 - **Authors:** Sania Fatima Sayed, John W. Holloway, Reyer Zwiggelaar, Faisal I. Rezwan
 - **Subjects:** Subjects:
Sound (cs.SD); Machine Learning (cs.LG); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2609.31165

 - **Pdf link:** https://arxiv.org/pdf/2609.31165

 - **Abstract**
 Speech-breath segmentation is a fundamental preprocessing step in respiratory audio analysis, enabling applications such as respiratory acoustic biomarker extraction, lung function prediction and disease monitoring. Existing approaches, including threshold methods, Fourier Transform-based techniques, and unsupervised and pretrained voice activity detection (VAD) models, primarily focus on speech detection and often classify breathing events as non-speech or silence, limiting their applicability for precise breath detection. To address this limitation, we propose BreathGRU, a semi-supervised Bidirectional Gated Recurrent Unit (BiGRU) framework specifically designed for speech-breath segmentation. The proposed framework combines frame-level acoustic feature extraction with bidirectional recurrent modelling, pseudo-label refinement and duration-constrained Segmental Viterbi decoding to produce speech and breath segmentation. BreathGRU was evaluated against the existing approaches, using manually annotated recordings. Performance was assessed using event-based, time-based, overlap-based, duration-based and boundary-based segmentation metrics. Experiment results demonstrated that BreathGRU achieved the highest breath event recall (0.83), the lowest onset-localisation error (0.14s) and the highest Mean Match Intersection over Union (0.81), with competitive overall segmentation performance compared to large pretrained VAD models like Silero. Qualitative evaluation on manually annotated recordings further showed close agreement between BreathGRU and manual annotation, with better breath detection compared to Silero. These findings demonstrate that explicit breath event modelling provides advantages over general-purpose VAD models and establish BreathGRU as an effective speech-breath segmentation framework which can be applied for respiratory audio analysis and pulmonary healthcare applications.
#### BAT-CLIP: Trimodal Alignment of Brain, Audio and Text
 - **Authors:** Suhyun Kim, Jinmo Han, Danny Dongyeop Han, Ahhyun Lucy Lee, Jewoon Lee, Yonghyeon Gwon, Zach Paris, Chun Kee Chung, Saewoong Bahk, Nam Soo Kim, Seong Jae Hwang, Jiook Cha
 - **Subjects:** Subjects:
Sound (cs.SD); Artificial Intelligence (cs.AI); Machine Learning (cs.LG); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2609.31180

 - **Pdf link:** https://arxiv.org/pdf/2609.31180

 - **Abstract**
 Decoding and interpreting naturalistic speech from the brain increasingly relies on alignment to pretrained speech and language representation spaces. However, current CLIP-style brain-speech alignment ground neural activity to a single anchor modality-audio or text-despite the brain's inherently multimodal speech processing. This induces a trade-off: audio anchoring preserves temporal structure but weakens linguistic separability, while text anchoring captures semantics yet discards acoustic detail. We propose BAT-CLIP, the first CLIP-style trimodal alignment framework for iEEG that jointly aligns neural embeddings to both pretrained audio and text anchors in a shared, frozen audio-text manifold. On the naturalistic Podcast benchmark, BAT-CLIP yields more robust representations than bimodal CLIP baselines. We also highlight the importance of using self-supervised foundation models for CLIP training.
#### Acoustic-to-Text KV Compression for Full-Duplex Speech Models
 - **Authors:** Yejin Lee, Seungbeom Kim, Yongha Lee, Kyuhong Shim
 - **Subjects:** Subjects:
Sound (cs.SD); Artificial Intelligence (cs.AI); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2609.31224

 - **Pdf link:** https://arxiv.org/pdf/2609.31224

 - **Abstract**
 Full-duplex speech language models continuously accumulate acoustic key-value (KV) states, making long-running interactions memory-intensive. During listening, the model can finish processing an audio unit before the next arrives; we term the remaining interval listening-time slack. We propose acoustic-to-text KV compression, which introduces a transcription side channel to convert incoming speech into compact textual memory within this interval. When the cache exceeds a target budget during inference, older acoustic states are evicted while transcripts and recent acoustic context remain. We train the side channel with LoRA using cross-entropy on transcription segments. To preserve listening and speaking behavior, we apply knowledge distillation to the original model's token-level output distributions at native prediction positions. On ten-minute LongSpeech sessions, our MiniCPM-o 4.5 implementation reduces peak streaming KV-cache size by 64.6% compared with the same model without eviction. The proposed method also improves transcription, temporal question answering, and summarization over the baseline. Full-Duplex-Bench evaluations further show comparable pause-handling, turn-taking, and interruption performance.


by Zyzzyva0381 (Windy). 


2026-09-28
