# Showing new listings for Friday, 18 September 2026
Auto update papers at about 2:30am UTC (10:30am Beijing time) every weekday.


阅读 `Usage.md`了解如何使用此repo实现个性化的Arxiv论文推送

See `Usage.md` for instructions on how to personalize the repo. 


Keyword list: ['text-to-speech', 'text to speech', 'tts', 'LLM-based', 'speech', 'voice']


Excluded: []


### Today: 10papers 
#### Decaf: A privacy preserving speech codec using speaker disentanglement and canonical voice conversion
 - **Authors:** Md Shakhrul Iman Siam, Dushyant Sharma, Stanislav Yu. Kruchinin, Peter Skala
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2609.19304

 - **Pdf link:** https://arxiv.org/pdf/2609.19304

 - **Abstract**
 We present DECAF, a privacy preserving neural speech codec that obfuscates a speaker's voice while preserving linguistic content while maintaining automatic speech recognition (ASR) performance at very low bitrates, inspired by decaffeination. At the transmitter end, speech is encoded into speaker independent content embeddings, which are compressed using residual vector quantization and transmitted without any speaker related information. At the receiver, a canonical speaker embedding, shared a priori between endpoints, is used for waveform reconstruction, enabling deterministic and consistent obfuscation of a speaker's voice. The proposed framework leverages an information bottleneck applied to self supervised representations, along with a separate speaker embedding branch, to achieve effective speaker content disentanglement. We further incorporate a CTC-based auxiliary objective, encouraging content representations that are well aligned with downstream ASR tasks. We show that DECAF operating at a bit rate of 0.5 kbps achieves an Equal Error Rate (EER) of up to 43.5% for a speaker verification system, while maintaining competitive ASR performance, yielding a relative reduction in word error rate of 33.2% compared to a state of the art method.
#### PersianVox: A Prosody-Aware Approach for Speech Dataset Generation from In-the-Wild Data
 - **Authors:** Saeedreza Zouashkiani, Soheil Khalesi, Saman Soleimani Roudi, Sajjad Amini, Shahrokh Ghaemmaghami
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2609.19324

 - **Pdf link:** https://arxiv.org/pdf/2609.19324

 - **Abstract**
 Advancement of zero-shot text-to-speech synthesis is currently hindered for low-resource languages by the scarcity of large-scale, high-fidelity speech datasets. Traditional alignment-based methods require rare verbatim transcripts, while standard in-the-wild pipelines often rely on single-model automatic speech recognition and silence-based segmentation, leading to transcription errors and truncated prosody. To address these challenges for the Persian language, this paper introduces PersianVox, a fully automated pipeline designed to generate high-quality speech corpora from unlabeled web data. Our approach integrates a novel prosody-aware segmentation strategy that utilizes acoustic turn-detection to preserve linguistic completeness and optimize utterance duration for long-context modeling. Furthermore, we employ a dual-model agreement mechanism, leveraging two distinct model architectures to filter unreliable transcriptions without ground truth. This pipeline yields a 2,400-hour multi-speaker dataset, the largest open-source speech resource available for Persian to date. Additionally, we provide the first comparative benchmark of speech quality assessment methods for Persian, releasing a human-annotated subset to facilitate future research.
#### Multimodal Conversational Context for LLM-Based ASR: Data Construction, Training, and Benchmark
 - **Authors:** Longhao Li, Jian Tang, Yuxiang Kong, Jie Chen, Binbin Zhang, Lei Xie, Xiangang Li
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2609.19765

 - **Pdf link:** https://arxiv.org/pdf/2609.19765

 - **Abstract**
 Conversational context provides semantic and acoustic cues across turns for automatic speech recognition (ASR), but relying on historical transcripts can propagate recognition errors and discard pronunciation and speaker information. We present a multimodal conversational-context framework for LLM-based ASR that integrates a scenario-controlled data pipeline, scalable multimodal context training, and systematic evaluation. We construct dialogues around entities and their confusable forms and interleave historical user speech with assistant text responses for supervised fine-tuning. We also introduce MM-ContextASR Bench, which evaluates contextual understanding and entity error correction across five scenarios. Experiments with Qwen3-Omni and Step-Audio-2-mini reveal limitations in handling irrelevant and erroneous history and show that our data construction and training improve context utilization, with multimodal context achieving the highest overall entity recall on both models. Further experiments on accent, dialect, and target-speaker ASR demonstrate the value of historical speech. The benchmark data and evaluation code are publicly available at this https URL.
#### Consensus-Guided Shared-Specific Tri-View Learning for Speech Emotion Recognition
 - **Authors:** Bing Huang, Yujian Ma, Xikun Lu, Xianquan Jiang, Jinqiu Sang
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2609.19826

 - **Pdf link:** https://arxiv.org/pdf/2609.19826

 - **Abstract**
 Speech emotion recognition (SER) benefits from heterogeneous acoustic representations, but views derived from the same utterance contain both overlapping emotional evidence and representation-dependent cues. Direct fusion may therefore propagate redundant information or obscure complementary details. To address this issue, we propose Tri-view Consensus-Guided Fusion (TriCGF) for jointly modeling spectrogram, Mel-frequency cepstral coefficients, and HuBERT representations. TriCGF organizes each view into common and view-specific components before fusion. Cross-view Consensus Learning aggregates the common components into a global reference, while View-wise Gated Integration adaptively combines this reference with each view-specific component. A soft difference regularizer further discourages excessive information overlap. Under speaker-independent evaluation, TriCGF achieves 74.19% weighted accuracy (WA) and 75.17% unweighted accuracy (UA) on IEMOCAP, and 94.36% WA and 94.28% UA on EmoDB, outperforming representative SER methods on both datasets.
#### Foreground Voice Activity Detection: Learning Speaker Selectivity from Supervision
 - **Authors:** Guangzhao Yang, Muhammad Huzaifah, Yu Pan, Jinya Sakurai, Ningjie Bai
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2609.19856

 - **Pdf link:** https://arxiv.org/pdf/2609.19856

 - **Abstract**
 Voice activity detection (VAD) fronts most voice-agent pipelines, yet production detectors treat all human speech, background talkers included, as valid activity; in crowded settings this floods recognition, stalls turn-taking, and triggers false barge-in. We formalize Foreground VAD (FVAD): a frame-synchronous, enrollment-free task in which only the dominant speaker, defined by sustained presence rather than instantaneous loudness, is positive, and which reduces to conventional VAD when a single speaker is present. We show that foreground selectivity is largely governed by training supervision: the crucial ingredient is an augmentation recipe pairing foreground-only labels with competing-speaker mixing, generated fully automatically without human annotation. To quantify selectivity we introduce the Background False-Alarm Rate (BG-FAR), gated by foreground F1, and build a controlled benchmark, Mix-Interference, complemented by an adapted VOiCES for real-world far-field evaluation. Across equal-size backbones, Mamba and LSTM perform on par while a longer-context attention model is no better, suggesting that training supervision plays a substantially larger role than temporal modeling capacity in achieving foreground selectivity. The resulting lightweight streaming model, Mamba-FVAD, outperforms commercial VADs and enrollment-based speaker-aware systems in foreground selectivity while staying competitive on conventional VAD, at 1-2 ms per-frame CPU latency.
#### Alignment-Path Distillation from Non-streaming ASR-LLMs for Streaming Speech Recognition
 - **Authors:** Yan Jia, Kai Huang, Junjie Chen, Feng-Long Xie, Xu Tang, Yao Hu
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2609.20121

 - **Pdf link:** https://arxiv.org/pdf/2609.20121

 - **Abstract**
 In this paper, we propose an alignment-path distillation framework for streaming automatic speech recognition (ASR) with large language models (LLMs). Interleaved streaming ASR-LLMs use forced alignments (FA) from alignment models, such as those trained with connectionist temporal classification (CTC), to construct speech-text training sequences. However, alignments obtained from a separate acoustic model may be inconsistent with those learned by LLM-based ASR. This motivates us to transfer alignment information from a non-streaming ASR-LLM to improve streaming recognition. Specifically, we extract monotonic alignment paths from a non-streaming teacher's soft text-audio attention and use them to construct interleaved training sequences. The framework also includes logit and hidden-state distillation to learn from the teacher's output distributions and internal representations. Experimental results show that, without logit or hidden-state distillation, training with teacher-derived alignment paths achieves a 5.2% relative error rate reduction compared with training using forced alignments. When both models use logit and hidden-state distillation, teacher-derived alignments yield a 3.9% relative error rate reduction, with similar mean emission latency but higher flicker. The complete framework achieves a 16.6% relative error rate reduction compared with training using forced alignments without logit or hidden-state distillation.
#### Model-Agnostic and Language-Agnostic Voice Pipeline Improvement for the Agriculture Domain
 - **Authors:** Aakash Singh, Lakshmi Pedapudi, Chandrashekar M S, Sanyam Singh, Naga Ganesh, Vineet Singh
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Artificial Intelligence (cs.AI); Computation and Language (cs.CL); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2609.20504

 - **Pdf link:** https://arxiv.org/pdf/2609.20504

 - **Abstract**
 FarmerChat is Digital Green's AI-powered agricultural advisory assistant for smallholder farmers, who access it in their own language through text, voice, or photographs. Voice is a critical channel for this population, yet field-recorded speech is challenging for general-purpose automatic speech recognition (ASR) because recordings frequently contain machinery noise, background media, competing speakers, and domain-specific agricultural vocabulary. These conditions disproportionately affect crop, pest, chemical, and quantity terms that carry the meaning of a farmer's query. We present a modular, model-agnostic pipeline for improving ASR quality in FarmerChat without fine-tuning or replacing the underlying ASR model. The pipeline combines gated audio enhancement, speaker diarization and target-speaker selection, ASR, domain-aware correction using a weighted agricultural lexicon, and a quality gate for detecting unreliable transcripts. Only the diarization stage is fine-tuned; all other stages use off-the-shelf models behind common interfaces. We evaluate the pipeline on human-annotated FarmerChat recordings in Hindi, Telugu, and Odia using word error rate (WER) and a domain-weighted error rate that gives greater importance to agricultural terminology. The largest improvements occur on multi-speaker recordings, where target-speaker selection prevents competing speech from entering the transcript. Across the full corpus, the pipeline reduces WER by 16-23% relative on three cloud ASR models and by 5% on an on-device model. On multi-speaker recordings, the reductions are 32-42% for the cloud models and 16% for the on-device model. All reported reductions are statistically significant. These results show that targeted preprocessing, speaker selection, and domain-aware post-processing can substantially improve agricultural speech transcription while preserving the underlying ASR model.
#### A Deep Neural Network for Predicting Continuous Human EEG Across the Auditory Pathway in Response to Sound
 - **Authors:** Thomas J Stoll, Ross K Maddox
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2609.20595

 - **Pdf link:** https://arxiv.org/pdf/2609.20595

 - **Abstract**
 Computational models of auditory physiology commonly target specific responses or stages of the auditory pathway, limiting their ability to integrate findings across experimental paradigms and neural timescales. We present a foundation model of human auditory electrophysiology: a causal neural network trained to map binaural acoustic waveforms directly to high-sample-rate EEG. The model was trained on approximately 250 hours of EEG data from 92 subjects, with varied electrode montages and stimuli spanning tonebursts, speech, and music. We tested whether the model recovered effects of stimulus rate, frequency, and presentation method on auditory brainstem responses (ABRs); subcortical and cortical temporal response functions (TRFs) to continuous speech; and the click-evoked binaural interaction component (BIC). Predicted ABRs and TRFs reproduced established response morphology and stimulus-dependent effects, with model-grand-average correlations falling within the corresponding subject-level human distributions. The model-predicted BIC metrics closely resembled the values reported in the literature. These findings demonstrate that a single audio-to-EEG model can capture auditory physiology across paradigms and timescales, supporting future in silico experimentation and hearing technology applications.
#### VākQA: A Benchmark and Evaluation Study for Telugu Spoken Factoid Question Answering
 - **Authors:** Bhavana Akkiraju, Ravi Sastry Kolluru, Sri Charan D, Srihari Bandarupalli, Santosh Kesiraju, Anil Vuppala
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2609.19879

 - **Pdf link:** https://arxiv.org/pdf/2609.19879

 - **Abstract**
 Question answering has advanced rapidly with large language models, but predominantly for high-resource languages, in both text and spoken settings. Spoken question answering (SQA) benchmark for Telugu remains unexplored, and the reliability of automatic evaluation in this setting remains unquantified. We introduce VākQA, a Telugu SQA benchmark of 2,001 factoid question-answer pairs across six domains, with 2.53 hours of speech audio, bilingual transcriptions, and human-verified reference answers. We first validate evaluation methods against human judgements: Gemini-as-a-judge best approximates human ratings but is non-uniformly strict, while open-weight judges systematically penalize correct Telugu answers that differ in surface form from the reference. Using this validated setup, we benchmark proprietary and open-weight models across input modality, language, and domain. We observe that Telugu phrasing retains cultural specificity that is lost in translation, speech input introduces phonetic confusions that alter question meaning, and cascaded ASR-MT errors compound progressively. VākQA is publicly released.
#### Reading Emotions in the Token Space: Discriminative Adaptation of SpeechLLMs for Emotion Recognition
 - **Authors:** Hasindri Watawana, Sergio Burdisso, Esaú Villatoro-Tello, Manjunath K E, Kadri Hacioglu, Petr Motlicek, Andreas Stolcke
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Artificial Intelligence (cs.AI); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2609.20081

 - **Pdf link:** https://arxiv.org/pdf/2609.20081

 - **Abstract**
 SpeechLLMs have shown strong potential for emotion recognition, yet they read the predicted emotion off a generative decoder not suited for classification: it can emit labels outside the target set and favors frequent classes. We propose a discriminative adaptation that reads the final prompt token's hidden state through a classification head, producing a label in one forward pass without modifying the backbone. Because this readout starts from the hidden state the model would otherwise decode, it gives a controlled comparison of generative and discriminative inference in an otherwise identical speechLLM. We keep the head a single linear layer, trading little accuracy for interpretability: each emotion becomes one direction in the LLM output token space, revealing associated tokens. On IEMOCAP, across two speechLLM architectures, it improves Macro F1 and removes hallucinations, with largest gains on realistic ASR transcripts. Our analysis reveals that these emotion directions encode indirect associations mirroring biases in web-scale text.


by Zyzzyva0381 (Windy). 


2026-09-18
