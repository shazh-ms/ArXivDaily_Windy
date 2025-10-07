# Showing new listings for Tuesday, 7 October 2025
Auto update papers at about 2:30am UTC (10:30am Beijing time) every weekday.


阅读 `Usage.md`了解如何使用此repo实现个性化的Arxiv论文推送

See `Usage.md` for instructions on how to personalize the repo. 


Keyword list: ['text-to-speech', 'text to speech', 'tts', 'LLM-based', 'speech', 'voice']


Excluded: []


### Today: 16papers 
#### Scaling Multi-Talker ASR with Speaker-Agnostic Activity Streams
 - **Authors:** Xiluo He, Alexander Polok, Jesús Villalba, Thomas Thebaud, Matthew Maciejewski
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2510.03630

 - **Pdf link:** https://arxiv.org/pdf/2510.03630

 - **Abstract**
 An increasingly common training paradigm for multi-talker automatic speech recognition (ASR) is to use speaker activity signals to adapt single-speaker ASR models for overlapping speech. Although effective, these systems require running the ASR model once per speaker, resulting in inference costs that scale with the number of speakers and limiting their practicality. In this work, we propose a method that decouples the inference cost of activity-conditioned ASR systems from the number of speakers by converting speaker-specific activity outputs into two speaker-agnostic streams. A central challenge is that naïvely merging speaker activities into streams significantly degrades recognition, since pretrained ASR models assume contiguous, single-speaker inputs. To address this, we design new heuristics aimed at preserving conversational continuity and maintaining compatibility with existing systems. We show that our approach is compatible with Diarization-Conditioned Whisper (DiCoW) to greatly reduce runtimes on the AMI and ICSI meeting datasets while retaining competitive performance.
#### Adapting Diarization-Conditioned Whisper for End-to-End Multi-Talker Speech Recognition
 - **Authors:** Martin Kocour, Martin Karafiat, Alexander Polok, Dominik Klement, Lukáš Burget, Jan Černocký
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Computation and Language (cs.CL); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2510.03723

 - **Pdf link:** https://arxiv.org/pdf/2510.03723

 - **Abstract**
 We propose a speaker-attributed (SA) Whisper-based model for multi-talker speech recognition that combines target-speaker modeling with serialized output training (SOT). Our approach leverages a Diarization-Conditioned Whisper (DiCoW) encoder to extract target-speaker embeddings, which are concatenated into a single representation and passed to a shared decoder. This enables the model to transcribe overlapping speech as a serialized output stream with speaker tags and timestamps. In contrast to target-speaker ASR systems such as DiCoW, which decode each speaker separately, our approach performs joint decoding, allowing the decoder to condition on the context of all speakers simultaneously. Experiments show that the model outperforms existing SOT-based approaches and surpasses DiCoW on multi-talker mixtures (e.g., LibriMix).
#### A MATLAB toolbox for Computation of Speech Transmission Index (STI)
 - **Authors:** Pavel Rajmic, Jiří Schimmel, Šimon Cieslar
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2510.03825

 - **Pdf link:** https://arxiv.org/pdf/2510.03825

 - **Abstract**
 The speech transmission index (STI) is a popular simple metric for the prediction of speech intelligibility when speech is passed through a transmission channel. Computation of STI from acoustic measurements is described in the IEC 60268-16:2020 standard. Though, reliable implementations of STI are not publicly accessible and are frequently limited to the use with a proprietary measurement hardware. We present a Matlab STI implementation of both the direct and indirect approaches according to the standard, including the shortened STIPA protocol. The suggested implementation meets prescribed requirements, as evidenced by tests on reference signals. Additionally, we conducted a verification measurement in comparison to a commercial measurement device. Our software comes with open source code.
#### A Multilingual Framework for Dysarthria: Detection, Severity Classification, Speech-to-Text, and Clean Speech Generation
 - **Authors:** Ananya Raghu, Anisha Raghu, Nithika Vivek, Sofie Budman, Omar Mansour
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2510.03986

 - **Pdf link:** https://arxiv.org/pdf/2510.03986

 - **Abstract**
 Dysarthria is a motor speech disorder that results in slow and often incomprehensible speech. Speech intelligibility significantly impacts communication, leading to barriers in social interactions. Dysarthria is often a characteristic of neurological diseases including Parkinson's and ALS, yet current tools lack generalizability across languages and levels of severity. In this study, we present a unified AI-based multilingual framework that addresses six key components: (1) binary dysarthria detection, (2) severity classification, (3) clean speech generation, (4) speech-to-text conversion, (5) emotion detection, and (6) voice cloning. We analyze datasets in English, Russian, and German, using spectrogram-based visualizations and acoustic feature extraction to inform model training. Our binary detection model achieved 97% accuracy across all three languages, demonstrating strong generalization across languages. The severity classification model also reached 97% test accuracy, with interpretable results showing model attention focused on lower harmonics. Our translation pipeline, trained on paired Russian dysarthric and clean speech, reconstructed intelligible outputs with low training (0.03) and test (0.06) L1 losses. Given the limited availability of English dysarthric-clean pairs, we fine-tuned the Russian model on English data and achieved improved losses of 0.02 (train) and 0.03 (test), highlighting the promise of cross-lingual transfer learning for low-resource settings. Our speech-to-text pipeline achieved a Word Error Rate of 0.1367 after three epochs, indicating accurate transcription on dysarthric speech and enabling downstream emotion recognition and voice cloning from transcribed speech. Overall, the results and products of this study can be used to diagnose dysarthria and improve communication and understanding for patients across different languages.
#### MoME: Mixture of Matryoshka Experts for Audio-Visual Speech Recognition
 - **Authors:** Umberto Cappellazzo, Minsu Kim, Pingchuan Ma, Honglie Chen, Xubo Liu, Stavros Petridis, Maja Pantic
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Computer Vision and Pattern Recognition (cs.CV); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2510.04136

 - **Pdf link:** https://arxiv.org/pdf/2510.04136

 - **Abstract**
 Large language models (LLMs) have recently shown strong potential in audio-visual speech recognition (AVSR), but their high computational demands and sensitivity to token granularity limit their practicality in resource-constrained settings. Token compression methods can reduce inference cost, but they require fixing a compression rate in advance and produce a single fixed-length output, offering no flexibility to balance information density and efficiency at inference time. Matryoshka representation learning (MRL) addresses this by enabling a single model to operate across multiple token granularities, allowing compression rates to be adjusted dynamically. However, current MRL-based methods treat each scale independently during training, limiting cross-scale generalization, robustness at high compression, and interpretability. To overcome these limitations, we propose MoME (Mixture of Matryoshka Experts), a novel framework that integrates sparse Mixture-of-Experts (MoE) into MRL-based LLMs for AVSR. MoME augments a frozen LLM with top-k routed and shared experts, allowing dynamic capacity allocation across scales and modalities. A shared router promotes consistent expert activation across granularities, enabling compressed sequences to benefit from representations learned at lower compression. Experiments on LRS2 and LRS3 demonstrate that MoME achieves state-of-the-art performance across AVSR, ASR, and VSR tasks, while requiring significantly fewer parameters and maintaining robustness under noise. MoME unifies the adaptability of MRL with the efficiency of MoE, offering a scalable and interpretable solution for resource-aware speech recognition.
#### Drax: Speech Recognition with Discrete Flow Matching
 - **Authors:** Aviv Navon, Aviv Shamsian, Neta Glazer, Yael Segal-Feldman, Gill Hetz, Joseph Keshet, Ethan Fetaya
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Machine Learning (cs.LG); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2510.04162

 - **Pdf link:** https://arxiv.org/pdf/2510.04162

 - **Abstract**
 Diffusion and flow-based non-autoregressive (NAR) models have shown strong promise in large language modeling, however, their potential for automatic speech recognition (ASR) remains largely unexplored. We propose Drax, a discrete flow matching framework for ASR that enables efficient parallel decoding. To better align training with inference, we construct an audio-conditioned probability path that guides the model through trajectories resembling likely intermediate inference errors, rather than direct random noise to target transitions. Our theoretical analysis links the generalization gap to divergences between training and inference occupancies, controlled by cumulative velocity errors, thereby motivating our design choice. Empirical evaluation demonstrates that our approach attains recognition accuracy on par with state-of-the-art speech models while offering improved accuracy-efficiency trade-offs, highlighting discrete flow matching as a promising direction for advancing NAR ASR.
#### Probing Whisper for Dysarthric Speech in Detection and Assessment
 - **Authors:** Zhengjun Yue, Devendra Kayande, Zoran Cvetkovic, Erfan Loweimi
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2510.04219

 - **Pdf link:** https://arxiv.org/pdf/2510.04219

 - **Abstract**
 Large-scale end-to-end models such as Whisper have shown strong performance on diverse speech tasks, but their internal behavior on pathological speech remains poorly understood. Understanding how dysarthric speech is represented across layers is critical for building reliable and explainable clinical assessment tools. This study probes the Whisper-Medium model encoder for dysarthric speech for detection and assessment (i.e., severity classification). We evaluate layer-wise embeddings with a linear classifier under both single-task and multi-task settings, and complement these results with Silhouette scores and mutual information to provide perspectives on layer informativeness. To examine adaptability, we repeat the analysis after fine-tuning Whisper on a dysarthric speech recognition task. Across metrics, the mid-level encoder layers (13-15) emerge as most informative, while fine-tuning induces only modest changes. The findings improve the interpretability of Whisper's embeddings and highlight the potential of probing analyses to guide the use of large-scale pretrained models for pathological speech.
#### UniVoice: Unifying Autoregressive ASR and Flow-Matching based TTS with Large Language Models
 - **Authors:** Wenhao Guan, Zhikang Niu, Ziyue Jiang, Kaidi Wang, Peijie Chen, Qingyang Hong, Lin Li, Xie Chen
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2510.04593

 - **Pdf link:** https://arxiv.org/pdf/2510.04593

 - **Abstract**
 Large language models (LLMs) have demonstrated promising performance in both automatic speech recognition (ASR) and text-to-speech (TTS) systems, gradually becoming the mainstream approach. However, most current approaches address these tasks separately rather than through a unified framework. This work aims to integrate these two tasks into one unified model. Although discrete speech tokenization enables joint modeling, its inherent information loss limits performance in both recognition and generation. In this work, we present UniVoice, a unified LLM framework through continuous representations that seamlessly integrates speech recognition and synthesis within a single model. Our approach combines the strengths of autoregressive modeling for speech recognition with flow matching for high-quality generation. To mitigate the inherent divergence between autoregressive and flow-matching models, we further design a dual attention mechanism, which switches between a causal mask for recognition and a bidirectional attention mask for synthesis. Furthermore, the proposed text-prefix-conditioned speech infilling method enables high-fidelity zero-shot voice cloning. Experimental results demonstrate that our method can achieve or exceed current single-task modeling methods in both ASR and zero-shot TTS tasks. This work explores new possibilities for end-to-end speech understanding and generation.
#### Perceptual Evaluation of Extrapolated Spatial Room Impulse Responses From a Mono Source
 - **Authors:** Ben Heritage, Fiona Ryder, Michael McLoughlin, Karolina Prawda
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2510.04937

 - **Pdf link:** https://arxiv.org/pdf/2510.04937

 - **Abstract**
 Immersion in virtual and augmented reality solutions is reliant on plausible spatial audio. However, plausibly representing a space for immersive audio often requires many individual acoustic measurements of source-microphone pairs with specialist spatial microphones, making the procedure time-consuming and expensive. In this study, we evaluate the plausibility of extrapolated and spatialised Room Impulse Responses (RIRs) by using a 3-Alternative Forced Choice (3AFC) listening test. The stimuli comprised of RIRs from three spaces convolved with speech, orchestral, and instrumental music. When asked to select which stimuli was artificial out of one extrapolated and two real stimuli, an overall accuracy of 38% was achieved from 20 participants (5 percentage points above the expected guessing rate). Given the listening test result, this study shows that it is possible to extrapolate plausible spatial RIRs from mono measurements, decreasing the need for time and specialist equipment in acoustic measurements.
#### MuFFIN: Multifaceted Pronunciation Feedback Model with Interactive Hierarchical Neural Modeling
 - **Authors:** Bi-Cheng Yan, Ming-Kang Tsai, Berlin Chen
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Artificial Intelligence (cs.AI)
 - **Arxiv link:** https://arxiv.org/abs/2510.04956

 - **Pdf link:** https://arxiv.org/pdf/2510.04956

 - **Abstract**
 Computer-assisted pronunciation training (CAPT) manages to facilitate second-language (L2) learners to practice pronunciation skills by offering timely and instructive feedback. To examine pronunciation proficiency from multiple facets, existing methods for CAPT broadly fall into two categories: mispronunciation detection and diagnosis (MDD) as well as automatic pronunciation assessment (APA). The former aims to pinpoint phonetic pronunciation errors and provide diagnostic feedback, while the latter seeks instead to quantify pronunciation proficiency pertaining to various aspects. Despite the natural complementarity between MDD and APA, researchers and practitioners, however, often treat them as independent tasks with disparate modeling paradigms. In light of this, we in this paper first introduce MuFFIN, a Multi-Faceted pronunciation Feedback model with an Interactive hierarchical Neural architecture, to jointly address the tasks of MDD and APA. To better capture the nuanced distinctions between phonemes in the feature space, a novel phoneme-contrastive ordinal regularization mechanism is then put forward to optimize the proposed model to generate more phoneme-discriminative features while factoring in the ordinality of the aspect scores. In addition, to address the intricate data imbalance problem in MDD, we design a simple yet effective training objective, which is specifically tailored to perturb the outputs of a phoneme classifier with the phoneme-specific variations, so as to better render the distribution of predicted phonemes meanwhile considering their mispronunciation characteristics. A series of experiments conducted on the Speechocean762 benchmark dataset demonstrates the efficacy of our method in relation to several cutting-edge baselines, showing state-of-the-art performance on both the APA and MDD tasks.
#### Audio Forensics Evaluation (SAFE) Challenge
 - **Authors:** Kirill Trapeznikov, Paul Cummer, Pranay Pherwani, Jai Aslam, Michael S. Davinroy, Peter Bautista, Laura Cassani, Matthew Stamm, Jill Crisman
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2510.03387

 - **Pdf link:** https://arxiv.org/pdf/2510.03387

 - **Abstract**
 The increasing realism of synthetic speech generated by advanced text-to-speech (TTS) models, coupled with post-processing and laundering techniques, presents a significant challenge for audio forensic detection. In this paper, we introduce the SAFE (Synthetic Audio Forensics Evaluation) Challenge, a fully blind evaluation framework designed to benchmark detection models across progressively harder scenarios: raw synthetic speech, processed audio (e.g., compression, resampling), and laundered audio intended to evade forensic analysis. The SAFE challenge consisted of a total of 90 hours of audio and 21,000 audio samples split across 21 different real sources and 17 different TTS models and 3 tasks. We present the challenge, evaluation design and tasks, dataset details, and initial insights into the strengths and limitations of current approaches, offering a foundation for advancing synthetic audio detection research. More information is available at \href{this https URL}{this https URL}.
#### Cross-Lingual Multi-Granularity Framework for Interpretable Parkinson's Disease Diagnosis from Speech
 - **Authors:** Ilias Tougui, Mehdi Zakroum, Mounir Ghogho
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2510.03758

 - **Pdf link:** https://arxiv.org/pdf/2510.03758

 - **Abstract**
 Parkinson's Disease (PD) affects over 10 million people worldwide, with speech impairments in up to 89% of patients. Current speech-based detection systems analyze entire utterances, potentially overlooking the diagnostic value of specific phonetic elements. We developed a granularity-aware approach for multilingual PD detection using an automated pipeline that extracts time-aligned phonemes, syllables, and words from recordings. Using Italian, Spanish, and English datasets, we implemented a bidirectional LSTM with multi-head attention to compare diagnostic performance across the different granularity levels. Phoneme-level analysis achieved superior performance with AUROC of 93.78% +- 2.34% and accuracy of 92.17% +- 2.43%. This demonstrates enhanced diagnostic capability for cross-linguistic PD detection. Importantly, attention analysis revealed that the most informative speech features align with those used in established clinical protocols: sustained vowels (/a/, /e/, /o/, /i/) at phoneme level, diadochokinetic syllables (/ta/, /pa/, /la/, /ka/) at syllable level, and /pataka/ sequences at word level. Source code will be available at this https URL.
#### GDiffuSE: Diffusion-based speech enhancement with noise model guidance
 - **Authors:** Efrayim Yanir, David Burshtein, Sharon Gannot
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2510.04157

 - **Pdf link:** https://arxiv.org/pdf/2510.04157

 - **Abstract**
 This paper introduces a novel speech enhancement (SE) approach based on a denoising diffusion probabilistic model (DDPM), termed Guided diffusion for speech enhancement (GDiffuSE). In contrast to conventional methods that directly map noisy speech to clean speech, our method employs a lightweight helper model to estimate the noise distribution, which is then incorporated into the diffusion denoising process via a guidance mechanism. This design improves robustness by enabling seamless adaptation to unseen noise types and by leveraging large-scale DDPMs originally trained for speech generation in the context of SE. We evaluate our approach on noisy signals obtained by adding noise samples from the BBC sound effects database to LibriSpeech utterances, showing consistent improvements over state-of-the-art baselines under mismatched noise conditions. Examples are available at our project webpage.
#### Machine Unlearning in Speech Emotion Recognition via Forget Set Alone
 - **Authors:** Zhao Ren, Rathi Adarshi Rammohan, Kevin Scheck, Tanja Schultz
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2510.04251

 - **Pdf link:** https://arxiv.org/pdf/2510.04251

 - **Abstract**
 Speech emotion recognition aims to identify emotional states from speech signals and has been widely applied in human-computer interaction, education, healthcare, and many other fields. However, since speech data contain rich sensitive information, partial data can be required to be deleted by speakers due to privacy concerns. Current machine unlearning approaches largely depend on data beyond the samples to be forgotten. However, this reliance poses challenges when data redistribution is restricted and demands substantial computational resources in the context of big data. We propose a novel adversarial-attack-based approach that fine-tunes a pre-trained speech emotion recognition model using only the data to be forgotten. The experimental results demonstrate that the proposed approach can effectively remove the knowledge of the data to be forgotten from the model, while preserving high model performance on the test set for emotion recognition.
#### Evaluating Self-Supervised Speech Models via Text-Based LLMS
 - **Authors:** Takashi Maekaku, Keita Goto, Jinchuan Tian, Yusuke Shinohara, Shinji Watanabe
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2510.04463

 - **Pdf link:** https://arxiv.org/pdf/2510.04463

 - **Abstract**
 Self-Supervised Learning (SSL) has gained traction for its ability to learn rich representations with low labeling costs, applicable across diverse downstream tasks. However, assessing the downstream-task performance remains challenging due to the cost of extra training and evaluation. Existing methods for task-agnostic evaluation also require extra training or hyperparameter tuning. We propose a novel evaluation metric using large language models (LLMs). By inputting discrete token sequences and minimal domain cues derived from SSL models into LLMs, we obtain the mean log-likelihood; these cues guide in-context learning, rendering the score more reliable without extra training or hyperparameter tuning. Experimental results show a correlation between LLM-based scores and automatic speech recognition task. Additionally, our findings reveal that LLMs not only functions as an SSL evaluation tools but also provides inference-time embeddings that are useful for speaker verification task.
#### Speak, Edit, Repeat: High-Fidelity Voice Editing and Zero-Shot TTS with Cross-Attentive Mamba
 - **Authors:** Baher Mohammad, Magauiya Zhussip, Stamatios Lefkimmiatis
 - **Subjects:** Subjects:
Sound (cs.SD); Artificial Intelligence (cs.AI); Computation and Language (cs.CL); Machine Learning (cs.LG); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2510.04738

 - **Pdf link:** https://arxiv.org/pdf/2510.04738

 - **Abstract**
 We introduce MAVE (Mamba with Cross-Attention for Voice Editing and Synthesis), a novel autoregressive architecture for text-conditioned voice editing and high-fidelity text-to-speech (TTS) synthesis, built on a cross-attentive Mamba backbone. MAVE achieves state-of-the-art performance in speech editing and very competitive results in zero-shot TTS, while not being explicitly trained on the latter task, outperforming leading autoregressive and diffusion models on diverse, real-world audio. By integrating Mamba for efficient audio sequence modeling with cross-attention for precise text-acoustic alignment, MAVE enables context-aware voice editing with exceptional naturalness and speaker consistency. In pairwise human evaluations on a random 40-sample subset of the RealEdit benchmark (400 judgments), 57.2% of listeners rated MAVE - edited speech as perceptually equal to the original, while 24.8% prefered the original and 18.0% MAVE - demonstrating that in the majority of cases edits are indistinguishable from the source. MAVE compares favorably with VoiceCraft and FluentSpeech both on pairwise comparisons and standalone mean opinion score (MOS) evaluations. For zero-shot TTS, MAVE exceeds VoiceCraft in both speaker similarity and naturalness, without requiring multiple inference runs or post-processing. Remarkably, these quality gains come with a significantly lower memory cost and approximately the same latency: MAVE requires ~6x less memory than VoiceCraft during inference on utterances from the RealEdit database (mean duration: 6.21s, A100, FP16, batch size 1). Our results demonstrate that MAVE establishes a new standard for flexible, high-fidelity voice editing and synthesis through the synergistic integration of structured state-space modeling and cross-modal attention.


by Zyzzyva0381 (Windy). 


2025-10-07
