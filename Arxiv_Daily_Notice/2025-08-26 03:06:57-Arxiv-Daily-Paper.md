# Showing new listings for Tuesday, 26 August 2025
Auto update papers at about 2:30am UTC (10:30am Beijing time) every weekday.


阅读 `Usage.md`了解如何使用此repo实现个性化的Arxiv论文推送

See `Usage.md` for instructions on how to personalize the repo. 


Keyword list: ['text-to-speech', 'text to speech', 'tts', 'LLM-based', 'speech', 'voice']


Excluded: []


### Today: 10papers 
#### Localization using Angle-of-Arrival Triangulation
 - **Authors:** Amod K. Agrawal
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Human-Computer Interaction (cs.HC); Networking and Internet Architecture (cs.NI); Sound (cs.SD); Signal Processing (eess.SP)
 - **Arxiv link:** https://arxiv.org/abs/2508.16908

 - **Pdf link:** https://arxiv.org/pdf/2508.16908

 - **Abstract**
 Indoor localization is a long-standing challenge in mobile computing, with significant implications for enabling location-aware and intelligent applications within smart environments such as homes, offices, and retail spaces. As AI assistants such as Amazon Alexa and Google Nest become increasingly pervasive, microphone-equipped devices are emerging as key components of everyday life and home automation. This paper introduces a passive, infrastructure-light system for localizing human speakers using speech signals captured by two or more spatially distributed smart devices. The proposed approach, GCC+, extends the Generalized Cross-Correlation with Phase Transform (GCC-PHAT) method to estimate the Angle-of-Arrival (AoA) of audio signals at each device and applies robust triangulation techniques to infer the speaker's two-dimensional position. To further improve temporal resolution and localization accuracy, feature-space expansion and subsample interpolation techniques are employed for precise Time Difference of Arrival (TDoA) estimation. The system operates without requiring hardware modifications, prior calibration, explicit user cooperation, or knowledge of the speaker's signal content, thereby offering a highly practical solution for real-world deployment. Experimental evaluation in a real-world home environment yields a median AoA estimation error of 2.2 degrees and a median localization error of 1.25 m, demonstrating the feasibility and effectiveness of audio-based localization for enabling context-aware, privacy-preserving ambient intelligence.
#### Pinhole Effect on Linkability and Dispersion in Speaker Anonymization
 - **Authors:** Kong Aik Lee, Zeyan Liu, Liping Chen, Zhenhua Ling
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2508.17134

 - **Pdf link:** https://arxiv.org/pdf/2508.17134

 - **Abstract**
 Speaker anonymization aims to conceal speaker-specific attributes in speech signals, making the anonymized speech unlinkable to the original speaker identity. Recent approaches achieve this by disentangling speech into content and speaker components, replacing the latter with pseudo speakers. The anonymized speech can be mapped either to a common pseudo speaker shared across utterances or to distinct pseudo speakers unique to each utterance. This paper investigates the impact of these mapping strategies on three key dimensions: speaker linkability, dispersion in the anonymized speaker space, and de-identification from the original identity. Our findings show that using distinct pseudo speakers increases speaker dispersion and reduces linkability compared to common pseudo-speaker mapping, thereby enhancing privacy preservation. These observations are interpreted through the proposed pinhole effect, a conceptual framework introduced to explain the relationship between mapping strategies and anonymization performance. The hypothesis is validated through empirical evaluation.
#### Objective and Subjective Evaluation of Diffusion-Based Speech Enhancement for Dysarthric Speech
 - **Authors:** Dimme de Groot, Tanvina Patel, Devendra Kayande, Odette Scharenborg, Zhengjun Yue
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2508.17980

 - **Pdf link:** https://arxiv.org/pdf/2508.17980

 - **Abstract**
 Dysarthric speech poses significant challenges for automatic speech recognition (ASR) systems due to its high variability and reduced intelligibility. In this work we explore the use of diffusion models for dysarthric speech enhancement, which is based on the hypothesis that using diffusion-based speech enhancement moves the distribution of dysarthric speech closer to that of typical speech, which could potentially improve dysarthric speech recognition performance. We assess the effect of two diffusion-based and one signal-processing-based speech enhancement algorithms on intelligibility and speech quality of two English dysarthric speech corpora. We applied speech enhancement to both typical and dysarthric speech and evaluate the ASR performance using Whisper-Turbo, and the subjective and objective speech quality of the original and enhanced dysarthric speech. We also fine-tuned Whisper-Turbo on the enhanced speech to assess its impact on recognition performance.
#### Unseen Speaker and Language Adaptation for Lightweight Text-To-Speech with Adapters
 - **Authors:** Alessio Falai, Ziyao Zhang, Akos Gangoly
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Computation and Language (cs.CL); Machine Learning (cs.LG); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2508.18006

 - **Pdf link:** https://arxiv.org/pdf/2508.18006

 - **Abstract**
 In this paper we investigate cross-lingual Text-To-Speech (TTS) synthesis through the lens of adapters, in the context of lightweight TTS systems. In particular, we compare the tasks of unseen speaker and language adaptation with the goal of synthesising a target voice in a target language, in which the target voice has no recordings therein. Results from objective evaluations demonstrate the effectiveness of adapters in learning language-specific and speaker-specific information, allowing pre-trained models to learn unseen speaker identities or languages, while avoiding catastrophic forgetting of the original model's speaker or language information. Additionally, to measure how native the generated voices are in terms of accent, we propose and validate an objective metric inspired by mispronunciation detection techniques in second-language (L2) learners. The paper also provides insights into the impact of adapter placement, configuration and the number of speakers used.
#### TaDiCodec: Text-aware Diffusion Speech Tokenizer for Speech Language Modeling
 - **Authors:** Yuancheng Wang, Dekun Chen, Xueyao Zhang, Junan Zhang, Jiaqi Li, Zhizheng Wu
 - **Subjects:** Subjects:
Sound (cs.SD); Machine Learning (cs.LG); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2508.16790

 - **Pdf link:** https://arxiv.org/pdf/2508.16790

 - **Abstract**
 Speech tokenizers serve as foundational components for speech language models, yet current designs exhibit several limitations, including: 1) dependence on multi-layer residual vector quantization structures or high frame rates, 2) reliance on auxiliary pre-trained models for semantic distillation, and 3) requirements for complex two-stage training processes. In this work, we introduce the Text-aware Diffusion Transformer Speech Codec (TaDiCodec), a novel approach designed to overcome these challenges. TaDiCodec employs end-to-end optimization for quantization and reconstruction through a diffusion autoencoder, while integrating text guidance into the diffusion decoder to enhance reconstruction quality and achieve optimal compression. TaDiCodec achieves an extremely low frame rate of 6.25 Hz and a corresponding bitrate of 0.0875 kbps with a single-layer codebook for 24 kHz speech, while maintaining superior performance on critical speech generation evaluation metrics such as Word Error Rate (WER), speaker similarity (SIM), and speech quality (UTMOS). Notably, TaDiCodec employs a single-stage, end-to-end training paradigm, and obviating the need for auxiliary pre-trained models. We also validate the compatibility of TaDiCodec in language model based zero-shot text-to-speech with both autoregressive modeling and masked generative modeling, demonstrating its effectiveness and efficiency for speech language modeling, as well as a significantly small reconstruction-generation gap. We will open source our code and model checkpoints. Audio samples are are available at https:/tadicodec.github.io/. We release code and model checkpoints at https:/github.com/HeCheng0625/Diffusion-Speech-Tokenizer.
#### Multi-Metric Preference Alignment for Generative Speech Restoration
 - **Authors:** Junan Zhang, Xueyao Zhang, Jing Yang, Yuancheng Wang, Fan Fan, Zhizheng Wu
 - **Subjects:** Subjects:
Sound (cs.SD); Artificial Intelligence (cs.AI); Machine Learning (cs.LG); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2508.17229

 - **Pdf link:** https://arxiv.org/pdf/2508.17229

 - **Abstract**
 Recent generative models have significantly advanced speech restoration tasks, yet their training objectives often misalign with human perceptual preferences, resulting in suboptimal quality. While post-training alignment has proven effective in other generative domains like text and image generation, its application to generative speech restoration remains largely under-explored. This work investigates the challenges of applying preference-based post-training to this task, focusing on how to define a robust preference signal and curate high-quality data to avoid reward hacking. To address these challenges, we propose a multi-metric preference alignment strategy. We construct a new dataset, GenSR-Pref, comprising 80K preference pairs, where each chosen sample is unanimously favored by a complementary suite of metrics covering perceptual quality, signal fidelity, content consistency, and timbre preservation. This principled approach ensures a holistic preference signal. Applying Direct Preference Optimization (DPO) with our dataset, we observe consistent and significant performance gains across three diverse generative paradigms: autoregressive models (AR), masked generative models (MGM), and flow-matching models (FM) on various restoration benchmarks, in both objective and subjective evaluations. Ablation studies confirm the superiority of our multi-metric strategy over single-metric approaches in mitigating reward hacking. Furthermore, we demonstrate that our aligned models can serve as powerful ''data annotators'', generating high-quality pseudo-labels to serve as a supervision signal for traditional discriminative models in data-scarce scenarios like singing voice restoration. Demo Page:this https URL
#### EMO-Reasoning: Benchmarking Emotional Reasoning Capabilities in Spoken Dialogue Systems
 - **Authors:** Jingwen Liu, Kan Jen Cheng, Jiachen Lian, Akshay Anand, Rishi Jain, Faith Qiao, Robin Netzorg, Huang-Cheng Chou, Tingle Li, Guan-Ting Lin, Gopala Anumanchipalli
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2508.17623

 - **Pdf link:** https://arxiv.org/pdf/2508.17623

 - **Abstract**
 Speech emotions play a crucial role in human-computer interaction, shaping engagement and context-aware communication. Despite recent advances in spoken dialogue systems, a holistic system for evaluating emotional reasoning is still lacking. To address this, we introduce EMO-Reasoning, a benchmark for assessing emotional coherence in dialogue systems. It leverages a curated dataset generated via text-to-speech to simulate diverse emotional states, overcoming the scarcity of emotional speech data. We further propose the Cross-turn Emotion Reasoning Score to assess the emotion transitions in multi-turn dialogues. Evaluating seven dialogue systems through continuous, categorical, and perceptual metrics, we show that our framework effectively detects emotional inconsistencies, providing insights for improving current dialogue systems. By releasing a systematic evaluation benchmark, we aim to advance emotion-aware spoken dialogue modeling toward more natural and adaptive interactions.
#### Zero-shot Context Biasing with Trie-based Decoding using Synthetic Multi-Pronunciation
 - **Authors:** Changsong Liu, Yizhou Peng, Eng Siong Chng
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2508.17796

 - **Pdf link:** https://arxiv.org/pdf/2508.17796

 - **Abstract**
 Contextual automatic speech recognition (ASR) systems allow for recognizing out-of-vocabulary (OOV) words, such as named entities or rare words. However, it remains challenging due to limited training data and ambiguous or inconsistent pronunciations. In this paper, we propose a synthesis-driven multi-pronunciation contextual biasing method that performs zero-shot contextual ASR on a pretrained Whisper model. Specifically, we leverage text-to-speech (TTS) systems to synthesize diverse speech samples containing each target rare word, and then use the pretrained Whisper model to extract multiple predicted pronunciation variants. These variant token sequences are compiled into a prefix-trie, which assigns rewards to beam hypotheses in a shallow-fusion manner during beam-search decoding. After which, any recognized variant is mapped back to the original rare word in the final transcription. The evaluation results on the Librispeech dataset show that our method reduces biased word error rate (WER) by 42% on test-clean and 43% on test-other while maintaining unbiased WER essentially unchanged.
#### FasterVoiceGrad: Faster One-step Diffusion-Based Voice Conversion with Adversarial Diffusion Conversion Distillation
 - **Authors:** Takuhiro Kaneko, Hirokazu Kameoka, Kou Tanaka, Yuto Kondo
 - **Subjects:** Subjects:
Sound (cs.SD); Artificial Intelligence (cs.AI); Machine Learning (cs.LG); Audio and Speech Processing (eess.AS); Machine Learning (stat.ML)
 - **Arxiv link:** https://arxiv.org/abs/2508.17868

 - **Pdf link:** https://arxiv.org/pdf/2508.17868

 - **Abstract**
 A diffusion-based voice conversion (VC) model (e.g., VoiceGrad) can achieve high speech quality and speaker similarity; however, its conversion process is slow owing to iterative sampling. FastVoiceGrad overcomes this limitation by distilling VoiceGrad into a one-step diffusion model. However, it still requires a computationally intensive content encoder to disentangle the speaker's identity and content, which slows conversion. Therefore, we propose FasterVoiceGrad, a novel one-step diffusion-based VC model obtained by simultaneously distilling a diffusion model and content encoder using adversarial diffusion conversion distillation (ADCD), where distillation is performed in the conversion process while leveraging adversarial and score distillation training. Experimental evaluations of one-shot VC demonstrated that FasterVoiceGrad achieves competitive VC performance compared to FastVoiceGrad, with 6.6-6.9 and 1.8 times faster speed on a GPU and CPU, respectively.
#### Vocoder-Projected Feature Discriminator
 - **Authors:** Takuhiro Kaneko, Hirokazu Kameoka, Kou Tanaka, Yuto Kondo
 - **Subjects:** Subjects:
Sound (cs.SD); Artificial Intelligence (cs.AI); Machine Learning (cs.LG); Audio and Speech Processing (eess.AS); Machine Learning (stat.ML)
 - **Arxiv link:** https://arxiv.org/abs/2508.17874

 - **Pdf link:** https://arxiv.org/pdf/2508.17874

 - **Abstract**
 In text-to-speech (TTS) and voice conversion (VC), acoustic features, such as mel spectrograms, are typically used as synthesis or conversion targets owing to their compactness and ease of learning. However, because the ultimate goal is to generate high-quality waveforms, employing a vocoder to convert these features into waveforms and applying adversarial training in the time domain is reasonable. Nevertheless, upsampling the waveform introduces significant time and memory overheads. To address this issue, we propose a vocoder-projected feature discriminator (VPFD), which uses vocoder features for adversarial training. Experiments on diffusion-based VC distillation demonstrated that a pretrained and frozen vocoder feature extractor with a single upsampling step is necessary and sufficient to achieve a VC performance comparable to that of waveform discriminators while reducing the training time and memory consumption by 9.6 and 11.4 times, respectively.


by Zyzzyva0381 (Windy). 


2025-08-26
