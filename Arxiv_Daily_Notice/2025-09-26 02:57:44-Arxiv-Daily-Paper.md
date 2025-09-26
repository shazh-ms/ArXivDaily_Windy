# Showing new listings for Friday, 26 September 2025
Auto update papers at about 2:30am UTC (10:30am Beijing time) every weekday.


阅读 `Usage.md`了解如何使用此repo实现个性化的Arxiv论文推送

See `Usage.md` for instructions on how to personalize the repo. 


Keyword list: ['text-to-speech', 'text to speech', 'tts', 'LLM-based', 'speech', 'voice']


Excluded: []


### Today: 15papers 
#### Data-Efficient ASR Personalization for Non-Normative Speech Using an Uncertainty-Based Phoneme Difficulty Score for Guided Sampling
 - **Authors:** Niclas Pokel, Pehuén Moure, Roman Boehringer, Yingqiang Gao
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Artificial Intelligence (cs.AI); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2509.20396

 - **Pdf link:** https://arxiv.org/pdf/2509.20396

 - **Abstract**
 Automatic speech recognition (ASR) systems struggle with non-normative speech from individuals with impairments caused by conditions like cerebral palsy or structural anomalies. The high acoustic variability and scarcity of training data severely degrade model performance. This work introduces a data-efficient personalization method that quantifies phoneme-level uncertainty to guide fine-tuning. We leverage Monte Carlo Dropout to estimate which phonemes a model finds most difficult and use these estimates for a targeted oversampling strategy. We validate our method on English and German datasets. Crucially, we demonstrate that our model-derived uncertainty strongly correlates with phonemes identified as challenging in an expert clinical logopedic report, marking, to our knowledge, the first work to successfully align model uncertainty with expert assessment of speech difficulty. Our results show that this clinically-validated, uncertainty-guided sampling significantly improves ASR accuracy, delivering a practical framework for personalized and inclusive ASR.
#### Variational Low-Rank Adaptation for Personalized Impaired Speech Recognition
 - **Authors:** Niclas Pokel, Pehuén Moure, Roman Boehringer, Shih-Chii Liu, Yingqiang Gao
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Artificial Intelligence (cs.AI)
 - **Arxiv link:** https://arxiv.org/abs/2509.20397

 - **Pdf link:** https://arxiv.org/pdf/2509.20397

 - **Abstract**
 Speech impairments resulting from congenital disorders, such as cerebral palsy, down syndrome, or apert syndrome, as well as acquired brain injuries due to stroke, traumatic accidents, or tumors, present major challenges to automatic speech recognition (ASR) systems. Despite recent advancements, state-of-the-art ASR models like Whisper still struggle with non-normative speech due to limited training data availability and high acoustic variability. Moreover, collecting and annotating non-normative speech is burdensome: speaking is effortful for many affected individuals, while laborious annotation often requires caregivers familiar with the speaker. This work introduces a novel ASR personalization method based on Bayesian Low-rank Adaptation for data-efficient fine-tuning. We validate our method on the English UA-Speech dataset and a newly collected German speech dataset, BF-Sprache, from a child with structural speech impairment. The dataset and approach are designed to reflect the challenges of low-resource settings that include individuals with speech impairments. Our method significantly improves ASR accuracy for impaired speech while maintaining data and annotation efficiency, offering a practical path toward inclusive ASR.
#### Phoenix-VAD: Streaming Semantic Endpoint Detection for Full-Duplex Speech Interaction
 - **Authors:** Weijie Wu, Wenhao Guan, Kaidi Wang, Peijie Chen, Zhuanling Zha, Junbo Li, Jun Fang, Lin Li, Qingyang Hong
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2509.20410

 - **Pdf link:** https://arxiv.org/pdf/2509.20410

 - **Abstract**
 Spoken dialogue models have significantly advanced intelligent human\textendash computer interaction, yet they lack a plug\textendash and\textendash play full\textendash duplex prediction module for semantic endpoint detection, hindering seamless audio interactions. In this paper, we introduce Phoenix\textendashVAD, an LLM\textendash based model that enables streaming semantic endpoint detection. Specifically, Phoenix\textendash VAD leverages the semantic comprehension capability of the LLM and a sliding window training strategy to achieve reliable semantic endpoint detection while supporting streaming inference. Experiments on both semantically complete and incomplete speech scenarios indicate that Phoenix\textendash VAD achieves excellent and competitive performance. Furthermore, this design enables the full\textendash duplex prediction module to be optimized independently of the dialogue model, providing more reliable and flexible support for next\textendash generation human\textendash computer interaction.
#### Objective Evaluation of Prosody and Intelligibility in Speech Synthesis via Conditional Prediction of Discrete Tokens
 - **Authors:** Ismail Rasim Ulgen, Zongyang Du, Junchen Lu, Philipp Koehn, Berrak Sisman
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Machine Learning (cs.LG); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2509.20485

 - **Pdf link:** https://arxiv.org/pdf/2509.20485

 - **Abstract**
 Objective evaluation of synthesized speech is critical for advancing speech generation systems, yet existing metrics for intelligibility and prosody remain limited in scope and weakly correlated with human perception. Word Error Rate (WER) provides only a coarse text-based measure of intelligibility, while F0-RMSE and related pitch-based metrics offer a narrow, reference-dependent view of prosody. To address these limitations, we propose TTScore, a targeted and reference-free evaluation framework based on conditional prediction of discrete speech tokens. TTScore employs two sequence-to-sequence predictors conditioned on input text: TTScore-int, which measures intelligibility through content tokens, and TTScore-pro, which evaluates prosody through prosody tokens. For each synthesized utterance, the predictors compute the likelihood of the corresponding token sequences, yielding interpretable scores that capture alignment with intended linguistic content and prosodic structure. Experiments on the SOMOS, VoiceMOS, and TTSArena benchmarks demonstrate that TTScore-int and TTScore-pro provide reliable, aspect-specific evaluation and achieve stronger correlations with human judgments of overall quality than existing intelligibility and prosody-focused metrics.
#### Real-Time System for Audio-Visual Target Speech Enhancement
 - **Authors:** T. Aleksandra Ma, Sile Yin, Li-Chia Yang, Shuo Zhang
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Emerging Technologies (cs.ET); Machine Learning (cs.LG)
 - **Arxiv link:** https://arxiv.org/abs/2509.20741

 - **Pdf link:** https://arxiv.org/pdf/2509.20741

 - **Abstract**
 We present a live demonstration for RAVEN, a real-time audio-visual speech enhancement system designed to run entirely on a CPU. In single-channel, audio-only settings, speech enhancement is traditionally approached as the task of extracting clean speech from environmental noise. More recent work has explored the use of visual cues, such as lip movements, to improve robustness, particularly in the presence of interfering speakers. However, to our knowledge, no prior work has demonstrated an interactive system for real-time audio-visual speech enhancement operating on CPU hardware. RAVEN fills this gap by using pretrained visual embeddings from an audio-visual speech recognition model to encode lip movement information. The system generalizes across environmental noise, interfering speakers, transient sounds, and even singing voices. In this demonstration, attendees will be able to experience live audio-visual target speech enhancement using a microphone and webcam setup, with clean speech playback through headphones.
#### SPADE: Structured Pruning and Adaptive Distillation for Efficient LLM-TTS
 - **Authors:** Tan Dat Nguyen, Jaehun Kim, Ji-Hoon Kim, Shukjae Choi, Youshin Lim, Joon Son Chung
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2509.20802

 - **Pdf link:** https://arxiv.org/pdf/2509.20802

 - **Abstract**
 The goal of this paper is to introduce SPADE, a framework for Structured Pruning and Adaptive Distillation for Efficient Large Language Model-based text-to-speech (LLM-TTS). Recent LLM-TTS systems achieve strong controllability and zero-shot generalization, but their large parameter counts and high latency limit real-world deployment. SPADE addresses this by combining (i) a pruning step guided by a word-error-rate-based layer importance index to remove non-essential Transformer layers, with (ii) multi-level knowledge distillation to restore autoregressive coherence. On zero-shot benchmarks, SPADE preserves near-parity perceptual quality while halving Transformer depth, reducing VRAM usage by up to 20%, and achieving up to 1.7x faster real-time factor with less than 5% of the original training data. These results show that compact LLM-TTS models can maintain naturalness and speaker similarity while enabling practical real-time speech generation. Audio samples are available at this https URL.
#### PAS-SE: Personalized Auxiliary-Sensor Speech Enhancement for Voice Pickup in Hearables
 - **Authors:** Mattes Ohlenbusch, Mikolaj Kegler, Marko Stamenovic
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2509.20875

 - **Pdf link:** https://arxiv.org/pdf/2509.20875

 - **Abstract**
 Speech enhancement for voice pickup in hearables aims to improve the user's voice by suppressing noise and interfering talkers, while maintaining own-voice quality. For single-channel methods, it is particularly challenging to distinguish the target from interfering talkers without additional context. In this paper, we compare two strategies to resolve this ambiguity: personalized speech enhancement (PSE), which uses enrollment utterances to represent the target, and auxiliary-sensor speech enhancement (AS-SE), which uses in-ear microphones as additional input. We evaluate the strategies on two public datasets, employing different auxiliary sensor arrays, to investigate their cross-dataset generalization. We propose training-time augmentations to facilitate cross-dataset generalization of AS-SE systems. We also show that combining PSE and AS-SE (PAS-SE) provides complementary performance benefits, especially when enrollment speech is recorded with the in-ear microphone. We further demonstrate that PAS-SE personalized with noisy in-ear enrollments maintains performance benefits over the AS-SE system.
#### TF-Restormer: Complex Spectral Prediction for Speech Restoration
 - **Authors:** Ui-Hyeop Shin, Jaehyun Ko, Woocheol Jeong, Hyuing-Min Park
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2509.21003

 - **Pdf link:** https://arxiv.org/pdf/2509.21003

 - **Abstract**
 Speech restoration in real-world conditions is challenging due to compounded distortions such as clipping, band-pass filtering, digital artifacts, noise, and reverberation, and low sampling rates. Existing systems, including vocoder-based approaches, often sacrifice signal fidelity, while diffusion models remain impractical for streaming. Moreover, most assume a fixed target sampling rate, requiring external resampling that leads to redundant computations. We present TF-Restormer, an encoder-decoder architecture that concentrates analysis on input-bandwidth with a time-frequency dual-path encoder and reconstructs missing high-frequency bands through a light decoder with frequency extension queries. It enables efficient and universal restoration across arbitrary input-output rates without redundant resampling. To support adversarial training across diverse rates, we introduce a shared sampling-frequency-independent (SFI) STFT discriminator. TF-Restormer further supports streaming with a causal time module, and improves robustness under extreme degradations by injecting spectral inductive bias into the frequency module. Finally, we propose a scaled log-spectral loss that stabilizes optimization under severe conditions while emphasizing well-predicted spectral details. As a single model across sampling rates, TF-Restormer consistently outperforms prior systems, achieving balanced gains in signal fidelity and perceptual quality, while its streaming mode maintains competitive effectiveness for real-time application. Code and demos are available at this https URL.
#### Are Modern Speech Enhancement Systems Vulnerable to Adversarial Attacks?
 - **Authors:** Rostislav Makarov, Lea Schönherr, Timo Gerkmann
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Machine Learning (cs.LG); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2509.21087

 - **Pdf link:** https://arxiv.org/pdf/2509.21087

 - **Abstract**
 Machine learning approaches for speech enhancement are becoming increasingly expressive, enabling ever more powerful modifications of input signals. In this paper, we demonstrate that this expressiveness introduces a vulnerability: advanced speech enhancement models can be susceptible to adversarial attacks. Specifically, we show that adversarial noise, carefully crafted and psychoacoustically masked by the original input, can be injected such that the enhanced speech output conveys an entirely different semantic meaning. We experimentally verify that contemporary predictive speech enhancement models can indeed be manipulated in this way. Furthermore, we highlight that diffusion models with stochastic samplers exhibit inherent robustness to such adversarial attacks by design.
#### Hybrid Real- And Complex-Valued Neural Network Concept For Low-Complexity Phase-Aware Speech Enhancement
 - **Authors:** Luan Vinícius Fiorio, Alex Young, Ronald M. Aarts
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2509.21185

 - **Pdf link:** https://arxiv.org/pdf/2509.21185

 - **Abstract**
 In this paper, we propose hybrid real- and complex-valued neural networks for speech enhancement. Real- or complex-valued models are either inefficient or present high complexity. We devise a straightforward design method for extending a real-valued network into its hybrid counterpart. Based on speech intelligibility and quality metrics, we compare the real, complex, and hybrid versions of a convolutional and a convolutional-recurrent architecture. The hybrid network consistently outperforms its counterparts with the same number of parameters. Additionally, the hybrid models' complexity in terms of multiply-accumulate operations is substantially lower than that of their counterparts.
#### MeanSE: Efficient Generative Speech Enhancement with Mean Flows
 - **Authors:** Jiahe Wang, Hongyu Wang, Wei Wang, Lei Yang, Chenda Li, Wangyou Zhang, Lufen Tan, Yanmin Qian
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2509.21214

 - **Pdf link:** https://arxiv.org/pdf/2509.21214

 - **Abstract**
 Speech enhancement (SE) improves degraded speech's quality, with generative models like flow matching gaining attention for their outstanding perceptual quality. However, the flow-based model requires multiple numbers of function evaluations (NFEs) to achieve stable and satisfactory performance, leading to high computational load and poor 1-NFE performance. In this paper, we propose MeanSE, an efficient generative speech enhancement model using mean flows, which models the average velocity field to achieve high-quality 1-NFE enhancement. Experimental results demonstrate that our proposed MeanSE significantly outperforms the flow matching baseline with a single NFE, exhibiting extremely better out-of-domain generalization capabilities.
#### Why Speech Deepfake Detectors Won't Generalize: The Limits of Detection in an Open World
 - **Authors:** Visar Berisha, Prad Kadambi, Isabella Lenz
 - **Subjects:** Subjects:
Cryptography and Security (cs.CR); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2509.20405

 - **Pdf link:** https://arxiv.org/pdf/2509.20405

 - **Abstract**
 Speech deepfake detectors are often evaluated on clean, benchmark-style conditions, but deployment occurs in an open world of shifting devices, sampling rates, codecs, environments, and attack families. This creates a ``coverage debt" for AI-based detectors: every new condition multiplies with existing ones, producing data blind spots that grow faster than data can be collected. Because attackers can target these uncovered regions, worst-case performance (not average benchmark scores) determines security. To demonstrate the impact of the coverage debt problem, we analyze results from a recent cross-testing framework. Grouping performance by bona fide domain and spoof release year, two patterns emerge: newer synthesizers erase the legacy artifacts detectors rely on, and conversational speech domains (teleconferencing, interviews, social media) are consistently the hardest to secure. These findings show that detection alone should not be relied upon for high-stakes decisions. Detectors should be treated as auxiliary signals within layered defenses that include provenance, personhood credentials, and policy safeguards.
#### Building Tailored Speech Recognizers for Japanese Speaking Assessment
 - **Authors:** Yotaro Kubo, Richard Sproat, Chihiro Taguchi, Llion Jones
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2509.20655

 - **Pdf link:** https://arxiv.org/pdf/2509.20655

 - **Abstract**
 This paper presents methods for building speech recognizers tailored for Japanese speaking assessment tasks. Specifically, we build a speech recognizer that outputs phonemic labels with accent markers. Although Japanese is resource-rich, there is only a small amount of data for training models to produce accurate phonemic transcriptions that include accent marks. We propose two methods to mitigate data sparsity. First, a multitask training scheme introduces auxiliary loss functions to estimate orthographic text labels and pitch patterns of the input signal, so that utterances with only orthographic annotations can be leveraged in training. The second fuses two estimators, one over phonetic alphabet strings, and the other over text token sequences. To combine these estimates we develop an algorithm based on the finite-state transducer framework. Our results indicate that the use of multitask learning and fusion is effective for building an accurate phonemic recognizer. We show that this approach is advantageous compared to the use of generic multilingual recognizers. The relative advantages of the proposed methods were also compared. Our proposed methods reduced the average of mora-label error rates from 12.3% to 7.1% over the CSJ core evaluation sets.
#### MI-Fuse: Label Fusion for Unsupervised Domain Adaptation with Closed-Source Large-Audio Language Model
 - **Authors:** Hsiao-Ying Huang, Yi-Cheng Lin, Hung-yi Lee
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2509.20706

 - **Pdf link:** https://arxiv.org/pdf/2509.20706

 - **Abstract**
 Large audio-language models (LALMs) show strong zero-shot ability on speech tasks, suggesting promise for speech emotion recognition (SER). However, SER in real-world deployments often fails under domain mismatch, where source data are unavailable and powerful LALMs are accessible only through an API. We ask: given only unlabeled target-domain audio and an API-only LALM, can a student model be adapted to outperform the LALM in the target domain? To this end, we propose MI-Fuse, a denoised label fusion framework that supplements the LALM with a source-domain trained SER classifier as an auxiliary teacher. The framework draws multiple stochastic predictions from both teachers, weights their mean distributions by mutual-information-based uncertainty, and stabilizes training with an exponential moving average teacher. Experiments across three public emotion datasets and six cross-domain transfers show consistent gains, with the student surpassing the LALM and outperforming the strongest baseline by 3.9%. This approach strengthens emotion-aware speech systems without sharing source data, enabling realistic adaptation.
#### SingVERSE: A Diverse, Real-World Benchmark for Singing Voice Enhancement
 - **Authors:** Shaohan Jiang, Junan Zhang, Yunjia Zhang, Jing Yang, Fan Fan, Zhizheng Wu
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2509.20969

 - **Pdf link:** https://arxiv.org/pdf/2509.20969

 - **Abstract**
 This paper presents a benchmark for singing voice enhancement. The development of singing voice enhancement is limited by the lack of realistic evaluation data. To address this gap, this paper introduces SingVERSE, the first real-world benchmark for singing voice enhancement, covering diverse acoustic scenarios and providing paired, studio-quality clean references. Leveraging SingVERSE, we conduct a comprehensive evaluation of state-of-the-art models and uncover a consistent trade-off between perceptual quality and intelligibility. Finally, we show that training on in-domain singing data substantially improves enhancement performance without degrading speech capabilities, establishing a simple yet effective path forward. This work offers the community a foundational benchmark together with critical insights to guide future advances in this underexplored domain. Demopage: this https URL


by Zyzzyva0381 (Windy). 


2025-09-26
