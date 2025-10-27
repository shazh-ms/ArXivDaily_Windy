# Showing new listings for Monday, 27 October 2025
Auto update papers at about 2:30am UTC (10:30am Beijing time) every weekday.


阅读 `Usage.md`了解如何使用此repo实现个性化的Arxiv论文推送

See `Usage.md` for instructions on how to personalize the repo. 


Keyword list: ['text-to-speech', 'text to speech', 'tts', 'LLM-based', 'speech', 'voice']


Excluded: []


### Today: 8papers 
#### Can large audio language models understand child stuttering speech? speech summarization, and source separation
 - **Authors:** Chibuzor Okocha, Maya Bakri, Christan Grant
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Computation and Language (cs.CL); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2510.20850

 - **Pdf link:** https://arxiv.org/pdf/2510.20850

 - **Abstract**
 Child speech differs from adult speech in acoustics, prosody, and language development, and disfluencies (repetitions, prolongations, blocks) further challenge Automatic Speech Recognition (ASR) and downstream Natural Language Processing (NLP). Recent large audio-language models (LALMs) demonstrate strong cross-modal audio understanding; however, their behavior in disfluent child speech remains underexplored. We evaluate several state-of-the-art LALMs in two settings: an interview (mixed speakers) and a reading task (single child). The tasks are (i) single-channel source separation to isolate the child and (ii) child-only summarization that preserves clinically relevant disfluencies and avoids adult-speech leakage. Evaluation combines Large Language Model (LLM) as a judge, human expert ratings, and BERTScore (F1), and we report agreement between models and between models and humans to assess reliability. Our findings delineate the conditions under which LALMs produce faithful child-only summaries from mixed audio and where they fail, offering practical guidance for clinical and educational deployments. We provide prompts and evaluation scripts to support replication.
#### Data-Centric Lessons To Improve Speech-Language Pretraining
 - **Authors:** Vishaal Udandarao, Zhiyun Lu, Xuankai Chang, Yongqiang Wang, Violet Z. Yao, Albin Madapally Jose, Fartash Faghri, Josh Gardner, Chung-Cheng Chiu
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Computation and Language (cs.CL); Machine Learning (cs.LG)
 - **Arxiv link:** https://arxiv.org/abs/2510.20860

 - **Pdf link:** https://arxiv.org/pdf/2510.20860

 - **Abstract**
 Spoken Question-Answering (SQA) is a core capability for useful and interactive artificial intelligence systems. Recently, several speech-language models (SpeechLMs) have been released with a specific focus on improving their SQA performance. However, a lack of controlled ablations of pretraining data processing and curation makes it challenging to understand what factors account for performance, despite substantial gains from similar studies in other data modalities. In this work, we address this gap by conducting a data-centric exploration for pretraining SpeechLMs. We focus on three research questions fundamental to speech-language pretraining data: (1) how to process raw web-crawled audio content for speech-text pretraining, (2) how to construct synthetic pretraining datasets to augment web-crawled data and (3) how to interleave (text, audio) segments into training sequences. We apply the insights from our controlled data-centric ablations to pretrain a 3.8B-parameter SpeechLM, called SpeLangy, that outperforms models that are up to 3x larger by 10.2% absolute performance. We hope our findings highlight the impact of effective data curation for speech-language pretraining and guide future data-centric exploration in SpeechLMs.
#### refess-qi: reference-free evaluation for speech separation with joint quality and intelligibility scoring
 - **Authors:** Ari Frummer, Helin Wang, Tianyu Cao, Adi Arbel, Yuval Sieradzki, Oren Gal, Jesús Villalba, Thomas Thebaud, Najim Dehak
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2510.21014

 - **Pdf link:** https://arxiv.org/pdf/2510.21014

 - **Abstract**
 Source separation is a crucial pre-processing step for various speech processing tasks, such as automatic speech recognition (ASR). Traditionally, the evaluation metrics for speech separation rely on the matched reference audios and corresponding transcriptions to assess audio quality and intelligibility. However, they cannot be used to evaluate real-world mixtures for which no reference exists. This paper introduces a text-free reference-free evaluation framework based on self-supervised learning (SSL) representations. The proposed framework utilize the mixture and separated tracks to predict jointly audio quality, through the Scale Invariant Signal to Noise Ratio (SI-SNR) metric, and speech intelligibility through the Word Error Rate (WER) metric. We conducted experiments on the WHAMR! dataset, which shows a WER estimation with a mean absolute error (MAE) of 17\% and a Pearson correlation coefficient (PCC) of 0.77; and SI-SNR estimation with an MAE of 1.38 and PCC of 0.95. We further demonstrate the robustness of our estimator by using various SSL representations.
#### PhoenixCodec: Taming Neural Speech Coding for Extreme Low-Resource Scenarios
 - **Authors:** Zixiang Wan, Haoran Zhao, Guochang Zhang, Runqiang Han, Jianqiang Wei, Yuexian Zou
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2510.21196

 - **Pdf link:** https://arxiv.org/pdf/2510.21196

 - **Abstract**
 This paper presents PhoenixCodec, a comprehensive neural speech coding and decoding framework designed for extremely low-resource conditions. The proposed system integrates an optimized asymmetric frequency-time architecture, a Cyclical Calibration and Refinement (CCR) training strategy, and a noise-invariant fine-tuning procedure. Under stringent constraints - computation below 700 MFLOPs, latency less than 30 ms, and dual-rate support at 1 kbps and 6 kbps - existing methods face a trade-off between efficiency and quality. PhoenixCodec addresses these challenges by alleviating the resource scattering of conventional decoders, employing CCR to escape local optima, and enhancing robustness through noisy-sample fine-tuning. In the LRAC 2025 Challenge Track 1, the proposed system ranked third overall and demonstrated the best performance at 1 kbps in both real-world noise and reverberation and intelligibility in clean tests, confirming its effectiveness.
#### SpecTokenizer: A Lightweight Streaming Codec in the Compressed Spectrum Domain
 - **Authors:** Zixiang Wan, Guochang Zhang, Yifeng He, Jianqiang Wei
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2510.21209

 - **Pdf link:** https://arxiv.org/pdf/2510.21209

 - **Abstract**
 Neural Audio Codecs (NACs) have gained growing attention in recent years as technologies for audio compression and audio representation in speech language models. While mainstream NACs typically require G-level computation and M-level parameters, the performance of lightweight and streaming NACs remains underexplored. This paper proposes SpecTokenizer, a lightweight streaming codec that operates in the compressed spectral domain. Composed solely of alternating CNN and RNN layers, SpecTokenizer achieves greater efficiency and better representational capability through multi-scale modeling in the compressed spectrum domain. At 4 kbps, the proposed SpecTokenizer achieves comparable or superior performance compared to the codec with state-of-the-art lightweight architecture while requiring only 20% of the computation and 10% of the parameters. Furthermore, it significantly outperforms the codec when using similar computational and storage resources.
#### Are These Even Words? Quantifying the Gibberishness of Generative Speech Models
 - **Authors:** Danilo de Oliveira, Tal Peer, Jonas Rochdi, Timo Gerkmann
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2510.21317

 - **Pdf link:** https://arxiv.org/pdf/2510.21317

 - **Abstract**
 Significant research efforts are currently being dedicated to non-intrusive quality and intelligibility assessment, especially given how it enables curation of large scale datasets of in-the-wild speech data. However, with the increasing capabilities of generative models to synthesize high quality speech, new types of artifacts become relevant, such as generative hallucinations. While intrusive metrics are able to spot such sort of discrepancies from a reference signal, it is not clear how current non-intrusive methods react to high-quality phoneme confusions or, more extremely, gibberish speech. In this paper we explore how to factor in this aspect under a fully unsupervised setting by leveraging language models. Additionally, we publish a dataset of high-quality synthesized gibberish speech for further development of measures to assess implausible sentences in spoken language, alongside code for calculating scores from a variety of speech language models.
#### Compressing Quaternion Convolutional Neural Networks for Audio Classification
 - **Authors:** Arshdeep Singh, Vinayak Abrol, Mark D. Plumbley
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Artificial Intelligence (cs.AI); Sound (cs.SD); Signal Processing (eess.SP)
 - **Arxiv link:** https://arxiv.org/abs/2510.21388

 - **Pdf link:** https://arxiv.org/pdf/2510.21388

 - **Abstract**
 Conventional Convolutional Neural Networks (CNNs) in the real domain have been widely used for audio classification. However, their convolution operations process multi-channel inputs independently, limiting the ability to capture correlations among channels. This can lead to suboptimal feature learning, particularly for complex audio patterns such as multi-channel spectrogram representations. Quaternion Convolutional Neural Networks (QCNNs) address this limitation by employing quaternion algebra to jointly capture inter-channel dependencies, enabling more compact models with fewer learnable parameters while better exploiting the multi-dimensional nature of audio signals. However, QCNNs exhibit higher computational complexity due to the overhead of quaternion operations, resulting in increased inference latency and reduced efficiency compared to conventional CNNs, posing challenges for deployment on resource-constrained platforms. To address this challenge, this study explores knowledge distillation (KD) and pruning, to reduce the computational complexity of QCNNs while maintaining performance. Our experiments on audio classification reveal that pruning QCNNs achieves similar or superior performance compared to KD while requiring less computational effort. Compared to conventional CNNs and Transformer-based architectures, pruned QCNNs achieve competitive performance with a reduced learnable parameter count and computational complexity. On the AudioSet dataset, pruned QCNNs reduce computational cost by 50\% and parameter count by 80\%, while maintaining performance comparable to the conventional CNNs. Furthermore, pruned QCNNs generalize well across multiple audio classification benchmarks, including GTZAN for music genre recognition, ESC-50 for environmental sound classification and RAVDESS for speech emotion recognition.
#### FlexIO: Flexible Single- and Multi-Channel Speech Separation and Enhancement
 - **Authors:** Yoshiki Masuyama, Kohei Saijo, Francesco Paissan, Jiangyu Han, Marc Delcroix, Ryo Aihara, François G. Germain, Gordon Wichern, Jonathan Le Roux
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS); Signal Processing (eess.SP)
 - **Arxiv link:** https://arxiv.org/abs/2510.21485

 - **Pdf link:** https://arxiv.org/pdf/2510.21485

 - **Abstract**
 Speech separation and enhancement (SSE) has advanced remarkably and achieved promising results in controlled settings, such as a fixed number of speakers and a fixed array configuration. Towards a universal SSE system, single-channel systems have been extended to deal with a variable number of speakers (i.e., outputs). Meanwhile, multi-channel systems accommodating various array configurations (i.e., inputs) have been developed. However, these attempts have been pursued separately. In this paper, we propose a flexible input and output SSE system, named FlexIO. It performs conditional separation using prompt vectors, one per speaker as a condition, allowing separation of an arbitrary number of speakers. Multi-channel mixtures are processed together with the prompt vectors via an array-agnostic channel communication mechanism. Our experiments demonstrate that FlexIO successfully covers diverse conditions with one to five microphones and one to three speakers. We also confirm the robustness of FlexIO on CHiME-4 real data.


by Zyzzyva0381 (Windy). 


2025-10-27
