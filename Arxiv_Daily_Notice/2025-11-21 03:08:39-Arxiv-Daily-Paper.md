# Showing new listings for Friday, 21 November 2025
Auto update papers at about 2:30am UTC (10:30am Beijing time) every weekday.


阅读 `Usage.md`了解如何使用此repo实现个性化的Arxiv论文推送

See `Usage.md` for instructions on how to personalize the repo. 


Keyword list: ['text-to-speech', 'text to speech', 'tts', 'LLM-based', 'speech', 'voice']


Excluded: []


### Today: 4papers 
#### A Generalized Weighted Overlap-Add (WOLA) Filter Bank for Improved Subband System Identification
 - **Authors:** Mohit Sharma (1), Robbe Van Rompaey (2), Wouter Lanneer (2), Marc Moonen (1) ((1) Department of Electrical Engineering (ESAT), KU Leuven, Belgium, (2) Nokia Bell Labs, Antwerp, Belgium)
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Signal Processing (eess.SP)
 - **Arxiv link:** https://arxiv.org/abs/2511.15766

 - **Pdf link:** https://arxiv.org/pdf/2511.15766

 - **Abstract**
 This paper addresses the challenges in short-time Fourier transform (STFT) domain subband adaptive filtering, in particular, subband system identification. Previous studies in this area have primarily focused on setups with subband filtering at a downsampled rate, implemented using the weighted overlap-add (WOLA) filter bank, popular in audio and speech-processing for its reduced complexity. However, this traditional approach imposes constraints on the subband filters when transformed to their full-rate representation. This paper makes three key contributions. First, it introduces a generalized WOLA filter bank that repositions subband filters before the downsampling operation, eliminating the constraints on subband filters inherent in the conventional WOLA filter bank. Second, it investigates the mean square error (MSE) performance of the generalized WOLA filter bank for full-band system identification, establishing analytical ties between the order of subband filters, the full-band system impulse response length, the decimation factor, and the prototype filters. Third, to address the increased computational complexity of the generalized WOLA, the paper proposes a low-complexity implementation termed per-tone weighted overlap-add (PT-WOLA), which maintains computational complexity on par with conventional WOLA. Analytical and empirical evidence demonstrates that the proposed generalized WOLA filter bank significantly enhances the performance of subband system identification.
#### Train Short, Infer Long: Speech-LLM Enables Zero-Shot Streamable Joint ASR and Diarization on Long Audio
 - **Authors:** Mohan Shi, Xiong Xiao, Ruchao Fan, Shaoshi Ling, Jinyu Li
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2511.16046

 - **Pdf link:** https://arxiv.org/pdf/2511.16046

 - **Abstract**
 Joint automatic speech recognition (ASR) and speaker diarization aim to answer the question "who spoke what" in multi-speaker scenarios. In this paper, we present an end-to-end speech large language model (Speech-LLM) for Joint strEamable DIarization and aSr (JEDIS-LLM). The model is trained only on short audio under 20s but is capable of streamable inference on long-form audio without additional training. This is achieved by introducing a Speaker Prompt Cache (SPC) with an on-the-fly update mechanism during chunk-wise streaming inference, inspired by the autoregressive nature of LLMs. The SPC also allows the seamless use of pre-enrolled speaker profiles which is common in many scenarios like meeting transcription. To further enhance diarization capability, we incorporate word-level speaker supervision into the speech encoder during training. Experimental results demonstrate that our system outperforms strong baselines, including Sortformer and Meta-Cat in the local setting on audio up to 20s, and DiarizationLM on long-form audio, despite being fully end-to-end and streamable while DiarizationLM follows a cascaded offline pipeline. To the best of our knowledge, this is the first work enabling zero-shot streamable joint ASR and diarization on long audio using a Speech-LLM trained only on short audio, achieving state-of-the-art performance.
#### SUNAC: Source-aware Unified Neural Audio Codec
 - **Authors:** Ryo Aihara, Yoshiki Masuyama, Francesco Paissan, FranÃ§ois G. Germain, Gordon Wichern, Jonathan Le Roux
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Signal Processing (eess.SP)
 - **Arxiv link:** https://arxiv.org/abs/2511.16126

 - **Pdf link:** https://arxiv.org/pdf/2511.16126

 - **Abstract**
 Neural audio codecs (NACs) provide compact representations that can be leveraged in many downstream applications, in particular large language models. Yet most NACs encode mixtures of multiple sources in an entangled manner, which may impede efficient downstream processing in applications that need access to only a subset of the sources (e.g., analysis of a particular type of sound, transcription of a given speaker, etc). To address this, we propose a source-aware codec that encodes individual sources directly from mixtures, conditioned on source type prompts. This enables user-driven selection of which source(s) to encode, including separately encoding multiple sources of the same type (e.g., multiple speech signals). Experiments show that our model achieves competitive resynthesis and separation quality relative to a cascade of source separation followed by a conventional NAC, with lower computational cost.
#### Codec2Vec: Self-Supervised Speech Representation Learning Using Neural Speech Codecs
 - **Authors:** Wei-Cheng Tseng, David Harwath
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Computation and Language (cs.CL)
 - **Arxiv link:** https://arxiv.org/abs/2511.16639

 - **Pdf link:** https://arxiv.org/pdf/2511.16639

 - **Abstract**
 Recent advancements in neural audio codecs have not only enabled superior audio compression but also enhanced speech synthesis techniques. Researchers are now exploring their potential as universal acoustic feature extractors for a broader range of speech processing tasks. Building on this trend, we introduce Codec2Vec, the first speech representation learning framework that relies exclusively on discrete audio codec units. This approach offers several advantages, including improved data storage and transmission efficiency, faster training, and enhanced data privacy. We explore masked prediction with various training target derivation strategies to thoroughly understand the effectiveness of this framework. Evaluated on the SUPERB benchmark, Codec2Vec achieves competitive performance compared to continuous-input models while reducing storage requirements by up to 16.5x and training time by 2.3x, showcasing its scalability and efficiency.


by Zyzzyva0381 (Windy). 


2025-11-21
