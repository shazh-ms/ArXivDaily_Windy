# Showing new listings for Thursday, 23 October 2025
Auto update papers at about 2:30am UTC (10:30am Beijing time) every weekday.


阅读 `Usage.md`了解如何使用此repo实现个性化的Arxiv论文推送

See `Usage.md` for instructions on how to personalize the repo. 


Keyword list: ['text-to-speech', 'text to speech', 'tts', 'LLM-based', 'speech', 'voice']


Excluded: []


### Today: 7papers 
#### RIR-Mega: a large-scale simulated room impulse response dataset for machine learning and room acoustics modeling
 - **Authors:** Mandip Goswami
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Signal Processing (eess.SP)
 - **Arxiv link:** https://arxiv.org/abs/2510.18917

 - **Pdf link:** https://arxiv.org/pdf/2510.18917

 - **Abstract**
 Room impulse responses are a core resource for dereverberation, robust speech recognition, source localization, and room acoustics estimation. We present RIR-Mega, a large collection of simulated RIRs described by a compact, machine friendly metadata schema and distributed with simple tools for validation and reuse. The dataset ships with a Hugging Face Datasets loader, scripts for metadata checks and checksums, and a reference regression baseline that predicts RT60 like targets from waveforms. On a train and validation split of 36,000 and 4,000 examples, a small Random Forest on lightweight time and spectral features reaches a mean absolute error near 0.013 s and a root mean square error near 0.022 s. We host a subset with 1,000 linear array RIRs and 3,000 circular array RIRs on Hugging Face for streaming and quick tests, and preserve the complete 50,000 RIR archive on Zenodo. The dataset and code are public to support reproducible studies.
#### StutterZero and StutterFormer: End-to-End Speech Conversion for Stuttering Transcription and Correction
 - **Authors:** Qianheng Xu
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Artificial Intelligence (cs.AI); Computation and Language (cs.CL)
 - **Arxiv link:** https://arxiv.org/abs/2510.18938

 - **Pdf link:** https://arxiv.org/pdf/2510.18938

 - **Abstract**
 Over 70 million people worldwide experience stuttering, yet most automatic speech systems misinterpret disfluent utterances or fail to transcribe them accurately. Existing methods for stutter correction rely on handcrafted feature extraction or multi-stage automatic speech recognition (ASR) and text-to-speech (TTS) pipelines, which separate transcription from audio reconstruction and often amplify distortions. This work introduces StutterZero and StutterFormer, the first end-to-end waveform-to-waveform models that directly convert stuttered speech into fluent speech while jointly predicting its transcription. StutterZero employs a convolutional-bidirectional LSTM encoder-decoder with attention, whereas StutterFormer integrates a dual-stream Transformer with shared acoustic-linguistic representations. Both architectures are trained on paired stuttered-fluent data synthesized from the SEP-28K and LibriStutter corpora and evaluated on unseen speakers from the FluencyBank dataset. Across all benchmarks, StutterZero had a 24% decrease in Word Error Rate (WER) and a 31% improvement in semantic similarity (BERTScore) compared to the leading Whisper-Medium model. StutterFormer achieved better results, with a 28% decrease in WER and a 34% improvement in BERTScore. The results validate the feasibility of direct end-to-end stutter-to-fluent speech conversion, offering new opportunities for inclusive human-computer interaction, speech therapy, and accessibility-oriented AI systems.
#### An Efficient Neural Network for Modeling Human Auditory Neurograms for Speech
 - **Authors:** Eylon Zohar, Israel Nelken, Boaz Rafaely
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2510.19354

 - **Pdf link:** https://arxiv.org/pdf/2510.19354

 - **Abstract**
 Classical auditory-periphery models, exemplified by Bruce et al., 2018, provide high-fidelity simulations but are stochastic and computationally demanding, limiting large-scale experimentation and low-latency use. Prior neural encoders approximate aspects of the periphery; however, few are explicitly trained to reproduce the deterministic, rate-domain neurogram , hindering like-for-like evaluation. We present a compact convolutional encoder that approximates the Bruce mean-rate pathway and maps audio to a multi-frequency neurogram. We deliberately omit stochastic spiking effects and focus on a deterministic mapping (identical outputs for identical inputs). Using a computationally efficient design, the encoder achieves close correspondence to the reference while significantly reducing computation, enabling efficient modeling and front-end processing for auditory neuroscience and audio signal processing applications.
#### EchoFake: A Replay-Aware Dataset for Practical Speech Deepfake Detection
 - **Authors:** Tong Zhang, Yihuan Huang, Yanzhen Ren
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Artificial Intelligence (cs.AI); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2510.19414

 - **Pdf link:** https://arxiv.org/pdf/2510.19414

 - **Abstract**
 The growing prevalence of speech deepfakes has raised serious concerns, particularly in real-world scenarios such as telephone fraud and identity theft. While many anti-spoofing systems have demonstrated promising performance on lab-generated synthetic speech, they often fail when confronted with physical replay attacks-a common and low-cost form of attack used in practical settings. Our experiments show that models trained on existing datasets exhibit severe performance degradation, with average accuracy dropping to 59.6% when evaluated on replayed audio. To bridge this gap, we present EchoFake, a comprehensive dataset comprising more than 120 hours of audio from over 13,000 speakers, featuring both cutting-edge zero-shot text-to-speech (TTS) speech and physical replay recordings collected under varied devices and real-world environmental settings. Additionally, we evaluate three baseline detection models and show that models trained on EchoFake achieve lower average EERs across datasets, indicating better generalization. By introducing more practical challenges relevant to real-world deployment, EchoFake offers a more realistic foundation for advancing spoofing detection methods.
#### Relative Transfer Matrix Estimator using Covariance Subtraction
 - **Authors:** Wageesha N. Manamperi, Thushara D. Abhayapala
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2510.19439

 - **Pdf link:** https://arxiv.org/pdf/2510.19439

 - **Abstract**
 The Relative Transfer Matrix (ReTM), recently introduced as a generalization of the relative transfer function for multiple receivers and sources, shows promising performance when applied to speech enhancement and speaker separation in noisy environments. Blindly estimating the ReTM of sound sources by exploiting the covariance matrices of multichannel recordings is highly beneficial for practical applications. In this paper, we use covariance subtraction to present a flexible and practically viable method for estimating the ReTM for a select set of independent sound sources. To show the versatility of the method, we validated it through a speaker separation application under reverberant conditions. Separation performance is evaluated at low signal-to-noise ratio levels in comparison with existing ReTM-based and relative transfer function-based estimators, in both simulated and real-life environments.
#### Re-evaluating Minimum Bayes Risk Decoding for Automatic Speech Recognition
 - **Authors:** Yuu Jinnai
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Machine Learning (cs.LG); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2510.19471

 - **Pdf link:** https://arxiv.org/pdf/2510.19471

 - **Abstract**
 Recent work has shown that sample-based Minimum Bayes Risk (MBR) decoding outperforms beam search in text-to-text generation tasks, such as machine translation, text summarization, and image captioning. On the other hand, beam search is the current practice for speech-to-text tasks such as automatic speech recognition (ASR) and Speech Translation (ST). Given that MBR decoding is effective in text-to-text generation tasks, it is reasonable to expect it to also be effective for speech-to-text tasks. In this paper, we evaluate MBR decoding for ASR and ST tasks on English and Japanese using Whisper and its derivative models. We observe that the accuracy of MBR decoding outperforms that of beam search in most of the experimental settings we have evaluated. The results show that MBR decoding is a promising method for offline ASR and ST tasks that require high accuracy. The code is available at this https URL
#### Which Evaluation for Which Model? A Taxonomy for Speech Model Assessment
 - **Authors:** Maureen de Seyssel, Eeshan Gunesh Dhekane
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2510.19509

 - **Pdf link:** https://arxiv.org/pdf/2510.19509

 - **Abstract**
 Speech foundation models have recently achieved remarkable capabilities across a wide range of tasks. However, their evaluation remains disjointed across tasks and model types. Different models excel at distinct aspects of speech processing and thus require different evaluation protocols. This paper proposes a unified taxonomy that addresses the question: Which evaluation is appropriate for which model? The taxonomy defines three orthogonal axes: the \textbf{evaluation aspect} being measured, the model capabilities required to attempt the task, and the task or protocol requirements needed to perform it. We classify a broad set of existing evaluations and benchmarks along these axes, spanning areas such as representation learning, speech generation, and interactive dialogue. By mapping each evaluation to the capabilities a model exposes (e.g., speech generation, real-time processing) and to its methodological demands (e.g., fine-tuning data, human judgment), the taxonomy provides a principled framework for aligning models with suitable evaluation methods. It also reveals systematic gaps, such as limited coverage of prosody, interaction, or reasoning, that highlight priorities for future benchmark design. Overall, this work offers a conceptual foundation and practical guide for selecting, interpreting, and extending evaluations of speech models.


by Zyzzyva0381 (Windy). 


2025-10-23
