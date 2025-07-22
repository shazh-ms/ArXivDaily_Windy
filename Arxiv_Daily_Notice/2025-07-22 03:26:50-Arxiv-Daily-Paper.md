# Showing new listings for Tuesday, 22 July 2025
Auto update papers at about 2:30am UTC (10:30am Beijing time) every weekday.


阅读 `Usage.md`了解如何使用此repo实现个性化的Arxiv论文推送

See `Usage.md` for instructions on how to personalize the repo. 


Keyword list: ['text-to-speech', 'text to speech', 'tts', 'LLM-based', 'speech', 'voice']


Excluded: []


### Today: 14papers 
#### Towards Accurate Phonetic Error Detection Through Phoneme Similarity Modeling
 - **Authors:** Xuanru Zhou, Jiachen Lian, Cheol Jun Cho, Tejas Prabhune, Shuhe Li, William Li, Rodrigo Ortiz, Zoe Ezzes, Jet Vonk, Brittany Morin, Rian Bogley, Lisa Wauters, Zachary Miller, Maria Gorno-Tempini, Gopala Anumanchipalli
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2507.14346

 - **Pdf link:** https://arxiv.org/pdf/2507.14346

 - **Abstract**
 Phonetic error detection, a core subtask of automatic pronunciation assessment, identifies pronunciation deviations at the phoneme level. Speech variability from accents and dysfluencies challenges accurate phoneme recognition, with current models failing to capture these discrepancies effectively. We propose a verbatim phoneme recognition framework using multi-task training with novel phoneme similarity modeling that transcribes what speakers actually say rather than what they're supposed to say. We develop and open-source \textit{VCTK-accent}, a simulated dataset containing phonetic errors, and propose two novel metrics for assessing pronunciation differences. Our work establishes a new benchmark for phonetic error detection.
#### Adapting Whisper for Lightweight and Efficient Automatic Speech Recognition of Children for On-device Edge Applications
 - **Authors:** Satwik Dutta, Shruthigna Chandupatla, John Hansen
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Human-Computer Interaction (cs.HC); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2507.14451

 - **Pdf link:** https://arxiv.org/pdf/2507.14451

 - **Abstract**
 Reliability on cloud providers for ASR inference to support child-centered voice-based applications is becoming challenging due to regulatory and privacy challenges. Motivated by a privacy-preserving design, this study aims to develop a lightweight & efficient Whisper ASR system capable of running on a Raspberry Pi. Upon evaluation of the MyST corpus and by examining various filtering strategies to fine-tune the `this http URL' model, a Word Error Rate (WER) of 15.9% was achieved (11.8% filtered). A low-rank compression reduces the encoder size by 0.51M with 1.26x faster inference in GPU, with 11% relative WER increase. During inference on Pi, the compressed version required ~2 GFLOPS fewer computations. The RTF for both the models ranged between [0.23-0.41] for various input audio durations. Analyzing the RAM usage and CPU temperature showed that the PI was capable of handling both the tiny models, however it was noticed that small models initiated additional overhead/thermal throttling.
#### Conan: A Chunkwise Online Network for Zero-Shot Adaptive Voice Conversion
 - **Authors:** Yu Zhang, Baotong Tian, Zhiyao Duan
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Computation and Language (cs.CL); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2507.14534

 - **Pdf link:** https://arxiv.org/pdf/2507.14534

 - **Abstract**
 Zero-shot online voice conversion (VC) holds significant promise for real-time communications and entertainment. However, current VC models struggle to preserve semantic fidelity under real-time constraints, deliver natural-sounding conversions, and adapt effectively to unseen speaker characteristics. To address these challenges, we introduce Conan, a chunkwise online zero-shot voice conversion model that preserves the content of the source while matching the voice timbre and styles of reference speech. Conan comprises three core components: 1) a Stream Content Extractor that leverages Emformer for low-latency streaming content encoding; 2) an Adaptive Style Encoder that extracts fine-grained stylistic features from reference speech for enhanced style adaptation; 3) a Causal Shuffle Vocoder that implements a fully causal HiFiGAN using a pixel-shuffle mechanism. Experimental evaluations demonstrate that Conan outperforms baseline models in subjective and objective metrics. Audio samples can be found at this https URL.
#### Parameter-Efficient Fine-Tuning of Foundation Models for CLP Speech Classification
 - **Authors:** Susmita Bhattacharjee, Jagabandhu Mishra, H.S. Shekhawat, S. R. Mahadeva Prasanna
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Signal Processing (eess.SP)
 - **Arxiv link:** https://arxiv.org/abs/2507.14898

 - **Pdf link:** https://arxiv.org/pdf/2507.14898

 - **Abstract**
 We propose the use of parameter-efficient fine-tuning (PEFT) of foundation models for cleft lip and palate (CLP) detection and severity classification. In CLP, nasalization increases with severity due to the abnormal passage between the oral and nasal tracts; this causes oral stops to be replaced by glottal stops and alters formant trajectories and vowel space. Since foundation models are trained for grapheme prediction or long-term quantized representation prediction, they may better discriminate CLP severity when fine-tuned on domain-specific data. We conduct experiments on two datasets: English (NMCPC) and Kannada (AIISH). We perform a comparative analysis using embeddings from self-supervised models Wav2Vec2 and WavLM, and the weakly supervised Whisper, each paired with SVM classifiers, and compare them with traditional handcrafted features eGeMAPS and ComParE. Finally, we fine-tune the best-performing Whisper model using PEFT techniques: Low-Rank Adapter (LoRA) and Decomposed Low-Rank Adapter (DoRA). Our results demonstrate that the proposed approach achieves relative improvements of 26.4% and 63.4% in macro-average F1 score over the best foundation model and handcrafted feature baselines on the NMCPC dataset, and improvements of 6.1% and 52.9% on the AIISH dataset, respectively.
#### DMOSpeech 2: Reinforcement Learning for Duration Prediction in Metric-Optimized Speech Synthesis
 - **Authors:** Yinghao Aaron Li, Xilin Jiang, Fei Tao, Cheng Niu, Kaifeng Xu, Juntong Song, Nima Mesgarani
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2507.14988

 - **Pdf link:** https://arxiv.org/pdf/2507.14988

 - **Abstract**
 Diffusion-based text-to-speech (TTS) systems have made remarkable progress in zero-shot speech synthesis, yet optimizing all components for perceptual metrics remains challenging. Prior work with DMOSpeech demonstrated direct metric optimization for speech generation components, but duration prediction remained unoptimized. This paper presents DMOSpeech 2, which extends metric optimization to the duration predictor through a reinforcement learning approach. The proposed system implements a novel duration policy framework using group relative preference optimization (GRPO) with speaker similarity and word error rate as reward signals. By optimizing this previously unoptimized component, DMOSpeech 2 creates a more complete metric-optimized synthesis pipeline. Additionally, this paper introduces teacher-guided sampling, a hybrid approach leveraging a teacher model for initial denoising steps before transitioning to the student model, significantly improving output diversity while maintaining efficiency. Comprehensive evaluations demonstrate superior performance across all metrics compared to previous systems, while reducing sampling steps by half without quality degradation. These advances represent a significant step toward speech synthesis systems with metric optimization across multiple components. The audio samples, code and pre-trained models are available at this https URL.
#### Mixture to Beamformed Mixture: Leveraging Beamformed Mixture as Weak-Supervision for Speech Enhancement and Noise-Robust ASR
 - **Authors:** Zhong-Qiu Wang, Ruizhe Pang
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2507.15229

 - **Pdf link:** https://arxiv.org/pdf/2507.15229

 - **Abstract**
 In multi-channel speech enhancement and robust automatic speech recognition (ASR), beamforming can typically improve the signal-to-noise ratio (SNR) of the target speaker and produce reliable enhancement with little distortion to target speech. With this observation, we propose to leverage beamformed mixture, which has a higher SNR of the target speaker than the input mixture, as a weak supervision to train deep neural networks (DNNs) to enhance the input mixture. This way, we can train enhancement models using pairs of real-recorded mixture and its beamformed mixture, and potentially realize better generalization to real mixtures, compared with only training the models on simulated mixtures, which usually mismatch real mixtures. Evaluation results on the real-recorded CHiME-4 dataset show the effectiveness of the proposed algorithm.
#### Multi-Sampling-Frequency Naturalness MOS Prediction Using Self-Supervised Learning Model with Sampling-Frequency-Independent Layer
 - **Authors:** Go Nishikawa, Wataru Nakata, Yuki Saito, Kanami Imamura, Hiroshi Saruwatari, Tomohiko Nakamura
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2507.14647

 - **Pdf link:** https://arxiv.org/pdf/2507.14647

 - **Abstract**
 We introduce our submission to the AudioMOS Challenge (AMC) 2025 Track 3: mean opinion score (MOS) prediction for speech with multiple sampling frequencies (SFs). Our submitted model integrates an SF-independent (SFI) convolutional layer into a self-supervised learning (SSL) model to achieve SFI speech feature extraction for MOS prediction. We present some strategies to improve the MOS prediction performance of our model: distilling knowledge from a pretrained non-SFI-SSL model and pretraining with a large-scale MOS dataset. Our submission to the AMC 2025 Track 3 ranked the first in one evaluation metric and the fourth in the final ranking. We also report the results of our ablation study to investigate essential factors of our model.
#### Frame-level Temporal Difference Learning for Partial Deepfake Speech Detection
 - **Authors:** Menglu Li, Xiao-Ping Zhang, Lian Zhao
 - **Subjects:** Subjects:
Sound (cs.SD); Cryptography and Security (cs.CR); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2507.15101

 - **Pdf link:** https://arxiv.org/pdf/2507.15101

 - **Abstract**
 Detecting partial deepfake speech is essential due to its potential for subtle misinformation. However, existing methods depend on costly frame-level annotations during training, limiting real-world scalability. Also, they focus on detecting transition artifacts between bonafide and deepfake segments. As deepfake generation techniques increasingly smooth these transitions, detection has become more challenging. To address this, our work introduces a new perspective by analyzing frame-level temporal differences and reveals that deepfake speech exhibits erratic directional changes and unnatural local transitions compared to bonafide speech. Based on this finding, we propose a Temporal Difference Attention Module (TDAM) that redefines partial deepfake detection as identifying unnatural temporal variations, without relying on explicit boundary annotations. A dual-level hierarchical difference representation captures temporal irregularities at both fine and coarse scales, while adaptive average pooling preserves essential patterns across variable-length inputs to minimize information loss. Our TDAM-AvgPool model achieves state-of-the-art performance, with an EER of 0.59% on the PartialSpoof dataset and 0.03% on the HAD dataset, which significantly outperforms the existing methods without requiring frame-level supervision.
#### Exploiting Context-dependent Duration Features for Voice Anonymization Attack Systems
 - **Authors:** Natalia Tomashenko, Emmanuel Vincent, Marc Tommasi
 - **Subjects:** Subjects:
Sound (cs.SD); Computation and Language (cs.CL); Cryptography and Security (cs.CR); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2507.15214

 - **Pdf link:** https://arxiv.org/pdf/2507.15214

 - **Abstract**
 The temporal dynamics of speech, encompassing variations in rhythm, intonation, and speaking rate, contain important and unique information about speaker identity. This paper proposes a new method for representing speaker characteristics by extracting context-dependent duration embeddings from speech temporal dynamics. We develop novel attack models using these representations and analyze the potential vulnerabilities in speaker verification and voice anonymization this http URL experimental results show that the developed attack models provide a significant improvement in speaker verification performance for both original and anonymized data in comparison with simpler representations of speech temporal dynamics reported in the literature.
#### EchoVoices: Preserving Generational Voices and Memories for Seniors and Children
 - **Authors:** Haiying Xu, Haoze Liu, Mingshi Li, Siyu Cai, Guangxuan Zheng, Yuhuang Jia, Jinghua Zhao, Yong Qin
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2507.15221

 - **Pdf link:** https://arxiv.org/pdf/2507.15221

 - **Abstract**
 Recent breakthroughs in intelligent speech and digital human technologies have primarily targeted mainstream adult users, often overlooking the distinct vocal patterns and interaction styles of seniors and children. These demographics possess distinct vocal characteristics, linguistic styles, and interaction patterns that challenge conventional ASR, TTS, and LLM systems. To address this, we introduce EchoVoices, an end-to-end digital human pipeline dedicated to creating persistent digital personas for seniors and children, ensuring their voices and memories are preserved for future generations. Our system integrates three core innovations: a k-NN-enhanced Whisper model for robust speech recognition of atypical speech; an age-adaptive VITS model for high-fidelity, speaker-aware speech synthesis; and an LLM-driven agent that automatically generates persona cards and leverages a RAG-based memory system for conversational consistency. Our experiments, conducted on the SeniorTalk and ChildMandarin datasets, demonstrate significant improvements in recognition accuracy, synthesis quality, and speaker similarity. EchoVoices provides a comprehensive framework for preserving generational voices, offering a new means of intergenerational connection and the creation of lasting digital legacies.
#### A2TTS: TTS for Low Resource Indian Languages
 - **Authors:** Ayush Singh Bhadoriya, Abhishek Nikunj Shinde, Isha Pandey, Ganesh Ramakrishnan
 - **Subjects:** Subjects:
Sound (cs.SD); Artificial Intelligence (cs.AI); Computation and Language (cs.CL); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2507.15272

 - **Pdf link:** https://arxiv.org/pdf/2507.15272

 - **Abstract**
 We present a speaker conditioned text-to-speech (TTS) system aimed at addressing challenges in generating speech for unseen speakers and supporting diverse Indian languages. Our method leverages a diffusion-based TTS architecture, where a speaker encoder extracts embeddings from short reference audio samples to condition the DDPM decoder for multispeaker generation. To further enhance prosody and naturalness, we employ a cross-attention based duration prediction mechanism that utilizes reference audio, enabling more accurate and speaker consistent timing. This results in speech that closely resembles the target speaker while improving duration modeling and overall expressiveness. Additionally, to improve zero-shot generation, we employed classifier free guidance, allowing the system to generate speech more near speech for unknown speakers. Using this approach, we trained language-specific speaker-conditioned models. Using the IndicSUPERB dataset for multiple Indian languages such as Bengali, Gujarati, Hindi, Marathi, Malayalam, Punjabi and Tamil.
#### STITCH: Simultaneous Thinking and Talking with Chunked Reasoning for Spoken Language Models
 - **Authors:** Cheng-Han Chiang, Xiaofei Wang, Linjie Li, Chung-Ching Lin, Kevin Lin, Shujie Liu, Zhendong Wang, Zhengyuan Yang, Hung-yi Lee, Lijuan Wang
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2507.15375

 - **Pdf link:** https://arxiv.org/pdf/2507.15375

 - **Abstract**
 Spoken Language Models (SLMs) are designed to take speech inputs and produce spoken responses. However, current SLMs lack the ability to perform an internal, unspoken thinking process before responding. In contrast, humans typically engage in complex mental reasoning internally, enabling them to communicate ideas clearly and concisely. Thus, integrating an unspoken thought process into SLMs is highly desirable. While naively generating a complete chain-of-thought (CoT) reasoning before starting to talk can enable thinking for SLMs, this induces additional latency for the speech response, as the CoT reasoning can be arbitrarily long. To solve this issue, we propose Stitch, a novel generation method that alternates between the generation of unspoken reasoning chunks and spoken response chunks. Since the audio duration of a chunk of spoken response is much longer than the time to generate the tokens in a chunk of spoken response, we use the remaining free time to generate the unspoken reasoning tokens. When a chunk of audio is played to the user, the model continues to generate the next unspoken reasoning chunk, achieving simultaneous thinking and talking. Remarkably, Stitch matches the latency of baselines that cannot generate unspoken CoT by design while outperforming those baselines by 15% on math reasoning datasets; Stitch also performs equally well on non-reasoning datasets as those baseline models. Some animations and demonstrations are on the project page: this https URL.
#### Neuro-MSBG: An End-to-End Neural Model for Hearing Loss Simulation
 - **Authors:** Hui-Guan Yuan, Ryandhimas E. Zezario, Shafique Ahmed, Hsin-Min Wang, Kai-Lung Hua, Yu Tsao
 - **Subjects:** Subjects:
Sound (cs.SD); Artificial Intelligence (cs.AI); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2507.15396

 - **Pdf link:** https://arxiv.org/pdf/2507.15396

 - **Abstract**
 Hearing loss simulation models are essential for hearing aid deployment. However, existing models have high computational complexity and latency, which limits real-time applications and lack direct integration with speech processing systems. To address these issues, we propose Neuro-MSBG, a lightweight end-to-end model with a personalized audiogram encoder for effective time-frequency modeling. Experiments show that Neuro-MSBG supports parallel inference and retains the intelligibility and perceptual quality of the original MSBG, with a Spearman's rank correlation coefficient (SRCC) of 0.9247 for Short-Time Objective Intelligibility (STOI) and 0.8671 for Perceptual Evaluation of Speech Quality (PESQ). Neuro-MSBG reduces simulation runtime by a factor of 46 (from 0.970 seconds to 0.021 seconds for a 1 second input), further demonstrating its efficiency and practicality.
#### An Investigation of Test-time Adaptation for Audio Classification under Background Noise
 - **Authors:** Weichuang Shao, Iman Yi Liao, Tomas Henrique Bode Maul, Tissa Chandesa
 - **Subjects:** Subjects:
Machine Learning (cs.LG); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2507.15523

 - **Pdf link:** https://arxiv.org/pdf/2507.15523

 - **Abstract**
 Domain shift is a prominent problem in Deep Learning, causing a model pre-trained on a source dataset to suffer significant performance degradation on test datasets. This research aims to address the issue of audio classification under domain shift caused by background noise using Test-Time Adaptation (TTA), a technique that adapts a pre-trained model during testing using only unlabelled test data before making predictions. We adopt two common TTA methods, TTT and TENT, and a state-of-the-art method CoNMix, and investigate their respective performance on two popular audio classification datasets, AudioMNIST (AM) and SpeechCommands V1 (SC), against different types of background noise and noise severity levels. The experimental results reveal that our proposed modified version of CoNMix produced the highest classification accuracy under domain shift (5.31% error rate under 10 dB exercise bike background noise and 12.75% error rate under 3 dB running tap background noise for AM) compared to TTT and TENT. The literature search provided no evidence of similar works, thereby motivating the work reported here as the first study to leverage TTA techniques for audio classification under domain shift.


by Zyzzyva0381 (Windy). 


2025-07-22
