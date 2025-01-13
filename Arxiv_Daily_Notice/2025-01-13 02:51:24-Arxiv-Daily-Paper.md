# Showing new listings for Monday, 13 January 2025
Auto update papers at about 2:30am UTC (10:30am Beijing time) every weekday.


阅读 `Usage.md`了解如何使用此repo实现个性化的Arxiv论文推送

See `Usage.md` for instructions on how to personalize the repo. 


Keyword list: ['acoustic', 'reinforcement learning', 'reverb', 'meta', 'separate', 'reconstruction', 'noise', 'enhance', 'localization', 'speech']


Excluded: []


### Today: 13papers 
#### Mel-Spectrogram Inversion via Alternating Direction Method of Multipliers
 - **Authors:** Yoshiki Masuyama, Natsuki Ueno, Nobutaka Ono
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2501.05557

 - **Pdf link:** https://arxiv.org/pdf/2501.05557

 - **Abstract**
 Signal reconstruction from its mel-spectrogram is known as mel-spectrogram inversion and has many applications, including speech and foley sound synthesis. In this paper, we propose a mel-spectrogram inversion method based on a rigorous optimization algorithm. To reconstruct a time-domain signal with inverse short-time Fourier transform (STFT), both full-band STFT magnitude and phase should be predicted from a given mel-spectrogram. Their joint estimation has outperformed the cascaded full-band magnitude prediction and phase reconstruction by preventing error accumulation. However, the existing joint estimation method requires many iterations, and there remains room for performance improvement. We present an alternating direction method of multipliers (ADMM)-based joint estimation method motivated by its success in various nonconvex optimization problems including phase reconstruction. An efficient update of each variable is derived by exploiting the conditional independence among the variables. Our experiments demonstrate the effectiveness of the proposed method on speech and foley sounds.
#### Sub-band Domain Multi-Hypothesis Acoustic Echo Canceler Based Acoustic Scene Analysis
 - **Authors:** Benjamin J Southwell, Yin-Lee Ho, David Gunawan
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2501.05652

 - **Pdf link:** https://arxiv.org/pdf/2501.05652

 - **Abstract**
 This paper introduces a novel approach for acoustic scene analysis by exploiting an ensemble of statistics extracted from a sub-band domain multi-hypothesis acoustic echo canceler (SDMH-AEC). A well-designed SDMH-AEC employs multiple adaptive filtering strategies with potentially complementary behaviours during convergence, perturbations, and steady-state conditions. By aggregating statistics across the sub-bands, we derive a feature vector that exhibits strong discriminative power for distinguishing different acoustic events and estimating acoustic parameters. The complementary nature of the SDMH-AEC filters provides a rich source of information that can be extracted at insignificant cost for acoustic scene analysis tasks. We demonstrate the effectiveness of the proposed approach experimentally with real data containing double-talk, echo path change and events where the full-duplex device is physically moved. The extracted features enable acoustic scene analysis using existing echo cancellation algorithms and techniques.
#### MARS6: A Small and Robust Hierarchical-Codec Text-to-Speech Model
 - **Authors:** Matthew Baas, Pieter Scholtz, Arnav Mehta, Elliott Dyson, Akshat Prakash, Herman Kamper
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Computation and Language (cs.CL)
 - **Arxiv link:** https://arxiv.org/abs/2501.05787

 - **Pdf link:** https://arxiv.org/pdf/2501.05787

 - **Abstract**
 Codec-based text-to-speech (TTS) models have shown impressive quality with zero-shot voice cloning abilities. However, they often struggle with more expressive references or complex text inputs. We present MARS6, a robust encoder-decoder transformer for rapid, expressive TTS. MARS6 is built on recent improvements in spoken language modelling. Utilizing a hierarchical setup for its decoder, new speech tokens are processed at a rate of only 12 Hz, enabling efficient modelling of long-form text while retaining reconstruction quality. We combine several recent training and inference techniques to reduce repetitive generation and improve output stability and quality. This enables the 70M-parameter MARS6 to achieve similar performance to models many times larger. We show this in objective and subjective evaluations, comparing TTS output quality and reference speaker cloning ability. Project page: this https URL
#### Large Model Empowered Streaming Semantic Communications for Speech Translation
 - **Authors:** Zhenzi Weng, Geoffrey Ye Li
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2501.05859

 - **Pdf link:** https://arxiv.org/pdf/2501.05859

 - **Abstract**
 Semantic communications have been explored to perform downstream intelligent tasks by extracting and transmitting essential information. In this paper, we introduce a large model-empowered streaming semantic communication system for speech translation across various languages, named LaSC-ST. Specifically, we devise an edge-device collaborative semantic communication architecture by offloading the intricate semantic extraction module to edge servers, thereby reducing the computational burden on local devices. To support multilingual speech translation, pre-trained large speech models are utilized to learn unified semantic features from speech in different languages, breaking the constraint of a single input language and enhancing the practicality of the LaSC-ST. Moreover, the input speech is sequentially streamed into the developed system as short speech segments, which enables low transmission latency without the degradation in speech translation quality. A novel dynamic speech segmentation algorithm is proposed to further minimize the transmission latency by adaptively adjusting the duration of speech segments. According to simulation results, the LaSC-ST provides more accurate speech translation and achieves streaming transmission with lower latency compared to existing non-streaming semantic communication systems.
#### Estimation and Restoration of Unknown Nonlinear Distortion using Diffusion
 - **Authors:** Michal Švento, Eloi Moliner, Lauri Juvela, Alec Wright, Vesa Välimäki
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2501.05959

 - **Pdf link:** https://arxiv.org/pdf/2501.05959

 - **Abstract**
 The restoration of nonlinearly distorted audio signals, alongside the identification of the applied memoryless nonlinear operation, is studied. The paper focuses on the difficult but practically important case in which both the nonlinearity and the original input signal are unknown. The proposed method uses a generative diffusion model trained unconditionally on guitar or speech signals to jointly model and invert the nonlinear system at inference time. Both the memoryless nonlinear function model and the restored audio signal are obtained as output. Successful example case studies are presented including inversion of hard and soft clipping, digital quantization, half-wave rectification, and wavefolding nonlinearities. Our results suggest that, out of the nonlinear functions tested here, the cubic Catmull-Rom spline is best suited to approximating these nonlinearities. In the case of guitar recordings, comparisons with informed and supervised methods show that the proposed blind method is at least as good as they are in terms of objective metrics. Experiments on distorted speech show that the proposed blind method outperforms general-purpose speech enhancement techniques and restores the original voice quality. The proposed method can be applied to audio effects modeling, restoration of music and speech recordings, and characterization of analog recording media.
#### Low-Resource Text-to-Speech Synthesis Using Noise-Augmented Training of ForwardTacotron
 - **Authors:** Kishor Kayyar Lakshminarayana, Frank Zalkow, Christian Dittmar, Nicola Pia, Emanuel A.P. Habets
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2501.05976

 - **Pdf link:** https://arxiv.org/pdf/2501.05976

 - **Abstract**
 In recent years, several text-to-speech systems have been proposed to synthesize natural speech in zero-shot, few-shot, and low-resource scenarios. However, these methods typically require training with data from many different speakers. The speech quality across the speaker set typically is diverse and imposes an upper limit on the quality achievable for the low-resource speaker. In the current work, we achieve high-quality speech synthesis using as little as five minutes of speech from the desired speaker by augmenting the low-resource speaker data with noise and employing multiple sampling techniques during training. Our method requires only four high-quality, high-resource speakers, which are easy to obtain and use in practice. Our low-complexity method achieves improved speaker similarity compared to the state-of-the-art zero-shot method HierSpeech++ and the recent low-resource method AdapterMix while maintaining comparable naturalness. Our proposed approach can also reduce the data requirements for speech synthesis for new speakers and languages.
#### Unmasking Deepfakes: Leveraging Augmentations and Features Variability for Deepfake Speech Detection
 - **Authors:** Inbal Rimon, Oren Gal, Haim Permuter
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2501.05545

 - **Pdf link:** https://arxiv.org/pdf/2501.05545

 - **Abstract**
 The detection of deepfake speech has become increasingly challenging with the rapid evolution of deepfake technologies. In this paper, we propose a hybrid architecture for deepfake speech detection, combining a self-supervised learning framework for feature extraction with a classifier head to form an end-to-end model. Our approach incorporates both audio-level and feature-level augmentation techniques. Specifically, we introduce and analyze various masking strategies for augmenting raw audio spectrograms and for enhancing feature representations during training. We incorporate compression augmentations during the pretraining phase of the feature extractor to address the limitations of small, single-language datasets. We evaluate the model on the ASVSpoof5 (ASVSpoof 2024) challenge, achieving state-of-the-art results in Track 1 under closed conditions with an Equal Error Rate of 4.37%. By employing different pretrained feature extractors, the model achieves an enhanced EER of 3.39%. Our model demonstrates robust performance against unseen deepfake attacks and exhibits strong generalization across different codecs.
#### FreeSVC: Towards Zero-shot Multilingual Singing Voice Conversion
 - **Authors:** Alef Iury Siqueira Ferreira, Lucas Rafael Gris, Augusto Seben da Rosa, Frederico Santos de Oliveira, Edresson Casanova, Rafael Teixeira Sousa, Arnaldo Candido Junior, Anderson da Silva Soares, Arlindo Galvão Filho
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2501.05586

 - **Pdf link:** https://arxiv.org/pdf/2501.05586

 - **Abstract**
 This work presents FreeSVC, a promising multilingual singing voice conversion approach that leverages an enhanced VITS model with Speaker-invariant Clustering (SPIN) for better content representation and the State-of-the-Art (SOTA) speaker encoder ECAPA2. FreeSVC incorporates trainable language embeddings to handle multiple languages and employs an advanced speaker encoder to disentangle speaker characteristics from linguistic content. Designed for zero-shot learning, FreeSVC enables cross-lingual singing voice conversion without extensive language-specific training. We demonstrate that a multilingual content extractor is crucial for optimal cross-language conversion. Our source code and models are publicly available.
#### CognoSpeak: an automatic, remote assessment of early cognitive decline in real-world conversational speech
 - **Authors:** Madhurananda Pahar, Fuxiang Tao, Bahman Mirheidari, Nathan Pevy, Rebecca Bright, Swapnil Gadgil, Lise Sproson, Dorota Braun, Caitlin Illingworth, Daniel Blackburn, Heidi Christensen
 - **Subjects:** Subjects:
Sound (cs.SD); Machine Learning (cs.LG); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2501.05755

 - **Pdf link:** https://arxiv.org/pdf/2501.05755

 - **Abstract**
 The early signs of cognitive decline are often noticeable in conversational speech, and identifying those signs is crucial in dealing with later and more serious stages of neurodegenerative diseases. Clinical detection is costly and time-consuming and although there has been recent progress in the automatic detection of speech-based cues, those systems are trained on relatively small databases, lacking detailed metadata and demographic information. This paper presents CognoSpeak and its associated data collection efforts. CognoSpeak asks memory-probing long and short-term questions and administers standard cognitive tasks such as verbal and semantic fluency and picture description using a virtual agent on a mobile or web platform. In addition, it collects multimodal data such as audio and video along with a rich set of metadata from primary and secondary care, memory clinics and remote settings like people's homes. Here, we present results from 126 subjects whose audio was manually transcribed. Several classic classifiers, as well as large language model-based classifiers, have been investigated and evaluated across the different types of prompts. We demonstrate a high level of performance; in particular, we achieved an F1-score of 0.873 using a DistilBERT model to discriminate people with cognitive impairment (dementia and people with mild cognitive impairment (MCI)) from healthy volunteers using the memory responses, fluency tasks and cookie theft picture description. CognoSpeak is an automatic, remote, low-cost, repeatable, non-invasive and less stressful alternative to existing clinical cognitive assessments.
#### Towards Early Prediction of Self-Supervised Speech Model Performance
 - **Authors:** Ryan Whetten, Lucas Maison, Titouan Parcollet, Marco Dinarelli, Yannick Estève
 - **Subjects:** Subjects:
Sound (cs.SD); Computation and Language (cs.CL); Machine Learning (cs.LG); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2501.05966

 - **Pdf link:** https://arxiv.org/pdf/2501.05966

 - **Abstract**
 In Self-Supervised Learning (SSL), pre-training and evaluation are resource intensive. In the speech domain, current indicators of the quality of SSL models during pre-training, such as the loss, do not correlate well with downstream performance. Consequently, it is often difficult to gauge the final downstream performance in a cost efficient manner during pre-training. In this work, we propose unsupervised efficient methods that give insights into the quality of the pre-training of SSL speech models, namely, measuring the cluster quality and rank of the embeddings of the SSL model. Results show that measures of cluster quality and rank correlate better with downstream performance than the pre-training loss with only one hour of unlabeled audio, reducing the need for GPU hours and labeled data in SSL model evaluation.
#### Comparing Self-Supervised Learning Models Pre-Trained on Human Speech and Animal Vocalizations for Bioacoustics Processing
 - **Authors:** Eklavya Sarkar, Mathew Magimai.-Doss
 - **Subjects:** Subjects:
Machine Learning (cs.LG); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2501.05987

 - **Pdf link:** https://arxiv.org/pdf/2501.05987

 - **Abstract**
 Self-supervised learning (SSL) foundation models have emerged as powerful, domain-agnostic, general-purpose feature extractors applicable to a wide range of tasks. Such models pre-trained on human speech have demonstrated high transferability for bioacoustic processing. This paper investigates (i) whether SSL models pre-trained directly on animal vocalizations offer a significant advantage over those pre-trained on speech, and (ii) whether fine-tuning speech-pretrained models on automatic speech recognition (ASR) tasks can enhance bioacoustic classification. We conduct a comparative analysis using three diverse bioacoustic datasets and two different bioacoustic tasks. Results indicate that pre-training on bioacoustic data provides only marginal improvements over speech-pretrained models, with comparable performance in most scenarios. Fine-tuning on ASR tasks yields mixed outcomes, suggesting that the general-purpose representations learned during SSL pre-training are already well-suited for bioacoustic tasks. These findings highlight the robustness of speech-pretrained SSL models for bioacoustics and imply that extensive fine-tuning may not be necessary for optimal performance.
#### Benchmarking Rotary Position Embeddings for Automatic Speech Recognition
 - **Authors:** Shucong Zhang, Titouan Parcollet, Rogier van Dalen, Sourav Bhattacharya
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Artificial Intelligence (cs.AI); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2501.06051

 - **Pdf link:** https://arxiv.org/pdf/2501.06051

 - **Abstract**
 Rotary Position Embedding (RoPE) encodes relative and absolute positional information in Transformer-based models through rotation matrices applied to input vectors within sequences. While RoPE has demonstrated superior performance compared to other positional embedding technologies in natural language processing tasks, its effectiveness in speech processing applications remains understudied. In this work, we conduct a comprehensive evaluation of RoPE across diverse automatic speech recognition (ASR) tasks. Our experimental results demonstrate that for ASR tasks, RoPE consistently achieves lower error rates compared to the currently widely used relative positional embedding. To facilitate further research, we release the implementation and all experimental recipes through the SpeechBrain toolkit.
#### xLSTM-SENet: xLSTM for Single-Channel Speech Enhancement
 - **Authors:** Nikolai Lund Kühne, Jan Østergaard, Jesper Jensen, Zheng-Hua Tan
 - **Subjects:** Subjects:
Sound (cs.SD); Artificial Intelligence (cs.AI); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2501.06146

 - **Pdf link:** https://arxiv.org/pdf/2501.06146

 - **Abstract**
 While attention-based architectures, such as Conformers, excel in speech enhancement, they face challenges such as scalability with respect to input sequence length. In contrast, the recently proposed Extended Long Short-Term Memory (xLSTM) architecture offers linear scalability. However, xLSTM-based models remain unexplored for speech enhancement. This paper introduces xLSTM-SENet, the first xLSTM-based single-channel speech enhancement system. A comparative analysis reveals that xLSTM-and notably, even LSTM-can match or outperform state-of-the-art Mamba- and Conformer-based systems across various model sizes in speech enhancement on the VoiceBank+Demand dataset. Through ablation studies, we identify key architectural design choices such as exponential gating and bidirectionality contributing to its effectiveness. Our best xLSTM-based model, xLSTM-SENet2, outperforms state-of-the-art Mamba- and Conformer-based systems on the Voicebank+DEMAND dataset.


by Zyzzyva0381 (Windy). 


2025-01-13
