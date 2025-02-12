# Showing new listings for Tuesday, 11 February 2025
Auto update papers at about 2:30am UTC (10:30am Beijing time) every weekday.


阅读 `Usage.md`了解如何使用此repo实现个性化的Arxiv论文推送

See `Usage.md` for instructions on how to personalize the repo. 


Keyword list: ['acoustic', 'reinforcement learning', 'reverb', 'meta', 'separate', 'reconstruction', 'noise', 'enhance', 'localization', 'speech']


Excluded: []


### Today: 18papers 
#### Distillation and Pruning for Scalable Self-Supervised Representation-Based Speech Quality Assessment
 - **Authors:** Benjamin Stahl, Hannes Gamper
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2502.05356

 - **Pdf link:** https://arxiv.org/pdf/2502.05356

 - **Abstract**
 In this paper, we investigate distillation and pruning methods to reduce model size for non-intrusive speech quality assessment based on self-supervised representations. Our experiments build on XLS-R-SQA, a speech quality assessment model using wav2vec 2.0 XLS-R embeddings. We retrain this model on a large compilation of mean opinion score datasets, encompassing over 100,000 labeled clips. For distillation, using this model as a teacher, we generate pseudo-labels on unlabeled degraded speech signals and train student models of varying sizes. For pruning, we use a data-driven strategy. While data-driven pruning performs better at larger model sizes, distillation on unlabeled data is more effective for smaller model sizes. Distillation can halve the gap between the baseline's correlation with ground-truth MOS labels and that of the XLS-R-based teacher model, while reducing model size by two orders of magnitude compared to the teacher model.
#### Unbiased Sliced Wasserstein Kernels for High-Quality Audio Captioning
 - **Authors:** Manh Luong, Khai Nguyen, Dinh Phung, Gholamreza Haffari, Lizhen Qu
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Artificial Intelligence (cs.AI); Machine Learning (cs.LG)
 - **Arxiv link:** https://arxiv.org/abs/2502.05435

 - **Pdf link:** https://arxiv.org/pdf/2502.05435

 - **Abstract**
 Teacher-forcing training for audio captioning usually leads to exposure bias due to training and inference mismatch. Prior works propose the contrastive method to deal with caption degeneration. However, the contrastive method ignores the temporal information when measuring similarity across acoustic and linguistic modalities, leading to inferior performance. In this work, we develop the temporal-similarity score by introducing the unbiased sliced Wasserstein RBF (USW-RBF) kernel equipped with rotary positional embedding to account for temporal information across modalities. In contrast to the conventional sliced Wasserstein RBF kernel, we can form an unbiased estimation of USW-RBF kernel via Monte Carlo estimation. Therefore, it is well-suited to stochastic gradient optimization algorithms, and its approximation error decreases at a parametric rate of $\mathcal{O}(L^{-1/2})$ with $L$ Monte Carlo samples. Additionally, we introduce an audio captioning framework based on the unbiased sliced Wasserstein kernel, incorporating stochastic decoding methods to mitigate caption degeneration during the generation process. We conduct extensive quantitative and qualitative experiments on two datasets, AudioCaps and Clotho, to illustrate the capability of generating high-quality audio captions. Experimental results show that our framework is able to increase caption length, lexical diversity, and text-to-audio self-retrieval accuracy.
#### Less is More for Synthetic Speech Detection in the Wild
 - **Authors:** Ashi Garg, Zexin Cai, Henry Li Xinyuan, Leibny Paola GarcÃ­a-Perera, Kevin Duh, Sanjeev Khudanpur, Matthew Wiesner, Nicholas Andrews
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2502.05674

 - **Pdf link:** https://arxiv.org/pdf/2502.05674

 - **Abstract**
 Driven by advances in self-supervised learning for speech, state-of-the-art synthetic speech detectors have achieved low error rates on popular benchmarks such as ASVspoof. However, prior benchmarks do not address the wide range of real-world variability in speech. Are reported error rates realistic in real-world conditions? To assess detector failure modes and robustness under controlled distribution shifts, we introduce ShiftySpeech, a benchmark with more than 3000 hours of synthetic speech from 7 domains, 6 TTS systems, 12 vocoders, and 3 languages. We found that all distribution shifts degraded model performance, and contrary to prior findings, training on more vocoders, speakers, or with data augmentation did not guarantee better generalization. In fact, we found that training on less diverse data resulted in better generalization, and that a detector fit using samples from a single carefully selected vocoder and a single speaker achieved state-of-the-art results on the challenging In-the-Wild benchmark.
#### Target Speaker Lipreading by Audio-Visual Self-Distillation Pretraining and Speaker Adaptation
 - **Authors:** Jing-Xuan Zhang, Tingzhi Mao, Longjiang Guo, Jin Li, Lichen Zhang
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2502.05758

 - **Pdf link:** https://arxiv.org/pdf/2502.05758

 - **Abstract**
 Lipreading is an important technique for facilitating human-computer interaction in noisy environments. Our previously developed self-supervised learning method, AV2vec, which leverages multimodal self-distillation, has demonstrated promising performance in speaker-independent lipreading on the English LRS3 dataset. However, AV2vec faces challenges such as high training costs and a potential scarcity of audio-visual data for lipreading in languages other than English, such as Chinese. Additionally, most studies concentrate on speakerindependent lipreading models, which struggle to account for the substantial variation in speaking styles across di?erent speakers. To address these issues, we propose a comprehensive approach. First, we investigate cross-lingual transfer learning, adapting a pre-trained AV2vec model from a source language and optimizing it for the lipreading task in a target language. Second, we enhance the accuracy of lipreading for specific target speakers through a speaker adaptation strategy, which is not extensively explored in previous research. Third, after analyzing the complementary performance of lipreading with lip region-of-interest (ROI) and face inputs, we introduce a model ensembling strategy that integrates both, signi?cantly boosting model performance. Our method achieved a character error rate (CER) of 77.3% on the evaluation set of the ChatCLR dataset, which is lower than the top result from the 2024 Chat-scenario Chinese Lipreading Challenge.
#### Non-invasive electromyographic speech neuroprosthesis: a geometric perspective
 - **Authors:** Harshavardhana T. Gowda, Ferdous Rahimi, Lee M. Miller
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2502.05762

 - **Pdf link:** https://arxiv.org/pdf/2502.05762

 - **Abstract**
 In this article, we present a high-bandwidth egocentric neuromuscular speech interface for translating silently voiced speech articulations into textand audio. Specifically, we collect electromyogram (EMG) signals from multiple articulatorysites on the face and neck as individuals articulate speech in an alaryngeal manner to perform EMG-to-text or EMG-to-audio translation. Such an interface is useful for restoring audible speech in individuals who have lost the ability to speak intelligibly due to laryngectomy, neuromuscular disease, stroke, or trauma-induced damage (e.g., radiotherapy toxicity) to speech articulators. Previous works have focused on training text or speech synthesis models using EMG collected during audible speech articulations or by transferring audio targets from EMG collected during audible articulation to EMG collected during silent articulation. However, such paradigms are not suited for individuals who have already lost the ability to audibly articulate speech. We are the first to present an alignment-free EMG-to-text and EMG-to-audio conversion using only EMG collected during silently articulated speech in an open-sourced manner. On a limited vocabulary corpora, our approach achieves almost 2.4x improvement in word error rate with a model that is 25x smaller by leveraging the inherent geometry of EMG.
#### Audio-Visual Representation Learning via Knowledge Distillation from Speech Foundation Models
 - **Authors:** Jing-Xuan Zhang, Genshun Wan, Jianqing Gao, Zhen-Hua Ling
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2502.05766

 - **Pdf link:** https://arxiv.org/pdf/2502.05766

 - **Abstract**
 Audio-visual representation learning is crucial for advancing multimodal speech processing tasks, such as lipreading and audio-visual speech recognition. Recently, speech foundation models (SFMs) have shown remarkable generalization capabilities across various speech-related tasks. Building on this progress, we propose an audio-visual representation learning model that leverages cross-modal knowledge distillation from SFMs. In our method, SFMs serve as teachers, from which multi-layer hidden representations are extracted using clean audio inputs. We also introduce a multi-teacher ensemble method to distill the student, which receives audio-visual data as inputs. A novel representational knowledge distillation loss is employed to train the student during pretraining, which is also applied during finetuning to further enhance the performance on downstream tasks. Our experiments utilized both a self-supervised SFM, WavLM, and a supervised SFM, iFLYTEK-speech. The results demonstrated that our proposed method achieved superior or at least comparable performance to previous state-of-the-art baselines across automatic speech recognition, visual speech recognition, and audio-visual speech recognition tasks. Additionally, comprehensive ablation studies and the visualization of learned representations were conducted to evaluate the effectiveness of our proposed method.
#### Synergistic Effects of Knowledge Distillation and Structured Pruning for Self-Supervised Speech Models
 - **Authors:** Shiva Kumar C, Jitendra Kumar Dhiman, Nagaraj Adiga, Shatrughan Singh
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2502.05837

 - **Pdf link:** https://arxiv.org/pdf/2502.05837

 - **Abstract**
 Traditionally, Knowledge Distillation (KD) is used for model compression, often leading to suboptimal performance. In this paper, we evaluate the impact of combining KD loss with alternative pruning techniques, including Low-Rank Factorization (LRF) and l0 regularization, on a conformer-based pre-trained network under the paradigm of Self-Supervised Learning (SSL). We also propose a strategy to jointly prune and train an RNN-T-based ASR model, demonstrating that this approach yields superior performance compared to pruning a pre-trained network first and then using it for ASR training. This approach led to a significant reduction in word error rate: l0 and KD combination achieves the best non-streaming performance, with a 8.9% Relative Word Error Rate (RWER) improvement over the baseline, while LRF and KD combination yields the best results for streaming ASR, improving RWER by 13.4%.
#### On the use of Performer and Agent Attention for Spoken Language Identification
 - **Authors:** Jitendra Kumar dhiman, Jainag Ambati
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2502.05841

 - **Pdf link:** https://arxiv.org/pdf/2502.05841

 - **Abstract**
 One of the methods for language Identification (LID) involves deriving speech representation from pre-trained models using self-supervised learning, followed by fine-tuning the model for the LID task. State-of-the-art approaches for LID use an attention-based statistical pooling layer to facilitate the aggregation of contextual information across time frames of the embedding vectors extracted from the pre-trained model. In this paper, we delve into exploring recently proposed attention mechanisms, namely performer and agent-attention, in conjunction with the statistical pooling layer. The LID experiments are performed on three datasets: VoxPopuli, FLEURS, and VoxLingua. We compare their performance against vanilla self-attention. Our findings suggest that performer-attention outperforms self-attention and agent-attention exhibits comparable or occasionally superior performance to self-attention, while also being computationally less expensive.
#### Recent Advances in Discrete Speech Tokens: A Review
 - **Authors:** Yiwei Guo, Zhihan Li, Hankun Wang, Bohan Li, Chongtian Shao, Hanglei Zhang, Chenpeng Du, Xie Chen, Shujie Liu, Kai Yu
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Artificial Intelligence (cs.AI); Multimedia (cs.MM); Sound (cs.SD); Signal Processing (eess.SP)
 - **Arxiv link:** https://arxiv.org/abs/2502.06490

 - **Pdf link:** https://arxiv.org/pdf/2502.06490

 - **Abstract**
 The rapid advancement of speech generation technologies in the era of large language models (LLMs) has established discrete speech tokens as a foundational paradigm for speech representation. These tokens, characterized by their discrete, compact, and concise nature, are not only advantageous for efficient transmission and storage, but also inherently compatible with the language modeling framework, enabling seamless integration of speech into text-dominated LLM architectures. Current research categorizes discrete speech tokens into two principal classes: acoustic tokens and semantic tokens, each of which has evolved into a rich research domain characterized by unique design philosophies and methodological approaches. This survey systematically synthesizes the existing taxonomy and recent innovations in discrete speech tokenization, conducts a critical examination of the strengths and limitations of each paradigm, and presents systematic experimental comparisons across token types. Furthermore, we identify persistent challenges in the field and propose potential research directions, aiming to offer actionable insights to inspire future advancements in the development and application of discrete speech tokens.
#### Aligner-Encoders: Self-Attention Transformers Can Be Self-Transducers
 - **Authors:** Adam Stooke, Rohit Prabhavalkar, Khe Chai Sim, Pedro Moreno Mengibar
 - **Subjects:** Subjects:
Sound (cs.SD); Artificial Intelligence (cs.AI); Machine Learning (cs.LG); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2502.05232

 - **Pdf link:** https://arxiv.org/pdf/2502.05232

 - **Abstract**
 Modern systems for automatic speech recognition, including the RNN-Transducer and Attention-based Encoder-Decoder (AED), are designed so that the encoder is not required to alter the time-position of information from the audio sequence into the embedding; alignment to the final text output is processed during decoding. We discover that the transformer-based encoder adopted in recent years is actually capable of performing the alignment internally during the forward pass, prior to decoding. This new phenomenon enables a simpler and more efficient model, the "Aligner-Encoder". To train it, we discard the dynamic programming of RNN-T in favor of the frame-wise cross-entropy loss of AED, while the decoder employs the lighter text-only recurrence of RNN-T without learned cross-attention -- it simply scans embedding frames in order from the beginning, producing one token each until predicting the end-of-message. We conduct experiments demonstrating performance remarkably close to the state of the art, including a special inference configuration enabling long-form recognition. In a representative comparison, we measure the total inference time for our model to be 2x faster than RNN-T and 16x faster than AED. Lastly, we find that the audio-text alignment is clearly visible in the self-attention weights of a certain layer, which could be said to perform "self-transduction".
#### Koel-TTS: Enhancing LLM based Speech Generation with Preference Alignment and Classifier Free Guidance
 - **Authors:** Shehzeen Hussain, Paarth Neekhara, Xuesong Yang, Edresson Casanova, Subhankar Ghosh, Mikyas T. Desta, Roy Fejgin, Rafael Valle, Jason Li
 - **Subjects:** Subjects:
Sound (cs.SD); Artificial Intelligence (cs.AI); Machine Learning (cs.LG); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2502.05236

 - **Pdf link:** https://arxiv.org/pdf/2502.05236

 - **Abstract**
 While autoregressive speech token generation models produce speech with remarkable variety and naturalness, their inherent lack of controllability often results in issues such as hallucinations and undesired vocalizations that do not conform to conditioning inputs. We introduce Koel-TTS, a suite of enhanced encoder-decoder Transformer TTS models that address these challenges by incorporating preference alignment techniques guided by automatic speech recognition and speaker verification models. Additionally, we incorporate classifier-free guidance to further improve synthesis adherence to the transcript and reference speaker audio. Our experiments demonstrate that these optimizations significantly enhance target speaker similarity, intelligibility, and naturalness of synthesized speech. Notably, Koel-TTS directly maps text and context audio to acoustic tokens, and on the aforementioned metrics, outperforms state-of-the-art TTS models, despite being trained on a significantly smaller dataset. Audio samples and demos are available on our website.
#### Enhancing Expressive Voice Conversion with Discrete Pitch-Conditioned Flow Matching Model
 - **Authors:** Jialong Zuo, Shengpeng Ji, Minghui Fang, Ziyue Jiang, Xize Cheng, Qian Yang, Wenrui Liu, Guangyan Zhang, Zehai Tu, Yiwen Guo, Zhou Zhao
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2502.05471

 - **Pdf link:** https://arxiv.org/pdf/2502.05471

 - **Abstract**
 This paper introduces PFlow-VC, a conditional flow matching voice conversion model that leverages fine-grained discrete pitch tokens and target speaker prompt information for expressive voice conversion (VC). Previous VC works primarily focus on speaker conversion, with further exploration needed in enhancing expressiveness (such as prosody and emotion) for timbre conversion. Unlike previous methods, we adopt a simple and efficient approach to enhance the style expressiveness of voice conversion models. Specifically, we pretrain a self-supervised pitch VQVAE model to discretize speaker-irrelevant pitch information and leverage a masked pitch-conditioned flow matching model for Mel-spectrogram synthesis, which provides in-context pitch modeling capabilities for the speaker conversion model, effectively improving the voice style transfer capacity. Additionally, we improve timbre similarity by combining global timbre embeddings with time-varying timbre tokens. Experiments on unseen LibriTTS test-clean and emotional speech dataset ESD show the superiority of the PFlow-VC model in both timbre conversion and style transfer. Audio samples are available on the demo page this https URL.
#### IndexTTS: An Industrial-Level Controllable and Efficient Zero-Shot Text-To-Speech System
 - **Authors:** Wei Deng, Siyi Zhou, Jingchen Shu, Jinchao Wang, Lu Wang
 - **Subjects:** Subjects:
Sound (cs.SD); Artificial Intelligence (cs.AI); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2502.05512

 - **Pdf link:** https://arxiv.org/pdf/2502.05512

 - **Abstract**
 Recently, large language model (LLM) based text-to-speech (TTS) systems have gradually become the mainstream in the industry due to their high naturalness and powerful zero-shot voice cloning this http URL, we introduce the IndexTTS system, which is mainly based on the XTTS and Tortoise model. We add some novel improvements. Specifically, in Chinese scenarios, we adopt a hybrid modeling method that combines characters and pinyin, making the pronunciations of polyphonic characters and long-tail characters controllable. We also performed a comparative analysis of the Vector Quantization (VQ) with Finite-Scalar Quantization (FSQ) for codebook utilization of acoustic speech tokens. To further enhance the effect and stability of voice cloning, we introduce a conformer-based speech conditional encoder and replace the speechcode decoder with BigVGAN2. Compared with XTTS, it has achieved significant improvements in naturalness, content consistency, and zero-shot voice cloning. As for the popular TTS systems in the open-source, such as Fish-Speech, CosyVoice2, FireRedTTS and F5-TTS, IndexTTS has a relatively simple training process, more controllable usage, and faster inference speed. Moreover, its performance surpasses that of these systems. Our demos are available at this https URL.
#### Gender Bias in Instruction-Guided Speech Synthesis Models
 - **Authors:** Chun-Yi Kuan, Hung-yi Lee
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Machine Learning (cs.LG); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2502.05649

 - **Pdf link:** https://arxiv.org/pdf/2502.05649

 - **Abstract**
 Recent advancements in controllable expressive speech synthesis, especially in text-to-speech (TTS) models, have allowed for the generation of speech with specific styles guided by textual descriptions, known as style prompts. While this development enhances the flexibility and naturalness of synthesized speech, there remains a significant gap in understanding how these models handle vague or abstract style prompts. This study investigates the potential gender bias in how models interpret occupation-related prompts, specifically examining their responses to instructions like "Act like a nurse". We explore whether these models exhibit tendencies to amplify gender stereotypes when interpreting such prompts. Our experimental results reveal the model's tendency to exhibit gender bias for certain occupations. Moreover, models of different sizes show varying degrees of this bias across these occupations.
#### Large Language Model-based Nonnegative Matrix Factorization For Cardiorespiratory Sound Separation
 - **Authors:** Yasaman Torabi, Shahram Shirani, James P. Reilly
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS); Signal Processing (eess.SP)
 - **Arxiv link:** https://arxiv.org/abs/2502.05757

 - **Pdf link:** https://arxiv.org/pdf/2502.05757

 - **Abstract**
 This study represents the first integration of large language models (LLMs) with non-negative matrix factorization (NMF), marking a novel advancement in the source separation field. The LLM is employed in two unique ways: enhancing the separation results by providing detailed insights for disease prediction and operating in a feedback loop to optimize a fundamental frequency penalty added to the NMF cost function. We tested the algorithm on two datasets: 100 synthesized mixtures of real measurements, and 210 recordings of heart and lung sounds from a clinical manikin including both individual and mixed sounds, captured using a digital stethoscope. The approach consistently outperformed existing methods, demonstrating its potential to significantly enhance medical sound analysis for disease diagnostics.
#### Temporal Working Memory: Query-Guided Segment Refinement for Enhanced Multimodal Understanding
 - **Authors:** Xingjian Diao, Chunhui Zhang, Weiyi Wu, Zhongyu Ouyang, Peijun Qing, Ming Cheng, Soroush Vosoughi, Jiang Gui
 - **Subjects:** Subjects:
Computer Vision and Pattern Recognition (cs.CV); Multimedia (cs.MM); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2502.06020

 - **Pdf link:** https://arxiv.org/pdf/2502.06020

 - **Abstract**
 Multimodal foundation models (MFMs) have demonstrated significant success in tasks such as visual captioning, question answering, and image-text retrieval. However, these models face inherent limitations due to their finite internal capacity, which restricts their ability to process extended temporal sequences, a crucial requirement for comprehensive video and audio analysis. To overcome these challenges, we introduce a specialized cognitive module, temporal working memory (TWM), which aims to enhance the temporal modeling capabilities of MFMs. It selectively retains task-relevant information across temporal dimensions, ensuring that critical details are preserved throughout the processing of video and audio content. The TWM uses a query-guided attention approach to focus on the most informative multimodal segments within temporal sequences. By retaining only the most relevant content, TWM optimizes the use of the model's limited capacity, enhancing its temporal modeling ability. This plug-and-play module can be easily integrated into existing MFMs. With our TWM, nine state-of-the-art models exhibit significant performance improvements across tasks such as video captioning, question answering, and video-text retrieval. By enhancing temporal modeling, TWM extends the capability of MFMs to handle complex, time-sensitive data effectively. Our code is available at this https URL.
#### An adaptive filter bank based neural network approach for time delay estimation and speech enhancement
 - **Authors:** Lu Ma
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2502.06098

 - **Pdf link:** https://arxiv.org/pdf/2502.06098

 - **Abstract**
 Time delay estimation (TDE) plays a key role in acoustic echo cancellation (AEC) using adaptive filter method. Considerable residual echo will be left if estimation error arises. Here, in this paper, we proposed an adaptive filter bank based neural network approach where the delay is estimated by a bank of adaptive filters with overlapped time scope, and all the energy of filter weights are concatenated and feed to a classification network. The index with maximal probability is chosen as the estimated delay. Based on this TDE, an AEC scheme is designed using a neural network for residual echo and noise suppression, and the optimally-modified log-spectral amplitude (OMLSA) algorithm is adopted to make it robust. Also, a robust automatic gain control (AGC) scheme with spectrum smoothing method is designed to amplify speech segments. Performance evaluations reveal that higher performance can be achieved for our scheme.
#### Automatic Identification of Samples in Hip-Hop Music via Multi-Loss Training and an Artificial Dataset
 - **Authors:** Huw Cheston, Jan Van Balen, Simon Durand
 - **Subjects:** Subjects:
Sound (cs.SD); Machine Learning (cs.LG); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2502.06364

 - **Pdf link:** https://arxiv.org/pdf/2502.06364

 - **Abstract**
 Sampling, the practice of reusing recorded music or sounds from another source in a new work, is common in popular music genres like hip-hop and rap. Numerous services have emerged that allow users to identify connections between samples and the songs that incorporate them, with the goal of enhancing music discovery. Designing a system that can perform the same task automatically is challenging, as samples are commonly altered with audio effects like pitch- and time-stretching and may only be seconds long. Progress on this task has been minimal and is further blocked by the limited availability of training data. Here, we show that a convolutional neural network trained on an artificial dataset can identify real-world samples in commercial hip-hop music. We extract vocal, harmonic, and percussive elements from several databases of non-commercial music recordings using audio source separation, and train the model to fingerprint a subset of these elements in transformed versions of the original audio. We optimize the model using a joint classification and metric learning loss and show that it achieves 13% greater precision on real-world instances of sampling than a fingerprinting system using acoustic landmarks, and that it can recognize samples that have been both pitch shifted and time stretched. We also show that, for half of the commercial music recordings we tested, our model is capable of locating the position of a sample to within five seconds.


by Zyzzyva0381 (Windy). 


2025-02-12
