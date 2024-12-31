# Showing new listings for Monday, 30 December 2024
Auto update papers at about 2:30am UTC (10:30am Beijing time) every weekday.


阅读 `Usage.md`了解如何使用此repo实现个性化的Arxiv论文推送

See `Usage.md` for instructions on how to personalize the repo. 


Keyword list: ['acoustic', 'reinforcement learning', 'reverb', 'meta', 'separate', 'reconstruction', 'noise', 'enhance', 'localization', 'speech']


Excluded: []


### Today: 23papers 
#### Investigating Acoustic-Textual Emotional Inconsistency Information for Automatic Depression Detection
 - **Authors:** Rongfeng Su, Changqing Xu, Xinyi Wu, Feng Xu, Xie Chen, Lan Wangt, Nan Yan
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Artificial Intelligence (cs.AI); Computation and Language (cs.CL)
 - **Arxiv link:** https://arxiv.org/abs/2412.18614

 - **Pdf link:** https://arxiv.org/pdf/2412.18614

 - **Abstract**
 Previous studies have demonstrated that emotional features from a single acoustic sentiment label can enhance depression diagnosis accuracy. Additionally, according to the Emotion Context-Insensitivity theory and our pilot study, individuals with depression might convey negative emotional content in an unexpectedly calm manner, showing a high degree of inconsistency in emotional expressions during natural conversations. So far, few studies have recognized and leveraged the emotional expression inconsistency for depression detection. In this paper, a multimodal cross-attention method is presented to capture the Acoustic-Textual Emotional Inconsistency (ATEI) information. This is achieved by analyzing the intricate local and long-term dependencies of emotional expressions across acoustic and textual domains, as well as the mismatch between the emotional content within both domains. A Transformer-based model is then proposed to integrate this ATEI information with various fusion strategies for detecting depression. Furthermore, a scaling technique is employed to adjust the ATEI feature degree during the fusion process, thereby enhancing the model's ability to discern patients with depression across varying levels of severity. To best of our knowledge, this work is the first to incorporate emotional expression inconsistency information into depression detection. Experimental results on a counseling conversational dataset illustrate the effectiveness of our method.
#### Structured Speaker-Deficiency Adaptation of Foundation Models for Dysarthric and Elderly Speech Recognition
 - **Authors:** Shujie Hu, Xurong Xie, Mengzhe Geng, Jiajun Deng, Zengrui Jin, Tianzi Wang, Mingyu Cui, Guinan Li, Zhaoqing Li, Helen Meng, Xunying Liu
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2412.18832

 - **Pdf link:** https://arxiv.org/pdf/2412.18832

 - **Abstract**
 Data-intensive fine-tuning of speech foundation models (SFMs) to scarce and diverse dysarthric and elderly speech leads to data bias and poor generalization to unseen speakers. This paper proposes novel structured speaker-deficiency adaptation approaches for SSL pre-trained SFMs on such data. Speaker and speech deficiency invariant SFMs were constructed in their supervised adaptive fine-tuning stage to reduce undue bias to training data speakers, and serves as a more neutral and robust starting point for test time unsupervised adaptation. Speech variability attributed to speaker identity and speech impairment severity, or aging induced neurocognitive decline, are modelled using separate adapters that can be combined together to model any seen or unseen speaker. Experiments on the UASpeech dysarthric and DementiaBank Pitt elderly speech corpora suggest structured speaker-deficiency adaptation of HuBERT and Wav2vec2-conformer models consistently outperforms baseline SFMs using either: a) no adapters; b) global adapters shared among all speakers; or c) single attribute adapters modelling speaker or deficiency labels alone by statistically significant WER reductions up to 3.01% and 1.50% absolute (10.86% and 6.94% relative) on the two tasks respectively. The lowest published WER of 19.45% (49.34% on very low intelligibility, 33.17% on unseen words) is obtained on the UASpeech test set of 16 dysarthric speakers.
#### Enhancing Audiovisual Speech Recognition through Bifocal Preference Optimization
 - **Authors:** Yihan Wu, Yichen Lu, Yifan Peng, Xihua Wang, Ruihua Song, Shinji Watanabe
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Artificial Intelligence (cs.AI)
 - **Arxiv link:** https://arxiv.org/abs/2412.19005

 - **Pdf link:** https://arxiv.org/pdf/2412.19005

 - **Abstract**
 Audiovisual Automatic Speech Recognition (AV-ASR) aims to improve speech recognition accuracy by leveraging visual signals. It is particularly challenging in unconstrained real-world scenarios across various domains due to noisy acoustic environments, spontaneous speech, and the uncertain use of visual information. Most previous works fine-tune audio-only ASR models on audiovisual datasets, optimizing them for conventional ASR objectives. However, they often neglect visual features and common errors in unconstrained video scenarios. In this paper, we propose using a preference optimization strategy to improve speech recognition accuracy for real-world videos. First, we create preference data via simulating common errors that occurred in AV-ASR from two focals: manipulating the audio or vision input and rewriting the output transcript. Second, we propose BPO-AVASR, a Bifocal Preference Optimization method to improve AV-ASR models by leveraging both input-side and output-side preference. Extensive experiments demonstrate that our approach significantly improves speech recognition accuracy across various domains, outperforming previous state-of-the-art models on real-world video speech recognition.
#### Attacking Voice Anonymization Systems with Augmented Feature and Speaker Identity Difference
 - **Authors:** Yanzhe Zhang, Zhonghao Bi, Feiyang Xiao, Xuefeng Yang, Qiaoxi Zhu, Jian Guan
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2412.19068

 - **Pdf link:** https://arxiv.org/pdf/2412.19068

 - **Abstract**
 This study focuses on the First VoicePrivacy Attacker Challenge within the ICASSP 2025 Signal Processing Grand Challenge, which aims to develop speaker verification systems capable of determining whether two anonymized speech signals are from the same speaker. However, differences between feature distributions of original and anonymized speech complicate this task. To address this challenge, we propose an attacker system that combines Data Augmentation enhanced feature representation and Speaker Identity Difference enhanced classifier to improve verification performance, termed DA-SID. Specifically, data augmentation strategies (i.e., data fusion and SpecAugment) are utilized to mitigate feature distribution gaps, while probabilistic linear discriminant analysis (PLDA) is employed to further enhance speaker identity difference. Our system significantly outperforms the baseline, demonstrating exceptional effectiveness and robustness against various voice anonymization systems, ultimately securing a top-5 ranking in the challenge.
#### Robust Speech and Natural Language Processing Models for Depression Screening
 - **Authors:** Y. Lu, A. Harati, T. Rutowski, R. Oliveira, P. Chlebek, E. Shriberg
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Computation and Language (cs.CL)
 - **Arxiv link:** https://arxiv.org/abs/2412.19072

 - **Pdf link:** https://arxiv.org/pdf/2412.19072

 - **Abstract**
 Depression is a global health concern with a critical need for increased patient screening. Speech technology offers advantages for remote screening but must perform robustly across patients. We have described two deep learning models developed for this purpose. One model is based on acoustics; the other is based on natural language processing. Both models employ transfer learning. Data from a depression-labeled corpus in which 11,000 unique users interacted with a human-machine application using conversational speech is used. Results on binary depression classification have shown that both models perform at or above AUC=0.80 on unseen data with no speaker overlap. Performance is further analyzed as a function of test subset characteristics, finding that the models are generally robust over speaker and session variables. We conclude that models based on these approaches offer promise for generalized automated depression screening.
#### Graph-Enhanced Dual-Stream Feature Fusion with Pre-Trained Model for Acoustic Traffic Monitoring
 - **Authors:** Shitong Fan, Feiyang Xiao, Wenbo Wang, Shuhan Qi, Qiaoxi Zhu, Wenwu Wang, Jian Guan
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Signal Processing (eess.SP)
 - **Arxiv link:** https://arxiv.org/abs/2412.19078

 - **Pdf link:** https://arxiv.org/pdf/2412.19078

 - **Abstract**
 Microphone array techniques are widely used in sound source localization and smart city acoustic-based traffic monitoring, but these applications face significant challenges due to the scarcity of labeled real-world traffic audio data and the complexity and diversity of application scenarios. The DCASE Challenge's Task 10 focuses on using multi-channel audio signals to count vehicles (cars or commercial vehicles) and identify their directions (left-to-right or vice versa). In this paper, we propose a graph-enhanced dual-stream feature fusion network (GEDF-Net) for acoustic traffic monitoring, which simultaneously considers vehicle type and direction to improve detection. We propose a graph-enhanced dual-stream feature fusion strategy which consists of a vehicle type feature extraction (VTFE) branch, a vehicle direction feature extraction (VDFE) branch, and a frame-level feature fusion module to combine the type and direction feature for enhanced performance. A pre-trained model (PANNs) is used in the VTFE branch to mitigate data scarcity and enhance the type features, followed by a graph attention mechanism to exploit temporal relationships and highlight important audio events within these features. The frame-level fusion of direction and type features enables fine-grained feature representation, resulting in better detection performance. Experiments demonstrate the effectiveness of our proposed method. GEDF-Net is our submission that achieved 1st place in the DCASE 2024 Challenge Task 10.
#### Causal Speech Enhancement with Predicting Semantics based on Quantized Self-supervised Learning Features
 - **Authors:** Emiru Tsunoo, Yuki Saito, Wataru Nakata, Hiroshi Saruwatari
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2412.19248

 - **Pdf link:** https://arxiv.org/pdf/2412.19248

 - **Abstract**
 Real-time speech enhancement (SE) is essential to online speech communication. Causal SE models use only the previous context while predicting future information, such as phoneme continuation, may help performing causal SE. The phonetic information is often represented by quantizing latent features of self-supervised learning (SSL) models. This work is the first to incorporate SSL features with causality into an SE model. The causal SSL features are encoded and combined with spectrogram features using feature-wise linear modulation to estimate a mask for enhancing the noisy input speech. Simultaneously, we quantize the causal SSL features using vector quantization to represent phonetic characteristics as semantic tokens. The model not only encodes SSL features but also predicts the future semantic tokens in multi-task learning (MTL). The experimental results using VoiceBank + DEMAND dataset show that our proposed method achieves 2.88 in PESQ, especially with semantic prediction MTL, in which we confirm that the semantic prediction played an important role in causal SE.
#### VoiceDiT: Dual-Condition Diffusion Transformer for Environment-Aware Speech Synthesis
 - **Authors:** Jaemin Jung, Junseok Ahn, Chaeyoung Jung, Tan Dat Nguyen, Youngjoon Jang, Joon Son Chung
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2412.19259

 - **Pdf link:** https://arxiv.org/pdf/2412.19259

 - **Abstract**
 We present VoiceDiT, a multi-modal generative model for producing environment-aware speech and audio from text and visual prompts. While aligning speech with text is crucial for intelligible speech, achieving this alignment in noisy conditions remains a significant and underexplored challenge in the field. To address this, we present a novel audio generation pipeline named VoiceDiT. This pipeline includes three key components: (1) the creation of a large-scale synthetic speech dataset for pre-training and a refined real-world speech dataset for fine-tuning, (2) the Dual-DiT, a model designed to efficiently preserve aligned speech information while accurately reflecting environmental conditions, and (3) a diffusion-based Image-to-Audio Translator that allows the model to bridge the gap between audio and image, facilitating the generation of environmental sound that aligns with the multi-modal prompts. Extensive experimental results demonstrate that VoiceDiT outperforms previous models on real-world datasets, showcasing significant improvements in both audio quality and modality integration.
#### Towards a Single ASR Model That Generalizes to Disordered Speech
 - **Authors:** Jimmy Tobin, Katrin Tomanek, Subhashini Venugopalan
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2412.19315

 - **Pdf link:** https://arxiv.org/pdf/2412.19315

 - **Abstract**
 This study investigates the impact of integrating a dataset of disordered speech recordings ($\sim$1,000 hours) into the fine-tuning of a near state-of-the-art ASR baseline system. Contrary to what one might expect, despite the data being less than 1% of the training data of the ASR system, we find a considerable improvement in disordered speech recognition accuracy. Specifically, we observe a 33% improvement on prompted speech, and a 26% improvement on a newly gathered spontaneous, conversational dataset of disordered speech. Importantly, there is no significant performance decline on standard speech recognition benchmarks. Further, we observe that the proposed tuning strategy helps close the gap between the baseline system and personalized models by 64% highlighting the significant progress as well as the room for improvement. Given the substantial benefits of our findings, this experiment suggests that from a fairness perspective, incorporating a small fraction of high quality disordered speech data in a training recipe is an easy step that could be done to make speech technology more accessible for users with speech disabilities.
#### Meta-Learning-Based Delayless Subband Adaptive Filter using Complex Self-Attention for Active Noise Control
 - **Authors:** Pengxing Feng, Hing Cheung So
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Machine Learning (cs.LG); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2412.19471

 - **Pdf link:** https://arxiv.org/pdf/2412.19471

 - **Abstract**
 Active noise control typically employs adaptive filtering to generate secondary noise, where the least mean square algorithm is the most widely used. However, traditional updating rules are linear and exhibit limited effectiveness in addressing nonlinear environments and nonstationary noise. To tackle this challenge, we reformulate the active noise control problem as a meta-learning problem and propose a meta-learning-based delayless subband adaptive filter with deep neural networks. The core idea is to utilize a neural network as an adaptive algorithm that can adapt to different environments and types of noise. The neural network will train under noisy observations, implying that it recognizes the optimized updating rule without true labels. A single-headed attention recurrent neural network is devised with learnable feature embedding to update the adaptive filter weight efficiently, enabling accurate computation of the secondary source to attenuate the unwanted primary noise. In order to relax the time constraint on updating the adaptive filter weights, the delayless subband architecture is employed, which will allow the system to be updated less frequently as the downsampling factor increases. In addition, the delayless subband architecture does not introduce additional time delays in active noise control systems. A skip updating strategy is introduced to decrease the updating frequency further so that machines with limited resources have more possibility to board our meta-learning-based model. Extensive multi-condition training ensures generalization and robustness against various types of noise and environments. Simulation results demonstrate that our meta-learning-based model achieves superior noise reduction performance compared to traditional methods.
#### Simi-SFX: A similarity-based conditioning method for controllable sound effect synthesis
 - **Authors:** Yunyi Liu, Craig Jin
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2412.18710

 - **Pdf link:** https://arxiv.org/pdf/2412.18710

 - **Abstract**
 Generating sound effects with controllable variations is a challenging task, traditionally addressed using sophisticated physical models that require in-depth knowledge of signal processing parameters and algorithms. In the era of generative and large language models, text has emerged as a common, human-interpretable interface for controlling sound synthesis. However, the discrete and qualitative nature of language tokens makes it difficult to capture subtle timbral variations across different sounds. In this research, we propose a novel similarity-based conditioning method for sound synthesis, leveraging differentiable digital signal processing (DDSP). This approach combines the use of latent space for learning and controlling audio timbre with an intuitive guiding vector, normalized within the range [0,1], to encode categorical acoustic information. By utilizing pre-trained audio representation models, our method achieves expressive and fine-grained timbre control. To benchmark our approach, we introduce two sound effect datasets--Footstep-set and Impact-set--designed to evaluate both controllability and sound quality. Regression analysis demonstrates that the proposed similarity score effectively controls timbre variations and enables creative applications such as timbre interpolation between discrete classes. Our work provides a robust and versatile framework for sound effect synthesis, bridging the gap between traditional signal processing and modern machine learning techniques.
#### Intra- and Inter-modal Context Interaction Modeling for Conversational Speech Synthesis
 - **Authors:** Zhenqi Jia, Rui Liu
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2412.18733

 - **Pdf link:** https://arxiv.org/pdf/2412.18733

 - **Abstract**
 Conversational Speech Synthesis (CSS) aims to effectively take the multimodal dialogue history (MDH) to generate speech with appropriate conversational prosody for target utterance. The key challenge of CSS is to model the interaction between the MDH and the target utterance. Note that text and speech modalities in MDH have their own unique influences, and they complement each other to produce a comprehensive impact on the target utterance. Previous works did not explicitly model such intra-modal and inter-modal interactions. To address this issue, we propose a new intra-modal and inter-modal context interaction scheme-based CSS system, termed III-CSS. Specifically, in the training phase, we combine the MDH with the text and speech modalities in the target utterance to obtain four modal combinations, including Historical Text-Next Text, Historical Speech-Next Speech, Historical Text-Next Speech, and Historical Speech-Next Text. Then, we design two contrastive learning-based intra-modal and two inter-modal interaction modules to deeply learn the intra-modal and inter-modal context interaction. In the inference phase, we take MDH and adopt trained interaction modules to fully infer the speech prosody of the target utterance's text content. Subjective and objective experiments on the DailyTalk dataset show that III-CSS outperforms the advanced baselines in terms of prosody expressiveness. Code and speech samples are available at this https URL.
#### Towards Expressive Video Dubbing with Multiscale Multimodal Context Interaction
 - **Authors:** Yuan Zhao, Rui Liu, Gaoxiang Cong
 - **Subjects:** Subjects:
Multimedia (cs.MM); Computation and Language (cs.CL); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2412.18748

 - **Pdf link:** https://arxiv.org/pdf/2412.18748

 - **Abstract**
 Automatic Video Dubbing (AVD) generates speech aligned with lip motion and facial emotion from scripts. Recent research focuses on modeling multimodal context to enhance prosody expressiveness but overlooks two key issues: 1) Multiscale prosody expression attributes in the context influence the current sentence's prosody. 2) Prosody cues in context interact with the current sentence, impacting the final prosody expressiveness. To tackle these challenges, we propose M2CI-Dubber, a Multiscale Multimodal Context Interaction scheme for AVD. This scheme includes two shared M2CI encoders to model the multiscale multimodal context and facilitate its deep interaction with the current sentence. By extracting global and local features for each modality in the context, utilizing attention-based mechanisms for aggregation and interaction, and employing an interaction-based graph attention network for fusion, the proposed approach enhances the prosody expressiveness of synthesized speech for the current sentence. Experiments on the Chem dataset show our model outperforms baselines in dubbing expressiveness. The code and demos are available at \textcolor[rgb]{0.93,0.0,0.47}{this https URL}.
#### MRI2Speech: Speech Synthesis from Articulatory Movements Recorded by Real-time MRI
 - **Authors:** Neil Shah, Ayan Kashyap, Shirish Karande, Vineet Gandhi
 - **Subjects:** Subjects:
Sound (cs.SD); Artificial Intelligence (cs.AI); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2412.18836

 - **Pdf link:** https://arxiv.org/pdf/2412.18836

 - **Abstract**
 Previous real-time MRI (rtMRI)-based speech synthesis models depend heavily on noisy ground-truth speech. Applying loss directly over ground truth mel-spectrograms entangles speech content with MRI noise, resulting in poor intelligibility. We introduce a novel approach that adapts the multi-modal self-supervised AV-HuBERT model for text prediction from rtMRI and incorporates a new flow-based duration predictor for speaker-specific alignment. The predicted text and durations are then used by a speech decoder to synthesize aligned speech in any novel voice. We conduct thorough experiments on two datasets and demonstrate our method's generalization ability to unseen speakers. We assess our framework's performance by masking parts of the rtMRI video to evaluate the impact of different articulators on text prediction. Our method achieves a $15.18\%$ Word Error Rate (WER) on the USC-TIMIT MRI corpus, marking a huge improvement over the current state-of-the-art. Speech samples are available at \url{this https URL}
#### Advancing NAM-to-Speech Conversion with Novel Methods and the MultiNAM Dataset
 - **Authors:** Neil Shah, Shirish Karande, Vineet Gandhi
 - **Subjects:** Subjects:
Sound (cs.SD); Artificial Intelligence (cs.AI); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2412.18839

 - **Pdf link:** https://arxiv.org/pdf/2412.18839

 - **Abstract**
 Current Non-Audible Murmur (NAM)-to-speech techniques rely on voice cloning to simulate ground-truth speech from paired whispers. However, the simulated speech often lacks intelligibility and fails to generalize well across different speakers. To address this issue, we focus on learning phoneme-level alignments from paired whispers and text and employ a Text-to-Speech (TTS) system to simulate the ground-truth. To reduce dependence on whispers, we learn phoneme alignments directly from NAMs, though the quality is constrained by the available training data. To further mitigate reliance on NAM/whisper data for ground-truth simulation, we propose incorporating the lip modality to infer speech and introduce a novel diffusion-based method that leverages recent advancements in lip-to-speech technology. Additionally, we release the MultiNAM dataset with over $7.96$ hours of paired NAM, whisper, video, and text data from two speakers and benchmark all methods on this dataset. Speech samples and the dataset are available at \url{this https URL}
#### Preventing output saturation in active noise control: An output-constrained Kalman filter approach
 - **Authors:** Junwei Ji, Dongyuan Shi, Boxiang Wang, Xiaoyi Shen, Zhengding Luo, Woon-Seng Gan
 - **Subjects:** Subjects:
Systems and Control (eess.SY); Audio and Speech Processing (eess.AS); Signal Processing (eess.SP)
 - **Arxiv link:** https://arxiv.org/abs/2412.18887

 - **Pdf link:** https://arxiv.org/pdf/2412.18887

 - **Abstract**
 The Kalman filter (KF)-based active noise control (ANC) system demonstrates superior tracking and faster convergence compared to the least mean square (LMS) method, particularly in dynamic noise cancellation scenarios. However, in environments with extremely high noise levels, the power of the control signal can exceed the system's rated output power due to hardware limitations, leading to output saturation and subsequent non-linearity. To mitigate this issue, a modified KF with an output constraint is proposed. In this approach, the disturbance treated as an measurement is re-scaled by a constraint factor, which is determined by the system's rated power, the secondary path gain, and the disturbance power. As a result, the output power of the system, i.e. the control signal, is indirectly constrained within the maximum output of the system, ensuring stability. Simulation results indicate that the proposed algorithm not only achieves rapid suppression of dynamic noise but also effectively prevents non-linearity due to output saturation, highlighting its practical significance.
#### Robust Target Speaker Direction of Arrival Estimation
 - **Authors:** Zixuan Li, Shulin He, Xueliang Zhang
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2412.18913

 - **Pdf link:** https://arxiv.org/pdf/2412.18913

 - **Abstract**
 In multi-speaker environments the direction of arrival (DOA) of a target speaker is key for improving speech clarity and extracting target speaker's voice. However, traditional DOA estimation methods often struggle in the presence of noise, reverberation, and particularly when competing speakers are present. To address these challenges, we propose RTS-DOA, a robust real-time DOA estimation system. This system innovatively uses the registered speech of the target speaker as a reference and leverages full-band and sub-band spectral information from a microphone array to estimate the DOA of the target speaker's voice. Specifically, the system comprises a speech enhancement module for initially improving speech quality, a spatial module for learning spatial information, and a speaker module for extracting voiceprint features. Experimental results on the LibriSpeech dataset demonstrate that our RTS-DOA system effectively tackles multi-speaker scenarios and established new optimal benchmarks.
#### Indonesian-English Code-Switching Speech Synthesizer Utilizing Multilingual STEN-TTS and Bert LID
 - **Authors:** Ahmad Alfani Handoyo, Chung Tran, Dessi Puji Lestari, Sakriani Sakti
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Artificial Intelligence (cs.AI); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2412.19043

 - **Pdf link:** https://arxiv.org/pdf/2412.19043

 - **Abstract**
 Multilingual text-to-speech systems convert text into speech across multiple languages. In many cases, text sentences may contain segments in different languages, a phenomenon known as code-switching. This is particularly common in Indonesia, especially between Indonesian and English. Despite its significance, no research has yet developed a multilingual TTS system capable of handling code-switching between these two languages. This study addresses Indonesian-English code-switching in STEN-TTS. Key modifications include adding a language identification component to the text-to-phoneme conversion using finetuned BERT for per-word language identification, as well as removing language embedding from the base model. Experimental results demonstrate that the code-switching model achieves superior naturalness and improved speech intelligibility compared to the Indonesian and English baseline STEN-TTS models.
#### BSDB-Net: Band-Split Dual-Branch Network with Selective State Spaces Mechanism for Monaural Speech Enhancement
 - **Authors:** Cunhang Fan, Enrui Liu, Andong Li, Jianhua Tao, Jian Zhou, Jiahao Li, Chengshi Zheng, Zhao Lv
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2412.19099

 - **Pdf link:** https://arxiv.org/pdf/2412.19099

 - **Abstract**
 Although the complex spectrum-based speech enhancement(SE) methods have achieved significant performance, coupling amplitude and phase can lead to a compensation effect, where amplitude information is sacrificed to compensate for the phase that is harmful to SE. In addition, to further improve the performance of SE, many modules are stacked onto SE, resulting in increased model complexity that limits the application of SE. To address these problems, we proposed a dual-path network based on compressed frequency using Mamba. First, we extract amplitude and phase information through parallel dual branches. This approach leverages structured complex spectra to implicitly capture phase information and solves the compensation effect by decoupling amplitude and phase, and the network incorporates an interaction module to suppress unnecessary parts and recover missing components from the other branch. Second, to reduce network complexity, the network introduces a band-split strategy to compress the frequency dimension. To further reduce complexity while maintaining good performance, we designed a Mamba-based module that models the time and frequency dimensions under linear complexity. Finally, compared to baselines, our model achieves an average 8.3 times reduction in computational complexity while maintaining superior performance. Furthermore, it achieves a 25 times reduction in complexity compared to transformer-based models.
#### CoheDancers: Enhancing Interactive Group Dance Generation through Music-Driven Coherence Decomposition
 - **Authors:** Kaixing Yang, Xulong Tang, Haoyu Wu, Qinliang Xue, Biao Qin, Hongyan Liu, Zhaoxin Fan
 - **Subjects:** Subjects:
Sound (cs.SD); Multimedia (cs.MM); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2412.19123

 - **Pdf link:** https://arxiv.org/pdf/2412.19123

 - **Abstract**
 Dance generation is crucial and challenging, particularly in domains like dance performance and virtual gaming. In the current body of literature, most methodologies focus on Solo Music2Dance. While there are efforts directed towards Group Music2Dance, these often suffer from a lack of coherence, resulting in aesthetically poor dance performances. Thus, we introduce CoheDancers, a novel framework for Music-Driven Interactive Group Dance Generation. CoheDancers aims to enhance group dance generation coherence by decomposing it into three key aspects: synchronization, naturalness, and fluidity. Correspondingly, we develop a Cycle Consistency based Dance Synchronization strategy to foster music-dance correspondences, an Auto-Regressive-based Exposure Bias Correction strategy to enhance the fluidity of the generated dances, and an Adversarial Training Strategy to augment the naturalness of the group dance output. Collectively, these strategies enable CohdeDancers to produce highly coherent group dances with superior quality. Furthermore, to establish better benchmarks for Group Music2Dance, we construct the most diverse and comprehensive open-source dataset to date, I-Dancers, featuring rich dancer interactions, and create comprehensive evaluation metrics. Experimental evaluations on I-Dancers and other extant datasets substantiate that CoheDancers achieves unprecedented state-of-the-art performance. Code will be released.
#### Personalized Dynamic Music Emotion Recognition with Dual-Scale Attention-Based Meta-Learning
 - **Authors:** Dengming Zhang, Weitao You, Ziheng Liu, Lingyun Sun, Pei Chen
 - **Subjects:** Subjects:
Sound (cs.SD); Information Retrieval (cs.IR); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2412.19200

 - **Pdf link:** https://arxiv.org/pdf/2412.19200

 - **Abstract**
 Dynamic Music Emotion Recognition (DMER) aims to predict the emotion of different moments in music, playing a crucial role in music information retrieval. The existing DMER methods struggle to capture long-term dependencies when dealing with sequence data, which limits their performance. Furthermore, these methods often overlook the influence of individual differences on emotion perception, even though everyone has their own personalized emotional perception in the real world. Motivated by these issues, we explore more effective sequence processing methods and introduce the Personalized DMER (PDMER) problem, which requires models to predict emotions that align with personalized perception. Specifically, we propose a Dual-Scale Attention-Based Meta-Learning (DSAML) method. This method fuses features from a dual-scale feature extractor and captures both short and long-term dependencies using a dual-scale attention transformer, improving the performance in traditional DMER. To achieve PDMER, we design a novel task construction strategy that divides tasks by annotators. Samples in a task are annotated by the same annotator, ensuring consistent perception. Leveraging this strategy alongside meta-learning, DSAML can predict personalized perception of emotions with just one personalized annotation sample. Our objective and subjective experiments demonstrate that our method can achieve state-of-the-art performance in both traditional DMER and PDMER.
#### Improving Generalization for AI-Synthesized Voice Detection
 - **Authors:** Hainan Ren, Lin Li, Chun-Hao Liu, Xin Wang, Shu Hu
 - **Subjects:** Subjects:
Sound (cs.SD); Machine Learning (cs.LG); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2412.19279

 - **Pdf link:** https://arxiv.org/pdf/2412.19279

 - **Abstract**
 AI-synthesized voice technology has the potential to create realistic human voices for beneficial applications, but it can also be misused for malicious purposes. While existing AI-synthesized voice detection models excel in intra-domain evaluation, they face challenges in generalizing across different domains, potentially becoming obsolete as new voice generators emerge. Current solutions use diverse data and advanced machine learning techniques (e.g., domain-invariant representation, self-supervised learning), but are limited by predefined vocoders and sensitivity to factors like background noise and speaker identity. In this work, we introduce an innovative disentanglement framework aimed at extracting domain-agnostic artifact features related to vocoders. Utilizing these features, we enhance model learning in a flat loss landscape, enabling escape from suboptimal solutions and improving generalization. Extensive experiments on benchmarks show our approach outperforms state-of-the-art methods, achieving up to 5.12% improvement in the equal error rate metric in intra-domain and 7.59% in cross-domain evaluations.
#### Enhancing Whisper's Accuracy and Speed for Indian Languages through Prompt-Tuning and Tokenization
 - **Authors:** Kumud Tripathi, Raj Gothi, Pankaj Wasnik
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2412.19785

 - **Pdf link:** https://arxiv.org/pdf/2412.19785

 - **Abstract**
 Automatic speech recognition has recently seen a significant advancement with large foundational models such as Whisper. However, these models often struggle to perform well in low-resource languages, such as Indian languages. This paper explores two novel approaches to enhance Whisper's multilingual speech recognition performance in Indian languages. First, we propose prompt-tuning with language family information, which enhances Whisper's accuracy in linguistically similar languages. Second, we introduce a novel tokenizer that reduces the number of generated tokens, thereby accelerating Whisper's inference speed. Our extensive experiments demonstrate that the tokenizer significantly reduces inference time, while prompt-tuning enhances accuracy across various Whisper model sizes, including Small, Medium, and Large. Together, these techniques achieve a balance between optimal WER and inference speed.


by Zyzzyva0381 (Windy). 


2024-12-31
