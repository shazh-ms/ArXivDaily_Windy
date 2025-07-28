# Showing new listings for Monday, 28 July 2025
Auto update papers at about 2:30am UTC (10:30am Beijing time) every weekday.


阅读 `Usage.md`了解如何使用此repo实现个性化的Arxiv论文推送

See `Usage.md` for instructions on how to personalize the repo. 


Keyword list: ['text-to-speech', 'text to speech', 'tts', 'LLM-based', 'speech', 'voice']


Excluded: []


### Today: 11papers 
#### FD-Bench: A Full-Duplex Benchmarking Pipeline Designed for Full Duplex Spoken Dialogue Systems
 - **Authors:** Yizhou Peng, Yi-Wen Chao, Dianwen Ng, Yukun Ma, Chongjia Ni, Bin Ma, Eng Siong Chng
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Computation and Language (cs.CL)
 - **Arxiv link:** https://arxiv.org/abs/2507.19040

 - **Pdf link:** https://arxiv.org/pdf/2507.19040

 - **Abstract**
 Full-duplex spoken dialogue systems (FDSDS) enable more natural human-machine interactions by allowing real-time user interruptions and backchanneling, compared to traditional SDS that rely on turn-taking. However, existing benchmarks lack metrics for FD scenes, e.g., evaluating model performance during user interruptions. In this paper, we present a comprehensive FD benchmarking pipeline utilizing LLMs, TTS, and ASR to address this gap. It assesses FDSDS's ability to handle user interruptions, manage delays, and maintain robustness in challenging scenarios with diverse novel metrics. We applied our benchmark to three open-source FDSDS (Moshi, Freeze-omni, and VITA-1.5) using over 40 hours of generated speech, with 293 simulated conversations and 1,200 interruptions. The results show that all models continue to face challenges, such as failing to respond to user interruptions, under frequent disruptions and noisy conditions. Demonstrations, data, and code will be released.
#### Assessment of Personality Dimensions Across Situations Using Conversational Speech
 - **Authors:** Alice Zhang, Skanda Muralidhar, Daniel Gatica-Perez, Mathew Magimai-Doss
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Artificial Intelligence (cs.AI); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2507.19137

 - **Pdf link:** https://arxiv.org/pdf/2507.19137

 - **Abstract**
 Prior research indicates that users prefer assistive technologies whose personalities align with their own. This has sparked interest in automatic personality perception (APP), which aims to predict an individual's perceived personality traits. Previous studies in APP have treated personalities as static traits, independent of context. However, perceived personalities can vary by context and situation as shown in psychological research. In this study, we investigate the relationship between conversational speech and perceived personality for participants engaged in two work situations (a neutral interview and a stressful client interaction). Our key findings are: 1) perceived personalities differ significantly across interactions, 2) loudness, sound level, and spectral flux features are indicative of perceived extraversion, agreeableness, conscientiousness, and openness in neutral interactions, while neuroticism correlates with these features in stressful contexts, 3) handcrafted acoustic features and non-verbal features outperform speaker embeddings in inference of perceived personality, and 4) stressful interactions are more predictive of neuroticism, aligning with existing psychological research.
#### Should Top-Down Clustering Affect Boundaries in Unsupervised Word Discovery?
 - **Authors:** Simon Malan, Benjamin van Niekerk, Herman Kamper
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Computation and Language (cs.CL); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2507.19204

 - **Pdf link:** https://arxiv.org/pdf/2507.19204

 - **Abstract**
 We investigate the problem of segmenting unlabeled speech into word-like units and clustering these to create a lexicon. Prior work can be categorized into two frameworks. Bottom-up methods first determine boundaries and then cluster the fixed segmented words into a lexicon. In contrast, top-down methods incorporate information from the clustered words to inform boundary selection. However, it is unclear whether top-down information is necessary to improve segmentation. To explore this, we look at two similar approaches that differ in whether top-down clustering informs boundary selection. Our simple bottom-up strategy predicts word boundaries using the dissimilarity between adjacent self-supervised features, then clusters the resulting segments to construct a lexicon. Our top-down system is an updated version of the ES-KMeans dynamic programming method that iteratively uses K-means to update its boundaries. On the five-language ZeroSpeech benchmarks, both approaches achieve comparable state-of-the-art results, with the bottom-up system being nearly five times faster. Through detailed analyses, we show that the top-down influence of ES-KMeans can be beneficial (depending on factors like the candidate boundaries), but in many cases the simple bottom-up method performs just as well. For both methods, we show that the clustering step is a limiting factor. Therefore, we recommend that future work focus on improved clustering techniques and learning more discriminative word-like representations. Project code repository: this https URL.
#### Comparison of Knowledge Distillation Methods for Low-complexity Multi-microphone Speech Enhancement using the FT-JNF Architecture
 - **Authors:** Robert Metzger, Mattes Ohlenbusch, Christian Rollwage, Simon Doclo
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2507.19208

 - **Pdf link:** https://arxiv.org/pdf/2507.19208

 - **Abstract**
 Multi-microphone speech enhancement using deep neural networks (DNNs) has significantly progressed in recent years. However, many proposed DNN-based speech enhancement algorithms cannot be implemented on devices with limited hardware resources. Only lowering the complexity of such systems by reducing the number of parameters often results in worse performance. Knowledge Distillation (KD) is a promising approach for reducing DNN model size while preserving performance. In this paper, we consider the recently proposed Frequency-Time Joint Non-linear Filter (FT-JNF) architecture and investigate several KD methods to train smaller (student) models from a large pre-trained (teacher) model. Five KD methods are evaluated using direct output matching, the self-similarity of intermediate layers, and fused multi-layer losses. Experimental results on a simulated dataset using a compact array with five microphones show that three KD methods substantially improve the performance of student models compared to training without KD. A student model with only 25% of the teacher model's parameters achieves comparable PESQ scores at 0 dB SNR. Furthermore, a reduction of up to 96% in model size can be achieved with only a minimal decrease in PESQ scores.
#### Binaural Target Speaker Extraction using HRTFs and a Complex-Valued Neural Network
 - **Authors:** Yoav Ellinson, Sharon Gannot
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2507.19369

 - **Pdf link:** https://arxiv.org/pdf/2507.19369

 - **Abstract**
 In this work, we aim to imitate the human ability to selectively attend to a single speaker, even in the presence of multiple simultaneous talkers. We propose a novel approach for binaural target speaker extraction that leverages the listener's Head-Related Transfer Function (HRTF) to isolate the desired speaker. Notably, our method does not rely on speaker embeddings, making it speaker-independent and enabling strong generalization across multiple speech datasets in different languages. We employ a fully complex-valued neural network that operates directly on the complex-valued Short-Time Fourier Transform (STFT) of the mixed audio signals. This deviates from conventional approaches that use spectrograms or treat the real and imaginary components of the STFT as separate real-valued inputs. We first evaluate the method in an anechoic, noise-free scenario, where it demonstrates excellent extraction performance while effectively preserving the binaural cues of the target signal. We then test a modified variant under mild reverberation conditions. This version remains robust in reverberant environments, maintaining speech clarity, preserving source directionality, and simultaneously reducing reverberation.
#### HH-Codec: High Compression High-fidelity Discrete Neural Codec for Spoken Language Modeling
 - **Authors:** Rongkun Xue, Yazhe Niu, Shuai Hu, Zixin Yin, Yongqiang Yao, Jing Yang
 - **Subjects:** Subjects:
Sound (cs.SD); Artificial Intelligence (cs.AI); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2507.18897

 - **Pdf link:** https://arxiv.org/pdf/2507.18897

 - **Abstract**
 Discrete speech tokenization is a fundamental component in speech codecs. However, in large-scale speech-to-speech systems, the complexity of parallel streams from multiple quantizers and the computational cost of high-time-dimensional codecs pose significant challenges. In this paper, we introduce HH-Codec, a neural codec that achieves extreme compression at 24 tokens per second for 24 kHz audio while relying on single-quantizer inference. Our approach involves a carefully designed Vector Quantization space for Spoken Language Modeling, optimizing compression efficiency while minimizing information loss. Building on this, we propose an asymmetric encoder-decoder architecture (Audio-VQ-Mel-Audio) that leverages dual supervision and progressive training to enhance reconstruction stability and fidelity. HH-Codec achieves state-of-the-art performance in speech reconstruction with an ultra-low bandwidth of 0.3 kbps. We further evaluate its effectiveness in codebook utilization and generative model adaptation, with extensive ablations validating the necessity of each module. HH-Codec is available at this https URL.
#### MLLM-based Speech Recognition: When and How is Multimodality Beneficial?
 - **Authors:** Yiwen Guan, Viet Anh Trinh, Vivek Voleti, Jacob Whitehill
 - **Subjects:** Subjects:
Sound (cs.SD); Computation and Language (cs.CL); Multimedia (cs.MM); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2507.19037

 - **Pdf link:** https://arxiv.org/pdf/2507.19037

 - **Abstract**
 Recent advances in multi-modal large language models (MLLMs) have opened new possibilities for unified modeling of speech, text, images, and other modalities. Building on our prior work, this paper examines the conditions and model architectures under which multiple input modalities can improve automatic speech recognition (ASR) accuracy in noisy environments. Through experiments on synthetic and real-world data, we find that (1) harnessing more modalities usually improves ASR accuracy, as each modality provides complementary information, but the improvement depends on the amount of auditory noise. (2) Synchronized modalities (e.g., lip movements) are more useful at high noise levels whereas unsynchronized modalities (e.g., image context) are most helpful at moderate noise levels. (3) Higher-quality visual representations consistently improve ASR accuracy, highlighting the importance of developing more powerful visual encoders. (4) Mamba exhibits similar trends regarding the benefits of multimodality as do Transformers. (5) The input order of modalities as well as their weights in the loss function can significantly impact accuracy. These findings both offer practical insights and help to deepen our understanding of multi-modal speech recognition under challenging conditions.
#### From Continuous to Discrete: Cross-Domain Collaborative General Speech Enhancement via Hierarchical Language Models
 - **Authors:** Zhaoxi Mu, Rilin Chen, Andong Li, Meng Yu, Xinyu Yang, Dong Yu
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2507.19062

 - **Pdf link:** https://arxiv.org/pdf/2507.19062

 - **Abstract**
 This paper introduces OmniGSE, a novel general speech enhancement (GSE) framework designed to mitigate the diverse distortions that speech signals encounter in real-world scenarios. These distortions include background noise, reverberation, bandwidth limitations, signal clipping, and network packet loss. Existing methods typically focus on optimizing for a single type of distortion, often struggling to effectively handle the simultaneous presence of multiple distortions in complex scenarios. OmniGSE bridges this gap by integrating the strengths of discriminative and generative approaches through a two-stage architecture that enables cross-domain collaborative optimization. In the first stage, continuous features are enhanced using a lightweight channel-split NAC-RoFormer. In the second stage, discrete tokens are generated to reconstruct high-quality speech through language models. Specifically, we designed a hierarchical language model structure consisting of a RootLM and multiple BranchLMs. The RootLM models general acoustic features across codebook layers, while the BranchLMs explicitly capture the progressive relationships between different codebook levels. Experimental results demonstrate that OmniGSE surpasses existing models across multiple benchmarks, particularly excelling in scenarios involving compound distortions. These findings underscore the framework's potential for robust and versatile speech enhancement in real-world applications.
#### Face2VoiceSync: Lightweight Face-Voice Consistency for Text-Driven Talking Face Generation
 - **Authors:** Fang Kang, Yin Cao, Haoyu Chen
 - **Subjects:** Subjects:
Sound (cs.SD); Computer Vision and Pattern Recognition (cs.CV); Multimedia (cs.MM); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2507.19225

 - **Pdf link:** https://arxiv.org/pdf/2507.19225

 - **Abstract**
 Recent studies in speech-driven talking face generation achieve promising results, but their reliance on fixed-driven speech limits further applications (e.g., face-voice mismatch). Thus, we extend the task to a more challenging setting: given a face image and text to speak, generating both talking face animation and its corresponding speeches. Accordingly, we propose a novel framework, Face2VoiceSync, with several novel contributions: 1) Voice-Face Alignment, ensuring generated voices match facial appearance; 2) Diversity \& Manipulation, enabling generated voice control over paralinguistic features space; 3) Efficient Training, using a lightweight VAE to bridge visual and audio large-pretrained models, with significantly fewer trainable parameters than existing methods; 4) New Evaluation Metric, fairly assessing the diversity and identity consistency. Experiments show Face2VoiceSync achieves both visual and audio state-of-the-art performances on a single 40GB GPU.
#### The Eloquence team submission for task 1 of MLC-SLM challenge
 - **Authors:** Lorenzo Concina, Jordi Luque, Alessio Brutti, Marco Matassoni, Yuchen Zhang
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2507.19308

 - **Pdf link:** https://arxiv.org/pdf/2507.19308

 - **Abstract**
 In this paper, we present our studies and experiments carried out for the task 1 of the Challenge and Workshop on Multilingual Conversational Speech Language Model (MLC-SLM), which focuses on advancing multilingual conversational speech recognition through the development of speech language models architectures. Given the increasing relevance of real-world conversational data for building robust Spoken Dialogue Systems, we explore three approaches to multilingual ASR. First, we conduct an evaluation of the official baseline to better understand its strengths and limitations, by training two projectors (linear and qformer) with different foundation models. Second we leverage the SLAM-ASR framework to train a custom multilingual linear projector. Finally we investigate the role of contrastive learning and the extended conversational context in enhancing the robustness of recognition.
#### SpeechIQ: Speech Intelligence Quotient Across Cognitive Levels in Voice Understanding Large Language Models
 - **Authors:** Zhen Wan, Chao-Han Huck Yang, Yahan Yu, Jinchuan Tian, Sheng Li, Ke Hu, Zhehuai Chen, Shinji Watanabe, Fei Cheng, Chenhui Chu, Sadao Kurohashi
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Artificial Intelligence (cs.AI); Symbolic Computation (cs.SC); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2507.19361

 - **Pdf link:** https://arxiv.org/pdf/2507.19361

 - **Abstract**
 We introduce Speech-based Intelligence Quotient (SIQ) as a new form of human cognition-inspired evaluation pipeline for voice understanding large language models, LLM Voice, designed to assess their voice understanding ability. Moving beyond popular voice understanding metrics such as word error rate (WER), SIQ examines LLM Voice across three cognitive levels motivated by Bloom's Taxonomy: (1) Remembering (i.e., WER for verbatim accuracy); (2) Understanding (i.e., similarity of LLM's interpretations); and (3) Application (i.e., QA accuracy for simulating downstream tasks). We demonstrate that SIQ not only quantifies voice understanding abilities but also provides unified comparisons between cascaded methods (e.g., ASR LLM) and end-to-end models, identifies annotation errors in existing benchmarks, and detects hallucinations in LLM Voice. Our framework represents a first-of-its-kind intelligence examination that bridges cognitive principles with voice-oriented benchmarks, while exposing overlooked challenges in multi-modal training.


by Zyzzyva0381 (Windy). 


2025-07-28
