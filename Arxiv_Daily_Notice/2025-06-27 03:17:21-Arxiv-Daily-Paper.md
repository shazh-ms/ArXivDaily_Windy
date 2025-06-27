# Showing new listings for Friday, 27 June 2025
Auto update papers at about 2:30am UTC (10:30am Beijing time) every weekday.


阅读 `Usage.md`了解如何使用此repo实现个性化的Arxiv论文推送

See `Usage.md` for instructions on how to personalize the repo. 


Keyword list: ['text-to-speech', 'text to speech', 'tts', 'LLM-based', 'speech', 'voice']


Excluded: []


### Today: 6papers 
#### CodecSlime: Temporal Redundancy Compression of Neural Speech Codec via Dynamic Frame Rate
 - **Authors:** Hankun Wang, Yiwei Guo, Chongtian Shao, Bohan Li, Xie Chen, Kai Yu
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2506.21074

 - **Pdf link:** https://arxiv.org/pdf/2506.21074

 - **Abstract**
 Neural speech codecs have been widely used in audio compression and various downstream tasks. Current mainstream codecs are fixed-frame-rate (FFR), which allocate the same number of tokens to every equal-duration slice. However, speech is inherently non-uniform in temporal information density. As a result, many tokens are wasted on steady-state segments like long vowels and silences. To address this mismatch, we present CodecSlime, a plugin-style method for compressing temporal redundancy through supporting dynamic frame rate (DFR) on neural speech codecs for the first time. Our method is unsupervised and architecture-agnostic, combining two key innovations, ScheDFR and Melt-and-Cool, for adapting inference and training, respectively. When integrated into a typical VQ-GAN codec backbone and operating at 40 Hz DFR ($\approx$ 600 bps), the reconstruction WER of CodecSlime is reduced by up to 46% relative to conventional FFR baselines with the same model architecture and similar bitrates, while other metrics are also competitive. CodecSlime also enables flexible trade-offs between reconstruction quality and bitrate: a single model supports inference at multiple frame rates and consistently outperforms FFR models at the corresponding frame rates. Audio samples are available at this https URL.
#### Post-training for Deepfake Speech Detection
 - **Authors:** Wanying Ge, Xin Wang, Xuechen Liu, Junichi Yamagishi
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2506.21090

 - **Pdf link:** https://arxiv.org/pdf/2506.21090

 - **Abstract**
 We introduce a post-training approach that adapts self-supervised learning (SSL) models for deepfake speech detection by bridging the gap between general pre-training and domain-specific fine-tuning. We present AntiDeepfake models, a series of post-trained models developed using a large-scale multilingual speech dataset containing over 56,000 hours of genuine speech and 18,000 hours of speech with various artifacts in over one hundred languages. Experimental results show that the post-trained models already exhibit strong robustness and generalization to unseen deepfake speech. When they are further fine-tuned on the Deepfake-Eval-2024 dataset, these models consistently surpass existing state-of-the-art detectors that do not leverage post-training. Model checkpoints and source code are available online.
#### Hybrid Deep Learning and Signal Processing for Arabic Dialect Recognition in Low-Resource Settings
 - **Authors:** Ghazal Al-Shwayyat, Omer Nezih Gerek
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Computation and Language (cs.CL); Sound (cs.SD); Signal Processing (eess.SP)
 - **Arxiv link:** https://arxiv.org/abs/2506.21386

 - **Pdf link:** https://arxiv.org/pdf/2506.21386

 - **Abstract**
 Arabic dialect recognition presents a significant challenge in speech technology due to the linguistic diversity of Arabic and the scarcity of large annotated datasets, particularly for underrepresented dialects. This research investigates hybrid modeling strategies that integrate classical signal processing techniques with deep learning architectures to address this problem in low-resource scenarios. Two hybrid models were developed and evaluated: (1) Mel-Frequency Cepstral Coefficients (MFCC) combined with a Convolutional Neural Network (CNN), and (2) Discrete Wavelet Transform (DWT) features combined with a Recurrent Neural Network (RNN). The models were trained on a dialect-filtered subset of the Common Voice Arabic dataset, with dialect labels assigned based on speaker metadata. Experimental results demonstrate that the MFCC + CNN architecture achieved superior performance, with an accuracy of 91.2% and strong precision, recall, and F1-scores, significantly outperforming the Wavelet + RNN configuration, which achieved an accuracy of 66.5%. These findings highlight the effectiveness of leveraging spectral features with convolutional models for Arabic dialect recognition, especially when working with limited labeled data. The study also identifies limitations related to dataset size, potential regional overlaps in labeling, and model optimization, providing a roadmap for future research. Recommendations for further improvement include the adoption of larger annotated corpora, integration of self-supervised learning techniques, and exploration of advanced neural architectures such as Transformers. Overall, this research establishes a strong baseline for future developments in Arabic dialect recognition within resource-constrained environments.
#### A Multi-Stage Framework for Multimodal Controllable Speech Synthesis
 - **Authors:** Rui Niu, Weihao Wu, Jie Chen, Long Ma, Zhiyong Wu
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2506.20945

 - **Pdf link:** https://arxiv.org/pdf/2506.20945

 - **Abstract**
 Controllable speech synthesis aims to control the style of generated speech using reference input, which can be of various modalities. Existing face-based methods struggle with robustness and generalization due to data quality constraints, while text prompt methods offer limited diversity and fine-grained control. Although multimodal approaches aim to integrate various modalities, their reliance on fully matched training data significantly constrains their performance and applicability. This paper proposes a 3-stage multimodal controllable speech synthesis framework to address these challenges. For face encoder, we use supervised learning and knowledge distillation to tackle generalization issues. Furthermore, the text encoder is trained on both text-face and text-speech data to enhance the diversity of the generated speech. Experimental results demonstrate that this method outperforms single-modal baseline methods in both face based and text prompt based speech synthesis, highlighting its effectiveness in generating high-quality speech.
#### Prompt-Guided Turn-Taking Prediction
 - **Authors:** Koji Inoue, Mikey Elmers, Yahui Fu, Zi Haur Pang, Divesh Lala, Keiko Ochi, Tatsuya Kawahara
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2506.21191

 - **Pdf link:** https://arxiv.org/pdf/2506.21191

 - **Abstract**
 Turn-taking prediction models are essential components in spoken dialogue systems and conversational robots. Recent approaches leverage transformer-based architectures to predict speech activity continuously and in real-time. In this study, we propose a novel model that enables turn-taking prediction to be dynamically controlled via textual prompts. This approach allows intuitive and explicit control through instructions such as "faster" or "calmer" adapting dynamically to conversational partners and contexts. The proposed model builds upon a transformer-based voice activity projection (VAP) model, incorporating textual prompt embeddings into both channel-wise transformers and a cross-channel transformer. We evaluated the feasibility of our approach using over 950 hours of human-human spoken dialogue data. Since textual prompt data for the proposed approach was not available in existing datasets, we utilized a large language model (LLM) to generate synthetic prompt sentences. Experimental results demonstrated that the proposed model improved prediction accuracy and effectively varied turn-taking timing behaviors according to the textual prompts.
#### Aligning Spoken Dialogue Models from User Interactions
 - **Authors:** Anne Wu, Laurent Mazaré, Neil Zeghidour, Alexandre Défossez
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Machine Learning (cs.LG); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2506.21463

 - **Pdf link:** https://arxiv.org/pdf/2506.21463

 - **Abstract**
 We propose a novel preference alignment framework for improving spoken dialogue models on real-time conversations from user interactions. Current preference learning methods primarily focus on text-based language models, and are not directly suited to the complexities of real-time speech interactions, with richer dynamics (e.g. interruption, interjection) and no explicit segmentation between speaker this http URL create a large-scale dataset of more than 150,000 preference pairs from raw multi-turn speech conversations, annotated with AI feedback, to cover preferences over both linguistic content and temporal context variations. We leverage offline alignment methods to finetune a full-duplex autoregressive speech-to-speech model. Extensive experiments demonstrate that feedback on generic conversations can be consistently effective in improving spoken dialogue models to produce more factual, safer and more contextually aligned interactions. We deploy the finetuned model and conduct holistic human evaluations to assess the impact beyond single-turn conversations. Our findings shed light on the importance of a well-calibrated balance among various dynamics, crucial for natural real-time speech dialogue systems.


by Zyzzyva0381 (Windy). 


2025-06-27
