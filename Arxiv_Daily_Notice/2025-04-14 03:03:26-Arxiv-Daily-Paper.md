# Showing new listings for Monday, 14 April 2025
Auto update papers at about 2:30am UTC (10:30am Beijing time) every weekday.


阅读 `Usage.md`了解如何使用此repo实现个性化的Arxiv论文推送

See `Usage.md` for instructions on how to personalize the repo. 


Keyword list: ['acoustic', 'reinforcement learning', 'reverb', 'meta', 'separate', 'reconstruction', 'noise', 'enhance', 'localization', 'speech']


Excluded: []


### Today: 8papers 
#### Mitigating Timbre Leakage with Universal Semantic Mapping Residual Block for Voice Conversion
 - **Authors:** Na Li, Chuke Wang, Yu Gu, Zhifeng Li
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Artificial Intelligence (cs.AI); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2504.08524

 - **Pdf link:** https://arxiv.org/pdf/2504.08524

 - **Abstract**
 Voice conversion (VC) transforms source speech into a target voice by preserving the content. However, timbre information from the source speaker is inherently embedded in the content representations, causing significant timbre leakage and reducing similarity to the target speaker. To address this, we introduce a residual block to a content extractor. The residual block consists of two weighted branches: 1) universal semantic dictionary based Content Feature Re-expression (CFR) module, supplying timbre-free content representation. 2) skip connection to the original content layer, providing complementary fine-grained information. In the CFR module, each dictionary entry in the universal semantic dictionary represents a phoneme class, computed statistically using speech from multiple speakers, creating a stable, speaker-independent semantic set. We introduce a CFR method to obtain timbre-free content representations by expressing each content frame as a weighted linear combination of dictionary entries using corresponding phoneme posteriors as weights. Extensive experiments across various VC frameworks demonstrate that our approach effectively mitigates timbre leakage and significantly improves similarity to the target speaker.
#### Reverberation-based Features for Sound Event Localization and Detection with Distance Estimation
 - **Authors:** Davide Berghi, Philip J. B. Jackson
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD); Signal Processing (eess.SP)
 - **Arxiv link:** https://arxiv.org/abs/2504.08644

 - **Pdf link:** https://arxiv.org/pdf/2504.08644

 - **Abstract**
 Sound event localization and detection (SELD) involves predicting active sound event classes over time while estimating their positions. The localization subtask in SELD is usually treated as a direction of arrival estimation problem, ignoring source distance. Only recently, SELD was extended to 3D by incorporating distance estimation, enabling the prediction of sound event positions in 3D space (3D SELD). However, existing methods lack input features designed for distance estimation. We argue that reverberation encodes valuable information for this task. This paper introduces two novel feature formats for 3D SELD based on reverberation: one using direct-to-reverberant ratio (DRR) and another leveraging signal autocorrelation to provide the model with insights into early reflections. Pre-training on synthetic data improves relative distance error (RDE) and overall SELD score, with autocorrelation-based features reducing RDE by over 3 percentage points on the STARSS23 dataset. The code to extract the features is available at this http URL.
#### From Speech to Summary: A Comprehensive Survey of Speech Summarization
 - **Authors:** Fabian Retkowski, Maike Züfle, Andreas Sudmann, Dinah Pfau, Jan Niehues, Alexander Waibel
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2504.08024

 - **Pdf link:** https://arxiv.org/pdf/2504.08024

 - **Abstract**
 Speech summarization has become an essential tool for efficiently managing and accessing the growing volume of spoken and audiovisual content. However, despite its increasing importance, speech summarization is still not clearly defined and intersects with several research areas, including speech recognition, text summarization, and specific applications like meeting summarization. This survey not only examines existing datasets and evaluation methodologies, which are crucial for assessing the effectiveness of summarization approaches but also synthesizes recent developments in the field, highlighting the shift from traditional systems to advanced models like fine-tuned cascaded architectures and end-to-end solutions.
#### Generalized Multilingual Text-to-Speech Generation with Language-Aware Style Adaptation
 - **Authors:** Haowei Lou, Hye-young Paik, Sheng Li, Wen Hu, Lina Yao
 - **Subjects:** Subjects:
Sound (cs.SD); Computation and Language (cs.CL); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2504.08274

 - **Pdf link:** https://arxiv.org/pdf/2504.08274

 - **Abstract**
 Text-to-Speech (TTS) models can generate natural, human-like speech across multiple languages by transforming phonemes into waveforms. However, multilingual TTS remains challenging due to discrepancies in phoneme vocabularies and variations in prosody and speaking style across languages. Existing approaches either train separate models for each language, which achieve high performance at the cost of increased computational resources, or use a unified model for multiple languages that struggles to capture fine-grained, language-specific style variations. In this work, we propose LanStyleTTS, a non-autoregressive, language-aware style adaptive TTS framework that standardizes phoneme representations and enables fine-grained, phoneme-level style control across languages. This design supports a unified multilingual TTS model capable of producing accurate and high-quality speech without the need to train language-specific models. We evaluate LanStyleTTS by integrating it with several state-of-the-art non-autoregressive TTS architectures. Results show consistent performance improvements across different model backbones. Furthermore, we investigate a range of acoustic feature representations, including mel-spectrograms and autoencoder-derived latent features. Our experiments demonstrate that latent encodings can significantly reduce model size and computational cost while preserving high-quality speech generation.
#### Location-Oriented Sound Event Localization and Detection with Spatial Mapping and Regression Localization
 - **Authors:** Xueping Zhang, Yaxiong Chen, Ruilin Yao, Yunfei Zi, Shengwu Xiong
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2504.08365

 - **Pdf link:** https://arxiv.org/pdf/2504.08365

 - **Abstract**
 Sound Event Localization and Detection (SELD) combines the Sound Event Detection (SED) with the corresponding Direction Of Arrival (DOA). Recently, adopted event oriented multi-track methods affect the generality in polyphonic environments due to the limitation of the number of tracks. To enhance the generality in polyphonic environments, we propose Spatial Mapping and Regression Localization for SELD (SMRL-SELD). SMRL-SELD segments the 3D spatial space, mapping it to a 2D plane, and a new regression localization loss is proposed to help the results converge toward the location of the corresponding event. SMRL-SELD is location-oriented, allowing the model to learn event features based on orientation. Thus, the method enables the model to process polyphonic sounds regardless of the number of overlapping events. We conducted experiments on STARSS23 and STARSS22 datasets and our proposed SMRL-SELD outperforms the existing SELD methods in overall evaluation and polyphony environments.
#### Passive Underwater Acoustic Signal Separation based on Feature Decoupling Dual-path Network
 - **Authors:** Yucheng Liu, Longyu Jiang
 - **Subjects:** Subjects:
Sound (cs.SD); Artificial Intelligence (cs.AI); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2504.08371

 - **Pdf link:** https://arxiv.org/pdf/2504.08371

 - **Abstract**
 Signal separation in the passive underwater acoustic domain has heavily relied on deep learning techniques to isolate ship radiated noise. However, the separation networks commonly used in this domain stem from speech separation applications and may not fully consider the unique aspects of underwater acoustics beforehand, such as the influence of different propagation media, signal frequencies and modulation characteristics. This oversight highlights the need for tailored approaches that account for the specific characteristics of underwater sound propagation. This study introduces a novel temporal network designed to separate ship radiated noise by employing a dual-path model and a feature decoupling approach. The mixed signals' features are transformed into a space where they exhibit greater independence, with each dimension's significance decoupled. Subsequently, a fusion of local and global attention mechanisms is employed in the separation layer. Extensive comparisons showcase the effectiveness of this method when compared to other prevalent network models, as evidenced by its performance in the ShipsEar and DeepShip datasets.
#### On the Design of Diffusion-based Neural Speech Codecs
 - **Authors:** Pietro Foti, Andreas Brendel
 - **Subjects:** Subjects:
Sound (cs.SD); Artificial Intelligence (cs.AI); Multimedia (cs.MM); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2504.08470

 - **Pdf link:** https://arxiv.org/pdf/2504.08470

 - **Abstract**
 Recently, neural speech codecs (NSCs) trained as generative models have shown superior performance compared to conventional codecs at low bitrates. Although most state-of-the-art NSCs are trained as Generative Adversarial Networks (GANs), Diffusion Models (DMs), a recent class of generative models, represent a promising alternative due to their superior performance in image generation relative to GANs. Consequently, DMs have been successfully applied for audio and speech coding among various other audio generation applications. However, the design of diffusion-based NSCs has not yet been explored in a systematic way. We address this by providing a comprehensive analysis of diffusion-based NSCs divided into three contributions. First, we propose a categorization based on the conditioning and output domains of the DM. This simple conceptual framework allows us to define a design space for diffusion-based NSCs and to assign a category to existing approaches in the literature. Second, we systematically investigate unexplored designs by creating and evaluating new diffusion-based NSCs within the conceptual framework. Finally, we compare the proposed models to existing GAN and DM baselines through objective metrics and subjective listening tests.
#### On The Landscape of Spoken Language Models: A Comprehensive Survey
 - **Authors:** Siddhant Arora, Kai-Wei Chang, Chung-Ming Chien, Yifan Peng, Haibin Wu, Yossi Adi, Emmanuel Dupoux, Hung-Yi Lee, Karen Livescu, Shinji Watanabe
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2504.08528

 - **Pdf link:** https://arxiv.org/pdf/2504.08528

 - **Abstract**
 The field of spoken language processing is undergoing a shift from training custom-built, task-specific models toward using and optimizing spoken language models (SLMs) which act as universal speech processing systems. This trend is similar to the progression toward universal language models that has taken place in the field of (text) natural language processing. SLMs include both "pure" language models of speech -- models of the distribution of tokenized speech sequences -- and models that combine speech encoders with text language models, often including both spoken and written input or output. Work in this area is very diverse, with a range of terminology and evaluation settings. This paper aims to contribute an improved understanding of SLMs via a unifying literature survey of recent work in the context of the evolution of the field. Our survey categorizes the work in this area by model architecture, training, and evaluation choices, and describes some key challenges and directions for future work.


by Zyzzyva0381 (Windy). 


2025-04-14
