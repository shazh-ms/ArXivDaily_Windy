# Showing new listings for Friday, 29 August 2025
Auto update papers at about 2:30am UTC (10:30am Beijing time) every weekday.


阅读 `Usage.md`了解如何使用此repo实现个性化的Arxiv论文推送

See `Usage.md` for instructions on how to personalize the repo. 


Keyword list: ['text-to-speech', 'text to speech', 'tts', 'LLM-based', 'speech', 'voice']


Excluded: []


### Today: 7papers 
#### Unifying Diarization, Separation, and ASR with Multi-Speaker Encoder
 - **Authors:** Muhammad Shakeel, Yui Sudo, Yifan Peng, Chyi-Jiunn Lin, Shinji Watanabe
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Computation and Language (cs.CL); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2508.20474

 - **Pdf link:** https://arxiv.org/pdf/2508.20474

 - **Abstract**
 This paper presents a unified multi-speaker encoder (UME), a novel architecture that jointly learns representations for speaker diarization (SD), speech separation (SS), and multi-speaker automatic speech recognition (ASR) tasks using a shared speech foundational encoder. We leverage the hidden representations from multiple layers of UME as a residual weighted-sum encoding (RWSE) to effectively use information from different semantic levels, contributing to bottom-up alignment between tasks. This joint training approach captures the inherent interdependencies among the tasks, enhancing overall performance on overlapping speech data. Our evaluations demonstrate that UME substantially improves over the single-task baselines dedicated to SD, SS, and multi-speaker ASR on LibriMix evaluation sets. Notably, for SD, UME outperforms the previous studies, achieving diarization error rates of 1.37% and 2.29% on Libri2Mix and Libri3Mix evaluation sets, respectively.
#### CodecBench: A Comprehensive Benchmark for Acoustic and Semantic Evaluation
 - **Authors:** Ruifan Deng, Yitian Gong, Qinghui Gao, Luozhijie Jin, Qinyuan Cheng, Zhaoye Fei, Shimin Li, Xipeng Qiu
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2508.20660

 - **Pdf link:** https://arxiv.org/pdf/2508.20660

 - **Abstract**
 With the rise of multimodal large language models (LLMs), audio codec plays an increasingly vital role in encoding audio into discrete tokens, enabling integration of audio into text-based LLMs. Current audio codec captures two types of information: acoustic and semantic. As audio codec is applied to diverse scenarios in speech language model , it needs to model increasingly complex information and adapt to varied contexts, such as scenarios with multiple speakers, background noise, or richer paralinguistic information. However, existing codec's own evaluation has been limited by simplistic metrics and scenarios, and existing benchmarks for audio codec are not designed for complex application scenarios, which limits the assessment performance on complex datasets for acoustic and semantic capabilities. We introduce CodecBench, a comprehensive evaluation dataset to assess audio codec performance from both acoustic and semantic perspectives across four data domains. Through this benchmark, we aim to identify current limitations, highlight future research directions, and foster advances in the development of audio codec. The codes are available at this https URL.
#### Leveraging Discriminative Latent Representations for Conditioning GAN-Based Speech Enhancement
 - **Authors:** Shrishti Saha Shetu, Emanuël A. P. Habets, Andreas Brendel
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2508.20859

 - **Pdf link:** https://arxiv.org/pdf/2508.20859

 - **Abstract**
 Generative speech enhancement methods based on generative adversarial networks (GANs) and diffusion models have shown promising results in various speech enhancement tasks. However, their performance in very low signal-to-noise ratio (SNR) scenarios remains under-explored and limited, as these conditions pose significant challenges to both discriminative and generative state-of-the-art methods. To address this, we propose a method that leverages latent features extracted from discriminative speech enhancement models as generic conditioning features to improve GAN-based speech enhancement. The proposed method, referred to as DisCoGAN, demonstrates performance improvements over baseline models, particularly in low-SNR scenarios, while also maintaining competitive or superior performance in high-SNR conditions and on real-world recordings. We also conduct a comprehensive evaluation of conventional GAN-based architectures, including GANs trained end-to-end, GANs as a first processing stage, and post-filtering GANs, as well as discriminative models under low-SNR conditions. We show that DisCoGAN consistently outperforms existing methods. Finally, we present an ablation study that investigates the contributions of individual components within DisCoGAN and analyzes the impact of the discriminative conditioning method on overall performance.
#### Multilingual Dataset Integration Strategies for Robust Audio Deepfake Detection: A SAFE Challenge System
 - **Authors:** Hashim Ali, Surya Subramani, Lekha Bollinani, Nithin Sai Adupa, Sali El-Loh, Hafiz Malik
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Machine Learning (cs.LG)
 - **Arxiv link:** https://arxiv.org/abs/2508.20983

 - **Pdf link:** https://arxiv.org/pdf/2508.20983

 - **Abstract**
 The SAFE Challenge evaluates synthetic speech detection across three tasks: unmodified audio, processed audio with compression artifacts, and laundered audio designed to evade detection. We systematically explore self-supervised learning (SSL) front-ends, training data compositions, and audio length configurations for robust deepfake detection. Our AASIST-based approach incorporates WavLM large frontend with RawBoost augmentation, trained on a multilingual dataset of 256,600 samples spanning 9 languages and over 70 TTS systems from CodecFake, MLAAD v5, SpoofCeleb, Famous Figures, and MAILABS. Through extensive experimentation with different SSL front-ends, three training data versions, and two audio lengths, we achieved second place in both Task 1 (unmodified audio detection) and Task 3 (laundered audio detection), demonstrating strong generalization and robustness.
#### Towards Inclusive Communication: A Unified LLM-Based Framework for Sign Language, Lip Movements, and Audio Understanding
 - **Authors:** Jeong Hun Yeo, Hyeongseop Rha, Sungjune Park, Junil Won, Yong Man Ro
 - **Subjects:** Subjects:
Computer Vision and Pattern Recognition (cs.CV); Multimedia (cs.MM); Audio and Speech Processing (eess.AS); Image and Video Processing (eess.IV)
 - **Arxiv link:** https://arxiv.org/abs/2508.20476

 - **Pdf link:** https://arxiv.org/pdf/2508.20476

 - **Abstract**
 Audio is the primary modality for human communication and has driven the success of Automatic Speech Recognition (ASR) technologies. However, such systems remain inherently inaccessible to individuals who are deaf or hard of hearing. Visual alternatives such as sign language and lip reading offer effective substitutes, and recent advances in Sign Language Translation (SLT) and Visual Speech Recognition (VSR) have improved audio-less communication. Yet, these modalities have largely been studied in isolation, and their integration within a unified framework remains underexplored. In this paper, we introduce the first unified framework capable of handling diverse combinations of sign language, lip movements, and audio for spoken-language text generation. We focus on three main objectives: (i) designing a unified, modality-agnostic architecture capable of effectively processing heterogeneous inputs; (ii) exploring the underexamined synergy among modalities, particularly the role of lip movements as non-manual cues in sign language comprehension; and (iii) achieving performance on par with or superior to state-of-the-art models specialized for individual tasks. Building on this framework, we achieve performance on par with or better than task-specific state-of-the-art models across SLT, VSR, ASR, and AVSR. Furthermore, our analysis reveals that explicitly modeling lip movements as a separate modality significantly improves SLT performance.
#### OLMoASR: Open Models and Data for Training Robust Speech Recognition Models
 - **Authors:** Huong Ngo, Matt Deitke, Martijn Bartelds, Sarah Pratt, Josh Gardner, Matt Jordan, Ludwig Schmidt
 - **Subjects:** Subjects:
Sound (cs.SD); Computation and Language (cs.CL); Machine Learning (cs.LG); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2508.20869

 - **Pdf link:** https://arxiv.org/pdf/2508.20869

 - **Abstract**
 Improvements in training data scale and quality have led to significant advances, yet its influence in speech recognition remains underexplored. In this paper, we present a large-scale dataset, OLMoASR-Pool, and series of models, OLMoASR, to study and develop robust zero-shot speech recognition models. Beginning from OLMoASR-Pool, a collection of 3M hours of English audio and 17M transcripts, we design text heuristic filters to remove low-quality or mistranscribed data. Our curation pipeline produces a new dataset containing 1M hours of high-quality audio-transcript pairs, which we call OLMoASR-Mix. We use OLMoASR-Mix to train the OLMoASR-Mix suite of models, ranging from 39M (this http URL) to 1.5B (this http URL) parameters. Across all model scales, OLMoASR achieves comparable average performance to OpenAI's Whisper on short and long-form speech recognition benchmarks. Notably, this http URL attains a 12.8\% and 11.0\% word error rate (WER) that is on par with Whisper's largest English-only model this http URL's 12.4\% and 10.5\% WER for short and long-form recognition respectively (at equivalent parameter count). OLMoASR-Pool, OLMoASR models, and filtering, training and evaluation code will be made publicly available to further research on robust speech processing.
#### Learning Robust Spatial Representations from Binaural Audio through Feature Distillation
 - **Authors:** Holger Severin Bovbjerg (1), Jan Østergaard (1), Jesper Jensen (1, 2), Shinji Watanabe (3), Zheng-Hua Tan ((1) Aalborg University (2) Eriksholm Research Centre, (3) Carnegie Mellon University)
 - **Subjects:** Subjects:
Sound (cs.SD); Machine Learning (cs.LG); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2508.20914

 - **Pdf link:** https://arxiv.org/pdf/2508.20914

 - **Abstract**
 Recently, deep representation learning has shown strong performance in multiple audio tasks. However, its use for learning spatial representations from multichannel audio is underexplored. We investigate the use of a pretraining stage based on feature distillation to learn a robust spatial representation of binaural speech without the need for data labels. In this framework, spatial features are computed from clean binaural speech samples to form prediction labels. These clean features are then predicted from corresponding augmented speech using a neural network. After pretraining, we throw away the spatial feature predictor and use the learned encoder weights to initialize a DoA estimation model which we fine-tune for DoA estimation. Our experiments demonstrate that the pretrained models show improved performance in noisy and reverberant environments after fine-tuning for direction-of-arrival estimation, when compared to fully supervised models and classic signal processing methods.


by Zyzzyva0381 (Windy). 


2025-08-29
