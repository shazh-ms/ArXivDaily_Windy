# Showing new listings for Monday, 31 March 2025
Auto update papers at about 2:30am UTC (10:30am Beijing time) every weekday.


阅读 `Usage.md`了解如何使用此repo实现个性化的Arxiv论文推送

See `Usage.md` for instructions on how to personalize the repo. 


Keyword list: ['acoustic', 'reinforcement learning', 'reverb', 'meta', 'separate', 'reconstruction', 'noise', 'enhance', 'localization', 'speech']


Excluded: []


### Today: 6papers 
#### Lend a Hand: Semi Training-Free Cued Speech Recognition via MLLM-Driven Hand Modeling for Barrier-free Communication
 - **Authors:** Guanjie Huang, Danny Hin Kwok Tsang, Li Liu
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2503.21785

 - **Pdf link:** https://arxiv.org/pdf/2503.21785

 - **Abstract**
 Cued Speech (CS) is an innovative visual communication system that integrates lip-reading with hand coding, designed to enhance effective communication for individuals with hearing impairments. Automatic CS Recognition (ACSR) refers to the AI-driven process of automatically recognizing hand gestures and lip movements in CS, converting them into text. However, previous work often relies on complex fusion modules and training techniques. Additionally, due to the limited amount of data in CS, the extraction of hand features, as well as recognition modeling, has consistently been subpar, significantly limiting the effectiveness of ACSR. To address this issue, we have innovatively explored the capabilities of Multimodal large language models (MLLMs) in recognizing hand shapes and positions in CS. More precisely, we propose a new Semi Training-Free paradigm for ACSR, named STF-ACSR. This approach leverages zero-shot recognition of hand movements through the Chinese CS Prompt Module (CCSPM), which equipped a training-free keyframe filtering and customized prompt engineering based on MLLM. It then integrates the recognition results into the lip-reading model using a Minimalist Fusion Module (MFM), effectively achieving superior recognition results. Furthermore, specifically for this study, we have supplemented the existing dataset of 6 normal hearing CS cuers by recording additional data from 8 cuers with hearing impairments, resulting in a new mixed dataset. Extensive experiments have demonstrated that STF-ACSR significantly outperforms previous methods on both normal and hearing-impaired data. Implementation and checkpoints are available at this https URL.
#### Baseline Systems and Evaluation Metrics for Spatial Semantic Segmentation of Sound Scenes
 - **Authors:** Binh Thien Nguyen, Masahiro Yasuda, Daiki Takeuchi, Daisuke Niizumi, Yasunori Ohishi, Noboru Harada
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2503.22088

 - **Pdf link:** https://arxiv.org/pdf/2503.22088

 - **Abstract**
 Immersive communication has made significant advancements, especially with the release of the codec for Immersive Voice and Audio Services. Aiming at its further realization, the DCASE 2025 Challenge has recently introduced a task for spatial semantic segmentation of sound scenes (S5), which focuses on detecting and separating sound events in spatial sound scenes. In this paper, we explore methods for addressing the S5 task. Specifically, we present baseline S5 systems that combine audio tagging (AT) and label-queried source separation (LSS) models. We investigate two LSS approaches based on the ResUNet architecture: a) extracting a single source for each detected event and b) querying multiple sources concurrently. Since each separated source in S5 is identified by its sound event class label, we propose new class-aware metrics to evaluate both the sound sources and labels simultaneously. Experimental results on first-order ambisonics spatial audio demonstrate the effectiveness of the proposed systems and confirm the efficacy of the metrics.
#### M2D2: Exploring General-purpose Audio-Language Representations Beyond CLAP
 - **Authors:** Daisuke Niizumi, Daiki Takeuchi, Masahiro Yasuda, Binh Thien Nguyen, Yasunori Ohishi, Noboru Harada
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2503.22104

 - **Pdf link:** https://arxiv.org/pdf/2503.22104

 - **Abstract**
 Contrastive language-audio pre-training (CLAP) has addressed audio-language tasks such as audio-text retrieval by aligning audio and text in a common feature space. While CLAP addresses general audio-language tasks, its audio features do not generalize well in audio tasks. In contrast, self-supervised learning (SSL) models learn general-purpose audio features that perform well in diverse audio tasks. We pursue representation learning that can be widely used in audio applications and hypothesize that a method that learns both general audio features and CLAP features should achieve our goal, which we call a general-purpose audio-language representation. To implement our hypothesis, we propose M2D2, a second-generation masked modeling duo (M2D) that combines an SSL M2D and CLAP. M2D2 learns two types of features using two modalities (audio and text) in a two-stage training process. It also utilizes advanced LLM-based sentence embeddings in CLAP training for powerful semantic supervision. In the first stage, M2D2 learns generalizable audio features from M2D and CLAP, where CLAP aligns the features with the fine LLM-based semantic embeddings. In the second stage, it learns CLAP features using the audio features learned from the LLM-based embeddings. Through these pre-training stages, M2D2 should enhance generalizability and performance in its audio and CLAP features. Experiments validated that M2D2 achieves effective general-purpose audio-language representation, highlighted with SOTA fine-tuning mAP of 49.0 for AudioSet, SOTA performance in music tasks, and top-level performance in audio-language tasks.
#### Make Some Noise: Towards LLM audio reasoning and generation using sound tokens
 - **Authors:** Shivam Mehta, Nebojsa Jojic, Hannes Gamper
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Artificial Intelligence (cs.AI); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2503.22275

 - **Pdf link:** https://arxiv.org/pdf/2503.22275

 - **Abstract**
 Integrating audio comprehension and generation into large language models (LLMs) remains challenging due to the continuous nature of audio and the resulting high sampling rates. Here, we introduce a novel approach that combines Variational Quantization with Conditional Flow Matching to convert audio into ultra-low bitrate discrete tokens of 0.23kpbs, allowing for seamless integration with text tokens in LLMs. We fine-tuned a pretrained text-based LLM using Low-Rank Adaptation (LoRA) to assess its effectiveness in achieving true multimodal capabilities, i.e., audio comprehension and generation. Our tokenizer outperforms a traditional VQ-VAE across various datasets with diverse acoustic events. Despite the substantial loss of fine-grained details through audio tokenization, our multimodal LLM trained with discrete tokens achieves competitive results in audio comprehension with state-of-the-art methods, though audio generation is poor. Our results highlight the need for larger, more diverse datasets and improved evaluation metrics to advance multimodal LLM performance.
#### Enhancing Dance-to-Music Generation via Negative Conditioning Latent Diffusion Model
 - **Authors:** Changchang Sun, Gaowen Liu, Charles Fleming, Yan Yan
 - **Subjects:** Subjects:
Sound (cs.SD); Computer Vision and Pattern Recognition (cs.CV); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2503.22138

 - **Pdf link:** https://arxiv.org/pdf/2503.22138

 - **Abstract**
 Conditional diffusion models have gained increasing attention since their impressive results for cross-modal synthesis, where the strong alignment between conditioning input and generated output can be achieved by training a time-conditioned U-Net augmented with cross-attention mechanism. In this paper, we focus on the problem of generating music synchronized with rhythmic visual cues of the given dance video. Considering that bi-directional guidance is more beneficial for training a diffusion model, we propose to enhance the quality of generated music and its synchronization with dance videos by adopting both positive rhythmic information and negative ones (PN-Diffusion) as conditions, where a dual diffusion and reverse processes is devised. Specifically, to train a sequential multi-modal U-Net structure, PN-Diffusion consists of a noise prediction objective for positive conditioning and an additional noise prediction objective for negative conditioning. To accurately define and select both positive and negative conditioning, we ingeniously utilize temporal correlations in dance videos, capturing positive and negative rhythmic cues by playing them forward and backward, respectively. Through subjective and objective evaluations of input-output correspondence in terms of dance-music beat alignment and the quality of generated music, experimental results on the AIST++ and TikTok dance video datasets demonstrate that our model outperforms SOTA dance-to-music generation models.
#### DeepAudio-V1:Towards Multi-Modal Multi-Stage End-to-End Video to Speech and Audio Generation
 - **Authors:** Haomin Zhang, Chang Liu, Junjie Zheng, Zihao Chen, Chaofan Ding, Xinhan Di
 - **Subjects:** Subjects:
Computer Vision and Pattern Recognition (cs.CV); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2503.22265

 - **Pdf link:** https://arxiv.org/pdf/2503.22265

 - **Abstract**
 Currently, high-quality, synchronized audio is synthesized using various multi-modal joint learning frameworks, leveraging video and optional text inputs. In the video-to-audio benchmarks, video-to-audio quality, semantic alignment, and audio-visual synchronization are effectively achieved. However, in real-world scenarios, speech and audio often coexist in videos simultaneously, and the end-to-end generation of synchronous speech and audio given video and text conditions are not well studied. Therefore, we propose an end-to-end multi-modal generation framework that simultaneously produces speech and audio based on video and text conditions. Furthermore, the advantages of video-to-audio (V2A) models for generating speech from videos remain unclear. The proposed framework, DeepAudio, consists of a video-to-audio (V2A) module, a text-to-speech (TTS) module, and a dynamic mixture of modality fusion (MoF) module. In the evaluation, the proposed end-to-end framework achieves state-of-the-art performance on the video-audio benchmark, video-speech benchmark, and text-speech benchmark. In detail, our framework achieves comparable results in the comparison with state-of-the-art models for the video-audio and text-speech benchmarks, and surpassing state-of-the-art models in the video-speech benchmark, with WER 16.57% to 3.15% (+80.99%), SPK-SIM 78.30% to 89.38% (+14.15%), EMO-SIM 66.24% to 75.56% (+14.07%), MCD 8.59 to 7.98 (+7.10%), MCD SL 11.05 to 9.40 (+14.93%) across a variety of dubbing settings.


by Zyzzyva0381 (Windy). 


2025-03-31
