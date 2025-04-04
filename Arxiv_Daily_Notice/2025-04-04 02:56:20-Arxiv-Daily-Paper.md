# Showing new listings for Friday, 4 April 2025
Auto update papers at about 2:30am UTC (10:30am Beijing time) every weekday.


阅读 `Usage.md`了解如何使用此repo实现个性化的Arxiv论文推送

See `Usage.md` for instructions on how to personalize the repo. 


Keyword list: ['acoustic', 'reinforcement learning', 'reverb', 'meta', 'separate', 'reconstruction', 'noise', 'enhance', 'localization', 'speech']


Excluded: []


### Today: 6papers 
#### Causal Self-supervised Pretrained Frontend with Predictive Code for Speech Separation
 - **Authors:** Wupeng Wang, Zexu Pan, Xinke Li, Shuai Wang, Haizhou Li
 - **Subjects:** Subjects:
Sound (cs.SD); Machine Learning (cs.LG); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2504.02302

 - **Pdf link:** https://arxiv.org/pdf/2504.02302

 - **Abstract**
 Speech separation (SS) seeks to disentangle a multi-talker speech mixture into single-talker speech streams. Although SS can be generally achieved using offline methods, such a processing paradigm is not suitable for real-time streaming applications. Causal separation models, which rely only on past and present information, offer a promising solution for real-time streaming. However, these models typically suffer from notable performance degradation due to the absence of future context. In this paper, we introduce a novel frontend that is designed to mitigate the mismatch between training and run-time inference by implicitly incorporating future information into causal models through predictive patterns. The pretrained frontend employs a transformer decoder network with a causal convolutional encoder as the backbone and is pretrained in a self-supervised manner with two innovative pretext tasks: autoregressive hybrid prediction and contextual knowledge distillation. These tasks enable the model to capture predictive patterns directly from mixtures in a self-supervised manner. The pretrained frontend subsequently serves as a feature extractor to generate high-quality predictive patterns. Comprehensive evaluations on synthetic and real-world datasets validated the effectiveness of the proposed pretrained frontend.
#### VoiceCraft-Dub: Automated Video Dubbing with Neural Codec Language Models
 - **Authors:** Kim Sung-Bin, Jeongsoo Choi, Puyuan Peng, Joon Son Chung, Tae-Hyun Oh, David Harwath
 - **Subjects:** Subjects:
Computer Vision and Pattern Recognition (cs.CV); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2504.02386

 - **Pdf link:** https://arxiv.org/pdf/2504.02386

 - **Abstract**
 We present VoiceCraft-Dub, a novel approach for automated video dubbing that synthesizes high-quality speech from text and facial cues. This task has broad applications in filmmaking, multimedia creation, and assisting voice-impaired individuals. Building on the success of Neural Codec Language Models (NCLMs) for speech synthesis, our method extends their capabilities by incorporating video features, ensuring that synthesized speech is time-synchronized and expressively aligned with facial movements while preserving natural prosody. To inject visual cues, we design adapters to align facial features with the NCLM token space and introduce audio-visual fusion layers to merge audio-visual information within the NCLM framework. Additionally, we curate CelebV-Dub, a new dataset of expressive, real-world videos specifically designed for automated video dubbing. Extensive experiments show that our model achieves high-quality, intelligible, and natural speech synthesis with accurate lip synchronization, outperforming existing methods in human perception and performing favorably in objective evaluations. We also adapt VoiceCraft-Dub for the video-to-speech task, demonstrating its versatility for various applications.
#### Scaling Analysis of Interleaved Speech-Text Language Models
 - **Authors:** Gallil Maimon, Michael Hassid, Amit Roth, Yossi Adi
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2504.02398

 - **Pdf link:** https://arxiv.org/pdf/2504.02398

 - **Abstract**
 Existing Speech Language Model (SLM) scaling analysis paints a bleak picture. They predict that SLMs require much more compute and data compared to text, leading some to question the feasibility of training high-quality SLMs. However, modern SLMs are often initialised from pre-trained TextLMs using speech-text interleaving to allow knowledge transfer. This raises the question - Do interleaved SLMs scale more efficiently than textless-SLMs? In this paper we answer a resounding, yes! We conduct scaling analysis of interleaved SLMs by training several dozen and analysing the scaling trends. We see that under this setup SLMs scale more efficiently with compute. Additionally, our results indicate that the scaling-dynamics are significantly different than textless-SLMs, suggesting one should allocate notably more of the compute budget for increasing model size over training tokens. We also study the role of synthetic data and TextLM model families in unlocking this potential. Results suggest, that our scaled up model achieves comparable performance with leading models on speech semantic metrics while using less compute and data than other approaches. We open source models, samples, and data - this https URL.
#### EvMic: Event-based Non-contact sound recovery from effective spatial-temporal modeling
 - **Authors:** Hao Yin, Shi Guo, Xu Jia, Xudong XU, Lu Zhang, Si Liu, Dong Wang, Huchuan Lu, Tianfan Xue
 - **Subjects:** Subjects:
Sound (cs.SD); Artificial Intelligence (cs.AI); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2504.02402

 - **Pdf link:** https://arxiv.org/pdf/2504.02402

 - **Abstract**
 When sound waves hit an object, they induce vibrations that produce high-frequency and subtle visual changes, which can be used for recovering the sound. Early studies always encounter trade-offs related to sampling rate, bandwidth, field of view, and the simplicity of the optical path. Recent advances in event camera hardware show good potential for its application in visual sound recovery, because of its superior ability in capturing high-frequency signals. However, existing event-based vibration recovery methods are still sub-optimal for sound recovery. In this work, we propose a novel pipeline for non-contact sound recovery, fully utilizing spatial-temporal information from the event stream. We first generate a large training set using a novel simulation pipeline. Then we designed a network that leverages the sparsity of events to capture spatial information and uses Mamba to model long-term temporal information. Lastly, we train a spatial aggregation block to aggregate information from different locations to further improve signal quality. To capture event signals caused by sound waves, we also designed an imaging system using a laser matrix to enhance the gradient and collected multiple data sequences for testing. Experimental results on synthetic and real-world data demonstrate the effectiveness of our method.
#### F5R-TTS: Improving Flow Matching based Text-to-Speech with Group Relative Policy Optimization
 - **Authors:** Xiaohui Sun, Ruitong Xiao, Jianye Mo, Bowen Wu, Qun Yu, Baoxun Wang
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2504.02407

 - **Pdf link:** https://arxiv.org/pdf/2504.02407

 - **Abstract**
 We present F5R-TTS, a novel text-to-speech (TTS) system that integrates Gradient Reward Policy Optimization (GRPO) into a flow-matching based architecture. By reformulating the deterministic outputs of flow-matching TTS into probabilistic Gaussian distributions, our approach enables seamless integration of reinforcement learning algorithms. During pretraining, we train a probabilistically reformulated flow-matching based model which is derived from F5-TTS with an open-source dataset. In the subsequent reinforcement learning (RL) phase, we employ a GRPO-driven enhancement stage that leverages dual reward metrics: word error rate (WER) computed via automatic speech recognition and speaker similarity (SIM) assessed by verification models. Experimental results on zero-shot voice cloning demonstrate that F5R-TTS achieves significant improvements in both speech intelligibility (relatively 29.5\% WER reduction) and speaker similarity (relatively 4.6\% SIM score increase) compared to conventional flow-matching based TTS systems. Audio samples are available at this https URL.
#### LinTO Audio and Textual Datasets to Train and Evaluate Automatic Speech Recognition in Tunisian Arabic Dialect
 - **Authors:** Hedi Naouara, Jean-Pierre Lorré, Jérôme Louradour
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2504.02604

 - **Pdf link:** https://arxiv.org/pdf/2504.02604

 - **Abstract**
 Developing Automatic Speech Recognition (ASR) systems for Tunisian Arabic Dialect is challenging due to the dialect's linguistic complexity and the scarcity of annotated speech datasets. To address these challenges, we propose the LinTO audio and textual datasets -- comprehensive resources that capture phonological and lexical features of Tunisian Arabic Dialect. These datasets include a variety of texts from numerous sources and real-world audio samples featuring diverse speakers and code-switching between Tunisian Arabic Dialect and English or French. By providing high-quality audio paired with precise transcriptions, the LinTO audio and textual datasets aim to provide qualitative material to build and benchmark ASR systems for the Tunisian Arabic Dialect. Keywords -- Tunisian Arabic Dialect, Speech-to-Text, Low-Resource Languages, Audio Data Augmentation


by Zyzzyva0381 (Windy). 


2025-04-04
