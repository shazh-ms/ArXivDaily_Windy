# Showing new listings for Friday, 15 August 2025
Auto update papers at about 2:30am UTC (10:30am Beijing time) every weekday.


阅读 `Usage.md`了解如何使用此repo实现个性化的Arxiv论文推送

See `Usage.md` for instructions on how to personalize the repo. 


Keyword list: ['text-to-speech', 'text to speech', 'tts', 'LLM-based', 'speech', 'voice']


Excluded: []


### Today: 8papers 
#### Layer-Wise Analysis of Self-Supervised Representations for Age and Gender Classification in Children's Speech
 - **Authors:** Abhijit Sinha, Harishankar Kumar, Mohit Joshi, Hemant Kumar Kathania, Shrikanth Narayanan, Sudarsana Reddy Kadiri
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Artificial Intelligence (cs.AI); Human-Computer Interaction (cs.HC); Machine Learning (cs.LG); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2508.10332

 - **Pdf link:** https://arxiv.org/pdf/2508.10332

 - **Abstract**
 Children's speech presents challenges for age and gender classification due to high variability in pitch, articulation, and developmental traits. While self-supervised learning (SSL) models perform well on adult speech tasks, their ability to encode speaker traits in children remains underexplored. This paper presents a detailed layer-wise analysis of four Wav2Vec2 variants using the PFSTAR and CMU Kids datasets. Results show that early layers (1-7) capture speaker-specific cues more effectively than deeper layers, which increasingly focus on linguistic information. Applying PCA further improves classification, reducing redundancy and highlighting the most informative components. The Wav2Vec2-large-lv60 model achieves 97.14% (age) and 98.20% (gender) on CMU Kids; base-100h and large-lv60 models reach 86.05% and 95.00% on PFSTAR. These results reveal how speaker traits are structured across SSL model depth and support more targeted, adaptive strategies for child-aware speech interfaces.
#### Towards Frame-level Quality Predictions of Synthetic Speech
 - **Authors:** Michael Kuhlmann, Fritz Seebauer, Petra Wagner, Reinhold Haeb-Umbach
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2508.10374

 - **Pdf link:** https://arxiv.org/pdf/2508.10374

 - **Abstract**
 While automatic subjective speech quality assessment has witnessed much progress, an open question is whether an automatic quality assessment at frame resolution is possible. This would be highly desirable, as it adds explainability to the assessment of speech synthesis systems. Here, we take first steps towards this goal by identifying issues of existing quality predictors that prevent sensible frame-level prediction. Further, we define criteria that a frame-level predictor should fulfill. We also suggest a chunk-based processing that avoids the impact of a localized distortion on the score of neighboring frames. Finally, we measure in experiments with localized artificial distortions the localization performance of a set of frame-level quality predictors and show that they can outperform detection performance of human annotations obtained from a crowd-sourced perception experiment.
#### Exploring Cross-Utterance Speech Contexts for Conformer-Transducer Speech Recognition Systems
 - **Authors:** Mingyu Cui, Mengzhe Geng, Jiajun Deng, Chengxi Deng, Jiawen Kang, Shujie Hu, Guinan Li, Tianzi Wang, Zhaoqing Li, Xie Chen, Xunying Liu
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2508.10456

 - **Pdf link:** https://arxiv.org/pdf/2508.10456

 - **Abstract**
 This paper investigates four types of cross-utterance speech contexts modeling approaches for streaming and non-streaming Conformer-Transformer (C-T) ASR systems: i) input audio feature concatenation; ii) cross-utterance Encoder embedding concatenation; iii) cross-utterance Encoder embedding pooling projection; or iv) a novel chunk-based approach applied to C-T models for the first time. An efficient batch-training scheme is proposed for contextual C-Ts that uses spliced speech utterances within each minibatch to minimize the synchronization overhead while preserving the sequential order of cross-utterance speech contexts. Experiments are conducted on four benchmark speech datasets across three languages: the English GigaSpeech and Mandarin Wenetspeech corpora used in contextual C-T models pre-training; and the English DementiaBank Pitt and Cantonese JCCOCC MoCA elderly speech datasets used in domain fine-tuning. The best performing contextual C-T systems consistently outperform their respective baselines using no cross-utterance speech contexts in pre-training and fine-tuning stages with statistically significant average word error rate (WER) or character error rate (CER) reductions up to 0.9%, 1.1%, 0.51%, and 0.98% absolute (6.0%, 5.4%, 2.0%, and 3.4% relative) on the four tasks respectively. Their performance competitiveness against Wav2vec2.0-Conformer, XLSR-128, and Whisper models highlights the potential benefit of incorporating cross-utterance speech contexts into current speech foundation models.
#### Whisper Smarter, not Harder: Adversarial Attack on Partial Suppression
 - **Authors:** Zheng Jie Wong, Bingquan Shen
 - **Subjects:** Subjects:
Sound (cs.SD); Cryptography and Security (cs.CR); Machine Learning (cs.LG); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2508.09994

 - **Pdf link:** https://arxiv.org/pdf/2508.09994

 - **Abstract**
 Currently, Automatic Speech Recognition (ASR) models are deployed in an extensive range of applications. However, recent studies have demonstrated the possibility of adversarial attack on these models which could potentially suppress or disrupt model output. We investigate and verify the robustness of these attacks and explore if it is possible to increase their imperceptibility. We additionally find that by relaxing the optimisation objective from complete suppression to partial suppression, we can further decrease the imperceptibility of the attack. We also explore possible defences against these attacks and show a low-pass filter defence could potentially serve as an effective defence.
#### Beyond Hard Sharing: Efficient Multi-Task Speech-to-Text Modeling with Supervised Mixture of Experts
 - **Authors:** Hojun Jin, Eunsoo Hong, Ziwon Hyung, Sungjun Lim, Seungjin Lee, Keunseok Cho
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Artificial Intelligence (cs.AI); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2508.10009

 - **Pdf link:** https://arxiv.org/pdf/2508.10009

 - **Abstract**
 Hard-parameter sharing is a common strategy to train a single model jointly across diverse tasks. However, this often leads to task interference, impeding overall model performance. To address the issue, we propose a simple yet effective Supervised Mixture of Experts (S-MoE). Unlike traditional Mixture of Experts models, S-MoE eliminates the need for training gating functions by utilizing special guiding tokens to route each task to its designated expert. By assigning each task to a separate feedforward network, S-MoE overcomes the limitations of hard-parameter sharing. We further apply S-MoE to a speech-to-text model, enabling the model to process mixed-bandwidth input while jointly performing automatic speech recognition (ASR) and speech translation (ST). Experimental results demonstrate the effectiveness of the proposed S-MoE, achieving a 6.35% relative improvement in Word Error Rate (WER) when applied to both the encoder and decoder.
#### MCP2OSC: Parametric Control by Natural Language
 - **Authors:** Yuan-Yi Fan
 - **Subjects:** Subjects:
Human-Computer Interaction (cs.HC); Artificial Intelligence (cs.AI); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2508.10414

 - **Pdf link:** https://arxiv.org/pdf/2508.10414

 - **Abstract**
 Text prompts enable intuitive content creation but may fall short in achieving high precision for intricate tasks; knob or slider controls offer precise adjustments at the cost of increased complexity. To address the gap between knobs and prompts, a new MCP (Model Context Protocol) server and a unique set of prompt design criteria are presented to enable exploring parametric OSC (OpenSoundControl) control by natural language prompts. Demonstrated by 14 practical QA examples with best practices and the generalized prompt templates, this study finds Claude integrated with the MCP2OSC server effective in generating OSC messages by natural language, interpreting, searching, and visualizing OSC messages, validating and debugging OSC messages, and managing OSC address patterns. MCP2OSC enhances human-machine collaboration by leveraging LLM (Large Language Model) to handle intricate OSC development tasks, and by empowering human creativity with an intuitive language interface featuring flexible precision controls: a prompt-based OSC tool. This study provides a novel perspective on the creative MCP application at the network protocol level by utilizing LLM's strength in directly processing and generating human-readable OSC messages. The results suggest its potential for a LLM-based universal control mechanism for multimedia devices.
#### Alternating Approach-Putt Models for Multi-Stage Speech Enhancement
 - **Authors:** Iksoon Jeong, Kyung-Joong Kim, Kang-Hun Ahn
 - **Subjects:** Subjects:
Sound (cs.SD); Artificial Intelligence (cs.AI); Machine Learning (cs.LG); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2508.10436

 - **Pdf link:** https://arxiv.org/pdf/2508.10436

 - **Abstract**
 Speech enhancement using artificial neural networks aims to remove noise from noisy speech signals while preserving the speech content. However, speech enhancement networks often introduce distortions to the speech signal, referred to as artifacts, which can degrade audio quality. In this work, we propose a post-processing neural network designed to mitigate artifacts introduced by speech enhancement models. Inspired by the analogy of making a `Putt' after an `Approach' in golf, we name our model PuttNet. We demonstrate that alternating between a speech enhancement model and the proposed Putt model leads to improved speech quality, as measured by perceptual quality scores (PESQ), objective intelligibility (STOI), and background noise intrusiveness (CBAK) scores. Furthermore, we illustrate with graphical analysis why this alternating Approach outperforms repeated application of either model alone.
#### Advances in Speech Separation: Techniques, Challenges, and Future Trends
 - **Authors:** Kai Li, Guo Chen, Wendi Sang, Yi Luo, Zhuo Chen, Shuai Wang, Shulin He, Zhong-Qiu Wang, Andong Li, Zhiyong Wu, Xiaolin Hu
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2508.10830

 - **Pdf link:** https://arxiv.org/pdf/2508.10830

 - **Abstract**
 The field of speech separation, addressing the "cocktail party problem", has seen revolutionary advances with DNNs. Speech separation enhances clarity in complex acoustic environments and serves as crucial pre-processing for speech recognition and speaker recognition. However, current literature focuses narrowly on specific architectures or isolated approaches, creating fragmented understanding. This survey addresses this gap by providing systematic examination of DNN-based speech separation techniques. Our work differentiates itself through: (I) Comprehensive perspective: We systematically investigate learning paradigms, separation scenarios with known/unknown speakers, comparative analysis of supervised/self-supervised/unsupervised frameworks, and architectural components from encoders to estimation strategies. (II) Timeliness: Coverage of cutting-edge developments ensures access to current innovations and benchmarks. (III) Unique insights: Beyond summarization, we evaluate technological trajectories, identify emerging patterns, and highlight promising directions including domain-robust frameworks, efficient architectures, multimodal integration, and novel self-supervised paradigms. (IV) Fair evaluation: We provide quantitative evaluations on standard datasets, revealing true capabilities and limitations of different methods. This comprehensive survey serves as an accessible reference for experienced researchers and newcomers navigating speech separation's complex landscape.


by Zyzzyva0381 (Windy). 


2025-08-15
