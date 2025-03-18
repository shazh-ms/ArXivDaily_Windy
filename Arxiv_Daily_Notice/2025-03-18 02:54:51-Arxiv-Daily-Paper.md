# Showing new listings for Monday, 17 March 2025
Auto update papers at about 2:30am UTC (10:30am Beijing time) every weekday.


阅读 `Usage.md`了解如何使用此repo实现个性化的Arxiv论文推送

See `Usage.md` for instructions on how to personalize the repo. 


Keyword list: ['acoustic', 'reinforcement learning', 'reverb', 'meta', 'separate', 'reconstruction', 'noise', 'enhance', 'localization', 'speech']


Excluded: []


### Today: 11papers 
#### EEG-Based Decoding of Sound Location: Comparing Free-Field to Headphone-Based Non-Individual HRTFs
 - **Authors:** Nils Marggraf-Turley, Martha Shiell, Niels Pontoppidan, Drew Cappotto, Lorenzo Picinali
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2503.10783

 - **Pdf link:** https://arxiv.org/pdf/2503.10783

 - **Abstract**
 Sound source localization relies on spatial cues such as interaural time differences (ITD), interaural level differences (ILD), and monaural spectral cues. Individually measured Head-Related Transfer Functions (HRTFs) facilitate precise spatial hearing but are impractical to measure, necessitating non-individual HRTFs, which may compromise localization accuracy and externalization. To further investigate this phenomenon, the neurophysiological differences between free-field and non-individual HRTF listening are explored by decoding sound locations from EEG-derived Event-Related Potentials (ERPs). Twenty-two participants localized stimuli under both conditions with EEG responses recorded and logistic regression classifiers trained to distinguish sound source locations. Lower cortical response amplitudes were observed for KEMAR compared to free-field, especially in front-central and occipital-parietal regions. ANOVA identified significant main effects of auralization condition (F(1, 21) = 34.56, p < 0.0001) and location (F(3, 63) = 18.17, p < 0.0001) on decoding accuracy (DA), which was higher in free-field and interaural-cue-dominated locations. DA negatively correlated with front-back confusion rates (r = -0.57, p < 0.01), linking neural DA to perceptual confusion. These findings demonstrate that headphone-based non-individual HRTFs elicit lower amplitude cortical responses to static, azimuthally-varying locations than free-field conditions. The correlation between EEG-based DA and front-back confusion underscores neurophysiological markers' potential for assessing spatial auditory discrimination.
#### MAVFlow: Preserving Paralinguistic Elements with Conditional Flow Matching for Zero-Shot AV2AV Multilingual Translation
 - **Authors:** Sungwoo Cho, Jeongsoo Choi, Sungnyun Kim, Se-Young Yun
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Computer Vision and Pattern Recognition (cs.CV); Machine Learning (cs.LG); Multimedia (cs.MM)
 - **Arxiv link:** https://arxiv.org/abs/2503.11026

 - **Pdf link:** https://arxiv.org/pdf/2503.11026

 - **Abstract**
 Despite recent advances in text-to-speech (TTS) models, audio-visual to audio-visual (AV2AV) translation still faces a critical challenge: maintaining speaker consistency between the original and translated vocal and facial features. To address this issue, we propose a conditional flow matching (CFM) zero-shot audio-visual renderer that utilizes strong dual guidance from both audio and visual modalities. By leveraging multi-modal guidance with CFM, our model robustly preserves speaker-specific characteristics and significantly enhances zero-shot AV2AV translation abilities. For the audio modality, we enhance the CFM process by integrating robust speaker embeddings with x-vectors, which serve to bolster speaker consistency. Additionally, we convey emotional nuances to the face rendering module. The guidance provided by both audio and visual cues remains independent of semantic or linguistic content, allowing our renderer to effectively handle zero-shot translation tasks for monolingual speakers in different languages. We empirically demonstrate that the inclusion of high-quality mel-spectrograms conditioned on facial information not only enhances the quality of the synthesized speech but also positively influences facial generation, leading to overall performance improvements.
#### Joint Training And Decoding for Multilingual End-to-End Simultaneous Speech Translation
 - **Authors:** Wuwei Huang, Renren Jin, Wen Zhang, Jian Luan, Bin Wang, Deyi Xiong
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2503.11080

 - **Pdf link:** https://arxiv.org/pdf/2503.11080

 - **Abstract**
 Recent studies on end-to-end speech translation(ST) have facilitated the exploration of multilingual end-to-end ST and end-to-end simultaneous ST. In this paper, we investigate end-to-end simultaneous speech translation in a one-to-many multilingual setting which is closer to applications in real scenarios. We explore a separate decoder architecture and a unified architecture for joint synchronous training in this scenario. To further explore knowledge transfer across languages, we propose an asynchronous training strategy on the proposed unified decoder architecture. A multi-way aligned multilingual end-to-end ST dataset was curated as a benchmark testbed to evaluate our methods. Experimental results demonstrate the effectiveness of our models on the collected dataset. Our codes and data are available at: this https URL.
#### Reinforcement Learning Outperforms Supervised Fine-Tuning: A Case Study on Audio Question Answering
 - **Authors:** Gang Li, Jizhong Liu, Heinrich Dinkel, Yadong Niu, Junbo Zhang, Jian Luan
 - **Subjects:** Subjects:
Sound (cs.SD); Artificial Intelligence (cs.AI); Computation and Language (cs.CL); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2503.11197

 - **Pdf link:** https://arxiv.org/pdf/2503.11197

 - **Abstract**
 Recently, reinforcement learning (RL) has been shown to greatly enhance the reasoning capabilities of large language models (LLMs), and RL-based approaches have been progressively applied to visual multimodal tasks. However, the audio modality has largely been overlooked in these developments. Thus, we conduct a series of RL explorations in audio understanding and reasoning, specifically focusing on the audio question answering (AQA) task. We leverage the group relative policy optimization (GRPO) algorithm to Qwen2-Audio-7B-Instruct, and our experiments demonstrated state-of-the-art performance on the MMAU Test-mini benchmark, achieving an accuracy rate of 64.5%. The main findings in this technical report are as follows: 1) The GRPO algorithm can be effectively applied to large audio language models (LALMs), even when the model has only 8.2B parameters; 2) With only 38k post-training samples, RL significantly outperforms supervised fine-tuning (SFT), indicating that RL-based approaches can be effective without large datasets; 3) The explicit reasoning process has not shown significant benefits for AQA tasks, and how to efficiently utilize deep thinking remains an open question for further research; 4) LALMs still lag far behind humans auditory-language reasoning, suggesting that the RL-based approaches warrant further exploration. Our project is available at this https URL and this https URL.
#### Comparative Study of Spike Encoding Methods for Environmental Sound Classification
 - **Authors:** Andres Larroza, Javier Naranjo-Alcazar, Vicent Ortiz Castelló, Pedro Zuccarello
 - **Subjects:** Subjects:
Sound (cs.SD); Emerging Technologies (cs.ET); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2503.11206

 - **Pdf link:** https://arxiv.org/pdf/2503.11206

 - **Abstract**
 Spiking Neural Networks (SNNs) offer a promising approach to reduce energy consumption and computational demands, making them particularly beneficial for embedded machine learning in edge applications. However, data from conventional digital sensors must first be converted into spike trains to be processed using neuromorphic computing technologies. The classification of environmental sounds presents unique challenges due to the high variability of frequencies, background noise, and overlapping acoustic events. Despite these challenges, most studies on spike-based audio encoding focus on speech processing, leaving non-speech environmental sounds underexplored. In this work, we conduct a comprehensive comparison of widely used spike encoding techniques, evaluating their effectiveness on the ESC-10 dataset. By understanding the impact of encoding choices on environmental sound processing, researchers and practitioners can select the most suitable approach for real-world applications such as smart surveillance, environmental monitoring, and industrial acoustic analysis. This study serves as a benchmark for spike encoding in environmental sound classification, providing a foundational reference for future research in neuromorphic audio processing.
#### Exploring the Potential of Large Multimodal Models as Effective Alternatives for Pronunciation Assessment
 - **Authors:** Ke Wang, Lei He, Kun Liu, Yan Deng, Wenning Wei, Sheng Zhao
 - **Subjects:** Subjects:
Sound (cs.SD); Computation and Language (cs.CL); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2503.11229

 - **Pdf link:** https://arxiv.org/pdf/2503.11229

 - **Abstract**
 Large Multimodal Models (LMMs) have demonstrated exceptional performance across a wide range of domains. This paper explores their potential in pronunciation assessment tasks, with a particular focus on evaluating the capabilities of the Generative Pre-trained Transformer (GPT) model, specifically GPT-4o. Our study investigates its ability to process speech and audio for pronunciation assessment across multiple levels of granularity and dimensions, with an emphasis on feedback generation and scoring. For our experiments, we use the publicly available Speechocean762 dataset. The evaluation focuses on two key aspects: multi-level scoring and the practicality of the generated feedback. Scoring results are compared against the manual scores provided in the Speechocean762 dataset, while feedback quality is assessed using Large Language Models (LLMs). The findings highlight the effectiveness of integrating LMMs with traditional methods for pronunciation assessment, offering insights into the model's strengths and identifying areas for further improvement.
#### A Data-Driven Exploration of Elevation Cues in HRTFs: An Explainable AI Perspective Across Multiple Datasets
 - **Authors:** Juan Antonio De Rus, Mario Montagud, Jesus Lopez-Ballester, Francesc J. Ferri, Maximo Cobos
 - **Subjects:** Subjects:
Signal Processing (eess.SP); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2503.11312

 - **Pdf link:** https://arxiv.org/pdf/2503.11312

 - **Abstract**
 Precise elevation perception in binaural audio remains a challenge, despite extensive research on head-related transfer functions (HRTFs) and spectral cues. While prior studies have advanced our understanding of sound localization cues, the interplay between spectral features and elevation perception is still not fully understood. This paper presents a comprehensive analysis of over 600 subjects from 11 diverse public HRTF datasets, employing a convolutional neural network (CNN) model combined with explainable artificial intelligence (XAI) techniques to investigate elevation cues. In addition to testing various HRTF pre-processing methods, we focus on both within-dataset and inter-dataset generalization and explainability, assessing the model's robustness across different HRTF variations stemming from subjects and measurement setups. By leveraging class activation mapping (CAM) saliency maps, we identify key frequency bands that may contribute to elevation perception, providing deeper insights into the spectral features that drive elevation-specific classification. This study offers new perspectives on HRTF modeling and elevation perception by analyzing diverse datasets and pre-processing techniques, expanding our understanding of these cues across a wide range of conditions.
#### MMS-LLaMA: Efficient LLM-based Audio-Visual Speech Recognition with Minimal Multimodal Speech Tokens
 - **Authors:** Jeong Hun Yeo, Hyeongseop Rha, Se Jin Park, Yong Man Ro
 - **Subjects:** Subjects:
Computer Vision and Pattern Recognition (cs.CV); Multimedia (cs.MM); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2503.11315

 - **Pdf link:** https://arxiv.org/pdf/2503.11315

 - **Abstract**
 Audio-Visual Speech Recognition (AVSR) achieves robust speech recognition in noisy environments by combining auditory and visual information. However, recent Large Language Model (LLM) based AVSR systems incur high computational costs due to the high temporal resolution of audio-visual speech processed by LLMs. In this work, we introduce an efficient multimodal speech LLM framework that minimizes token length while preserving essential linguistic content. Our approach employs an early av-fusion module for streamlined feature integration, an audio-visual speech Q-Former that dynamically allocates tokens based on input duration, and a refined query allocation strategy with a speech rate predictor to adjust token allocation according to speaking speed of each audio sample. Extensive experiments on the LRS3 dataset show that our method achieves state-of-the-art performance with a WER of 0.74% while using only 3.5 tokens per second. Moreover, our approach not only reduces token usage by 86% compared to the previous multimodal speech LLM framework, but also improves computational efficiency by reducing FLOPs by 35.7%.
#### Creating a Good Teacher for Knowledge Distillation in Acoustic Scene Classification
 - **Authors:** Tobias Morocutti, Florian Schmid, Khaled Koutini, Gerhard Widmer
 - **Subjects:** Subjects:
Sound (cs.SD); Machine Learning (cs.LG); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2503.11363

 - **Pdf link:** https://arxiv.org/pdf/2503.11363

 - **Abstract**
 Knowledge Distillation (KD) is a widespread technique for compressing the knowledge of large models into more compact and efficient models. KD has proved to be highly effective in building well-performing low-complexity Acoustic Scene Classification (ASC) systems and was used in all the top-ranked submissions to this task of the annual DCASE challenge in the past three years. There is extensive research available on establishing the KD process, designing efficient student models, and forming well-performing teacher ensembles. However, less research has been conducted on investigating which teacher model attributes are beneficial for low-complexity students. In this work, we try to close this gap by studying the effects on the student's performance when using different teacher network architectures, varying the teacher model size, training them with different device generalization methods, and applying different ensembling strategies. The results show that teacher model sizes, device generalization methods, the ensembling strategy and the ensemble size are key factors for a well-performing student network.
#### Exploring Performance-Complexity Trade-Offs in Sound Event Detection
 - **Authors:** Tobias Morocutti, Florian Schmid, Jonathan Greif, Francesco Foscarin, Gerhard Widmer
 - **Subjects:** Subjects:
Sound (cs.SD); Machine Learning (cs.LG); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2503.11373

 - **Pdf link:** https://arxiv.org/pdf/2503.11373

 - **Abstract**
 We target the problem of developing new low-complexity networks for the sound event detection task. Our goal is to meticulously analyze the performance-complexity trade-off, aiming to be competitive with the large state-of-the-art models, at a fraction of the computational requirements. We find that low-complexity convolutional models previously proposed for audio tagging can be effectively adapted for event detection (which requires frame-wise prediction) by adjusting convolutional strides, removing the global pooling, and, importantly, adding a sequence model before the (now frame-wise) classification heads. Systematic experiments reveal that the best choice for the sequence model type depends on which complexity metric is most important for the given application. We also investigate the impact of enhanced training strategies such as knowledge distillation. In the end, we show that combined with an optimized training strategy, we can reach event detection performance comparable to state-of-the-art transformers while requiring only around 5% of the parameters. We release all our pre-trained models and the code for reproducing this work to support future research in low-complexity sound event detection at this https URL.
#### Are Deep Speech Denoising Models Robust to Adversarial Noise?
 - **Authors:** Will Schwarzer, Philip S. Thomas, Andrea Fanelli, Xiaoyu Liu
 - **Subjects:** Subjects:
Sound (cs.SD); Machine Learning (cs.LG); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2503.11627

 - **Pdf link:** https://arxiv.org/pdf/2503.11627

 - **Abstract**
 Deep noise suppression (DNS) models enjoy widespread use throughout a variety of high-stakes speech applications. However, in this paper, we show that four recent DNS models can each be reduced to outputting unintelligible gibberish through the addition of imperceptible adversarial noise. Furthermore, our results show the near-term plausibility of targeted attacks, which could induce models to output arbitrary utterances, and over-the-air attacks. While the success of these attacks varies by model and setting, and attacks appear to be strongest when model-specific (i.e., white-box and non-transferable), our results highlight a pressing need for practical countermeasures in DNS systems.


by Zyzzyva0381 (Windy). 


2025-03-18
