# Showing new listings for Monday, 10 March 2025
Auto update papers at about 2:30am UTC (10:30am Beijing time) every weekday.


阅读 `Usage.md`了解如何使用此repo实现个性化的Arxiv论文推送

See `Usage.md` for instructions on how to personalize the repo. 


Keyword list: ['acoustic', 'reinforcement learning', 'reverb', 'meta', 'separate', 'reconstruction', 'noise', 'enhance', 'localization', 'speech']


Excluded: []


### Today: 6papers 
#### From Voice to Safety: Language AI Powered Pilot-ATC Communication Understanding for Airport Surface Movement Collision Risk Assessment
 - **Authors:** Yutian Pang, Andrew Paul Kendall, Alex Porcayo, Mariah Barsotti, Anahita Jain, John-Paul Clarke
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2503.04974

 - **Pdf link:** https://arxiv.org/pdf/2503.04974

 - **Abstract**
 This work integrates language AI-based voice communication understanding with collision risk assessment. The proposed framework consists of two major parts, (a) Automatic Speech Recognition (ASR); (b) surface collision risk modeling. ASR module generates information tables by processing voice communication transcripts, which serve as references for producing potential taxi plans and calculating the surface movement collision risk. For ASR, we collect and annotate our own Named Entity Recognition (NER) dataset based on open-sourced video recordings and safety investigation reports. Additionally, we refer to FAA Order JO 7110.65W and FAA Order JO 7340.2N to get the list of heuristic rules and phase contractions of communication between the pilot and the Air Traffic Controller (ATCo) used in daily aviation operations. Then, we propose the novel ATC Rule-Enhanced NER method, which integrates the heuristic rules into the model training and inference stages, resulting into hybrid rule-based NER model. We show the effectiveness of this hybrid approach by comparing different setups with different token-level embedding models. For the risk modeling, we adopt the node-link airport layout graph from NASA FACET and model the aircraft taxi speed at each link as a log-normal distribution and derive the total taxi time distribution. Then, we propose a spatiotemporal formulation of the risk probability of two aircraft moving across potential collision nodes during ground movement. We show the effectiveness of our approach by simulating two case studies, (a) the Henada airport runway collision accident happened in January 2024; (b) the KATL taxiway collision happened in September 2024. We show that, by understanding the pilot-ATC communication transcripts and analyzing surface movement patterns, the proposed model improves airport safety by providing risk assessment in time.
#### Musical Source Separation of Brazilian Percussion
 - **Authors:** Richa Namballa, Giovana Morais, Magdalena Fuentes
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Signal Processing (eess.SP)
 - **Arxiv link:** https://arxiv.org/abs/2503.04995

 - **Pdf link:** https://arxiv.org/pdf/2503.04995

 - **Abstract**
 Musical source separation (MSS) has recently seen a big breakthrough in separating instruments from a mixture in the context of Western music, but research on non-Western instruments is still limited due to a lack of data. In this demo, we use an existing dataset of Brazilian sama percussion to create artificial mixtures for training a U-Net model to separate the surdo drum, a traditional instrument in samba. Despite limited training data, the model effectively isolates the surdo, given the drum's repetitive patterns and its characteristic low-pitched timbre. These results suggest that MSS systems can be successfully harnessed to work in more culturally-inclusive scenarios without the need of collecting extensive amounts of data.
#### Direct Speech to Speech Translation: A Review
 - **Authors:** Mohammad Sarim, Saim Shakeel, Laeeba Javed, Jamaluddin, Mohammad Nadeem
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2503.04799

 - **Pdf link:** https://arxiv.org/pdf/2503.04799

 - **Abstract**
 Speech to speech translation (S2ST) is a transformative technology that bridges global communication gaps, enabling real time multilingual interactions in diplomacy, tourism, and international trade. Our review examines the evolution of S2ST, comparing traditional cascade models which rely on automatic speech recognition (ASR), machine translation (MT), and text to speech (TTS) components with newer end to end and direct speech translation (DST) models that bypass intermediate text representations. While cascade models offer modularity and optimized components, they suffer from error propagation, increased latency, and loss of prosody. In contrast, direct S2ST models retain speaker identity, reduce latency, and improve translation naturalness by preserving vocal characteristics and prosody. However, they remain limited by data sparsity, high computational costs, and generalization challenges for low-resource languages. The current work critically evaluates these approaches, their tradeoffs, and future directions for improving real time multilingual communication.
#### S2S-Arena, Evaluating Speech2Speech Protocols on Instruction Following with Paralinguistic Information
 - **Authors:** Feng Jiang, Zhiyu Lin, Fan Bu, Yuhao Du, Benyou Wang, Haizhou Li
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2503.05085

 - **Pdf link:** https://arxiv.org/pdf/2503.05085

 - **Abstract**
 The rapid development of large language models (LLMs) has brought significant attention to speech models, particularly recent progress in speech2speech protocols supporting speech input and output. However, the existing benchmarks adopt automatic text-based evaluators for evaluating the instruction following ability of these models lack consideration for paralinguistic information in both speech understanding and generation. To address these issues, we introduce S2S-Arena, a novel arena-style S2S benchmark that evaluates instruction-following capabilities with paralinguistic information in both speech-in and speech-out across real-world tasks. We design 154 samples that fused TTS and live recordings in four domains with 21 tasks and manually evaluate existing popular speech models in an arena-style manner. The experimental results show that: (1) in addition to the superior performance of GPT-4o, the speech model of cascaded ASR, LLM, and TTS outperforms the jointly trained model after text-speech alignment in speech2speech protocols; (2) considering paralinguistic information, the knowledgeability of the speech model mainly depends on the LLM backbone, and the multilingual support of that is limited by the speech module; (3) excellent speech models can already understand the paralinguistic information in speech input, but generating appropriate audio with paralinguistic information is still a challenge.
#### UniArray: Unified Spectral-Spatial Modeling for Array-Geometry-Agnostic Speech Separation
 - **Authors:** Weiguang Chen, Junjie Zhang, Jielong Yang, Eng Siong Chng, Xionghu Zhong
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2503.05110

 - **Pdf link:** https://arxiv.org/pdf/2503.05110

 - **Abstract**
 Array-geometry-agnostic speech separation (AGA-SS) aims to develop an effective separation method regardless of the microphone array geometry. Conventional methods rely on permutation-free operations, such as summation or attention mechanisms, to capture spatial information. However, these approaches often incur high computational costs or disrupt the effective use of spatial information during intra- and inter-channel interactions, leading to suboptimal performance. To address these issues, we propose UniArray, a novel approach that abandons the conventional interleaving manner. UniArray consists of three key components: a virtual microphone estimation (VME) module, a feature extraction and fusion module, and a hierarchical dual-path separator. The VME ensures robust performance across arrays with varying channel numbers. The feature extraction and fusion module leverages a spectral feature extraction module and a spatial dictionary learning (SDL) module to extract and fuse frequency-bin-level features, allowing the separator to focus on using the fused features. The hierarchical dual-path separator models feature dependencies along the time and frequency axes while maintaining computational efficiency. Experimental results show that UniArray outperforms state-of-the-art methods in SI-SDRi, WB-PESQ, NB-PESQ, and STOI across both seen and unseen array geometries.
#### DiVISe: Direct Visual-Input Speech Synthesis Preserving Speaker Characteristics And Intelligibility
 - **Authors:** Yifan Liu, Yu Fang, Zhouhan Lin
 - **Subjects:** Subjects:
Sound (cs.SD); Computer Vision and Pattern Recognition (cs.CV); Machine Learning (cs.LG); Multimedia (cs.MM); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2503.05223

 - **Pdf link:** https://arxiv.org/pdf/2503.05223

 - **Abstract**
 Video-to-speech (V2S) synthesis, the task of generating speech directly from silent video input, is inherently more challenging than other speech synthesis tasks due to the need to accurately reconstruct both speech content and speaker characteristics from visual cues alone. Recently, audio-visual pre-training has eliminated the need for additional acoustic hints in V2S, which previous methods often relied on to ensure training convergence. However, even with pre-training, existing methods continue to face challenges in achieving a balance between acoustic intelligibility and the preservation of speaker-specific characteristics. We analyzed this limitation and were motivated to introduce DiVISe (Direct Visual-Input Speech Synthesis), an end-to-end V2S model that predicts Mel-spectrograms directly from video frames alone. Despite not taking any acoustic hints, DiVISe effectively preserves speaker characteristics in the generated audio, and achieves superior performance on both objective and subjective metrics across the LRS2 and LRS3 datasets. Our results demonstrate that DiVISe not only outperforms existing V2S models in acoustic intelligibility but also scales more effectively with increased data and model parameters. Code and weights can be found at this https URL.


by Zyzzyva0381 (Windy). 


2025-03-11
