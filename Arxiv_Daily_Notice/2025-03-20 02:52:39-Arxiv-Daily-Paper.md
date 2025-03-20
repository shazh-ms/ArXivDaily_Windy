# Showing new listings for Thursday, 20 March 2025
Auto update papers at about 2:30am UTC (10:30am Beijing time) every weekday.


阅读 `Usage.md`了解如何使用此repo实现个性化的Arxiv论文推送

See `Usage.md` for instructions on how to personalize the repo. 


Keyword list: ['acoustic', 'reinforcement learning', 'reverb', 'meta', 'separate', 'reconstruction', 'noise', 'enhance', 'localization', 'speech']


Excluded: []


### Today: 5papers 
#### Analysis and Extension of Noisy-target Training for Unsupervised Target Signal Enhancement
 - **Authors:** Takuya Fujimura, Tomoki Toda
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2503.14854

 - **Pdf link:** https://arxiv.org/pdf/2503.14854

 - **Abstract**
 Deep neural network-based target signal enhancement (TSE) is usually trained in a supervised manner using clean target signals. However, collecting clean target signals is costly and such signals are not always available. Thus, it is desirable to develop an unsupervised method that does not rely on clean target signals. Among various studies on unsupervised TSE methods, Noisy-target Training (NyTT) has been established as a fundamental method. NyTT simply replaces clean target signals with noisy ones in the typical supervised training, and it has been experimentally shown to achieve TSE. Despite its effectiveness and simplicity, its mechanism and detailed behavior are still unclear. In this paper, to advance NyTT and, thus, unsupervised methods as a whole, we analyze NyTT from various perspectives. We experimentally demonstrate the mechanism of NyTT, the desirable conditions, and the effectiveness of utilizing noisy signals in situations where a small number of clean target signals are available. Furthermore, we propose an improved version of NyTT based on its properties and explore its capabilities in the dereverberation and declipping tasks, beyond the denoising task.
#### Solla: Towards a Speech-Oriented LLM That Hears Acoustic Context
 - **Authors:** Junyi Ao, Dekun Chen, Xiaohai Tian, Wenjie Feng, Jun Zhang, Lu Lu, Yuxuan Wang, Haizhou Li, Zhizheng Wu
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Computation and Language (cs.CL); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2503.15338

 - **Pdf link:** https://arxiv.org/pdf/2503.15338

 - **Abstract**
 Large Language Models (LLMs) have recently shown remarkable ability to process not only text but also multimodal inputs such as speech and audio. However, most existing models primarily focus on analyzing input signals using text instructions, overlooking scenarios in which speech instructions and audio are mixed and serve as inputs to the model. To address these challenges, we introduce Solla, a novel framework designed to understand speech-based questions and hear the acoustic context concurrently. Solla incorporates an audio tagging module to effectively identify and represent audio events, as well as an ASR-assisted prediction method to improve comprehension of spoken content. To rigorously evaluate Solla and other publicly available models, we propose a new benchmark dataset called SA-Eval, which includes three tasks: audio event classification, audio captioning, and audio question answering. SA-Eval has diverse speech instruction with various speaking styles, encompassing two difficulty levels, easy and hard, to capture the range of real-world acoustic conditions. Experimental results show that Solla performs on par with or outperforms baseline models on both the easy and hard test sets, underscoring its effectiveness in jointly understanding speech and audio.
#### PANDORA: Diffusion Policy Learning for Dexterous Robotic Piano Playing
 - **Authors:** Yanjia Huang, Renjie Li, Zhengzhong Tu
 - **Subjects:** Subjects:
Machine Learning (cs.LG); Robotics (cs.RO); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2503.14545

 - **Pdf link:** https://arxiv.org/pdf/2503.14545

 - **Abstract**
 We present PANDORA, a novel diffusion-based policy learning framework designed specifically for dexterous robotic piano performance. Our approach employs a conditional U-Net architecture enhanced with FiLM-based global conditioning, which iteratively denoises noisy action sequences into smooth, high-dimensional trajectories. To achieve precise key execution coupled with expressive musical performance, we design a composite reward function that integrates task-specific accuracy, audio fidelity, and high-level semantic feedback from a large language model (LLM) oracle. The LLM oracle assesses musical expressiveness and stylistic nuances, enabling dynamic, hand-specific reward adjustments. Further augmented by a residual inverse-kinematics refinement policy, PANDORA achieves state-of-the-art performance in the ROBOPIANIST environment, significantly outperforming baselines in both precision and expressiveness. Ablation studies validate the critical contributions of diffusion-based denoising and LLM-driven semantic feedback in enhancing robotic musicianship. Videos available at: this https URL
#### Shushing! Let's Imagine an Authentic Speech from the Silent Video
 - **Authors:** Jiaxin Ye, Hongming Shan
 - **Subjects:** Subjects:
Computer Vision and Pattern Recognition (cs.CV); Artificial Intelligence (cs.AI); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2503.14928

 - **Pdf link:** https://arxiv.org/pdf/2503.14928

 - **Abstract**
 Vision-guided speech generation aims to produce authentic speech from facial appearance or lip motions without relying on auditory signals, offering significant potential for applications such as dubbing in filmmaking and assisting individuals with aphonia. Despite recent progress, existing methods struggle to achieve unified cross-modal alignment across semantics, timbre, and emotional prosody from visual cues, prompting us to propose Consistent Video-to-Speech (CV2S) as an extended task to enhance cross-modal consistency. To tackle emerging challenges, we introduce ImaginTalk, a novel cross-modal diffusion framework that generates faithful speech using only visual input, operating within a discrete space. Specifically, we propose a discrete lip aligner that predicts discrete speech tokens from lip videos to capture semantic information, while an error detector identifies misaligned tokens, which are subsequently refined through masked language modeling with BERT. To further enhance the expressiveness of the generated speech, we develop a style diffusion transformer equipped with a face-style adapter that adaptively customizes identity and prosody dynamics across both the channel and temporal dimensions while ensuring synchronization with lip-aware semantic features. Extensive experiments demonstrate that ImaginTalk can generate high-fidelity speech with more accurate semantic details and greater expressiveness in timbre and emotion compared to state-of-the-art baselines. Demos are shown at our project page: this https URL.
#### InsectSet459: an open dataset of insect sounds for bioacoustic machine learning
 - **Authors:** Marius FaiÃŸ, Burooj Ghani, Dan Stowell
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2503.15074

 - **Pdf link:** https://arxiv.org/pdf/2503.15074

 - **Abstract**
 Automatic recognition of insect sound could help us understand changing biodiversity trends around the world -- but insect sounds are challenging to recognize even for deep learning. We present a new dataset comprised of 26399 audio files, from 459 species of Orthoptera and Cicadidae. It is the first large-scale dataset of insect sound that is easily applicable for developing novel deep-learning methods. Its recordings were made with a variety of audio recorders using varying sample rates to capture the extremely broad range of frequencies that insects produce. We benchmark performance with two state-of-the-art deep learning classifiers, demonstrating good performance but also significant room for improvement in acoustic insect classification. This dataset can serve as a realistic test case for implementing insect monitoring workflows, and as a challenging basis for the development of audio representation methods that can handle highly variable frequencies and/or sample rates.


by Zyzzyva0381 (Windy). 


2025-03-20
