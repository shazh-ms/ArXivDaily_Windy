# Showing new listings for Tuesday, 25 February 2025
Auto update papers at about 2:30am UTC (10:30am Beijing time) every weekday.


阅读 `Usage.md`了解如何使用此repo实现个性化的Arxiv论文推送

See `Usage.md` for instructions on how to personalize the repo. 


Keyword list: ['acoustic', 'reinforcement learning', 'reverb', 'meta', 'separate', 'reconstruction', 'noise', 'enhance', 'localization', 'speech']


Excluded: []


### Today: 15papers 
#### Speech Enhancement Using Continuous Embeddings of Neural Audio Codec
 - **Authors:** Haoyang Li, Jia Qi Yip, Tianyu Fan, Eng Siong Chng
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Artificial Intelligence (cs.AI); Machine Learning (cs.LG); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2502.16240

 - **Pdf link:** https://arxiv.org/pdf/2502.16240

 - **Abstract**
 Recent advancements in Neural Audio Codec (NAC) models have inspired their use in various speech processing tasks, including speech enhancement (SE). In this work, we propose a novel, efficient SE approach by leveraging the pre-quantization output of a pretrained NAC encoder. Unlike prior NAC-based SE methods, which process discrete speech tokens using Language Models (LMs), we perform SE within the continuous embedding space of the pretrained NAC, which is highly compressed along the time dimension for efficient representation. Our lightweight SE model, optimized through an embedding-level loss, delivers results comparable to SE baselines trained on larger datasets, with a significantly lower real-time factor of 0.005. Additionally, our method achieves a low GMAC of 3.94, reducing complexity 18-fold compared to Sepformer in a simulated cloud-based audio transmission environment. This work highlights a new, efficient NAC-based SE solution, particularly suitable for cloud applications where NAC is used to compress audio before transmission. Copyright 20XX IEEE. Personal use of this material is permitted. Permission from IEEE must be obtained for all other uses, in any current or future media, including reprinting/republishing this material for advertising or promotional purposes, creating new collective works, for resale or redistribution to servers or lists, or reuse of any copyrighted component of this work in other works.
#### voc2vec: A Foundation Model for Non-Verbal Vocalization
 - **Authors:** Alkis Koudounas, Moreno La Quatra, Marco Sabato Siniscalchi, Elena Baralis
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2502.16298

 - **Pdf link:** https://arxiv.org/pdf/2502.16298

 - **Abstract**
 Speech foundation models have demonstrated exceptional capabilities in speech-related tasks. Nevertheless, these models often struggle with non-verbal audio data, such as vocalizations, baby crying, etc., which are critical for various real-world applications. Audio foundation models well handle non-speech data but also fail to capture the nuanced features of non-verbal human sounds. In this work, we aim to overcome the above shortcoming and propose a novel foundation model, termed voc2vec, specifically designed for non-verbal human data leveraging exclusively open-source non-verbal audio datasets. We employ a collection of 10 datasets covering around 125 hours of non-verbal audio. Experimental results prove that voc2vec is effective in non-verbal vocalization classification, and it outperforms conventional speech and audio foundation models. Moreover, voc2vec consistently outperforms strong baselines, namely OpenSmile and emotion2vec, on six different benchmark datasets. To the best of the authors' knowledge, voc2vec is the first universal representation model for vocalization tasks.
#### Balancing Speech Understanding and Generation Using Continual Pre-training for Codec-based Speech LLM
 - **Authors:** Jiatong Shi, Chunlei Zhang, Jinchuan Tian, Junrui Ni, Hao Zhang, Shinji Watanabe, Dong Yu
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2502.16897

 - **Pdf link:** https://arxiv.org/pdf/2502.16897

 - **Abstract**
 Recent efforts have extended textual LLMs to the speech domain. Yet, a key challenge remains, which is balancing speech understanding and generation while avoiding catastrophic forgetting when integrating acoustically rich codec-based representations into models originally trained on text. In this work, we propose a novel approach that leverages continual pre-training (CPT) on a pre-trained textual LLM to create a codec-based speech language model. This strategy mitigates the modality gap between text and speech, preserving the linguistic reasoning of the original model while enabling high-fidelity speech synthesis. We validate our approach with extensive experiments across multiple tasks, including automatic speech recognition, text-to-speech, speech-to-text translation, and speech-to-speech translation (S2ST), demonstrating that our model achieves superior TTS performance and, notably, the first end-to-end S2ST system based on neural codecs.
#### Slamming: Training a Speech Language Model on One GPU in a Day
 - **Authors:** Gallil Maimon, Avishai Elmakies, Yossi Adi
 - **Subjects:** Subjects:
Machine Learning (cs.LG); Artificial Intelligence (cs.AI); Computation and Language (cs.CL); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2502.15814

 - **Pdf link:** https://arxiv.org/pdf/2502.15814

 - **Abstract**
 We introduce Slam, a recipe for training high-quality Speech Language Models (SLMs) on a single academic GPU in 24 hours. We do so through empirical analysis of model initialisation and architecture, synthetic training data, preference optimisation with synthetic data and tweaking all other components. We empirically demonstrate that this training recipe also scales well with more compute getting results on par with leading SLMs in a fraction of the compute cost. We hope these insights will make SLM training and research more accessible. In the context of SLM scaling laws, our results far outperform predicted compute optimal performance, giving an optimistic view to SLM feasibility. See code, data, models, samples at - this https URL .
#### LACTOSE: Linear Array of Conditions, TOpologies with Separated Error-backpropagation -- The Differentiable "IF" Conditional for Differentiable Digital Signal Processing
 - **Authors:** Christopher Johann Clarke
 - **Subjects:** Subjects:
Machine Learning (cs.LG); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2502.15829

 - **Pdf link:** https://arxiv.org/pdf/2502.15829

 - **Abstract**
 There has been difficulty utilising conditional statements as part of the neural network graph (e.g. if input $> x$, pass input to network $N$). This is due to the inability to backpropagate through branching conditions. The Linear Array of Conditions, TOpologies with Separated Error-backpropagation (LACTOSE) Algorithm addresses this issue and allows the conditional use of available machine learning layers for supervised learning models. In this paper, the LACTOSE algorithm is applied to a simple use of DDSP, however, the main point is the development of the "if" conditional for DDSP use. The LACTOSE algorithm stores trained parameters for each user-specified numerical range and loads the parameters dynamically during prediction.
#### Deriving Representative Structure from Music Corpora
 - **Authors:** Ilana Shapiro, Ruanqianqian (Lisa)Huang, Zachary Novack, Cheng-i Wang, Hao-Wen Dong, Taylor Berg-Kirkpatrick, Shlomo Dubnov, Sorin Lerner
 - **Subjects:** Subjects:
Sound (cs.SD); Artificial Intelligence (cs.AI); Logic in Computer Science (cs.LO); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2502.15849

 - **Pdf link:** https://arxiv.org/pdf/2502.15849

 - **Abstract**
 Western music is an innately hierarchical system of interacting levels of structure, from fine-grained melody to high-level form. In order to analyze music compositions holistically and at multiple granularities, we propose a unified, hierarchical meta-representation of musical structure called the structural temporal graph (STG). For a single piece, the STG is a data structure that defines a hierarchy of progressively finer structural musical features and the temporal relationships between them. We use the STG to enable a novel approach for deriving a representative structural summary of a music corpus, which we formalize as a dually NP-hard combinatorial optimization problem extending the Generalized Median Graph problem. Our approach first applies simulated annealing to develop a measure of structural distance between two music pieces rooted in graph isomorphism. Our approach then combines the formal guarantees of SMT solvers with nested simulated annealing over structural distances to produce a structurally sound, representative centroid STG for an entire corpus of STGs from individual pieces. To evaluate our approach, we conduct experiments verifying that structural distance accurately differentiates between music pieces, and that derived centroids accurately structurally characterize their corpora.
#### Understanding Zero-shot Rare Word Recognition Improvements Through LLM Integration
 - **Authors:** Haoxuan Wang
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2502.16142

 - **Pdf link:** https://arxiv.org/pdf/2502.16142

 - **Abstract**
 In this study, we investigate the integration of a large language model (LLM) with an automatic speech recognition (ASR) system, specifically focusing on enhancing rare word recognition performance. Using a 190,000-hour dataset primarily sourced from YouTube, pre-processed with Whisper V3 pseudo-labeling, we demonstrate that the LLM-ASR architecture outperforms traditional Zipformer-Transducer models in the zero-shot rare word recognition task, after training on a large dataset. Our analysis reveals that the LLM contributes significantly to improvements in rare word error rate (R-WER), while the speech encoder primarily determines overall transcription performance (Orthographic Word Error Rate, O-WER, and Normalized Word Error Rate, N-WER). Through extensive ablation studies, we highlight the importance of adapter integration in aligning speech encoder outputs with the LLM's linguistic capabilities. Furthermore, we emphasize the critical role of high-quality labeled data in achieving optimal performance. These findings provide valuable insights into the synergy between LLM-based ASR architectures, paving the way for future advancements in large-scale LLM-based speech recognition systems.
#### Improving Speech Enhancement by Cross- and Sub-band Processing with State Space Model
 - **Authors:** Jizhen Li, Weiping Tu, Yuhong Yang, Xinmeng Xu, Yiqun Zhang, Yanzhen Ren
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2502.16207

 - **Pdf link:** https://arxiv.org/pdf/2502.16207

 - **Abstract**
 Recently, the state space model (SSM) represented by Mamba has shown remarkable performance in long-term sequence modeling tasks, including speech enhancement. However, due to substantial differences in sub-band features, applying the same SSM to all sub-bands limits its inference capability. Additionally, when processing each time frame of the time-frequency representation, the SSM may forget certain high-frequency information of low energy, making the restoration of structure in the high-frequency bands challenging. For this reason, we propose Cross- and Sub-band Mamba (CSMamba). To assist the SSM in handling different sub-band features flexibly, we propose a band split block that splits the full-band into four sub-bands with different widths based on their information similarity. We then allocate independent weights to each sub-band, thereby reducing the inference burden on the SSM. Furthermore, to mitigate the forgetting of low-energy information in the high-frequency bands by the SSM, we introduce a spectrum restoration block that enhances the representation of the cross-band features from multiple perspectives. Experimental results on the DNS Challenge 2021 dataset demonstrate that CSMamba outperforms several state-of-the-art (SOTA) speech enhancement methods in three objective evaluation metrics with fewer parameters.
#### Audio-FLAN: A Preliminary Release
 - **Authors:** Liumeng Xue, Ziya Zhou, Jiahao Pan, Zixuan Li, Shuai Fan, Yinghao Ma, Sitong Cheng, Dongchao Yang, Haohan Guo, Yujia Xiao, Xinsheng Wang, Zixuan Shen, Chuanbo Zhu, Xinshen Zhang, Tianchi Liu, Ruibin Yuan, Zeyue Tian, Haohe Liu, Emmanouil Benetos, Ge Zhang, Yike Guo, Wei Xue
 - **Subjects:** Subjects:
Sound (cs.SD); Artificial Intelligence (cs.AI); Computation and Language (cs.CL); Multimedia (cs.MM); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2502.16584

 - **Pdf link:** https://arxiv.org/pdf/2502.16584

 - **Abstract**
 Recent advancements in audio tokenization have significantly enhanced the integration of audio capabilities into large language models (LLMs). However, audio understanding and generation are often treated as distinct tasks, hindering the development of truly unified audio-language models. While instruction tuning has demonstrated remarkable success in improving generalization and zero-shot learning across text and vision, its application to audio remains largely unexplored. A major obstacle is the lack of comprehensive datasets that unify audio understanding and generation. To address this, we introduce Audio-FLAN, a large-scale instruction-tuning dataset covering 80 diverse tasks across speech, music, and sound domains, with over 100 million instances. Audio-FLAN lays the foundation for unified audio-language models that can seamlessly handle both understanding (e.g., transcription, comprehension) and generation (e.g., speech, music, sound) tasks across a wide range of audio domains in a zero-shot manner. The Audio-FLAN dataset is available on HuggingFace and GitHub and will be continuously updated.
#### Target Speaker Extraction through Comparing Noisy Positive and Negative Audio Enrollments
 - **Authors:** Shitong Xu, Yiyuan Yang, Niki Trigoni, Andrew Markham
 - **Subjects:** Subjects:
Sound (cs.SD); Artificial Intelligence (cs.AI); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2502.16611

 - **Pdf link:** https://arxiv.org/pdf/2502.16611

 - **Abstract**
 Target speaker extraction focuses on isolating a specific speaker's voice from an audio mixture containing multiple speakers. To provide information about the target speaker's identity, prior works have utilized clean audio examples as conditioning inputs. However, such clean audio examples are not always readily available (e.g. It is impractical to obtain a clean audio example of a stranger's voice at a cocktail party without stepping away from the noisy environment). Limited prior research has explored extracting the target speaker's characteristics from noisy audio examples, which may include overlapping speech from disturbing speakers. In this work, we focus on target speaker extraction when multiple speakers are present during the enrollment stage, through leveraging differences between audio segments where the target speakers are speaking (Positive Enrollments) and segments where they are not (Negative Enrollments). Experiments show the effectiveness of our model architecture and the dedicated pretraining method for the proposed task. Our method achieves state-of-the-art performance in the proposed application settings and demonstrates strong generalizability across challenging and realistic scenarios.
#### AAD-LLM: Neural Attention-Driven Auditory Scene Understanding
 - **Authors:** Xilin Jiang, Sukru Samet Dindar, Vishal Choudhari, Stephan Bickel, Ashesh Mehta, Guy M McKhann, Adeen Flinker, Daniel Friedman, Nima Mesgarani
 - **Subjects:** Subjects:
Sound (cs.SD); Artificial Intelligence (cs.AI); Computation and Language (cs.CL); Human-Computer Interaction (cs.HC); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2502.16794

 - **Pdf link:** https://arxiv.org/pdf/2502.16794

 - **Abstract**
 Auditory foundation models, including auditory large language models (LLMs), process all sound inputs equally, independent of listener perception. However, human auditory perception is inherently selective: listeners focus on specific speakers while ignoring others in complex auditory scenes. Existing models do not incorporate this selectivity, limiting their ability to generate perception-aligned responses. To address this, we introduce Intention-Informed Auditory Scene Understanding (II-ASU) and present Auditory Attention-Driven LLM (AAD-LLM), a prototype system that integrates brain signals to infer listener attention. AAD-LLM extends an auditory LLM by incorporating intracranial electroencephalography (iEEG) recordings to decode which speaker a listener is attending to and refine responses accordingly. The model first predicts the attended speaker from neural activity, then conditions response generation on this inferred attentional state. We evaluate AAD-LLM on speaker description, speech transcription and extraction, and question answering in multitalker scenarios, with both objective and subjective ratings showing improved alignment with listener intention. By taking a first step toward intention-aware auditory AI, this work explores a new paradigm where listener perception informs machine listening, paving the way for future listener-centered auditory systems. Demo and code available: this https URL.
#### ENACT-Heart -- ENsemble-based Assessment Using CNN and Transformer on Heart Sounds
 - **Authors:** Jiho Han, Adnan Shaout
 - **Subjects:** Subjects:
Sound (cs.SD); Artificial Intelligence (cs.AI); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2502.16914

 - **Pdf link:** https://arxiv.org/pdf/2502.16914

 - **Abstract**
 This study explores the application of Vision Transformer (ViT) principles in audio analysis, specifically focusing on heart sounds. This paper introduces ENACT-Heart - a novel ensemble approach that leverages the complementary strengths of Convolutional Neural Networks (CNN) and ViT through a Mixture of Experts (MoE) framework, achieving a remarkable classification accuracy of 97.52%. This outperforms the individual contributions of ViT (93.88%) and CNN (95.45%), demonstrating the potential for enhanced diagnostic accuracy in cardiovascular health monitoring. These results demonstrate the potential of ensemble methods in enhancing classification performance for cardiovascular health monitoring and diagnosis.
#### Baichuan-Audio: A Unified Framework for End-to-End Speech Interaction
 - **Authors:** Tianpeng Li, Jun Liu, Tao Zhang, Yuanbo Fang, Da Pan, Mingrui Wang, Zheng Liang, Zehuan Li, Mingan Lin, Guosheng Dong, Jianhua Xu, Haoze Sun, Zenan Zhou, Weipeng Chen
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2502.17239

 - **Pdf link:** https://arxiv.org/pdf/2502.17239

 - **Abstract**
 We introduce Baichuan-Audio, an end-to-end audio large language model that seamlessly integrates audio understanding and generation. It features a text-guided aligned speech generation mechanism, enabling real-time speech interaction with both comprehension and generation capabilities. Baichuan-Audio leverages a pre-trained ASR model, followed by multi-codebook discretization of speech at a frame rate of 12.5 Hz. This multi-codebook setup ensures that speech tokens retain both semantic and acoustic information. To further enhance modeling, an independent audio head is employed to process audio tokens, effectively capturing their unique characteristics. To mitigate the loss of intelligence during pre-training and preserve the original capabilities of the LLM, we propose a two-stage pre-training strategy that maintains language understanding while enhancing audio modeling. Following alignment, the model excels in real-time speech-based conversation and exhibits outstanding question-answering capabilities, demonstrating its versatility and efficiency. The proposed model demonstrates superior performance in real-time spoken dialogue and exhibits strong question-answering abilities. Our code, model and training data are available at this https URL
#### Improving the Inclusivity of Dutch Speech Recognition by Fine-tuning Whisper on the JASMIN-CGN Corpus
 - **Authors:** Golshid Shekoufandeh, Paul Boersma, Antal van den Bosch
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2502.17284

 - **Pdf link:** https://arxiv.org/pdf/2502.17284

 - **Abstract**
 We test and study the variation in speech recognition of fine-tuned versions of the Whisper model on child, elderly and non-native Dutch speech from the JASMIN-CGN corpus. Our primary goal is to evaluate how speakers' age and linguistic background influence Whisper's performance. Whisper achieves varying Word Error Rates (WER) when fine-tuned on subpopulations of specific ages and linguistic backgrounds. Fine-tuned performance is remarkably better than zero-shot performance, achieving a relative reduction in WER of 81% for native children, 72% for non-native children, 67% for non-native adults, and 65% for native elderly people. Our findings underscore the importance of training speech recognition models like Whisper on underrepresented subpopulations such as children, the elderly, and non-native speakers.
#### Low-Rank and Sparse Model Merging for Multi-Lingual Speech Recognition and Translation
 - **Authors:** Qiuming Zhao, Guangzhi Sun, Chao Zhang, Mingxing Xu, Thomas Fang Zheng
 - **Subjects:** Subjects:
Sound (cs.SD); Artificial Intelligence (cs.AI); Computation and Language (cs.CL); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2502.17380

 - **Pdf link:** https://arxiv.org/pdf/2502.17380

 - **Abstract**
 Language diversity presents a significant challenge in speech-to-text (S2T) tasks, such as automatic speech recognition and translation. Traditional multi-task training approaches aim to address this by jointly optimizing multiple speech recognition and translation tasks across various languages. While models like Whisper, built on these strategies, demonstrate strong performance, they still face issues of high computational cost, language interference, suboptimal training configurations, and limited extensibility. To overcome these challenges, we introduce LoRS-Merging (low-rank and sparse model merging), a novel technique designed to efficiently integrate models trained on different languages or tasks while preserving performance and reducing computational overhead. LoRS-Merging combines low-rank and sparse pruning to retain essential structures while eliminating redundant parameters, mitigating language and task interference, and enhancing extensibility. Experimental results across a range of languages demonstrate that LoRS-Merging significantly outperforms conventional multi-lingual multi-task training baselines. Our findings suggest that model merging, particularly LoRS-Merging, is a scalable and effective complement to traditional multi-lingual training strategies for S2T applications.


by Zyzzyva0381 (Windy). 


2025-02-26
