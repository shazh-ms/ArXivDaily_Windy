# Showing new listings for Thursday, 14 August 2025
Auto update papers at about 2:30am UTC (10:30am Beijing time) every weekday.


阅读 `Usage.md`了解如何使用此repo实现个性化的Arxiv论文推送

See `Usage.md` for instructions on how to personalize the repo. 


Keyword list: ['text-to-speech', 'text to speech', 'tts', 'LLM-based', 'speech', 'voice']


Excluded: []


### Today: 6papers 
#### Objective Soups: Multilingual Multi-Task Modeling for Speech Processing
 - **Authors:** A F M Saif, Lisha Chen, Xiaodong Cui, Songtao Lu, Brian Kingsbury, Tianyi Chen
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Machine Learning (cs.LG); Optimization and Control (math.OC); Machine Learning (stat.ML)
 - **Arxiv link:** https://arxiv.org/abs/2508.09228

 - **Pdf link:** https://arxiv.org/pdf/2508.09228

 - **Abstract**
 Training a single model for multilingual, multi-task speech processing (MSP) is severely hampered by conflicting objectives between tasks like speech recognition and translation. While multi-objective optimization (MOO) aims to align gradient updates, its effectiveness diminishes as the number of tasks grows, making it difficult to find a common descent direction. This raises a fundamental question: should highly conflicting objectives be optimized jointly or separated into a hierarchical structure? To address this question, this paper investigates three multi-objective MSP formulations, which we refer to as \textbf{objective soup recipes}. These formulations apply multi-objective optimization at different optimization levels to mitigate potential conflicts among all objectives. To ensure efficiency, we introduce a lightweight layer-selection mechanism that computes the conflict-avoiding gradient using only the most problematic layers, minimizing computational and memory overhead. Extensive experiments on CoVoST v2, LibriSpeech, and AISHELL-1 reveal that a bi-level recipe separating recognition and translation tasks consistently outperforms standard flat optimization. Our work demonstrates that hierarchical MOO is a more effective and scalable approach for building state-of-the-art MSP models. Our code has been released at this https URL.
#### Fake-Mamba: Real-Time Speech Deepfake Detection Using Bidirectional Mamba as Self-Attention's Alternative
 - **Authors:** Xi Xuan, Zimo Zhu, Wenxin Zhang, Yi-Cheng Lin, Tomi Kinnunen
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Artificial Intelligence (cs.AI); Computation and Language (cs.CL); Machine Learning (cs.LG); Systems and Control (eess.SY)
 - **Arxiv link:** https://arxiv.org/abs/2508.09294

 - **Pdf link:** https://arxiv.org/pdf/2508.09294

 - **Abstract**
 Advances in speech synthesis intensify security threats, motivating real-time deepfake detection research. We investigate whether bidirectional Mamba can serve as a competitive alternative to Self-Attention in detecting synthetic speech. Our solution, Fake-Mamba, integrates an XLSR front-end with bidirectional Mamba to capture both local and global artifacts. Our core innovation introduces three efficient encoders: TransBiMamba, ConBiMamba, and PN-BiMamba. Leveraging XLSR's rich linguistic representations, PN-BiMamba can effectively capture the subtle cues of synthetic speech. Evaluated on ASVspoof 21 LA, 21 DF, and In-The-Wild benchmarks, Fake-Mamba achieves 0.97%, 1.74%, and 5.85% EER, respectively, representing substantial relative gains over SOTA models XLSR-Conformer and XLSR-Mamba. The framework maintains real-time inference across utterance lengths, demonstrating strong generalization and practical viability. The code is available at this https URL.
#### ProMode: A Speech Prosody Model Conditioned on Acoustic and Textual Inputs
 - **Authors:** Eray Eren, Qingju Liu, Hyeongwoo Kim, Pablo Garrido, Abeer Alwan
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Computation and Language (cs.CL); Machine Learning (cs.LG); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2508.09389

 - **Pdf link:** https://arxiv.org/pdf/2508.09389

 - **Abstract**
 Prosody conveys rich emotional and semantic information of the speech signal as well as individual idiosyncrasies. We propose a stand-alone model that maps text-to-prosodic features such as F0 and energy and can be used in downstream tasks such as TTS. The ProMode encoder takes as input acoustic features and time-aligned textual content, both are partially masked, and obtains a fixed-length latent prosodic embedding. The decoder predicts acoustics in the masked region using both the encoded prosody input and unmasked textual content. Trained on the GigaSpeech dataset, we compare our method with state-of-the-art style encoders. For F0 and energy predictions, we show consistent improvements for our model at different levels of granularity. We also integrate these predicted prosodic features into a TTS system and conduct perceptual tests, which show higher prosody preference compared to the baselines, demonstrating the model's potential in tasks where prosody modeling is important.
#### $\text{M}^3\text{PDB}$: A Multimodal, Multi-Label, Multilingual Prompt Database for Speech Generation
 - **Authors:** Boyu Zhu, Cheng Gong, Muyang Wu, Ruihao Jing, Fan Liu, Xiaolei Zhang, Chi Zhang, Xuelong Li
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2508.09702

 - **Pdf link:** https://arxiv.org/pdf/2508.09702

 - **Abstract**
 Recent advancements in zero-shot speech generation have enabled models to synthesize speech that mimics speaker identity and speaking style from speech prompts. However, these models' effectiveness is significantly limited in real-world scenarios where high-quality speech prompts are absent, incomplete, or out of domain. This issue arises primarily from a significant quality mismatch between the speech data utilized for model training and the input prompt speech during inference. To address this, we introduce $\text{M}^3\text{PDB}$, the first large-scale, multi-modal, multi-label, and multilingual prompt database designed for robust prompt selection in speech generation. Our dataset construction leverages a novel multi-modal, multi-agent annotation framework, enabling precise and hierarchical labeling across diverse modalities. Furthermore, we propose a lightweight yet effective prompt selection strategy tailored for real-time, resource-constrained inference settings. Experimental results demonstrate that our proposed database and selection strategy effectively support various challenging speech generation scenarios. We hope our work can inspire the community to shift focus from improving performance on standard benchmarks to addressing more realistic and diverse application scenarios in speech generation. Code and dataset are available at: this https URL.
#### Improving the Speaker Anonymization Evaluation's Robustness to Target Speakers with Adversarial Learning
 - **Authors:** Carlos Franzreb, Arnab Das, Tim Polzehl, Sebastian Möller
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Machine Learning (cs.LG)
 - **Arxiv link:** https://arxiv.org/abs/2508.09803

 - **Pdf link:** https://arxiv.org/pdf/2508.09803

 - **Abstract**
 The current privacy evaluation for speaker anonymization often overestimates privacy when a same-gender target selection algorithm (TSA) is used, although this TSA leaks the speaker's gender and should hence be more vulnerable. We hypothesize that this occurs because the evaluation does not account for the fact that anonymized speech contains information from both the source and target speakers. To address this, we propose to add a target classifier that measures the influence of target speaker information in the evaluation, which can also be removed with adversarial learning. Experiments demonstrate that this approach is effective for multiple anonymizers, particularly when using a same-gender TSA, leading to a more reliable assessment.
#### UtterTune: LoRA-Based Target-Language Pronunciation Edit and Control in Multilingual Text-to-Speech
 - **Authors:** Shuhei Kato
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2508.09767

 - **Pdf link:** https://arxiv.org/pdf/2508.09767

 - **Abstract**
 We propose UtterTune, a lightweight adaptation method that fine-tunes a multilingual text-to-speech (TTS) system based on a large language model (LLM) architecture, designed to enhance the controllability of pronunciation in a target language while preserving performance in others. While LLM architectures have enabled TTS models to achieve remarkable naturalness, accurately modeling grapheme-to-phoneme (G2P) mapping and prosody remains challenging, especially when the model omits an explicit G2P module and directly processes minimally encoded text (e.g., byte-pair encoding). UtterTune leverages low-rank adaptation to enable the control of segmental pronunciation and pitch accent at the phoneme level for Japanese speech, the target language in this paper, while maintaining naturalness and speaker similarity in a zero-shot setting. Objective and subjective evaluations confirm its effectiveness.


by Zyzzyva0381 (Windy). 


2025-08-14
