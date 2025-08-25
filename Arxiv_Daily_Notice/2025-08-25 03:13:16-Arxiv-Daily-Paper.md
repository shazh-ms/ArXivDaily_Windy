# Showing new listings for Monday, 25 August 2025
Auto update papers at about 2:30am UTC (10:30am Beijing time) every weekday.


阅读 `Usage.md`了解如何使用此repo实现个性化的Arxiv论文推送

See `Usage.md` for instructions on how to personalize the repo. 


Keyword list: ['text-to-speech', 'text to speech', 'tts', 'LLM-based', 'speech', 'voice']


Excluded: []


### Today: 4papers 
#### Hybrid Pruning: In-Situ Compression of Self-Supervised Speech Models for Speaker Verification and Anti-Spoofing
 - **Authors:** Junyi Peng, Lin Zhang, Jiangyu Han, Oldřich Plchot, Johan Rohdin, Themos Stafylakis, Shuai Wang, Jan Černocký
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2508.16232

 - **Pdf link:** https://arxiv.org/pdf/2508.16232

 - **Abstract**
 Although large-scale self-supervised learning (SSL) models like WavLM have achieved state-of-the-art performance in speech processing, their significant size impedes deployment on resource-constrained devices. While structured pruning is a key technique for model compression, existing methods typically separate it from task-specific fine-tuning. This multi-stage approach struggles to create optimal architectures tailored for diverse downstream tasks. In this work, we introduce a unified framework that integrates structured pruning into the downstream fine-tuning process. Our framework unifies these steps, jointly optimizing for task performance and model sparsity in a single stage. This allows the model to learn a compressed architecture specifically for the end task, eliminating the need for complex multi-stage pipelines and knowledge distillation. Our pruned models achieve up to a 70\% parameter reduction with negligible performance degradation on large-scale datasets, achieving equal error rates of 0.7\%, 0.8\%, and 1.6\% on Vox1-O, -E, and -H, respectively. Furthermore, our approach demonstrates improved generalization in low-resource scenarios, reducing overfitting and achieving a state-of-the-art 3.7\% EER on ASVspoof5.
#### Mini-Omni-Reasoner: Token-Level Thinking-in-Speaking in Large Speech Models
 - **Authors:** Zhifei Xie, Ziyang Ma, Zihang Liu, Kaiyu Pang, Hongyu Li, Jialin Zhang, Yue Liao, Deheng Ye, Chunyan Miao, Shuicheng Yan
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Artificial Intelligence (cs.AI); Machine Learning (cs.LG); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2508.15827

 - **Pdf link:** https://arxiv.org/pdf/2508.15827

 - **Abstract**
 Reasoning is essential for effective communication and decision-making. While recent advances in LLMs and MLLMs have shown that incorporating explicit reasoning significantly improves understanding and generalization, reasoning in LSMs remains in a nascent stage. Early efforts attempt to transfer the "Thinking-before-Speaking" paradigm from textual models to speech. However, this sequential formulation introduces notable latency, as spoken responses are delayed until reasoning is fully completed, impairing real-time interaction and communication efficiency. To address this, we propose Mini-Omni-Reasoner, a framework that enables reasoning within speech via a novel "Thinking-in-Speaking" formulation. Rather than completing reasoning before producing any verbal output, Mini-Omni-Reasoner interleaves silent reasoning tokens with spoken response tokens at the token level. This design allows continuous speech generation while embedding structured internal reasoning, leveraging the model's high-frequency token processing capability. Although interleaved, local semantic alignment is enforced to ensure that each response token is informed by its preceding reasoning. To support this framework, we introduce Spoken-Math-Problems-3M, a large-scale dataset tailored for interleaved reasoning and response. The dataset ensures that verbal tokens consistently follow relevant reasoning content, enabling accurate and efficient learning of speech-coupled reasoning. Built on a hierarchical Thinker-Talker architecture, Mini-Omni-Reasoner delivers fluent yet logically grounded spoken responses, maintaining both naturalness and precision. On the Spoken-MQA benchmark, it achieves a +19.1% gain in arithmetic reasoning and +6.4% in contextual understanding, with shorter outputs and zero decoding latency.
#### Beyond Transcription: Mechanistic Interpretability in ASR
 - **Authors:** Neta Glazer, Yael Segal-Feldman, Hilit Segev, Aviv Shamsian, Asaf Buchnick, Gill Hetz, Ethan Fetaya, Joseph Keshet, Aviv Navon
 - **Subjects:** Subjects:
Sound (cs.SD); Computation and Language (cs.CL); Machine Learning (cs.LG); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2508.15882

 - **Pdf link:** https://arxiv.org/pdf/2508.15882

 - **Abstract**
 Interpretability methods have recently gained significant attention, particularly in the context of large language models, enabling insights into linguistic representations, error detection, and model behaviors such as hallucinations and repetitions. However, these techniques remain underexplored in automatic speech recognition (ASR), despite their potential to advance both the performance and interpretability of ASR systems. In this work, we adapt and systematically apply established interpretability methods such as logit lens, linear probing, and activation patching, to examine how acoustic and semantic information evolves across layers in ASR systems. Our experiments reveal previously unknown internal dynamics, including specific encoder-decoder interactions responsible for repetition hallucinations and semantic biases encoded deep within acoustic representations. These insights demonstrate the benefits of extending and applying interpretability techniques to speech recognition, opening promising directions for future research on improving model transparency and robustness.
#### QvTAD: Differential Relative Attribute Learning for Voice Timbre Attribute Detection
 - **Authors:** Zhiyu Wu, Jingyi Fang, Yufei Tang, Yuanzhong Zheng, Yaoxuan Wang, Haojun Fei
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2508.15931

 - **Pdf link:** https://arxiv.org/pdf/2508.15931

 - **Abstract**
 Voice Timbre Attribute Detection (vTAD) plays a pivotal role in fine-grained timbre modeling for speech generation tasks. However, it remains challenging due to the inherently subjective nature of timbre descriptors and the severe label imbalance in existing datasets. In this work, we present QvTAD, a novel pairwise comparison framework based on differential attention, designed to enhance the modeling of perceptual timbre attributes. To address the label imbalance in the VCTK-RVA dataset, we introduce a graph-based data augmentation strategy that constructs a Directed Acyclic Graph and employs Disjoint-Set Union techniques to automatically mine unobserved utterance pairs with valid attribute comparisons. Our framework leverages speaker embeddings from a pretrained FACodec, and incorporates a Relative Timbre Shift-Aware Differential Attention module. This module explicitly models attribute-specific contrasts between paired utterances via differential denoising and contrast amplification mechanisms. Experimental results on the VCTK-RVA benchmark demonstrate that QvTAD achieves substantial improvements across multiple timbre descriptors, with particularly notable gains in cross-speaker generalization scenarios.


by Zyzzyva0381 (Windy). 


2025-08-25
