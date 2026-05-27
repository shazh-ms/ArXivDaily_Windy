# Showing new listings for Wednesday, 27 May 2026
Auto update papers at about 2:30am UTC (10:30am Beijing time) every weekday.


阅读 `Usage.md`了解如何使用此repo实现个性化的Arxiv论文推送

See `Usage.md` for instructions on how to personalize the repo. 


Keyword list: ['text-to-speech', 'text to speech', 'tts', 'LLM-based', 'speech', 'voice']


Excluded: []


### Today: 3papers 
#### CFMDCTCodec: A Low-Bitrate Neural Speech Codec with Noise-Prior-aware Conditional Flow Matching for MDCT-Spectral Enhancement
 - **Authors:** Xiao-Hang Jiang, Yang Ai, Hui-Peng Du, Zhen-Hua Ling, Ji Wu
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2605.26812

 - **Pdf link:** https://arxiv.org/pdf/2605.26812

 - **Abstract**
 High-quality speech coding at low bitrates is crucial for bandwidth-constrained applications, yet remains challenging due to the severe loss of quality-critical information in highly compressed representations. To overcome this challenge, we propose CFMDCTCodec, a low-bitrate neural speech codec that operates entirely in the modified discrete cosine transform (MDCT) domain. CFMDCTCodec integrates a lightweight encoder-quantizer-decoder-style MDCT-spectral codec with a noise-prior-aware, conditional-flow-matching (CFM)-based MDCT-spectral enhancer. Within this framework, the codec serves as a base module that compactly discretizes the MDCT spectrum extracted from speech and produces an initial coarse reconstruction, while the enhancer further restores fine-grained spectral details. The enhancer improves the decoded MDCT spectrum by integrating a conditional MDCT velocity-field filter with an ordinary differential equation (ODE) solver, under the guidance of an MDCT-derived magnitude-adaptive noise prior, aiming to emphasize perceptually significant high-energy regions while stabilizing low-energy and silent regions. Finally, the enhanced MDCT spectrum is reconstructed into the decoded speech using the inverse MDCT. When optimizing CFMDCTCodec, we adopt a unified non-adversarial training strategy that jointly combines reconstruction, quantization and CFM objectives. Both objective and subjective evaluations show that CFMDCTCodec outperforms competitive baselines in low-bitrate regimes, e.g., 0.65 kbps, while approaching the perceptual quality of large-scale codecs with significantly fewer parameters and computations.
#### Why Can't They Remember? Uncovering Representation and Retrieval Bottlenecks in Multi-Turn Acoustic Memory
 - **Authors:** Yang Xiao, Siyi Wang, Han Yin, Hong Jia, Vidhyasaharan Sethu, Eun-Jung Holden, Ting Dang
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2605.27039

 - **Pdf link:** https://arxiv.org/pdf/2605.27039

 - **Abstract**
 Large audio language models (LALMs) process both speech and environmental acoustic cues, yet struggle to retain non-speech information across multi-turn interactions. The performance gap between semantic (speech) and acoustic (non-speech) understanding remains poorly understood, and the underlying mechanisms of representation and retrieval are still unclear. This work introduces EnvMem, a controlled multi-turn benchmark designed to study this gap and identify the root causes of failures at the representation (i.e., latent embeddings) and retrieval levels (i.e., attention allocation). We further conduct post-hoc interventions to probe representational structure and attention dynamics. Our results reveal representational trajectory drift as the key failure mode, while showing that attention allocation plays a limited role in explaining the observed degradation. Overall, we provide a systematic framework for analyzing and improving non-linguistic memory in long-context LALMs, shedding light on future data and training design for robust acoustic memory modeling.
#### Beyond Binary: Speech Representations Across the Cognitive Score Hierarchy
 - **Authors:** Serli Kopar, Roshan Prakash Rane, Christian Mychajliw, Lydia Federmann, Gerhard Eschweiler, Daniela Berg, Sam Gijsen, Paula Andrea Perez-Toro, Kerstin Ritter
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Machine Learning (cs.LG); Sound (cs.SD); Audio and Speech Processing (eess.AS); Neurons and Cognition (q-bio.NC)
 - **Arxiv link:** https://arxiv.org/abs/2605.27189

 - **Pdf link:** https://arxiv.org/pdf/2605.27189

 - **Abstract**
 This study examines the relationship between speech representations and the hierarchical structure of cognitive assessment in mild cognitive impairment. Utilizing 5,754 German neuropsychological assessment recordings, we evaluate six cognitive tasks across three score levels: task, domain, and global levels. We compare hand-crafted acoustic features with self-supervised learning (SSL) embeddings. Results show that although SSL representations generally outperform hand-crafted features at lower levels, this trend reverses for MCI classification. Furthermore, task-specific constraints influence performance: tasks with greater response freedom exhibit performance dilution as hierarchical levels increase, suggesting ``specialist'' representations, whereas the performance of highly structured tasks increases toward higher levels, suggesting ``generalist'' representations. These findings show links between task constraints and assessment hierarchy in automated clinical speech analysis.


by Zyzzyva0381 (Windy). 


2026-05-27
