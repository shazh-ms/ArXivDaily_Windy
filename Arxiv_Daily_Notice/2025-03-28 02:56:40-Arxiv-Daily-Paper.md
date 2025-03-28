# Showing new listings for Friday, 28 March 2025
Auto update papers at about 2:30am UTC (10:30am Beijing time) every weekday.


阅读 `Usage.md`了解如何使用此repo实现个性化的Arxiv论文推送

See `Usage.md` for instructions on how to personalize the repo. 


Keyword list: ['acoustic', 'reinforcement learning', 'reverb', 'meta', 'separate', 'reconstruction', 'noise', 'enhance', 'localization', 'speech']


Excluded: []


### Today: 3papers 
#### A Low-Power Streaming Speech Enhancement Accelerator For Edge Devices
 - **Authors:** Ci-Hao Wu, Tian-Sheuan Chang
 - **Subjects:** Subjects:
Hardware Architecture (cs.AR); Artificial Intelligence (cs.AI); Multimedia (cs.MM); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2503.21335

 - **Pdf link:** https://arxiv.org/pdf/2503.21335

 - **Abstract**
 Transformer-based speech enhancement models yield impressive results. However, their heterogeneous and complex structure restricts model compression potential, resulting in greater complexity and reduced hardware efficiency. Additionally, these models are not tailored for streaming and low-power applications. Addressing these challenges, this paper proposes a low-power streaming speech enhancement accelerator through model and hardware optimization. The proposed high performance model is optimized for hardware execution with the co-design of model compression and target application, which reduces 93.9\% of model size by the proposed domain-aware and streaming-aware pruning techniques. The required latency is further reduced with batch normalization-based transformers. Additionally, we employed softmax-free attention, complemented by an extra batch normalization, facilitating simpler hardware design. The tailored hardware accommodates these diverse computing patterns by breaking them down into element-wise multiplication and accumulation (MAC). This is achieved through a 1-D processing array, utilizing configurable SRAM addressing, thereby minimizing hardware complexities and simplifying zero skipping. Using the TSMC 40nm CMOS process, the final implementation requires merely 207.8K gates and 53.75KB SRAM. It consumes only 8.08 mW for real-time inference at a 62.5MHz frequency.
#### A 71.2-$μ$W Speech Recognition Accelerator with Recurrent Spiking Neural Network
 - **Authors:** Chih-Chyau Yang, Tian-Sheuan Chang
 - **Subjects:** Subjects:
Hardware Architecture (cs.AR); Artificial Intelligence (cs.AI); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2503.21337

 - **Pdf link:** https://arxiv.org/pdf/2503.21337

 - **Abstract**
 This paper introduces a 71.2-$\mu$W speech recognition accelerator designed for edge devices' real-time applications, emphasizing an ultra low power design. Achieved through algorithm and hardware co-optimizations, we propose a compact recurrent spiking neural network with two recurrent layers, one fully connected layer, and a low time step (1 or 2). The 2.79-MB model undergoes pruning and 4-bit fixed-point quantization, shrinking it by 96.42\% to 0.1 MB. On the hardware front, we take advantage of \textit{mixed-level pruning}, \textit{zero-skipping} and \textit{merged spike} techniques, reducing complexity by 90.49\% to 13.86 MMAC/S. The \textit{parallel time-step execution} addresses inter-time-step data dependencies and enables weight buffer power savings through weight sharing. Capitalizing on the sparse spike activity, an input broadcasting scheme eliminates zero computations, further saving power. Implemented on the TSMC 28-nm process, the design operates in real time at 100 kHz, consuming 71.2 $\mu$W, surpassing state-of-the-art designs. At 500 MHz, it has 28.41 TOPS/W and 1903.11 GOPS/mm$^2$ in energy and area efficiency, respectively.
#### Magnitude-Phase Dual-Path Speech Enhancement Network based on Self-Supervised Embedding and Perceptual Contrast Stretch Boosting
 - **Authors:** Alimjan Mattursun, Liejun Wang, Yinfeng Yu, Chunyang Ma
 - **Subjects:** Subjects:
Sound (cs.SD); Artificial Intelligence (cs.AI); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2503.21571

 - **Pdf link:** https://arxiv.org/pdf/2503.21571

 - **Abstract**
 Speech self-supervised learning (SSL) has made great progress in various speech processing tasks, but there is still room for improvement in speech enhancement (SE). This paper presents BSP-MPNet, a dual-path framework that combines self-supervised features with magnitude-phase information for SE. The approach starts by applying the perceptual contrast stretching (PCS) algorithm to enhance the magnitude-phase spectrum. A magnitude-phase 2D coarse (MP-2DC) encoder then extracts coarse features from the enhanced spectrum. Next, a feature-separating self-supervised learning (FS-SSL) model generates self-supervised embeddings for the magnitude and phase components separately. These embeddings are fused to create cross-domain feature representations. Finally, two parallel RNN-enhanced multi-attention (REMA) mask decoders refine the features, apply them to the mask, and reconstruct the speech signal. We evaluate BSP-MPNet on the VoiceBank+DEMAND and WHAMR! datasets. Experimental results show that BSP-MPNet outperforms existing methods under various noise conditions, providing new directions for self-supervised speech enhancement research. The implementation of the BSP-MPNet code is available online\footnote[2]{this https URL. \label{s1}}


by Zyzzyva0381 (Windy). 


2025-03-28
