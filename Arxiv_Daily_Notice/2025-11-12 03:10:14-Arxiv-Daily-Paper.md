# Showing new listings for Wednesday, 12 November 2025
Auto update papers at about 2:30am UTC (10:30am Beijing time) every weekday.


阅读 `Usage.md`了解如何使用此repo实现个性化的Arxiv论文推送

See `Usage.md` for instructions on how to personalize the repo. 


Keyword list: ['text-to-speech', 'text to speech', 'tts', 'LLM-based', 'speech', 'voice']


Excluded: []


### Today: 4papers 
#### Pruning as Regularization: Sensitivity-Aware One-Shot Pruning in ASR
 - **Authors:** Julian Irigoyen, Arthur Söhler, Andreas Søeborg Kirkedal
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Computation and Language (cs.CL); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2511.08092

 - **Pdf link:** https://arxiv.org/pdf/2511.08092

 - **Abstract**
 We challenge the conventional view of neural network pruning as solely a compression technique, demonstrating that one-shot magnitude pruning serves as a powerful implicit regularizer for ASR. Using Whisper-small, we combine gradient- and Fisher-based sensitivity diagnostics with targeted, component-wise pruning. This reveals architectural asymmetries: decoder FFNs are pruning-fragile, whereas decoder self-attention and the last encoder layers contain redundancy that, when removed, improves generalization. Without fine-tuning, pruning 50% of decoder self-attention reduces WER by 2.38% absolute (20.44% relative) on LibriSpeech test-other; pruning the last four encoder layers at 50% instead yields a 1.72% absolute (14.8% relative) improvement. Gains persisted on Common Voice and TED-LIUM datasets. Beyond regularization benefits, our sensitivity-aware approach enables more aggressive one-shot compression. At 40% sparsity, where established global pruning approaches catastrophically fail, our method preserves near-baseline accuracy. This positions pruning as a first-class architectural design tool: knowing where to prune is as important as how much to prune.
#### Quantizing Whisper-small: How design choices affect ASR performance
 - **Authors:** Arthur Söhler, Julian Irigoyen, Andreas Søeborg Kirkedal
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Computation and Language (cs.CL); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2511.08093

 - **Pdf link:** https://arxiv.org/pdf/2511.08093

 - **Abstract**
 Large speech recognition models like Whisper-small achieve high accuracy but are difficult to deploy on edge devices due to their high computational demand. To this end, we present a unified, cross-library evaluation of post-training quantization (PTQ) on Whisper-small that disentangles the impact of quantization scheme, method, granularity, and bit-width. Our study is based on four libraries: PyTorch, Optimum-Quanto, HQQ, and bitsandbytes. Experiments on LibriSpeech test-clean and test-other show that dynamic int8 quantization with Quanto offers the best trade-off, reducing model size by 57% while improving on the baseline's word error rate. Static quantization performed worse, likely due to Whisper's Transformer architecture, while more aggressive formats (e.g., nf4, int3) achieved up to 71% compression at the cost of accuracy in noisy conditions. Overall, our results demonstrate that carefully chosen PTQ methods can substantially reduce model size and inference cost without retraining, enabling efficient deployment of Whisper-small on constrained hardware.
#### Unifying Model and Layer Fusion for Speech Foundation Models
 - **Authors:** Yi-Jen Shih, David Harwath
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Artificial Intelligence (cs.AI); Computation and Language (cs.CL)
 - **Arxiv link:** https://arxiv.org/abs/2511.08389

 - **Pdf link:** https://arxiv.org/pdf/2511.08389

 - **Abstract**
 Speech Foundation Models have gained significant attention recently. Prior works have shown that the fusion of representations from multiple layers of the same model or the fusion of multiple models can improve performance on downstream tasks. We unify these two fusion strategies by proposing an interface module that enables fusion across multiple upstream speech models while integrating information across their layers. We conduct extensive experiments on different self-supervised and supervised models across various speech tasks, including ASR and paralinguistic analysis, and demonstrate that our method outperforms prior fusion approaches. We further analyze its scalability concerning model size and count, highlighting the importance of selecting appropriate upstream models. Our results show that the proposed interface provides an additional performance boost when given a suitable upstream model selection, making it a promising approach for utilizing Speech Foundation Models.
#### HQ-SVC: Towards High-Quality Zero-Shot Singing Voice Conversion in Low-Resource Scenarios
 - **Authors:** Bingsong Bai, Yizhong Geng, Fengping Wang, Cong Wang, Puyuan Guo, Yingming Gao, Ya Li
 - **Subjects:** Subjects:
Sound (cs.SD); Artificial Intelligence (cs.AI); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2511.08496

 - **Pdf link:** https://arxiv.org/pdf/2511.08496

 - **Abstract**
 Zero-shot singing voice conversion (SVC) transforms a source singer's timbre to an unseen target speaker's voice while preserving melodic content without fine-tuning. Existing methods model speaker timbre and vocal content separately, losing essential acoustic information that degrades output quality while requiring significant computational resources. To overcome these limitations, we propose HQ-SVC, an efficient framework for high-quality zero-shot SVC. HQ-SVC first extracts jointly content and speaker features using a decoupled codec. It then enhances fidelity through pitch and volume modeling, preserving critical acoustic information typically lost in separate modeling approaches, and progressively refines outputs via differentiable signal processing and diffusion techniques. Evaluations confirm HQ-SVC significantly outperforms state-of-the-art zero-shot SVC methods in conversion quality and efficiency. Beyond voice conversion, HQ-SVC achieves superior voice naturalness compared to specialized audio super-resolution methods while natively supporting voice super-resolution tasks.


by Zyzzyva0381 (Windy). 


2025-11-12
