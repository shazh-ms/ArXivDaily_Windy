# Showing new listings for Monday, 19 January 2026
Auto update papers at about 2:30am UTC (10:30am Beijing time) every weekday.


阅读 `Usage.md`了解如何使用此repo实现个性化的Arxiv论文推送

See `Usage.md` for instructions on how to personalize the repo. 


Keyword list: ['text-to-speech', 'text to speech', 'tts', 'LLM-based', 'speech', 'voice']


Excluded: []


### Today: 3papers 
#### DSA-Tokenizer: Disentangled Semantic-Acoustic Tokenization via Flow Matching-based Hierarchical Fusion
 - **Authors:** Hanlin Zhang, Daxin Tan, Dehua Tao, Xiao Chen, Haochen Tan, Yunhe Li, Yuchen Cao, Jianping Wang, Linqi Song
 - **Subjects:** Subjects:
Sound (cs.SD); Artificial Intelligence (cs.AI); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2601.09239

 - **Pdf link:** https://arxiv.org/pdf/2601.09239

 - **Abstract**
 Speech tokenizers serve as the cornerstone of discrete Speech Large Language Models (Speech LLMs). Existing tokenizers either prioritize semantic encoding, fuse semantic content with acoustic style inseparably, or achieve incomplete semantic-acoustic disentanglement. To achieve better disentanglement, we propose DSA-Tokenizer, which explicitly disentangles speech into discrete semantic and acoustic tokens via distinct optimization constraints. Specifically, semantic tokens are supervised by ASR to capture linguistic content, while acoustic tokens focus on mel-spectrograms restoration to encode style. To eliminate rigid length constraints between the two sequences, we introduce a hierarchical Flow-Matching decoder that further improve speech generation quality. Furthermore, We employ a joint reconstruction-recombination training strategy to enforce this separation. DSA-Tokenizer enables high fidelity reconstruction and flexible recombination through robust disentanglement, facilitating controllable generation in speech LLMs. Our analysis highlights disentangled tokenization as a pivotal paradigm for future speech modeling. Audio samples are avaialble at this https URL. The code and model will be made publicly available after the paper has been accepted.
#### Unifying Speech Recognition, Synthesis and Conversion with Autoregressive Transformers
 - **Authors:** Runyuan Cai, Yu Lin, Yiming Wang, Chunlin Fu, Xiaodong Zeng
 - **Subjects:** Subjects:
Sound (cs.SD); Artificial Intelligence (cs.AI); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2601.10770

 - **Pdf link:** https://arxiv.org/pdf/2601.10770

 - **Abstract**
 Traditional speech systems typically rely on separate, task-specific models for text-to-speech (TTS), automatic speech recognition (ASR), and voice conversion (VC), resulting in fragmented pipelines that limit scalability, efficiency, and cross-task generalization. In this paper, we present General-Purpose Audio (GPA), a unified audio foundation model that integrates multiple core speech tasks within a single large language model (LLM) architecture. GPA operates on a shared discrete audio token space and supports instruction-driven task induction, enabling a single autoregressive model to flexibly perform TTS, ASR, and VC without architectural modifications. This unified design combines a fully autoregressive formulation over discrete speech tokens, joint multi-task training across speech domains, and a scalable inference pipeline that achieves high concurrency and throughput. The resulting model family supports efficient multi-scale deployment, including a lightweight 0.3B-parameter variant optimized for edge and resource-constrained environments. Together, these design choices demonstrate that a unified autoregressive architecture can achieve competitive performance across diverse speech tasks while remaining viable for low-latency, practical deployment.
#### FlashLabs Chroma 1.0: A Real-Time End-to-End Spoken Dialogue Model with Personalized Voice Cloning
 - **Authors:** Tanyu Chen, Tairan Chen, Kai Shen, Zhenghua Bao, Zhihui Zhang, Man Yuan, Yi Shi
 - **Subjects:** Subjects:
Sound (cs.SD); Computation and Language (cs.CL); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2601.11141

 - **Pdf link:** https://arxiv.org/pdf/2601.11141

 - **Abstract**
 Recent end-to-end spoken dialogue systems leverage speech tokenizers and neural audio codecs to enable LLMs to operate directly on discrete speech representations. However, these models often exhibit limited speaker identity preservation, hindering personalized voice interaction. In this work, we present Chroma 1.0, the first open-source, real-time, end-to-end spoken dialogue model that achieves both low-latency interaction and high-fidelity personalized voice cloning. Chroma achieves sub-second end-to-end latency through an interleaved text-audio token schedule (1:2) that supports streaming generation, while maintaining high-quality personalized voice synthesis across multi-turn conversations. Our experimental results demonstrate that Chroma achieves a 10.96% relative improvement in speaker similarity over the human baseline, with a Real-Time Factor (RTF) of 0.43, while maintaining strong reasoning and dialogue capabilities. Our code and models are publicly available at this https URL and this https URL .


by Zyzzyva0381 (Windy). 


2026-01-21
