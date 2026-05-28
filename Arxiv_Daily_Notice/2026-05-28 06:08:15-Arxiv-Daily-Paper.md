# Showing new listings for Thursday, 28 May 2026
Auto update papers at about 2:30am UTC (10:30am Beijing time) every weekday.


阅读 `Usage.md`了解如何使用此repo实现个性化的Arxiv论文推送

See `Usage.md` for instructions on how to personalize the repo. 


Keyword list: ['text-to-speech', 'text to speech', 'tts', 'LLM-based', 'speech', 'voice']


Excluded: []


### Today: 5papers 
#### LoSATok: Low-dimensional Semantic-Acoustic Tokenizer for Cross-Domain Audio Understanding and Generation
 - **Authors:** Zhisheng Zhang, Xiang Li, Yixuan Zhou, Jing Peng, Guoyang Zeng, Zhiyong Wu
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Artificial Intelligence (cs.AI); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2605.27840

 - **Pdf link:** https://arxiv.org/pdf/2605.27840

 - **Abstract**
 Audio tokenizers are fundamental to unifying audio understanding and generation. Understanding requires high-level semantics, while generation demands semantic and acoustic details. Existing unified tokenizers jointly encode both in high-dimensional continuous latents, which increases the modeling burden of Diffusion Transformers (DiTs) for generation. We propose LoSATok, a low-dimensional audio tokenizer for cross-domain audio understanding and generation. Motivated by the observation that 1280-dimensional semantic encoder features are compressible, we introduce a Semantic Bottleneck that compresses them into 128 dimensions, regularized by the proposed time-relation loss for temporal feature consistency. We further design a dual-level semantic supervision method that leverages both high- and low-dimensional semantic signals, enabling the tokenizer to jointly capture semantics and acoustic details within a compact latent space. Experiments on speech, music, and general audio show that SemBo preserves strong low-dimensional semantic capacity and LoSATok retains competitive understanding performance compared with several semantic representations, while consistently improving DiT modeling performance on speech, music, and audio generation. These results demonstrate that LoSATok's low-dimensional representations can effectively support audio understanding and generation. Our code is provided at this https URL.
#### I Hear, Therefore I Trust: A Socio-Technical Investigation of Humans as Synthetic Speech Detectors
 - **Authors:** Lelia Erscoi (1), Tomi Kinnunen (1) ((1) Computational Speech Group, University of Eastern Finland)
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Artificial Intelligence (cs.AI); Human-Computer Interaction (cs.HC)
 - **Arxiv link:** https://arxiv.org/abs/2605.28064

 - **Pdf link:** https://arxiv.org/pdf/2605.28064

 - **Abstract**
 Automatic deepfake detection has received considerable research attention, yet the socio-technical environment in which humans actually encounter synthetic speech remains poorly understood. We investigate voice deepfake detection as a perceptual and contextual process, presenting a localization task in which 47 participants marked suspected synthetic segments across authentic, fully synthetic, and partially synthetic utterances under three manipulated trust cues: instructional framing, affective priming, and provenance labeling. Participants provided quality ratings on mechanicalness, expressiveness, intelligibility, clarity, calmness, and confidence of evaluation. Utterance class was the primary determinant of detection accuracy and perceptual quality; trust cues produced no main effects but motivated detection behavior. Fully synthetic speech was detected at below-chance levels. Quality ratings tracked utterance type, indicating implicit discrimination where overt detection failed.
#### Comprehensive Benchmarking of Long-Form Speech Generation in Diverse Scenarios
 - **Authors:** Changhao Pan, Rui Yang, Han Wang, Zhuan Zhou, Xuming He, Wenxiang Guo, Ziyue Jiang, Ruiqi Li, Yu Zhang, Chenyuhao Wen, Ke Lei, Xiang Yin, Jingyu Lu, Zhiyuan Zhu, Zhou Zhao
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2605.28618

 - **Pdf link:** https://arxiv.org/pdf/2605.28618

 - **Abstract**
 Recent advances in speech generation have enabled high-fidelity synthesis, yet systematic evaluation of models under long-context conditions remains largely underexplored. A comprehensive evaluation benchmark for long-form speech is indispensable for two reasons: 1) existing test scenarios are often confined to limited domains, creating a significant gap with the diverse downstream applications; 2) existing metrics overlook critical long-text factors such as consistency and coherence, failing to generalize reliably. To this end, we propose Swanbench-Speech, a comprehensive benchmark that decomposes long-form speech quality into specific, disentangled dimensions. SwanBench-Speech has three key properties. 1) Rich speech scenarios: Focusing on long-form speech generation and dialog generation, SwanBench-Speech covers acoustics, semantics, and expressiveness challenges, and consists of 1,101 samples spanning 17 common speech scenarios; 2) Comprehensive evaluation dimensions: Along the acoustics, semantics, and expressiveness axes, SwanBench-Speech defines an automated evaluation protocol with seven metrics to provide a comprehensive, accurate, and standardized assessment; 3) Valuable Insights: Through extensive experiments, we reveal that current models still struggle in highly expressive scenarios and exhibit a notable gap in consistency and hierarchy compared to real recordings.
#### MoDAl: Self-Supervised Neural Modality Discovery via Decorrelation for Speech Neuroprosthesis
 - **Authors:** Yuanhao Chen, Peter Chin
 - **Subjects:** Subjects:
Neurons and Cognition (q-bio.NC); Computation and Language (cs.CL); Human-Computer Interaction (cs.HC); Machine Learning (cs.LG); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2605.00025

 - **Pdf link:** https://arxiv.org/pdf/2605.00025

 - **Abstract**
 Speech neuroprosthesis systems decode intended speech from neural activity in the absence of audible output, offering a path to restoring communication for individuals with speech-impairing conditions. Current approaches decode predominantly from motor cortical areas, discarding others -- such as area 44, part of Broca's area -- that may encode complementary linguistic information. We introduce MoDAl (Modality Decorrelation and Alignment), a framework that discovers complementary neural modalities through the interplay of two objectives in a shared projection space. A contrastive loss aligns each of several parallel brain encoders with the text embeddings of a pretrained large language model (LLM), while a decorrelation loss prevents the encoders from coalescing to duplicative representations. We prove that these objectives are in productive tension: Contrastive alignment induces transitive modality coalescence, which decorrelation must counteract for the framework to discover diverse neurolinguistic modalities. On the Brain-to-Text Benchmark '24, MoDAl reduces word error rate (WER) from 26.3% to 21.6% compared to the previous best end-to-end method, with the gain from incorporating previously discarded area 44 signals arising entirely from the decorrelation mechanism. Analysis of the discovered modalities reveals functional specialization: Encoders receiving area 44 input capture structural and syntactic properties (sentence length, grammatical voice, wh-words), consistent with the neurolinguistic understanding of Broca's area.
#### Diffusion Large Language Models for Visual Speech Recognition
 - **Authors:** Jeong Hun Yeo, Chae Won Kim, Hyeongseop Rha, Yong Man Ro
 - **Subjects:** Subjects:
Artificial Intelligence (cs.AI); Computer Vision and Pattern Recognition (cs.CV); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2605.28456

 - **Pdf link:** https://arxiv.org/pdf/2605.28456

 - **Abstract**
 Existing Visual Speech Recognition (VSR) systems commonly rely on left-to-right autoregressive decoding, which can force premature decisions on visually ambiguous tokens before sufficient context is available. We propose DLLM-VSR, to the best of our knowledge, the first Diffusion Large Language Model (DLLM)-based VSR framework, formulating transcription as iterative masked denoising with flexible-order decoding. With confidence-based unmasking, DLLM-VSR commits high-confidence positions early and uses the committed tokens as bidirectional context to refine ambiguous ones. To adapt DLLMs to VSR, we introduce a two-stage masked-denoising training strategy that separates visual-to-text content alignment from length modeling. We further observe a performance gap with oracle-length decoding, which assumes access to the true transcript length, indicating that reducing target-length uncertainty can improve DLLM-based VSR. To reduce this gap, we develop length-guided candidate decoding, which uses video duration to construct plausible transcript-length hypotheses, decodes under multiple hypotheses, and reranks candidates using length plausibility and decoding confidence. The proposed method achieves a state-of-the-art WER of 19.5\% on LRS3 using only its labeled training data.


by Zyzzyva0381 (Windy). 


2026-05-28
