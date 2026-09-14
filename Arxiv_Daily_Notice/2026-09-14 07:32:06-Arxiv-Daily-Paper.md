# Showing new listings for Monday, 14 September 2026
Auto update papers at about 2:30am UTC (10:30am Beijing time) every weekday.


阅读 `Usage.md`了解如何使用此repo实现个性化的Arxiv论文推送

See `Usage.md` for instructions on how to personalize the repo. 


Keyword list: ['text-to-speech', 'text to speech', 'tts', 'LLM-based', 'speech', 'voice']


Excluded: []


### Today: 10papers 
#### VoxTubeS: Distributable Speaker-Anonymized Synthetic Speech Corpora and Their Analysis
 - **Authors:** Zhe Zhang, Yexin Lu, Junichi Yamagishi
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2609.12432

 - **Pdf link:** https://arxiv.org/pdf/2609.12432

 - **Abstract**
 Large speech corpora support research, but recordings can expose speaker identity because voice remains a recognizable biometric. Meanwhile, speech data derived from media can be difficult to redistribute reliably. We present \emph{VoxTubeS}, a family of speaker-anonymized synthetic speech corpora designed for redistribution, comprising three method families and seven variants derived from the VoxTube corpus, which is distributed under CC BY-NC-SA 4.0, using 1.29M quality-filtered English utterances from 1,511 speakers. The synthesis methods span voice conversion, latent-space anonymization, and controllable text-to-speech. We evaluate VoxTubeS using utterance-level unlinkability, conversation-level linkability and singling-out, downstream speaker verification, linguistic consistency, speaker diversity, and fairness metrics for gender and accents. Our comprehensive analysis exposes a complex trade-off: stronger identity suppression often reduces linkability but sacrifices utility and population diversity, whereas speaker consistency training improves both utterance- and conversation-level privacy while retaining comparable utility and a broader speaker space. Fairness varies independently of aggregate performance. No method dominates; VoxTubeS therefore treats corpus construction as a choice among operating points that balances privacy, utility, diversity, fairness, and responsible redistribution under the source license.
#### Location-based Training with Complementary Folded Linear Orderings for Multichannel Speech Separation
 - **Authors:** Kaixuan Yang, Stijn Kindt, Nilesh Madhu
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2609.12629

 - **Pdf link:** https://arxiv.org/pdf/2609.12629

 - **Abstract**
 Location-based training (LBT) effectively resolves the output permutation problem in multichannel speech separation by imposing deterministic spatial orderings. For planar microphone arrays, LBT typically adopts circular azimuth ordering to cover the full spatial range. However, the resulting cyclic topology introduces a discontinuity at the wrap-around point, increasing learning complexity and limiting the effective use of spatial cues. This work investigates this limitation by introducing location-based training with folded linear orderings (LBT-FLOs), which collapse circular azimuths into controlled linear orderings. While individual LBT-FLOs exhibit front-back ambiguity, each provides enhanced spatial discriminability over specific azimuth regions. Exploiting their complementarity, we propose an ensemble-style framework that selects among multiple LBT-FLOs using azimuth-guided scoring. Experiments across planar array geometries and reverberant conditions show modest but consistent improvements over circular-ordering LBT, with robustness to azimuth estimation errors.
#### X-Pred MeanFlow for Streaming Token-to-Mel Speech Decoding
 - **Authors:** Hanke Xie, Xiaming Ren, Qirui Zhan, Jingbin Hu, Wenhao Li, Haoyu Zhang, Ruonan You, Chengyou Wang, Yunxiang Chen, Houdun Liu, Su Feng, Lei Xie
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2609.12728

 - **Pdf link:** https://arxiv.org/pdf/2609.12728

 - **Abstract**
 Recent advancements in discrete token-based speech generation have highlighted the importance of efficient token-to-waveform synthesis in streaming and dialogue scenarios. Flow-matching acoustic decoders achieve high-quality token-to-mel generation, but their iterative sampling requires multiple neural function evaluations, limiting low-latency speech synthesis. MeanFlow reduces the sampling budget by modeling the average velocity over a temporal interval, yet maintaining high acoustic quality under extremely few-step token-to-mel generation remains challenging. To address this challenge, we propose X-Pred MeanFlow, a few-step streaming token-to-mel decoder that reparameterizes MeanFlow with mel-space prediction. The decoder predicts a generalized mel field and analytically derives the corresponding average velocity for sampling, thereby preserving the MeanFlow formulation while providing a direct acoustic prediction target. We further introduce layer-selective block-wise attention to enable continuous chunk-wise generation with bounded context. Experiments show that X-Pred MeanFlow improves few-step token-to-mel synthesis over Direct-$u$ MeanFlow and supports stable streaming generation. Speech samples are this http URL://renxiaming.this http URL
#### A Device to Control and Manipulate Occlusion Effects for Own Voice Perception Studies
 - **Authors:** Rouben Rehman, Simon Kersten, Aron Schliep, Janina Fels (Institute for Hearing Technology and Acoustics, RWTH Aachen University)
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2609.12845

 - **Pdf link:** https://arxiv.org/pdf/2609.12845

 - **Abstract**
 The occlusion effect (OE) refers to changes of the eardrum sound pressure through ear canal occlusion. It consists of two phenomena: an insertion loss (IL) attenuating air-conducted sounds, and an occlusion gain (OG) amplifying bone-conduction. Perceptual research on this is hindered by high variability of the OE across individuals, complicating repeatable presentation of precise OE conditions. Consequently, a method to control OE conditions reproducibly during perceptual experiments is needed. For such investigations, the system must enable separate control of IL and OG. We present an approach based on modified commercial earmuffs with integrated microphones. The design integrates dedicated impedance measurements and the derivation of digital filters, which are applied to the microphone signals to emulate arbitrary OE curves. The system is evaluated objectively through appropriate measurements. Results show that the headphones' inherent OG is guaranteed to be below 6dB above 115Hz, falling below 0dB above 155Hz. Emulation is shown to work accurately over the entire frequency range of interest. Limitations arise mainly due to a slight residual inherent OG for deep voices and the processing delay of the system. Future work will focus on the perceptual evaluation of the system and its application in perceptual studies.
#### AlignDPO: Preference-Gated Alignment for Reducing Hallucination in Decoder-Only TTS
 - **Authors:** Xiao Zhou, Oisín Turbitt, Kit Bower-Morris, Jonathan Carlton, Jamie Stacey, Kris Y. Hong
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2609.12855

 - **Pdf link:** https://arxiv.org/pdf/2609.12855

 - **Abstract**
 Decoder-only text-to-speech (TTS) models scale efficiently but remain prone to content hallucinations that arise from weak text-speech alignment during autoregressive generation. We find that robustness is governed by a non-monotone relation to the sharpness of the alignment-bearing attention heads: a moderate degree is best, whereas over-sharpening is no better than the unaligned backbone and even less robust. Guided by this, we present AlignDPO, a post-training method that reaches this moderate regime by folding a lightweight connectionist-temporal-classification (CTC) alignment term into Direct Preference Optimization (DPO), applied only to the chosen samples, with no architectural or inference-time change. On the Seed-TTS-Eval English set, this significantly reduces the content-hallucination and word error rates relative to a strong DPO baseline and lowers the severe content-hallucination rate to ~0.6% (from 4.4%); a listening study further finds it preferred for naturalness over both the backbone and that baseline. Alignment is thus best learned and kept moderate rather than maximized or imposed at decoding. Audio samples are available at this https URL.
#### Objective Intelligibility Prediction Using Distance Metrics on Speech Foundation Model Representations
 - **Authors:** Lyonel Behringer, Andreas Brendel
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2609.13046

 - **Pdf link:** https://arxiv.org/pdf/2609.13046

 - **Abstract**
 High-dimensional representations of pretrained speech foundation models have proven beneficial for objective speech quality and intelligibility prediction. While existing work on neural intelligibility prediction usually leverages such representations for task-specific fine-tuning, in this work we evaluate the usefulness of such representations for intelligibility prediction without any further training. We conduct a layer-wise analysis of multiple speech foundation models, correlating various embedding distances with subjective intelligibility scores. The results show that embeddings extracted from Whisper speech recognition models are best suited, with the last encoder and decoder layers yielding the best correlations when using the Fréchet Audio Distance. Notably, the evaluated distances outperform classical intelligibility metrics and are more robust than Word and Character Error Rates. Further, correlations improve with increasing size of the Whisper model from which embeddings are extracted.
#### MP-Bench: Evaluating Voice Agents as a Multiparty Conversation Participant
 - **Authors:** Yi-Jen Shih, Shih-Yun Shan Kuan, Guan-Ting Lin, Kai-Wei Chang, Siddhant Arora, Shu-wen Yang, Abdelrahman Mohamed, Shinji Watanabe, Hung-yi Lee, David Harwath
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Artificial Intelligence (cs.AI); Computation and Language (cs.CL)
 - **Arxiv link:** https://arxiv.org/abs/2609.13076

 - **Pdf link:** https://arxiv.org/pdf/2609.13076

 - **Abstract**
 Conversational voice agents have advanced significantly, offering increasingly natural human-machine interactions through both cascaded and end-to-end architectures. However, while recent benchmarks extensively evaluate dyadic interactions and passive audio comprehension, they largely overlook a prevalent real-world scenario: multi-party conversations. Evaluating agents in these settings is fundamentally more challenging than in dyadic interactions due to the exponentially greater conversational complexity. For voice agents to integrate seamlessly into human group dynamics, they must not only generate contextually appropriate responses but also demonstrate a nuanced understanding of open turn-taking. To address this gap, we introduce Multiparty Bench (MP-Bench), the first benchmark specifically designed to objectively evaluate conversational speech systems as active participants within multi-party contexts. MP-Bench assesses agent behavior along two primary dimensions: turn-taking awareness and response appropriateness. Additionally, we incorporate comprehension-based question-answering tasks as a complementary evaluation. By benchmarking 12 voice agents, we find that real-time voice agents stay at or below 22% on multiparty comprehension and remain near chance on multiparty turn-taking, exposing an open challenge for real-time voice agents under multiparty scenario.
#### DriftSE: Speech Enhancement with Generative Drifting
 - **Authors:** Liang Xu, Diego Caviedes-Nozal, W. Bastiaan Kleijn, Longfei Felix Yan, Rasmus Kongsgaard Olsson
 - **Subjects:** Subjects:
Sound (cs.SD); Artificial Intelligence (cs.AI); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2609.12252

 - **Pdf link:** https://arxiv.org/pdf/2609.12252

 - **Abstract**
 We propose DriftSE, a novel one-step generative framework for speech enhancement formulated as a latent distribution equilibrium problem. During training, the drifting field aligns the generator's pushforward distribution with the clean speech manifold through drifting in a latent domain. During inference, the drifting process is discarded, enabling one-step generation. We establish that its enhancement quality depends fundamentally on the choice of latent representation. Semantic latents preserve phonetic structure but fail to capture physical acoustic cues, whereas acoustic latents reconstruct the physical signal but risk linguistic hallucination. Therefore, we introduce dual-latent drifting, performing parallel drifting in both semantic and acoustic latents to simultaneously preserve phonetic intelligibility and acoustic fidelity. Additionally, we demonstrate that DriftSE enables fully unpaired training by aligning latent distributions rather than exact point-wise targets. Consequently, DriftSE facilitates cross-dataset learning in the absence of paired noisy-clean samples. Moreover, DriftSE exhibits broad architectural flexibility across different generator backbones. Extensive evaluations on additive denoising and convolutive dereverberation demonstrate robust one-step enhancement across both offline and real-time causal settings. Notably, DriftSE achieves state-of-the-art word error rates across all four evaluated datasets while strictly operating at 1 NFE. Code and audio examples are available online.
#### TokenMapper: A Step Toward Interoperable Speech Token Translation
 - **Authors:** Tal Kozakov, Tal Rosenwein, Eliya Nachmani
 - **Subjects:** Subjects:
Machine Learning (cs.LG); Artificial Intelligence (cs.AI); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2609.12563

 - **Pdf link:** https://arxiv.org/pdf/2609.12563

 - **Abstract**
 Neural audio codecs discretize speech into token sequences, but the resulting token spaces differ in vocabulary and codebook structure, preventing direct communication across models. This limitation affects applications such as conversational voice agents and speech to speech translation systems where multiple speech models must interact. As a result, transferring information between speech systems typically requires decoding to waveform audio and re-encoding with a second tokenizer, increasing latency and introducing potential information loss. To address these limitations, we present TokenMapper, a direction aware framework for direct token to token translation between heterogeneous speech tokenizers in the discrete domain. TokenMapper supports structurally mismatched token spaces, including mappings between single codebook and multi codebook representations, under a shared effective token rate. Experiments on GLM-4-Voice, MiMi and DualCodec show consistent cross model performance. Specifically, translation WER approaches native reconstructions within 2.5-6.8% absolute WER, human MOS for TokenMapper outputs ranges from 2.29 to 4.39, following the same direction level trends as UTMOS and end to end latency is reduced by 4.8-94.5% relative to waveform bridging, reaching up to 972 ms per utterance. These results provide a practical step toward cross model speech token interoperability without intermediate waveform reconstruction.
#### StepAudio 3 Gen Technical Report
 - **Authors:** Bin Lin, Bo Zhao, Boyang Wang, Boyang Zhang, Boyong Wu, Chao Yan, Chen Geng, Chen Wu, Cheng Yi, Chengli Feng, Chenglin Zhu, DanNi Wan, Daxin Jiang, Dongqing Pang, Fei Tian, Feng Tian, Future Li, Gang Yu, Guanglong Yang, Jia Peng, Jiahao Song, Jiamin Fan, Jiangjie Zhen, Jianzheng Gao, Jun Chen, Li Xie, Lifang Zhang, Lingli Ji, Liying Shi, Lun Cai, Min Xu, Na Wang, Peilin Li, Peng Yang, Pengfei Tan, Qingjian Lin, Ruijie Xiong, Runze Li, Shenghua Hu, Shi Qiu, Siqi Tu, Siyi Zhou, Tianjiao Deng, Wanying Lu, Weiming Niu, Wen Sun, WenWen Qu, Xiangyu Zhang, Xianwei Zhang, XiaoSu Su, Xing Chen, Xinyu Liu, Xuerui Yang, Yang Li, Yang Yang, Yechang Huang, Yibo Zhu, Yifan Zhang, Yiyang Xu, Yu Fu, Yu Luo, Yu Zhou, Yumang Wang, Yunzhou Ju, Yuxiang Yang, Zekai Liu, Zengwei Yao, Zhenwei Mou, Zheqi Dai, Zhiyue Wu, Zichao Zhou
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2609.12945

 - **Pdf link:** https://arxiv.org/pdf/2609.12945

 - **Abstract**
 We introduce StepAudio 3 Gen, a general-purpose audio generation model that supports zero-shot text-to-speech (TTS), voice design, vocal generation, sound effects, music, vibe speech, and mixtures of multiple audio types within a unified framework. At its core, StepAudio 3 Gen is a discrete autoregressive generator that models audio directly over residual vector quantization (RVQ) tokens, departing from the diffusion Transformer-based continuous generation paradigm prevalent in recent general audio models. Its StepAudio Tokenizer represents general audio at 12.5 Hz in a shared $16 \times 2048$ residual code space, jointly quantizing semantic and waveform-level acoustic features so that each code layer preserves both types of information. For generation, the backbone predicts the first codebook along the time axis using autoregressive modeling, while a lightweight causal Transformer completes the remaining fifteen codebooks along the codebook axis. Our study further identifies three key design principles: (1) interference-aware progressive pretraining for acquiring audio capabilities while preserving the textual abilities of the large language model, (2) RVQ Adaptor for effectively incorporating multi-codebook acoustic representations, and (3) discrete autoregressive modeling over a shared representation across general audio domains. With progressive pretraining, multi-task instruction training, and supervised fine-tuning, StepAudio 3 Gen achieves state-of-the-art performance on both TTS and voice design, while retaining strong generation capabilities across speech, vocals, sound effects, and music. Audio samples are available at this https URL.


by Zyzzyva0381 (Windy). 


2026-09-14
