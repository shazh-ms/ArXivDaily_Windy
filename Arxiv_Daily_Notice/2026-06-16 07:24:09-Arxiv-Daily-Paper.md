# Showing new listings for Tuesday, 16 June 2026
Auto update papers at about 2:30am UTC (10:30am Beijing time) every weekday.


阅读 `Usage.md`了解如何使用此repo实现个性化的Arxiv论文推送

See `Usage.md` for instructions on how to personalize the repo. 


Keyword list: ['text-to-speech', 'text to speech', 'tts', 'LLM-based', 'speech', 'voice']


Excluded: []


### Today: 23papers 
#### Pixel-TTS: Image based Text Rendering for Robust Text-to-Speech
 - **Authors:** Adarsh Arigala, Arjun Gangwar, S Umesh, Yova Kementchedjhieva
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Artificial Intelligence (cs.AI); Computer Vision and Pattern Recognition (cs.CV); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2606.14750

 - **Pdf link:** https://arxiv.org/pdf/2606.14750

 - **Abstract**
 Recent advances in pixel-based text modeling show that representing text as images enables models to exploit visual cues for language understanding. Grounding text in its visual form allows structurally similar characters with different Unicode encodings to produce similar embeddings, benefiting cross-lingual and zero-shot scenarios. Conventional text-based approaches treat each character independently, limiting generalization to unseen characters and requiring embedding expansion during cross-lingual adaptation. We propose Pixel-TTS, the first framework for visually grounded speech synthesis. It renders text as images and projects them through a 2D convolutional layer to generate embeddings. This design eliminates embedding matrix expansion during fine-tuning while improving robustness to unseen characters and orthographic variations. Extensive experiments show Pixel-TTS achieves competitive performance with strong baselines, faster convergence and robust zero-shot generalization.
#### From Physics to Representation: Audio Learning with Synthetic Pre-training via Procedural Generation
 - **Authors:** Fengrui Liu, Ruiyang Huang, Qijian Zheng, Yuanfang Wang, Feng Liu
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Machine Learning (cs.LG); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2606.14791

 - **Pdf link:** https://arxiv.org/pdf/2606.14791

 - **Abstract**
 Self-supervised learning advances audio representation for multimedia analysis. However, prevailing data-centric approaches rely on massive real-world corpora, increasing training costs, curation burdens, and privacy barriers. To address this, we present AudioPG, a procedural synthesis framework eliminating real audio recordings during pre-training. AudioPG trains a Transformer-based masked autoencoder on waveforms generated on-the-fly from basic acoustic primitives and composition rules. The encoder transfers effectively to real audio benchmarks, achieving 90.60% accuracy on ESC-50, 0.546 mAP on FSD50K, 88.17% on UrbanSound8K, and 97.03% on Speech Commands V2. Notably, pre-training completes in under 20 minutes on a single GPU. Latent space analysis reveals physical factors, including fundamental frequency and relative intensity, emerge in orthogonal subspaces, making representations linearly decodable. These results establish procedural synthesis as an efficient, interpretable pre-training signal when large-scale corpora are unavailable. Our code is available at: this https URL.
#### VoxWatermark: A Large-Scale Benchmark for Audio Watermark Detection under Perturbations
 - **Authors:** Farnaz Sedaghati, Yuxi Wang, Zicheng Weng, Wei Rao
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2606.15187

 - **Pdf link:** https://arxiv.org/pdf/2606.15187

 - **Abstract**
 With the rapid deployment of speech generation systems in open environments, providing verifiable source attribution and copyright accountability for audio content has become critical. A gap in current research is the lack of a unified benchmark that systematically compares different watermark injection methods under realistic distribution shifts. To address this, we build VoxWatermark by applying 10 watermarking methods (4 neural and 6 traditional) with unified injection and annotation on multilingual, multi-source corpora, and introducing no-box, black-box, and white-box perturbations to simulate real recording and transmission conditions. Based on this benchmark, we propose AudioWMD as a robust baseline detector for large-scale, multi-method, cross-distribution settings. Results show that injection-method diversity and distribution shifts affect detection stability, while validating the effectiveness and scalability of AudioWMD. Dataset and code are publicly available.
#### DuraMark: Duration-Embedded Watermarking in LLM-based TTS
 - **Authors:** Zhenwei Mou, Weili Jiang, Liping Chen, Zhen-Hua Ling, Kong Aik Lee, Kai Gao, Boyu Zhao
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2606.15264

 - **Pdf link:** https://arxiv.org/pdf/2606.15264

 - **Abstract**
 Large language model (LLM)-based text-to-speech (TTS) models have achieved remarkable voice cloning capabilities, raising concerns about potential deepfake misuse. Speech watermarking mitigates this by embedding traceable information into generated speech. Mainstream watermarking methods operate at the signal level (waveform or spectrogram), rendering the watermark vulnerable to generative attacks (e.g., neural codec and vocoder). To address this, we propose DuraMark, a robust information-level watermarking framework. It utilizes syllable duration editing to achieve watermark embedding. Specifically, DuraMark integrates a duration-controllable LLM-based TTS model to edit syllable durations during synthesis, coupled with a duration extractor to extract these durations for detection. Experiments demonstrate DuraMark's superior robustness against generative attacks, significantly outperforming signal-level baselines. Audio samples are available at this https URL.
#### Dynamic Prosody Prediction in LLM-based TTS for Improving Speaker Similarity
 - **Authors:** Zhenwei Mou, Liping Chen, Yajun Hu, Zhen-Hua Ling, Xin Fang, Jianqing Gao
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2606.15267

 - **Pdf link:** https://arxiv.org/pdf/2606.15267

 - **Abstract**
 Personalized text-to-speech (TTS) aims to clone the target speaker in the synthesized speech, imitating both the voice and speaking style. Current large language model (LLM)-based TTS methods ignore the style-specific prosodic patterns in generated speech, resulting in deficient style learning and thus limiting speaker similarity in synthesized speech. To this end, we investigate the prosody learning conditioned on the synthesized speech, and propose to predict the prosody of the current syllable based on previously predicted speech. Experimental results obtained on three datasets demonstrated the efficacy of the proposed dynamic prosody prediction method in enhancing the prosody learning capability, thereby improving the speaker similarity of the generated speech. Audio samples are available at this https URL.
#### DDPO-VC: Speaker De-Identification via Diffusion Denoising Policy Optimization
 - **Authors:** Liming Wang, Cody Karjadi, Rhoda Au, James Glass
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2606.15313

 - **Pdf link:** https://arxiv.org/pdf/2606.15313

 - **Abstract**
 A key challenge of speaker de-identification is the balance between privacy and utility. Many utility variables, such as the cognitive health status of the speaker, are correlated with the privacy variable, such as the speaker identity, violating the independence assumption held by the disentanglement-based approaches, causing leakage of private information and the loss of useful information for downstream tasks. To tackle this challenge, we propose a general framework, DDPO-VC, for speaker de-identification through reinforcement learning-based post-training with diffusion models. Learning from reward signals combining knowledge from privacy-focused and utility-focused teachers, our method outperforms various strong \deid/ methods in both privacy preservation and cognitive utility on two commonly used dementia speech benchmarks. Please check out our code\footnote{\href{this https URL}{this https URL}} and demo\footnote{\href{this https URL}{this https URL}}.
#### Phonetically Explainable Speech Deepfake Detection
 - **Authors:** Manasi Chhibber, Jagabandhu Mishra, Tomi H. Kinnunen
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2606.15454

 - **Pdf link:** https://arxiv.org/pdf/2606.15454

 - **Abstract**
 Speech deepfake detection is predominantly treated as an opaque classification task where all temporal frames are aggregated equally. This ignores that different phonetic categories carry vastly different amounts of discriminative information. To address this, we propose a phoneme-guided cross-attention framework that transforms detection into an interpretable, phonetically grounded process. We factorize the spoofing posterior $P(\text{spoofed}\mid X, W)$, conditioned on the acoustic representation $X$ and the phonetic posteriorgram $W$. The resulting factorization can be written as $P(\text{spoofed} \mid X, W) = \sum_{i=1}^{M} w_i \cdot P(\text{spoofed} \mid X, Z = z_i)$, where $M$ denotes the number of phonetic classes, $P(\text{spoofed} \mid X, Z = z_i)$ is the spoofing probability for the $i$-th phonetic class $z_i$ conditioned on $X$, and each $w_i$ is the prevalence of phonetic class $z_i$ in the utterance. Our transformer-based architecture instantiates this through a cross-attention block in which phonetic queries selectively probe information in acoustic keys and values, with softmax-normalized pooling supplying explicit phone-presence weights. Unlike prior approaches that rely heavily on post-hoc explainability methods, our framework offers phonetic-explainability-by-design. We evaluate the framework on an LJSpeech-derived corpus, ASVspoof 2019 LA, and ASVspoof 5 Track 1. Per-phone importance rankings reveal that discriminative power concentrates on articulatory categories that generative models struggle to reproduce faithfully. Stops, fricatives, affricates, nasals, and silence-boundary closures rank most discriminative, while periodic vowels and semivowels rank lower. Beyond competitive performance, our model provides structural interpretability, yielding an inspectable per-articulatory category breakdown of the final verdict.
#### MambAdapter: Lightweight Mamba-Based Adapters for Parameter-Efficient Transfer Learning in Speech and Audio
 - **Authors:** Salman Hussain Ali, Umberto Cappellazzo, Mirco Ravanelli
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2606.15638

 - **Pdf link:** https://arxiv.org/pdf/2606.15638

 - **Abstract**
 Fine-tuning Transformer-based foundation models has become the dominant strategy for domain adaptation in audio and speech processing. To reduce the computational and memory costs of this process, parameter-efficient transfer learning (PETL) methods have been widely explored. Meanwhile, Mamba, a recent state-space model, has emerged as a promising alternative to Transformers for sequence modeling. In this work, we present MambAdapter, a parameter-efficient transfer learning approach that integrates Mamba into low-rank bottleneck adapters. Our design combines parameter sharing across adapters with the injection of a lightweight Mamba module, enabling more effective modeling of audio features. We demonstrate that MambAdapter matches or outperforms strong PETL baselines on four audio classification tasks and five speech recognition languages, even when operating under reduced parameter budgets.
#### Bridging the SEA Gap: An Initial Benchmark for Neural Audio Codec-Synthesized Speech Deepfakes in South-East Asian Languages
 - **Authors:** Orchid Chetia Phukan, Girish, Mohd Mujtaba Akhtar, Arun Balaji Buduru
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2606.15968

 - **Pdf link:** https://arxiv.org/pdf/2606.15968

 - **Abstract**
 Codecfakes (CFs) are a type of speech deepfakes generated through Audio Language Models (ALMs), with Neural Audio Codecs (NACs) forming the core mechanism for speech encoding and generation. CFs exhibit distributional characteristics that differ from vocoder-based deepfakes, causing detectors trained on vocoder data to generalize poorly to CFs detection. Although this has led to the development of CF detection benchmarks, existing resources are largely confined to English -- and to a limited extent Chinese -- leaving South-East Asian (SEA) languages unexplored. To bridge this gap, we introduce SEA-CF, the first large-scale benchmark for CF detection spanning multiple SEA languages, diverse speaker profiles, and a wide range of NAC architectures. SEA-CF is constructed by synthesizing publicly available real speech corpora. Our experiments show that state-of-the-art (SOTA) CF detectors trained on English-centric datasets fail to generalize to SEA speech due to language-specific phonetic structures, tonal variations, and rich prosodic diversity. We further conduct a comprehensive zero-shot and fine-tuned evaluation of recent SOTA ALMs on SEA-CF. Fine-tuning the ALMs improves performance, however, these are very large being impractical for real-world application due to their scale, particularly in low-resource and latency-constrained settings. To address this limitation, we propose a novel small-ALM, GARUDA tailored for CF detection, which delivers strong performance while remaining lightweight. Extensive evaluations demonstrate that the proposed Small-ALM outperforms strong end-to-end and ALM-based baselines, establishing a new, practical direction for robust CF detection in SEA languages and beyond.
#### Stabilizing Short Duration Speaker Verification through Neural Re-scoring with Hybrid Enrollment
 - **Authors:** Zhiqi Ai, Han Cheng, Shiyi Mu, Zhiyong Chen, Yongjin Zhou, Shugong Xu
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2606.16115

 - **Pdf link:** https://arxiv.org/pdf/2606.16115

 - **Abstract**
 Short-duration speaker verification (SDSV) is crucial for personalized keyword spotting, where test utterances are typically shorter than three seconds. Limited speech duration results in unstable speaker representations and increased sensitivity to noise and phoneme variations, thereby degrading performance. To investigate this issue, we construct VoxPhrase, a large-scale SDSV corpus automatically segmented from the VoxCeleb dataset. Our analysis shows that text-dependent (TD) enrollment is constrained by duration and yields unstable speaker representations. In contrast, although text-independent (TI) enrollment introduces content mismatch, its representations become more stable as the enrollment duration increases. Accordingly, we propose a hybrid-enrollment neural re-scoring framework that combines TD and TI enrollment and performs frame-level comparison via parallel cross-attention. Experiments on VoxPhrase demonstrate consistent improvements across multiple speaker models.
#### Towards Robust Generative Speech Enhancement Using Vector Quantisation-Based Neural Audio Codec
 - **Authors:** Haixin Zhao, Nilesh Madhu
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2606.16464

 - **Pdf link:** https://arxiv.org/pdf/2606.16464

 - **Abstract**
 This work investigates modelling strategies in continuous and discrete latent spaces in the vector quantisation (VQ)-based neural audio codec (NAC) speech enhancement (SE), along with the role of VQ regularisation. We propose cNAC-SE and dNAC-SE frameworks that predict continuous representations and discrete tokens in latent space, respectively. Theoretical analysis and visualisations in latent space are performed to exhibit their inherent modelling mechanisms. Experimental results show that the fully fine-tuned cNAC-SE model consistently outperforms all dNAC-SE variants across diverse test conditions and achieves leading performance among established generative approaches in DNS-MOS metrics. Comparison with the discriminative counterpart shows that VQ enhances robustness through an intrinsic effect of clean-prior-constrained regularisation, independent of discrete token processing. This highlights the transferable value of VQ regularisation to other continuous modelling methods.
#### Decoding while Adapting: Zero-Shot Online Speaker Adaptation via Audio-Textual Prompts for Elderly Speech Recognition
 - **Authors:** Chengxi Deng, Xurong Xie, Shujie Hu, Mengzhe Geng, Tianzi Wang, Youjun Chen, Huimeng Wang, Haoning Xu, Jiajun Deng, Xunying Liu
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2606.16539

 - **Pdf link:** https://arxiv.org/pdf/2606.16539

 - **Abstract**
 This paper proposes a novel cross-utterance audio-textual prompts based speaker adaptation approach for elderly speech recognition. It enables zero-shot, real-time adaptation to unseen speakers. Speech and text embeddings are extracted from the current and a few preceding utterances, before being fused in a cross-modal manner to produce compact speaker prompts that are more consistent than i/x-vectors and ECAPA-TDNN features. Experiments on the English DementiaBank Pitt and Cantonese JCCOCC MoCA elderly speech datasets suggest that the proposed online adaptation outperforms the speaker-independent (SI) model by statistically significant word error rate (WER) or character error rate (CER) reductions of 0.61% and 1.22% absolute (2.99% and 4.48% relative). Real-time factor (RTF) speed-up ratios of up to 9.83 times are obtained over offline batch-mode adaptation.
#### Confidence Score Guided Incremental and Speaker Adaptive Pseudo-Labeling for Semi-Supervised Elderly Speech Recognition
 - **Authors:** Chengxi Deng, Xurong Xie, Shujie Hu, Jiajun Deng, Mengzhe Geng, Youjun Chen, Huimeng Wang, Haoning Xu, Guinan Li, Xunying Liu
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2606.16546

 - **Pdf link:** https://arxiv.org/pdf/2606.16546

 - **Abstract**
 This paper proposes a novel confidence score guided incremental and speaker adaptive pseudo-labeling approach for semi-supervised elderly speech recognition. It facilitates higher-quality pseudo-label selection and progressive refinement, while also mitigating speaker heterogeneity. A confidence estimation module is designed to rank the reliability of untranscribed data, enabling a curriculum learning trajectory that progressively folds in unlabeled data subsets from high to low confidence. Speaker-specific characteristics are captured through speaker adaptive training with learnable prompts. Experiments on the English DementiaBank Pitt and Cantonese JCCOCC MoCA elderly speech datasets suggest that the proposed method outperforms the semi-supervised baseline using no confidence scores guided incremental or speaker adaptive pseudo-labeling by statistically significant word error rate (WER) or character error rate (CER) reductions of 1.45% and 2.27% absolute (6.21% and 6.98% relative).
#### CraBERT: Efficient Phoneme Encoder Pre-Training via Cascade Fusion of Subword Representations for Text-to-Speech
 - **Authors:** Dong Yang, Yuki Saito, Wataru Nakata, Hiroshi Saruwatari
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2606.16668

 - **Pdf link:** https://arxiv.org/pdf/2606.16668

 - **Abstract**
 This paper introduces CraBERT, a pre-trained phoneme encoder (PPEnc) designed for efficient pre-training in text-to-speech (TTS). CraBERT employs a cascade-fusion architecture and a subword-phoneme alignment algorithm to integrate representations from a pre-trained subword-level BERT into a phoneme-level BERT. This design provides prior word- and sentence-level information, reducing the amount of pre-training required by the phoneme encoder. Subjective listening evaluations show that CraBERT achieves MOS values comparable to existing PPEncs after approximately one epoch of pre-training, whereas the baselines in our comparison are pre-trained for approximately ten epochs. These results demonstrate that CraBERT can efficiently learn representations suitable for improving the perceived naturalness and prosody of synthesized speech.
#### LLM-Based Synthetic Ground Truth Generation for Audio-Based Emotion Classification via In-Context Learning
 - **Authors:** Qing Huang, Pooja Pol, Jianing Zhang
 - **Subjects:** Subjects:
Sound (cs.SD); Machine Learning (cs.LG); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2606.14784

 - **Pdf link:** https://arxiv.org/pdf/2606.14784

 - **Abstract**
 Understanding human states and interaction dynamics is a core goal of human-computer interaction (HCI). As interaction paradigms become more immersive, virtual reality (VR) has emerged as a powerful platform for studying collaborative work. In such settings, evaluating team collaboration states, including team performance and team resilience, requires continuous and reliable inference of latent team-level cognitive and affective states from multi-modal sensor data, such as speech signals. However, generating ground truth labels for these latent states remains challenging due to sensor-induced noise, contextual variability, and sparse expert annotations. Traditional self-reporting approaches provide only static and delayed measurements and are therefore insufficient for capturing dynamic team processes reflected in continuous speech data. In this work, we propose a large language model (LLM)-driven, agentic inference workflow for automated emotion-related synthetic ground truth generation from streaming speech data in multi-user VR environments. Leveraging the generalization capabilities of LLMs, we use In-Context Learning (ICL) with few-shot demonstrations of paired audio-based samples and their corresponding transcriptions. ICL tends to achieve task adaptation comparable to model fine-tuning while circumventing the computational overhead of parameter updates. To construct informative and robust in-context prompts, we adopt a retrieval-based selection strategy that dynamically identifies relevant audio demonstrations based on similarity in the acoustic feature space.
#### Unifying Acoustic Features and Text with Multimodal LLMs for Neurodegenerative Screening
 - **Authors:** Qingfeng Zhang, Yuanxiong Guo, Yanmin Gong
 - **Subjects:** Subjects:
Sound (cs.SD); Artificial Intelligence (cs.AI); Machine Learning (cs.LG); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2606.14788

 - **Pdf link:** https://arxiv.org/pdf/2606.14788

 - **Abstract**
 Voice-based screening offers a scalable and non-invasive way to assess neurodegenerative diseases such as Alzheimer's disease (AD) and Parkinson's disease (PD), but their staging remains challenging due to the difficulty of integrating heterogeneous data. This paper presents NeurMLLM, an efficient multimodal generative framework for neurodegenerative disease staging. NeurMLLM first encodes the spectrograms and Mel-frequency cepstral coefficients of audio data with vision transformers and projects their representations into the embedding space of a large language model (LLM), where they are concatenated with transcript and demographic instruction tokens as a single unified sequence. The LLM is then instruction-tuned via Low-Rank Adaptation using task prompts to autoregressively predict a constrained label token, enabling a generative classification. By evaluating on the Bridge2AI-Voice dataset for fine-grained staging of AD and PD, we observe that NeurMLLM achieves strong performance, consistently outperforming classical machine learning methods and existing LLM-based approaches. The results show the high potential of multimodal LLMs in neurodegenerative disease staging, improving staging accuracy and supporting accessible deployment.
#### Spectro-Temporal Interference Confounds Phase Encoding in Spatial Audio Foundation Models
 - **Authors:** Yuxuan Chen, Haoyuan Yu, Peize He
 - **Subjects:** Subjects:
Sound (cs.SD); Artificial Intelligence (cs.AI); Computation and Language (cs.CL); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2606.14820

 - **Pdf link:** https://arxiv.org/pdf/2606.14820

 - **Abstract**
 Recent spatial self supervised audio models achieve high performance on localization tasks, raising questions about their encoding of microsecond interaural phase fine structures. We propose a psychoacoustic benchmark based on the binaural masking level difference to evaluate this. Using an equalization cancellation baseline and a GCC PHAT positive control we evaluate nine frozen audio models spanning binaural SSL, monaural SSL, and neural audio codecs. Four monaural negative controls yield zero BMLD confirming binaural specificity. Two general purpose binaural SSL models exhibit minimal phase sensitivity while dedicated binaural spatial SSL models achieve BMLD comparable to the analytical baseline. Progressive physical ablations show that general purpose binaural SSL models rely on spectro temporal interference textures rather than cross channel phase computation. High detection rates in speech reflect a confounding reliance on broadband envelopes rather than genuine phase encoding.
#### An Empirical Study on Learning Latent Representations for Emotional Speech Synthesis
 - **Authors:** Vinh Dang Quang, Huy Ngo Quang
 - **Subjects:** Subjects:
Sound (cs.SD); Artificial Intelligence (cs.AI); Computation and Language (cs.CL); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2606.14922

 - **Pdf link:** https://arxiv.org/pdf/2606.14922

 - **Abstract**
 For the last couple of years, the field of speech synthesis has improved dramatically thanks to deep learning. There are more and more deep learning-based TTS systems developed to make it possible to produce voices with high intelligibility and naturalness. Meanwhile, controlling the expressiveness is yet a big deal, generating speech in different styles or manners has received a lot of attention from community recently. This paper aims to give our solutions to deal with the task emotional speech synthesis (ESS) at VLSP 2022 which allows to generate humanlike natural-sounding voice from a given input text with desired emotional expression. By integrating speaker embedding, prosody bottleneck into FastSpeech 2, our systems can promisingly generate emotional speech of a single speaker (Sub-task 1), transfer speaking styles from another speaker to the target speaker with neutral non-expressive data while retaining the target speaker's identity (Sub-task 2).
#### AP-GRPO: Anchor-Gated Phonetic Alignment with Policy Optimization for Pathological Speech Reconstruction
 - **Authors:** Pengfei Zhang, Hoang H Nguyen, Yutong Song, Wenjun Huang, Tahmid Imtiaz Imu, Henry Peng Zou, Jiang Wu, Honghui Xu, Amir M. Rahmani
 - **Subjects:** Subjects:
Sound (cs.SD); Artificial Intelligence (cs.AI); Multimedia (cs.MM); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2606.15540

 - **Pdf link:** https://arxiv.org/pdf/2606.15540

 - **Abstract**
 Pathological speech from patients with neurodegenerative and neuromotor disorders is often acoustically distorted and linguistically fragmented, making pathological speech reconstruction necessary to recover intended textual content from distorted and incomplete speech recordings. Crucially, such recordings are rarely uniformly degraded: some words or short phrases remain reliable and can serve as audible anchors for reconstructing the corrupted surrounding content. We introduce Anchor-gated Phonetic Group Relative Policy Optimization (AP-GRPO), a GRPO framework with phonetic reward that aligns speech language models (SLMs) through audible-anchor preservation and inter-anchor phonetic compatibility to the original speech signal. AP-GRPO consists of: (i) an anchor-gated reward that matches reliable audible anchors in clear regions; and (ii) an inter-anchor phonetic alignment reward that evaluates whether recovered contents are phonetically supported by the corresponding corrupted inter-anchor speech span. Across four disease conditions, AP-GRPO improves faithful speech reconstruction, and the learned anchor constraint automatically adapts to each condition and thus reveals interpretable disease-specific profiles: conditions with severe articulatory degradation require stronger anchor enforcement, whereas milder impairment or linguistically impaired conditions rely more on phonetic alignment for inter-anchor recovery.
#### NVMOS: Non-Verbal Vocalization Quality Assessment in Speech
 - **Authors:** Jialong Mai, Jinxin Ji, Xiaofen Xing, Wencui Liu, Xiangmin Xu
 - **Subjects:** Subjects:
Sound (cs.SD); Artificial Intelligence (cs.AI); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2606.15888

 - **Pdf link:** https://arxiv.org/pdf/2606.15888

 - **Abstract**
 Non-verbal vocalizations (NVs), such as laughter, sighs, and coughs, are important acoustic cues for emotion and intent. Existing speech quality assessment methods typically focus on overall naturalness, while non-verbal TTS evaluations mainly examine whether a target NV appears with the correct type and position. However, the perceptual quality of NV events themselves remains underexplored. To address this gap, we construct an NV-MOS dataset containing outputs from multiple NV-TTS systems and naturally occurring NV samples, with ratings collected from three acoustic experts on a perceptual quality scale. We further analyze audio-capable multimodal large language models such as Gemini and find clear inconsistencies between their scores and expert ratings. These results suggest that general-purpose multimodal models cannot reliably replace human judgments for NV quality assessment. We then propose NVMOS, to our knowledge the first model that can reliably predict the perceptual quality of NV events in speech. Experimental results show that, with a local NV-event focusing module, NVMOS reaches expert-level or stronger agreement with human MOS.
#### ArtBoost: Synthetic Articulatory Data Augmentation for Acoustic-to-Articulatory Inversion
 - **Authors:** Hyung Kyu Kim, Byungchan Hwang, Hak Gu Kim
 - **Subjects:** Subjects:
Sound (cs.SD); Artificial Intelligence (cs.AI); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2606.16327

 - **Pdf link:** https://arxiv.org/pdf/2606.16327

 - **Abstract**
 Recent acoustic-to-articulatory inversion (AAI) models rely on electromagnetic articulography (EMA) data, which are costly and limited in scale. To address this limitation, we propose \textit{ArtBoost}, a novel data augmentation strategy that leverages large-scale speech--mesh datasets originally developed for speech-driven 3D facial animation to improve AAI under limited EMA supervision. \textit{ArtBoost} extracts pseudo articulatory trajectories from visible facial anchors and uses them for pre-training before fine-tuning on real EMA data. Experiments show consistent improvements in PCC and RMSE. Trajectory analyses confirm that the pseudo articulatory signals reflect physically meaningful visible articulatory dynamics. Additional evaluations across different AAI architectures demonstrate stable performance gains, indicating that \textit{ArtBoost} can be integrated into diverse AAI models. These results suggest that speech--mesh data provide an effective and scalable source of articulatory supervision for AAI. Project page: this https URL
#### Joycent: Diffusion-based Accent TTS without Accented Phone Prediction
 - **Authors:** Xintong Wang, Ye Wang
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2606.16417

 - **Pdf link:** https://arxiv.org/pdf/2606.16417

 - **Abstract**
 Accent text-to-speech (TTS) aims to synthesize speech with target accents. Existing accent TTS systems typically rely on a two-stage pipeline that first converts standard phone sequences into accented phone sequences and then synthesizes accented speech. However, such approaches suffer from error accumulation and require paired standard-accented phone sequence data, which is often limited in practice. Moreover, text-based accented phone representations are insufficient to model acoustic accent characteristics such as prosody and rhythm. In this work, we propose Joycent, a diffusion-based accent TTS model that synthesizes accented speech directly from standard phone sequences and speech references without accented phone prediction. Joycent integrates accent and speaker representations through conditional layer normalization (CLN) in the text encoder. We introduce WhisAID, a Mandarin accent identification model trained on accented Mandarin speech to extract accent representations. Experimental results show that Joycent improves accentedness while preserving speaker identity compared with baseline systems. We release our code and demos at: this https URL.
#### Probing Low Frame Rate Degradation in Neural Audio Codecs
 - **Authors:** Alex Gichamba, Moise Busogi
 - **Subjects:** Subjects:
Sound (cs.SD); Artificial Intelligence (cs.AI); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2606.16969

 - **Pdf link:** https://arxiv.org/pdf/2606.16969

 - **Abstract**
 Low frame rates in neural audio codecs are attractive for autoregressive speech synthesis, where the generation cost scales linearly with the sequence length. Recent work has demonstrated that codecs can operate at 12.5 Hz and below, but the mechanisms underlying low frame rate degradation remain insufficiently understood. We investigate these mechanisms through a controlled frame rate ablation. We reproduce a quality cliff at 6.25 Hz reported in previous works and evaluate candidate explanations: phonemic collisions and codebook saturation, neither of which shows evidence of a fundamental barrier. The cliff is instead caused by suboptimal training configuration: fixed clip duration during training yields too few tokens at low frame rates, starving the decoder of inter-token context. Once corrected, WER degrades smoothly with phonemic load down to 3.1 Hz and 1.6 Hz, suggesting the inference-time efficiency gains of low frame rate codecs are more accessible than previously assumed.


by Zyzzyva0381 (Windy). 


2026-06-16
