# Showing new listings for Monday, 27 April 2026
Auto update papers at about 2:30am UTC (10:30am Beijing time) every weekday.


阅读 `Usage.md`了解如何使用此repo实现个性化的Arxiv论文推送

See `Usage.md` for instructions on how to personalize the repo. 


Keyword list: ['text-to-speech', 'text to speech', 'tts', 'LLM-based', 'speech', 'voice']


Excluded: []


### Today: 4papers 
#### Advancing automatic speech recognition using feature fusion with self-supervised learning features: A case study on Fearless Steps Apollo corpus
 - **Authors:** Szu-Jui Chen, John H.L. Hansen
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2604.22203

 - **Pdf link:** https://arxiv.org/pdf/2604.22203

 - **Abstract**
 Using self-supervised learning (SSL) models has significantly improved performance for downstream speech tasks, surpassing the capabilities of traditional hand-crafted features. This study investigates the amalgamation of SSL models, with the aim to leverage both their individual strengths and refine extracted features to achieve improved speech recognition models for naturalistic scenarios. Our research investigates the massive naturalistic Fearless Steps (FS) APOLLO resource, with particular focus on the FS Challenge (FSC) Phase-4 corpus, providing the inaugural analysis of this dataset. Additionally, we incorporate the CHiME-6 dataset to evaluate performance across diverse naturalistic speech scenarios. While exploring previously proposed Feature Refinement Loss and fusion methods, we found these methods to be less effective on the FSC Phase-4 corpus. To address this, we introduce a novel deep cross-attention (DCA) fusion method, designed to elevate performance, especially for the FSC Phase-4 corpus. Our objective is to foster creation of superior FS APOLLO community resources, catering to the diverse needs of researchers across various disciplines. The proposed solution achieves an absolute +1.1% improvement in WER, providing effective meta-data creation for the massive FS APOLLO community resource.
#### UniSonate: A Unified Model for Speech, Music, and Sound Effect Generation with Text Instructions
 - **Authors:** Chunyu Qiang, Xiaopeng Wang, Kang Yin, Yuzhe Liang, Yuxin Guo, Teng Ma, Ziyu Zhang, Tianrui Wang, Cheng Gong, Yushen Chen, Ruibo Fu, Chen Zhang, Longbiao Wang, Jianwu Dang
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Artificial Intelligence (cs.AI); Computation and Language (cs.CL); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2604.22209

 - **Pdf link:** https://arxiv.org/pdf/2604.22209

 - **Abstract**
 Generative audio modeling has largely been fragmented into specialized tasks, text-to-speech (TTS), text-to-music (TTM), and text-to-audio (TTA), each operating under heterogeneous control paradigms. Unifying these modalities remains a fundamental challenge due to the intrinsic dissonance between structured semantic representations (speech/music) and unstructured acoustic textures (sound effects). In this paper, we introduce UniSonate, a unified flow-matching framework capable of synthesizing speech, music, and sound effects through a standardized, reference-free natural language instruction interface. To reconcile structural disparities, we propose a novel dynamic token injection mechanism that projects unstructured environmental sounds into a structured temporal latent space, enabling precise duration control within a phoneme-driven Multimodal Diffusion Transformer (MM-DiT). Coupled with a multi-stage curriculum learning strategy, this approach effectively mitigates cross-modal optimization conflicts. Extensive experiments demonstrate that UniSonate achieves state-of-the-art performance in instruction-based TTS (WER 1.47%) and TTM (SongEval Coherence 3.18), while maintaining competitive fidelity in TTA. Crucially, we observe positive transfer, where joint training on diverse audio data significantly enhances structural coherence and prosodic expressiveness compared to single-task baselines. Audio samples are available at this https URL.
#### DM-ASR: Diarization-aware Multi-speaker ASR with Large Language Models
 - **Authors:** Li Li, Ming Cheng, Weixin Zhu, Yannan Wang, Juan Liu, Ming Li
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2604.22467

 - **Pdf link:** https://arxiv.org/pdf/2604.22467

 - **Abstract**
 Multi-speaker automatic speech recognition (ASR) aims to transcribe conversational speech involving multiple speakers, requiring the model to capture not only what was said, but also who said it and sometimes when it was spoken. Recent Speech-LLM approaches have shown the potential of unified modeling for this task, but jointly learning speaker attribution, temporal structure, and lexical recognition remains difficult and data-intensive. At the current stage, leveraging reliable speaker diarization as an explicit structural prior provides a practical and efficient way to simplify this task. To effectively exploit such priors, we propose DM-ASR, a diarization-aware multi-speaker ASR framework that reformulates the task as a multi-turn dialogue generation process. Given an audio chunk and diarization results, DM-ASR decomposes transcription into a sequence of speaker- and time-conditioned queries, each corresponding to one speaker in one time segment. This formulation converts multi-speaker recognition into a series of structured sub-tasks, explicitly decoupling speaker-temporal structure from linguistic content and enabling effective integration of diarization cues with the reasoning capability of large language models. We further introduce an optional word-level timestamp prediction mechanism that interleaves word and timestamp tokens, yielding richer structured outputs and better transcription quality. Our analysis shows that diarization systems provide more reliable speaker identities and segment-level boundaries, while LLMs excel at modeling linguistic content and long-range dependencies, demonstrating their complementary strengths. Experiments on Mandarin and English benchmarks show that the proposed approach achieves strong performance with relatively small models and training data, while remaining competitive with or outperforming existing unified approaches.
#### TTS-PRISM: A Perceptual Reasoning and Interpretable Speech Model for Fine-Grained Diagnosis
 - **Authors:** Xi Wang, Jie Wang, Xingchen Song, Baijun Song, Jingran Xie, Jiahe Shao, Zijian Lin, Di Wu, Meng Meng, Jian Luan, Zhiyong Wu
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2604.22225

 - **Pdf link:** https://arxiv.org/pdf/2604.22225

 - **Abstract**
 While generative text-to-speech (TTS) models approach human-level quality, monolithic metrics fail to diagnose fine-grained acoustic artifacts or explain perceptual collapse. To address this, we propose TTS-PRISM, a multi-dimensional diagnostic framework for Mandarin. First, we establish a 12-dimensional schema spanning stability to advanced expressiveness. Second, we design a targeted synthesis pipeline with adversarial perturbations and expert anchors to build a high-quality diagnostic dataset. Third, schema-driven instruction tuning embeds explicit scoring criteria and reasoning into an efficient end-to-end model. Experiments on a 1,600-sample Gold Test Set show TTS-PRISM outperforms generalist models in human alignment. Profiling six TTS paradigms establishes intuitive diagnostic flags that reveal fine-grained capability differences. TTS-PRISM is open-source, with code and checkpoints at this https URL.


by Zyzzyva0381 (Windy). 


2026-04-27
