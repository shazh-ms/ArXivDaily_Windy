# Showing new listings for Wednesday, 3 June 2026
Auto update papers at about 2:30am UTC (10:30am Beijing time) every weekday.


阅读 `Usage.md`了解如何使用此repo实现个性化的Arxiv论文推送

See `Usage.md` for instructions on how to personalize the repo. 


Keyword list: ['text-to-speech', 'text to speech', 'tts', 'LLM-based', 'speech', 'voice']


Excluded: []


### Today: 11papers 
#### FSA-GRPO: Teaching Auditory LLMs to Use Few-shot Demonstrations
 - **Authors:** Haolong Zheng, Siyin Wang, Xulin Fan, Zengrui Jin, Mark Hasegawa-Johnson
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Artificial Intelligence (cs.AI); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2606.02615

 - **Pdf link:** https://arxiv.org/pdf/2606.02615

 - **Abstract**
 Few-shot prompting provides an effective way to adapt auditory large language models to low-resource tasks such as children's speech recognition. However, most auditory large language models are not explicitly trained to perform inference in this demonstration-conditioned format, limiting the extent to which they can benefit from few-shot prompting. To address this limitation, we introduce Few-Shot Aware GRPO (FSA-GRPO), an RL-based post-training recipe that uses a specially designed reward to encourage the model to leverage few-shot demonstrations, thereby strengthening its few-shot adaptation ability. Notably, training with only high-resource adult ASR data improves the model's general few-shot adaptation ability, yielding gains not only in children's speech recognition but also in speech translation and audio understanding. We further study data selection and auxiliary reward weighting to identify an effective training recipe. Our experiments show that when in-domain data are unavailable or cannot be used for training, FSA-GRPO is more effective than direct tuning on related out-of-domain data.
#### Wavelet as Tokenizer: Preliminary Results on a Shared Wavelet Token Schema for Natural Signals
 - **Authors:** Shenghao Ding
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Artificial Intelligence (cs.AI); Computer Vision and Pattern Recognition (cs.CV); Machine Learning (cs.LG); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2606.02631

 - **Pdf link:** https://arxiv.org/pdf/2606.02631

 - **Abstract**
 This paper studies whether audio, images, and video can share a common wavelet token schema rather than relying on separate modality-specific latent grids. It introduces a preliminary continuous-token model built around a one-level Haar DWT/IDWT frontend, a shared coefficient-token layout, optional structural metadata, lightweight modality value adapters, and a shared token-wise encoder-decoder trunk. On Speech Commands, EuroSAT RGB, and DAVIS 2017 data, a dense shared model reaches 39.92 dB audio, 29.37 dB image, and 23.93 dB video PSNR. A matched-rate sweep under continuous latent scalar budgets indicates that the visual gains are not explained solely by latent capacity, while also showing that additive metadata embeddings are not a universal source of improvement. Finally, fixed-rate energy selection provides a strong non-parametric baseline: energy_global improves average PSNR over uniform selection by 16.73 dB for audio, 16.90 dB for images, and 15.86 dB for video under compressed keep ratios. Masked sparse training reaches 34.45 dB video PSNR with 50% of dense tokens. The results support a unified wavelet token schema and sparse token interface, while stopping short of establishing a universal discrete vocabulary.
#### SVHalluc: Benchmarking Speech-Vision Hallucination in Audio-Visual Large Language Models
 - **Authors:** Chenshuang Zhang, Kyeong Seon Kim, Chengxin Liu, Tae-Hyun Oh
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Artificial Intelligence (cs.AI); Computer Vision and Pattern Recognition (cs.CV); Machine Learning (cs.LG); Multimedia (cs.MM); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2606.02642

 - **Pdf link:** https://arxiv.org/pdf/2606.02642

 - **Abstract**
 Despite the success of audio-visual large-language models (LLMs), they can produce plausible but ungrounded outputs, termed hallucination. Existing benchmarks focus on environmental sounds (e.g., dog barking) to indicate event occurrence. In contrast, human speech carries fundamentally different, rich semantics and temporal structures, yet it remains unexplored whether current models can accurately align speech content with corresponding visual signals. In this work, we show that speech content can induce hallucinations in audio-visual LLMs. To systematically study this, we introduce SVHalluc, the first comprehensive benchmark for evaluating speech-vision hallucination in audio-visual LLMs. Our benchmark diagnoses speech-vision hallucinations from two critical and complementary aspects: semantic and temporal. Experimental results demonstrate that state-of-the-art open-source audio-visual LLMs struggle with aligning speech content with corresponding visual signals, with a near-random accuracy on multiple tasks. In contrast, Gemini 2.5 Pro significantly outperforms the open-source models. Our analysis suggests that their failures stem from limited ability in cross-modality understanding, despite strong performance in single-modality perception. Our work uncovers a new and fundamental limitation of current audio-visual LLMs and highlights the need for speech-grounded video comprehension. Project page: this https URL.
#### A Comparison of Generative and Discriminative Methods for Speech Enhancement: Robustness, Complexity, and Hallucination
 - **Authors:** Shrishti Saha Shetu, Emanuël A. P. Habets, Andreas Brendel
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2606.02913

 - **Pdf link:** https://arxiv.org/pdf/2606.02913

 - **Abstract**
 In this study, we conduct a comprehensive comparative analysis of generative and discriminative deep learning-based speech enhancement methods, specifically in noise reduction tasks. Our investigation focuses on evaluating their effectiveness under high and low signal-to-noise ratio conditions, considering both matched and mismatched training scenarios. We further investigate the impact of training data volume, model convergence speed, and interpret the performance differences in terms of objective results for the considered training paradigms. Additionally, we compare the complexity-performance trade-off and the practical viability of these approaches. To further strengthen the evaluation, we study the hallucination characteristics of generative approaches in terms of word error rate and phoneme similarity. The insights derived from this study provide empirical evidence to assist researchers and practitioners in understanding whether the perceptual gains of different approaches justify their computational cost in practical applications.
#### AnyAudio-Judge: A Dynamic Rubric-Based Benchmark and Evaluator for Audio Instruction Following
 - **Authors:** Haitao Li, Tian Tan, Yuguang Yang, Shan Yang, Xie Chen
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Artificial Intelligence (cs.AI); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2606.03116

 - **Pdf link:** https://arxiv.org/pdf/2606.03116

 - **Abstract**
 The rapid advancement of instruction-guided audio generation has highlighted the critical need for robust alignment evaluation. Current automated evaluation methods heavily rely on holistic scoring from general-purpose large language models, which struggle to decouple complex instructions, lack interpretability, and fail to capture fine-grained attribute mismatches. To address this, we introduce a novel dynamic rubric-based evaluation paradigm that adaptively decomposes complex audio captions into a variable number of independent, verifiable binary rubric items. To rigorously benchmark this capability, we propose the AnyAudio-Judge Bench, a comprehensive, bilingual benchmark comprising 7,920 meticulously curated samples across four diverse audio domains (speech, sound, music, and mixed), featuring deliberately constructed hard negatives. Furthermore, we construct a large-scale corpus of 105K samples with explicit Chain-of-Thought (CoT) rationales to train our dedicated evaluator, the AnyAudio-Judge model. By employing a training pipeline that combines Supervised Fine-Tuning (SFT) and Group Relative Policy Optimization (GRPO), our model successfully aligns its reasoning paths with the rubric-based scoring mechanism. Extensive experiments demonstrate that AnyAudio-Judge not only significantly enhances zero-shot alignment detection compared to state-of-the-art baselines, but also provides precise and interpretable reward signals that substantially improve instruction alignment in downstream reinforcement learning for audio generation.
#### SpeakerCard-1M: An Evidence-Grounded Speaker Card Corpus for In-the-Wild Speaker Verification
 - **Authors:** Junyi Peng, Oldřich Plchot, Xiao Song, Dading Chong, Lichun Fan, Hang Su, Themos Stafylakis, Junjie Li, Kong Aik Lee, Shuai Wang, Jan Černocký
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2606.03283

 - **Pdf link:** https://arxiv.org/pdf/2606.03283

 - **Abstract**
 Modern speaker verification (SV) systems rely on speaker embeddings that are effective but difficult to interpret or query in natural language. Most existing speech-text corpora target controllable synthesis or utterance-level captioning, and provide limited speaker-level supervision for in-the-wild speaker recognition. This paper introduces SpeakerCard-1M, a bilingual speaker-centric resource for evidence-grounded SV, derived from VoxCeleb1/2 and CN-Celeb1/2, where the "-1M" suffix refers to the 1.78M utterance-level captions contained in the release. We adopt a tool-first, LLM-last approach: ten acoustic probes produce field-level evidence, the evidence is aggregated into speaker profiles under a schema that separates relatively stable traits from utterance-level states, and bilingual Speaker Cards are rendered by a constrained LLM that sees only the structured fields. The release includes 56.7K Speaker Card records over 10.2K speakers, 1.78M utterance-level captions, and speaker-ID-disjoint hard-negative triplets. We further define two SV-oriented cross-modal protocols, bidirectional Speaker-Text Retrieval (T2S-R / S2T-R) and Attribute-Conditioned Verification (AC-Verify), and compare a dual-encoder baseline against recent audio language models under a zero-shot forced-choice setting. Joint audio-text training increases VoxCeleb1-O EER by 0.31% absolute over the audio-only baseline. Under a style-symmetric LLM-generated counterfactual protocol, eight recent audio language models (7B-30B+ parameters, both open- and closed-source) score 49-77% on pitch-level AC-Verify under two-way forced choice, compared with 88.66% reached by our dual encoder.
#### WavTTS: Towards High-Quality Zero-Shot TTS via Direct Raw Waveform Modeling
 - **Authors:** Wenxi Chen, Dongya Jia, Yushen Chen, Zhikang Niu, Yuzhe Liang, Xiquan Li, Ruiqi Yan, Ziyang Ma, Guanrou Yang, Sanyuan Chen, Yue Wang, Zhuo Chen, Kai Yu, Xie Chen
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2606.03455

 - **Pdf link:** https://arxiv.org/pdf/2606.03455

 - **Abstract**
 Recently, diffusion models operating on VAE latents or mel-spectrograms have become the dominant paradigm for zero-shot TTS. Although these compressed representations improve generation efficiency, they inevitably suffer from information loss and non-end-to-end training. Theoretically, directly modeling raw waveforms circumvents these issues; however, this direction remains underexplored and is often deemed difficult due to the extremely long sequence length of audio signals. To overcome this, we propose WavTTS, the first raw waveform generative TTS model that substantially narrows the gap with latent-space generative models. Built upon the flow matching with Diffusion Transformer (DiT), WavTTS directly models speech waveforms via a simple patchification strategy, while integrating multi-scale mel-spectrogram supervision to provide perceptual guidance during training. Furthermore, we investigate the impact of prediction targets and noise scheduling in waveform diffusion, and develop an effective schedule design to improve generation quality. Evaluations on open-source benchmarks demonstrate that WavTTS closely approaches the performance of current state-of-the-art latent generative zero-shot TTS models, while substantially outperforming previous end-to-end speech generation models. Our findings demonstrate the feasibility of scaling diffusion-based TTS directly in the waveform space, opening a new direction for end-to-end speech generation.
#### SegTune: Structured and Fine-Grained Control for Song Generation
 - **Authors:** Yuejiao Wang, Zihao Ji, Pengfei Cai, Xu Li, Haorui Zheng, Zewen Song, Zhongliang Liu, Chen Zhang, Pengfei Wan
 - **Subjects:** Subjects:
Sound (cs.SD); Artificial Intelligence (cs.AI); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2606.02638

 - **Pdf link:** https://arxiv.org/pdf/2606.02638

 - **Abstract**
 Recent advances in neural song generation have enabled high-quality synthesis from lyrics and global textual prompts. However, most systems fail to model temporally varying attributes of songs, severely limiting fine-grained control over musical structure and dynamics. To address this, we propose SegTune, a Diffusion Transformer-based framework enabling structured and fine-grained controllability by allowing users or large language models (LLMs) to specify local musical descriptions aligned to song segments. These segment prompts are temporally broadcast to corresponding time windows, while global prompts ensure stylistic coherence. To support precise lyric-to-music alignment, we introduce an LLM-based duration predictor that autoregressively generates sentence-level timestamps in LyRiCs format. We further construct a large-scale data pipeline for high-quality song collection with aligned lyrics and prompts, and propose new metrics to evaluate segment alignment and vocal consistency. Experiments demonstrate that SegTune outperforms existing baselines in both musicality and controllability. Visit our project page (this https URL) for codes and more generated songs.
#### EntangleCodec: A Unified Discrete Audio Tokenizer via Semantic-Acoustic Entanglement
 - **Authors:** Hui Li, Yangfan Gao, Junlin Shang, Changhao Jiang, Tao Gui, Qi Zhang, Xuanjing Huang
 - **Subjects:** Subjects:
Sound (cs.SD); Artificial Intelligence (cs.AI); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2606.02739

 - **Pdf link:** https://arxiv.org/pdf/2606.02739

 - **Abstract**
 Audio tokenizers serve as the discrete interface between continuous audio and Audio Language Models (ALMs), but existing tokenizers often struggle to support both understanding and generation. Reconstruction-oriented codecs preserve acoustic fidelity but lack rich semantics, while semantic-aware tokenizers typically rely on separate semantic and acoustic streams, introducing redundancy or misalignment. We propose \textbf{EntangleCodec}, a unified discrete audio tokenizer that learns caption-aligned semantic-acoustic representations before quantization. By aligning audio with rich captions rather than ASR transcripts, EntangleCodec captures linguistic content, speaker identity, emotion, prosody, and acoustic scenes within a compact token stream. A flow-matching diffusion decoder further enables high-quality reconstruction across speech, music, and general audio. EntangleCodec achieves reconstruction quality competitive with specialized codecs, outperforms all codec-based baselines on audio understanding by up to \textbf{+7.4\%} on MMAR, and supports both TTS and TTA generation in a unified framework. Furthermore, EntangleCodec-based audio language models demonstrate strong scaling behavior: even at \textit{0.6B} parameters, the model surpasses specialized continuous-representation LLMs with over \textit{13B} parameters across three benchmarks using \textbf{22$\times$} fewer parameters; scaling to \textit{8B} further establishes new state-of-the-art results on MMAR, highlighting that representation quality is as critical as model scale in audio language modeling. Code and model weights are available at this https URL.
#### Benchmarking Speech-to-Speech Translation Models
 - **Authors:** Alkis Koudounas, Hayato Futami, Quentin Jodelet, Osamu Take, Shinji Watanabe, Emiru Tsunoo
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2606.03241

 - **Pdf link:** https://arxiv.org/pdf/2606.03241

 - **Abstract**
 Speech-to-speech translation (S2ST) has advanced rapidly, but offline evaluation lacks a unified protocol: studies report non-overlapping metric subsets, preventing direct comparisons. We introduce COMPASS, a unified and reproducible benchmarking framework integrating 46 metrics across eight dimensions, and deploy it on 1,248 model-language configurations from FLEURS and CVSS, spanning cascaded and end-to-end architectures over ten language pairs. Architectures exhibit complementary strengths: best-vs-worst gaps exceed 30\% on naturalness and speaker preservation but remain within a few points on translation quality, so single-metric rankings systematically misrepresent system quality. Correlation filtering reduces 46 metrics to 10 per direction, with three axes requiring different metrics across X$\to$EN and EN$\to$X (e.g., TER/UTMOS vs. ChrF++/NISQA-MOS); these subsets preserve rankings (Spearman's $\rho>0.80$) while cutting evaluation time by $\approx 2.5\times$. Human validation across dubbing, podcasts, and medical domains shows standalone MOS predictors fail to predict listener preference, while top domain-specific metrics correlate with human judgment ($\rho \geq 0.90$). We release COMPASS as a foundation for domain-aware S2ST evaluation.
#### Efficient ASR Training with Conversations that Never Happened
 - **Authors:** Máté Gedeon, Péter Mihajlik
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Artificial Intelligence (cs.AI); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2606.03957

 - **Pdf link:** https://arxiv.org/pdf/2606.03957

 - **Abstract**
 Conversational ASR for lower-resource languages and niche domains is limited by the scarcity of domain-matched multi-speaker training data. We propose an augmentation pipeline that generates scenario-level dialogues with participant metadata, maps speaker attributes to TTS voice profiles, and assembles synthesized utterances into speaker-aware simulated conversations. We evaluated five LLM families under single-generator, fixed-budget mixture, and scale-up settings using the same FastConformer-Large training recipe for each one. We ran comprehensive evaluations on the Hungarian BEA-Dialogue benchmark corpus, with the method itself being applicable to any language given the resources for each component. The results show that synthetic conversations consistently improve speech recognition performance, but generator choice and data composition strongly affect the gains. Our largest training configuration, using only 67 hours of real conversations and 636 hours of simulated data, achieves better performance on the evaluation benchmark than a zero-shot model trained on 2700 hours of Hungarian speech. These findings indicate that LLM-generated conversational data synthesized with TTS is a practical complement to real conversational corpora for speech model training.


by Zyzzyva0381 (Windy). 


2026-06-03
