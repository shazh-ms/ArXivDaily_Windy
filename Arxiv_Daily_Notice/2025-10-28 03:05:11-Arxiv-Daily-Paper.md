# Showing new listings for Tuesday, 28 October 2025
Auto update papers at about 2:30am UTC (10:30am Beijing time) every weekday.


阅读 `Usage.md`了解如何使用此repo实现个性化的Arxiv论文推送

See `Usage.md` for instructions on how to personalize the repo. 


Keyword list: ['text-to-speech', 'text to speech', 'tts', 'LLM-based', 'speech', 'voice']


Excluded: []


### Today: 11papers 
#### Bridging the Perceptual - Statistical Gap in Dysarthria Assessment: Why Machine Learning Still Falls Short
 - **Authors:** Krishna Gurugubelli
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Machine Learning (cs.LG)
 - **Arxiv link:** https://arxiv.org/abs/2510.22237

 - **Pdf link:** https://arxiv.org/pdf/2510.22237

 - **Abstract**
 Automated dysarthria detection and severity assessment from speech have attracted significant research attention due to their potential clinical impact. Despite rapid progress in acoustic modeling and deep learning, models still fall short of human expert performance. This manuscript provides a comprehensive analysis of the reasons behind this gap, emphasizing a conceptual divergence we term the ``perceptual-statistical gap''. We detail human expert perceptual processes, survey machine learning representations and methods, review existing literature on feature sets and modeling strategies, and present a theoretical analysis of limits imposed by label noise and inter-rater variability. We further outline practical strategies to narrow the gap, perceptually motivated features, self-supervised pretraining, ASR-informed objectives, multimodal fusion, human-in-the-loop training, and explainability methods. Finally, we propose experimental protocols and evaluation metrics aligned with clinical goals to guide future research toward clinically reliable and interpretable dysarthria assessment tools.
#### UltraVoice: Scaling Fine-Grained Style-Controlled Speech Conversations for Spoken Dialogue Models
 - **Authors:** Wenming Tu, Guanrou Yang, Ruiqi Yan, Wenxi Chen, Ziyang Ma, Yipeng Kang, Kai Yu, Xie Chen, Zilong Zheng
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Computation and Language (cs.CL)
 - **Arxiv link:** https://arxiv.org/abs/2510.22588

 - **Pdf link:** https://arxiv.org/pdf/2510.22588

 - **Abstract**
 Spoken dialogue models currently lack the ability for fine-grained speech style control, a critical capability for human-like interaction that is often overlooked in favor of purely functional capabilities like reasoning and question answering. To address this limitation, we introduce UltraVoice, the first large-scale speech dialogue dataset engineered for multiple fine-grained speech style control. Encompassing over 830 hours of speech dialogues, UltraVoice provides instructions across six key speech stylistic dimensions: emotion, speed, volume, accent, language, and composite styles. Fine-tuning leading models such as SLAM-Omni and VocalNet on UltraVoice significantly enhances their fine-grained speech stylistic controllability without degrading core conversational abilities. Specifically, our fine-tuned models achieve improvements of 29.12-42.33% in Mean Opinion Score (MOS) and 14.61-40.09 percentage points in Instruction Following Rate (IFR) on multi-dimensional control tasks designed in the UltraVoice. Moreover, on the URO-Bench benchmark, our fine-tuned models demonstrate substantial gains in core understanding, reasoning, and conversational abilities, with average improvements of +10.84% on the Basic setting and +7.87% on the Pro setting. Furthermore, the dataset's utility extends to training controllable Text-to-Speech (TTS) models, underscoring its high quality and broad applicability for expressive speech synthesis. The complete dataset and model checkpoints are available at: this https URL.
#### Mitigating Attention Sinks and Massive Activations in Audio-Visual Speech Recognition with LLMS
 - **Authors:** Anand, Umberto Cappellazzo, Stavros Petridis, Maja Pantic
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Computer Vision and Pattern Recognition (cs.CV); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2510.22603

 - **Pdf link:** https://arxiv.org/pdf/2510.22603

 - **Abstract**
 Large language models (LLMs) have recently advanced auditory speech recognition (ASR), visual speech recognition (VSR), and audio-visual speech recognition (AVSR). However, understanding of their internal dynamics under fine-tuning remains limited. In natural language processing, recent work has revealed attention sinks, tokens that attract disproportionately high attention, and associated massive activations in which some features of sink tokens exhibit huge activation in LLMs. In this work, we are the first to study these phenomena in multimodal speech recognition. Through a detailed analysis of audio-visual LLMs, we identify attention sinks and massive activations not only at the BOS token but also at intermediate low-semantic tokens across ASR, VSR, and AVSR. We show that massive activations originate in the MLP layers and correspond to fixed feature indices across all sink tokens. We further show that intermediate sink tokens exhibit high cosine similarity to the BOS token, thereby amplifying attention and activation. Building on these insights, we introduce a simple decorrelation loss that reduces cosine similarity between BOS and other tokens, effectively mitigating intermediate sinks and massive activations. Furthermore, our method improves word error rate (WER) under high audio-visual feature downsampling while remaining stable at lower downsampling rates.
#### HyBeam: Hybrid Microphone-Beamforming Array-Agnostic Speech Enhancement for Wearables
 - **Authors:** Yuval Bar Ilan (1), Boaz Rafaely (1), Vladimir Tourbabin (2) ((1) School of Electrical and Computer Engineering, Ben-Gurion University of the Negev, Beer-Sheva, Israel (2) Reality Labs Research, Meta, Redmond, WA, USA)
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Signal Processing (eess.SP)
 - **Arxiv link:** https://arxiv.org/abs/2510.22637

 - **Pdf link:** https://arxiv.org/pdf/2510.22637

 - **Abstract**
 Speech enhancement is a fundamental challenge in signal processing, particularly when robustness is required across diverse acoustic conditions and microphone setups. Deep learning methods have been successful for speech enhancement, but often assume fixed array geometries, limiting their use in mobile, embedded, and wearable devices. Existing array-agnostic approaches typically rely on either raw microphone signals or beamformer outputs, but both have drawbacks under changing geometries. We introduce HyBeam, a hybrid framework that uses raw microphone signals at low frequencies and beamformer signals at higher frequencies, exploiting their complementary strengths while remaining highly array-agnostic. Simulations across diverse rooms and wearable array configurations demonstrate that HyBeam consistently surpasses microphone-only and beamformer-only baselines in PESQ, STOI, and SI-SDR. A bandwise analysis shows that the hybrid approach leverages beamformer directivity at high frequencies and microphone cues at low frequencies, outperforming either method alone across all bands.
#### Adapting Speech Foundation Models with Large Language Models for Unified Speech Recognition
 - **Authors:** Jing-Xuan Zhang, Genshun Wan, Jin Li, Jianqing Gao
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2510.22961

 - **Pdf link:** https://arxiv.org/pdf/2510.22961

 - **Abstract**
 Unified speech recognition aims to perform auditory, visual, and audiovisual speech recognition within a single model framework. While speech foundation models (SFMs) have demonstrated remarkable performance in auditory tasks, their adaptation to multimodal scenarios remains underexplored. This paper presents UASR-LLM, a novel framework that adapts frozen SFMs to unified VSR, ASR, and AVSR tasks by leveraging large language models (LLMs) as text decoders. Our approach introduces visual representations into multiple SFM layers through visual injection modules, enabling multimodal input processing and unified hidden representations. The augmented SFMs connect with decoder-only LLMs via a feed-forward adaptor, where concatenated representations and instruction prompts guide speech transcription. We implement a twostage training strategy: visual injection pretraining followed by speech recognition finetuning. SFM parameters remain frozen throughout training, with only visual injection modules optimized initially, and LLMs finetuned using LoRA parameters subsequently. Experimental results demonstrate superior performance over state-of-the-art baselines across VSR, ASR, and AVSR tasks under both clean and noisy conditions. Ablation studies confirm generalization across various SFMs and LLMs, validating the proposed training strategy.
#### Treble10: A high-quality dataset for far-field speech recognition, dereverberation, and enhancement
 - **Authors:** Sarabeth S. Mullins, Georg Götz, Eric Bezzam, Steven Zheng, Daniel Gert Nielsen
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Machine Learning (cs.LG)
 - **Arxiv link:** https://arxiv.org/abs/2510.23141

 - **Pdf link:** https://arxiv.org/pdf/2510.23141

 - **Abstract**
 Accurate far-field speech datasets are critical for tasks such as automatic speech recognition (ASR), dereverberation, speech enhancement, and source separation. However, current datasets are limited by the trade-off between acoustic realism and scalability. Measured corpora provide faithful physics but are expensive, low-coverage, and rarely include paired clean and reverberant data. In contrast, most simulation-based datasets rely on simplified geometrical acoustics, thus failing to reproduce key physical phenomena like diffraction, scattering, and interference that govern sound propagation in complex environments. We introduce Treble10, a large-scale, physically accurate room-acoustic dataset. Treble10 contains over 3000 broadband room impulse responses (RIRs) simulated in 10 fully furnished real-world rooms, using a hybrid simulation paradigm implemented in the Treble SDK that combines a wave-based and geometrical acoustics solver. The dataset provides six complementary subsets, spanning mono, 8th-order Ambisonics, and 6-channel device RIRs, as well as pre-convolved reverberant speech scenes paired with LibriSpeech utterances. All signals are simulated at 32 kHz, accurately modelling low-frequency wave effects and high-frequency reflections. Treble10 bridges the realism gap between measurement and simulation, enabling reproducible, physically grounded evaluation and large-scale data augmentation for far-field speech tasks. The dataset is openly available via the Hugging Face Hub, and is intended as both a benchmark and a template for next-generation simulation-driven audio research.
#### Matching Reverberant Speech Through Learned Acoustic Embeddings and Feedback Delay Networks
 - **Authors:** Philipp Götz, Gloria Dal Santo, Sebastian J. Schlecht, Vesa Välimäki, Emanuël A.P. Habets
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2510.23158

 - **Pdf link:** https://arxiv.org/pdf/2510.23158

 - **Abstract**
 Reverberation conveys critical acoustic cues about the environment, supporting spatial awareness and immersion. For auditory augmented reality (AAR) systems, generating perceptually plausible reverberation in real time remains a key challenge, especially when explicit acoustic measurements are unavailable. We address this by formulating blind estimation of artificial reverberation parameters as a reverberant signal matching task, leveraging a learned room-acoustic prior. Furthermore, we propose a feedback delay network (FDN) structure that reproduces both frequency-dependent decay times and the direct-to-reverberation ratio of a target space. Experimental evaluation against a leading automatic FDN tuning method demonstrates improvements in estimated room-acoustic parameters and perceptual plausibility of artificial reverberant speech. These results highlight the potential of our approach for efficient, perceptually consistent reverberation rendering in AAR applications.
#### LibriConvo: Simulating Conversations from Read Literature for ASR and Diarization
 - **Authors:** Máté Gedeon, Péter Mihajlik
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Computation and Language (cs.CL); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2510.23320

 - **Pdf link:** https://arxiv.org/pdf/2510.23320

 - **Abstract**
 We introduce LibriConvo, a simulated multi-speaker conversational dataset based on speaker-aware conversation simulation (SASC), designed to support training and evaluation of speaker diarization and automatic speech recognition (ASR) systems. Unlike prior resources that mostly rely on semantically disconnected utterances and implausible temporal gaps, LibriConvo ensures semantic coherence and realistic conversational timing. Our pipeline leverages CallHome with external VAD for reliable boundaries, applies compression to reduce unnaturally long silences, and organizes LibriTTS utterances by book to maintain contextual consistency. Acoustic realism is enhanced via a novel room impulse response selection procedure that ranks speaker-microphone configurations by spatial plausibility, balancing realism and diversity. The dataset comprises 240.1 hours across 1,496 dialogues with 830 unique speakers, split in a speaker-disjoint manner for robust evaluation. Baselines show that the sortformer model outperforms the pyannote pipeline in diarization, while a fine-tuned Fast Conformer-CTC XLarge with Serialized Output Training achieves 7.29\% WER for ASR, surpassing zero-shot Whisper-large-v3. LibriConvo provides a valuable resource for advancing multi-speaker speech processing research with realistic conversational dynamics and controlled experimental conditions.
#### SoulX-Podcast: Towards Realistic Long-form Podcasts with Dialectal and Paralinguistic Diversity
 - **Authors:** Hanke Xie, Haopeng Lin, Wenxiao Cao, Dake Guo, Wenjie Tian, Jun Wu, Hanlin Wen, Ruixuan Shang, Hongmei Liu, Zhiqi Jiang, Yuepeng Jiang, Wenxi Chen, Ruiqi Yan, Jiale Qian, Yichao Yan, Shunshun Yin, Ming Tao, Xie Chen, Lei Xie, Xinsheng Wang
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2510.23541

 - **Pdf link:** https://arxiv.org/pdf/2510.23541

 - **Abstract**
 Recent advances in text-to-speech (TTS) synthesis have significantly improved speech expressiveness and naturalness. However, most existing systems are tailored for single-speaker synthesis and fall short in generating coherent multi-speaker conversational speech. This technical report presents SoulX-Podcast, a system designed for podcast-style multi-turn, multi-speaker dialogic speech generation, while also achieving state-of-the-art performance in conventional TTS tasks. To meet the higher naturalness demands of multi-turn spoken dialogue, SoulX-Podcast integrates a range of paralinguistic controls and supports both Mandarin and English, as well as several Chinese dialects, including Sichuanese, Henanese, and Cantonese, enabling more personalized podcast-style speech generation. Experimental results demonstrate that SoulX-Podcast can continuously produce over 90 minutes of conversation with stable speaker timbre and smooth speaker transitions. Moreover, speakers exhibit contextually adaptive prosody, reflecting natural rhythm and intonation changes as dialogues progress. Across multiple evaluation metrics, SoulX-Podcast achieves state-of-the-art performance in both monologue TTS and multi-turn conversational speech synthesis.
#### Beyond IVR Touch-Tones: Customer Intent Routing using LLMs
 - **Authors:** Sergio Rojas-Galeano
 - **Subjects:** Subjects:
Human-Computer Interaction (cs.HC); Artificial Intelligence (cs.AI); Computation and Language (cs.CL); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2510.21715

 - **Pdf link:** https://arxiv.org/pdf/2510.21715

 - **Abstract**
 Widespread frustration with rigid touch-tone Interactive Voice Response (IVR) systems for customer service underscores the need for more direct and intuitive language interaction. While speech technologies are necessary, the key challenge lies in routing intents from user phrasings to IVR menu paths, a task where Large Language Models (LLMs) show strong potential. Progress, however, is limited by data scarcity, as real IVR structures and interactions are often proprietary. We present a novel LLM-based methodology to address this gap. Using three distinct models, we synthesized a realistic 23-node IVR structure, generated 920 user intents (230 base and 690 augmented), and performed the routing task. We evaluate two prompt designs: descriptive hierarchical menus and flattened path representations, across both base and augmented datasets. Results show that flattened paths consistently yield higher accuracy, reaching 89.13% on the base dataset compared to 81.30% with the descriptive format, while augmentation introduces linguistic noise that slightly reduces performance. Confusion matrix analysis further suggests that low-performing routes may reflect not only model limitations but also redundancies in menu design. Overall, our findings demonstrate proof-of-concept that LLMs can enable IVR routing through a smoother, more seamless user experience -- moving customer service one step ahead of touch-tone menus.
#### Low-Resource Audio Codec (LRAC): 2025 Challenge Description
 - **Authors:** Kamil Wojcicki, Yusuf Ziya Isik, Laura Lechler, Mansur Yesilbursa, Ivana Balić, Wolfgang Mack, Rafał Łaganowski, Guoqing Zhang, Yossi Adi, Minje Kim, Shinji Watanabe
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2510.23312

 - **Pdf link:** https://arxiv.org/pdf/2510.23312

 - **Abstract**
 While recent neural audio codecs deliver superior speech quality at ultralow bitrates over traditional methods, their practical adoption is hindered by obstacles related to low-resource operation and robustness to acoustic distortions. Edge deployment scenarios demand codecs that operate under stringent compute constraints while maintaining low latency and bitrate. The presence of background noise and reverberation further necessitates designs that are resilient to such degradations. The performance of neural codecs under these constraints and their integration with speech enhancement remain largely unaddressed. To catalyze progress in this area, we introduce the 2025 Low-Resource Audio Codec Challenge, which targets the development of neural and hybrid codecs for resource-constrained applications. Participants are supported with a standardized training dataset, two baseline systems, and a comprehensive evaluation framework. The challenge is expected to yield valuable insights applicable to both codec design and related downstream audio tasks.


by Zyzzyva0381 (Windy). 


2025-10-28
