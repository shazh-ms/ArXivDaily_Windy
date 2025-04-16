# Showing new listings for Wednesday, 16 April 2025
Auto update papers at about 2:30am UTC (10:30am Beijing time) every weekday.


阅读 `Usage.md`了解如何使用此repo实现个性化的Arxiv论文推送

See `Usage.md` for instructions on how to personalize the repo. 


Keyword list: ['acoustic', 'reinforcement learning', 'reverb', 'meta', 'separate', 'reconstruction', 'noise', 'enhance', 'localization', 'speech']


Excluded: []


### Today: 5papers 
#### Will AI shape the way we speak? The emerging sociolinguistic influence of synthetic voices
 - **Authors:** Éva Székely, Jūra Miniota, Míša (Michaela)Hejná
 - **Subjects:** Subjects:
Computers and Society (cs.CY); Artificial Intelligence (cs.AI); Computation and Language (cs.CL); Human-Computer Interaction (cs.HC); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2504.10650

 - **Pdf link:** https://arxiv.org/pdf/2504.10650

 - **Abstract**
 The growing prevalence of conversational voice interfaces, powered by developments in both speech and language technologies, raises important questions about their influence on human communication. While written communication can signal identity through lexical and stylistic choices, voice-based interactions inherently amplify socioindexical elements - such as accent, intonation, and speech style - which more prominently convey social identity and group affiliation. There is evidence that even passive media such as television is likely to influence the audience's linguistic patterns. Unlike passive media, conversational AI is interactive, creating a more immersive and reciprocal dynamic that holds a greater potential to impact how individuals speak in everyday interactions. Such heightened influence can be expected to arise from phenomena such as acoustic-prosodic entrainment and linguistic accommodation, which occur naturally during interaction and enable users to adapt their speech patterns in response to the system. While this phenomenon is still emerging, its potential societal impact could provide organisations, movements, and brands with a subtle yet powerful avenue for shaping and controlling public perception and social identity. We argue that the socioindexical influence of AI-generated speech warrants attention and should become a focus of interdisciplinary research, leveraging new and existing methodologies and technologies to better understand its implications.
#### Deep Audio Watermarks are Shallow: Limitations of Post-Hoc Watermarking Techniques for Speech
 - **Authors:** Patrick O'Reilly, Zeyu Jin, Jiaqi Su, Bryan Pardo
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2504.10782

 - **Pdf link:** https://arxiv.org/pdf/2504.10782

 - **Abstract**
 In the audio modality, state-of-the-art watermarking methods leverage deep neural networks to allow the embedding of human-imperceptible signatures in generated audio. The ideal is to embed signatures that can be detected with high accuracy when the watermarked audio is altered via compression, filtering, or other transformations. Existing audio watermarking techniques operate in a post-hoc manner, manipulating "low-level" features of audio recordings after generation (e.g. through the addition of a low-magnitude watermark signal). We show that this post-hoc formulation makes existing audio watermarks vulnerable to transformation-based removal attacks. Focusing on speech audio, we (1) unify and extend existing evaluations of the effect of audio transformations on watermark detectability, and (2) demonstrate that state-of-the-art post-hoc audio watermarks can be removed with no knowledge of the watermarking scheme and minimal degradation in audio quality.
#### SonicSieve: Bringing Directional Speech Extraction to Smartphones Using Acoustic Microstructures
 - **Authors:** Kuang Yuan, Yifeng Wang, Xiyuxing Zhang, Chengyi Shen, Swarun Kumar, Justin Chan
 - **Subjects:** Subjects:
Sound (cs.SD); Human-Computer Interaction (cs.HC); Machine Learning (cs.LG); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2504.10793

 - **Pdf link:** https://arxiv.org/pdf/2504.10793

 - **Abstract**
 Imagine placing your smartphone on a table in a noisy restaurant and clearly capturing the voices of friends seated around you, or recording a lecturer's voice with clarity in a reverberant auditorium. We introduce SonicSieve, the first intelligent directional speech extraction system for smartphones using a bio-inspired acoustic microstructure. Our passive design embeds directional cues onto incoming speech without any additional electronics. It attaches to the in-line mic of low-cost wired earphones which can be attached to smartphones. We present an end-to-end neural network that processes the raw audio mixtures in real-time on mobile devices. Our results show that SonicSieve achieves a signal quality improvement of 5.0 dB when focusing on a 30° angular region. Additionally, the performance of our system based on only two microphones exceeds that of conventional 5-microphone arrays.
#### Generalized Audio Deepfake Detection Using Frame-level Latent Information Entropy
 - **Authors:** Botao Zhao, Zuheng Kang, Yayun He, Xiaoyang Qu, Junqing Peng, Jing Xiao, Jianzong Wang
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2504.10819

 - **Pdf link:** https://arxiv.org/pdf/2504.10819

 - **Abstract**
 Generalizability, the capacity of a robust model to perform effectively on unseen data, is crucial for audio deepfake detection due to the rapid evolution of text-to-speech (TTS) and voice conversion (VC) technologies. A promising approach to differentiate between bonafide and spoof samples lies in identifying intrinsic disparities to enhance model generalizability. From an information-theoretic perspective, we hypothesize the information content is one of the intrinsic differences: bonafide sample represents a dense, information-rich sampling of the real world, whereas spoof sample is typically derived from lower-dimensional, less informative representations. To implement this, we introduce frame-level latent information entropy detector(f-InfoED), a framework that extracts distinctive information entropy from latent representations at the frame level to identify audio deepfakes. Furthermore, we present AdaLAM, which extends large pre-trained audio models with trainable adapters for enhanced feature extraction. To facilitate comprehensive evaluation, the audio deepfake forensics 2024 (ADFF 2024) dataset was built by the latest TTS and VC methods. Extensive experiments demonstrate that our proposed approach achieves state-of-the-art performance and exhibits remarkable generalization capabilities. Further analytical studies confirms the efficacy of AdaLAM in extracting discriminative audio features and f-InfoED in leveraging latent entropy information for more generalized deepfake detection.
#### Dopamine Audiobook: A Training-free MLLM Agent for Emotional and Human-like Audiobook Generation
 - **Authors:** Yan Rong, Shan Yang, Guangzhi Lei, Li Liu
 - **Subjects:** Subjects:
Sound (cs.SD); Multimedia (cs.MM); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2504.11002

 - **Pdf link:** https://arxiv.org/pdf/2504.11002

 - **Abstract**
 Audiobook generation, which creates vivid and emotion-rich audio works, faces challenges in conveying complex emotions, achieving human-like qualities, and aligning evaluations with human preferences. Existing text-to-speech (TTS) methods are often limited to specific scenarios, struggle with emotional transitions, and lack automatic human-aligned evaluation benchmarks, instead relying on either misaligned automated metrics or costly human assessments. To address these issues, we propose Dopamine Audiobook, a new unified training-free system leveraging a multimodal large language model (MLLM) as an AI agent for emotional and human-like audiobook generation and evaluation. Specifically, we first design a flow-based emotion-enhanced framework that decomposes complex emotional speech synthesis into controllable sub-tasks. Then, we propose an adaptive model selection module that dynamically selects the most suitable TTS methods from a set of existing state-of-the-art (SOTA) TTS methods for diverse scenarios. We further enhance emotional expressiveness through paralinguistic augmentation and prosody retrieval at word and utterance levels. For evaluation, we propose a novel GPT-based evaluation framework incorporating self-critique, perspective-taking, and psychological MagicEmo prompts to ensure human-aligned and self-aligned assessments. Experiments show that our method generates long speech with superior emotional expression to SOTA TTS models in various metrics. Importantly, our evaluation framework demonstrates better alignment with human preferences and transferability across audio tasks. Project website with audio samples can be found at this https URL.


by Zyzzyva0381 (Windy). 


2025-04-16
