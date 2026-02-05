# Showing new listings for Thursday, 5 February 2026
Auto update papers at about 2:30am UTC (10:30am Beijing time) every weekday.


阅读 `Usage.md`了解如何使用此repo实现个性化的Arxiv论文推送

See `Usage.md` for instructions on how to personalize the repo. 


Keyword list: ['text-to-speech', 'text to speech', 'tts', 'LLM-based', 'speech', 'voice']


Excluded: []


### Today: 7papers 
#### Benchmarking Automatic Speech Recognition for Indian Languages in Agricultural Contexts
 - **Authors:** Chandrashekar M S, Vineet Singh, Lakshmi Pedapudi
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Artificial Intelligence (cs.AI); Computation and Language (cs.CL); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2602.03868

 - **Pdf link:** https://arxiv.org/pdf/2602.03868

 - **Abstract**
 The digitization of agricultural advisory services in India requires robust Automatic Speech Recognition (ASR) systems capable of accurately transcribing domain-specific terminology in multiple Indian languages. This paper presents a benchmarking framework for evaluating ASR performance in agricultural contexts across Hindi, Telugu, and Odia languages. We introduce evaluation metrics including Agriculture Weighted Word Error Rate (AWWER) and domain-specific utility scoring to complement traditional metrics. Our evaluation of 10,934 audio recordings, each transcribed by up to 10 ASR models, reveals performance variations across languages and models, with Hindi achieving the best overall performance (WER: 16.2%) while Odia presents the greatest challenges (best WER: 35.1%, achieved only with speaker diarization). We characterize audio quality challenges inherent to real-world agricultural field recordings and demonstrate that speaker diarization with best-speaker selection can substantially reduce WER for multi-speaker recordings (upto 66% depending on the proportion of multi-speaker audio). We identify recurring error patterns in agricultural terminology and provide practical recommendations for improving ASR systems in low-resource agricultural domains. The study establishes baseline benchmarks for future agricultural ASR development.
#### Sounding Highlights: Dual-Pathway Audio Encoders for Audio-Visual Video Highlight Detection
 - **Authors:** Seohyun Joo, Yoori Oh
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Artificial Intelligence (cs.AI); Computer Vision and Pattern Recognition (cs.CV); Multimedia (cs.MM); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2602.03891

 - **Pdf link:** https://arxiv.org/pdf/2602.03891

 - **Abstract**
 Audio-visual video highlight detection aims to automatically identify the most salient moments in videos by leveraging both visual and auditory cues. However, existing models often underutilize the audio modality, focusing on high-level semantic features while failing to fully leverage the rich, dynamic characteristics of sound. To address this limitation, we propose a novel framework, Dual-Pathway Audio Encoders for Video Highlight Detection (DAViHD). The dual-pathway audio encoder is composed of a semantic pathway for content understanding and a dynamic pathway that captures spectro-temporal dynamics. The semantic pathway extracts high-level information by identifying the content within the audio, such as speech, music, or specific sound events. The dynamic pathway employs a frequency-adaptive mechanism as time evolves to jointly model these dynamics, enabling it to identify transient acoustic events via salient spectral bands and rapid energy changes. We integrate the novel audio encoder into a full audio-visual framework and achieve new state-of-the-art performance on the large-scale this http URL benchmark. Our results demonstrate that a sophisticated, dual-faceted audio representation is key to advancing the field of highlight detection.
#### Universal Robust Speech Adaptation for Cross-Domain Speech Recognition and Enhancement
 - **Authors:** Chien-Chun Wang, Hung-Shin Lee, Hsin-Min Wang, Berlin Chen
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Computation and Language (cs.CL); Machine Learning (cs.LG); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2602.04307

 - **Pdf link:** https://arxiv.org/pdf/2602.04307

 - **Abstract**
 Pre-trained models for automatic speech recognition (ASR) and speech enhancement (SE) have exhibited remarkable capabilities under matched noise and channel conditions. However, these models often suffer from severe performance degradation when confronted with domain shifts, particularly in the presence of unseen noise and channel distortions. In view of this, we in this paper present URSA-GAN, a unified and domain-aware generative framework specifically designed to mitigate mismatches in both noise and channel conditions. URSA-GAN leverages a dual-embedding architecture that consists of a noise encoder and a channel encoder, each pre-trained with limited in-domain data to capture domain-relevant representations. These embeddings condition a GAN-based speech generator, facilitating the synthesis of speech that is acoustically aligned with the target domain while preserving phonetic content. To enhance generalization further, we propose dynamic stochastic perturbation, a novel regularization technique that introduces controlled variability into the embeddings during generation, promoting robustness to unseen domains. Empirical results demonstrate that URSA-GAN effectively reduces character error rates in ASR and improves perceptual metrics in SE across diverse noisy and mismatched channel scenarios. Notably, evaluations on compound test conditions with both channel and noise degradations confirm the generalization ability of URSA-GAN, yielding relative improvements of 16.16% in ASR performance and 15.58% in SE metrics.
#### LALM-as-a-Judge: Benchmarking Large Audio-Language Models for Safety Evaluation in Multi-Turn Spoken Dialogues
 - **Authors:** Amir Ivry, Shinji Watanabe
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2602.04796

 - **Pdf link:** https://arxiv.org/pdf/2602.04796

 - **Abstract**
 Spoken dialogues with and between voice agents are becoming increasingly common, yet assessing them for their socially harmful content such as violence, harassment, and hate remains text-centric and fails to account for audio-specific cues and transcription errors. We present LALM-as-a-Judge, the first controlled benchmark and systematic study of large audio-language models (LALMs) as safety judges for multi-turn spoken dialogues. We generate 24,000 unsafe and synthetic spoken dialogues in English that consist of 3-10 turns, by having a single dialogue turn including content with one of 8 harmful categories (e.g., violence) and on one of 5 grades, from very mild to severe. On 160 dialogues, 5 human raters confirmed reliable unsafe detection and a meaningful severity scale. We benchmark three open-source LALMs: Qwen2-Audio, Audio Flamingo 3, and MERaLiON as zero-shot judges that output a scalar safety score in [0,1] across audio-only, transcription-only, or multimodal inputs, along with a transcription-only LLaMA baseline. We measure the judges' sensitivity to detecting unsafe content, the specificity in ordering severity levels, and the stability of the score in dialogue turns. Results reveal architecture- and modality-dependent trade-offs: the most sensitive judge is also the least stable across turns, while stable configurations sacrifice detection of mild harmful content. Transcription quality is a key bottleneck: Whisper-Large may significantly reduce sensitivity for transcription-only modes, while largely preserving severity ordering. Audio becomes crucial when paralinguistic cues or transcription fidelity are category-critical. We summarize all findings and provide actionable guidance for practitioners.
#### Decoding Ambiguous Emotions with Test-Time Scaling in Audio-Language Models
 - **Authors:** Hong Jia, Weibin Li, Jingyao Wu, Xiaofeng Yu, Yan Gao, Jintao Cheng, Xiaoyu Tang, Feng Xia, Ting Dang
 - **Subjects:** Subjects:
Sound (cs.SD); Artificial Intelligence (cs.AI); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2602.03873

 - **Pdf link:** https://arxiv.org/pdf/2602.03873

 - **Abstract**
 Emotion recognition from human speech is a critical enabler for socially aware conversational AI. However, while most prior work frames emotion recognition as a categorical classification problem, real-world affective states are often ambiguous, overlapping, and context-dependent, posing significant challenges for both annotation and automatic modeling. Recent large-scale audio language models (ALMs) offer new opportunities for nuanced affective reasoning without explicit emotion supervision, but their capacity to handle ambiguous emotions remains underexplored. At the same time, advances in inference-time techniques such as test-time scaling (TTS) have shown promise for improving generalization and adaptability in hard NLP tasks, but their relevance to affective computing is still largely unknown. In this work, we introduce the first benchmark for ambiguous emotion recognition in speech with ALMs under test-time scaling. Our evaluation systematically compares eight state-of-the-art ALMs and five TTS strategies across three prominent speech emotion datasets. We further provide an in-depth analysis of the interaction between model capacity, TTS, and affective ambiguity, offering new insights into the computational and representational challenges of ambiguous emotion understanding. Our benchmark establishes a foundation for developing more robust, context-aware, and emotionally intelligent speech-based AI systems, and highlights key future directions for bridging the gap between model assumptions and the complexity of real-world human emotion.
#### Frontend Token Enhancement for Token-Based Speech Recognition
 - **Authors:** Takanori Ashihara, Shota Horiguchi, Kohei Matsuura, Tsubasa Ochiai, Marc Delcroix
 - **Subjects:** Subjects:
Sound (cs.SD); Computation and Language (cs.CL); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2602.04217

 - **Pdf link:** https://arxiv.org/pdf/2602.04217

 - **Abstract**
 Discretized representations of speech signals are efficient alternatives to continuous features for various speech applications, including automatic speech recognition (ASR) and speech language models. However, these representations, such as semantic or phonetic tokens derived from clustering outputs of self-supervised learning (SSL) speech models, are susceptible to environmental noise, which can degrade backend task performance. In this work, we introduce a frontend system that estimates clean speech tokens from noisy speech and evaluate it on an ASR backend using semantic tokens. We consider four types of enhancement models based on their input/output domains: wave-to-wave, token-to-token, continuous SSL features-to-token, and wave-to-token. These models are trained independently of ASR backends. Experiments on the CHiME-4 dataset demonstrate that wave-to-token enhancement achieves the best performance among the frontends. Moreover, it mostly outperforms the ASR system based on continuous SSL features.
#### Speaker-Aware Simulation Improves Conversational Speech Recognition
 - **Authors:** Máté Gedeon, Péter Mihajlik
 - **Subjects:** Subjects:
Sound (cs.SD); Computation and Language (cs.CL); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2602.04776

 - **Pdf link:** https://arxiv.org/pdf/2602.04776

 - **Abstract**
 Automatic speech recognition (ASR) for conversational speech remains challenging due to the limited availability of large-scale, well-annotated multi-speaker dialogue data and the complex temporal dynamics of natural interactions. Speaker-aware simulated conversations (SASC) offer an effective data augmentation strategy by transforming single-speaker recordings into realistic multi-speaker dialogues. However, prior work has primarily focused on English data, leaving questions about the applicability to lower-resource languages. In this paper, we adapt and implement the SASC framework for Hungarian conversational ASR. We further propose C-SASC, an extended variant that incorporates pause modeling conditioned on utterance duration, enabling a more faithful representation of local temporal dependencies observed in human conversation while retaining the simplicity and efficiency of the original approach. We generate synthetic Hungarian dialogues from the BEA-Large corpus and combine them with real conversational data for ASR training. Both SASC and C-SASC are evaluated extensively under a wide range of simulation configurations, using conversational statistics derived from CallHome, BEA-Dialogue, and GRASS corpora. Experimental results show that speaker-aware conversational simulation consistently improves recognition performance over naive concatenation-based augmentation. While the additional duration conditioning in C-SASC yields modest but systematic gains--most notably in character-level error rates--its effectiveness depends on the match between source conversational statistics and the target domain. Overall, our findings confirm the robustness of speaker-aware conversational simulation for Hungarian ASR and highlight the benefits and limitations of increasingly detailed temporal modeling in synthetic dialogue generation.


by Zyzzyva0381 (Windy). 


2026-02-05
