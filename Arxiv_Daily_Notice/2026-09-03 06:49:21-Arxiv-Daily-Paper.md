# Showing new listings for Thursday, 3 September 2026
Auto update papers at about 2:30am UTC (10:30am Beijing time) every weekday.


阅读 `Usage.md`了解如何使用此repo实现个性化的Arxiv论文推送

See `Usage.md` for instructions on how to personalize the repo. 


Keyword list: ['text-to-speech', 'text to speech', 'tts', 'LLM-based', 'speech', 'voice']


Excluded: []


### Today: 4papers 
#### Sensing Bone-Conducted Speech with Earbuds
 - **Authors:** Christoph Weyer, Peter Jax
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2609.02165

 - **Pdf link:** https://arxiv.org/pdf/2609.02165

 - **Abstract**
 Clear capture of the wearer's own voice (OV) is essential when using earbuds for mobile communication. However, OV capture remains challenging in noisy environments. Bone-conducted (BC) speech, which can be sensed as vibrations of the earbud housing, can be used to improve OV capture. However, neither bandwidth nor spatial characteristics of OV-induced earbud vibrations have been analyzed in detail, despite both characteristics being relevant, e.g., for sensor choice and placement. This study investigates both characteristics, based on measurements with two earbud models. Spectrally, results indicate that OV-induced earbud vibrations exhibit a low-pass characteristic, with a steep roll-off of -93 dB per decade above 400 Hz. Thus, sensors with comparatively low noise floors are required to sense the vibrations above \SI{1}{\kilo\hertz}. Spatially, results indicate that the earbuds mainly vibrate in and out of the ear canal entrance, with high consistency between subjects and fits. Simulations confirm that this enables capture of the high-power vibrations below 400 Hz by a single-axis sensor with less than 1.5 dB mean attenuation.
#### VAANI Noise Event Dataset: A curated spontaneous speech dataset annotated with timestamps for noise events
 - **Authors:** Pavan Kumar J, Agneedh Basu, Pranav Bhat, Sujith Pulikodan, Suryansh Shukla, Nihar Desai Prasanta K. Ghosh
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2609.02474

 - **Pdf link:** https://arxiv.org/pdf/2609.02474

 - **Abstract**
 Most public sound-event corpora are optimized either for general audio tagging or for clean speech separation, and comparatively few provide strong timestamped noise annotations layered directly on top of spontaneous, real-world speech. We present the VAANI Noise Event Timestamp Dataset, a derived annotation layer built on Project VAANI field recordings of spontaneous speech collected across 165 Indian districts in 105 languages. Unlike synthetically mixed corpora, VAANI captures speech and ambient noise in situ and simultaneously, and annotates each recording with fine-grained start/end timestamps for overlapping background noise events organized into a compact seven-class semantic taxonomy: animal, traffic, baby/child, music, signal/alarm, appliance, and non-speech human. This combination of spontaneous multilingual Indic speech, authentic regional soundscapes, and span-level noise tags that may overlap with speech targets tasks that existing datasets address only partially: noise-robust Automatic Speech Recognition (ASR), sound event detection (SED), and speech enhancement. We position VAANI against nine widely used corpora and benchmarks, including WHAM!, AVA-Speech, MUSAN, FSD50K, CHiME-6, AudioSet, DESED, the India-specific iNoise noise database, and the Kathbath-Noisy noisy-ASR benchmarks, and describe the annotation protocol and quality-control procedure used to produce the timestamped tags.
#### VibeVoice-ASR-Streaming Technical Report
 - **Authors:** Yujie Tu, Zhiliang Peng, Jianwei Yu, Li Dong, Songchen Xu, Yaoyao Chang, Wenhui Wang, Zilong Wang, Zehua Wang, Yan Xia, Jiajun Zhang, Xie Chen, Furu Wei
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2609.02812

 - **Pdf link:** https://arxiv.org/pdf/2609.02812

 - **Abstract**
 Traditional speaker-attributed ASR systems treated ASR and speaker diarization as two separate tasks. Recently, end-to-end models such as VibeVoice-ASR have unified the two tasks within a single model. However, existing unified models still mainly support offline recognition, making it difficult to meet the low-latency requirements of real-time voice assistants and agents. To tackle this issue, we present VibeVoice-ASR-Streaming, one of the first LLM-based end-to-end approaches to streaming speaker-attributed ASR. It interleaves fixed-size audio chunks, a small amount of lookahead audio and previous text. This allows the model to produce ''who said what'' as speech arrives, without a separate diarization stage. For transcription accuracy, our 7B model achieves the lowest average WER/CER across five evaluation sets. For speaker attribution, it achieves the best or tied-best on 12 of 13 evaluation settings. We release the 1.5B and 7B model weights together with inference code.
#### Hearing the Whispers: Black-Box Membership Inference Attacks on Finetuned TTS Models
 - **Authors:** Kunlin Cai, Kaiyuan Zhang, Zihang Xiang, Jinghuai Zhang, Abeer Alwan, Fnu Suya, Yuan Tian
 - **Subjects:** Subjects:
Cryptography and Security (cs.CR); Machine Learning (cs.LG); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2609.01723

 - **Pdf link:** https://arxiv.org/pdf/2609.01723

 - **Abstract**
 Text-to-Speech (TTS) foundation models are increasingly fine-tuned on private datasets to synthesize highly personalized voices, introducing severe privacy risks by exposing both biometric identities and sensitive speech content. Existing black-box membership inference attacks (MIAs) follow a two-stage pipeline of query generation and representation engineering, both of which face unique challenges when adapted to TTS. For query generation, dual conditioning on synthesis text and reference speech creates a large and underexplored query design space with no established criterion for identifying an effective query. For representation engineering, the multi-level speech characteristics and temporal variability of speech make low-level representations and direct comparisons inadequate for capturing membership signals. To address these challenges, we present the first black-box MIA framework explicitly tailored to TTS models at both the speaker and record levels. For query generation, we characterize the feasible query space and establish two criteria, scorable extent and memorization elicitation, for evaluating five representative queries, identifying recitation as the strongest. For representation engineering, we obtain multi-level speech representations from embedding models and temporally align the generated and target audio for fine-grained comparison. Evaluations across three state-of-the-art TTS models (CosyVoice2, F5-TTS, and XTTS-v2) fine-tuned on two benchmark datasets (VCTK and British Dialect) reveal severe privacy leakage: speaker-level AUC remains above 0.80 and approaches 1.0 in the strongest settings, while record-level AUC ranges from 0.80 to 0.90 and remains effective even in challenging scenarios where both members and non-members are of the same speakers. We further identify speech characteristics associated with disproportionate vulnerability to memorization.


by Zyzzyva0381 (Windy). 


2026-09-03
