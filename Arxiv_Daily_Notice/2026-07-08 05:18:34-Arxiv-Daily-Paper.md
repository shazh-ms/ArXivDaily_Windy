# Showing new listings for Wednesday, 8 July 2026
Auto update papers at about 2:30am UTC (10:30am Beijing time) every weekday.


阅读 `Usage.md`了解如何使用此repo实现个性化的Arxiv论文推送

See `Usage.md` for instructions on how to personalize the repo. 


Keyword list: ['text-to-speech', 'text to speech', 'tts', 'LLM-based', 'speech', 'voice']


Excluded: []


### Today: 3papers 
#### Goodbye Equal Error Rate, Hello Local Information Disclosure: Evaluating Voice Anonymisation against 1-to-N Linkage Threats
 - **Authors:** Dāvis Šterns, Konstantinos Drossos, Natasha Fernandes, Tom Bäckström, Catuscia Palamidessi
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2607.06259

 - **Pdf link:** https://arxiv.org/pdf/2607.06259

 - **Abstract**
 Voice anonymisation aims to protect speaker identity. Currently, its empirical privacy evaluation heavily relies on the Equal Error Rate (EER). Originally designed for biometric verification, EER aggregates scores globally, implicitly assuming an attacker is only trying to verify if two specific voice samples match (a 1-to-1 comparison). This introduces a threat model mismatch with real-world database linkage attacks, where an attacker searches across a fixed set of N enrolled identities (a 1-to-N closed-set search), allowing global averages to obscure localised privacy failures. While recent 1-to-N metrics address this aggregation issue, they abstract away the magnitude of the biometric evidence. In this paper, we propose a modular, information-theoretic evaluation framework explicitly designed for the 1-to-N linkage threat model. Within this framework, our core metric, Local Information Disclosure (LID), quantifies the exact privacy loss of a single trial utterance in bits by calibrating its raw similarity scores into the attacker's posterior confidence for each enrolled identity. Evaluating top-performing systems from the VoicePrivacy 2024 Challenge reveals that systems exhibiting near-perfect EERs (48 %) can still suffer from localised vulnerabilities with worst-case disclosures reaching 1 bit per trial utterance (effectively doubling the attacker's success rate over a random guess). We demonstrate that adopting localised privacy metrics is essential for capturing worst-case risks and aligning with strict privacy regulations.
#### WordVoice: Explicit and Decoupled Multi-Dimensional Word-Level Control for LLM-Based TTS
 - **Authors:** Sihang Nie, Jinxin Ji, Xiaofen Xing, Deyi Tuo, Chengbin Jin, Jialong Mai, Xiangmin Xu
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Computation and Language (cs.CL); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2607.06461

 - **Pdf link:** https://arxiv.org/pdf/2607.06461

 - **Abstract**
 While recent Large Language Model (LLM)-based Text-to-Speech (TTS) systems have achieved remarkable naturalness, they predominantly rely on implicit end-to-end generation paradigms, resulting in coarse-grained control. In scenarios demanding precise stylistic interventions and strict temporal alignment, such as audiobook narration and video dubbing, the inability to explicitly manipulate word-level acoustic attributes remains a critical bottleneck. This limitation is primarily amplified by the severe scarcity of fine-grained annotated datasets and the architectural challenge of integrating multi-dimensional control signals into discrete autoregressive generation. To address this, we propose a unified framework for highly precise word-level control. First, we construct WordVoice-5A, a massive 4.7k-hour bilingual dataset featuring five-dimensional word-level annotations (duration, boundary, energy, pitch and tone) developed through a rigorous linguistically-guided pipeline. Second, we introduce WordVoice to transform the implicit generation process into an explicit, highly controllable paradigm. Specifically, we introduce a bound-token mechanism within the LLM to formulate an explicit ``acoustic planning'' process, enabling adaptive multi-task prosodic planning and flexible manual intervention. Furthermore, we augment the token-to-waveform stage with a fine-grained acoustic modulation module, bridging the resolution gap to strictly align word-level attributes between highly compressed discrete tokens and continuous waveforms. Extensive experiments demonstrate that WordVoice achieves superior, decoupled control over multiple acoustic dimensions while maintaining competitive zero-shot synthesis stability. The code and audio samples are publicly available at this https URL.
#### Revisiting the Relation Between Language Model Perplexity and ASR Word Error Rate for Modern End-to-End Speech Recognition
 - **Authors:** Mohammad Zeineldeen, Albert Zeyer, Haoran Zhang, Robin Schmitt, Ralf Schlüter, Hermann Ney
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2607.05612

 - **Pdf link:** https://arxiv.org/pdf/2607.05612

 - **Abstract**
 Language model (LM) perplexity (PPL) has historically been used as a proxy for automatic speech recognition (ASR) word error rate (WER), with prior work reporting an approximately linear relation in log-log space. Modern end-to-end ASR systems challenge this assumption because they already contain internal language modeling capacity, are often evaluated without external language models, and can now be combined with neural LMs and large language models (LLMs) through different recognition strategies. This paper revisits the relation between PPL and WER for modern ASR systems. We study whether external LMs still improve current end-to-end ASR systems, whether the PPL-WER relation remains linear in log-log space, how encoder context length affects this relation, and how LLM perplexities fit into the trend observed for standard neural LMs. We further investigate internal language modeling (ILM) in attention-based encoder-decoder systems and show that ILM subtraction changes the observed PPL-WER relation, indicating that the decoder's internal LM must be considered when interpreting the effect of external LM quality.


by Zyzzyva0381 (Windy). 


2026-07-08
