# Showing new listings for Friday, 15 May 2026
Auto update papers at about 2:30am UTC (10:30am Beijing time) every weekday.


阅读 `Usage.md`了解如何使用此repo实现个性化的Arxiv论文推送

See `Usage.md` for instructions on how to personalize the repo. 


Keyword list: ['text-to-speech', 'text to speech', 'tts', 'LLM-based', 'speech', 'voice']


Excluded: []


### Today: 3papers 
#### A Benchmark for Early-stage Parkinson's Disease Detection from Speech
 - **Authors:** Terry Yi Zhong, Cristian Tejedor-Garcia, Khiet P. Truong, Janna Maas, Louis ten Bosch, Bastiaan R. Bloem
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Artificial Intelligence (cs.AI); Computation and Language (cs.CL); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2605.14066

 - **Pdf link:** https://arxiv.org/pdf/2605.14066

 - **Abstract**
 Early-stage Parkinson's disease (EarlyPD) detection from speech is clinically meaningful yet underexplored, and published results are hard to compare because studies differ in datasets, languages, tasks, evaluation protocols, and EarlyPD definitions. To address this issue, we propose the first benchmark for speech-based EarlyPD detection, with a speaker-independent split designed for fair and replicable cross-method evaluation on researcher-accessible datasets. The benchmark covers three common speech tasks and evaluates methods under different training-resource settings. We also present multi-dimensional evaluation breakdowns by dataset, aggregation level, gender, and disease stage to support fine-grained comparisons and clinical adoption. Our results provide a replicable reference and actionable insights, encouraging the adoption of this publicly available benchmark to advance robust and clinically meaningful EarlyPD detection from speech.
#### Streaming Speech-to-Text Translation with a SpeechLLM
 - **Authors:** Titouan Parcollet, Shucong Zhang, Xianrui Zheng, Rogier C. van Dalen
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Artificial Intelligence (cs.AI); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2605.14766

 - **Pdf link:** https://arxiv.org/pdf/2605.14766

 - **Abstract**
 Normally, a system that translates speech into text consists of separate modules for speech recognition and text-to-text translation. Combining those tasks into a SpeechLLM promises to exploit paralinguistic information in the speech and to reduce cascaded errors. But existing SpeechLLM systems are slow since they do not work in a real streaming fashion: they wait for a complete utterance of audio before outputting a translation, or output tokens at fixed intervals, which is not suitable for real applications. This work proposes an LLM-based architecture for real streaming speech-to-text translation. The LLM learns not just to emit output tokens, but also to decide whether it has seen enough audio to do so. The system is trained using automatic alignments of the input speech and the output text. In experiments on different language pairs, the system achieves a translation quality close to the non-streaming baseline, but with a latency of only 1-2 seconds.
#### SpeakerLLM: A Speaker-Specialized Audio-LLM for Speaker Understanding and Verification Reasoning
 - **Authors:** KiHyun Nam, Jungwoo Heo, Siu Bae, Ha-Jin Yu, Joon Son Chung
 - **Subjects:** Subjects:
Sound (cs.SD); Artificial Intelligence (cs.AI); Machine Learning (cs.LG); Multimedia (cs.MM); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2605.15044

 - **Pdf link:** https://arxiv.org/pdf/2605.15044

 - **Abstract**
 As audio-first agents become increasingly common in physical AI, conversational robots, and screenless wearables, audio large language models (audio-LLMs) must integrate speaker-specific understanding to support user authorization, personalization, and context-aware interaction. This requires modeling who is speaking, how the voice sounds, and how recording conditions affect speaker cues. Conventional speaker verification systems provide strong scalar scores but little linguistic evidence, while current audio-LLMs and speaker-aware language models have limited ability to organize speaker information beyond binary labels or descriptive profiles. We present SpeakerLLM, a speaker-specialized audio-LLM framework that unifies single-utterance speaker profiling, recording-condition understanding, utterance-pair speaker comparison, and evidence-organized verification reasoning within a natural-language interface. We construct verification-reasoning targets and a decision-composition policy that separate profile-level evidence from the final same-or-different decision and organize recording condition, profile evidence, and the decision into a structured trace. At its core, SpeakerLLM uses a hierarchical speaker tokenizer designed to capture multiple granularities of speaker evidence. Utterance-level speaker embeddings summarize identity and profile-level cues, whereas frame-level speaker features preserve fine-grained acoustic descriptors. Experiments show that SpeakerLLM-Base improves speaker-profile and recording-condition understanding over general audio-LLMs, while SpeakerLLM-VR preserves strong generated-verdict accuracy and produces decision traces grounded in the supervised verification reasoning schema. We will release the metadata-enriched supervision dataset and target-construction code for reproducibility.


by Zyzzyva0381 (Windy). 


2026-05-15
