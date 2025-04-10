# Showing new listings for Thursday, 10 April 2025
Auto update papers at about 2:30am UTC (10:30am Beijing time) every weekday.


阅读 `Usage.md`了解如何使用此repo实现个性化的Arxiv论文推送

See `Usage.md` for instructions on how to personalize the repo. 


Keyword list: ['acoustic', 'reinforcement learning', 'reverb', 'meta', 'separate', 'reconstruction', 'noise', 'enhance', 'localization', 'speech']


Excluded: []


### Today: 3papers 
#### RNN-Transducer-based Losses for Speech Recognition on Noisy Targets
 - **Authors:** Vladimir Bataev
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Artificial Intelligence (cs.AI); Computation and Language (cs.CL); Machine Learning (cs.LG); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2504.06963

 - **Pdf link:** https://arxiv.org/pdf/2504.06963

 - **Abstract**
 Training speech recognition systems on noisy transcripts is a significant challenge in industrial pipelines, where datasets are enormous and ensuring accurate transcription for every instance is difficult. In this work, we introduce novel loss functions to mitigate the impact of transcription errors in RNN-Transducer models. Our Star-Transducer loss addresses deletion errors by incorporating "skip frame" transitions in the loss lattice, restoring over 90% of the system's performance compared to models trained with accurate transcripts. The Bypass-Transducer loss uses "skip token" transitions to tackle insertion errors, recovering more than 60% of the quality. Finally, the Target-Robust Transducer loss merges these approaches, offering robust performance against arbitrary errors. Experimental results demonstrate that the Target-Robust Transducer loss significantly improves RNN-T performance on noisy data by restoring over 70% of the quality compared to well-transcribed data.
#### A Cascaded Architecture for Extractive Summarization of Multimedia Content via Audio-to-Text Alignment
 - **Authors:** Tanzir Hossain, Ar-Rafi Islam, Md. Sabbir Hossain, Annajiat Alim Rasel
 - **Subjects:** Subjects:
Information Retrieval (cs.IR); Artificial Intelligence (cs.AI); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2504.06275

 - **Pdf link:** https://arxiv.org/pdf/2504.06275

 - **Abstract**
 This study presents a cascaded architecture for extractive summarization of multimedia content via audio-to-text alignment. The proposed framework addresses the challenge of extracting key insights from multimedia sources like YouTube videos. It integrates audio-to-text conversion using Microsoft Azure Speech with advanced extractive summarization models, including Whisper, Pegasus, and Facebook BART XSum. The system employs tools such as Pytube, Pydub, and SpeechRecognition for content retrieval, audio extraction, and transcription. Linguistic analysis is enhanced through named entity recognition and semantic role labeling. Evaluation using ROUGE and F1 scores demonstrates that the cascaded architecture outperforms conventional summarization methods, despite challenges like transcription errors. Future improvements may include model fine-tuning and real-time processing. This study contributes to multimedia summarization by improving information retrieval, accessibility, and user experience.
#### TASTE: Text-Aligned Speech Tokenization and Embedding for Spoken Language Modeling
 - **Authors:** Liang-Hsuan Tseng, Yi-Chang Chen, Kuan-Yi Lee, Da-Shan Shiu, Hung-yi Lee
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2504.07053

 - **Pdf link:** https://arxiv.org/pdf/2504.07053

 - **Abstract**
 Large Language Models (LLMs) excel in text-based natural language processing tasks but remain constrained by their reliance on textual inputs and outputs. To enable more natural human-LLM interaction, recent progress have focused on deriving a spoken language model (SLM) that can not only listen but also generate speech. To achieve this, a promising direction is to conduct speech-text joint modeling. However, recent SLM still lag behind text LLM due to the modality mismatch. One significant mismatch can be the sequence lengths between speech and text tokens. To address this, we introduce Text-Aligned Speech Tokenization and Embedding (TASTE), a method that directly addresses the modality gap by aligning speech token with the corresponding text transcription during the tokenization stage. We propose a method that can achieve this through the special aggregation mechanism and with speech reconstruction as the training objective. We conduct extensive experiments and show that TASTE can preserve essential paralinguistic information while dramatically reducing the token sequence length. Furthermore, by leveraging TASTE, we can adapt text-based LLMs into effective SLMs with parameter-efficient fine-tuning techniques such as Low-Rank Adaptation (LoRA). Experimental results on benchmark tasks, including SALMON and StoryCloze, demonstrate that TASTE-based SLMs perform similarly to previous full-finetuning methods. To our knowledge, TASTE is the first end-to-end approach that utilizes a reconstruction objective to automatically learn a text-aligned speech tokenization and embedding suitable for spoken language modeling. Our demo, code, and models are publicly available at this https URL.


by Zyzzyva0381 (Windy). 


2025-04-10
