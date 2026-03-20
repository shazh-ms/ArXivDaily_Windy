# Showing new listings for Friday, 20 March 2026
Auto update papers at about 2:30am UTC (10:30am Beijing time) every weekday.


阅读 `Usage.md`了解如何使用此repo实现个性化的Arxiv论文推送

See `Usage.md` for instructions on how to personalize the repo. 


Keyword list: ['text-to-speech', 'text to speech', 'tts', 'LLM-based', 'speech', 'voice']


Excluded: []


### Today: 5papers 
#### PCOV-KWS: Multi-task Learning for Personalized Customizable Open Vocabulary Keyword Spotting
 - **Authors:** Jianan Pan, Kejie Huang
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Artificial Intelligence (cs.AI); Computation and Language (cs.CL); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2603.18023

 - **Pdf link:** https://arxiv.org/pdf/2603.18023

 - **Abstract**
 As advancements in technologies like Internet of Things (IoT), Automatic Speech Recognition (ASR), Speaker Verification (SV), and Text-to-Speech (TTS) lead to increased usage of intelligent voice assistants, the demand for privacy and personalization has escalated. In this paper, we introduce a multi-task learning framework for personalized, customizable open-vocabulary Keyword Spotting (PCOV-KWS). This framework employs a lightweight network to simultaneously perform Keyword Spotting (KWS) and SV to address personalized KWS requirements. We have integrated a training criterion distinct from softmax-based loss, transforming multi-class classification into multiple binary classifications, which eliminates inter-category competition, while an optimization strategy for multi-task loss weighting is employed during training. We evaluated our PCOV-KWS system in multiple datasets, demonstrating that it outperforms the baselines in evaluation results, while also requiring fewer parameters and lower computational resources.
#### ARTT: Augmented Reverberant-Target Training for Unsupervised Monaural Speech Dereverberation
 - **Authors:** Siqi Song, Fulin Wu, Zhong-Qiu Wang
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2603.18485

 - **Pdf link:** https://arxiv.org/pdf/2603.18485

 - **Abstract**
 Due to the absence of clean reference signals and spatial cues, monaural unsupervised speech dereverberation is a challenging ill-posed inverse problem. To realize it, we propose augmented reverberant-target training (ARTT), which consists of two stages. In the first stage, reverberant-target training (RTT) is proposed to first further reverberate the observed reverberant mixture signal, and then train a deep neural network (DNN) to recover the observed reverberant mixture via discriminative training. Although the target signal to fit is reverberant, we find that the resulting DNN can effectively reduce reverberation. In the second stage, an online self-distillation mechanism based on the mean-teacher algorithm is proposed to further improve dereverberation. Evaluation results demonstrate that ARTT achieves strong unsupervised dereverberation performance, significantly outperforming previous baselines.
#### Modeling Overlapped Speech with Shuffles
 - **Authors:** Matthew Wiesner, Samuele Cornell, Alexander Polok, Lucas Ondel Yang, Lukáš Burget, Sanjeev Khudanpur
 - **Subjects:** Subjects:
Sound (cs.SD); Computation and Language (cs.CL); Machine Learning (cs.LG); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2603.17769

 - **Pdf link:** https://arxiv.org/pdf/2603.17769

 - **Abstract**
 We propose to model parallel streams of data, such as overlapped speech, using shuffles. Specifically, this paper shows how the shuffle product and partial order finite-state automata (FSAs) can be used for alignment and speaker-attributed transcription of overlapped speech. We train using the total score on these FSAs as a loss function, marginalizing over all possible serializations of overlapping sequences at subword, word, and phrase levels. To reduce graph size, we impose temporal constraints by constructing partial order FSAs. We address speaker attribution by modeling (token, speaker) tuples directly. Viterbi alignment through the shuffle product FSA directly enables one-pass alignment. We evaluate performance on synthetic LibriSpeech overlaps. To our knowledge, this is the first algorithm that enables single-pass alignment of multi-talker recordings. All algorithms are implemented using k2 / Icefall.
#### DEAF: A Benchmark for Diagnostic Evaluation of Acoustic Faithfulness in Audio Language Models
 - **Authors:** Jiaqi Xiong, Yunjia Qi, Qi Cao, Yu Zheng, Weisheng Xu, Ziteng Wang, Ruofan Liao, Yutong Zhang, Sichen Liu
 - **Subjects:** Subjects:
Artificial Intelligence (cs.AI); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2603.18048

 - **Pdf link:** https://arxiv.org/pdf/2603.18048

 - **Abstract**
 Recent Audio Multimodal Large Language Models (Audio MLLMs) demonstrate impressive performance on speech benchmarks, yet it remains unclear whether these models genuinely process acoustic signals or rely on text-based semantic inference. To systematically study this question, we introduce DEAF (Diagnostic Evaluation of Acoustic Faithfulness), a benchmark of over 2,700 conflict stimuli spanning three acoustic dimensions: emotional prosody, background sounds, and speaker identity. Then, we design a controlled multi-level evaluation framework that progressively increases textual influence, ranging from semantic conflicts in the content to misleading prompts and their combination, allowing us to disentangle content-driven bias from prompt-induced sycophancy. We further introduce diagnostic metrics to quantify model reliance on textual cues over acoustic signals. Our evaluation of seven Audio MLLMs reveals a consistent pattern of text dominance: models are sensitive to acoustic variations, yet predictions are predominantly driven by textual inputs, revealing a gap between high performance on standard speech benchmarks and genuine acoustic understanding.
#### DiscoPhon: Benchmarking the Unsupervised Discovery of Phoneme Inventories With Discrete Speech Units
 - **Authors:** Maxime Poli, Manel Khentout, Angelo Ortiz Tandazo, Ewan Dunbar, Emmanuel Chemla, Emmanuel Dupoux
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2603.18612

 - **Pdf link:** https://arxiv.org/pdf/2603.18612

 - **Abstract**
 We introduce DiscoPhon, a multilingual benchmark for evaluating unsupervised phoneme discovery from discrete speech units. DiscoPhon covers 6 dev and 6 test languages, chosen to span a wide range of phonemic contrasts. Given only 10 hours of speech in a previously unseen language, systems must produce discrete units that are mapped to a predefined phoneme inventory, through either a many-to-one or a one-to-one assignment. The resulting sequences are evaluated for unit quality, recognition and segmentation. We provide four pretrained multilingual HuBERT and SpidR baselines, and show that phonemic information is available enough in current models for derived units to correlate well with phonemes, though with variations across languages.


by Zyzzyva0381 (Windy). 


2026-03-20
