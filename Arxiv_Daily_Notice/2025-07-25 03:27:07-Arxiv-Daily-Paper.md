# Showing new listings for Friday, 25 July 2025
Auto update papers at about 2:30am UTC (10:30am Beijing time) every weekday.


阅读 `Usage.md`了解如何使用此repo实现个性化的Arxiv论文推送

See `Usage.md` for instructions on how to personalize the repo. 


Keyword list: ['text-to-speech', 'text to speech', 'tts', 'LLM-based', 'speech', 'voice']


Excluded: []


### Today: 11papers 
#### ASR-Guided Speaker-Role Diarization and Diarization-Guided ASR Decoding
 - **Authors:** Arindam Ghosh, Mark Fuhs, Bongjun Kim, Anurag Chowdhury, Monika Woszczyna
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Artificial Intelligence (cs.AI); Machine Learning (cs.LG)
 - **Arxiv link:** https://arxiv.org/abs/2507.17765

 - **Pdf link:** https://arxiv.org/pdf/2507.17765

 - **Abstract**
 From an application standpoint, speaker-role diarization (RD), such as doctor vs. patient, host vs. guest, etc. is often more useful than traditional speaker diarization (SD), which assigns generic labels like speaker-1, speaker-2 etc. In the context of joint automatic speech recognition (ASR) + SD (who spoke what?), recent end-to-end models employ an auxiliary SD transducer, synchronized with the ASR transducer, to predict speakers per word. In this paper, we extend this framework to RD with three key contributions: (1) we simplify the training via forced alignment and cross-entropy loss instead of RNNT loss, (2) we show that word prediction and role prediction require different amounts of predictor's context, leading to separate task-specific predictors, unlike existing shared-predictor models, and (3) we propose a way to leverage RD posterior activity to influence ASR decoding and reduce small-word deletion errors.
#### A Concept-based approach to Voice Disorder Detection
 - **Authors:** Davide Ghia, Gabriele Ciravegna, Alkis Koudounas, Marco Fantini, Erika Crosetti, Giovanni Succo, Tania Cerquitelli
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Machine Learning (cs.LG); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2507.17799

 - **Pdf link:** https://arxiv.org/pdf/2507.17799

 - **Abstract**
 Voice disorders affect a significant portion of the population, and the ability to diagnose them using automated, non-invasive techniques would represent a substantial advancement in healthcare, improving the quality of life of patients. Recent studies have demonstrated that artificial intelligence models, particularly Deep Neural Networks (DNNs), can effectively address this task. However, due to their complexity, the decision-making process of such models often remain opaque, limiting their trustworthiness in clinical contexts. This paper investigates an alternative approach based on Explainable AI (XAI), a field that aims to improve the interpretability of DNNs by providing different forms of explanations. Specifically, this works focuses on concept-based models such as Concept Bottleneck Model (CBM) and Concept Embedding Model (CEM) and how they can achieve performance comparable to traditional deep learning methods, while offering a more transparent and interpretable decision framework.
#### Recent Trends in Distant Conversational Speech Recognition: A Review of CHiME-7 and 8 DASR Challenges
 - **Authors:** Samuele Cornell, Christoph Boeddeker, Taejin Park, He Huang, Desh Raj, Matthew Wiesner, Yoshiki Masuyama, Xuankai Chang, Zhong-Qiu Wang, Stefano Squartini, Paola Garcia, Shinji Watanabe
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Computation and Language (cs.CL); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2507.18161

 - **Pdf link:** https://arxiv.org/pdf/2507.18161

 - **Abstract**
 The CHiME-7 and 8 distant speech recognition (DASR) challenges focus on multi-channel, generalizable, joint automatic speech recognition (ASR) and diarization of conversational speech. With participation from 9 teams submitting 32 diverse systems, these challenges have contributed to state-of-the-art research in the field. This paper outlines the challenges' design, evaluation metrics, datasets, and baseline systems while analyzing key trends from participant submissions. From this analysis it emerges that: 1) Most participants use end-to-end (e2e) ASR systems, whereas hybrid systems were prevalent in previous CHiME challenges. This transition is mainly due to the availability of robust large-scale pre-trained models, which lowers the data burden for e2e-ASR. 2) Despite recent advances in neural speech separation and enhancement (SSE), all teams still heavily rely on guided source separation, suggesting that current neural SSE techniques are still unable to reliably deal with complex scenarios and different recording setups. 3) All best systems employ diarization refinement via target-speaker diarization techniques. Accurate speaker counting in the first diarization pass is thus crucial to avoid compounding errors and CHiME-8 DASR participants especially focused on this part. 4) Downstream evaluation via meeting summarization can correlate weakly with transcription quality due to the remarkable effectiveness of large-language models in handling errors. On the NOTSOFAR-1 scenario, even systems with over 50\% time-constrained minimum permutation WER can perform roughly on par with the most effective ones (around 11\%). 5) Despite recent progress, accurately transcribing spontaneous speech in challenging acoustic environments remains difficult, even when using computationally intensive system ensembles.
#### SpecASR: Accelerating LLM-based Automatic Speech Recognition via Speculative Decoding
 - **Authors:** Linye Wei, Shuzhang Zhong, Songqiang Xu, Runsheng Wang, Ru Huang, Meng Li
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2507.18181

 - **Pdf link:** https://arxiv.org/pdf/2507.18181

 - **Abstract**
 Large language model (LLM)-based automatic speech recognition (ASR) has recently attracted a lot of attention due to its high recognition accuracy and enhanced multi-dialect support. However, the high decoding latency of LLMs challenges the real-time ASR requirements. Although speculative decoding has been explored for better decoding efficiency, they usually ignore the key characteristics of the ASR task and achieve limited speedup. To further reduce the real-time ASR latency, in this paper, we propose a novel speculative decoding framework specialized for ASR, dubbed SpecASR. SpecASR is developed based on our core observation that ASR decoding is audio-conditioned, which results in high output alignment between small and large ASR models, even given output mismatches in intermediate decoding steps. Therefore, SpecASR features an adaptive draft sequence generation process that dynamically modifies the draft sequence length to maximize the token acceptance length. SpecASR further proposes a draft sequence recycling strategy that reuses the previously generated draft sequence to reduce the draft ASR model latency. Moreover, a two-pass sparse token tree generation algorithm is also proposed to balance the latency of draft and target ASR models. With extensive experimental results, we demonstrate SpecASR achieves 3.04x-3.79x and 1.25x-1.84x speedup over the baseline autoregressive decoding and speculative decoding, respectively, without any loss in recognition accuracy.
#### Speech Enhancement with Dual-path Multi-Channel Linear Prediction Filter and Multi-norm Beamforming
 - **Authors:** Chengyuan Qin, Wenmeng Xiong, Jing Zhou, Maoshen Jia, Changchun Bao
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2507.18350

 - **Pdf link:** https://arxiv.org/pdf/2507.18350

 - **Abstract**
 In this paper, we propose a speech enhancement method us ing dual-path Multi-Channel Linear Prediction (MCLP) filters and multi-norm beamforming. Specifically, the MCLP part in the proposed method is designed with dual-path filters in both time and frequency dimensions. For the beamforming part, we minimize the power of the microphone array output as well as the l1 norm of the denoised signals while preserving source sig nals from the target directions. An efficient method to select the prediction orders in the dual-path filters is also proposed, which is robust for signals with different reverberation time (T60) val ues and can be applied to other MCLP-based methods. Eval uations demonstrate that our proposed method outperforms the baseline methods for speech enhancement, particularly in high reverberation scenarios.
#### Streaming Sortformer: Speaker Cache-Based Online Speaker Diarization with Arrival-Time Ordering
 - **Authors:** Ivan Medennikov, Taejin Park, Weiqing Wang, He Huang, Kunal Dhawan, Jinhan Wang, Jagadeesh Balam, Boris Ginsburg
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2507.18446

 - **Pdf link:** https://arxiv.org/pdf/2507.18446

 - **Abstract**
 This paper presents a streaming extension for the Sortformer speaker diarization framework, whose key property is the arrival-time ordering of output speakers. The proposed approach employs an Arrival-Order Speaker Cache (AOSC) to store frame-level acoustic embeddings of previously observed speakers. Unlike conventional speaker-tracing buffers, AOSC orders embeddings by speaker index corresponding to their arrival time order, and is dynamically updated by selecting frames with the highest scores based on the model's past predictions. Notably, the number of stored embeddings per speaker is determined dynamically by the update mechanism, ensuring efficient cache utilization and precise speaker tracking. Experiments on benchmark datasets confirm the effectiveness and flexibility of our approach, even in low-latency setups. These results establish Streaming Sortformer as a robust solution for real-time multi-speaker tracking and a foundation for streaming multi-talker speech processing.
#### Speaker Disentanglement of Speech Pre-trained Model Based on Interpretability
 - **Authors:** Xiaoxu Zhu, Junhua Li
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2507.17851

 - **Pdf link:** https://arxiv.org/pdf/2507.17851

 - **Abstract**
 Speech pretrained models contain task-specific information across different layers, but decoupling content and timbre information remains challenging as removing speaker-specific information often causes content loss. Current research lacks direct metrics to quantify timbre residual in model encodings, relying on indirect evaluation through downstream tasks. This paper addresses these challenges through interpretability-based speaker disentanglement in speech pretraining models. We quantitatively evaluate timbre residual in model embeddings and improve speaker disentanglement using interpretive representations. Our contributions include: (1) InterpTRQE-SptME Benchmark - a timbre residual recognition framework using interpretability. The benchmark concatenates content embeddings with timbre embeddings for speaker classification, then applies Gradient SHAP Explainer to quantify timbre residual. We evaluate seven speech pretraining model variations. (2) InterpTF-SptME method - an interpretability-based timbre filtering approach using SHAP Noise and SHAP Cropping techniques. This model-agnostic method transforms intermediate encodings to remove timbre while preserving content. Experiments on VCTK dataset with HuBERT LARGE demonstrate successful content preservation and significant speaker disentanglement optimization. Results show the SHAP Noise method can reduce timbre residual from 18.05% to near 0% while maintaining content integrity, contributing to enhanced performance in content-related speech processing tasks and preventing timbre privacy leakage.
#### The TEA-ASLP System for Multilingual Conversational Speech Recognition and Speech Diarization in MLC-SLM 2025 Challenge
 - **Authors:** Hongfei Xue, Kaixun Huang, Zhikai Zhou, Shen Huang, Shidong Shang
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2507.18051

 - **Pdf link:** https://arxiv.org/pdf/2507.18051

 - **Abstract**
 This paper presents the TEA-ASLP's system submitted to the MLC-SLM 2025 Challenge, addressing multilingual conversational automatic speech recognition (ASR) in Task I and speech diarization ASR in Task II. For Task I, we enhance Ideal-LLM model by integrating known language identification and a multilingual MOE LoRA structure, along with using CTC-predicted tokens as prompts to improve autoregressive generation. The model is trained on approximately 180k hours of multilingual ASR data. In Task II, we replace the baseline English-Chinese speaker diarization model with a more suitable English-only version. Our approach achieves a 30.8% reduction in word error rate (WER) compared to the baseline speech language model, resulting in a final WER of 9.60% in Task I and a time-constrained minimum-permutation WER of 17.49% in Task II, earning first and second place in the respective challenge tasks.
#### TELEVAL: A Dynamic Benchmark Designed for Spoken Language Models in Chinese Interactive Scenarios
 - **Authors:** Zehan Li, Hongjie Chen, Yuxin Zhang, Jing Zhou, Xuening Wang, Hang Lv, Mengjie Du, Yaodong Song, Jie Lian, Jian Kang, Jie Li, Yongxiang Li, Zhongjiang He, Xuelong Li
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Artificial Intelligence (cs.AI); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2507.18061

 - **Pdf link:** https://arxiv.org/pdf/2507.18061

 - **Abstract**
 Spoken language models (SLMs) have seen rapid progress in recent years, along with the development of numerous benchmarks for evaluating their performance. However, most existing benchmarks primarily focus on evaluating whether SLMs can perform complex tasks comparable to those tackled by large language models (LLMs), often failing to align with how users naturally interact in real-world conversational scenarios. In this paper, we propose TELEVAL, a dynamic benchmark specifically designed to evaluate SLMs' effectiveness as conversational agents in realistic Chinese interactive settings. TELEVAL defines three evaluation dimensions: Explicit Semantics, Paralinguistic and Implicit Semantics, and System Abilities. It adopts a dialogue format consistent with real-world usage and evaluates text and audio outputs separately. TELEVAL particularly focuses on the model's ability to extract implicit cues from user speech and respond appropriately without additional instructions. Our experiments demonstrate that despite recent progress, existing SLMs still have considerable room for improvement in natural conversational tasks. We hope that TELEVAL can serve as a user-centered evaluation framework that directly reflects the user experience and contributes to the development of more capable dialogue-oriented SLMs.
#### GOAT-SLM: A Spoken Language Model with Paralinguistic and Speaker Characteristic Awareness
 - **Authors:** Hongjie Chen, Zehan Li, Yaodong Song, Wenming Deng, Yitong Yao, Yuxin Zhang, Hang Lv, Xuechao Zhu, Jian Kang, Jie Lian, Jie Li, Chao Wang, Shuangyong Song, Yongxiang Li, Zhongjiang He
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Artificial Intelligence (cs.AI); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2507.18119

 - **Pdf link:** https://arxiv.org/pdf/2507.18119

 - **Abstract**
 Recent advances in end-to-end spoken language models (SLMs) have significantly improved the ability of AI systems to engage in natural spoken interactions. However, most existing models treat speech merely as a vehicle for linguistic content, often overlooking the rich paralinguistic and speaker characteristic cues embedded in human speech, such as dialect, age, emotion, and non-speech vocalizations. In this work, we introduce GOAT-SLM, a novel spoken language model with paralinguistic and speaker characteristic awareness, designed to extend spoken language modeling beyond text semantics. GOAT-SLM adopts a dual-modality head architecture that decouples linguistic modeling from acoustic realization, enabling robust language understanding while supporting expressive and adaptive speech generation. To enhance model efficiency and versatility, we propose a modular, staged training strategy that progressively aligns linguistic, paralinguistic, and speaker characteristic information using large-scale speech-text corpora. Experimental results on TELEVAL, a multi-dimensional evaluation benchmark, demonstrate that GOAT-SLM achieves well-balanced performance across both semantic and non-semantic tasks, and outperforms existing open-source models in handling emotion, dialectal variation, and age-sensitive interactions. This work highlights the importance of modeling beyond linguistic content and advances the development of more natural, adaptive, and socially aware spoken language systems.
#### Tiny is not small enough: High-quality, low-resource facial animation models through hybrid knowledge distillation
 - **Authors:** Zhen Han, Mattias Teye, Derek Yadgaroff, Judith Bütepage
 - **Subjects:** Subjects:
Graphics (cs.GR); Machine Learning (cs.LG); Multimedia (cs.MM); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2507.18352

 - **Pdf link:** https://arxiv.org/pdf/2507.18352

 - **Abstract**
 The training of high-quality, robust machine learning models for speech-driven 3D facial animation requires a large, diverse dataset of high-quality audio-animation pairs. To overcome the lack of such a dataset, recent work has introduced large pre-trained speech encoders that are robust to variations in the input audio and, therefore, enable the facial animation model to generalize across speakers, audio quality, and languages. However, the resulting facial animation models are prohibitively large and lend themselves only to offline inference on a dedicated machine. In this work, we explore on-device, real-time facial animation models in the context of game development. We overcome the lack of large datasets by using hybrid knowledge distillation with pseudo-labeling. Given a large audio dataset, we employ a high-performing teacher model to train very small student models. In contrast to the pre-trained speech encoders, our student models only consist of convolutional and fully-connected layers, removing the need for attention context or recurrent updates. In our experiments, we demonstrate that we can reduce the memory footprint to up to 3.4 MB and required future audio context to up to 81 ms while maintaining high-quality animations. This paves the way for on-device inference, an important step towards realistic, model-driven digital characters.


by Zyzzyva0381 (Windy). 


2025-07-25
