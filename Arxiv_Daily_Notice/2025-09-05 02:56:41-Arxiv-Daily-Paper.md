# Showing new listings for Friday, 5 September 2025
Auto update papers at about 2:30am UTC (10:30am Beijing time) every weekday.


阅读 `Usage.md`了解如何使用此repo实现个性化的Arxiv论文推送

See `Usage.md` for instructions on how to personalize the repo. 


Keyword list: ['text-to-speech', 'text to speech', 'tts', 'LLM-based', 'speech', 'voice']


Excluded: []


### Today: 5papers 
#### LibriQuote: A Speech Dataset of Fictional Character Utterances for Expressive Zero-Shot Speech Synthesis
 - **Authors:** Gaspard Michel, Elena V. Epure, Christophe Cerisara
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Computation and Language (cs.CL); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2509.04072

 - **Pdf link:** https://arxiv.org/pdf/2509.04072

 - **Abstract**
 Text-to-speech (TTS) systems have recently achieved more expressive and natural speech synthesis by scaling to large speech datasets. However, the proportion of expressive speech in such large-scale corpora is often unclear. Besides, existing expressive speech corpora are typically smaller in scale and primarily used for benchmarking TTS systems. In this paper, we introduce the LibriQuote dataset, an English corpus derived from read audiobooks, designed for both fine-tuning and benchmarking expressive zero-shot TTS system. The training dataset includes 12.7K hours of read, non-expressive speech and 5.3K hours of mostly expressive speech drawn from character quotations. Each utterance in the expressive subset is supplemented with the context in which it was written, along with pseudo-labels of speech verbs and adverbs used to describe the quotation (\textit{e.g. ``he whispered softly''}). Additionally, we provide a challenging 7.5 hour test set intended for benchmarking TTS systems: given a neutral reference speech as input, we evaluate system's ability to synthesize an expressive utterance while preserving reference timbre. We validate qualitatively the test set by showing that it covers a wide range of emotions compared to non-expressive speech, along with various accents. Extensive subjective and objective evaluations show that fine-tuning a baseline TTS system on LibriQuote significantly improves its synthesized speech intelligibility, and that recent systems fail to synthesize speech as expressive and natural as the ground-truth utterances. The dataset and evaluation code are freely available. Audio samples can be found at this https URL.
#### Test-Time Adaptation for Speech Enhancement via Domain Invariant Embedding Transformation
 - **Authors:** Tobias Raichle, Niels Edinger, Bin Yang
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2509.04280

 - **Pdf link:** https://arxiv.org/pdf/2509.04280

 - **Abstract**
 Deep learning-based speech enhancement models achieve remarkable performance when test distributions match training conditions, but often degrade when deployed in unpredictable real-world environments with domain shifts. To address this challenge, we present LaDen (latent denoising), the first test-time adaptation method specifically designed for speech enhancement. Our approach leverages powerful pre-trained speech representations to perform latent denoising, approximating clean speech representations through a linear transformation of noisy embeddings. We show that this transformation generalizes well across domains, enabling effective pseudo-labeling for target domains without labeled target data. The resulting pseudo-labels enable effective test-time adaptation of speech enhancement models across diverse acoustic environments. We propose a comprehensive benchmark spanning multiple datasets with various domain shifts, including changes in noise types, speaker characteristics, and languages. Our extensive experiments demonstrate that LaDen consistently outperforms baseline methods across perceptual metrics, particularly for speaker and language domain shifts.
#### Speech-Based Cognitive Screening: A Systematic Evaluation of LLM Adaptation Strategies
 - **Authors:** Fatemeh Taherinezhad, Mohamad Javad Momeni Nezhad, Sepehr Karimi, Sina Rashidi, Ali Zolnour, Maryam Dadkhah, Yasaman Haghbin, Hossein AzadMaleki, Maryam Zolnoori
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Artificial Intelligence (cs.AI); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2509.03525

 - **Pdf link:** https://arxiv.org/pdf/2509.03525

 - **Abstract**
 Over half of US adults with Alzheimer disease and related dementias remain undiagnosed, and speech-based screening offers a scalable detection approach. We compared large language model adaptation strategies for dementia detection using the DementiaBank speech corpus, evaluating nine text-only models and three multimodal audio-text models on recordings from DementiaBank speech corpus. Adaptations included in-context learning with different demonstration selection policies, reasoning-augmented prompting, parameter-efficient fine-tuning, and multimodal integration. Results showed that class-centroid demonstrations achieved the highest in-context learning performance, reasoning improved smaller models, and token-level fine-tuning generally produced the best scores. Adding a classification head substantially improved underperforming models. Among multimodal models, fine-tuned audio-text systems performed well but did not surpass the top text-only models. These findings highlight that model adaptation strategies, including demonstration selection, reasoning design, and tuning method, critically influence speech-based dementia detection, and that properly adapted open-weight models can match or exceed commercial systems.
#### Enhancing Speech Large Language Models through Reinforced Behavior Alignment
 - **Authors:** Yansong Liu, Jiateng Li, Yuan Liu
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2509.03526

 - **Pdf link:** https://arxiv.org/pdf/2509.03526

 - **Abstract**
 The recent advancements of Large Language Models (LLMs) have spurred considerable research interest in extending their linguistic capabilities beyond text to other modalities, which leads to emergence of speech-based LLMs (SpeechLMs) with capability of processing user request in either speech or textual formats. However, owing to inter-modal discrepancies, these SpeechLMs still exhibit a significant performance gap compared to their text-based LLM counterparts in instruction-following, particularly when confronted with the dynamic and variable nature of user speech. To address this challenge, this paper introduces a framework termed Reinforced Behavior Alignment (RBA), designed to bolster the language generation proficiency of SpeechLMs. Instead of relying on supervised fine-tuning from human annotations, RBA employs a self-synthesis methodology to generate extensive, high-fidelity alignment data by a powerful teacher LLM. Then SpeechLMs is aligned its behavior with that of a teacher using a reinforcement learning-based approach. Experimental results demonstrate that this method effectively enhances the instruction-following capabilities of SpeechLMs that outperform conventional distillation baselines. Crucially, we demonstrate that RBA can be seamlessly extended to tasks such including spoken question answering and speech-to-text translation, attaining state-of-the-art performance on open benchmarks with only self-generated data.
#### SwinSRGAN: Swin Transformer-based Generative Adversarial Network for High-Fidelity Speech Super-Resolution
 - **Authors:** Jiajun Yuan, Xiaochen Wang, Yuhang Xiao, Yulin Wu, Chenhao Hu, Xueyang Lv
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2509.03913

 - **Pdf link:** https://arxiv.org/pdf/2509.03913

 - **Abstract**
 Speech super-resolution (SR) reconstructs high-frequency content from low-resolution speech signals. Existing systems often suffer from representation mismatch in two-stage mel-vocoder pipelines and from over-smoothing of hallucinated high-band content by CNN-only generators. Diffusion and flow models are computationally expensive, and their robustness across domains and sampling rates remains limited. We propose SwinSRGAN, an end-to-end framework operating on Modified Discrete Cosine Transform (MDCT) magnitudes. It is a Swin Transformer-based U-Net that captures long-range spectro-temporal dependencies with a hybrid adversarial scheme combines time-domain MPD/MSD discriminators with a multi-band MDCT discriminator specialized for the high-frequency band. We employs a sparse-aware regularizer on arcsinh-compressed MDCT to better preserve transient components. The system upsamples inputs at various sampling rates to 48 kHz in a single pass and operates in real time. On standard benchmarks, SwinSRGAN reduces objective error and improves ABX preference scores. In zero-shot tests on HiFi-TTS without fine-tuning, it outperforms NVSR and mdctGAN, demonstrating strong generalization across datasets


by Zyzzyva0381 (Windy). 


2025-09-05
