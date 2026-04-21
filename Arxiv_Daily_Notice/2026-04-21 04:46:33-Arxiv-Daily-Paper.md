# Showing new listings for Tuesday, 21 April 2026
Auto update papers at about 2:30am UTC (10:30am Beijing time) every weekday.


阅读 `Usage.md`了解如何使用此repo实现个性化的Arxiv论文推送

See `Usage.md` for instructions on how to personalize the repo. 


Keyword list: ['text-to-speech', 'text to speech', 'tts', 'LLM-based', 'speech', 'voice']


Excluded: []


### Today: 10papers 
#### SAND: The Challenge on Speech Analysis for Neurodegenerative Disease Assessment
 - **Authors:** Giovanna Sannino, Ivanoe De Falco, Nadia Brancati, Laura Verde, Maria Frucci, Daniel Riccio, Vincenzo Bevilacqua, Antonio Di Marino, Lucia Aruta, Valentina Virginia Iuzzolino, Gianmaria Senerchia, Myriam Spisto, Raffaele Dubbioso
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Artificial Intelligence (cs.AI); Computer Vision and Pattern Recognition (cs.CV); Machine Learning (cs.LG)
 - **Arxiv link:** https://arxiv.org/abs/2604.16445

 - **Pdf link:** https://arxiv.org/pdf/2604.16445

 - **Abstract**
 Recent advances in Artificial Intelligence (AI) and the exploration of noninvasive, objective biomarkers, such as speech signals, have encouraged the development of algorithms to support the early diagnosis of neurodegenerative diseases, including Amyotrophic Lateral Sclerosis (ALS). Voice changes in subjects suffering from ALS typically manifest as progressive dysarthria, which is a prominent neurodegenerative symptom because it affects patients as the disease progresses. Since voice signals are complex data, the development and use of advanced AI techniques are fundamental to extracting distinctive patterns from them. Validating AI algorithms for ALS diagnosis and monitoring using voice signals is challenging, particularly due to the lack of annotated reference datasets. In this work, we present the outcome of a collaboration between a multidisciplinary team of clinicians and Machine Learning experts to create both a clinically annotated validation dataset and the "Speech Analysis for Neurodegenerative Diseases" (SAND) challenge based on it. Specifically, by analyzing voice disorders, the SAND challenge provides an opportunity to develop, test, and evaluate AI models for the automatic early identification and prediction of ALS disease progression.
#### Neural Encoding Detection is Not All You Need for Synthetic Speech Detection
 - **Authors:** Luca Cuccovillo, Xin Wang, Milica Gerhardt, Patrick Aichroth
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2604.16700

 - **Pdf link:** https://arxiv.org/pdf/2604.16700

 - **Abstract**
 This paper reviews the current state and emerging trends in synthetic speech detection. It outlines the main data-driven approaches, discusses the advantages and drawbacks of focusing future research solely on neural encoding detection, and offers recommendations for promising research directions. Unlike works that introduce new detection methods or datasets, this paper aims to guide future state-of-the-art research in the field and to highlight the risk of overcommitting to approaches that may not stand the test of time.
#### Anonymization, Not Elimination: Utility-Preserved Speech Anonymization
 - **Authors:** Yunchong Xiao, Yuxiang Zhao, Ziyang Ma, Shuai Wang, Kai Yu, Jiachun Liao, Xie Chen
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2604.17000

 - **Pdf link:** https://arxiv.org/pdf/2604.17000

 - **Abstract**
 The growing reliance on large-scale speech data has made privacy protection a critical concern. However, existing anonymization approaches often degrade data utility, for example by disrupting acoustic continuity or reducing vocal diversity, which compromises the value of speech data for downstream tasks such as Automatic Speech Recognition (ASR), Text-to-Speech (TTS), and Speech Emotion Recognition (SER). Current evaluation practices are also limited, as they mainly rely on direct testing of anonymized speech with pretrained models, providing only a partial view of utility. To address these issues, we propose a novel two-stage framework that protects both linguistic content and acoustic identity while maintaining usability. For content privacy, we employ a generative speech editing model to seamlessly replace personally identifiable information (PII), and for voice privacy, we introduce F3-VA, a flow-matching-based anonymization framework with a three-stage design that produces diverse and distinct anonymized speakers. To enable a more comprehensive assessment, we evaluate privacy using both acoustic- and content-based speaker verification metrics, and assess utility by training ASR, TTS, and SER models from scratch. Experimental results show that our framework achieves stronger privacy protection with minimal utility degradation compared to baselines from the VoicePrivacy Challenge, while the proposed evaluation protocol provides a more realistic reflection of the utility of anonymized speech under privacy protection.
#### VIBE: Voice-Induced open-ended Bias Evaluation for Large Audio-Language Models via Real-World Speech
 - **Authors:** Yi-Cheng Lin, Yusuke Hirota, Sung-Feng Huang, Hung-yi Lee
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Computation and Language (cs.CL); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2604.17248

 - **Pdf link:** https://arxiv.org/pdf/2604.17248

 - **Abstract**
 Large Audio-Language Models (LALMs) are increasingly integrated into daily applications, yet their generative biases remain underexplored. Existing speech fairness benchmarks rely on synthetic speech and Multiple-Choice Questions (MCQs), both offering a fragmented view of fairness. We propose VIBE, a framework that evaluates generative bias through open-ended tasks such as personalized recommendations, using real-world human recordings. Unlike MCQs, our method allows stereotypical associations to manifest organically without predefined options, making it easily extensible to new tasks. Evaluating 11 state-of-the-art LALMs reveals systematic biases in realistic scenarios. We find that gender cues often trigger larger distributional shifts than accent cues, indicating that current LALMs reproduce social stereotypes.
#### HCFD: A Benchmark for Audio Deepfake Detection in Healthcare
 - **Authors:** Mohd Mujtaba Akhtar, Girish, Muskaan Singh
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2604.17642

 - **Pdf link:** https://arxiv.org/pdf/2604.17642

 - **Abstract**
 In this study, we present Healthcare Codec-Fake Detection (HCFD), a new task for detecting codec-fakes under pathological speech conditions. We intentionally focus on codec based synthetic speech in this work, since neural codec decoding forms a core building block in modern speech generation pipelines. First, we release Healthcare CodecFake, the first pathology-aware dataset containing paired real and NAC-synthesized speech across multipl clinical conditions and codec families. Our evaluations show that SOTA codec-fake detectors trained primarily on healthy speech perform poorly on Healthcare CodecFake, highlighting the need for HCFD-specific models. Second, we demonstrate that PaSST outperforms existing speech-based models for HCFD, benefiting from its patch-based spectro-temporal representation. Finally, we propose PHOENIX-Mamba, a geometry-aware framework that models codec-fakes as multiple self-discovered modes in hyperbolic space and achieves the strongest performance on HCFD across clinical conditions and codecs. Experiments on HCFK show that PHOENIX-Mamba (PaSST) achieves the best overall performance, reaching 97.04 Acc on E-Dep, 96.73 on E-Alz, and 96.57 on E-Dys, while maintaining strong results on Chinese with 94.41 (Dep), 94.40 (Alz), and 93.20 (Dys). This geometry-aware formulation enables self-discovered clustering of heterogeneous codec-fake modes in hyperbolic space, facilitating robust discrimination under pathological speech variability. PHOENIX-Mamba achieves topmost performance on the HCFD task across clinical conditions and codecs.
#### Prosody as Supervision: Bridging the Non-Verbal--Verbal for Multilingual Speech Emotion Recognition
 - **Authors:** Girish, Mohd Mujtaba Akhtar, Muskaan Singh
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2604.17647

 - **Pdf link:** https://arxiv.org/pdf/2604.17647

 - **Abstract**
 In this work, we introduce a paralinguistic supervision paradigm for low-resource multilingual speech emotion recognition (LRM-SER) that leverages non-verbal vocalizations to exploit prosody-centric emotion cues. Unlike conventional SER systems that rely heavily on labeled verbal speech and suffer from poor cross-lingual transfer, our approach reformulates LRM-SER as non-verbal-to-verbal transfer, where supervision from a labeled non-verbal source domain is adapted to unlabeled verbal speech across multiple target languages. To this end, we propose NOVA ARC, a geometry-aware framework that models affective structure in the Poincaré ball, discretizes paralinguistic patterns via a hyperbolic vector-quantized prosody codebook, and captures emotion intensity through a hyperbolic emotion lens. For unsupervised adaptation, NOVA-ARC performs optimal transport based prototype alignment between source emotion prototypes and target utterances, inducing soft supervision for unlabeled speech while being stabilized through consistency regularization. Experiments show that NOVA-ARC delivers the strongest performance under both non-verbal-to-verbal adaptation and the complementary verbal-to-verbal transfer setting, consistently outperforming Euclidean counterparts and strong SSL baselines. To the best of our knowledge, this work is the first to move beyond verbal-speech-centric supervision by introducing a non-verbal-to-verbal transfer paradigm for SER.
#### MINT-Bench: A Comprehensive Multilingual Benchmark for Instruction-Following Text-to-Speech
 - **Authors:** Huakang Chen, Jingbin Hu, Liumeng Xue, Qirui Zhan, Wenhao Li, Guobin Ma, Hanke Xie, Dake Guo, Linhan Ma, Yuepeng Jiang, Bengu Wu, Pengyuan Xie, Chuan Xie, Qiang Zhang, Lei Xie
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2604.17958

 - **Pdf link:** https://arxiv.org/pdf/2604.17958

 - **Abstract**
 Instruction-following text-to-speech (TTS) has emerged as an important capability for controllable and expressive speech generation, yet its evaluation remains underdeveloped due to limited benchmark coverage, weak diagnostic granularity, and insufficient multilingual support. We present \textbf{MINT-Bench}, a comprehensive multilingual benchmark for instruction-following TTS. MINT-Bench is built upon a hierarchical multi-axis taxonomy, a scalable multi-stage data construction pipeline, and a hierarchical hybrid evaluation protocol that jointly assesses content consistency, instruction following, and perceptual quality. Experiments across ten languages show that current systems remain far from solved: frontier commercial systems lead overall, while leading open-source models become highly competitive and can even outperform commercial counterparts in localized settings such as Chinese. The benchmark further reveals that harder compositional and paralinguistic controls remain major bottlenecks for current systems. We release MINT-Bench together with the data construction and evaluation toolkit to support future research on controllable, multilingual, and diagnostically grounded TTS evaluation. The leaderboard and demo are available at this https URL
#### NIM4-ASR: Towards Efficient, Robust, and Customizable Real-Time LLM-Based ASR
 - **Authors:** Yuan Xie, Jiaqi Song, Guang Qiu, Xianliang Wang, Kai Qiao, Junfeng Yuan, Shengqing Liu, Yi Zhang, Bowen Chen, Ming Lei, Jie Gao, Jie Wu
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Computation and Language (cs.CL); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2604.18105

 - **Pdf link:** https://arxiv.org/pdf/2604.18105

 - **Abstract**
 Integrating large language models (LLMs) into automatic speech recognition (ASR) has become a mainstream paradigm in recent years. Although existing LLM-based ASR models demonstrate impressive performance on public benchmarks, their training remains predominantly data-driven, leaving key practical challenges insufficiently addressed -- particularly limited downward scalability in resource-constrained deployments and hallucinations under acoustically challenging conditions. To address these issues, we present NIM4-ASR, a production-oriented LLM-based ASR framework optimized for both efficiency and robustness. Grounded in a principled delineation of functional roles between the encoder and the LLM, we redesign the multi-stage training paradigm to align each module with its intended capability boundary. Specifically, we reformulate the pre-training architecture and objective to mitigate the modality gap and improve parameter efficiency; introduce an iterative asynchronous SFT stage to preserve acoustic fidelity and constrain representation drift; and design an ASR-specialized reinforcement learning stage to further enhance recognition quality and robustness. We additionally incorporate a suite of production-oriented optimizations, including robustness under noisy and silent conditions, real-time streaming inference, and hotword customization via retrieval-augmented generation (RAG). Experiments show that NIM4-ASR achieves state-of-the-art performance on multiple public benchmarks with merely 2.3B parameters, while substantially outperforming larger-scale competitors on internal benchmarks -- particularly in entity-intensive real-world scenarios. NIM4-ASR further supports million-scale hotword customization via RAG with sub-millisecond retrieval latency, enabling efficient adaptation to emerging entities and personalized user requirements.
#### Towards Building Speech Large Language Models for Multitask Understanding in Low-Resource Languages
 - **Authors:** Mingchen Shao, Bingshen Mu, Chengyou Wang, Hai Li, Ying Yan, Zhonghua Fu, Lei Xie
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2509.14804

 - **Pdf link:** https://arxiv.org/pdf/2509.14804

 - **Abstract**
 Speech large language models (SLLMs) built on speech encoders, adapters, and LLMs demonstrate remarkable multitask understanding performance in high-resource languages such as English and Chinese. However, their effectiveness substantially degrades in low-resource languages such as Thai. This limitation arises from three factors: (1) existing commonly used speech encoders, like the Whisper family, underperform in low-resource languages and lack support for broader spoken language understanding tasks; (2) the ASR-based alignment paradigm requires training the entire SLLM, leading to high computational cost; (3) paired speech-text data in low-resource languages is scarce. To overcome these challenges in the low-resource language Thai, we introduce XLSR-Thai, the first self-supervised learning (SSL) speech encoder for Thai. It is obtained by continuously training the standard SSL XLSR model on 36,000 hours of Thai speech data. Furthermore, we propose U-Align, a speech-text alignment method that is more resource-efficient and multitask-effective than typical ASR-based alignment. Finally, we present Thai-SUP, a pipeline for generating Thai spoken language understanding data from high-resource languages, yielding the first Thai spoken language understanding dataset of over 1,000 hours. Multiple experiments demonstrate the effectiveness of our methods in building a Thai multitask-understanding SLLM. We open-source XLSR-Thai and Thai-SUP to facilitate future research.
#### MoVE: Translating Laughter and Tears via Mixture of Vocalization Experts in Speech-to-Speech Translation
 - **Authors:** Szu-Chi Chen, I-Ning Tsai, Yi-Cheng Lin, Sung-Feng Huang, Hung-yi Lee
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Artificial Intelligence (cs.AI); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2604.17435

 - **Pdf link:** https://arxiv.org/pdf/2604.17435

 - **Abstract**
 Recent Speech-to-Speech Translation (S2ST) systems achieve strong semantic accuracy yet consistently strip away non-verbal vocalizations (NVs), such as laughter and crying that convey pragmatic intent, which severely limits real-world utility. We address this via three contributions. First, we propose a synthesis pipeline for building scalable expressive datasets to overcome the data scarcity limitation. Second, we propose MoVE, a Mixture-of-LoRA-Experts architecture with expressive-specialized adapters and a soft-weighting router that blends experts for capturing hybrid expressive states. Third, we show pretrained AudioLLMs enable striking data efficiency: 30 minutes of curated data is enough for strong performance. On English-Chinese S2ST, while comparing with strong baselines, MoVE reproduces target NVs in 76% of cases and achieves the highest human-rated naturalness and emotional fidelity among all compared systems, where existing S2ST systems preserve at most 14% of NVs.


by Zyzzyva0381 (Windy). 


2026-04-21
