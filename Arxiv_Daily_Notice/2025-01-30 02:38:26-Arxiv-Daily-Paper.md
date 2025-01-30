# Showing new listings for Thursday, 30 January 2025
Auto update papers at about 2:30am UTC (10:30am Beijing time) every weekday.


阅读 `Usage.md`了解如何使用此repo实现个性化的Arxiv论文推送

See `Usage.md` for instructions on how to personalize the repo. 


Keyword list: ['acoustic', 'reinforcement learning', 'reverb', 'meta', 'separate', 'reconstruction', 'noise', 'enhance', 'localization', 'speech']


Excluded: []


### Today: 8papers 
#### Audio Large Language Models Can Be Descriptive Speech Quality Evaluators
 - **Authors:** Chen Chen, Yuchen Hu, Siyin Wang, Helin Wang, Zhehuai Chen, Chao Zhang, Chao-Han Huck Yang, Eng Siong Chng
 - **Subjects:** Subjects:
Sound (cs.SD); Computation and Language (cs.CL); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2501.17202

 - **Pdf link:** https://arxiv.org/pdf/2501.17202

 - **Abstract**
 An ideal multimodal agent should be aware of the quality of its input modalities. Recent advances have enabled large language models (LLMs) to incorporate auditory systems for handling various speech-related tasks. However, most audio LLMs remain unaware of the quality of the speech they process. This limitation arises because speech quality evaluation is typically excluded from multi-task training due to the lack of suitable datasets. To address this, we introduce the first natural language-based speech evaluation corpus, generated from authentic human ratings. In addition to the overall Mean Opinion Score (MOS), this corpus offers detailed analysis across multiple dimensions and identifies causes of quality degradation. It also enables descriptive comparisons between two speech samples (A/B tests) with human-like judgment. Leveraging this corpus, we propose an alignment approach with LLM distillation (ALLD) to guide the audio LLM in extracting relevant information from raw speech and generating meaningful responses. Experimental results demonstrate that ALLD outperforms the previous state-of-the-art regression model in MOS prediction, with a mean square error of 0.17 and an A/B test accuracy of 98.6%. Additionally, the generated responses achieve BLEU scores of 25.8 and 30.2 on two tasks, surpassing the capabilities of task-specific models. This work advances the comprehensive perception of speech signals by audio LLMs, contributing to the development of real-world auditory and sensory intelligent agents.
#### Summary of the NOTSOFAR-1 Challenge: Highlights and Learnings
 - **Authors:** Igor Abramovski, Alon Vinnikov, Shalev Shaer, Naoyuki Kanda, Xiaofei Wang, Amir Ivry, Eyal Krupka
 - **Subjects:** Subjects:
Sound (cs.SD); Machine Learning (cs.LG); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2501.17304

 - **Pdf link:** https://arxiv.org/pdf/2501.17304

 - **Abstract**
 The first Natural Office Talkers in Settings of Far-field Audio Recordings (NOTSOFAR-1) Challenge is a pivotal initiative that sets new benchmarks by offering datasets more representative of the needs of real-world business applications than those previously available. The challenge provides a unique combination of 280 recorded meetings across 30 diverse environments, capturing real-world acoustic conditions and conversational dynamics, and a 1000-hour simulated training dataset, synthesized with enhanced authenticity for real-world generalization, incorporating 15,000 real acoustic transfer functions. In this paper, we provide an overview of the systems submitted to the challenge and analyze the top-performing approaches, hypothesizing the factors behind their success. Additionally, we highlight promising directions left unexplored by participants. By presenting key findings and actionable insights, this work aims to drive further innovation and progress in DASR research and applications.
#### Compact Neural TTS Voices for Accessibility
 - **Authors:** Kunal Jain, Eoin Murphy, Deepanshu Gupta, Jonathan Dyke, Saumya Shah, Vasilieios Tsiaras, Petko Petkov, Alistair Conkie
 - **Subjects:** Subjects:
Sound (cs.SD); Machine Learning (cs.LG); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2501.17332

 - **Pdf link:** https://arxiv.org/pdf/2501.17332

 - **Abstract**
 Contemporary text-to-speech solutions for accessibility applications can typically be classified into two categories: (i) device-based statistical parametric speech synthesis (SPSS) or unit selection (USEL) and (ii) cloud-based neural TTS. SPSS and USEL offer low latency and low disk footprint at the expense of naturalness and audio quality. Cloud-based neural TTS systems provide significantly better audio quality and naturalness but regress in terms of latency and responsiveness, rendering these impractical for real-world applications. More recently, neural TTS models were made deployable to run on handheld devices. Nevertheless, latency remains higher than SPSS and USEL, while disk footprint prohibits pre-installation for multiple voices at once. In this work, we describe a high-quality compact neural TTS system achieving latency on the order of 15 ms with low disk footprint. The proposed solution is capable of running on low-power devices.
#### Music2Latent2: Audio Compression with Summary Embeddings and Autoregressive Decoding
 - **Authors:** Marco Pasini, Stefan Lattner, George Fazekas
 - **Subjects:** Subjects:
Sound (cs.SD); Artificial Intelligence (cs.AI); Machine Learning (cs.LG); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2501.17578

 - **Pdf link:** https://arxiv.org/pdf/2501.17578

 - **Abstract**
 Efficiently compressing high-dimensional audio signals into a compact and informative latent space is crucial for various tasks, including generative modeling and music information retrieval (MIR). Existing audio autoencoders, however, often struggle to achieve high compression ratios while preserving audio fidelity and facilitating efficient downstream applications. We introduce Music2Latent2, a novel audio autoencoder that addresses these limitations by leveraging consistency models and a novel approach to representation learning based on unordered latent embeddings, which we call summary embeddings. Unlike conventional methods that encode local audio features into ordered sequences, Music2Latent2 compresses audio signals into sets of summary embeddings, where each embedding can capture distinct global features of the input sample. This enables to achieve higher reconstruction quality at the same compression ratio. To handle arbitrary audio lengths, Music2Latent2 employs an autoregressive consistency model trained on two consecutive audio chunks with causal masking, ensuring coherent reconstruction across segment boundaries. Additionally, we propose a novel two-step decoding procedure that leverages the denoising capabilities of consistency models to further refine the generated audio at no additional cost. Our experiments demonstrate that Music2Latent2 outperforms existing continuous audio autoencoders regarding audio quality and performance on downstream tasks. Music2Latent2 paves the way for new possibilities in audio compression.
#### VoicePrompter: Robust Zero-Shot Voice Conversion with Voice Prompt and Conditional Flow Matching
 - **Authors:** Ha-Yeong Choi, Jaehan Park
 - **Subjects:** Subjects:
Sound (cs.SD); Artificial Intelligence (cs.AI); Audio and Speech Processing (eess.AS); Signal Processing (eess.SP)
 - **Arxiv link:** https://arxiv.org/abs/2501.17612

 - **Pdf link:** https://arxiv.org/pdf/2501.17612

 - **Abstract**
 Despite remarkable advancements in recent voice conversion (VC) systems, enhancing speaker similarity in zero-shot scenarios remains challenging. This challenge arises from the difficulty of generalizing and adapting speaker characteristics in speech within zero-shot environments, which is further complicated by mismatch between the training and inference processes. To address these challenges, we propose VoicePrompter, a robust zero-shot VC model that leverages in-context learning with voice prompts. VoicePrompter is composed of (1) a factorization method that disentangles speech components and (2) a DiT-based conditional flow matching (CFM) decoder that conditions on these factorized features and voice prompts. Additionally, (3) latent mixup is used to enhance in-context learning by combining various speaker features. This approach improves speaker similarity and naturalness in zero-shot VC by applying mixup to latent representations. Experimental results demonstrate that VoicePrompter outperforms existing zero-shot VC systems in terms of speaker similarity, speech intelligibility, and audio quality. Our demo is available at \url{this https URL}.
#### Cross-lingual Embedding Clustering for Hierarchical Softmax in Low-Resource Multilingual Speech Recognition
 - **Authors:** Zhengdong Yang, Qianying Liu, Sheng Li, Fei Cheng, Chenhui Chu
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2501.17615

 - **Pdf link:** https://arxiv.org/pdf/2501.17615

 - **Abstract**
 We present a novel approach centered on the decoding stage of Automatic Speech Recognition (ASR) that enhances multilingual performance, especially for low-resource languages. It utilizes a cross-lingual embedding clustering method to construct a hierarchical Softmax (H-Softmax) decoder, which enables similar tokens across different languages to share similar decoder representations. It addresses the limitations of the previous Huffman-based H-Softmax method, which relied on shallow features in token similarity assessments. Through experiments on a downsampled dataset of 15 languages, we demonstrate the effectiveness of our approach in improving low-resource multilingual ASR accuracy.
#### A computational loudness model for electrical stimulation with cochlear implants
 - **Authors:** Franklin Alvarez, Yixuan Zhang, Daniel Kipping, Waldo Nogueira
 - **Subjects:** Subjects:
Neurons and Cognition (q-bio.NC); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2501.17640

 - **Pdf link:** https://arxiv.org/pdf/2501.17640

 - **Abstract**
 Cochlear implants (CIs) are devices that restore the sense of hearing in people with severe sensorineural hearing loss. An electrode array inserted in the cochlea bypasses the natural transducer mechanism that transforms mechanical sound waves into neural activity by artificially stimulating the auditory nerve fibers with electrical pulses. The perception of sounds is possible because the brain extracts features from this neural activity, and loudness is among the most fundamental perceptual features. A computational model that uses a three-dimensional (3D) representation of the peripheral auditory system of CI users was developed to predict categorical loudness from the simulated peripheral neural activity. In contrast, current state-of-the-art computational loudness models predict loudness from the electrical pulses with minimal parametrization of the electrode-nerve interface. In the proposed model, the spikes produced in a population of auditory nerve fibers were grouped by cochlear places, a physiological representation of the auditory filters in psychoacoustics, to be transformed into loudness contribution. Then, a loudness index was obtained with a spatiotemporal integration over this loudness contribution. This index served to define the simulated threshold of hearing (THL) and most comfortable loudness (MCL) levels resembling the growth function in CI users. The performance of real CI users in loudness summation experiments was also used to validate the computational model. These experiments studied the effect of stimulation rate, electrode separation and amplitude modulation. The proposed model provides a new set of perceptual features that can be used in computational frameworks for CIs and narrows the gap between simulations and the human peripheral neural activity.
#### acoupi: An Open-Source Python Framework for Deploying Bioacoustic AI Models on Edge Devices
 - **Authors:** Aude Vuilliomenet, Santiago Martínez Balvanera, Oisin Mac Aodha, Kate E. Jones, Duncan Wilson
 - **Subjects:** Subjects:
Sound (cs.SD); Machine Learning (cs.LG); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2501.17841

 - **Pdf link:** https://arxiv.org/pdf/2501.17841

 - **Abstract**
 1. Passive acoustic monitoring (PAM) coupled with artificial intelligence (AI) is becoming an essential tool for biodiversity monitoring. Traditional PAM systems require manual data offloading and impose substantial demands on storage and computing infrastructure. The combination of on-device AI-based processing and network connectivity enables local data analysis and transmission of only relevant information, greatly reducing storage needs. However, programming these devices for robust operation is challenging, requiring expertise in embedded systems and software engineering. Despite the increase in AI-based models for bioacoustics, their full potential remains unrealized without accessible tools to deploy them on custom hardware and tailor device behaviour to specific monitoring goals. 2. To address this challenge, we develop acoupi, an open-source Python framework that simplifies the creation and deployment of smart bioacoustic devices. acoupi integrates audio recording, AI-based data processing, data management, and real-time wireless messaging into a unified and configurable framework. By modularising key elements of the bioacoustic monitoring workflow, acoupi allows users to easily customise, extend, or select specific components to fit their unique monitoring needs. 3. We demonstrate the flexibility of acoupi by integrating two bioacoustic classifiers: BirdNET, for the classification of bird species, and BatDetect2, for the classification of UK bat species. We test the reliability of acoupi over a month-long deployment of two acoupi-powered devices in a UK urban park. 4. acoupi can be deployed on low-cost hardware such as the Raspberry Pi and can be customised for various applications. acoupi standardised framework and simplified tools facilitate the adoption of AI-powered PAM systems for researchers and conservationists. acoupi is on GitHub at this https URL.


by Zyzzyva0381 (Windy). 


2025-01-30
