# Showing new listings for Thursday, 2 July 2026
Auto update papers at about 2:30am UTC (10:30am Beijing time) every weekday.


阅读 `Usage.md`了解如何使用此repo实现个性化的Arxiv论文推送

See `Usage.md` for instructions on how to personalize the repo. 


Keyword list: ['text-to-speech', 'text to speech', 'tts', 'LLM-based', 'speech', 'voice']


Excluded: []


### Today: 5papers 
#### Do Multimodal Large Language Models Need Reasoning to Classify Dementia from Speech?
 - **Authors:** Liming Wang, Neguine Rezaii, Bradford C. Dickerson, James Glass
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2607.00260

 - **Pdf link:** https://arxiv.org/pdf/2607.00260

 - **Abstract**
 Multimodal large language models (MLLMs) have emerged as a promising approach for improving the accuracy, transferability, and explainability of automatic dementia classification (ADC) systems from voice recordings. Yet it remains unclear whether their reasoning capabilities are beneficial for ADC, and how such capabilities should be leveraged. In this paper, we conduct a careful evaluation of reasoning MLLMs for ADC and show that naive strategies, such as relying on text-based rationales, can lead to hallucinated and inconsistent rationales for diagnosis and yield inferior ADC performance compared with LLM-free baselines. To overcome this limitation, we propose \textbf{De}mentia \textbf{T}hinker with Nonlinear \textbf{A}daptor and Re\textbf{i}nforcement \textbf{L}earning (DeTAiL), an adaptor-based framework that exploits the internal representations of reasoning MLLMs for improved dementia classification. Across two dementia datasets with distinct test formats and label granularities, DeTAiL consistently outperforms strong baselines and methods that rely on text-based rationales. Code and demo will be released upon acceptance.
#### From Objectives to Applications: Aligning Architectural Biases in Audio Self-Supervised Learning
 - **Authors:** Kele Xu, Yulu Fang, Boda Zhou, Yulin Sun, Qisheng Xu, Qiya Song, Jin Zhang, Cheng Yang, Huaimin Wang
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2607.00387

 - **Pdf link:** https://arxiv.org/pdf/2607.00387

 - **Abstract**
 This paper examines audio self-supervised learning (SSL) through the alignment between pretraining objectives, architectural inductive biases, and downstream applications. Rather than treating SSL methods as a chronological sequence of pretext tasks or model families, we ask how different supervisory signals shape the representations that models are expected to learn. The discussion is organized around five paradigms: auxiliary tasks, contrastive learning, generative reconstruction, discrete token prediction, and multimodal alignment. These objectives place different demands on the model, from local structural sensitivity and contrastive invariance to contextual inference, discrete semantic abstraction, and multimodal grounding. We relate these demands to the biases of CNNs, recurrent and State Space Models, Transformers, and hybrid architectures, showing how local acoustic compression, sequential state propagation, content-dependent global routing, and local--global integration support different forms of audio SSL. The same view is then used to interpret downstream applications in speech processing, environmental sound analysis, music information retrieval, medical and bioacoustic analysis, and multimodal audio understanding as practical tests of whether learned representations and architectural choices generalize across domains. We also review benchmark protocols and open challenges, including tokenization bottlenecks, long-context efficiency, robustness, and secure multimodal deployment, and discuss how codec-based tokenization and audio-language modeling extend this objective--architecture--application pipeline. The accompanying repository is released at this https URL.
#### AmbiDrop: Ambisonics-Based Array-Agnostic Neural Speech Enhancement
 - **Authors:** Michael Tatarjitzky, Vladimir Tourbabin, Boaz Rafaely
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2607.00548

 - **Pdf link:** https://arxiv.org/pdf/2607.00548

 - **Abstract**
 Multichannel Deep Neural Networks (DNNs) have significantly improved speech enhancement performance; however, they typically remain constrained by reliance on fixed microphone array geometries, leading to poor generalization on unseen or irregular configurations. Current array-agnostic approaches often rely on high-complexity architectures or massive, diverse datasets, yet they still struggle to generalize to out-of-distribution layouts. In this paper, we present an in-depth analysis of AmbiDrop, a recently proposed framework that achieves geometry independence by leveraging ideal Ambisonics as the DNN input. By employing a channel-wise dropout layer during training to simulate Ambisonics encoding errors, AmbiDrop decouples the learning process from the physical sensor arrangement. During inference, microphone signals from arbitrary array configurations are transformed into the Ambisonics domain via Ambisonics Signal Matching (ASM) before processing. Extensive experiments demonstrate that AmbiDrop maintains high robustness across a diverse suite of unseen simulated arrays and real-world recordings. Furthermore, our results show that the framework is resilient to sensor failures and remains effective even with reduced network scales, making it highly suitable for deployment on resource-constrained edge devices and versatile wearable hardware.
#### Positive-Incentive Noise Predictor for Adversarial Purification in Speaker Verification
 - **Authors:** Yibo Bai, Sizhou Chen, Michele Panariello, Hao Ma, Xiao-Lei Zhang, Xuelong Li, Massimiliano Todisco, Nicholas Evan
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2607.00899

 - **Pdf link:** https://arxiv.org/pdf/2607.00899

 - **Abstract**
 Modern automatic speaker verification (ASV) systems are vulnerable to adversarial perturbations. Diffusion-based purification has recently shown strong effectiveness against such perturbations, but its reverse denoising process requires iterative sampling and leads to high inference latency. We find that the forward noising process provides most of the robustness gain. Motivated by this observation, we reformulate adversarial purification as a learnable noising problem, and propose the Positive-Incentive Noise Predictor (PnP), the first framework that explicitly introduces positive-incentive noise ({\pi}-noise) into the purification task. PnP learns input-adaptive {\pi}-noise and mixes it with the input to improve the robustness of downstream ASV systems. Experiments on four advanced ASV backbones show that PnP effectively defends against adversarial attacks while preserving performance on natural speech. Compared with representative purification baselines, the proposed framework provides a competitive balance among defense effectiveness, impact on genuine utterances, and inference efficiency under white-box, black-box, and defender-aware adaptive attacks, with a real-time factor as low as 0.014. Moreover, PnP can be cascaded with a diffusion denoiser to further improve the perceptual quality of purified utterances. Code and purified audio examples are available at this https URL
#### Speech Playground: An Interactive Tool for Speech Analysis and Comparison
 - **Authors:** Stephen McIntosh, Daisuke Saito, Nobuaki Minematsu
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2607.00418

 - **Pdf link:** https://arxiv.org/pdf/2607.00418

 - **Abstract**
 This paper presents Speech Playground, an interactive speech visualization and comparison tool. While existing tools such as Praat are excellent, it can be cumbersome to integrate them with modern deep learning representations and use them for comparison. Speech Playground addresses this by combining a Python backend with a web-based frontend for interactive exploration of multiple feature types, including continuous, discrete, and variable-length representations. It includes TextGrid and forced alignment support together with configurable distance and alignment settings for visual and auditory comparison. Speech Playground is intended for use in speech research, representation validation, and computer-aided pronunciation training (CAPT)-oriented experimentation.


by Zyzzyva0381 (Windy). 


2026-07-02
