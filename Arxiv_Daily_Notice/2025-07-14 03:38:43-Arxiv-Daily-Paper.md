# Showing new listings for Monday, 14 July 2025
Auto update papers at about 2:30am UTC (10:30am Beijing time) every weekday.


阅读 `Usage.md`了解如何使用此repo实现个性化的Arxiv论文推送

See `Usage.md` for instructions on how to personalize the repo. 


Keyword list: ['text-to-speech', 'text to speech', 'tts', 'LLM-based', 'speech', 'voice']


Excluded: []


### Today: 11papers 
#### DARAS: Dynamic Audio-Room Acoustic Synthesis for Blind Room Impulse Response Estimation
 - **Authors:** Chunxi Wang, Maoshen Jia, Wenyu Jin
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2507.08135

 - **Pdf link:** https://arxiv.org/pdf/2507.08135

 - **Abstract**
 Room Impulse Responses (RIRs) accurately characterize acoustic properties of indoor environments and play a crucial role in applications such as speech enhancement, speech recognition, and audio rendering in augmented reality (AR) and virtual reality (VR). Existing blind estimation methods struggle to achieve practical accuracy. To overcome this challenge, we propose the dynamic audio-room acoustic synthesis (DARAS) model, a novel deep learning framework that is explicitly designed for blind RIR estimation from monaural reverberant speech signals. First, a dedicated deep audio encoder effectively extracts relevant nonlinear latent space features. Second, the Mamba-based self-supervised blind room parameter estimation (MASS-BRPE) module, utilizing the efficient Mamba state space model (SSM), accurately estimates key room acoustic parameters and features. Third, the system incorporates a hybrid-path cross-attention feature fusion module, enhancing deep integration between audio and room acoustic features. Finally, our proposed dynamic acoustic tuning (DAT) decoder adaptively segments early reflections and late reverberation to improve the realism of synthesized RIRs. Experimental results, including a MUSHRA-based subjective listening study, demonstrate that DARAS substantially outperforms existing baseline models, providing a robust and effective solution for practical blind RIR estimation in real-world acoustic environments.
#### RawTFNet: A Lightweight CNN Architecture for Speech Anti-spoofing
 - **Authors:** Yang Xiao, Ting Dang, Rohan Kumar Das
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2507.08227

 - **Pdf link:** https://arxiv.org/pdf/2507.08227

 - **Abstract**
 Automatic speaker verification (ASV) systems are often affected by spoofing attacks. Recent transformer-based models have improved anti-spoofing performance by learning strong feature representations. However, these models usually need high computing power. To address this, we introduce RawTFNet, a lightweight CNN model designed for audio signals. The RawTFNet separates feature processing along time and frequency dimensions, which helps to capture the fine-grained details of synthetic speech. We tested RawTFNet on the ASVspoof 2021 LA and DF evaluation datasets. The results show that RawTFNet reaches comparable performance to that of the state-of-the-art models, while also using fewer computing resources. The code and models will be made publicly available.
#### RepeaTTS: Towards Feature Discovery through Repeated Fine-Tuning
 - **Authors:** Atli Sigurgeirsson, Simon King
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Artificial Intelligence (cs.AI); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2507.08012

 - **Pdf link:** https://arxiv.org/pdf/2507.08012

 - **Abstract**
 A Prompt-based Text-To-Speech model allows a user to control different aspects of speech, such as speaking rate and perceived gender, through natural language instruction. Although user-friendly, such approaches are on one hand constrained: control is limited to acoustic features exposed to the model during training, and too flexible on the other: the same inputs yields uncontrollable variation that are reflected in the corpus statistics. We investigate a novel fine-tuning regime to address both of these issues at the same time by exploiting the uncontrollable variance of the model. Through principal component analysis of thousands of synthesised samples, we determine latent features that account for the highest proportion of the output variance and incorporate them as new labels for secondary fine-tuning. We evaluate the proposed methods on two models trained on an expressive Icelandic speech corpus, one with emotional disclosure and one without. In the case of the model without emotional disclosure, the method yields both continuous and discrete features that improve overall controllability of the model.
#### Modèle physique variationnel pour l'estimation de réponses impulsionnelles de salles
 - **Authors:** Louis Lalay (LTCI, IP Paris, S2A), Mathieu Fontaine (LTCI, IP Paris, S2A), Roland Badeau (S2A, LTCI, IP Paris)
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS); Signal Processing (eess.SP); Classical Physics (physics.class-ph)
 - **Arxiv link:** https://arxiv.org/abs/2507.08051

 - **Pdf link:** https://arxiv.org/pdf/2507.08051

 - **Abstract**
 Room impulse response estimation is essential for tasks like speech dereverberation, which improves automatic speech recognition. Most existing methods rely on either statistical signal processing or deep neural networks designed to replicate signal processing principles. However, combining statistical and physical modeling for RIR estimation remains largely unexplored. This paper proposes a novel approach integrating both aspects through a theoretically grounded model. The RIR is decomposed into interpretable parameters: white Gaussian noise filtered by a frequency-dependent exponential decay (e.g. modeling wall absorption) and an autoregressive filter (e.g. modeling microphone response). A variational free-energy cost function enables practical parameter estimation. As a proof of concept, we show that given dry and reverberant speech signals, the proposed method outperforms classical deconvolution in noisy environments, as validated by objective metrics.
#### Audio Flamingo 3: Advancing Audio Intelligence with Fully Open Large Audio Language Models
 - **Authors:** Arushi Goel, Sreyan Ghosh, Jaehyeon Kim, Sonal Kumar, Zhifeng Kong, Sang-gil Lee, Chao-Han Huck Yang, Ramani Duraiswami, Dinesh Manocha, Rafael Valle, Bryan Catanzaro
 - **Subjects:** Subjects:
Sound (cs.SD); Artificial Intelligence (cs.AI); Computation and Language (cs.CL); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2507.08128

 - **Pdf link:** https://arxiv.org/pdf/2507.08128

 - **Abstract**
 We present Audio Flamingo 3 (AF3), a fully open state-of-the-art (SOTA) large audio-language model that advances reasoning and understanding across speech, sound, and music. AF3 introduces: (i) AF-Whisper, a unified audio encoder trained using a novel strategy for joint representation learning across all 3 modalities of speech, sound, and music; (ii) flexible, on-demand thinking, allowing the model to do chain-of-thought-type reasoning before answering; (iii) multi-turn, multi-audio chat; (iv) long audio understanding and reasoning (including speech) up to 10 minutes; and (v) voice-to-voice interaction. To enable these capabilities, we propose several large-scale training datasets curated using novel strategies, including AudioSkills-XL, LongAudio-XL, AF-Think, and AF-Chat, and train AF3 with a novel five-stage curriculum-based training strategy. Trained on only open-source audio data, AF3 achieves new SOTA results on over 20+ (long) audio understanding and reasoning benchmarks, surpassing both open-weight and closed-source models trained on much larger datasets.
#### Active Learning for Text-to-Speech Synthesis with Informative Sample Collection
 - **Authors:** Kentaro Seki, Shinnosuke Takamichi, Takaaki Saeki, Hiroshi Saruwatari
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2507.08319

 - **Pdf link:** https://arxiv.org/pdf/2507.08319

 - **Abstract**
 The construction of high-quality datasets is a cornerstone of modern text-to-speech (TTS) systems. However, the increasing scale of available data poses significant challenges, including storage constraints. To address these issues, we propose a TTS corpus construction method based on active learning. Unlike traditional feed-forward and model-agnostic corpus construction approaches, our method iteratively alternates between data collection and model training, thereby focusing on acquiring data that is more informative for model improvement. This approach enables the construction of a data-efficient corpus. Experimental results demonstrate that the corpus constructed using our method enables higher-quality speech synthesis than corpora of the same size.
#### Enforcing Speech Content Privacy in Environmental Sound Recordings using Segment-wise Waveform Reversal
 - **Authors:** Modan Tailleur, Mathieu Lagrange, Pierre Aumond, Vincent Tourre
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2507.08412

 - **Pdf link:** https://arxiv.org/pdf/2507.08412

 - **Abstract**
 Environmental sound recordings often contain intelligible speech, raising privacy concerns that limit analysis, sharing and reuse of data. In this paper, we introduce a method that renders speech unintelligible while preserving both the integrity of the acoustic scene, and the overall audio quality. Our approach involves reversing waveform segments to distort speech content. This process is enhanced through a voice activity detection and speech separation pipeline, which allows for more precise targeting of speech. In order to demonstrate the effectivness of the proposed approach, we consider a three-part evaluation protocol that assesses: 1) speech intelligibility using Word Error Rate (WER), 2) sound sources detectability using Sound source Classification Accuracy-Drop (SCAD) from a widely used pre-trained model, and 3) audio quality using the Fréchet Audio Distance (FAD), computed with our reference dataset that contains unaltered speech. Experiments on this simulated evaluation dataset, which consists of linear mixtures of speech and environmental sound scenes, show that our method achieves satisfactory speech intelligibility reduction (97.9% WER), minimal degradation of the sound sources detectability (2.7% SCAD), and high perceptual quality (FAD of 1.40). An ablation study further highlights the contribution of each component of the pipeline. We also show that incorporating random splicing to our speech content privacy enforcement method can enhance the algorithm's robustness to attempt to recover the clean speech, at a slight cost of audio quality.
#### MIDI-VALLE: Improving Expressive Piano Performance Synthesis Through Neural Codec Language Modelling
 - **Authors:** Jingjing Tang, Xin Wang, Zhe Zhang, Junichi Yamagishi, Geraint Wiggins, George Fazekas
 - **Subjects:** Subjects:
Sound (cs.SD); Artificial Intelligence (cs.AI); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2507.08530

 - **Pdf link:** https://arxiv.org/pdf/2507.08530

 - **Abstract**
 Generating expressive audio performances from music scores requires models to capture both instrument acoustics and human interpretation. Traditional music performance synthesis pipelines follow a two-stage approach, first generating expressive performance MIDI from a score, then synthesising the MIDI into audio. However, the synthesis models often struggle to generalise across diverse MIDI sources, musical styles, and recording environments. To address these challenges, we propose MIDI-VALLE, a neural codec language model adapted from the VALLE framework, which was originally designed for zero-shot personalised text-to-speech (TTS) synthesis. For performance MIDI-to-audio synthesis, we improve the architecture to condition on a reference audio performance and its corresponding MIDI. Unlike previous TTS-based systems that rely on piano rolls, MIDI-VALLE encodes both MIDI and audio as discrete tokens, facilitating a more consistent and robust modelling of piano performances. Furthermore, the model's generalisation ability is enhanced by training on an extensive and diverse piano performance dataset. Evaluation results show that MIDI-VALLE significantly outperforms a state-of-the-art baseline, achieving over 75% lower Frechet Audio Distance on the ATEPP and Maestro datasets. In the listening test, MIDI-VALLE received 202 votes compared to 58 for the baseline, demonstrating improved synthesis quality and generalisation across diverse performance MIDI inputs.
#### Unlocking Speech Instruction Data Potential with Query Rewriting
 - **Authors:** Yonghua Hei, Yibo Yan, Shuliang Liu, Huiyu Zhou, Linfeng Zhang, Xuming Hu
 - **Subjects:** Subjects:
Artificial Intelligence (cs.AI); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2507.08603

 - **Pdf link:** https://arxiv.org/pdf/2507.08603

 - **Abstract**
 End-to-end Large Speech Language Models~(\textbf{LSLMs}) demonstrate strong potential in response latency and speech comprehension capabilities, showcasing general intelligence across speech understanding tasks. However, the ability to follow speech instructions has not been fully realized due to the lack of datasets and heavily biased training tasks. Leveraging the rich ASR datasets, previous approaches have used Large Language Models~(\textbf{LLMs}) to continue the linguistic information of speech to construct speech instruction datasets. Yet, due to the gap between LLM-generated results and real human responses, the continuation methods further amplify these shortcomings. Given the high costs of collecting and annotating speech instruction datasets by humans, using speech synthesis to construct large-scale speech instruction datasets has become a balanced and robust alternative. Although modern Text-To-Speech~(\textbf{TTS}) models have achieved near-human-level synthesis quality, it is challenging to appropriately convert out-of-distribution text instruction to speech due to the limitations of the training data distribution in TTS models. To address this issue, we propose a query rewriting framework with multi-LLM knowledge fusion, employing multiple agents to annotate and validate the synthesized speech, making it possible to construct high-quality speech instruction datasets without relying on human annotation. Experiments show that this method can transform text instructions into distributions more suitable for TTS models for speech synthesis through zero-shot rewriting, increasing data usability from 72\% to 93\%. It also demonstrates unique advantages in rewriting tasks that require complex knowledge and context-related abilities.
#### Phoneme-Level Analysis for Person-of-Interest Speech Deepfake Detection
 - **Authors:** Davide Salvi, Viola Negroni, Sara Mandelli, Paolo Bestagini, Stefano Tubaro
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2507.08626

 - **Pdf link:** https://arxiv.org/pdf/2507.08626

 - **Abstract**
 Recent advances in generative AI have made the creation of speech deepfakes widely accessible, posing serious challenges to digital trust. To counter this, various speech deepfake detection strategies have been proposed, including Person-of-Interest (POI) approaches, which focus on identifying impersonations of specific individuals by modeling and analyzing their unique vocal traits. Despite their excellent performance, the existing methods offer limited granularity and lack interpretability. In this work, we propose a POI-based speech deepfake detection method that operates at the phoneme level. Our approach decomposes reference audio into phonemes to construct a detailed speaker profile. In inference, phonemes from a test sample are individually compared against this profile, enabling fine-grained detection of synthetic artifacts. The proposed method achieves comparable accuracy to traditional approaches while offering superior robustness and interpretability, key aspects in multimedia forensics. By focusing on phoneme analysis, this work explores a novel direction for explainable, speaker-centric deepfake detection.
#### On Barriers to Archival Audio Processing
 - **Authors:** Peter Sullivan, Muhammad Abdul-Mageed
 - **Subjects:** Subjects:
Sound (cs.SD); Artificial Intelligence (cs.AI); Computation and Language (cs.CL); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2507.08768

 - **Pdf link:** https://arxiv.org/pdf/2507.08768

 - **Abstract**
 In this study, we leverage a unique UNESCO collection of mid-20th century radio recordings to probe the robustness of modern off-the-shelf language identification (LID) and speaker recognition (SR) methods, especially with respect to the impact of multilingual speakers and cross-age recordings. Our findings suggest that LID systems, such as Whisper, are increasingly adept at handling second-language and accented speech. However, speaker embeddings remain a fragile component of speech processing pipelines that is prone to biases related to the channel, age, and language. Issues which will need to be overcome should archives aim to employ SR methods for speaker indexing.


by Zyzzyva0381 (Windy). 


2025-07-14
