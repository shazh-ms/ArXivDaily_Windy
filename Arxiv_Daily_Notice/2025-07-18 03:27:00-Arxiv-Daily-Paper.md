# Showing new listings for Friday, 18 July 2025
Auto update papers at about 2:30am UTC (10:30am Beijing time) every weekday.


阅读 `Usage.md`了解如何使用此repo实现个性化的Arxiv论文推送

See `Usage.md` for instructions on how to personalize the repo. 


Keyword list: ['text-to-speech', 'text to speech', 'tts', 'LLM-based', 'speech', 'voice']


Excluded: []


### Today: 12papers 
#### Enhancing In-Domain and Out-Domain EmoFake Detection via Cooperative Multilingual Speech Foundation Models
 - **Authors:** Orchid Chetia Phukan, Mohd Mujtaba Akhtar, Girish, Arun Balaji Buduru
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2507.12595

 - **Pdf link:** https://arxiv.org/pdf/2507.12595

 - **Abstract**
 In this work, we address EmoFake Detection (EFD). We hypothesize that multilingual speech foundation models (SFMs) will be particularly effective for EFD due to their pre-training across diverse languages, enabling a nuanced understanding of variations in pitch, tone, and intensity. To validate this, we conduct a comprehensive comparative analysis of state-of-the-art (SOTA) SFMs. Our results shows the superiority of multilingual SFMs for same language (in-domain) as well as cross-lingual (out-domain) evaluation. To our end, we also propose, THAMA for fusion of foundation models (FMs) motivated by related research where combining FMs have shown improved performance. THAMA leverages the complementary conjunction of tucker decomposition and hadamard product for effective fusion. With THAMA, synergized with cooperative multilingual SFMs achieves topmost performance across in-domain and out-domain settings, outperforming individual FMs, baseline fusion techniques, and prior SOTA methods.
#### UniSLU: Unified Spoken Language Understanding from Heterogeneous Cross-Task Datasets
 - **Authors:** Zhichao Sheng, Shilin Zhou, Chen Gong, Zhenghua Li
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Artificial Intelligence (cs.AI); Computation and Language (cs.CL); Multimedia (cs.MM); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2507.12951

 - **Pdf link:** https://arxiv.org/pdf/2507.12951

 - **Abstract**
 Spoken Language Understanding (SLU) plays a crucial role in speech-centric multimedia applications, enabling machines to comprehend spoken language in scenarios such as meetings, interviews, and customer service interactions. SLU encompasses multiple tasks, including Automatic Speech Recognition (ASR), spoken Named Entity Recognition (NER), and spoken Sentiment Analysis (SA). However, existing methods often rely on separate model architectures for individual tasks such as spoken NER and SA, which increases system complexity, limits cross-task interaction, and fails to fully exploit heterogeneous datasets available across tasks. To address these limitations, we propose UniSLU, a unified framework that jointly models multiple SLU tasks within a single architecture. Specifically, we propose a unified representation for diverse SLU tasks, enabling full utilization of heterogeneous datasets across multiple tasks. Built upon this representation, we propose a unified generative method that jointly models ASR, spoken NER, and SA tasks, enhancing task interactions and enabling seamless integration with large language models to harness their powerful generative capabilities. Extensive experiments on public SLU datasets demonstrate the effectiveness of our approach, achieving superior SLU performance compared to several benchmark methods, making it well-suited for real-world speech-based multimedia scenarios. We will release all code and models at github to facilitate future research.
#### AVFSNet: Audio-Visual Speech Separation for Flexible Number of Speakers with Multi-Scale and Multi-Task Learning
 - **Authors:** Daning Zhang, Ying Wei
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2507.12972

 - **Pdf link:** https://arxiv.org/pdf/2507.12972

 - **Abstract**
 Separating target speech from mixed signals containing flexible speaker quantities presents a challenging task. While existing methods demonstrate strong separation performance and noise robustness, they predominantly assume prior knowledge of speaker counts in mixtures. The limited research addressing unknown speaker quantity scenarios exhibits significantly constrained generalization capabilities in real acoustic environments. To overcome these challenges, this paper proposes AVFSNet -- an audio-visual speech separation model integrating multi-scale encoding and parallel architecture -- jointly optimized for speaker counting and multi-speaker separation tasks. The model independently separates each speaker in parallel while enhancing environmental noise adaptability through visual information integration. Comprehensive experimental evaluations demonstrate that AVFSNet achieves state-of-the-art results across multiple evaluation metrics and delivers outstanding performance on diverse datasets.
#### Task-Specific Audio Coding for Machines: Machine-Learned Latent Features Are Codes for That Machine
 - **Authors:** Anastasia Kuznetsova, Inseon Jang, Wootaek Lim, Minje Kim
 - **Subjects:** Subjects:
Sound (cs.SD); Artificial Intelligence (cs.AI); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2507.12701

 - **Pdf link:** https://arxiv.org/pdf/2507.12701

 - **Abstract**
 Neural audio codecs, leveraging quantization algorithms, have significantly impacted various speech/audio tasks. While high-fidelity reconstruction is paramount for human perception, audio coding for machines (ACoM) prioritizes efficient compression and downstream task performance, disregarding perceptual nuances. This work introduces an efficient ACoM method that can compress and quantize any chosen intermediate feature representation of an already trained speech/audio downstream model. Our approach employs task-specific loss guidance alongside residual vector quantization (RVQ) losses, providing ultra-low bitrates (i.e., less than 200 bps) with a minimal loss of the downstream model performance. The resulting tokenizer is adaptable to various bitrates and model sizes for flexible deployment. Evaluated on automatic speech recognition and audio classification, our method demonstrates its efficacy and potential for broader task and architectural applicability through appropriate regularization.
#### AudioJudge: Understanding What Works in Large Audio Model Based Speech Evaluation
 - **Authors:** Potsawee Manakul, Woody Haosheng Gan, Michael J. Ryan, Ali Sartaz Khan, Warit Sirichotedumrong, Kunat Pipatanakul, William Held, Diyi Yang
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2507.12705

 - **Pdf link:** https://arxiv.org/pdf/2507.12705

 - **Abstract**
 Current speech evaluation suffers from two critical limitations: the need and difficulty of designing specialized systems targeting individual audio characteristics, and poor correlation between automatic evaluation methods and human preferences. This work presents a systematic study of Large Audio Model (LAM) as a Judge, AudioJudge, investigating whether it can provide a unified evaluation framework that addresses both challenges. We systematically explore AudioJudge across audio characteristic detection tasks, including pronunciation, speaking rate, speaker identification and speech quality, and system-level human preference simulation for automated benchmarking. We investigate different prompt engineering strategies, finding that audio concatenation combined with in-context learning significantly improves performance across both audio characteristic detection and human preference simulation tasks. We further introduce a multi-aspect ensemble AudioJudge to enable general-purpose multi-aspect audio evaluation. This method decomposes speech assessment into specialized judges for lexical content, speech quality, and paralinguistic features, achieving up to 0.91 Spearman correlation with human preferences on our system ranking benchmark. Robustness analysis reveals that while LAMs maintain strong performance under acoustic noise, they exhibit significant verbosity and positional biases that require careful mitigation.
#### Cross-Modal Watermarking for Authentic Audio Recovery and Tamper Localization in Synthesized Audiovisual Forgeries
 - **Authors:** Minyoung Kim, Sehwan Park, Sungmin Cha, Paul Hongsuck Seo
 - **Subjects:** Subjects:
Sound (cs.SD); Multimedia (cs.MM); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2507.12723

 - **Pdf link:** https://arxiv.org/pdf/2507.12723

 - **Abstract**
 Recent advances in voice cloning and lip synchronization models have enabled Synthesized Audiovisual Forgeries (SAVFs), where both audio and visuals are manipulated to mimic a target speaker. This significantly increases the risk of misinformation by making fake content seem real. To address this issue, existing methods detect or localize manipulations but cannot recover the authentic audio that conveys the semantic content of the message. This limitation reduces their effectiveness in combating audiovisual misinformation. In this work, we introduce the task of Authentic Audio Recovery (AAR) and Tamper Localization in Audio (TLA) from SAVFs and propose a cross-modal watermarking framework to embed authentic audio into visuals before manipulation. This enables AAR, TLA, and a robust defense against misinformation. Extensive experiments demonstrate the strong performance of our method in AAR and TLA against various manipulations, including voice cloning and lip synchronization.
#### Sample-Constrained Black Box Optimization for Audio Personalization
 - **Authors:** Rajalaxmi Rajagopalan, Yu-Lin Wei, Romit Roy Choudhury
 - **Subjects:** Subjects:
Sound (cs.SD); Machine Learning (cs.LG); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2507.12773

 - **Pdf link:** https://arxiv.org/pdf/2507.12773

 - **Abstract**
 We consider the problem of personalizing audio to maximize user experience. Briefly, we aim to find a filter $h^*$, which applied to any music or speech, will maximize the user's satisfaction. This is a black-box optimization problem since the user's satisfaction function is unknown. Substantive work has been done on this topic where the key idea is to play audio samples to the user, each shaped by a different filter $h_i$, and query the user for their satisfaction scores $f(h_i)$. A family of ``surrogate" functions is then designed to fit these scores and the optimization method gradually refines these functions to arrive at the filter $\hat{h}^*$ that maximizes satisfaction. In certain applications, we observe that a second type of querying is possible where users can tell us the individual elements $h^*[j]$ of the optimal filter $h^*$. Consider an analogy from cooking where the goal is to cook a recipe that maximizes user satisfaction. A user can be asked to score various cooked recipes (e.g., tofu fried rice) or to score individual ingredients (say, salt, sugar, rice, chicken, etc.). Given a budget of $B$ queries, where a query can be of either type, our goal is to find the recipe that will maximize this user's satisfaction. Our proposal builds on Sparse Gaussian Process Regression (GPR) and shows how a hybrid approach can outperform any one type of querying. Our results are validated through simulations and real world experiments, where volunteers gave feedback on music/speech audio and were able to achieve high satisfaction levels. We believe this idea of hybrid querying opens new problems in black-box optimization and solutions can benefit other applications beyond audio personalization.
#### Autoregressive Speech Enhancement via Acoustic Tokens
 - **Authors:** Luca Della Libera, Cem Subakan, Mirco Ravanelli
 - **Subjects:** Subjects:
Sound (cs.SD); Machine Learning (cs.LG); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2507.12825

 - **Pdf link:** https://arxiv.org/pdf/2507.12825

 - **Abstract**
 In speech processing pipelines, improving the quality and intelligibility of real-world recordings is crucial. While supervised regression is the primary method for speech enhancement, audio tokenization is emerging as a promising alternative for a smooth integration with other modalities. However, research on speech enhancement using discrete representations is still limited. Previous work has mainly focused on semantic tokens, which tend to discard key acoustic details such as speaker identity. Additionally, these studies typically employ non-autoregressive models, assuming conditional independence of outputs and overlooking the potential improvements offered by autoregressive modeling. To address these gaps we: 1) conduct a comprehensive study of the performance of acoustic tokens for speech enhancement, including the effect of bitrate and noise strength; 2) introduce a novel transducer-based autoregressive architecture specifically designed for this task. Experiments on VoiceBank and Libri1Mix datasets show that acoustic tokens outperform semantic tokens in terms of preserving speaker identity, and that our autoregressive approach can further improve performance. Nevertheless, we observe that discrete representations still fall short compared to continuous ones, highlighting the need for further research in this area.
#### Best Practices and Considerations for Child Speech Corpus Collection and Curation in Educational, Clinical, and Forensic Scenarios
 - **Authors:** John Hansen, Satwik Dutta, Ellen Grand
 - **Subjects:** Subjects:
Sound (cs.SD); Computers and Society (cs.CY); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2507.12870

 - **Pdf link:** https://arxiv.org/pdf/2507.12870

 - **Abstract**
 A child's spoken ability continues to change until their adult age. Until 7-8yrs, their speech sound development and language structure evolve rapidly. This dynamic shift in their spoken communication skills and data privacy make it challenging to curate technology-ready speech corpora for children. This study aims to bridge this gap and provide researchers and practitioners with the best practices and considerations for developing such a corpus based on an intended goal. Although primarily focused on educational goals, applications of child speech data have spread across fields including clinical and forensics fields. Motivated by this goal, we describe the WHO, WHAT, WHEN, and WHERE of data collection inspired by prior collection efforts and our experience/knowledge. We also provide a guide to establish collaboration, trust, and for navigating the human subjects research protocol. This study concludes with guidelines for corpus quality check, triage, and annotation.
#### SHIELD: A Secure and Highly Enhanced Integrated Learning for Robust Deepfake Detection against Adversarial Attacks
 - **Authors:** Kutub Uddin, Awais Khan, Muhammad Umar Farooq, Khalid Malik
 - **Subjects:** Subjects:
Sound (cs.SD); Artificial Intelligence (cs.AI); Cryptography and Security (cs.CR); Machine Learning (cs.LG); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2507.13170

 - **Pdf link:** https://arxiv.org/pdf/2507.13170

 - **Abstract**
 Audio plays a crucial role in applications like speaker verification, voice-enabled smart devices, and audio conferencing. However, audio manipulations, such as deepfakes, pose significant risks by enabling the spread of misinformation. Our empirical analysis reveals that existing methods for detecting deepfake audio are often vulnerable to anti-forensic (AF) attacks, particularly those attacked using generative adversarial networks. In this article, we propose a novel collaborative learning method called SHIELD to defend against generative AF attacks. To expose AF signatures, we integrate an auxiliary generative model, called the defense (DF) generative model, which facilitates collaborative learning by combining input and output. Furthermore, we design a triplet model to capture correlations for real and AF attacked audios with real-generated and attacked-generated audios using auxiliary generative models. The proposed SHIELD strengthens the defense against generative AF attacks and achieves robust performance across various generative models. The proposed AF significantly reduces the average detection accuracy from 95.49% to 59.77% for ASVspoof2019, from 99.44% to 38.45% for In-the-Wild, and from 98.41% to 51.18% for HalfTruth for three different generative models. The proposed SHIELD mechanism is robust against AF attacks and achieves an average accuracy of 98.13%, 98.58%, and 99.57% in match, and 98.78%, 98.62%, and 98.85% in mismatch settings for the ASVspoof2019, In-the-Wild, and HalfTruth datasets, respectively.
#### Automatically assessing oral narratives of Afrikaans and isiXhosa children
 - **Authors:** R. Louw (1), E. Sharratt (1), F. de Wet (1), C. Jacobs (1), A. Smith (1), H. Kamper (1) ((1) Stellenbosch University)
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2507.13205

 - **Pdf link:** https://arxiv.org/pdf/2507.13205

 - **Abstract**
 Developing narrative and comprehension skills in early childhood is critical for later literacy. However, teachers in large preschool classrooms struggle to accurately identify students who require intervention. We present a system for automatically assessing oral narratives of preschool children in Afrikaans and isiXhosa. The system uses automatic speech recognition followed by a machine learning scoring model to predict narrative and comprehension scores. For scoring predicted transcripts, we compare a linear model to a large language model (LLM). The LLM-based system outperforms the linear model in most cases, but the linear system is competitive despite its simplicity. The LLM-based system is comparable to a human expert in flagging children who require intervention. We lay the foundation for automatic oral assessments in classrooms, giving teachers extra capacity to focus on personalised support for children's learning.
#### Voxtral
 - **Authors:** Alexander H. Liu, Andy Ehrenberg, Andy Lo, Clément Denoix, Corentin Barreau, Guillaume Lample, Jean-Malo Delignon, Khyathi Raghavi Chandu, Patrick von Platen, Pavankumar Reddy Muddireddy, Sanchit Gandhi, Soham Ghosh, Srijan Mishra, Thomas Foubert, Abhinav Rastogi, Adam Yang, Albert Q. Jiang, Alexandre Sablayrolles, Amélie Héliou, Amélie Martin, Anmol Agarwal, Antoine Roux, Arthur Darcet, Arthur Mensch, Baptiste Bout, Baptiste Rozière, Baudouin De Monicault, Chris Bamford, Christian Wallenwein, Christophe Renaudin, Clémence Lanfranchi, Darius Dabert, Devendra Singh Chaplot, Devon Mizelle, Diego de las Casas, Elliot Chane-Sane, Emilien Fugier, Emma Bou Hanna, Gabrielle Berrada, Gauthier Delerce, Gauthier Guinet, Georgii Novikov, Guillaume Martin, Himanshu Jaju, Jan Ludziejewski, Jason Rute, Jean-Hadrien Chabran, Jessica Chudnovsky, Joachim Studnia, Joep Barmentlo, Jonas Amar, Josselin Somerville Roberts, Julien Denize, Karan Saxena, Karmesh Yadav, Kartik Khandelwal, Kush Jain, Lélio Renard Lavaud, Léonard Blier, Lingxiao Zhao, Louis Martin, Lucile Saulnier, Luyu Gao, Marie Pellat, Mathilde Guillaumin, Mathis Felardos, Matthieu Dinot, Maxime Darrin, Maximilian Augustin, Mickaël Seznec, Neha Gupta, Nikhil Raghuraman, Olivier Duchenne, Patricia Wang, Patryk Saffer, Paul Jacob, Paul Wambergue, Paula Kurylowicz, Philomène Chagniot, Pierre Stock, Pravesh Agrawal, Rémi Delacourt, Romain Sauvestre, Roman Soletskyi, Sagar Vaze, Sandeep Subramanian, Saurabh Garg, Shashwat Dalal, Siddharth Gandhi, Sumukh Aithal, Szymon Antoniak, Teven Le Scao, Thibault Schueller, Thibaut Lavril, Thomas Robert, Thomas Wang, Timothée Lacroix, Tom Bewley, Valeriia Nemychnikova, Victor Paltz
 - **Subjects:** Subjects:
Sound (cs.SD); Artificial Intelligence (cs.AI); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2507.13264

 - **Pdf link:** https://arxiv.org/pdf/2507.13264

 - **Abstract**
 We present Voxtral Mini and Voxtral Small, two multimodal audio chat models. Voxtral is trained to comprehend both spoken audio and text documents, achieving state-of-the-art performance across a diverse range of audio benchmarks, while preserving strong text capabilities. Voxtral Small outperforms a number of closed-source models, while being small enough to run locally. A 32K context window enables the model to handle audio files up to 40 minutes in duration and long multi-turn conversations. We also contribute three benchmarks for evaluating speech understanding models on knowledge and trivia. Both Voxtral models are released under Apache 2.0 license.


by Zyzzyva0381 (Windy). 


2025-07-18
