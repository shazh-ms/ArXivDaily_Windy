# Showing new listings for Thursday, 18 September 2025
Auto update papers at about 2:30am UTC (10:30am Beijing time) every weekday.


阅读 `Usage.md`了解如何使用此repo实现个性化的Arxiv论文推送

See `Usage.md` for instructions on how to personalize the repo. 


Keyword list: ['text-to-speech', 'text to speech', 'tts', 'LLM-based', 'speech', 'voice']


Excluded: []


### Today: 11papers 
#### TICL: Text-Embedding KNN For Speech In-Context Learning Unlocks Speech Recognition Abilities of Large Multimodal Models
 - **Authors:** Haolong Zheng, Yekaterina Yegorova, Mark Hasegawa-Johnson
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Artificial Intelligence (cs.AI); Computation and Language (cs.CL); Machine Learning (cs.LG); Multimedia (cs.MM)
 - **Arxiv link:** https://arxiv.org/abs/2509.13395

 - **Pdf link:** https://arxiv.org/pdf/2509.13395

 - **Abstract**
 Speech foundation models have recently demonstrated the ability to perform Speech In-Context Learning (SICL). Selecting effective in-context examples is crucial for SICL performance, yet selection methodologies remain underexplored. In this work, we propose Text-Embedding KNN for SICL (TICL), a simple pipeline that uses semantic context to enhance off-the-shelf large multimodal models' speech recognition ability without fine-tuning. Across challenging automatic speech recognition tasks, including accented English, multilingual speech, and children's speech, our method enables models to surpass zero-shot performance with up to 84.7% relative WER reduction. We conduct ablation studies to show the robustness and efficiency of our method.
#### Enhancing Speaker-Independent Dysarthric Speech Severity Classification with DSSCNet and Cross-Corpus Adaptation
 - **Authors:** Arnab Kumar Roy, Hemant Kumar Kathania, Paban Sapkota
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2509.13442

 - **Pdf link:** https://arxiv.org/pdf/2509.13442

 - **Abstract**
 Dysarthric speech severity classification is crucial for objective clinical assessment and progress monitoring in individuals with motor speech disorders. Although prior methods have addressed this task, achieving robust generalization in speaker-independent (SID) scenarios remains challenging. This work introduces DSSCNet, a novel deep neural architecture that combines Convolutional, Squeeze-Excitation (SE), and Residual network, helping it extract discriminative representations of dysarthric speech from mel spectrograms. The addition of SE block selectively focuses on the important features of the dysarthric speech, thereby minimizing loss and enhancing overall model performance. We also propose a cross-corpus fine-tuning framework for severity classification, adapted from detection-based transfer learning approaches. DSSCNet is evaluated on two benchmark dysarthric speech corpora: TORGO and UA-Speech under speaker-independent evaluation protocols: One-Speaker-Per-Severity (OSPS) and Leave-One-Speaker-Out (LOSO) protocols. DSSCNet achieves accuracies of 56.84% and 62.62% under OSPS and 63.47% and 64.18% under LOSO setting on TORGO and UA-Speech respectively outperforming existing state-of-the-art methods. Upon fine-tuning, the performance improves substantially, with DSSCNet achieving up to 75.80% accuracy on TORGO and 68.25% on UA-Speech in OSPS, and up to 77.76% and 79.44%, respectively, in LOSO. These results demonstrate the effectiveness and generalizability of DSSCNet for fine-grained severity classification across diverse dysarthric speech datasets.
#### A Distilled Low-Latency Neural Vocoder with Explicit Amplitude and Phase Prediction
 - **Authors:** Hui-Peng Du, Yang Ai, Zhen-Hua Ling
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2509.13667

 - **Pdf link:** https://arxiv.org/pdf/2509.13667

 - **Abstract**
 The majority of mainstream neural vocoders primarily focus on speech quality and generation speed, while overlooking latency, which is a critical factor in real-time applications. Excessive latency leads to noticeable delays in user interaction, severely degrading the user experience and rendering such systems impractical for real-time use. Therefore, this paper proposes DLL-APNet, a Distilled Low-Latency neural vocoder which first predicts the Amplitude and Phase spectra explicitly from input mel spectrogram and then reconstructs the speech waveform via inverse short-time Fourier transform (iSTFT). The DLL-APNet vocoder leverages causal convolutions to constrain the utilization of information to current and historical contexts, effectively minimizing latency. To mitigate speech quality degradation caused by causal constraints, a knowledge distillation strategy is proposed, where a pre-trained non-causal teacher vocoder guides intermediate feature generation of the causal student DLL-APNet vocoder. Experimental results demonstrate that the proposed DLL-APNet vocoder produces higher-quality speech than other causal vocoders, while requiring fewer computational resources. Furthermore, the proposed DLL-APNet vocoder achieves speech quality on par with mainstream non-causal neural vocoders, validating its ability to deliver both high perceptual quality and low latency.
#### A High-Quality and Low-Complexity Streamable Neural Speech Codec with Knowledge Distillation
 - **Authors:** En-Wei Zhang, Hui-Peng Du, Xiao-Hang Jiang, Yang Ai, Zhen-Hua Ling
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2509.13670

 - **Pdf link:** https://arxiv.org/pdf/2509.13670

 - **Abstract**
 While many current neural speech codecs achieve impressive reconstructed speech quality, they often neglect latency and complexity considerations, limiting their practical deployment in downstream tasks such as real-time speech communication and efficient speech compression. In our previous work, we proposed StreamCodec, which enables streamable speech coding by leveraging model causalization and a scalar-vector-combined quantization strategy, but its reconstructed quality and complexity still have room for improvement. Therefore, this paper proposes an improved iteration of StreamCodec, named StreamCodec2. The StreamCodec2 supports streamable and lightweight speech coding by adopting a fully causal architecture and reducing the convolutional channels. To compensate for the speech quality degradation caused by model causalization and pruning, we introduce a non-causal, high-complexity teacher codec to guide the training of StreamCodec2 through knowledge distillation. Experimental results demonstrate that our proposed StreamCodec2, trained with the knowledge distillation strategy, can achieve high-quality speech reconstruction while maintaining low latency (only 20 ms), low computational complexity (only 910 MFLOPs), and low model complexity (only 5.4 M parameters).
#### Summary on The Multilingual Conversational Speech Language Model Challenge: Datasets, Tasks, Baselines, and Methods
 - **Authors:** Bingshen Mu, Pengcheng Guo, Zhaokai Sun, Shuai Wang, Hexin Liu, Mingchen Shao, Lei Xie, Eng Siong Chng, Longshuai Xiao, Qiangze Feng, Daliang Wang
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2509.13785

 - **Pdf link:** https://arxiv.org/pdf/2509.13785

 - **Abstract**
 This paper summarizes the Interspeech2025 Multilingual Conversational Speech Language Model (MLC-SLM) challenge, which aims to advance the exploration of building effective multilingual conversational speech LLMs (SLLMs). We provide a detailed description of the task settings for the MLC-SLM challenge, the released real-world multilingual conversational speech dataset totaling approximately 1,604 hours, and the baseline systems for participants. The MLC-SLM challenge attracts 78 teams from 13 countries to participate, with 489 valid leaderboard results and 14 technical reports for the two tasks. We distill valuable insights on building multilingual conversational SLLMs based on submissions from participants, aiming to contribute to the advancement of the community.
#### Mixture of Low-Rank Adapter Experts in Generalizable Audio Deepfake Detection
 - **Authors:** Janne Laakkonen, Ivan Kukanov, Ville HautamÃ¤ki
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Machine Learning (cs.LG); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2509.13878

 - **Pdf link:** https://arxiv.org/pdf/2509.13878

 - **Abstract**
 Foundation models such as Wav2Vec2 excel at representation learning in speech tasks, including audio deepfake detection. However, after being fine-tuned on a fixed set of bonafide and spoofed audio clips, they often fail to generalize to novel deepfake methods not represented in training. To address this, we propose a mixture-of-LoRA-experts approach that integrates multiple low-rank adapters (LoRA) into the model's attention layers. A routing mechanism selectively activates specialized experts, enhancing adaptability to evolving deepfake attacks. Experimental results show that our method outperforms standard fine-tuning in both in-domain and out-of-domain scenarios, reducing equal error rates relative to baseline models. Notably, our best MoE-LoRA model lowers the average out-of-domain EER from 8.55\% to 6.08\%, demonstrating its effectiveness in achieving generalizable audio deepfake detection.
#### Do You Hear What I Mean? Quantifying the Instruction-Perception Gap in Instruction-Guided Expressive Text-To-Speech Systems
 - **Authors:** Yi-Cheng Lin, Huang-Cheng Chou, Tzu-Chieh Wei, Kuan-Yu Chen, Hung-yi Lee
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2509.13989

 - **Pdf link:** https://arxiv.org/pdf/2509.13989

 - **Abstract**
 Instruction-guided text-to-speech (ITTS) enables users to control speech generation through natural language prompts, offering a more intuitive interface than traditional TTS. However, the alignment between user style instructions and listener perception remains largely unexplored. This work first presents a perceptual analysis of ITTS controllability across two expressive dimensions (adverbs of degree and graded emotion intensity) and collects human ratings on speaker age and word-level emphasis attributes. To comprehensively reveal the instruction-perception gap, we provide a data collection with large-scale human evaluations, named Expressive VOice Control (E-VOC) corpus. Furthermore, we reveal that (1) gpt-4o-mini-tts is the most reliable ITTS model with great alignment between instruction and generated utterances across acoustic dimensions. (2) The 5 analyzed ITTS systems tend to generate Adult voices even when the instructions ask to use child or Elderly voices. (3) Fine-grained control remains a major challenge, indicating that most ITTS systems have substantial room for improvement in interpreting slightly different attribute instructions.
#### A Lightweight Fourier-based Network for Binaural Speech Enhancement with Spatial Cue Preservation
 - **Authors:** Xikun Lu, Yujian Ma, Xianquan Jiang, Xuelong Wang, Jinqiu Sang
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2509.14076

 - **Pdf link:** https://arxiv.org/pdf/2509.14076

 - **Abstract**
 Binaural speech enhancement faces a severe trade-off challenge, where state-of-the-art performance is achieved by computationally intensive architectures, while lightweight solutions often come at the cost of significant performance degradation. To bridge this gap, we propose the Global Adaptive Fourier Network (GAF-Net), a lightweight deep complex network that aims to establish a balance between performance and computational efficiency. The GAF-Net architecture consists of three components. First, a dual-feature encoder combining short-time Fourier transform and gammatone features enhances the robustness of acoustic representation. Second, a channel-independent globally adaptive Fourier modulator efficiently captures long-term temporal dependencies while preserving the spatial cues. Finally, a dynamic gating mechanism is implemented to reduce processing artifacts. Experimental results show that GAF-Net achieves competitive performance, particularly in terms of binaural cues (ILD and IPD error) and objective intelligibility (MBSTOI), with fewer parameters and computational cost. These results confirm that GAF-Net provides a feasible way to achieve high-fidelity binaural processing on resource-constrained devices.
#### Read to Hear: A Zero-Shot Pronunciation Assessment Using Textual Descriptions and LLMs
 - **Authors:** Yu-Wen Chen, Melody Ma, Julia Hirschberg
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2509.14187

 - **Pdf link:** https://arxiv.org/pdf/2509.14187

 - **Abstract**
 Automatic pronunciation assessment is typically performed by acoustic models trained on audio-score pairs. Although effective, these systems provide only numerical scores, without the information needed to help learners understand their errors. Meanwhile, large language models (LLMs) have proven effective in supporting language learning, but their potential for assessing pronunciation remains unexplored. In this work, we introduce TextPA, a zero-shot, Textual description-based Pronunciation Assessment approach. TextPA utilizes human-readable representations of speech signals, which are fed into an LLM to assess pronunciation accuracy and fluency, while also providing reasoning behind the assigned scores. Finally, a phoneme sequence match scoring method is used to refine the accuracy scores. Our work highlights a previously overlooked direction for pronunciation assessment. Instead of relying on supervised training with audio-score examples, we exploit the rich pronunciation knowledge embedded in written text. Experimental results show that our approach is both cost-efficient and competitive in performance. Furthermore, TextPA significantly improves the performance of conventional audio-score-trained models on out-of-domain data by offering a complementary perspective.
#### Canary-1B-v2 & Parakeet-TDT-0.6B-v3: Efficient and High-Performance Models for Multilingual ASR and AST
 - **Authors:** Monica Sekoyan, Nithin Rao Koluguri, Nune Tadevosyan, Piotr Zelasko, Travis Bartley, Nick Karpov, Jagadeesh Balam, Boris Ginsburg
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2509.14128

 - **Pdf link:** https://arxiv.org/pdf/2509.14128

 - **Abstract**
 This report introduces Canary-1B-v2, a fast, robust multilingual model for Automatic Speech Recognition (ASR) and Speech-to-Text Translation (AST). Built with a FastConformer encoder and Transformer decoder, it supports 25 languages primarily European. The model was trained on 1.7M hours of total data samples, including Granary and NeMo ASR Set 3.0, with non-speech audio added to reduce hallucinations for ASR and AST. We describe its two-stage pre-training and fine-tuning process with dynamic data balancing, as well as experiments with an nGPT encoder. Results show nGPT scales well with massive data, while FastConformer excels after fine-tuning. For timestamps, Canary-1B-v2 uses the NeMo Forced Aligner (NFA) with an auxiliary CTC model, providing reliable segment-level timestamps for ASR and AST. Evaluations show Canary-1B-v2 outperforms Whisper-large-v3 on English ASR while being 10x faster, and delivers competitive multilingual ASR and AST performance against larger models like Seamless-M4T-v2-large and LLM-based systems. We also release Parakeet-TDT-0.6B-v3, a successor to v2, offering multilingual ASR across the same 25 languages with just 600M parameters.
#### CS-FLEURS: A Massively Multilingual and Code-Switched Speech Dataset
 - **Authors:** Brian Yan, Injy Hamed, Shuichiro Shimizu, Vasista Lodagala, William Chen, Olga Iakovenko, Bashar Talafha, Amir Hussein, Alexander Polok, Kalvin Chang, Dominik Klement, Sara Althubaiti, Puyuan Peng, Matthew Wiesner, Thamar Solorio, Ahmed Ali, Sanjeev Khudanpur, Shinji Watanabe, Chih-Chen Chen, Zhen Wu, Karim Benharrak, Anuj Diwan, Samuele Cornell, Eunjung Yeo, Kwanghee Choi, Carlos Carvalho, Karen Rosero
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2509.14161

 - **Pdf link:** https://arxiv.org/pdf/2509.14161

 - **Abstract**
 We present CS-FLEURS, a new dataset for developing and evaluating code-switched speech recognition and translation systems beyond high-resourced languages. CS-FLEURS consists of 4 test sets which cover in total 113 unique code-switched language pairs across 52 languages: 1) a 14 X-English language pair set with real voices reading synthetically generated code-switched sentences, 2) a 16 X-English language pair set with generative text-to-speech 3) a 60 {Arabic, Mandarin, Hindi, Spanish}-X language pair set with the generative text-to-speech, and 4) a 45 X-English lower-resourced language pair test set with concatenative text-to-speech. Besides the four test sets, CS-FLEURS also provides a training set with 128 hours of generative text-to-speech data across 16 X-English language pairs. Our hope is that CS-FLEURS helps to broaden the scope of future code-switched speech research. Dataset link: this https URL.


by Zyzzyva0381 (Windy). 


2025-09-18
