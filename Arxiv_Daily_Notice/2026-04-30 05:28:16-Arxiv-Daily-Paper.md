# Showing new listings for Thursday, 30 April 2026
Auto update papers at about 2:30am UTC (10:30am Beijing time) every weekday.


阅读 `Usage.md`了解如何使用此repo实现个性化的Arxiv论文推送

See `Usage.md` for instructions on how to personalize the repo. 


Keyword list: ['text-to-speech', 'text to speech', 'tts', 'LLM-based', 'speech', 'voice']


Excluded: []


### Today: 7papers 
#### One Voice, Many Tongues: Cross-Lingual Voice Cloning for Scientific Speech
 - **Authors:** Amanuel Gizachew Abebe, Yasmin Moslem
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Computation and Language (cs.CL)
 - **Arxiv link:** https://arxiv.org/abs/2604.26136

 - **Pdf link:** https://arxiv.org/pdf/2604.26136

 - **Abstract**
 Preserving a speaker's voice identity while generating speech in a different language remains a fundamental challenge in spoken language technology, particularly in specialized domains such as scientific communication. In this paper, we address this challenge through our system submission to the International Conference on Spoken Language Translation (IWSLT 2026), the Cross-Lingual Voice Cloning shared task. First, we evaluate several state-of-the-art voice cloning models for cross-lingual speech generation of scientific texts in Arabic, Chinese, and French. Then, we build voice cloning systems based on the OmniVoice foundation model. We employ data augmentation via multi-model ensemble distillation from the ACL 60/60 corpus. We investigate the effect of using this synthetic data for fine-tuning, demonstrating consistent improvements in intelligibility (WER and CER) across languages while preserving speaker similarity.
#### DiffAnon: Diffusion-based Prosody Control for Voice Anonymization
 - **Authors:** Ismail Rasim Ulgen, Zexin Cai, Nicholas Andrews, Philipp Koehn, Berrak Sisman
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Machine Learning (cs.LG); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2604.26281

 - **Pdf link:** https://arxiv.org/pdf/2604.26281

 - **Abstract**
 To preserve or not to preserve prosody is a central question in voice anonymization. Prosody conveys meaning and affect, yet is tightly coupled with speaker identity. Existing methods either discard prosody for privacy or lack a principled mechanism to control the utility-privacy trade-off, operating at fixed design points. We propose DiffAnon, a diffusion-based anonymization method with classifier-free guidance (CFG) that provides explicit, continuous inference-time control over prosody preservation. DiffAnon refines acoustic detail over semantic embeddings of an RVQ codec, enabling smooth interpolation between anonymization strength and prosodic fidelity within a single model. To the best of our knowledge, it is the first voice anonymization framework to provide structured, interpolatable inference-time prosody control. Experiments demonstrate structured trade-off behavior, achieving strong utility while maintaining competitive privacy across controllable operating points.
#### SPG-Codec: Exploring the Role and Boundaries of Semantic Priors in Ultra-Low-Bitrate Neural Speech Coding
 - **Authors:** Mingyu Zhao, Zijian Lin, Kun Wei, Zhiyong Wu
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2604.26296

 - **Pdf link:** https://arxiv.org/pdf/2604.26296

 - **Abstract**
 Conventional neural speech codecs suffer from severe intelligibility degradation at ultra-low bitrates, where the bottleneck transitions from acoustic distortion to semantic loss. To address this issue, this paper conducts a systematic investigation into the role and fundamental limits of integrating frozen semantic priors -- specifically HuBERT and Whisper -- into neural speech coding. We introduce and quantitatively validate a novel Semantic Retirement phenomenon: while semantic constraints reduce the Word Error Rate (WER) by up to ~10% relatively at 1.5 kbps, their benefits rapidly diminish beyond 6 kbps, indicating a practical capacity boundary. We further uncover a clear trade-off between different prior types: acoustic-rich priors (HuBERT) better preserve prosodic and timbral details, whereas high-level linguistic priors (Whisper) effectively suppress phonetic hallucinations in noisy environments (reducing hallucination rates by 26 percent) and substantially narrow the generalization gap for unseen speakers. Building on these findings, we propose a bitrate-aware regulation strategy that dynamically adjusts prior strength to optimize the trade-off between semantic consistency and perceptual naturalness. Extensive experimental evaluations confirm that our approach achieves competitive intelligibility and noise robustness compared to existing baselines, offering a principled pathway toward ultra-low-bitrate generative speech coding.
#### Dual-LoRA: Parameter-Efficient Adversarial Disentanglement for Cross-Lingual Speaker Verification
 - **Authors:** Qituan Shangguan, Junhao Du, Kunyang Peng, Feng Xue, Hui Zhang, Xinsheng Wang, Kai Yu, Shuai Wang
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2604.26327

 - **Pdf link:** https://arxiv.org/pdf/2604.26327

 - **Abstract**
 Cross-lingual speaker verification suffers from severe language-speaker entanglement. This causes systematic degradation in the hardest scenario: correctly accepting utterances from the same speaker across different languages while rejecting those from different speakers sharing the same language. Standard adversarial disentanglement degrades speaker discriminability; blind discriminators inadvertently penalize speaker-discriminative traits that merely correlate with language. To address this, we propose Dual-LoRA, injecting trainable task-factorized LoRA adapters into a frozen pre-trained backbone. Our core innovation is a Language-Anchored Adversary: by grounding the discriminator with an explicit language branch, adversarial gradients target true linguistic cues rather than arbitrary correlations, preserving essential speaker characteristics. Evaluated on the TidyVoice benchmark, our system achieves a 0.91% validation EER and achieves 3rd place in the official challenge.
#### The False Resonance: A Critical Examination of Emotion Embedding Similarity for Speech Generation Evaluation
 - **Authors:** Yun-Shao Tsai, Yi-Cheng Lin, Huang-Cheng Chou, Tzu-Wen Hsu, Yun-Man Hsu, Chun Wei Chen, Shrikanth Narayanan, Hung-yi Lee
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Computation and Language (cs.CL)
 - **Arxiv link:** https://arxiv.org/abs/2604.26347

 - **Pdf link:** https://arxiv.org/pdf/2604.26347

 - **Abstract**
 Objective metrics for emotional expressiveness are vital for speech generation, particularly in expressive synthesis and voice conversion requiring emotional prosody transfer. To quantify this, the field widely relies on emotion similarity between reference and generated samples. This approach computes cosine similarity of embeddings from encoders like emotion2vec, assuming they capture affective cues despite linguistic and speaker variations. We challenge this assumption through controlled adversarial tasks and human alignment tests. Despite high classification accuracy, these latent spaces are unsuitable for zero-shot similarity evaluation. Representational limitations cause linguistic and speaker interference to overshadow emotional features, degrading discriminative ability. Consequently, the metric misaligns with human perception. This acoustic vulnerability reveals it rewards acoustic mimicry over genuine emotional synthesis.
#### Speech Emotion Recognition Using MFCC Features and LSTM-Based Deep Learning Model
 - **Authors:** Adelekun Oluwademilade, Ademola Adedamola, Abiola Abdulhakeem, Akinpelu Azeezat, Eraiyetan Israel, Omotosho Oluwadunsin, Ibenye Ikechukwu, Ayuba Muhammad, Olusanya Olamide, Kamorudeen Amuda
 - **Subjects:** Subjects:
Sound (cs.SD); Artificial Intelligence (cs.AI); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2604.25938

 - **Pdf link:** https://arxiv.org/pdf/2604.25938

 - **Abstract**
 Speech Emotion Recognition (SER) is the use of machines to detect the emotional state of humans based on the speech, which is gaining importance in natural human-computer interaction. Speech is a very valuable source of information, as emotions modify the patterns of speech; pitch, energy and even timing. Nonetheless, SER is not an easy task because speakers are not constant, and situations vary when recording and the sound similarity between specific feelings. In this work, the author introduces a speech emotion recognition system relying on the Mel-Frequency Cepstral Coefficient and Long Short-Term Memory (LSTM) neural network, as a feature extraction method. The Toronto Emotional Speech Set (TESS) speech signal was pre-processed, and transformed into MFCC features to understand the important aspects in terms of time. The resultant features were then introduced to LSTM model, which is able to learn long term features of sequential audio data. The trained model was measured over several emotion classes occurring in the dataset. As seen in the results of experiments, the proposed MFCC-LSTM approach succeeds in capturing the patterns of emotions in speech and provides highly realistic classifications in all the chosen emotion classifications. This study presents a speech emotion recognition system using Mel-Frequency Cepstral Coefficients (MFCCs) as features and a deep learning LSTM classifier. A Support Vector Machine (SVM) with an RBF kernel served as a classical baseline, achieving 98% accuracy, against which the proposed LSTM model, achieving 99% accuracy, was validated. Overall, it is possible to confirm that LSTM-based architectures can be used to address the task of speech emotion recognition. Actual applications of the proposed system may be virtual assistants and mental health surveillance.
#### Recurrence-Based Nonlinear Vocal Dynamics as Digital Biomarkers for Depression Detection from Conversational Speech
 - **Authors:** Himadri S Samanta
 - **Subjects:** Subjects:
Sound (cs.SD); Machine Learning (cs.LG); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2604.26242

 - **Pdf link:** https://arxiv.org/pdf/2604.26242

 - **Abstract**
 Digital biomarkers for depression have largely relied on static acoustic descriptors, pooled summary statistics, or conventional machine learning representations. Such approaches may miss nonlinear temporal organization embedded in conversational vocal dynamics. We hypothesized that depression is associated with altered recurrence structure in vocal state trajectories, reflecting changes in how the vocal system revisits acoustic states over time. Using the depression subset of the DAIC-WOZ corpus with 142 labeled participants, we modeled frame-level COVAREP trajectories as nonlinear dynamical systems and derived recurrence-based biomarkers from 74 vocal channels. Logistic regression with feature selection and stratified cross-validation evaluated classification performance. Recurrence-based biomarkers achieved a mean cross-validated AUC of 0.689, exceeding static acoustic baselines, entropy-dynamics features, Hurst exponent features, determinism features, and Lyapunov-like instability proxies. Permutation testing indicated statistical significance with $p=0.004$. Pooled cross-validated predictions yielded AUC 0.665 with a 95\% bootstrap confidence interval of [0.568, 0.758]. These findings suggest that depression may be characterized by altered recurrence structure in conversational vocal dynamics and support nonlinear state-space analysis as a promising direction for digital psychiatric biomarkers.


by Zyzzyva0381 (Windy). 


2026-04-30
