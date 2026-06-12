# Showing new listings for Friday, 12 June 2026
Auto update papers at about 2:30am UTC (10:30am Beijing time) every weekday.


阅读 `Usage.md`了解如何使用此repo实现个性化的Arxiv论文推送

See `Usage.md` for instructions on how to personalize the repo. 


Keyword list: ['text-to-speech', 'text to speech', 'tts', 'LLM-based', 'speech', 'voice']


Excluded: []


### Today: 5papers 
#### Balancing ASR and diarization in end-to-end LLMs for multi-talker speech recognition
 - **Authors:** Naijun Zheng, Yuke Lin, Sanli Tian, Mengtian Li, Zhiwei Lin, Longshuai Xiao, Dandan Tu
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2606.13095

 - **Pdf link:** https://arxiv.org/pdf/2606.13095

 - **Abstract**
 Multi-talker speech recognition is often addressed by combining automatic speech recognition (ASR) and speaker diarization in a pipeline system. Recently, LLM-based approaches have shown promise by jointly modeling semantic and speaker information, but they typically require large-scale multi-talker corpora that are costly to annotate. In this paper, we investigate how to efficiently train an LLM-based system with limited real-recorded data while maintaining high accuracy in speaker attribution. We propose several strategies: (1) a dual-encoder architecture to extract semantic and speaker features, (2) a feature interleaving format to merge these features as the inputs to the LLM, (3) a length-aware speaker ID loss to enhance diarization capability, and (4) an adaptive threshold strategy for ASR loss computation to mitigate hallucinations caused by speech overlaps. These strategies balance training between ASR and diarization tasks. Our system outperforms open-source baseline approaches, achieving relative improvements of 18% on the AliMeeting corpus and 24% on the Aishell4 corpus.
#### Generating Training Targets for Real-World Speech Enhancement via Close-to-Distant Microphone Projection
 - **Authors:** Tomohiro Nakatani, Rintaro Ikeshita, Naoyuki Kamo, Marc Delcroix, Shoko Araki
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2606.13109

 - **Pdf link:** https://arxiv.org/pdf/2606.13109

 - **Abstract**
 Training neural networks (NNs) for speech enhancement (SE) in distant speech-capturing scenarios requires paired distorted and clean reference speech signals. While such data are often generated through simulation, the mismatch between simulated and real recordings significantly limits SE accuracy. To address this issue, we propose Close-to-Distant microphone Projection (C2D projection), a method that generates paired data from real recordings captured by close and distant microphones. C2D projection estimates an optimal projection matrix that transforms close-microphone inputs into clean reference signals aligned with distant-microphone recordings, while simultaneously performing denoising. We show this projection can be effectively realized using a variant of the Parametric Multichannel Wiener Filter (PMWF). Experimental results demonstrate that an NN trained with C2D-projected data outperforms the state-of-the-art Guided Source Separation (GSS) on the challenging CHiME6 dinner party ASR task under oracle diarization, when using the enhanced output from GSS as an auxiliary input to the NN.
#### Endpoint Anticipation for Low-Latency Spoken Dialogue
 - **Authors:** Sathvik Udupa, Shinji Watanabe, Petr Schwarz, Jan Cernocky
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2606.13450

 - **Pdf link:** https://arxiv.org/pdf/2606.13450

 - **Abstract**
 While low-latency interaction is critical for spoken dialogue, cascaded architectures are often bottlenecked by reactive turn-completion detection. We propose Endpoint Anticipation, shifting from reactive detection to proactive forecasting of end-of-turn signals. Our speech-based model anticipates endpoints upto 2.56 seconds in advance, enabling speculative execution of LLM and TTS pipelines on partial context. We introduce metrics to quantify the trade-off between realized latency reduction and computational redundancy. Evaluation across conversational and task-oriented datasets shows our model consistently outperforms competitive VAP-based baselines. Integration with the Unmute framework demonstrates a 505 ms average latency reduction with a 28.4% increase in speculative computation, effectively masking sequential bottlenecks to enable complex reasoning in real-time speech-to-speech interaction.
#### Adaptive Turn-Taking for Real-time Multi-Party Voice Agents
 - **Authors:** Soumyajit Mitra, Prabhat Pandey, Abhinav Jain, Shanmukha Sahith, K V Vijay Girish
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Artificial Intelligence (cs.AI); Computation and Language (cs.CL)
 - **Arxiv link:** https://arxiv.org/abs/2606.13544

 - **Pdf link:** https://arxiv.org/pdf/2606.13544

 - **Abstract**
 Turn-taking in multi-party spoken conversations remains a fundamental challenge for voice-based agents, particularly under dynamic floor competition and varying user expectations. We propose ModeratorLM, a role-playing voice agent that conditions turn-taking behavior on an explicitly assigned role in multi-party settings. The system is built on a speech large language model operating in chunk-wise streaming manner. We further introduce a reasoning-augmented variant that incorporates chain-of-thought reasoning over conversational context and the assigned role. We construct RolePlayConv, a large-scale synthetic dataset of spoken multi-party conversations with diverse assistant roles. Experiments on real-world meeting data and RolePlayConv show improved turn-taking precision by over 40% and recall by more than 70%, while substantially reducing false-positive interruptions compared to non-role-conditioned baselines.
#### A beam--membrane biomechanical vocal fold model incorporating posturing and glottal conformation
 - **Authors:** Mohamed A. Serry, Matías Zañartu, Sean D. Peterson
 - **Subjects:** Subjects:
Medical Physics (physics.med-ph); Audio and Speech Processing (eess.AS); Biological Physics (physics.bio-ph); Computational Physics (physics.comp-ph); Fluid Dynamics (physics.flu-dyn)
 - **Arxiv link:** https://arxiv.org/abs/2606.13480

 - **Pdf link:** https://arxiv.org/pdf/2606.13480

 - **Abstract**
 The posture of the vocal folds produced by laryngeal muscle activation plays a central role in determining the dynamics of voice production. Abnormal vocal fold configurations are frequently associated with inefficient phonation and a variety of voice disorders. Although diverse glottal closure patterns have been observed clinically, the biomechanical mechanisms governing their dynamic behavior and resulting phonatory characteristics remain incompletely understood. Moreover, existing numerical models that incorporate the effects of the intrinsic musculature on posturing and glottal conformation are computationally expensive, which limits their suitability for large-scale parametric investigations. In this work, we introduce a computationally inexpensive vocal fold (VF) model wherein the body and cover VF layers are treated as a composite beam and a coupled membrane, respectively. Intrinsic laryngeal muscle activation, in addition to positioning the arytenoid cartilages and cricothyroid joint, introduces moments at the boundaries of the structure that influence glottal conformation. The model produces phonatory characteristics that are qualitatively consistent with those reported in high-fidelity finite-element models and clinical studies, thereby supporting its predictive capability while offering substantial computational advantage. The proposed framework provides biomechanical insights into the influence of incomplete glottal closure on phonation dynamics and may serve as a computationally tractable tool for investigating mechanisms underlying certain voice disorders.


by Zyzzyva0381 (Windy). 


2026-06-12
