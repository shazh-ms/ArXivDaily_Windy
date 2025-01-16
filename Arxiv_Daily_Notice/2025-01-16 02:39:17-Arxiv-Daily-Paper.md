# Showing new listings for Thursday, 16 January 2025
Auto update papers at about 2:30am UTC (10:30am Beijing time) every weekday.


阅读 `Usage.md`了解如何使用此repo实现个性化的Arxiv论文推送

See `Usage.md` for instructions on how to personalize the repo. 


Keyword list: ['acoustic', 'reinforcement learning', 'reverb', 'meta', 'separate', 'reconstruction', 'noise', 'enhance', 'localization', 'speech']


Excluded: []


### Today: 7papers 
#### SEAL: Speaker Error Correction using Acoustic-conditioned Large Language Models
 - **Authors:** Anurag Kumar, Rohit Paturi, Amber Afshan, Sundararajan Srinivasan
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Artificial Intelligence (cs.AI); Computation and Language (cs.CL); Machine Learning (cs.LG); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2501.08421

 - **Pdf link:** https://arxiv.org/pdf/2501.08421

 - **Abstract**
 Speaker Diarization (SD) is a crucial component of modern end-to-end ASR pipelines. Traditional SD systems, which are typically audio-based and operate independently of ASR, often introduce speaker errors, particularly during speaker transitions and overlapping speech. Recently, language models including fine-tuned large language models (LLMs) have shown to be effective as a second-pass speaker error corrector by leveraging lexical context in the transcribed output. In this work, we introduce a novel acoustic conditioning approach to provide more fine-grained information from the acoustic diarizer to the LLM. We also show that a simpler constrained decoding strategy reduces LLM hallucinations, while avoiding complicated post-processing. Our approach significantly reduces the speaker error rates by 24-43% across Fisher, Callhome, and RT03-CTS datasets, compared to the first-pass Acoustic SD.
#### Speech Synthesis along Perceptual Voice Quality Dimensions
 - **Authors:** Frederik Rautenberg, Michael Kuhlmann, Fritz Seebauer, Jana Wiechmann, Petra Wagner, Reinhold Haeb-Umbach
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2501.08791

 - **Pdf link:** https://arxiv.org/pdf/2501.08791

 - **Abstract**
 While expressive speech synthesis or voice conversion systems mainly focus on controlling or manipulating abstract prosodic characteristics of speech, such as emotion or accent, we here address the control of perceptual voice qualities (PVQs) recognized by phonetic experts, which are speech properties at a lower level of abstraction. The ability to manipulate PVQs can be a valuable tool for teaching speech pathologists in training or voice actors. In this paper, we integrate a Conditional Continuous-Normalizing-Flow-based method into a Text-to-Speech system to modify perceptual voice attributes on a continuous scale. Unlike previous approaches, our system avoids direct manipulation of acoustic correlates and instead learns from examples. We demonstrate the system's capability by manipulating four voice qualities: Roughness, breathiness, resonance and weight. Phonetic experts evaluated these modifications, both for seen and unseen speaker conditions. The results highlight both the system's strengths and areas for improvement.
#### Selective Attention Merging for low resource tasks: A case study of Child ASR
 - **Authors:** Natarajan Balaji Shankar, Zilai Wang, Eray Eren, Abeer Alwan
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2501.08468

 - **Pdf link:** https://arxiv.org/pdf/2501.08468

 - **Abstract**
 While Speech Foundation Models (SFMs) excel in various speech tasks, their performance for low-resource tasks such as child Automatic Speech Recognition (ASR) is hampered by limited pretraining data. To address this, we explore different model merging techniques to leverage knowledge from models trained on larger, more diverse speech corpora. This paper also introduces Selective Attention (SA) Merge, a novel method that selectively merges task vectors from attention matrices to enhance SFM performance on low-resource tasks. Experiments on the MyST database show significant reductions in relative word error rate of up to 14%, outperforming existing model merging and data augmentation techniques. By combining data augmentation techniques with SA Merge, we achieve a new state-of-the-art WER of 8.69 on the MyST database for the Whisper-small model, highlighting the potential of SA Merge for improving low-resource ASR.
#### Towards Lightweight and Stable Zero-shot TTS with Self-distilled Representation Disentanglement
 - **Authors:** Qianniu Chen, Xiaoyang Hao, Bowen Li, Yue Liu, Li Lu
 - **Subjects:** Subjects:
Sound (cs.SD); Artificial Intelligence (cs.AI); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2501.08566

 - **Pdf link:** https://arxiv.org/pdf/2501.08566

 - **Abstract**
 Zero-shot Text-To-Speech (TTS) synthesis shows great promise for personalized voice customization through voice cloning. However, current methods for achieving zero-shot TTS heavily rely on large model scales and extensive training datasets to ensure satisfactory performance and generalizability across various speakers. This raises concerns regarding both deployment costs and data security. In this paper, we present a lightweight and stable zero-shot TTS system. We introduce a novel TTS architecture designed to effectively model linguistic content and various speaker attributes from source speech and prompt speech, respectively. Furthermore, we present a two-stage self-distillation framework that constructs parallel data pairs for effectively disentangling linguistic content and speakers from the perspective of training data. Extensive experiments show that our system exhibits excellent performance and superior stability on the zero-shot TTS tasks. Moreover, it shows markedly superior computational efficiency, with RTFs of 0.13 and 0.012 on the CPU and GPU, respectively.
#### Adaptive Data Augmentation with NaturalSpeech3 for Far-field Speaker Verification
 - **Authors:** Li Zhang, Jiyao Liu, Lei Xie
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2501.08691

 - **Pdf link:** https://arxiv.org/pdf/2501.08691

 - **Abstract**
 The scarcity of speaker-annotated far-field speech presents a significant challenge in developing high-performance far-field speaker verification (SV) systems. While data augmentation using large-scale near-field speech has been a common strategy to address this limitation, the mismatch in acoustic environments between near-field and far-field speech significantly hinders the improvement of far-field SV effectiveness. In this paper, we propose an adaptive speech augmentation approach leveraging NaturalSpeech3, a pre-trained foundation text-to-speech (TTS) model, to convert near-field speech into far-field speech by incorporating far-field acoustic ambient noise for data augmentation. Specifically, we utilize FACodec from NaturalSpeech3 to decompose the speech waveform into distinct embedding subspaces-content, prosody, speaker, and residual (acoustic details) embeddings-and reconstruct the speech waveform from these disentangled representations. In our method, the prosody, content, and residual embeddings of far-field speech are combined with speaker embeddings from near-field speech to generate augmented pseudo far-field speech that maintains the speaker identity from the out-domain near-field speech while preserving the acoustic environment of the in-domain far-field speech. This approach not only serves as an effective strategy for augmenting training data for far-field speaker verification but also extends to cross-data augmentation for enrollment and test speech in evaluation this http URL results on FFSVC demonstrate that the adaptive data augmentation method significantly outperforms traditional approaches, such as random noise addition and reverberation, as well as other competitive data augmentation strategies.
#### Subject Disentanglement Neural Network for Speech Envelope Reconstruction from EEG
 - **Authors:** Li Zhang, Jiyao Liu
 - **Subjects:** Subjects:
Signal Processing (eess.SP); Human-Computer Interaction (cs.HC); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2501.08693

 - **Pdf link:** https://arxiv.org/pdf/2501.08693

 - **Abstract**
 Reconstructing speech envelopes from EEG signals is essential for exploring neural mechanisms underlying speech perception. Yet, EEG variability across subjects and physiological artifacts complicate accurate reconstruction. To address this problem, we introduce Subject Disentangling Neural Network (SDN-Net), which disentangles subject identity information from reconstructed speech envelopes to enhance cross-subject reconstruction accuracy. SDN-Net integrates three key components: MLA-Codec, MPN-MI, and CTA-MTDNN. The MLA-Codec, a fully convolutional neural network, decodes EEG signals into speech envelopes. The CTA-MTDNN module, a multi-scale time-delay neural network with channel and temporal attention, extracts subject identity features from EEG signals. Lastly, the MPN-MI module, a mutual information estimator with a multi-layer perceptron, supervises the removal of subject identity information from the reconstructed speech envelope. Experiments on the Auditory EEG Decoding Dataset demonstrate that SDN-Net achieves superior performance in inner- and cross-subject speech envelope reconstruction compared to recent state-of-the-art methods.
#### Discrimination loss vs. SRT: A model-based approach towards harmonizing speech test interpretations
 - **Authors:** Mareike Buhl, Eugen Kludt, Lena Schell-Majoor, Paul Avan, Marta Campi
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS); Medical Physics (physics.med-ph)
 - **Arxiv link:** https://arxiv.org/abs/2501.08921

 - **Pdf link:** https://arxiv.org/pdf/2501.08921

 - **Abstract**
 Objective: Speech tests aim to estimate discrimination loss or speech recognition threshold (SRT). This paper investigates the potential to estimate SRTs from clinical data that target at characterizing the discrimination loss. Knowledge about the relationship between the speech test outcome variables--conceptually linked via the psychometric function--is important towards integration of data from different databases. Design: Depending on the available data, different SRT estimation procedures were compared and evaluated. A novel, model-based SRT estimation procedure was proposed that deals with incomplete patient data. Interpretations of supra-threshold deficits were assessed for the two interpretation modes. Study sample: Data for 27009 patients with Freiburg monosyllabic speech test (FMST) and audiogram (AG) results from the same day were included in the retrospective analysis. Results: The model-based SRT estimation procedure provided accurate SRTs, but with large deviations in the estimated slope. Supra-threshold hearing loss components differed between the two interpretation modes. Conclusions: The model-based procedure can be used for SRT estimation, and its properties relate to data availability for individual patients. All SRT procedures are influenced by the uncertainty of the word recognition scores. In the future, the proposed approach can be used to assess additional differences between speech tests.


by Zyzzyva0381 (Windy). 


2025-01-16
