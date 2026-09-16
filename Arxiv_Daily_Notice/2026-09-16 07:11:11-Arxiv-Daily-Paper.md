# Showing new listings for Wednesday, 16 September 2026
Auto update papers at about 2:30am UTC (10:30am Beijing time) every weekday.


阅读 `Usage.md`了解如何使用此repo实现个性化的Arxiv论文推送

See `Usage.md` for instructions on how to personalize the repo. 


Keyword list: ['text-to-speech', 'text to speech', 'tts', 'LLM-based', 'speech', 'voice']


Excluded: []


### Today: 6papers 
#### Language Orthogonalization for Zero-Shot Cross-Lingual Audio Deepfake Detection
 - **Authors:** Minu Kim, Ji Sub Um, Hoirin Kim
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Computation and Language (cs.CL)
 - **Arxiv link:** https://arxiv.org/abs/2609.16458

 - **Pdf link:** https://arxiv.org/pdf/2609.16458

 - **Abstract**
 Audio deepfake detectors need to transfer to languages absent from training, as multilingual speech synthesis outpaces labeled anti-spoofing resources. While detectors increasingly rely on self-supervised speech models (S3Ms), these backbones encode language-dependent structure that confounds spoof cues. We address this confound through language orthogonalization, a target-free ridge map that removes S3M variation projected onto continuous language-identification (LID) embeddings. Across six languages, six S3M backbones, and all Leave-N-Out settings, it consistently reduces EER across unseen languages. Cross-lingual EER correlates with LID-space distance, where orthogonalization yields larger gains for more distant transfers.
#### The Evolving Bottleneck in Speech Generation: Interface Co-design and Staged Alignment from CosyVoice to Qwen-Audio-3.0-TTS
 - **Authors:** Qian Chen, Xiangang Li, Xiang Lv, Han Zhao, Tianyu Zhao
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2609.16514

 - **Pdf link:** https://arxiv.org/pdf/2609.16514

 - **Abstract**
 Speech synthesis systems are commonly narrated as a sequence of larger models, better tokenizers, and broader data. This technical retrospective offers a different account of the CosyVoice lineage, from CosyVoice through CosyVoice 2 and CosyVoice 3 to Qwen-Audio-3.0-TTS: progress came from repeatedly relocating the system's dominant bottleneck. Across the lineage, a stable decomposition separates an autoregressive language model that plans speech from a flow-matching model that renders acoustics. What changes is the contract between them. CosyVoice establishes supervised semantic tokens as a content-aligned interface; CosyVoice 2 makes that interface causally available for streaming and removes the utterance-level speaker embedding from the language model; CosyVoice 3 improves the learnability and coverage of the interface through multitask supervision, scaling, and differentiable reward optimization; and Qwen-Audio-3.0-TTS reduces token rate, conditions its renderer on continuous language-model hidden states instead of token embeddings, and progressively aligns the coupled system. We formalize this history through four interface dimensions---representation, ownership, availability, and gradient reach---and separate within-paper evidence from cross-paper comparison. The resulting synthesis connects discrete autoregressive, continuous non-autoregressive, hybrid, and continuous autoregressive speech-generation paradigms, and yields practical principles for diagnosing and training modular speech generators.
#### Differentiable and Severity-invariant Discrete Tokens for Dysarthric Speech Recognition
 - **Authors:** Huimeng Wang, Xurong Xie, Mengzhe Geng, Haoning Xu, Jiajun Deng, Youjun Chen, Chengxi Deng, Xunying Liu
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2609.16855

 - **Pdf link:** https://arxiv.org/pdf/2609.16855

 - **Abstract**
 This paper proposes novel differentiable and severity-invariant (DSI) discrete token approaches that are not only tightly integrated with downstream dysarthric speech recognition tasks, but also minimise discrete token diversity across speech impairment severity groups. Experiments conducted on the UASpeech and TORGO corpora suggest that Conformer models trained using the DSI tokens outperform the comparable baseline HuBERT discrete/continuous features by statistically significant WER reductions of 2.22\%/0.78\% absolute (9.14\%/3.41\% relative) and 1.78\%/1.06\% absolute (18.43\%/11.86\% relative) on the two tasks, respectively. After system combination, the lowest WERs of 18.90\% and 6.38\% were obtained on UASpeech and TORGO. Phoneme-specific T-SNE visualizations show that severity-invariant regularization reduces severity-dependent variation by producing greater overlap and less distinct boundaries among severity-group distributions.
#### Counting Closures in Spanish Trills: A Multi-Corpus Acoustic Study
 - **Authors:** Mateo Cámara, Maria F. Alcala-Durand
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2609.17424

 - **Pdf link:** https://arxiv.org/pdf/2609.17424

 - **Abstract**
 The Spanish trill /r/ is canonically described as a short sequence of lingual closures, yet large-scale acoustic evidence across corpora is scarce, and automatic counters locating envelope peaks tend to conflate each closure with its release. We present a closure-based detector that locates closures gated by a quality filter and cross-checked against an independent autocorrelation-based period estimator. Applied to 3,560 well-formed (voiced, periodic) trill tokens from 356 speakers across six Spanish corpora, the detector yields a median of two closures and an inter-closure period near 36ms, matching the descriptive literature on all corpora. At the speaker level, phonotactic context is the only factor with a robust, medium effect: onset trills (word-initial and post-/n,l,s/) show more closures than intervocalic rr. We find no robust evidence of a sex effect once closures are counted directly. We report reference values and release a reproducible measurement pipeline for Spanish trills.
#### EMODY Flow: Emotion-Aware Audio-Driven Full-Body Motion Generation
 - **Authors:** Harsh Kumar Agarwal, Xavier Alameda-Pineda, Olivier Perrotin
 - **Subjects:** Subjects:
Graphics (cs.GR); Computer Vision and Pattern Recognition (cs.CV); Machine Learning (cs.LG); Multimedia (cs.MM); Robotics (cs.RO); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2609.16011

 - **Pdf link:** https://arxiv.org/pdf/2609.16011

 - **Abstract**
 Embodied conversational agents require synchronized full-body motion (body gestures and facial expressions) that aligns with speech and emotional state. Omni-modal large language models excel at multimodal understanding but produce only linguistic outputs, leaving a critical gap in embodied response generation. We identify and address a failure of emotion conditioning: like other conditional generators that under-use weak conditioning signals, a flow-matching model given both a rich audio embedding and a discrete emotion label suppresses the emotion, generating near-identical motion regardless of the specified emotion. We present EMODY Flow, a lightweight (around 35M parameters) flow-matching framework that attaches to a frozen Qwen-3 Omni model and reuses its internal Mimi audio-codecs to condition two parallel DiT generators - one for SMPL-X body pose, one for FLAME facial expressions. A training-time auxiliary emotion classifier restores emotion sensitivity by forcing generated motion to be emotion-identifiable. EMODY Flow sets a new state of the art on BEAT2 gesture quality, with FGD 0.302, Beat Correlation 0.853, and Diversity 24.62 - improving over the best prior results by 26%, 5%, and 62% respectively - and transfers to zero-shot facial animation on TFHP without domain-specific fine-tuning. Beyond these quantitative gains, the classifier yields clearly emotion-separated motion, which we demonstrate qualitatively through a multidimensional-scaling analysis of the generated gestures.
#### Self-Distilled Pronunciation and Accent Control for Neural Text-to-Speech
 - **Authors:** Shuhei Kato
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2609.17234

 - **Pdf link:** https://arxiv.org/pdf/2609.17234

 - **Abstract**
 Text-to-speech that reads raw text has no lexicon: a rare word is read as guessed. Remedies train a reading-and-accent channel on recorded speech or edit words one at a time from exemplars. We do neither. The frozen backbone reads a sentence containing a common word it already says correctly, and its own output then serves as the teacher for the same sentence, with that word replaced by a tagged, accented reading; this training pair is the whole idea. On Sarashina2.2-TTS, screened raters at Fleiss' kappa = 0.85 hear the prescribed accent on 0.89 of unseen words against 0.57 for kana, which cannot express one; kana wins no pair; naturalness is not measurably hurt. Moved untuned to autoregressive, diffusion, and encoder-decoder backbones, it transfers reading, 0.25 to 0.47 above no edit on 319 words, and on CosyVoice 2 accent on two words in three, but not on Irodori; the paper locates why.


by Zyzzyva0381 (Windy). 


2026-09-16
