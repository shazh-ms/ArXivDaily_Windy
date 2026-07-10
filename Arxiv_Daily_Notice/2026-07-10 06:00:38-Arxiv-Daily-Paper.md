# Showing new listings for Friday, 10 July 2026
Auto update papers at about 2:30am UTC (10:30am Beijing time) every weekday.


阅读 `Usage.md`了解如何使用此repo实现个性化的Arxiv论文推送

See `Usage.md` for instructions on how to personalize the repo. 


Keyword list: ['text-to-speech', 'text to speech', 'tts', 'LLM-based', 'speech', 'voice']


Excluded: []


### Today: 4papers 
#### On the Role of Conversational Timing in Synthetic Training Data for ASR
 - **Authors:** Máté Gedeon, Péter Mihajlik
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Artificial Intelligence (cs.AI); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2607.08371

 - **Pdf link:** https://arxiv.org/pdf/2607.08371

 - **Abstract**
 Synthetic multi-speaker conversations are widely used to train conversational automatic speech recognition (ASR) systems, but it remains unclear which timing properties make simulated data most useful. This paper studies conversational timing as a controllable training variable rather than merely as a corpus statistic to be reproduced. We parameterize pause and overlap timing distributions with an exponential-tilting family estimated from multiple conversational corpora, and then explore the resulting four-dimensional parameter space with Latin hypercube sampling and multi-objective Bayesian optimization. Each sampled timing configuration is used to generate simulated training conversations, train an ASR system, and evaluate concatenated-permutation word and character error rates (cpWER and cpCER) on a Hungarian dialogue corpus. The results show that downstream ASR behavior is explained more directly by induced timing statistics than by raw simulator coordinates or corpus proximity. In particular, higher overlap exposure is associated with lower cpWER, whereas longer and more variable gaps are associated with higher cpWER; cpCER follows the same trend, but with weaker statistical support. Bayesian optimization yields modest aggregate improvements, but its main value is analytical: it produces controlled timing interventions that reveal an overlap--gap trade-off in simulated conversational training data. These findings suggest that realistic simulation should be complemented by task-relevant diagnostics of overlap, gap, and timing-variability profiles.
#### Why Do You Say It Like That? A Phoneme-Level Framework for Explainable Speech Deepfake Detection
 - **Authors:** Anna Taylor, Michele Panariello, Massimiliano Todisco, Chiara Galdi, Nicholas Evans, Driss Matrouf
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2607.08586

 - **Pdf link:** https://arxiv.org/pdf/2607.08586

 - **Abstract**
 As the accuracy of speech deepfake detection improves with the use of self-supervised representations such as wav2vec 2.0 and HuBERT, understanding why the speech is classified as bona fide or deepfake remains an open challenge. In pursuit of more trustworthy and interpretable artificial intelligence, we introduce a phoneme-level analysis framework that connects model predictions to measurable phonetic units. Our post-hoc explainability method is generally applicable to a variety of speech deepfake detection systems based on convolutional neural networks since it leverages Gradient-weighted Class Activation Mapping in conjunction with speech recognition to generate saliency maps aligned with phonemes and pauses. This pipeline reveals statistically significant attack- and speaker-dependent phonetic cues associated with spoofed speech in terms that humans can understand. Experiments using ASVspoof 5 show comparable detection performance to similar architectures while providing linguistic interpretations across speakers and spoofing conditions.
#### Multimodal Digital Biomarker for Asthma: Complementary Roles of Vocal, Clinical and Demographic Factors
 - **Authors:** Vladimir Despotovic, Milena Despotovic, Abir Elbeji, Petr V. Nazarov, Guy Fagherazzi
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2607.08714

 - **Pdf link:** https://arxiv.org/pdf/2607.08714

 - **Abstract**
 Asthma affects over 260 million people worldwide, yet diagnosis remains dependent on spirometry and specialist assessment, limiting accessibility in primary care and low-resource settings. Vocal biomarkers offer a promising non-invasive alternative, but prior studies have largely focused on acoustic features without integrating clinical context. We present a multimodal Mixture-of-Experts framework for asthma detection that adaptively combines acoustic embeddings from sustained vowel phonation and reading passage tasks with structured clinical and demographic data. The model was evaluated on a matched cohort of 1,218 asthma cases and healthy controls from the Colive Voice study. The multimodal model achieved an AUROC of 0.85 and Brier score of 0.17, outperforming unimodal and bimodal approaches. Adaptive gating analysis revealed increased reliance on audio features in participants with greater respiratory symptom burden, whereas clinical features contributed more strongly in less symptomatic individuals. These findings support scalable and explainable asthma screening using smartphone-collected voice recordings.
#### A Reliability Assessment of LALM Audio Judges for Full-Duplex Voice Agents
 - **Authors:** A. Sayyad, J. Emmons, S. Jones, T. Lin, H. Krishnan
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Artificial Intelligence (cs.AI); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2607.07985

 - **Pdf link:** https://arxiv.org/pdf/2607.07985

 - **Abstract**
 We report the empirical reliability of Gemini models as audio judges that score full-duplex agent conversations directly from the raw stereo waveform, tested across three models in the Gemini family: 2.5 Flash, 3.5 Flash, and 3.1 Pro. Our primary evidence base uses Gemini 2.5 Flash as the ground-truth model, validated against three calibrated human raters on 209 stereo sessions, scored on 8 production dimensions: 152 full-duplex conversations across 13 accent-and-condition strata, together with 57 adversarial defect-injected clips. The evidence for Gemini 2.5 Flash is consistent across three tests. (i) On 5 of 8 dimensions the LALM-human Spearman rho departs from the pairwise human-human rho by at most 0.07, and on 7 of 8 dimensions the two quantities 95 percent bootstrap confidence intervals overlap. (ii) The LALM agrees with the three-rater human mean within 1 point on 60 to 92 percent of sessions on 6 of 8 dimensions. (iii) On 45 of 48 (defect, dimension) cells the LALM is as sensitive as humans or better under Newcombe-Wilson 95 percent confidence intervals, though most of these are underpowered nulls rather than demonstrated parity. Rank-ordering ability transfers across the Gemini family: 3.5 Flash improves simple agreement to 8 of 8 dimensions, while 3.1 Pro rates several dimensions markedly lower than humans despite comparable rank correlation. A model swap should be re-validated on calibration specifically, not assumed from rank-correlation alone. We identify four areas where deployment requires care, and we estimate that human rating alone for our current evaluation cadence costs roughly two orders of magnitude more than the equivalent LALM workload. The data presented here provides a defensible empirical basis for deploying the LALM as a substitute or fourth rater on the dimensions where the evidence supports it.


by Zyzzyva0381 (Windy). 


2026-07-10
