# Showing new listings for Monday, 5 October 2026
Auto update papers at about 2:30am UTC (10:30am Beijing time) every weekday.


阅读 `Usage.md`了解如何使用此repo实现个性化的Arxiv论文推送

See `Usage.md` for instructions on how to personalize the repo. 


Keyword list: ['text-to-speech', 'text to speech', 'tts', 'LLM-based', 'speech', 'voice']


Excluded: []


### Today: 5papers 
#### Acoustic gap placement in second-language read speech production
 - **Authors:** Peyman Jahanbin
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Signal Processing (eess.SP)
 - **Arxiv link:** https://arxiv.org/abs/2610.02582

 - **Pdf link:** https://arxiv.org/pdf/2610.02582

 - **Abstract**
 Speakers can differ not only in how much silence they produce but also in where interruptions fall, information that global pause counts obscure. We analyzed 115 publicly available Speech Accent Archive recordings of the same read passage: 57 speakers whose archive metadata listed Mandarin as their first language and a mainland-China birthplace, and 58 American English first-language speakers born in the U.S. Midwest. Word-level forced alignment was used to extract interword acoustic gaps and classify each position as punctuation-marked or unpunctuated. At a 250 ms threshold, punctuation-marked gaps were similar for the Mandarin and English groups (means 5.56 and 5.47), whereas unpunctuated-position gaps were more frequent in the Mandarin group (means 3.67 and 0.43; adjusted rate ratio 11.98, 95% confidence interval 6.53-22.00). A crossed-effects logistic model with random intercepts for speakers and passage positions confirmed that the contrast was disproportionately concentrated at unpunctuated positions (interaction odds ratio 9.44, 95% confidence interval 5.23-17.02). The pattern persisted at a 500 ms threshold, after removing positions adjacent to punctuation, after excluding severe transcript-deviation cases, and when analysis was restricted to acoustically confirmed silence. Nearly half of the Mandarin group's unpunctuated gaps occurred at positions classified as within-phrase in an exploratory single-coder annotation. In this fixed-passage corpus, acoustic gap placement relative to textual and syntactic structure distinguished the groups more clearly than punctuation-marked pausing. The results support placement-sensitive measurement as a reproducible dimension of breakdown fluency while not establishing a causal mechanism, proficiency difference, or perceptual consequence.
#### Unsupervised Instantaneous Phase and Frequency Tracking by Inverse Voice Synthesis
 - **Authors:** Chin-Yun Yu, György Fazekas
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2610.03058

 - **Pdf link:** https://arxiv.org/pdf/2610.03058

 - **Abstract**
 Knowledge-driven neural vocoders struggle to learn reliable fundamental frequency end-to-end, because spectral objectives provide weak supervision of periodic structure and lack phase information. We address this with a source-filter model whose alias-free additive source makes the instantaneous phase of the glottal cycle explicit; differentiating it yields the instantaneous frequency, and thus $F_0$, without an external tracker. Waveform error supervises only the deterministic harmonic path, while a spectral loss covers the full signal. On M4Singer and LM-SSD, the reconstruction is phase-aligned, reaching a signal-to-reconstruction-error ratio of 8.1 dB, while neural baselines remain negative. However, GOLF, given an external $F_0$, still reaches lower spectral distortion. On LM-SSD, the recovered $F_0$ attains the highest overall accuracy of any method tested, including supervised neural pitch trackers applied off the shelf, and the glottal closure instants come within 0.53 points of REAPER's identification rate, without any $F_0$ label.
#### Augmenting Large Audio Language Models with Low-Level Acoustic Features for Dysarthric Speech Detection
 - **Authors:** Mahdi Amiri, Hatef Otroshi Shahreza, Pascal Frossard, Ina Kodrasi
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2610.03352

 - **Pdf link:** https://arxiv.org/pdf/2610.03352

 - **Abstract**
 Automatic dysarthric speech detection approaches can support traditional clinical diagnosis, which relies on costly and time-consuming evaluation by a speech and language pathologist. Existing automatic approaches predominantly rely on deep learning (DL). More recently, Large Audio Language Models (LALMs) have emerged as a promising alternative given their strong performance across various tasks, but their application to dysarthric speech detection has not yet been established. We propose a framework that fine-tunes LALMs for dysarthric speech detection on speech recordings combined with textual information comprising low-level acoustic features and speaker demographics. Across two LALMs, our framework outperforms DL-based baselines, with Qwen2-Audio-Instruct achieving state-of-the-art performance. An ablation study shows that incorporating acoustic features and speaker demographics during fine-tuning improves LALM performance, while LALMs alone exhibit only chance-level zero-shot performance. These findings establish an effective approach for adapting LALMs to dysarthric speech detection.
#### Multiclass Speech Classification Under Noise Disparity
 - **Authors:** Mahdi Amiri, Sayantan Biswas, Mingchi Hou, Pascal Frossard, Ina Kodrasi
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2610.03381

 - **Pdf link:** https://arxiv.org/pdf/2610.03381

 - **Abstract**
 We investigate noise disparity in multi-class speech classification tasks and develop a strategy to prevent classifiers from exploiting class-dependent noise characteristics. Building on our previous work for binary classification, we introduce a multi-class cross-augmentation scheme that exposes each class to the noise characteristics of the other classes, thereby removing the association between individual noise conditions and class labels. We compare this training-based approach with speech enhancement (SE) as a preprocessing strategy, which aims to suppress noise-related cues directly from the input. Experiments on multi-class emotion recognition show that cross-augmentation effectively mitigates the effect of noise disparity across a range of signal-to-noise-ratios, while SE has a detrimental effect on model performance.
#### Learning Jazz Pianist Style with Cross-Attention Conditioning
 - **Authors:** Drew Edwards, Akira Maezawa, Simon Dixon
 - **Subjects:** Subjects:
Sound (cs.SD); Machine Learning (cs.LG); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2610.02918

 - **Pdf link:** https://arxiv.org/pdf/2610.02918

 - **Abstract**
 Jazz pianists develop distinctive traits that experienced listeners can often identify within seconds, yet the features underlying this recognition resist formal description. We study jazz pianist style through the lens of a pretrained symbolic music transformer, showing that its learned representations already encode pianist identity well enough for highly accurate classification across two benchmarks. We then augment the transformer with cross-attention over learned pianist identity embeddings, enabling it to generate music conditioned on a specific artist's style. Two evaluation protocols confirm that the generator captures meaningful stylistic structure: a sliding-window classifier consistently attributes conditioned continuations to the correct artist, far above unconditioned baselines; and a classifier trained entirely on synthetic generations identifies real pianists across 12 classes with 87% chunk-level and 95% song-level accuracy. Finally, we repurpose the classifier to locate the most characteristic moments within a performance, surfacing the specific musical gestures that distinguish each pianist's voice.


by Zyzzyva0381 (Windy). 


2026-10-05
