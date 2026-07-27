# Showing new listings for Monday, 27 July 2026
Auto update papers at about 2:30am UTC (10:30am Beijing time) every weekday.


阅读 `Usage.md`了解如何使用此repo实现个性化的Arxiv论文推送

See `Usage.md` for instructions on how to personalize the repo. 


Keyword list: ['text-to-speech', 'text to speech', 'tts', 'LLM-based', 'speech', 'voice']


Excluded: []


### Today: 2papers 
#### How Meta-Learning Shapes LoRA Adapter Geometry in Speech Deepfake Detection
 - **Authors:** Ivan Kukanov, Janne Laakkonen, Ville HautamÃ¤ki
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2607.22010

 - **Pdf link:** https://arxiv.org/pdf/2607.22010

 - **Abstract**
 Meta-learning for domain generalization (MLDG) improves out-of-distribution speech deepfake detection over empirical risk minimization (ERM) when both objectives train low-rank adapters on the same frozen self-supervised speech model. Because the architecture and adapter capacity are held fixed, this gap points to differences in how the training objective shapes the adapter, yet the field characterizes objectives through error rates rather than through the geometry of the solution they reach. We introduce a descriptive diagnostic for this question: holding architecture, rank, data, and seeds fixed and varying only the objective, we use the empirical Fisher on the finished adapter to compare the geometry that ERM and MLDG leave behind. We characterize each adapter with effective-rank diagnostics that separate where the adapter changes from where those changes matter to the loss, resolved by projection and by depth. Applied to ERM and MLDG, the diagnostic shows that the objective does not reshape all adapter projections alike: the loss-relevant update concentrates in the query and key projections while becoming more distributed in the output projection, consistently across six corpora and most strongly in the upper layers. The same contrast appears in the merged update independently of the low-rank factorization, indicating that it reflects the geometry of the effective update rather than the parameterization. These results show that the gap between ERM and MLDG is not only a difference in error rate, but a difference in how loss-relevant capacity is organized inside the adapter, and that loss-aware adapter geometry is a way to see it.
#### MEUSLI: a Multilingual Projector for LLM-based ASR and Beyond
 - **Authors:** Lorenzo Concina, Seraphina Fong, Marco Matassoni, Alessio Brutti
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Artificial Intelligence (cs.AI); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2607.22100

 - **Pdf link:** https://arxiv.org/pdf/2607.22100

 - **Abstract**
 Lightweight projectors are an established way to connect pre-trained speech encoders with large language models (LLMs), mapping acoustic features into token-level embeddings for tasks like ASR and spoken question answering. Existing systems, however, typically only support a few languages and are often limited to English. We introduce MEUSLI, the first open-science multilingual projector family that links a Whisper encoder with open-source multilingual LLMs, enabling fully open-source end-to-end ASR in 28 European languages. MEUSLI extends prior monolingual pipelines, delivering strong results across high- and low-resource languages. Using proper continual leaning techniques, MEUSLI can be easily extended to other languages not seen in training. We further demonstrate that the MEUSLI projector can be leveraged beyond ASR, enabling multilingual speech translation and topic identification with only a few hours of task specific supervision per language. Overall, MEUSLI provides a solid foundation for multilingual speech understanding tasks, supporting scalable and inclu- sive open-source SpeechLLM


by Zyzzyva0381 (Windy). 


2026-07-27
