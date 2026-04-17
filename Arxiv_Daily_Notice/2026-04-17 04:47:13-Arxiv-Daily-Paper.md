# Showing new listings for Friday, 17 April 2026
Auto update papers at about 2:30am UTC (10:30am Beijing time) every weekday.


阅读 `Usage.md`了解如何使用此repo实现个性化的Arxiv论文推送

See `Usage.md` for instructions on how to personalize the repo. 


Keyword list: ['text-to-speech', 'text to speech', 'tts', 'LLM-based', 'speech', 'voice']


Excluded: []


### Today: 6papers 
#### HARNESS: Lightweight Distilled Arabic Speech Foundation Models
 - **Authors:** Vrunda N. Sukhadia, Shammur Absar Chowdhury
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Artificial Intelligence (cs.AI); Computation and Language (cs.CL)
 - **Arxiv link:** https://arxiv.org/abs/2604.14186

 - **Pdf link:** https://arxiv.org/pdf/2604.14186

 - **Abstract**
 Large self-supervised speech (SSL) models achieve strong downstream performance, but their size limits deployment in resource-constrained settings. We present HArnESS, an Arabic-centric self-supervised speech model family trained from scratch with iterative self-distillation, together with lightweight student variants that offer strong accuracy-efficiency trade-offs on Automatic Speech Recognition (ASR), Dialect Identification (DID), and Speech Emotion Recognition (SER). Our approach begins with a large bilingual Arabic-English teacher and progressively distills its knowledge into compressed student models while preserving Arabic-relevant acoustic and paralinguistic representations. We further study PCA-based compression of the teacher supervision signal to better match the capacity of shallow and thin students. Compared with HuBERT and XLS-R, HArnESS consistently improves performance on Arabic downstream tasks, while the compressed models remain competitive under substantial structural reduction. These results position HArnESS as a practical and accessible Arabic-centric SSL foundation for real-world speech applications.
#### Who is Speaking or Who is Depressed? A Controlled Study of Speaker Leakage in Speech-Based Depression Detection
 - **Authors:** Hsiang-Chen Yeh, Luqi Sun, Aurosweta Mahapatra, Shreeram Suresh Chandra, Emily Mower Provost, Berrak Sisman
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2604.14354

 - **Pdf link:** https://arxiv.org/pdf/2604.14354

 - **Abstract**
 This study investigates whether speech-based depression detection models learn depression-related acoustic biomarkers or instead rely on speaker identity cues. Using the DAIC-WOZ dataset, we propose a data-splitting strategy that controls speaker overlap between training and test sets while keeping the training size constant, and evaluate three models of varying complexity. Results show that speaker overlap significantly boosts performance, whereas accuracy drops sharply on unseen speakers. Even with a Domain-Adversarial Neural Network, a substantial performance gap remains. These findings indicate that depression-related features extracted by current speech models are highly entangled with speaker identity. Conventional evaluation protocols may therefore overestimate generalization and clinical utility, highlighting the need for strictly speaker-independent evaluation.
#### UniPASE: A Generative Model for Universal Speech Enhancement with High Fidelity and Low Hallucinations
 - **Authors:** Xiaobin Rong, Zheng Wang, Yushi Wang, Jun Gao, Jing Lu
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2604.14606

 - **Pdf link:** https://arxiv.org/pdf/2604.14606

 - **Abstract**
 Universal speech enhancement (USE) aims to restore speech signals from diverse distortions across multiple sampling rates. We propose UniPASE, an extension of the low-hallucination PASE framework tailored for USE. At its core is DeWavLM-Omni, a unified representation-level enhancement module fine-tuned from WavLM via knowledge distillation on a large-scale supervised multi-distortion dataset. This module directly converts degraded waveforms into clean and linguistically faithful phonetic representations, ensuring robust enhancement with minimal linguistic hallucination. Based on these enhanced phonetic representations, an Adapter generates enhanced acoustic representations containing rich acoustic details, which a neural Vocoder uses to reconstruct corresponding high-fidelity 16-kHz waveforms. A PostNet then converts the waveforms to 48~kHz before resampling them to their original rates, enabling seamless handling of inputs and outputs at multiple sampling rates. Experimental results on several evaluation datasets, covering sub-tasks and full tasks, demonstrate that UniPASE achieves superior or competitive performance compared with existing state-of-the-art models. The proposed model also serves as the backbone of our submission to the URGENT 2026 Challenge, which achieved 1st place in the objective evaluation. The source code and audio demos are available at this https URL.
#### From Black Box to Glass Box: Cross-Model ASR Disagreement to Prioto Review in Ambient AI Scribe Documentation
 - **Authors:** Abdolamir Karbalaie, Fernando Seoane, Farhad Abtahi
 - **Subjects:** Subjects:
Sound (cs.SD); Artificial Intelligence (cs.AI); Computation and Language (cs.CL); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2604.14152

 - **Pdf link:** https://arxiv.org/pdf/2604.14152

 - **Abstract**
 Ambient AI "scribe" systems promise to reduce clinical documentation burden, but automatic speech recognition (ASR) errors can remain unnoticed without careful review, and high-quality human reference transcripts are often unavailable for calibrating uncertainty. We investigate whether cross-model disagreement among heterogeneous ASR systems can act as a reference-free uncertainty signal to prioritize human verification in medical transcription workflows. Using 50 publicly available medical education audio clips (8 h 14 min), we transcribed each clip with eight ASR systems spanning commercial APIs and open-source engines. We aligned multi-model outputs, built consensus pseudo-references, and quantified token-level agreement using a majority-strength metric; we further characterized disagreements by type (content vs. punctuation/formatting) and assessed per-model agreement via leave-one-model-out (jackknife) consensus scoring. Inter-model reliability was low (ICC[2,1] = 0.131), indicating heterogeneous failure modes across systems. Across 76,398 evaluated token positions, 72.1% showed near-unanimous agreement (7-8 models), while 2.5% fell into high-risk bands (0-3 models), with high-risk mass varying from 0.7% to 11.4% across accent groups. Low-agreement regions were enriched for content disagreements, with the content fraction increasing from 53.9% to 73.9% across quintiles of high-risk mass. These results suggest that cross-model disagreement provides a sparse, localizable signal that can surface potentially unreliable transcript spans without human-verified references, enabling targeted review; clinical accuracy of flagged regions remains to be established.
#### VoxSafeBench: Not Just What Is Said, but Who, How, and Where
 - **Authors:** Yuxiang Wang, Hongyu Liu, Yijiang Xu, Qinke Ni, Li Wang, Wan Lin, Kunyu Feng, Dekun Chen, Xu Tan, Lei Wang, Jie Shi, Zhizheng Wu
 - **Subjects:** Subjects:
Sound (cs.SD); Machine Learning (cs.LG); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2604.14548

 - **Pdf link:** https://arxiv.org/pdf/2604.14548

 - **Abstract**
 As speech language models (SLMs) transition from personal devices into shared, multi-user environments, their responses must account for far more than the words alone. Who is speaking, how they sound, and where the conversation takes place can each turn an otherwise benign request into one that is unsafe, unfair, or privacy-violating. Existing benchmarks, however, largely focus on basic audio comprehension, study individual risks in isolation, or conflate content that is inherently harmful with content that only becomes problematic due to its acoustic context. We introduce VoxSafeBench, among the first benchmarks to jointly evaluate social alignment in SLMs across three dimensions: safety, fairness, and privacy. VoxSafeBench adopts a Two-Tier design: Tier1 evaluates content-centric risks using matched text and audio inputs, while Tier2 targets audio-conditioned risks in which the transcript is benign but the appropriate response hinges on the speaker, paralinguistic cues, or the surrounding environment. To validate Tier2, we include intermediate perception probes and confirm that frontier SLMs can successfully detect these acoustic cues yet still fail to act on them appropriately. Across 22 tasks with bilingual coverage, we find that safeguards appearing robust on text often degrade in speech: safety awareness drops for speaker- and scene-conditioned risks, fairness erodes when demographic differences are conveyed vocally, and privacy protections falter when contextual cues arrive acoustically. Together, these results expose a pervasive speech grounding gap: current SLMs frequently recognize the relevant social norm in text but fail to apply it when the decisive cue must be grounded in speech. Code and data are publicly available at: this https URL
#### The Acoustic Camouflage Phenomenon: Re-evaluating Speech Features for Financial Risk Prediction
 - **Authors:** Dhruvin Dungrani, Disha Dungrani
 - **Subjects:** Subjects:
Sound (cs.SD); Machine Learning (cs.LG); Audio and Speech Processing (eess.AS); Computational Finance (q-fin.CP); Statistical Finance (q-fin.ST)
 - **Arxiv link:** https://arxiv.org/abs/2604.14619

 - **Pdf link:** https://arxiv.org/pdf/2604.14619

 - **Abstract**
 In computational paralinguistics, detecting cognitive load and deception from speech signals is a heavily researched domain. Recent efforts have attempted to apply these acoustic frameworks to corporate earnings calls to predict catastrophic stock market volatility. In this study, we empirically investigate the limits of acoustic feature extraction (pitch, jitter, and hesitation) when applied to highly trained speakers in in-the-wild teleconference environments. Utilizing a two-stream late-fusion architecture, we contrast an acoustic-based stream with a baseline Natural Language Processing (NLP) stream. The isolated NLP model achieved a recall of 66.25% for tail-risk downside events. Surprisingly, integrating acoustic features via late fusion significantly degraded performance, reducing recall to 47.08%. We identify this degradation as Acoustic Camouflage, where media-trained vocal regulation introduces contradictory noise that disrupts multimodal meta-learners. We present these findings as a boundary condition for speech processing applications in high-stakes financial forecasting.


by Zyzzyva0381 (Windy). 


2026-04-17
