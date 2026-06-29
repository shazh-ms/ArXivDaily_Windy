# Showing new listings for Monday, 29 June 2026
Auto update papers at about 2:30am UTC (10:30am Beijing time) every weekday.


阅读 `Usage.md`了解如何使用此repo实现个性化的Arxiv论文推送

See `Usage.md` for instructions on how to personalize the repo. 


Keyword list: ['text-to-speech', 'text to speech', 'tts', 'LLM-based', 'speech', 'voice']


Excluded: []


### Today: 5papers 
#### Screening Matters: A Comparative Study of Conventional and Crowdsourced Listening Tests
 - **Authors:** Anika Treffehn, Andrea Eichenseer, Emily Kratsch, Nicola Pia
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2606.28114

 - **Pdf link:** https://arxiv.org/pdf/2606.28114

 - **Abstract**
 Subjective evaluation remains the most reliable way of testing speech and audio coding techniques. Crowdsourcing the listening task is a cost-efficient and fast way of conducting this evaluation, but the quality of the results tends to be inferior to that of conventional listening tests done in the controlled environment of a laboratory. In this paper, classical and neural speech codecs are evaluated to compare P.808 against P.800 DCR tests. A statistical analysis is conducted to investigate the effectiveness of selected screening methods. The analysis shows that the crowdsourced evaluation can be improved by employing postscreening methods based on anchor ordering and rating span, and continuous screening methods like traps and gold standard questions, thus giving more value to the ratings obtained for the codecs under test. Based on these outcomes, a set of suitable screenings is proposed, for cost-effective, simplified, and bias-free enhancement of listening results.
#### HPRO: Hierarchical Progressive Reward Optimization via Preference Extraction for Emotional Text-to-Speech
 - **Authors:** Sihang Nie, Xiaofen Xing, Rui Xing, Haoming Li, Ruitong Xiao, Jingyuan Xing, Baiji Liu, Xiangmin Xu
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Computation and Language (cs.CL); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2606.28249

 - **Pdf link:** https://arxiv.org/pdf/2606.28249

 - **Abstract**
 Recently, Large Language Model (LLM)-based Text-to-Speech (TTS) models have achieved remarkable naturalness. However, the standard Supervised Fine-Tuning paradigm often converges to statistically averaged prosody, limiting emotional expressiveness. While preference-driven optimization offers a promising alternative, existing approaches suffer from two structural mismatches: information conflict, where content and emotion in a shared latent space produce conflicting gradients, leading to reward hacking and semantic degradation; and scale gap, where sparse sentence-level rewards struggle to guide dense frame-level generation. To overcome these challenges, we propose HPRO, a hierarchical progressive reward optimization framework. Within HPRO, we introduce the HD-Emo codec as a novel differentiable reward model to resolve the information conflict. It extracts speech into distinct content and style preference tokens, structurally isolating emotional optimization from semantic content. Building upon this structured preference space, HPRO bridges the scale gap by progressively aligning frame-, word- and sentence-level objectives. Experiments demonstrate that HPRO significantly enhances emotional expressiveness, while effectively preserving linguistic intelligibility. The code and audio samples are publicly available at this https URL.
#### Do Speech Emphasis Models Generalize across Languages and Emotions?
 - **Authors:** Megan Wei, Deepali Aneja, Jiaqi Su, Yunyun Wang, Haonan Chen, Zeyu Jin
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Artificial Intelligence (cs.AI); Machine Learning (cs.LG); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2606.27717

 - **Pdf link:** https://arxiv.org/pdf/2606.27717

 - **Abstract**
 Prosodic emphasis varies across languages, emotions, and speaking styles, yet existing emphasis detection models are largely trained and evaluated on monolingual neutral read speech. We introduce MMEE (Multilingual Multi-Emotion Emphasis), a corpus of 10,000 professionally recorded expressive utterances (14.13 hours) across 7 languages and 34 emotion/style categories, with three-level perceptual labels (10 annotations per sample). We benchmark two state-of-the-art architectures under monolingual, cross-lingual, multilingual, cross-emotion, cross-dataset, and data-scale settings. Monolingual models show limited zero-shot transfer, degrading across typologically distant languages, while multilingual training substantially improves robustness. Models transfer robustly between high- and low-arousal emotions; bidirectional transfer between synthetic and perceptual benchmarks suggests shared prosodic structure; and performance stays robust even at smaller training scales.
#### Dialogue to Detection: A Multimodal Hybrid NLP Pipeline for Insurance Fraud Detection
 - **Authors:** Muhammad Shakeel Akram, Amal Htait, Abdul Hamid Sadka, Emma Meisingseth, Karishma Jaitly
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Artificial Intelligence (cs.AI); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2606.28002

 - **Pdf link:** https://arxiv.org/pdf/2606.28002

 - **Abstract**
 Insurance fraud imposes substantial financial losses and operational inefficiencies, raising premiums and impacting trust among legitimate policyholders. Early detection at FNOL remains a persistent challenge. Existing approaches rely largely on private, text-only datasets, limiting progress on multimodal methods that integrate linguistic, behavioural, and speaker-based indicators. We introduce a synthetic multimodal framework that replicates FNOL conditions. It generates agent-customer dialogue transcripts and two-speaker audios, performs ASR and diarisation. Downstream modules combine NER, regex-based feature extraction, LLM-RAG retrieval, and speaker embeddings in a rule-based risk score to flag narrative reuse, structural inconsistencies, and cross-case voice repetition while balancing sensitivity and false positives. Dataset validation and component-level evaluations show stability and transfer potential, offering a reproducible baseline beyond text-only fraud detection.
#### DG^VoiC: Speaker Clustering for Fraud Investigation under Real Call-Centre Conditions
 - **Authors:** Muhammad Shakeel Akram, Amal Htait, Abdul Hamid Sadka, Emma Meisingseth, Karishma Jaitly
 - **Subjects:** Subjects:
Sound (cs.SD); Artificial Intelligence (cs.AI); Computation and Language (cs.CL); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2606.28048

 - **Pdf link:** https://arxiv.org/pdf/2606.28048

 - **Abstract**
 Insurance fraud remains costly and operationally difficult, particularly in call-centre workflows where many customer interactions begin at FNOL. While recent fraud detection methods mainly rely on structured data, text, or images, repeated speaker identity across calls remains underused as an investigative signal. This paper presents DG^VoiC, a voice clustering framework for customer verification and cross-profile speaker linking on anonymised real call-centre audio. The approach combines sensitive information-aligned anonymisation, speech-focused preprocessing, sliding-window speaker embedding extraction, and cosine similarity based clustering to identify repeated speakers under real telephony conditions. The method was evaluated on 121 recordings, with a curated reference subset of 56 samples in 22 human-agreed speaker clusters. used for validation. The best configuration achieved 96% AMI, 95% ARI, 98% completeness, 100% homogeneity, and 99% V-measure. These results show that speaker clustering can provide a strong additional signal for fraud investigation by helping analysts verify speaker consistency and surface repeated voices across customers.


by Zyzzyva0381 (Windy). 


2026-06-29
