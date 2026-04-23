# Showing new listings for Thursday, 23 April 2026
Auto update papers at about 2:30am UTC (10:30am Beijing time) every weekday.


阅读 `Usage.md`了解如何使用此repo实现个性化的Arxiv论文推送

See `Usage.md` for instructions on how to personalize the repo. 


Keyword list: ['text-to-speech', 'text to speech', 'tts', 'LLM-based', 'speech', 'voice']


Excluded: []


### Today: 7papers 
#### Explainable Speech Emotion Recognition: Weighted Attribute Fairness to Model Demographic Contributions to Social Bias
 - **Authors:** Tomisin Ogunnubi, Yupei Li, Björn Schuller
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Artificial Intelligence (cs.AI); Computation and Language (cs.CL)
 - **Arxiv link:** https://arxiv.org/abs/2604.19763

 - **Pdf link:** https://arxiv.org/pdf/2604.19763

 - **Abstract**
 Speech Emotion Recognition (SER) systems have growing applications in sensitive domains such as mental health and education, where biased predictions can cause harm. Traditional fairness metrics, such as Equalised Odds and Demographic Parity, often overlook the joint dependency between demographic attributes and model predictions. We propose a fairness modelling approach for SER that explicitly captures allocative bias by learning the joint relationship between demographic attributes and model error. We validate our fairness metric on synthetic data, then apply it to evaluate HuBERT and WavLM models finetuned on the CREMA-D dataset. Our results indicate that the proposed fairness model captures more mutual information between protected attributes and biases and quantifies the absolute contribution of individual attributes to bias in SSL-based SER models. Additionally, our analysis reveals indications of gender bias in both HuBERT and WavLM.
#### Enhancing ASR Performance in the Medical Domain for Dravidian Languages
 - **Authors:** Sri Charan Devarakonda, Ravi Sastry Kolluru, Manjula Sri Rayudu, Rashmi Kapoor, Madhu G, Anil Kumar Vuppala
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Artificial Intelligence (cs.AI); Computation and Language (cs.CL)
 - **Arxiv link:** https://arxiv.org/abs/2604.19797

 - **Pdf link:** https://arxiv.org/pdf/2604.19797

 - **Abstract**
 Automatic Speech Recognition (ASR) for low-resource Dravidian languages like Telugu and Kannada faces significant challenges in specialized medical domains due to limited annotated data and morphological complexity. This work proposes a novel confidence-aware training framework that integrates real and synthetic speech data through a hybrid confidence mechanism combining static perceptual and acoustic similarity metrics with dynamic model entropy. Unlike direct fine-tuning approaches, the proposed methodology employs both fixed-weight and learnable-weight confidence aggregation strategies to guide sample weighting during training, enabling effective utilization of heterogeneous data sources. The framework is evaluated on Telugu and Kannada medical datasets containing both real recordings and TTS-generated synthetic speech. A 5-gram KenLM language model is applied for post-decoding correction. Results show that the hybrid confidence-aware approach with learnable weights substantially reduces recognition errors: Telugu Word Error Rate (WER) decreases from 24.3% to 15.8% (8.5% absolute improvement), while Kannada WER drops from 31.7% to 25.4% (6.3% absolute improvement), both significantly outperforming standard fine-tuning baselines. These findings confirm that combining adaptive confidence-aware training with statistical language modeling delivers superior performance for domain-specific ASR in morphologically complex Dravidian languages.
#### Utterance-Level Methods for Identifying Reliable ASR-Output for Child Speech
 - **Authors:** Gus Lathouwers, Lingyun Gao, Catia Cucchiarini, Helmer Strik
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Artificial Intelligence (cs.AI); Computation and Language (cs.CL)
 - **Arxiv link:** https://arxiv.org/abs/2604.19801

 - **Pdf link:** https://arxiv.org/pdf/2604.19801

 - **Abstract**
 Automatic Speech Recognition (ASR) is increasingly used in applications involving child speech, such as language learning and literacy acquisition. However, the effectiveness of such applications is limited by high ASR error rates. The negative effects can be mitigated by identifying in advance which ASR-outputs are reliable. This work aims to develop two novel approaches for selecting reliable ASR-output at the utterance level, one for selecting reliable read speech and one for dialogue speech material. Evaluations were done on an English and a Dutch dataset, each with a baseline and finetuned model. The results show that utterance-level selection methods for identifying reliably transcribed speech recordings have high precision for the best strategy (P > 97.4) for both read speech and dialogue material, for both languages. Using the current optimal strategy allows 21.0% to 55.9% of dialogue/read speech datasets to be automatically selected with low (UER of < 2.6) error rates.
#### Indic-CodecFake meets SATYAM: Towards Detecting Neural Audio Codec Synthesized Speech Deepfakes in Indic Languages
 - **Authors:** Girish, Mohd Mujtaba Akhtar, Orchid Chetia Phukan, Arun Balaji Buduru
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2604.19949

 - **Pdf link:** https://arxiv.org/pdf/2604.19949

 - **Abstract**
 The rapid advancement of Audio Large Language Models (ALMs), driven by Neural Audio Codecs (NACs), has led to the emergence of highly realistic speech deepfakes, commonly referred to as CodecFakes (CFs). Consequently, CF detection has attracted increasing attention from the research community. However, existing studies predominantly focus on English or Chinese, leaving the vulnerability of Indic languages largely unexplored. To bridge this gap, we introduce Indic-CodecFake (ICF) dataset, the first large-scale benchmark comprising real and NAC-synthesized speech across multiple Indic languages, diverse speaker profiles, and multiple NAC types. We use IndicSUPERB as the real speech corpus for generation of ICF dataset. Our experiments demonstrate that state-of-the-art (SOTA) CF detectors trained on English-centric datasets fail to generalize to ICF, underscoring the challenges posed by phonetic diversity and prosodic variability in Indic speech. Further, we present systematic evaluation of SOTA ALMs in a zero-shot setting on ICF dataset. We evaluate these ALMs as they have shown effectiveness for different speech tasks. However, our findings reveal that current ALMs exhibit consistently poor performance. To address this, we propose SATYAM, a novel hyperbolic ALM tailored for CF detection in Indic languages. SATYAM integrates semantic representations from Whisper and prosodic representations from TRILLsson using through Bhattacharya distance in hyperbolic space and subsequently performs the same alignment procedure between the fused speech representation and an input conditioning prompt. This dual-stage fusion framework enables SATYAM to effectively model hierarchical relationships both within speech (semantic-prosodic) and across modalities (speech-text). Extensive evaluations show that SATYAM consistently outperforms competitive end-to-end and ALM-based baselines on the ICF benchmark.
#### Embedding-Based Intrusive Evaluation Metrics for Musical Source Separation Using MERT Representations
 - **Authors:** Paul A. Bereuter, Alois Sontacchi
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2604.20270

 - **Pdf link:** https://arxiv.org/pdf/2604.20270

 - **Abstract**
 Evaluation of musical source separation (MSS) has traditionally relied on Blind Source Separation Evaluation (BSS-Eval) metrics. However, recent work suggests that BSS-Eval metrics exhibit low correlation between metrics and perceptual audio quality ratings from a listening test, which is considered the gold standard evaluation method. As an alternative approach in singing voice separation, embedding-based intrusive metrics that leverage latent representations from large self-supervised audio models such as Music undERstanding with large-scale self-supervised Training (MERT) embeddings have been introduced. In this work, we analyze the correlation of perceptual audio quality ratings with two intrusive embedding-based metrics: a mean squared error (MSE) and an intrusive variant of the Fréchet Audio Distance (FAD) calculated on MERT embeddings. Experiments on two independent datasets show that these metrics correlate more strongly with perceptual audio quality ratings than traditional BSS-Eval metrics across all analyzed stem and model types.
#### KoALa-Bench: Evaluating Large Audio Language Models on Korean Speech Understanding and Faithfulness
 - **Authors:** Jinyoung Kim, Hyeongsoo Lim, Eunseo Seo, Minho Jang, Keunwoo Choi, Seungyoun Shin, Ji Won Yoon
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Artificial Intelligence (cs.AI); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2604.19782

 - **Pdf link:** https://arxiv.org/pdf/2604.19782

 - **Abstract**
 Recent advances in large audio language models (LALMs) have enabled multilingual speech understanding. However, benchmarks for evaluating LALMs remain scarce for non-English languages, with Korean being one such underexplored case. In this paper, we introduce KoALa-Bench, a comprehensive benchmark for evaluating Korean speech understanding and speech faithfulness of LALMs. In particular, KoALa-Bench comprises six tasks. Four tasks evaluate fundamental speech understanding capabilities, including automatic speech recognition, speech translation, speech question answering, and speech instruction following, while the remaining two tasks evaluate speech faithfulness, motivated by our observation that several LALMs often fail to fully leverage the speech modality. Furthermore, to reflect Korea-specific knowledge, our benchmark incorporates listening questions from the Korean college scholastic ability test as well as content covering Korean cultural domains. We conduct extensive experiments across six models, including both white-box and black-box ones. Our benchmark, evaluation code, and leaderboard are publicly available at this https URL.
#### Tonnetz Theory, Classical Harmony, and the Combinatorial Geometry of Abstract Musical Resources
 - **Authors:** Jeffrey R. Boland, Lane P. Hughston
 - **Subjects:** Subjects:
Combinatorics (math.CO); Audio and Speech Processing (eess.AS); Algebraic Geometry (math.AG)
 - **Arxiv link:** https://arxiv.org/abs/2604.19960

 - **Pdf link:** https://arxiv.org/pdf/2604.19960

 - **Abstract**
 In a previous submission, we established a fundamental relation between tone networks and configurations. It was shown that the Eulerian tonnetz can be represented by a $\{12_3\}$ of Daublebsky von Sterneck type D222. We also constructed a tonnetz for Tristan-genus chords (dominant sevenths and half-diminished sevenths) and we showed that this tonnetz can be represented by a $\{12_3\}$ of type D228. In both of these constructions the associated Levi graphs play an important role. Here we look at the tonnetze associated with some other musical systems, thereby offering several concrete examples of an abstract view of music as combinatorial geometry. First, we look at the tonal harmonies typical of the classical period. In the case of diatonic triads, we show the existence of a bipartite graph of type $\{7_3\}$ and girth four that represents the well-known relations between the seven diatonic degrees and their pitch classes. In the case of diatonic seventh chords, we obtain a Fano configuration $\{7_3\}$ which gives a complete characterization of the voice-leading relations that hold between such chords. Next, we construct a tonnetz for pentatonic music based on the Desargues configuration $\{10_3\}$ and we construct a tonnetz for the 12-tone system based on the Cremona-Richmond configuration $\{15_3\}$. Both can be used as a resource for musical compositions. Finally, we show that the relation between the chromatic pitch class set and the major triad set is also represented by a D222. The minor triads are in one-to-one correspondence with the members of a certain class of hexacycles in the Levi graph of this configuration. In this way, the characteristic duality between major and minor triads in the tonnetz can be broken.


by Zyzzyva0381 (Windy). 


2026-04-23
