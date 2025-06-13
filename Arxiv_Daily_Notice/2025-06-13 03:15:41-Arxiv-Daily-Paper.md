# Showing new listings for Friday, 13 June 2025
Auto update papers at about 2:30am UTC (10:30am Beijing time) every weekday.


阅读 `Usage.md`了解如何使用此repo实现个性化的Arxiv论文推送

See `Usage.md` for instructions on how to personalize the repo. 


Keyword list: ['text-to-speech', 'text to speech', 'tts', 'LLM-based', 'speech', 'voice']


Excluded: []


### Today: 12papers 
#### RT-VC: Real-Time Zero-Shot Voice Conversion with Speech Articulatory Coding
 - **Authors:** Yisi Liu, Chenyang Wang, Hanjo Kim, Raniya Khan, Gopala Anumanchipalli
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Artificial Intelligence (cs.AI)
 - **Arxiv link:** https://arxiv.org/abs/2506.10289

 - **Pdf link:** https://arxiv.org/pdf/2506.10289

 - **Abstract**
 Voice conversion has emerged as a pivotal technology in numerous applications ranging from assistive communication to entertainment. In this paper, we present RT-VC, a zero-shot real-time voice conversion system that delivers ultra-low latency and high-quality performance. Our approach leverages an articulatory feature space to naturally disentangle content and speaker characteristics, facilitating more robust and interpretable voice transformations. Additionally, the integration of differentiable digital signal processing (DDSP) enables efficient vocoding directly from articulatory features, significantly reducing conversion latency. Experimental evaluations demonstrate that, while maintaining synthesis quality comparable to the current state-of-the-art (SOTA) method, RT-VC achieves a CPU latency of 61.4 ms, representing a 13.3\% reduction in latency.
#### AC/DC: LLM-based Audio Comprehension via Dialogue Continuation
 - **Authors:** Yusuke Fujita, Tomoya Mizumoto, Atsushi Kojima, Lianbo Liu, Yui Sudo
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Computation and Language (cs.CL); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2506.10312

 - **Pdf link:** https://arxiv.org/pdf/2506.10312

 - **Abstract**
 We propose an instruction-following audio comprehension model that leverages the dialogue continuation ability of large language models (LLMs). Instead of directly generating target captions in training data, the proposed method trains a model to produce responses as if the input caption triggered a dialogue. This dialogue continuation training mitigates the caption variation problem. Learning to continue a dialogue effectively captures the caption's meaning beyond its surface-level words. As a result, our model enables zero-shot instruction-following capability without multitask instruction tuning, even trained solely on audio captioning datasets. Experiments on AudioCaps, WavCaps, and Clotho datasets with AudioBench audio-scene question-answering tests demonstrate our model's ability to follow various unseen instructions.
#### Joint ASR and Speaker Role Tagging with Serialized Output Training
 - **Authors:** Anfeng Xu, Tiantian Feng, Shrikanth Narayanan
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2506.10349

 - **Pdf link:** https://arxiv.org/pdf/2506.10349

 - **Abstract**
 Automatic Speech Recognition systems have made significant progress with large-scale pre-trained models. However, most current systems focus solely on transcribing the speech without identifying speaker roles, a function that is critical for conversational AI. In this work, we investigate the use of serialized output training (SOT) for joint ASR and speaker role tagging. By augmenting Whisper with role-specific tokens and fine-tuning it with SOT, we enable the model to generate role-aware transcriptions in a single decoding pass. We compare the SOT approach against a self-supervised previous baseline method on two real-world conversational datasets. Our findings show that this approach achieves more than 10% reduction in multi-talker WER, demonstrating its feasibility as a unified model for speaker-role aware speech transcription.
#### Robust Unsupervised Adaptation of a Speech Recogniser Using Entropy Minimisation and Speaker Codes
 - **Authors:** Rogier C. van Dalen, Shucong Zhang, Titouan Parcollet, Sourav Bhattacharya
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Computation and Language (cs.CL); Machine Learning (cs.LG)
 - **Arxiv link:** https://arxiv.org/abs/2506.10653

 - **Pdf link:** https://arxiv.org/pdf/2506.10653

 - **Abstract**
 Speech recognisers usually perform optimally only in a specific environment and need to be adapted to work well in another. For adaptation to a new speaker, there is often too little data for fine-tuning to be robust, and that data is usually unlabelled. This paper proposes a combination of approaches to make adaptation to a single minute of data robust. First, instead of estimating the adaptation parameters with cross-entropy on a single error-prone hypothesis or "pseudo-label", this paper proposes a novel loss function, the conditional entropy over complete hypotheses. Using multiple hypotheses makes adaptation more robust to errors in the initial recognition. Second, a "speaker code" characterises a speaker in a vector short enough that it requires little data to estimate. On a far-field noise-augmented version of Common Voice, the proposed scheme yields a 20% relative improvement in word error rate on one minute of adaptation data, increasing on 10 minutes to 29%.
#### FairASR: Fair Audio Contrastive Learning for Automatic Speech Recognition
 - **Authors:** Jongsuk Kim, Jaemyung Yu, Minchan Kwon, Junmo Kim
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2506.10747

 - **Pdf link:** https://arxiv.org/pdf/2506.10747

 - **Abstract**
 Large-scale ASR models have achieved remarkable gains in accuracy and robustness. However, fairness issues remain largely unaddressed despite their critical importance in real-world applications. In this work, we introduce FairASR, a system that mitigates demographic bias by learning representations that are uninformative about group membership, enabling fair generalization across demographic groups. Leveraging a multi-demographic dataset, our approach employs a gradient reversal layer to suppress demographic-discriminative features while maintaining the ability to capture generalizable speech patterns through an unsupervised contrastive loss. Experimental results show that FairASR delivers competitive overall ASR performance while significantly reducing performance disparities across different demographic groups.
#### Multimodal Emotion Coupling via Speech-to-Facial and Bodily Gestures in Dyadic Interaction
 - **Authors:** Von Ralph Dane Marquez Herbuela, Yukie Nagai
 - **Subjects:** Subjects:
Multimedia (cs.MM); Machine Learning (cs.LG); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2506.10010

 - **Pdf link:** https://arxiv.org/pdf/2506.10010

 - **Abstract**
 Human emotional expression emerges through coordinated vocal, facial, and gestural signals. While speech face alignment is well established, the broader dynamics linking emotionally expressive speech to regional facial and hand motion remains critical for gaining a deeper insight into how emotional and behavior cues are communicated in real interactions. Further modulating the coordination is the structure of conversational exchange like sequential turn taking, which creates stable temporal windows for multimodal synchrony, and simultaneous speech, often indicative of high arousal moments, disrupts this alignment and impacts emotional clarity. Understanding these dynamics enhances realtime emotion detection by improving the accuracy of timing and synchrony across modalities in both human interactions and AI systems. This study examines multimodal emotion coupling using region specific motion capture from dyadic interactions in the IEMOCAP corpus. Speech features included low level prosody, MFCCs, and model derived arousal, valence, and categorical emotions (Happy, Sad, Angry, Neutral), aligned with 3D facial and hand marker displacements. Expressive activeness was quantified through framewise displacement magnitudes, and speech to gesture prediction mapped speech features to facial and hand movements. Nonoverlapping speech consistently elicited greater activeness particularly in the lower face and mouth. Sadness showed increased expressivity during nonoverlap, while anger suppressed gestures during overlaps. Predictive mapping revealed highest accuracy for prosody and MFCCs in articulatory regions while arousal and valence had lower and more context sensitive correlations. Notably, hand speech synchrony was enhanced under low arousal and overlapping speech, but not for valence.
#### The 2025 PNPL Competition: Speech Detection and Phoneme Classification in the LibriBrain Dataset
 - **Authors:** Gilad Landau, Miran Özdogan, Gereon Elvers, Francesco Mantegna, Pratik Somaiya, Dulhan Jayalath, Luisa Kurth, Teyun Kwon, Brendan Shillingford, Greg Farquhar, Minqi Jiang, Karim Jerbi, Hamza Abdelhedi, Yorguin Mantilla Ramos, Caglar Gulcehre, Mark Woolrich, Natalie Voets, Oiwi Parker Jones
 - **Subjects:** Subjects:
Machine Learning (cs.LG); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2506.10165

 - **Pdf link:** https://arxiv.org/pdf/2506.10165

 - **Abstract**
 The advance of speech decoding from non-invasive brain data holds the potential for profound societal impact. Among its most promising applications is the restoration of communication to paralysed individuals affected by speech deficits such as dysarthria, without the need for high-risk surgical interventions. The ultimate aim of the 2025 PNPL competition is to produce the conditions for an "ImageNet moment" or breakthrough in non-invasive neural decoding, by harnessing the collective power of the machine learning community. To facilitate this vision we present the largest within-subject MEG dataset recorded to date (LibriBrain) together with a user-friendly Python library (pnpl) for easy data access and integration with deep learning frameworks. For the competition we define two foundational tasks (i.e. Speech Detection and Phoneme Classification from brain data), complete with standardised data splits and evaluation metrics, illustrative benchmark models, online tutorial code, a community discussion board, and public leaderboard for submissions. To promote accessibility and participation the competition features a Standard track that emphasises algorithmic innovation, as well as an Extended track that is expected to reward larger-scale computing, accelerating progress toward a non-invasive brain-computer interface for speech.
#### FedMLAC: Mutual Learning Driven Heterogeneous Federated Audio Classification
 - **Authors:** Jun Bai, Rajib Rana, Di Wu, Youyang Qu, Xiaohui Tao, Ji Zhang
 - **Subjects:** Subjects:
Sound (cs.SD); Distributed, Parallel, and Cluster Computing (cs.DC); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2506.10207

 - **Pdf link:** https://arxiv.org/pdf/2506.10207

 - **Abstract**
 Federated Learning (FL) provides a privacy-preserving paradigm for training audio classification (AC) models across distributed clients without sharing raw data. However, Federated Audio Classification (FedAC) faces three critical challenges that substantially hinder performance: data heterogeneity, model heterogeneity, and data poisoning. While prior works have attempted to address these issues, they are typically treated independently, lacking a unified and robust solution suited to real-world federated audio scenarios. To bridge this gap, we propose FedMLAC, a unified mutual learning framework designed to simultaneously tackle these challenges in FedAC. Specifically, FedMLAC introduces a dual-model architecture on each client, comprising a personalized local AC model and a lightweight, globally shared Plug-in model. Through bidirectional knowledge distillation, the Plug-in model enables global knowledge transfer while adapting to client-specific data distributions, thus supporting both generalization and personalization. To further enhance robustness against corrupted audio data, we develop a Layer-wise Pruning Aggregation (LPA) strategy that filters unreliable Plug-in model updates based on parameter deviations during server-side aggregation. Extensive experiments on four diverse audio classification benchmarks, spanning both speech and non-speech tasks, demonstrate that FedMLAC consistently outperforms existing state-of-the-art methods in terms of classification accuracy and robustness to noisy data.
#### Discrete Audio Tokens: More Than a Survey!
 - **Authors:** Pooneh Mousavi, Gallil Maimon, Adel Moumen, Darius Petermann, Jiatong Shi, Haibin Wu, Haici Yang, Anastasia Kuznetsova, Artem Ploujnikov, Ricard Marxer, Bhuvana Ramabhadran, Benjamin Elizalde, Loren Lugosch, Jinyu Li, Cem Subakan, Phil Woodland, Minje Kim, Hung-yi Lee, Shinji Watanabe, Yossi Adi, Mirco Ravanelli
 - **Subjects:** Subjects:
Sound (cs.SD); Artificial Intelligence (cs.AI); Computation and Language (cs.CL); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2506.10274

 - **Pdf link:** https://arxiv.org/pdf/2506.10274

 - **Abstract**
 Discrete audio tokens are compact representations that aim to preserve perceptual quality, phonetic content, and speaker characteristics while enabling efficient storage and inference, as well as competitive performance across diverse downstream this http URL provide a practical alternative to continuous features, enabling the integration of speech and audio into modern large language models (LLMs). As interest in token-based audio processing grows, various tokenization methods have emerged, and several surveys have reviewed the latest progress in the field. However, existing studies often focus on specific domains or tasks and lack a unified comparison across various benchmarks. This paper presents a systematic review and benchmark of discrete audio tokenizers, covering three domains: speech, music, and general audio. We propose a taxonomy of tokenization approaches based on encoder-decoder, quantization techniques, training paradigm, streamability, and application domains. We evaluate tokenizers on multiple benchmarks for reconstruction, downstream performance, and acoustic language modeling, and analyze trade-offs through controlled ablation studies. Our findings highlight key limitations, practical considerations, and open challenges, providing insight and guidance for future research in this rapidly evolving area. For more information, including our main results and tokenizer database, please refer to our website: this https URL.
#### Scheduled Interleaved Speech-Text Training for Speech-to-Speech Translation with LLMs
 - **Authors:** Hayato Futami, Emiru Tsunoo, Yosuke Kashiwagi, Yuki Ito, Hassan Shahmohammadi, Siddhant Arora, Shinji Watanabe
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2506.10299

 - **Pdf link:** https://arxiv.org/pdf/2506.10299

 - **Abstract**
 Speech-to-speech translation (S2ST) has been advanced with large language models (LLMs), which are fine-tuned on discrete speech units. In such approaches, modality adaptation from text to speech has been an issue. LLMs are trained on text-only data, which presents challenges to adapt them to speech modality with limited speech-to-speech data. To address the training difficulty, we propose scheduled interleaved speech--text training in this study. We use interleaved speech--text units instead of speech units during training, where aligned text tokens are interleaved at the word level. We gradually decrease the ratio of text as training progresses, to facilitate progressive modality adaptation from text to speech. We conduct experimental evaluations by fine-tuning LLaMA3.2-1B for S2ST on the CVSS dataset. We show that the proposed method consistently improves the translation performances, especially for languages with limited training data.
#### DanceChat: Large Language Model-Guided Music-to-Dance Generation
 - **Authors:** Qing Wang, Xiaohang Yang, Yilan Dong, Naveen Raj Govindaraj, Gregory Slabaugh, Shanxin Yuan
 - **Subjects:** Subjects:
Computer Vision and Pattern Recognition (cs.CV); Multimedia (cs.MM); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2506.10574

 - **Pdf link:** https://arxiv.org/pdf/2506.10574

 - **Abstract**
 Music-to-dance generation aims to synthesize human dance motion conditioned on musical input. Despite recent progress, significant challenges remain due to the semantic gap between music and dance motion, as music offers only abstract cues, such as melody, groove, and emotion, without explicitly specifying the physical movements. Moreover, a single piece of music can produce multiple plausible dance interpretations. This one-to-many mapping demands additional guidance, as music alone provides limited information for generating diverse dance movements. The challenge is further amplified by the scarcity of paired music and dance data, which restricts the modelâĂŹs ability to learn diverse dance patterns. In this paper, we introduce DanceChat, a Large Language Model (LLM)-guided music-to-dance generation approach. We use an LLM as a choreographer that provides textual motion instructions, offering explicit, high-level guidance for dance generation. This approach goes beyond implicit learning from music alone, enabling the model to generate dance that is both more diverse and better aligned with musical styles. Our approach consists of three components: (1) an LLM-based pseudo instruction generation module that produces textual dance guidance based on music style and structure, (2) a multi-modal feature extraction and fusion module that integrates music, rhythm, and textual guidance into a shared representation, and (3) a diffusion-based motion synthesis module together with a multi-modal alignment loss, which ensures that the generated dance is aligned with both musical and textual cues. Extensive experiments on AIST++ and human evaluations show that DanceChat outperforms state-of-the-art methods both qualitatively and quantitatively.
#### Analyzing the relationships between pretraining language, phonetic, tonal, and speaker information in self-supervised speech models
 - **Authors:** Michele Gubian, Ioana Krehan, Oli Liu, James Kirby, Sharon Goldwater
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2506.10855

 - **Pdf link:** https://arxiv.org/pdf/2506.10855

 - **Abstract**
 Analyses of self-supervised speech models have begun to reveal where and how they represent different types of information. However, almost all analyses have focused on English. Here, we examine how wav2vec2 models trained on four different languages encode both language-matched and non-matched speech. We use probing classifiers and geometric analyses to examine how phones, lexical tones, and speaker information are represented. We show that for all pretraining and test languages, the subspaces encoding phones, tones, and speakers are largely orthogonal, and that layerwise patterns of probing accuracy are similar, with a relatively small advantage for matched-language phone and tone (but not speaker) probes in the later layers. Our findings suggest that the structure of representations learned by wav2vec2 is largely independent of the speech material used during pretraining.


by Zyzzyva0381 (Windy). 


2025-06-13
