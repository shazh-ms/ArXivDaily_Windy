# Showing new listings for Tuesday, 17 June 2025
Auto update papers at about 2:30am UTC (10:30am Beijing time) every weekday.


阅读 `Usage.md`了解如何使用此repo实现个性化的Arxiv论文推送

See `Usage.md` for instructions on how to personalize the repo. 


Keyword list: ['text-to-speech', 'text to speech', 'tts', 'LLM-based', 'speech', 'voice']


Excluded: []


### Today: 23papers 
#### CMT-LLM: Contextual Multi-Talker ASR Utilizing Large Language Models
 - **Authors:** Jiajun He, Naoki Sawada, Koichi Miyazaki, Tomoki Toda
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Artificial Intelligence (cs.AI); Computation and Language (cs.CL); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2506.12059

 - **Pdf link:** https://arxiv.org/pdf/2506.12059

 - **Abstract**
 In real-world applications, automatic speech recognition (ASR) systems must handle overlapping speech from multiple speakers and recognize rare words like technical terms. Traditional methods address multi-talker ASR and contextual biasing separately, limiting performance in complex scenarios. We propose a unified framework that combines multi-talker overlapping speech recognition and contextual biasing into a single task. Our ASR method integrates pretrained speech encoders and large language models (LLMs), using optimized finetuning strategies. We also introduce a two-stage filtering algorithm to efficiently identify relevant rare words from large biasing lists and incorporate them into the LLM's prompt input, enhancing rare word recognition. Experiments show that our approach outperforms traditional contextual biasing methods, achieving a WER of 7.9% on LibriMix and 32.9% on AMI SDM when the biasing size is 1,000, demonstrating its effectiveness in complex speech scenarios.
#### Evaluating Logit-Based GOP Scores for Mispronunciation Detection
 - **Authors:** Aditya Kamlesh Parikh, Cristian Tejedor-Garcia, Catia Cucchiarini, Helmer Strik
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Artificial Intelligence (cs.AI); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2506.12067

 - **Pdf link:** https://arxiv.org/pdf/2506.12067

 - **Abstract**
 Pronunciation assessment relies on goodness of pronunciation (GOP) scores, traditionally derived from softmax-based posterior probabilities. However, posterior probabilities may suffer from overconfidence and poor phoneme separation, limiting their effectiveness. This study compares logit-based GOP scores with probability-based GOP scores for mispronunciation detection. We conducted our experiment on two L2 English speech datasets spoken by Dutch and Mandarin speakers, assessing classification performance and correlation with human ratings. Logit-based methods outperform probability-based GOP in classification, but their effectiveness depends on dataset characteristics. The maximum logit GOP shows the strongest alignment with human perception, while a combination of different GOP scores balances probability and logit features. The findings suggest that hybrid GOP methods incorporating uncertainty modeling and phoneme-specific weighting improve pronunciation assessment.
#### Seamless Dysfluent Speech Text Alignment for Disordered Speech Analysis
 - **Authors:** Zongli Ye, Jiachen Lian, Xuanru Zhou, Jinming Zhang, Haodong Li, Shuhe Li, Chenxu Guo, Anaisha Das, Peter Park, Zoe Ezzes, Jet Vonk, Brittany Morin, Rian Bogley, Lisa Wauters, Zachary Miller, Maria Gorno-Tempini, Gopala Anumanchipalli
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Artificial Intelligence (cs.AI); Computation and Language (cs.CL); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2506.12073

 - **Pdf link:** https://arxiv.org/pdf/2506.12073

 - **Abstract**
 Accurate alignment of dysfluent speech with intended text is crucial for automating the diagnosis of neurodegenerative speech disorders. Traditional methods often fail to model phoneme similarities effectively, limiting their performance. In this work, we propose Neural LCS, a novel approach for dysfluent text-text and speech-text alignment. Neural LCS addresses key challenges, including partial alignment and context-aware similarity mapping, by leveraging robust phoneme-level modeling. We evaluate our method on a large-scale simulated dataset, generated using advanced data simulation techniques, and real PPA data. Neural LCS significantly outperforms state-of-the-art models in both alignment accuracy and dysfluent speech segmentation. Our results demonstrate the potential of Neural LCS to enhance automated systems for diagnosing and analyzing speech disorders, offering a more accurate and linguistically grounded solution for dysfluent speech alignment.
#### Mitigating Non-Target Speaker Bias in Guided Speaker Embedding
 - **Authors:** Shota Horiguchi, Takanori Ashihara, Marc Delcroix, Atsushi Ando, Naohiro Tawara
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2506.12500

 - **Pdf link:** https://arxiv.org/pdf/2506.12500

 - **Abstract**
 Obtaining high-quality speaker embeddings in multi-speaker conditions is crucial for many applications. A recently proposed guided speaker embedding framework, which utilizes speech activities of target and non-target speakers as clues, drastically improved embeddings under severe overlap with small degradation in low-overlap cases. However, since extreme overlaps are rare in natural conversations, this degradation cannot be overlooked. This paper first reveals that the degradation is caused by the global-statistics-based modules, widely used in speaker embedding extractors, being overly sensitive to intervals containing only non-target speakers. As a countermeasure, we propose an extension of such modules that exploit the target speaker activity clues, to compute statistics from intervals where the target is active. The proposed method improves speaker verification performance in both low and high overlap ratios, and diarization performance on multiple datasets.
#### Towards Neural Audio Codec Source Parsing
 - **Authors:** Orchid Chetia Phukan, Girish, Mohd Mujtaba Akhtar, Arun Balaji Buduru, Rajesh Sharma
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2506.12627

 - **Pdf link:** https://arxiv.org/pdf/2506.12627

 - **Abstract**
 A new class of audio deepfakes-codecfakes (CFs)-has recently caught attention, synthesized by Audio Language Models that leverage neural audio codecs (NACs) in the backend. In response, the community has introduced dedicated benchmarks and tailored detection strategies. As the field advances, efforts have moved beyond binary detection toward source attribution, including open-set attribution, which aims to identify the NAC responsible for generation and flag novel, unseen ones during inference. This shift toward source attribution improves forensic interpretability and accountability. However, open-set attribution remains fundamentally limited: while it can detect that a NAC is unfamiliar, it cannot characterize or identify individual unseen codecs. It treats such inputs as generic ``unknowns'', lacking insight into their internal configuration. This leads to major shortcomings: limited generalization to new NACs and inability to resolve fine-grained variations within NAC families. To address these gaps, we propose Neural Audio Codec Source Parsing (NACSP) - a paradigm shift that reframes source attribution for CFs as structured regression over generative NAC parameters such as quantizers, bandwidth, and sampling rate. We formulate NACSP as a multi-task regression task for predicting these NAC parameters and establish the first comprehensive benchmark using various state-of-the-art speech pre-trained models (PTMs). To this end, we propose HYDRA, a novel framework that leverages hyperbolic geometry to disentangle complex latent properties from PTM representations. By employing task-specific attention over multiple curvature-aware hyperbolic subspaces, HYDRA enables superior multi-task generalization. Our extensive experiments show HYDRA achieves top results on benchmark CFs datasets compared to baselines operating in Euclidean space.
#### Magnetoencephalography (MEG) Based Non-Invasive Chinese Speech Decoding
 - **Authors:** Zhihong Jia, Hongbin Wang, Yuanzhong Shen, Feng Hu, Jiayu An, Kai Shu, Dongrui Wu
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2506.12817

 - **Pdf link:** https://arxiv.org/pdf/2506.12817

 - **Abstract**
 As an emerging paradigm of brain-computer interfaces (BCIs), speech BCI has the potential to directly reflect auditory perception and thoughts, offering a promising communication alternative for patients with aphasia. Chinese is one of the most widely spoken languages in the world, whereas there is very limited research on speech BCIs for Chinese language. This paper reports a text-magnetoencephalography (MEG) dataset for non-invasive Chinese speech BCIs. It also proposes a multi-modality assisted speech decoding (MASD) algorithm to capture both text and acoustic information embedded in brain signals during speech activities. Experiment results demonstrated the effectiveness of both our text-MEG dataset and our proposed MASD algorithm. To our knowledge, this is the first study on modality-assisted decoding for non-invasive speech BCIs.
#### ZipVoice: Fast and High-Quality Zero-Shot Text-to-Speech with Flow Matching
 - **Authors:** Han Zhu, Wei Kang, Zengwei Yao, Liyong Guo, Fangjun Kuang, Zhaoqing Li, Weiji Zhuang, Long Lin, Daniel Povey
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2506.13053

 - **Pdf link:** https://arxiv.org/pdf/2506.13053

 - **Abstract**
 Existing large-scale zero-shot text-to-speech (TTS) models deliver high speech quality but suffer from slow inference speeds due to massive parameters. To address this issue, this paper introduces ZipVoice, a high-quality flow-matching-based zero-shot TTS model with a compact model size and fast inference speed. Key designs include: 1) a Zipformer-based flow-matching decoder to maintain adequate modeling capabilities under constrained size; 2) Average upsampling-based initial speech-text alignment and Zipformer-based text encoder to improve speech intelligibility; 3) A flow distillation method to reduce sampling steps and eliminate the inference overhead associated with classifier-free guidance. Experiments on 100k hours multilingual datasets show that ZipVoice matches state-of-the-art models in speech quality, while being 3 times smaller and up to 30 times faster than a DiT-based flow-matching baseline. Codes, model checkpoints and demo samples are publicly available.
#### Instance-Specific Test-Time Training for Speech Editing in the Wild
 - **Authors:** Taewoo Kim, Uijong Lee, Hayoung Park, Choongsang Cho, Nam In Park, Young Han Lee
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2506.13295

 - **Pdf link:** https://arxiv.org/pdf/2506.13295

 - **Abstract**
 Speech editing systems aim to naturally modify speech content while preserving acoustic consistency and speaker identity. However, previous studies often struggle to adapt to unseen and diverse acoustic conditions, resulting in degraded editing performance in real-world scenarios. To address this, we propose an instance-specific test-time training method for speech editing in the wild. Our approach employs direct supervision from ground-truth acoustic features in unedited regions, and indirect supervision in edited regions via auxiliary losses based on duration constraints and phoneme prediction. This strategy mitigates the bandwidth discontinuity problem in speech editing, ensuring smooth acoustic transitions between unedited and edited regions. Additionally, it enables precise control over speech rate by adapting the model to target durations via mask length adjustment during test-time training. Experiments on in-the-wild benchmark datasets demonstrate that our method outperforms existing speech editing systems in both objective and subjective evaluations.
#### BUT System for the MLC-SLM Challenge
 - **Authors:** Alexander Polok, Jiangyu Han, Dominik Klement, Samuele Cornell, Jan Černocký, Lukáš Burget
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2506.13414

 - **Pdf link:** https://arxiv.org/pdf/2506.13414

 - **Abstract**
 We present a two-speaker automatic speech recognition (ASR) system that combines DiCoW -- a diarization-conditioned variant of Whisper -- with DiariZen, a diarization pipeline built on top of Pyannote. We first evaluate both systems in out-of-domain (OOD) multilingual scenarios without any fine-tuning. In this scenario, DiariZen consistently outperforms the baseline Pyannote diarization model, demonstrating strong generalization. Despite being fine-tuned on English-only data for target-speaker ASR, DiCoW retains solid multilingual performance, indicating that encoder modifications preserve Whisper's multilingual capabilities. We then fine-tune both DiCoW and DiariZen on the MLC-SLM challenge data. The fine-tuned DiariZen continues to outperform the fine-tuned Pyannote baseline, while DiCoW sees further gains from domain adaptation. Our final system achieves a micro-average tcpWER/CER of 16.75% and ranks second in Task 2 of the MLC-SLM challenge. Lastly, we identify several labeling inconsistencies in the training data -- such as missing speech segments and incorrect silence annotations -- which can hinder diarization fine-tuning. We propose simple mitigation strategies to address these issues and improve system robustness.
#### SpeechRefiner: Towards Perceptual Quality Refinement for Front-End Algorithms
 - **Authors:** Sirui Li, Shuai Wang, Zhijun Liu, Zhongjie Jiang, Yannan Wang, Haizhou Li
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2506.13709

 - **Pdf link:** https://arxiv.org/pdf/2506.13709

 - **Abstract**
 Speech pre-processing techniques such as denoising, de-reverberation, and separation, are commonly employed as front-ends for various downstream speech processing tasks. However, these methods can sometimes be inadequate, resulting in residual noise or the introduction of new artifacts. Such deficiencies are typically not captured by metrics like SI-SNR but are noticeable to human listeners. To address this, we introduce SpeechRefiner, a post-processing tool that utilizes Conditional Flow Matching (CFM) to improve the perceptual quality of speech. In this study, we benchmark SpeechRefiner against recent task-specific refinement methods and evaluate its performance within our internal processing pipeline, which integrates multiple front-end algorithms. Experiments show that SpeechRefiner exhibits strong generalization across diverse impairment sources, significantly enhancing speech perceptual quality. Audio demos can be found at this https URL.
#### TuneGenie: Reasoning-based LLM agents for preferential music generation
 - **Authors:** Amitesh Pandey, Jafarbek Arifdjanov, Ansh Tiwari
 - **Subjects:** Subjects:
Sound (cs.SD); Multiagent Systems (cs.MA); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2506.12083

 - **Pdf link:** https://arxiv.org/pdf/2506.12083

 - **Abstract**
 Recently, Large language models (LLMs) have shown great promise across a diversity of tasks, ranging from generating images to reasoning spatially. Considering their remarkable (and growing) textual reasoning capabilities, we investigate LLMs' potency in conducting analyses of an individual's preferences in music (based on playlist metadata, personal write-ups, etc.) and producing effective prompts (based on these analyses) to be passed to Suno AI (a generative AI tool for music production). Our proposition of a novel LLM-based textual representation to music model (which we call TuneGenie) and the various methods we develop to evaluate & benchmark similar models add to the increasing (and increasingly controversial) corpus of research on the use of AI in generating art.
#### Adapting Whisper for Streaming Speech Recognition via Two-Pass Decoding
 - **Authors:** Haoran Zhou, Xingchen Song, Brendan Fahy, Qiaochu Song, Binbin Zhang, Zhendong Peng, Anshul Wadhawan, Denglin Jiang, Apurv Verma, Vinay Ramesh, Srivas Prasad, Michele M. Franceschini
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2506.12154

 - **Pdf link:** https://arxiv.org/pdf/2506.12154

 - **Abstract**
 OpenAI Whisper is a family of robust Automatic Speech Recognition (ASR) models trained on 680,000 hours of audio. However, its encoder-decoder architecture, trained with a sequence-to-sequence objective, lacks native support for streaming ASR. In this paper, we fine-tune Whisper for streaming ASR using the WeNet toolkit by adopting a Unified Two-pass (U2) structure. We introduce an additional Connectionist Temporal Classification (CTC) decoder trained with causal attention masks to generate streaming partial transcripts, while the original Whisper decoder reranks these partial outputs. Our experiments on LibriSpeech and an earnings call dataset demonstrate that, with adequate fine-tuning data, Whisper can be adapted into a capable streaming ASR model. We also introduce a hybrid tokenizer approach, which uses a smaller token space for the CTC decoder while retaining Whisper's original token space for the attention decoder, resulting in improved data efficiency and generalization.
#### SSLAM: Enhancing Self-Supervised Models with Audio Mixtures for Polyphonic Soundscapes
 - **Authors:** Tony Alex, Sara Ahmed, Armin Mustafa, Muhammad Awais, Philip JB Jackson
 - **Subjects:** Subjects:
Sound (cs.SD); Artificial Intelligence (cs.AI); Machine Learning (cs.LG); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2506.12222

 - **Pdf link:** https://arxiv.org/pdf/2506.12222

 - **Abstract**
 Self-supervised pre-trained audio networks have seen widespread adoption in real-world systems, particularly in multi-modal large language models. These networks are often employed in a frozen state, under the assumption that the SSL pre-training has sufficiently equipped them to handle real-world audio. However, a critical question remains: how well do these models actually perform in real-world conditions, where audio is typically polyphonic and complex, involving multiple overlapping sound sources? Current audio SSL methods are often benchmarked on datasets predominantly featuring monophonic audio, such as environmental sounds, and speech. As a result, the ability of SSL models to generalize to polyphonic audio, a common characteristic in natural scenarios, remains underexplored. This limitation raises concerns about the practical robustness of SSL models in more realistic audio settings. To address this gap, we introduce Self-Supervised Learning from Audio Mixtures (SSLAM), a novel direction in audio SSL research, designed to improve, designed to improve the model's ability to learn from polyphonic data while maintaining strong performance on monophonic data. We thoroughly evaluate SSLAM on standard audio SSL benchmark datasets which are predominantly monophonic and conduct a comprehensive comparative analysis against SOTA methods using a range of high-quality, publicly available polyphonic datasets. SSLAM not only improves model performance on polyphonic audio, but also maintains or exceeds performance on standard audio SSL benchmarks. Notably, it achieves up to a 3.9\% improvement on the AudioSet-2M (AS-2M), reaching a mean average precision (mAP) of 50.2. For polyphonic datasets, SSLAM sets new SOTA in both linear evaluation and fine-tuning regimes with performance improvements of up to 9.1\% (mAP).
#### Improving Speech Enhancement with Multi-Metric Supervision from Learned Quality Assessment
 - **Authors:** Wei Wang, Wangyou Zhang, Chenda Li, Jiatong Shi, Shinji Watanabe, Yanmin Qian
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2506.12260

 - **Pdf link:** https://arxiv.org/pdf/2506.12260

 - **Abstract**
 Speech quality assessment (SQA) aims to predict the perceived quality of speech signals under a wide range of distortions. It is inherently connected to speech enhancement (SE), which seeks to improve speech quality by removing unwanted signal components. While SQA models are widely used to evaluate SE performance, their potential to guide SE training remains underexplored. In this work, we investigate a training framework that leverages a SQA model, trained to predict multiple evaluation metrics from a public SE leaderboard, as a supervisory signal for SE. This approach addresses a key limitation of conventional SE objectives, such as SI-SNR, which often fail to align with perceptual quality and generalize poorly across evaluation metrics. Moreover, it enables training on real-world data where clean references are unavailable. Experiments on both simulated and real-world test sets show that SQA-guided training consistently improves performance across a range of quality metrics.
#### Phonikud: Hebrew Grapheme-to-Phoneme Conversion for Real-Time Text-to-Speech
 - **Authors:** Yakov Kolani, Maxim Melichov, Cobi Calev, Morris Alper
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2506.12311

 - **Pdf link:** https://arxiv.org/pdf/2506.12311

 - **Abstract**
 Real-time text-to-speech (TTS) for Modern Hebrew is challenging due to the language's orthographic complexity. Existing solutions ignore crucial phonetic features such as stress that remain underspecified even when vowel marks are added. To address these limitations, we introduce Phonikud, a lightweight, open-source Hebrew grapheme-to-phoneme (G2P) system that outputs fully-specified IPA transcriptions. Our approach adapts an existing diacritization model with lightweight adaptors, incurring negligible additional latency. We also contribute the ILSpeech dataset of transcribed Hebrew speech with IPA annotations, serving as a benchmark for Hebrew G2P and as training data for TTS systems. Our results demonstrate that Phonikud G2P conversion more accurately predicts phonemes from Hebrew text compared to prior methods, and that this enables training of effective real-time Hebrew TTS models with superior speed-accuracy trade-offs. We release our code, data, and models at this https URL.
#### Speech-Language Models with Decoupled Tokenizers and Multi-Token Prediction
 - **Authors:** Xiaoran Fan, Zhichao Sun, Yangfan Gao, Jingfei Xiong, Hang Yan, Yifei Cao, Jiajun Sun, Shuo Li, Zhihao Zhang, Zhiheng Xi, Yuhao Zhou, Senjie Jin, Changhao Jiang, Junjie Ye, Ming Zhang, Rui Zheng, Zhenhua Han, Yunke Zhang, Demei Yan, Shaokang Dong, Tao Ji, Tao Gui, Qi Zhang, Xuanjing Huang
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Artificial Intelligence (cs.AI); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2506.12537

 - **Pdf link:** https://arxiv.org/pdf/2506.12537

 - **Abstract**
 Speech-language models (SLMs) offer a promising path toward unifying speech and text understanding and generation. However, challenges remain in achieving effective cross-modal alignment and high-quality speech generation. In this work, we systematically investigate the impact of key components (i.e., speech tokenizers, speech heads, and speaker modeling) on the performance of LLM-centric SLMs. We compare coupled, semi-decoupled, and fully decoupled speech tokenizers under a fair SLM framework and find that decoupled tokenization significantly improves alignment and synthesis quality. To address the information density mismatch between speech and text, we introduce multi-token prediction (MTP) into SLMs, enabling each hidden state to decode multiple speech tokens. This leads to up to 12$\times$ faster decoding and a substantial drop in word error rate (from 6.07 to 3.01). Furthermore, we propose a speaker-aware generation paradigm and introduce RoleTriviaQA, a large-scale role-playing knowledge QA benchmark with diverse speaker identities. Experiments demonstrate that our methods enhance both knowledge understanding and speaker consistency.
#### StreamMel: Real-Time Zero-shot Text-to-Speech via Interleaved Continuous Autoregressive Modeling
 - **Authors:** Hui Wang, Yifan Yang, Shujie Liu, Jinyu Li, Lingwei Meng, Yanqing Liu, Jiaming Zhou, Haoqin Sun, Yan Lu, Yong Qin
 - **Subjects:** Subjects:
Sound (cs.SD); Computation and Language (cs.CL); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2506.12570

 - **Pdf link:** https://arxiv.org/pdf/2506.12570

 - **Abstract**
 Recent advances in zero-shot text-to-speech (TTS) synthesis have achieved high-quality speech generation for unseen speakers, but most systems remain unsuitable for real-time applications because of their offline design. Current streaming TTS paradigms often rely on multi-stage pipelines and discrete representations, leading to increased computational cost and suboptimal system performance. In this work, we propose StreamMel, a pioneering single-stage streaming TTS framework that models continuous mel-spectrograms. By interleaving text tokens with acoustic frames, StreamMel enables low-latency, autoregressive synthesis while preserving high speaker similarity and naturalness. Experiments on LibriSpeech demonstrate that StreamMel outperforms existing streaming TTS baselines in both quality and latency. It even achieves performance comparable to offline systems while supporting efficient real-time generation, showcasing broad prospects for integration with real-time speech large language models. Audio samples are available at: this https URL.
#### SC-SOT: Conditioning the Decoder on Diarized Speaker Information for End-to-End Overlapped Speech Recognition
 - **Authors:** Yuta Hirano, Sakriani Sakti
 - **Subjects:** Subjects:
Sound (cs.SD); Computation and Language (cs.CL); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2506.12672

 - **Pdf link:** https://arxiv.org/pdf/2506.12672

 - **Abstract**
 We propose Speaker-Conditioned Serialized Output Training (SC-SOT), an enhanced SOT-based training for E2E multi-talker ASR. We first probe how SOT handles overlapped speech, and we found the decoder performs implicit speaker separation. We hypothesize this implicit separation is often insufficient due to ambiguous acoustic cues in overlapping regions. To address this, SC-SOT explicitly conditions the decoder on speaker information, providing detailed information about "who spoke when". Specifically, we enhance the decoder by incorporating: (1) speaker embeddings, which allow the model to focus on the acoustic characteristics of the target speaker, and (2) speaker activity information, which guides the model to suppress non-target speakers. The speaker embeddings are derived from a jointly trained E2E speaker diarization model, mitigating the need for speaker enrollment. Experimental results demonstrate the effectiveness of our conditioning approach on overlapped speech.
#### I$^2$S-TFCKD: Intra-Inter Set Knowledge Distillation with Time-Frequency Calibration for Speech Enhancement
 - **Authors:** Jiaming Cheng, Ruiyu Liang, Chao Xu, Ye Ni, Wei Zhou, Björn W. Schuller, Xiaoshuai Hao
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2506.13127

 - **Pdf link:** https://arxiv.org/pdf/2506.13127

 - **Abstract**
 In recent years, complexity compression of neural network (NN)-based speech enhancement (SE) models has gradually attracted the attention of researchers, especially in scenarios with limited hardware resources or strict latency requirements. The main difficulties and challenges lie in achieving a balance between complexity and performance according to the characteristics of the task. In this paper, we propose an intra-inter set knowledge distillation (KD) framework with time-frequency calibration (I$^2$S-TFCKD) for SE. Different from previous distillation strategies for SE, the proposed framework fully utilizes the time-frequency differential information of speech while promoting global knowledge flow. Firstly, we propose a multi-layer interactive distillation based on dual-stream time-frequency cross-calibration, which calculates the teacher-student similarity calibration weights in the time and frequency domains respectively and performs cross-weighting, thus enabling refined allocation of distillation contributions across different layers according to speech characteristics. Secondly, we construct a collaborative distillation paradigm for intra-set and inter-set correlations. Within a correlated set, multi-layer teacher-student features are pairwise matched for calibrated distillation. Subsequently, we generate representative features from each correlated set through residual fusion to form the fused feature set that enables inter-set knowledge interaction. The proposed distillation strategy is applied to the dual-path dilated convolutional recurrent network (DPDCRN) that ranked first in the SE track of the L3DAS23 challenge. Objective evaluations demonstrate that the proposed KD strategy consistently and effectively improves the performance of the low-complexity student model and outperforms other distillation schemes.
#### NTU Speechlab LLM-Based Multilingual ASR System for Interspeech MLC-SLM Challenge 2025
 - **Authors:** Yizhou Peng, Bin Wang, Yi-Wen Chao, Ziyang Ma, Haoyang Zhang, Hexin Liu, Xie Chen, Eng Siong Chng
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2506.13339

 - **Pdf link:** https://arxiv.org/pdf/2506.13339

 - **Abstract**
 This report details the NTU Speechlab system developed for the Interspeech 2025 Multilingual Conversational Speech and Language Model (MLC-SLM) Challenge (Task I), where we achieved 5th place. We present comprehensive analyses of our multilingual automatic speech recognition system, highlighting key advancements in model architecture, data selection, and training strategies. In particular, language-specific prompts and model averaging techniques were instrumental in boosting system performance across diverse languages. Compared to the initial baseline system, our final model reduced the average Mix Error Rate from 20.2% to 10.6%, representing an absolute improvement of 9.6% (a relative improvement of 48%) on the evaluation set. Our results demonstrate the effectiveness of our approach and offer practical insights for future Speech Large Language Models.
#### Bi-directional Context-Enhanced Speech Large Language Models for Multilingual Conversational ASR
 - **Authors:** Yizhou Peng, Hexin Liu, Eng Siong Chng
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2506.13396

 - **Pdf link:** https://arxiv.org/pdf/2506.13396

 - **Abstract**
 This paper introduces the integration of language-specific bi-directional context into a speech large language model (SLLM) to improve multilingual continuous conversational automatic speech recognition (ASR). We propose a character-level contextual masking strategy during training, which randomly removes portions of the context to enhance robustness and better emulate the flawed transcriptions that may occur during inference. For decoding, a two-stage pipeline is utilized: initial isolated segment decoding followed by context-aware re-decoding using neighboring hypotheses. Evaluated on the 1500-hour Multilingual Conversational Speech and Language Model (MLC-SLM) corpus covering eleven languages, our method achieves an 18% relative improvement compared to a strong baseline, outperforming even the model trained on 6000 hours of data for the MLC-SLM competition. These results underscore the significant benefit of incorporating contextual information in multilingual continuous conversational ASR.
#### Qwen vs. Gemma Integration with Whisper: A Comparative Study in Multilingual SpeechLLM Systems
 - **Authors:** Tuan Nguyen, Long-Vu Hoang, Huy-Dat Tran
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2506.13596

 - **Pdf link:** https://arxiv.org/pdf/2506.13596

 - **Abstract**
 This paper presents our system for the MLC-SLM Challenge 2025, focusing on multilingual speech recognition and language modeling with large language models (LLMs). Our approach combines a fine-tuned Whisper-large-v3 encoder with efficient projector architectures and various decoder configurations. We employ a three-stage training methodology that progressively optimizes the encoder, projector, and LLM components. Our system achieves competitive performance with a private test average WER/CER result of 16.63% using the Gemma3-12B and 18.6% using the Qwen2.5-7B as decoder-only language model.
#### Stream-Omni: Simultaneous Multimodal Interactions with Large Language-Vision-Speech Model
 - **Authors:** Shaolei Zhang, Shoutao Guo, Qingkai Fang, Yan Zhou, Yang Feng
 - **Subjects:** Subjects:
Artificial Intelligence (cs.AI); Computation and Language (cs.CL); Computer Vision and Pattern Recognition (cs.CV); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2506.13642

 - **Pdf link:** https://arxiv.org/pdf/2506.13642

 - **Abstract**
 The emergence of GPT-4o-like large multimodal models (LMMs) has raised the exploration of integrating text, vision, and speech modalities to support more flexible multimodal interaction. Existing LMMs typically concatenate representation of modalities along the sequence dimension and feed them into a large language model (LLM) backbone. While sequence-dimension concatenation is straightforward for modality integration, it often relies heavily on large-scale data to learn modality alignments. In this paper, we aim to model the relationships between modalities more purposefully, thereby achieving more efficient and flexible modality alignments. To this end, we propose Stream-Omni, a large language-vision-speech model with efficient modality alignments, which can simultaneously support interactions under various modality combinations. Stream-Omni employs LLM as the backbone and aligns the vision and speech to the text based on their relationships. For vision that is semantically complementary to text, Stream-Omni uses sequence-dimension concatenation to achieve vision-text alignment. For speech that is semantically consistent with text, Stream-Omni introduces a CTC-based layer-dimension mapping to achieve speech-text alignment. In this way, Stream-Omni can achieve modality alignments with less data (especially speech), enabling the transfer of text capabilities to other modalities. Experiments on various benchmarks demonstrate that Stream-Omni achieves strong performance on visual understanding, speech interaction, and vision-grounded speech interaction tasks. Owing to the layer-dimensional mapping, Stream-Omni can simultaneously provide intermediate text outputs (such as ASR transcriptions and model responses) during speech interaction, offering users a comprehensive multimodal experience.


by Zyzzyva0381 (Windy). 


2025-06-17
