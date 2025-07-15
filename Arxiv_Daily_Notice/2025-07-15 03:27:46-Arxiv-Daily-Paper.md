# Showing new listings for Tuesday, 15 July 2025
Auto update papers at about 2:30am UTC (10:30am Beijing time) every weekday.


阅读 `Usage.md`了解如何使用此repo实现个性化的Arxiv论文推送

See `Usage.md` for instructions on how to personalize the repo. 


Keyword list: ['text-to-speech', 'text to speech', 'tts', 'LLM-based', 'speech', 'voice']


Excluded: []


### Today: 22papers 
#### SemAlignVC: Enhancing zero-shot timbre conversion using semantic alignment
 - **Authors:** Shivam Mehta, Yingru Liu, Zhenyu Tang, Kainan Peng, Vimal Manohar, Shun Zhang, Mike Seltzer, Qing He, Mingbo Ma
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2507.09070

 - **Pdf link:** https://arxiv.org/pdf/2507.09070

 - **Abstract**
 Zero-shot voice conversion (VC) synthesizes speech in a target speaker's voice while preserving linguistic and paralinguistic content. However, timbre leakage-where source speaker traits persist-remains a challenge, especially in neural codec and LLM-based VC, where quantized representations entangle speaker identity with content. We introduce SemAlignVC, an architecture designed to prevent timbre leakage using SemAlign, a novel method that aligns text and audio representations to ensure speaker-independent semantic encoding. This disentangled representation conditions an autoregressive transformer for high-fidelity conversion without explicit speaker embeddings. Experiments show SemAlignVC significantly reduces timbre leakage, outperforming baselines in speaker timbre similarity, intelligibility, and naturalness, making it a robust, privacy-preserving, and generalizable VC solution. Audio samples can be accessed at this https URL
#### Can We Really Repurpose Multi-Speaker ASR Corpus for Speaker Diarization?
 - **Authors:** Shota Horiguchi, Naohiro Tawara, Takanori Ashihara, Atsushi Ando, Marc Delcroix
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2507.09226

 - **Pdf link:** https://arxiv.org/pdf/2507.09226

 - **Abstract**
 Neural speaker diarization is widely used for overlap-aware speaker diarization, but it requires large multi-speaker datasets for training. To meet this data requirement, large datasets are often constructed by combining multiple corpora, including those originally designed for multi-speaker automatic speech recognition (ASR). However, ASR datasets often feature loosely defined segment boundaries that do not align with the stricter conventions of diarization benchmarks. In this work, we show that such boundary looseness significantly impacts the diarization error rate, reducing evaluation reliability. We also reveal that models trained on data with varying boundary precision tend to learn dataset-specific looseness, leading to poor generalization across out-of-domain datasets. Training with standardized tight boundaries via forced alignment improves not only diarization performance, especially in streaming scenarios, but also ASR performance when combined with simple post-processing.
#### ZipVoice-Dialog: Non-Autoregressive Spoken Dialogue Generation with Flow Matching
 - **Authors:** Han Zhu, Wei Kang, Liyong Guo, Zengwei Yao, Fangjun Kuang, Weiji Zhuang, Zhaoqing Li, Zhifeng Han, Dong Zhang, Xin Zhang, Xingchen Song, Long Lin, Daniel Povey
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Computation and Language (cs.CL)
 - **Arxiv link:** https://arxiv.org/abs/2507.09318

 - **Pdf link:** https://arxiv.org/pdf/2507.09318

 - **Abstract**
 Generating spoken dialogue is more challenging than monologue text-to-speech (TTS) due to the need for realistic turn-taking and distinct speaker timbres. Existing spoken dialogue generation models, being auto-regressive, suffer from slow and unstable inference. To overcome these limitations, we introduce ZipVoice-Dialog, a non-autoregressive zero-shot spoken dialogue generation model built upon flow matching. Key designs include: 1) speaker-turn embeddings for precise speaker turn-taking; 2) a curriculum learning strategy for stable speech-text alignment; 3) specialized strategies to enable stereo dialogue generation. Additionally, recognizing the lack of open-source large-scale spoken dialogue datasets, we curated OpenDialog, a 6.8k-hour spoken dialogue dataset from in-the-wild speech data. Furthermore, we established a benchmark to comprehensively evaluate various models. Experimental results demonstrate that ZipVoice-Dialog achieves superior performance in intelligibility, speaker turn-taking accuracy, speaker similarity, and inference speed. Our codes, model checkpoints, demo samples, and the OpenDialog dataset are all publicly available at this https URL.
#### Microphone Occlusion Mitigation for Own-Voice Enhancement in Head-Worn Microphone Arrays Using Switching-Adaptive Beamforming
 - **Authors:** Wiebke Middelberg, Jung-Suk Lee, Saeed Bagheri Sereshki, Ali Aroudi, Vladimir Tourbabin, Daniel D. E. Wong
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2507.09350

 - **Pdf link:** https://arxiv.org/pdf/2507.09350

 - **Abstract**
 Enhancing the user's own-voice for head-worn microphone arrays is an important task in noisy environments to allow for easier speech communication and user-device interaction. However, a rarely addressed challenge is the change of the microphones' transfer functions when one or more of the microphones gets occluded by skin, clothes or hair. The underlying problem for beamforming-based speech enhancement is the (potentially rapidly) changing transfer functions of both the own-voice and the noise component that have to be accounted for to achieve optimal performance. In this paper, we address the problem of an occluded microphone in a head-worn microphone array. We investigate three alternative mitigation approaches by means of (i) conventional adaptive beamforming, (ii) switching between a-priori estimates of the beamformer coefficients for the occluded and unoccluded state, and (iii) a hybrid approach using a switching-adaptive beamformer. In an evaluation with real-world recordings and simulated occlusion, we demonstrate the advantages of the different approaches in terms of noise reduction, own-voice distortion and robustness against voice activity detection errors.
#### Controllable joint noise reduction and hearing loss compensation using a differentiable auditory model
 - **Authors:** Philippe Gonzalez, Torsten Dau, Tobias May
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2507.09372

 - **Pdf link:** https://arxiv.org/pdf/2507.09372

 - **Abstract**
 Deep learning-based hearing loss compensation (HLC) seeks to enhance speech intelligibility and quality for hearing impaired listeners using neural networks. One major challenge of HLC is the lack of a ground-truth target. Recent works have used neural networks to emulate non-differentiable auditory peripheral models in closed-loop frameworks, but this approach lacks flexibility. Alternatively, differentiable auditory models allow direct optimization, yet previous studies focused on individual listener profiles, or joint noise reduction (NR) and HLC without balancing each task. This work formulates NR and HLC as a multi-task learning problem, training a system to simultaneously predict denoised and compensated signals from noisy speech and audiograms using a differentiable auditory model. Results show the system achieves similar objective metric performance to systems trained for each task separately, while being able to adjust the balance between NR and HLC during inference.
#### The DKU System for Multi-Speaker Automatic Speech Recognition in MLC-SLM Challenge
 - **Authors:** Yuke Lin, Ming Cheng, Ze Li, Ming Li
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2507.09499

 - **Pdf link:** https://arxiv.org/pdf/2507.09499

 - **Abstract**
 We present the DKU system for Task 2 of the MLC-SLM Challenge, which aims to perform multi-speaker automatic speech recognition directly from raw audio without Oracle speaker labels or time boundaries. Our approach builds upon a diarization-aware framework integrating speaker embeddings and temporal utterance boundaries into a Qwen2.5-based large language model (LLM). Then, we enhance the system's multilingual performance by fine-tuning language-specific adapters and LoRA modules within the LLM decoder. Finally, our system achieves the tcpWER of 23.56\% and 18.08\% on the development and test sets of the MLC-SLM dataset, substantially outperforming the official baseline.
#### Aligning Generative Speech Enhancement with Human Preferences via Direct Preference Optimization
 - **Authors:** Haoyang Li, Nana Hou, Yuchen Hu, Jixun Yao, Sabato Marco Siniscalchi, Eng Siong Chng
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Artificial Intelligence (cs.AI); Machine Learning (cs.LG)
 - **Arxiv link:** https://arxiv.org/abs/2507.09929

 - **Pdf link:** https://arxiv.org/pdf/2507.09929

 - **Abstract**
 This work investigates speech enhancement (SE) from the perspective of language models (LMs). We propose a novel method that leverages Direct Preference Optimization (DPO) to improve the perceptual quality of enhanced speech. Using UTMOS, a neural MOS prediction model, as a proxy for human ratings, our approach guides optimization toward perceptually preferred outputs. This differs from existing LM-based SE methods that focus on maximizing the likelihood of clean speech tokens, which may misalign with human perception and degrade quality despite low prediction error. Experiments on the 2020 Deep Noise Suppression Challenge test sets demonstrate that applying DPO to a pretrained LM-based SE model yields consistent improvements across various speech quality metrics, with relative gains of up to 56%. To our knowledge, this is the first application of DPO to SE and the first to incorporate proxy perceptual feedback into LM-based SE training, pointing to a promising direction for perceptually aligned SE.
#### Cyclic Multichannel Wiener Filter for Acoustic Beamforming
 - **Authors:** Giovanni Bologni, Richard Heusdens, Richard C. Hendriks
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Signal Processing (eess.SP)
 - **Arxiv link:** https://arxiv.org/abs/2507.10159

 - **Pdf link:** https://arxiv.org/pdf/2507.10159

 - **Abstract**
 Acoustic beamforming models typically assume wide-sense stationarity of speech signals within short time frames. However, voiced speech is better modeled as a cyclostationary (CS) process, a random process whose mean and autocorrelation are $T_1$-periodic, where $\alpha_1=1/T_1$ corresponds to the fundamental frequency of vowels. Higher harmonic frequencies are found at integer multiples of the fundamental. This work introduces a cyclic multichannel Wiener filter (cMWF) for speech enhancement derived from a cyclostationary model. This beamformer exploits spectral correlation across the harmonic frequencies of the signal to further reduce the mean-squared error (MSE) between the target and the processed input. The proposed cMWF is optimal in the MSE sense and reduces to the MWF when the target is wide-sense stationary. Experiments on simulated data demonstrate considerable improvements in scale-invariant signal-to-distortion ratio (SI-SDR) on synthetic data but also indicate high sensitivity to the accuracy of the estimated fundamental frequency $\alpha_1$, which limits effectiveness on real data.
#### Harmonics to the Rescue: Why Voiced Speech is Not a Wss Process
 - **Authors:** Giovanni Bologni, Richard Heusdens, Richard C. Hendriks
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Signal Processing (eess.SP)
 - **Arxiv link:** https://arxiv.org/abs/2507.10176

 - **Pdf link:** https://arxiv.org/pdf/2507.10176

 - **Abstract**
 Speech processing algorithms often rely on statistical knowledge of the underlying process. Despite many years of research, however, the debate on the most appropriate statistical model for speech still continues. Speech is commonly modeled as a wide-sense stationary (WSS) process. However, the use of the WSS model for spectrally correlated processes is fundamentally wrong, as WSS implies spectral uncorrelation. In this paper, we demonstrate that voiced speech can be more accurately represented as a cyclostationary (CS) process. By employing the CS rather than the WSS model for processes that are inherently correlated across frequency, it is possible to improve the estimation of cross-power spectral densities (PSDs), source separation, and beamforming. We illustrate how the correlation between harmonic frequencies of CS processes can enhance system identification, and validate our findings using both simulated and real speech data.
#### Natural Language-based Assessment of L2 Oral Proficiency using LLMs
 - **Authors:** Stefano Bannò, Rao Ma, Mengjie Qian, Siyuan Tang, Kate Knill, Mark Gales
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Artificial Intelligence (cs.AI); Computation and Language (cs.CL)
 - **Arxiv link:** https://arxiv.org/abs/2507.10200

 - **Pdf link:** https://arxiv.org/pdf/2507.10200

 - **Abstract**
 Natural language-based assessment (NLA) is an approach to second language assessment that uses instructions - expressed in the form of can-do descriptors - originally intended for human examiners, aiming to determine whether large language models (LLMs) can interpret and apply them in ways comparable to human assessment. In this work, we explore the use of such descriptors with an open-source LLM, Qwen 2.5 72B, to assess responses from the publicly available S&I Corpus in a zero-shot setting. Our results show that this approach - relying solely on textual information - achieves competitive performance: while it does not outperform state-of-the-art speech LLMs fine-tuned for the task, it surpasses a BERT-based model trained specifically for this purpose. NLA proves particularly effective in mismatched task settings, is generalisable to other data types and languages, and offers greater interpretability, as it is grounded in clearly explainable, widely applicable language descriptors.
#### Less Stress, More Privacy: Stress Detection on Anonymized Speech of Air Traffic Controllers
 - **Authors:** Janaki Viswanathan, Alexander Blatt, Konrad Hagemann, Dietrich Klakow
 - **Subjects:** Subjects:
Sound (cs.SD); Computation and Language (cs.CL); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2507.08882

 - **Pdf link:** https://arxiv.org/pdf/2507.08882

 - **Abstract**
 Air traffic control (ATC) demands multi-tasking under time pressure with high consequences of an error. This can induce stress. Detecting stress is a key point in maintaining the high safety standards of ATC. However, processing ATC voice data entails privacy restrictions, e.g. the General Data Protection Regulation (GDPR) law. Anonymizing the ATC voice data is one way to comply with these restrictions. In this paper, different architectures for stress detection for anonymized ATCO speech are evaluated. Our best networks reach a stress detection accuracy of 93.6% on an anonymized version of the Speech Under Simulated and Actual Stress (SUSAS) dataset and an accuracy of 80.1% on our anonymized ATC simulation dataset. This shows that privacy does not have to be an impediment in building well-performing deep-learning-based models.
#### Mixture of LoRA Experts with Multi-Modal and Multi-Granularity LLM Generative Error Correction for Accented Speech Recognition
 - **Authors:** Bingshen Mu, Kun Wei, Pengcheng Guo, Lei Xie
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2507.09116

 - **Pdf link:** https://arxiv.org/pdf/2507.09116

 - **Abstract**
 Despite substantial improvements in ASR, performance tends to degrade when faced with adverse conditions such as speaker accents. Generative error correction (GER) leverages the rich linguistic knowledge and exceptional reasoning ability of LLMs, significantly outperforming typical LM methods. However, it lacks specificity in accented speech scenarios. In this study, we leverage GER to improve the accuracy of transcription predictions by addressing the two primary features of accented speech recognition. To fully leverage pronunciation information, we propose the multi-modal GER, which integrates pronunciation information from the speech modality, and the multi-granularity GER, which incorporates fine-grained phoneme-level information related to pronunciation. These two methods enable the LLM to utilize the pronunciation information of accented speech and the semantic information from word-level hypotheses for accurate transcription predictions through LoRA fine-tuning. On the one hand, we employ a three-stage training strategy to train separate multi-modal GER models for each accent to obtain mono-accent LoRA experts. By adopting our proposed HDMoLE method, which incorporates hierarchical routing and dynamic thresholds within the mixture of LoRA experts, we effectively merge multiple mono-accent LoRA experts within a single multi-modal GER to overcome the challenges posed by accent diversity. On the other hand, multi-granularity GER leverages the N-best word-level and phoneme-level hypotheses generated by the HDMoLE model to predict the final accented speech transcriptions. Experimental results on the multi-accent English dataset demonstrate the efficacy of our proposed methods. Our methods achieve a remarkable relative WER reduction of 67.35% compared to the Whisper-large-v3 baseline.
#### ClaritySpeech: Dementia Obfuscation in Speech
 - **Authors:** Dominika Woszczyk, Ranya Aloufi, Soteris Demetriou
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Cryptography and Security (cs.CR); Machine Learning (cs.LG); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2507.09282

 - **Pdf link:** https://arxiv.org/pdf/2507.09282

 - **Abstract**
 Dementia, a neurodegenerative disease, alters speech patterns, creating communication barriers and raising privacy concerns. Current speech technologies, such as automatic speech transcription (ASR), struggle with dementia and atypical speech, further challenging accessibility. This paper presents a novel dementia obfuscation in speech framework, ClaritySpeech, integrating ASR, text obfuscation, and zero-shot text-to-speech (TTS) to correct dementia-affected speech while preserving speaker identity in low-data environments without fine-tuning. Results show a 16% and 10% drop in mean F1 score across various adversarial settings and modalities (audio, text, fusion) for ADReSS and ADReSSo, respectively, maintaining 50% speaker similarity. We also find that our system improves WER (from 0.73 to 0.08 for ADReSS and 0.15 for ADReSSo) and speech quality from 1.65 to ~2.15, enhancing privacy and accessibility.
#### Voice Conversion for Lombard Speaking Style with Implicit and Explicit Acoustic Feature Conditioning
 - **Authors:** Dominika Woszczyk, Manuel Sam Ribeiro, Thomas Merritt, Daniel Korzekwa
 - **Subjects:** Subjects:
Sound (cs.SD); Computation and Language (cs.CL); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2507.09310

 - **Pdf link:** https://arxiv.org/pdf/2507.09310

 - **Abstract**
 Text-to-Speech (TTS) systems in Lombard speaking style can improve the overall intelligibility of speech, useful for hearing loss and noisy conditions. However, training those models requires a large amount of data and the Lombard effect is challenging to record due to speaker and noise variability and tiring recording conditions. Voice conversion (VC) has been shown to be a useful augmentation technique to train TTS systems in the absence of recorded data from the target speaker in the target speaking style. In this paper, we are concerned with Lombard speaking style transfer. Our goal is to convert speaker identity while preserving the acoustic attributes that define the Lombard speaking style. We compare voice conversion models with implicit and explicit acoustic feature conditioning. We observe that our proposed implicit conditioning strategy achieves an intelligibility gain comparable to the model conditioned on explicit acoustic features, while also preserving speaker similarity.
#### BENYO-S2ST-Corpus-1: A Bilingual English-to-Yoruba Direct Speech-to-Speech Translation Corpus
 - **Authors:** Emmanuel Adetiba, Abdultaofeek Abayomi, Raymond J. Kala, Ayodele H. Ifijeh, Oluwatobi E. Dare, Olabode Idowu-Bismark, Gabriel O. Sobola, Joy N. Adetiba, Monsurat Adepeju Lateef, Heather Cole-Lewis
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2507.09342

 - **Pdf link:** https://arxiv.org/pdf/2507.09342

 - **Abstract**
 There is a major shortage of Speech-to-Speech Translation (S2ST) datasets for high resource-to-low resource language pairs such as English-to-Yoruba. Thus, in this study, we curated the Bilingual English-to-Yoruba Speech-to-Speech Translation Corpus Version 1 (BENYO-S2ST-Corpus-1). The corpus is based on a hybrid architecture we developed for large-scale direct S2ST corpus creation at reduced cost. To achieve this, we leveraged non speech-to-speech Standard Yoruba (SY) real-time audios and transcripts in the YORULECT Corpus as well as the corresponding Standard English (SE) transcripts. YORULECT Corpus is small scale(1,504) samples, and it does not have paired English audios. Therefore, we generated the SE audios using pre-trained AI models (i.e. Facebook MMS). We also developed an audio augmentation algorithm named AcoustAug based on three latent acoustic features to generate augmented audios from the raw audios of the two languages. BENYO-S2ST-Corpus-1 has 12,032 audio samples per language, which gives a total of 24,064 sample size. The total audio duration for the two languages is 41.20 hours. This size is quite significant. Beyond building S2ST models, BENYO-S2ST-Corpus-1 can be used to build pretrained models or improve existing ones. The created corpus and Coqui framework were used to build a pretrained Yoruba TTS model (named YoruTTS-0.5) as a proof of concept. The YoruTTS-0.5 gave a F0 RMSE value of 63.54 after 1,000 epochs, which indicates moderate fundamental pitch similarity with the reference real-time audio. Ultimately, the corpus architecture in this study can be leveraged by researchers and developers to curate datasets for multilingual high-resource-to-low-resource African languages. This will bridge the huge digital divides in translations among high and low-resource language pairs. BENYO-S2ST-Corpus-1 and YoruTTS-0.5 are publicly available at (this https URL).
#### SC-TSE: Speaker Consistency-Aware Target Speaker Extraction
 - **Authors:** Shu Wu, Anbin Qi, Yanzhang Xie, Xiang Xie
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2507.09510

 - **Pdf link:** https://arxiv.org/pdf/2507.09510

 - **Abstract**
 Target Speaker Extraction (TSE) uses a reference cue to extract the target speech from a mixture. In TSE systems relying on audio cues, the speaker embedding from the enrolled speech is crucial to performance. However, these embeddings may suffer from speaker identity confusion. Unlike previous studies that focus on improving speaker embedding extraction, we improve TSE performance from the perspective of speaker consistency. In this paper, we propose a speaker consistency-aware target speaker extraction method that incorporates a centroid-based speaker consistency loss. This approach enhances TSE performance by ensuring speaker consistency between the enrolled and extracted speech. In addition, we integrate conditional loss suppression into the training process. The experimental results validate the effectiveness of our proposed methods in advancing the TSE performance. A speech demo is available online.\footnote{this https URL
#### THAI Speech Emotion Recognition (THAI-SER) corpus
 - **Authors:** Jilamika Wongpithayadisai, Chompakorn Chaksangchaichot, Soravitt Sangnark, Patawee Prakrankamanant, Krit Gangwanpongpun, Siwa Boonpunmongkol, Premmarin Milindasuta, Dangkamon Na-Pombejra, Sarana Nutanong, Ekapol Chuangsuwanich
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2507.09618

 - **Pdf link:** https://arxiv.org/pdf/2507.09618

 - **Abstract**
 We present the first sizeable corpus of Thai speech emotion recognition, THAI-SER, containing 41 hours and 36 minutes (27,854 utterances) from 100 recordings made in different recording environments: Zoom and two studio setups. The recordings contain both scripted and improvised sessions, acted by 200 professional actors (112 females and 88 males, aged 18 to 55) and were directed by professional directors. There are five primary emotions: neutral, angry, happy, sad, and frustrated, assigned to the actors when recording utterances. The utterances are annotated with an emotional category using crowdsourcing. To control the annotation process's quality, we also design an extensive filtering and quality control scheme to ensure that the majority agreement score remains above 0.71. We evaluate our annotated corpus using two metrics: inter-annotator reliability and human recognition accuracy. Inter-annotator reliability score was calculated using Krippendorff's alpha, where our corpus, after filtering, achieved an alpha score of 0.692, higher than a recommendation of 0.667. For human recognition accuracy, our corpus scored up to 0.772 post-filtering. We also provide the results of the model trained on the corpus evaluated on both in-corpus and cross-corpus setups. The corpus is publicly available under a Creative Commons BY-SA 4.0, as well as our codes for the experiments.
#### MB-RIRs: a Synthetic Room Impulse Response Dataset with Frequency-Dependent Absorption Coefficients
 - **Authors:** Enric Gusó, Joanna Luberadzka, Umut Sayin, Xavier Serra
 - **Subjects:** Subjects:
Sound (cs.SD); Machine Learning (cs.LG); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2507.09750

 - **Pdf link:** https://arxiv.org/pdf/2507.09750

 - **Abstract**
 We investigate the effects of four strategies for improving the ecological validity of synthetic room impulse response (RIR) datasets for monoaural Speech Enhancement (SE). We implement three features on top of the traditional image source method-based (ISM) shoebox RIRs: multiband absorption coefficients, source directivity and receiver directivity. We additionally consider mesh-based RIRs from the SoundSpaces dataset. We then train a DeepFilternet3 model for each RIR dataset and evaluate the performance on a test set of real RIRs both objectively and subjectively. We find that RIRs which use frequency-dependent acoustic absorption coefficients (MB-RIRs) can obtain +0.51dB of SDR and a +8.9 MUSHRA score when evaluated on real RIRs. The MB-RIRs dataset is publicly available for free download.
#### Knowing When to Quit: Probabilistic Early Exits for Speech Separation
 - **Authors:** Kenny Falkær Olsen. Mads Østergaard, Karl Ulbæk, Søren Føns Nielsen, Rasmus Malik Høegh Lindrup, Bjørn Sand Jensen, Morten Mørup
 - **Subjects:** Subjects:
Machine Learning (cs.LG); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2507.09768

 - **Pdf link:** https://arxiv.org/pdf/2507.09768

 - **Abstract**
 In recent years, deep learning-based single-channel speech separation has improved considerably, in large part driven by increasingly compute- and parameter-efficient neural network architectures. Most such architectures are, however, designed with a fixed compute and parameter budget, and consequently cannot scale to varying compute demands or resources, which limits their use in embedded and heterogeneous devices such as mobile phones and hearables. To enable such use-cases we design a neural network architecture for speech separation capable of early-exit, and we propose an uncertainty-aware probabilistic framework to jointly model the clean speech signal and error variance which we use to derive probabilistic early-exit conditions in terms of desired signal-to-noise ratios. We evaluate our methods on both speech separation and enhancement tasks, and we show that a single early-exit model can be competitive with state-of-the-art models trained at many compute and parameter budgets. Our framework enables fine-grained dynamic compute-scaling of speech separation networks while achieving state-of-the-art performance and interpretable exit conditions.
#### DualDub: Video-to-Soundtrack Generation via Joint Speech and Background Audio Synthesis
 - **Authors:** Wenjie Tian, Xinfa Zhu, Haohe Liu, Zhixian Zhao, Zihao Chen, Chaofan Ding, Xinhan Di, Junjie Zheng, Lei Xie
 - **Subjects:** Subjects:
Multimedia (cs.MM); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2507.10109

 - **Pdf link:** https://arxiv.org/pdf/2507.10109

 - **Abstract**
 While recent video-to-audio (V2A) models can generate realistic background audio from visual input, they largely overlook speech, an essential part of many video soundtracks. This paper proposes a new task, video-to-soundtrack (V2ST) generation, which aims to jointly produce synchronized background audio and speech within a unified framework. To tackle V2ST, we introduce DualDub, a unified framework built on a multimodal language model that integrates a multimodal encoder, a cross-modal aligner, and dual decoding heads for simultaneous background audio and speech generation. Specifically, our proposed cross-modal aligner employs causal and non-causal attention mechanisms to improve synchronization and acoustic harmony. Besides, to handle data scarcity, we design a curriculum learning strategy that progressively builds the multimodal capability. Finally, we introduce DualBench, the first benchmark for V2ST evaluation with a carefully curated test set and comprehensive metrics. Experimental results demonstrate that DualDub achieves state-of-the-art performance, generating high-quality and well-synchronized soundtracks with both speech and background audio.
#### DQLoRA: A Lightweight Domain-Aware Denoising ASR via Adapter-guided Distillation
 - **Authors:** Yiru Yang
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2507.10313

 - **Pdf link:** https://arxiv.org/pdf/2507.10313

 - **Abstract**
 We present a demo of DQLoRA, an Adapter-Guided Distillation framework for robust speech recognition under low-resource and noisy conditions. Our method employs a frozen Whisper model as the teacher to provide semantic supervision, and a lightweight Wav2Vec2 student equipped with QLoRA-based Adapters. Training is conducted on the FLEURS dataset augmented with DNS-style noise. The student is optimized by jointly minimizing CTC loss and KL-based distillation loss, enabling efficient adaptation while preserving recognition accuracy.
#### AudioMAE++: learning better masked audio representations with SwiGLU FFNs
 - **Authors:** Sarthak Yadav, Sergios Theodoridis, Zheng-Hua Tan
 - **Subjects:** Subjects:
Sound (cs.SD); Artificial Intelligence (cs.AI); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2507.10464

 - **Pdf link:** https://arxiv.org/pdf/2507.10464

 - **Abstract**
 Masked Autoencoders (MAEs) trained on audio spectrogram patches have emerged as a prominent approach for learning self-supervised audio representations. While several recent papers have evaluated key aspects of training MAEs on audio data, the majority of these approaches still leverage vanilla transformer building blocks, whereas the transformer community has seen steady integration of newer architectural advancements. In this work, we propose AudioMAE++, a revamped audio masked autoencoder with two such enhancements, namely macaron-style transformer blocks with gated linear units. When pretrained on the AudioSet dataset, the proposed AudioMAE++ models outperform existing MAE based approaches on 10 diverse downstream tasks, demonstrating excellent performance on audio classification and speech-based benchmarks. The proposed AudioMAE++ models also demonstrate excellent scaling characteristics, outperforming directly comparable standard MAE baselines with up to 4x more parameters.


by Zyzzyva0381 (Windy). 


2025-07-15
