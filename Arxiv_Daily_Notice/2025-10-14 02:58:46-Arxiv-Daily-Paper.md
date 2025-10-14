# Showing new listings for Monday, 13 October 2025
Auto update papers at about 2:30am UTC (10:30am Beijing time) every weekday.


阅读 `Usage.md`了解如何使用此repo实现个性化的Arxiv论文推送

See `Usage.md` for instructions on how to personalize the repo. 


Keyword list: ['text-to-speech', 'text to speech', 'tts', 'LLM-based', 'speech', 'voice']


Excluded: []


### Today: 19papers 
#### Articulation-Informed ASR: Integrating Articulatory Features into ASR via Auxiliary Speech Inversion and Cross-Attention Fusion
 - **Authors:** Ahmed Adel Attia, Jing Liu, Carol Espy Wilson
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Artificial Intelligence (cs.AI); Computation and Language (cs.CL); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2510.08585

 - **Pdf link:** https://arxiv.org/pdf/2510.08585

 - **Abstract**
 Prior works have investigated the use of articulatory features as complementary representations for automatic speech recognition (ASR), but their use was largely confined to shallow acoustic models. In this work, we revisit articulatory information in the era of deep learning and propose a framework that leverages articulatory representations both as an auxiliary task and as a pseudo-input to the recognition model. Specifically, we employ speech inversion as an auxiliary prediction task, and the predicted articulatory features are injected into the model as a query stream in a cross-attention module with acoustic embeddings as keys and values. Experiments on LibriSpeech demonstrate that our approach yields consistent improvements over strong transformer-based baselines, particularly under low-resource conditions. These findings suggest that articulatory features, once sidelined in ASR research, can provide meaningful benefits when reintroduced with modern architectures.
#### Dynamic Stress Detection: A Study of Temporal Progression Modelling of Stress in Speech
 - **Authors:** Vishakha Lall, Yisi Liu
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Artificial Intelligence (cs.AI); Computation and Language (cs.CL); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2510.08586

 - **Pdf link:** https://arxiv.org/pdf/2510.08586

 - **Abstract**
 Detecting psychological stress from speech is critical in high-pressure settings. While prior work has leveraged acoustic features for stress detection, most treat stress as a static label. In this work, we model stress as a temporally evolving phenomenon influenced by historical emotional state. We propose a dynamic labelling strategy that derives fine-grained stress annotations from emotional labels and introduce cross-attention-based sequential models, a Unidirectional LSTM and a Transformer Encoder, to capture temporal stress progression. Our approach achieves notable accuracy gains on MuSE (+5%) and StressID (+18%) over existing baselines, and generalises well to a custom real-world dataset. These results highlight the value of modelling stress as a dynamic construct in speech.
#### BaldWhisper: Faster Whisper with Head Shearing and Layer Merging
 - **Authors:** Yaya Sy, Christophe Cerisara, Irina Illina
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Artificial Intelligence (cs.AI); Computation and Language (cs.CL); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2510.08599

 - **Pdf link:** https://arxiv.org/pdf/2510.08599

 - **Abstract**
 Pruning large pre-trained transformers for low-resource languages is challenging, as it often requires massive retraining data to recover performance. For instance, Distill-Whisper prunes Whisper by 40% and retrains on 21,000 hours of speech, far beyond what is available for most languages. Can Whisper be made lighter and faster for edge devices in data-scarce settings? Focusing on Bambara with only 32h of speech-to-text data, we propose a new pruning recipe. Instead of vocabulary pruning, which is unsuitable due to frequent code-switching by Bambara speakers, we compress the embeddings with low-rank decomposition and feature distillation. Rather than removing layers, we merge them to limit performance loss. The final model preserves 90% of the original performance while being 48% smaller and 2.15x faster on a MacBook Air M1.
#### Look before Transcription: End-to-End SlideASR with Visually-Anchored Policy Optimization
 - **Authors:** Rui Hu, Delai Qiu, Yining Wang, Shengping Liu, Jitao Sang
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Computer Vision and Pattern Recognition (cs.CV); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2510.08618

 - **Pdf link:** https://arxiv.org/pdf/2510.08618

 - **Abstract**
 Automatic speech recognition (ASR) systems often struggle with domain-specific terminology, especially in specialized settings such as academic lectures. To address this, we define the SlideASR task, which leverages the rich visual information from presentation slides to improve transcription accuracy. Existing pipeline methods for this task tend to be complex and underperform. Although omni-modal large language models (OLLMs) provide a promising end-to-end framework, they frequently fail in practice by degenerating into simple optical character recognition (OCR) systems. To overcome this, we propose Visually-Anchored Policy Optimization (VAPO), a novel post-training method designed to control the model's reasoning process. Drawing on the Chain-of-Thought reasoning paradigm, VAPO enforces a structured "Look before Transcription" procedure using a <think><answer> format. Specifically, the model first performs OCR on the slide content within the think step, then generates the transcription by referencing this recognized visual information in the answer step. This reasoning process is optimized via reinforcement learning with four distinct rewards targeting format compliance, OCR accuracy, ASR quality, and visual anchoring consistency. To support further research, we construct SlideASR-Bench, a new entity-rich benchmark consisting of a synthetic dataset for training and testing, and a challenging real-world set for evaluation. Extensive experiments demonstrate that VAPO significantly improves recognition of domain-specific terms, establishing an effective end-to-end paradigm for SlideASR.
#### Impact of HRTF individualisation and head movements in a real/virtual localisation task
 - **Authors:** Vincent Martin, Lorenzo Picinali
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2510.09161

 - **Pdf link:** https://arxiv.org/pdf/2510.09161

 - **Abstract**
 The objective of Audio Augmented Reality (AAR) applications are to seamlessly integrate virtual sound sources within a real environment. It is critical for these applications that virtual sources are localised precisely at the intended position, and that the acoustic environments are accurately matched. One effective method for spatialising sound on headphones is through Head-Related Transfer Functions (HRTFs). These characterise how the physical features of a listener modify sound waves before they reach the eardrum. This study examines the influence of using individualised HRTFs on the localisation and the perceived realism of virtual sound sources associated with a real visual object. Participants were tasked with localising virtual and real speech sources presented via headphones and through a spherical loudspeaker array, respectively. The assessment focussed on perceived realism and sources location. All sources were associated with one of thirty real visual sources (loudspeakers) arranged in a semi-anechoic room. Various sound source renderings were compared, including single loudspeaker rendering and binaural rendering with individualised or non-individualised HRTFs. Additionally, the impact of head movements was explored: ten participants completed the same task with and without the possibility to move their head. The results showed that using individual HRTFs improved perceived realism but not localisation performance in the static scenario. Surprisingly, the opposite was observed when head movements were possible and encouraged.
#### Unsupervised lexicon learning from speech is limited by representations rather than clustering
 - **Authors:** Danel Adendorff, Simon Malan, Herman Kamper
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Computation and Language (cs.CL); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2510.09225

 - **Pdf link:** https://arxiv.org/pdf/2510.09225

 - **Abstract**
 Zero-resource word segmentation and clustering systems aim to tokenise speech into word-like units without access to text labels. Despite progress, the induced lexicons are still far from perfect. In an idealised setting with gold word boundaries, we ask whether performance is limited by the representation of word segments, or by the clustering methods that group them into word-like types. We combine a range of self-supervised speech features (continuous/discrete, frame/word-level) with different clustering methods (K-means, hierarchical, graph-based) on English and Mandarin data. The best system uses graph clustering with dynamic time warping on continuous features. Faster alternatives use graph clustering with cosine distance on averaged continuous features or edit distance on discrete unit sequences. Through controlled experiments that isolate either the representations or the clustering method, we demonstrate that representation variability across segments of the same word type -- rather than clustering -- is the primary factor limiting performance.
#### Effects of automotive microphone frequency response characteristics and noise conditions on speech and ASR quality -- an experimental evaluation
 - **Authors:** Michele Buccoli, Yu Du, Jacob Soendergaard, Simone Shawn Cazzaniga
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2510.09236

 - **Pdf link:** https://arxiv.org/pdf/2510.09236

 - **Abstract**
 Upon choosing microphones for automotive hands-free communication or Automatic Speech Recognition (ASR) applications, OEMs typically specify wideband, super wideband or even fullband requirements following established standard recommendations (e.g., ITU-P.1110, ITU-P.1120). In practice, it is often challenging to achieve the preferred bandwidth for an automotive microphone when considering limitations and constraints on microphone placement inside the cabin, and the automotive grade environmental robustness requirements. On the other hand, there seems to be no consensus or sufficient data on the effect of each microphone characteristic on the actual performance. As an attempt to answer this question, we used noise signals recorded in real vehicles and under various driving conditions to experimentally study the relationship between the microphones' characteristics and the final audio quality of speech communication and performance of ASR engines. We focus on how variations in microphone bandwidth and amplitude frequency response shapes affect the perceptual speech quality. The speech quality results are compared by using ETSI TS 103 281 metrics (S-MOS, N-MOS, G-MOS) and ancillary metrics such as SNR. The ASR results are evaluated with standard metrics such as Word Error Rate (WER). Findings from this study provide knowledge in the understanding of what microphone frequency response characteristics are more relevant for audio quality and choice of proper microphone specifications, particularly for automotive applications.
#### Target speaker anonymization in multi-speaker recordings
 - **Authors:** Natalia Tomashenko, Junichi Yamagishi, Xin Wang, Yun Liu, Emmanuel Vincent
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Computation and Language (cs.CL); Cryptography and Security (cs.CR)
 - **Arxiv link:** https://arxiv.org/abs/2510.09307

 - **Pdf link:** https://arxiv.org/pdf/2510.09307

 - **Abstract**
 Most of the existing speaker anonymization research has focused on single-speaker audio, leading to the development of techniques and evaluation metrics optimized for such condition. This study addresses the significant challenge of speaker anonymization within multi-speaker conversational audio, specifically when only a single target speaker needs to be anonymized. This scenario is highly relevant in contexts like call centers, where customer privacy necessitates anonymizing only the customer's voice in interactions with operators. Conventional anonymization methods are often not suitable for this task. Moreover, current evaluation methodology does not allow us to accurately assess privacy protection and utility in this complex multi-speaker scenario. This work aims to bridge these gaps by exploring effective strategies for targeted speaker anonymization in conversational audio, highlighting potential problems in their development and proposing corresponding improved evaluation methodologies.
#### A Study of the Removability of Speaker-Adversarial Perturbations
 - **Authors:** Liping Chen, Chenyang Guo, Kong Aik Lee, Zhen-Hua Ling, Wu Guo
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2510.09504

 - **Pdf link:** https://arxiv.org/pdf/2510.09504

 - **Abstract**
 Recent advancements in adversarial attacks have demonstrated their effectiveness in misleading speaker recognition models, making wrong predictions about speaker identities. On the other hand, defense techniques against speaker-adversarial attacks focus on reducing the effects of speaker-adversarial perturbations on speaker attribute extraction. These techniques do not seek to fully remove the perturbations and restore the original speech. To this end, this paper studies the removability of speaker-adversarial perturbations. Specifically, the investigation is conducted assuming various degrees of awareness of the perturbation generator across three scenarios: ignorant, semi-informed, and well-informed. Besides, we consider both the optimization-based and feedforward perturbation generation methods. Experiments conducted on the LibriSpeech dataset demonstrated that: 1) in the ignorant scenario, speaker-adversarial perturbations cannot be eliminated, although their impact on speaker attribute extraction is reduced, 2) in the semi-informed scenario, the speaker-adversarial perturbations cannot be fully removed, while those generated by the feedforward model can be considerably reduced, and 3) in the well-informed scenario, speaker-adversarial perturbations are nearly eliminated, allowing for the restoration of the original speech. Audio samples can be found in this https URL.
#### Evaluating Hallucinations in Multimodal LLMs with Spoken Queries under Diverse Acoustic Conditions
 - **Authors:** Hansol Park, Hoseong Ahn, Junwon Moon, Yejin Lee, Kyuhong Shim
 - **Subjects:** Subjects:
Sound (cs.SD); Artificial Intelligence (cs.AI); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2510.08581

 - **Pdf link:** https://arxiv.org/pdf/2510.08581

 - **Abstract**
 Hallucinations in vision-language models have been extensively studied using benchmarks that probe reliability in image-text settings. In contrast, the effect of spoken queries on multimodal hallucinations remains largely unexplored, despite the growing role of voice-driven interfaces. In this work, we investigate how spoken input influences hallucinations in multimodal large language models. We present RePOPE-Spk, an audio-augmented extension of the RePOPE benchmark, where queries are provided as speech under diverse acoustic conditions. Using RePOPE-Spk, we systematically evaluate both proprietary and open-source models. Experimental results show that hallucinations escalate when queries are spoken rather than written: error rates increase by 3% under clean speech and by up to 20% with environmental noise. Input order and query length further affect robustness, while strategies such as many-shot prompting and chain-of-thought reasoning offer partial but insufficient mitigation. These findings highlight a critical and underexplored challenge, opening new directions for building reliable voice interface systems.
#### Hierarchical Self-Supervised Representation Learning for Depression Detection from Speech
 - **Authors:** Yuxin Li, Eng Siong Chng, Cuntai Guan
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Artificial Intelligence (cs.AI); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2510.08593

 - **Pdf link:** https://arxiv.org/pdf/2510.08593

 - **Abstract**
 Speech-based depression detection (SDD) is a promising, non-invasive alternative to traditional clinical assessments. However, it remains limited by the difficulty of extracting meaningful features and capturing sparse, heterogeneous depressive cues over time. Pretrained self-supervised learning (SSL) models such as WavLM provide rich, multi-layer speech representations, yet most existing SDD methods rely only on the final layer or search for a single best-performing one. These approaches often overfit to specific datasets and fail to leverage the full hierarchical structure needed to detect subtle and persistent depression signals. To address this challenge, we propose HAREN-CTC, a novel architecture that integrates multi-layer SSL features using cross-attention within a multitask learning framework, combined with Connectionist Temporal Classification loss to handle sparse temporal supervision. HAREN-CTC comprises two key modules: a Hierarchical Adaptive Clustering module that reorganizes SSL features into complementary embeddings, and a Cross-Modal Fusion module that models inter-layer dependencies through cross-attention. The CTC objective enables alignment-aware training, allowing the model to track irregular temporal patterns of depressive speech cues. We evaluate HAREN-CTC under both an upper-bound setting with standard data splits and a generalization setting using five-fold cross-validation. The model achieves state-of-the-art macro F1-scores of 0.81 on DAIC-WOZ and 0.82 on MODMA, outperforming prior methods across both evaluation scenarios.
#### ControlAudio: Tackling Text-Guided, Timing-Indicated and Intelligible Audio Generation via Progressive Diffusion Modeling
 - **Authors:** Yuxuan Jiang, Zehua Chen, Zeqian Ju, Yusheng Dai, Weibei Dou, Jun Zhu
 - **Subjects:** Subjects:
Sound (cs.SD); Artificial Intelligence (cs.AI); Computation and Language (cs.CL); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2510.08878

 - **Pdf link:** https://arxiv.org/pdf/2510.08878

 - **Abstract**
 Text-to-audio (TTA) generation with fine-grained control signals, e.g., precise timing control or intelligible speech content, has been explored in recent works. However, constrained by data scarcity, their generation performance at scale is still compromised. In this study, we recast controllable TTA generation as a multi-task learning problem and introduce a progressive diffusion modeling approach, ControlAudio. Our method adeptly fits distributions conditioned on more fine-grained information, including text, timing, and phoneme features, through a step-by-step strategy. First, we propose a data construction method spanning both annotation and simulation, augmenting condition information in the sequence of text, timing, and phoneme. Second, at the model training stage, we pretrain a diffusion transformer (DiT) on large-scale text-audio pairs, achieving scalable TTA generation, and then incrementally integrate the timing and phoneme features with unified semantic representations, expanding controllability. Finally, at the inference stage, we propose progressively guided generation, which sequentially emphasizes more fine-grained information, aligning inherently with the coarse-to-fine sampling nature of DiT. Extensive experiments show that ControlAudio achieves state-of-the-art performance in terms of temporal accuracy and speech clarity, significantly outperforming existing methods on both objective and subjective evaluations. Demo samples are available at: this https URL.
#### VM-UNSSOR: Unsupervised Neural Speech Separation Enhanced by Higher-SNR Virtual Microphone Arrays
 - **Authors:** Shulin He, Zhong-Qiu Wang
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2510.08914

 - **Pdf link:** https://arxiv.org/pdf/2510.08914

 - **Abstract**
 Blind speech separation (BSS) aims to recover multiple speech sources from multi-channel, multi-speaker mixtures under unknown array geometry and room impulse responses. In unsupervised setup where clean target speech is not available for model training, UNSSOR proposes a mixture consistency (MC) loss for training deep neural networks (DNN) on over-determined training mixtures to realize unsupervised speech separation. However, when the number of microphones of the training mixtures decreases, the MC constraint weakens and the separation performance falls dramatically. To address this, we propose VM-UNSSOR, augmenting the observed training mixture signals recorded by a limited number of microphones with several higher-SNR virtual-microphone (VM) signals, which are obtained by applying linear spatial demixers (such as IVA and spatial clustering) to the observed training mixtures. As linear projections of the observed mixtures, the virtual-microphone signals can typically increase the SNR of each source and can be leveraged to compute extra MC losses to improve UNSSOR and address the frequency permutation problem in UNSSOR. On the SMS-WSJ dataset, in the over-determined six-microphone, two-speaker separation setup, VM-UNSSOR reaches 17.1 dB SI-SDR, while UNSSOR only obtains 14.7 dB; and in the determined two-microphone, two-speaker case, UNSSOR collapses to -2.7 dB SI-SDR, while VM-UNSSOR achieves 10.7 dB.
#### DiTSinger: Scaling Singing Voice Synthesis with Diffusion Transformer and Implicit Alignment
 - **Authors:** Zongcai Du, Guilin Deng, Xiaofeng Guo, Xin Gao, Linke Li, Kaichang Cheng, Fubo Han, Siyu Yang, Peng Liu, Pan Zhong, Qiang Fu
 - **Subjects:** Subjects:
Sound (cs.SD); Artificial Intelligence (cs.AI); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2510.09016

 - **Pdf link:** https://arxiv.org/pdf/2510.09016

 - **Abstract**
 Recent progress in diffusion-based Singing Voice Synthesis (SVS) demonstrates strong expressiveness but remains limited by data scarcity and model scalability. We introduce a two-stage pipeline: a compact seed set of human-sung recordings is constructed by pairing fixed melodies with diverse LLM-generated lyrics, and melody-specific models are trained to synthesize over 500 hours of high-quality Chinese singing data. Building on this corpus, we propose DiTSinger, a Diffusion Transformer with RoPE and qk-norm, systematically scaled in depth, width, and resolution for enhanced fidelity. Furthermore, we design an implicit alignment mechanism that obviates phoneme-level duration labels by constraining phoneme-to-acoustic attention within character-level spans, thereby improving robustness under noisy or uncertain alignments. Extensive experiments validate that our approach enables scalable, alignment-free, and high-fidelity SVS.
#### Déréverbération non-supervisée de la parole par modèle hybride
 - **Authors:** Louis Bahrman (IDS, S2A), Mathieu Fontaine (IDS, S2A), Gaël Richard (IDS, S2A)
 - **Subjects:** Subjects:
Sound (cs.SD); Artificial Intelligence (cs.AI); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2510.09025

 - **Pdf link:** https://arxiv.org/pdf/2510.09025

 - **Abstract**
 This paper introduces a new training strategy to improve speech dereverberation systems in an unsupervised manner using only reverberant speech. Most existing algorithms rely on paired dry/reverberant data, which is difficult to obtain. Our approach uses limited acoustic information, like the reverberation time (RT60), to train a dereverberation system. Experimental results demonstrate that our method achieves more consistent performance across various objective metrics than the state-of-the-art.
#### O_O-VC: Synthetic Data-Driven One-to-One Alignment for Any-to-Any Voice Conversion
 - **Authors:** Huu Tuong Tu, Huan Vu, cuong tien nguyen, Dien Hy Ngo, Nguyen Thi Thu Trang
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2510.09061

 - **Pdf link:** https://arxiv.org/pdf/2510.09061

 - **Abstract**
 Traditional voice conversion (VC) methods typically attempt to separate speaker identity and linguistic information into distinct representations, which are then combined to reconstruct the audio. However, effectively disentangling these factors remains challenging, often leading to information loss during training. In this paper, we propose a new approach that leverages synthetic speech data generated by a high-quality, pretrained multispeaker text-to-speech (TTS) model. Specifically, synthetic data pairs that share the same linguistic content but differ in speaker identity are used as input-output pairs to train the voice conversion model. This enables the model to learn a direct mapping between source and target voices, effectively capturing speaker-specific characteristics while preserving linguistic content. Additionally, we introduce a flexible training strategy for any-to-any voice conversion that generalizes well to unseen speakers and new languages, enhancing adaptability and performance in zero-shot scenarios. Our experiments show that our proposed method achieves a 16.35% relative reduction in word error rate and a 5.91% improvement in speaker cosine similarity, outperforming several state-of-the-art methods. Voice conversion samples can be accessed at: this https URL
#### FLToP CTC: Frame-Level Token Pruning via Relative Threshold for Efficient and Memory-Saving Decoding on Diverse Platforms
 - **Authors:** Atul Shree, Harshith Jupuru
 - **Subjects:** Subjects:
Machine Learning (cs.LG); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2510.09085

 - **Pdf link:** https://arxiv.org/pdf/2510.09085

 - **Abstract**
 CTC-based ASR systems face computational and memory bottlenecks in resource-limited environments. Traditional CTC decoders, requiring up to 90% of processing time in systems (e.g., wav2vec2-large on L4 GPUs), face inefficiencies due to exhaustive token-level operations. This paper introduces Frame Level Token Pruning for Connectionist Temporal Classification (FLToP CTC), a novel decoding algorithm that employs frame-level token pruning guided by a relative threshold probability. By dynamically eliminating low-probability tokens per frame, FLToP CTC reduces compute and memory demands while maintaining negligible WER degradation. On LibriSpeech, FLToP CTC achieves a 10.5x runtime speedup and 2.78x memory reduction versus standard CTC decoders. Its simplicity enables seamless integration into CTC decoders across platforms (CPUs, GPUs, etc.). FLToP CTC addresses CTC bottlenecks, offering scalability for resource-limited environments and realtime applications, enhancing speech recognition accessibility and efficiency.
#### The Speech-LLM Takes It All: A Truly Fully End-to-End Spoken Dialogue State Tracking Approach
 - **Authors:** Nizar El Ghazal, Antoine Caubrière, Valentin Vielzeuf
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Artificial Intelligence (cs.AI); Machine Learning (cs.LG); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2510.09424

 - **Pdf link:** https://arxiv.org/pdf/2510.09424

 - **Abstract**
 This paper presents a comparative study of context management strategies for end-to-end Spoken Dialog State Tracking using Speech-LLMs. We systematically evaluate traditional multimodal context (combining text history and spoken current turn), full spoken history, and compressed spoken history approaches. Our experiments on the SpokenWOZ corpus demonstrate that providing the full spoken conversation as input yields the highest performance among models of similar size, significantly surpassing prior methods. Furthermore, we show that attention-pooling-based compression of the spoken history offers a strong trade-off, maintaining competitive accuracy with reduced context size. Detailed analysis confirms that improvements stem from more effective context utilization.
#### Accent-Invariant Automatic Speech Recognition via Saliency-Driven Spectrogram Masking
 - **Authors:** Mohammad Hossein Sameti, Sepehr Harfi Moridani, Ali Zarean, Hossein Sameti
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2510.09528

 - **Pdf link:** https://arxiv.org/pdf/2510.09528

 - **Abstract**
 Pre-trained transformer-based models have significantly advanced automatic speech recognition (ASR), yet they remain sensitive to accent and dialectal variations, resulting in elevated word error rates (WER) in linguistically diverse languages such as English and Persian. To address this challenge, we propose an accent-invariant ASR framework that integrates accent and dialect classification into the recognition pipeline. Our approach involves training a spectrogram-based classifier to capture accent-specific cues, masking the regions most influential to its predictions, and using the masked spectrograms for data augmentation. This enhances the robustness of ASR models against accent variability. We evaluate the method using both English and Persian speech. For Persian, we introduce a newly collected dataset spanning multiple regional accents, establishing the first systematic benchmark for accent variation in Persian ASR that fills a critical gap in multilingual speech research and provides a foundation for future studies on low-resource, linguistically diverse languages. Experimental results with the Whisper model demonstrate that our masking and augmentation strategy yields substantial WER reductions in both English and Persian settings, confirming the effectiveness of the approach. This research advances the development of multilingual ASR systems that are resilient to accent and dialect diversity. Code and dataset are publicly available at: this https URL


by Zyzzyva0381 (Windy). 


2025-10-14
