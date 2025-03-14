# Showing new listings for Friday, 14 March 2025
Auto update papers at about 2:30am UTC (10:30am Beijing time) every weekday.


阅读 `Usage.md`了解如何使用此repo实现个性化的Arxiv论文推送

See `Usage.md` for instructions on how to personalize the repo. 


Keyword list: ['acoustic', 'reinforcement learning', 'reverb', 'meta', 'separate', 'reconstruction', 'noise', 'enhance', 'localization', 'speech']


Excluded: []


### Today: 8papers 
#### ValSub: Subsampling Validation Data to Mitigate Forgetting during ASR Personalization
 - **Authors:** Haaris Mehmood, Karthikeyan Saravanan, Pablo Peso Parada, David Tuckey, Mete Ozay, Gil Ho Lee, Jungin Lee, Seokyeong Jung
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2503.09906

 - **Pdf link:** https://arxiv.org/pdf/2503.09906

 - **Abstract**
 Automatic Speech Recognition (ASR) is widely used within consumer devices such as mobile phones. Recently, personalization or on-device model fine-tuning has shown that adaptation of ASR models towards target user speech improves their performance over rare words or accented speech. Despite these gains, fine-tuning on user data (target domain) risks the personalized model to forget knowledge about its original training distribution (source domain) i.e. catastrophic forgetting, leading to subpar general ASR performance. A simple and efficient approach to combat catastrophic forgetting is to measure forgetting via a validation set that represents the source domain distribution. However, such validation sets are large and impractical for mobile devices. Towards this, we propose a novel method to subsample a substantially large validation set into a smaller one while maintaining the ability to estimate forgetting. We demonstrate the efficacy of such a dataset in mitigating forgetting by utilizing it to dynamically determine the number of ideal fine-tuning epochs. When measuring the deviations in per user fine-tuning epochs against a 50x larger validation set (oracle), our method achieves a lower mean-absolute-error (3.39) compared to randomly selected subsets of the same size (3.78-8.65). Unlike random baselines, our method consistently tracks the oracle's behaviour across three different forgetting thresholds.
#### Sound Field Estimation: Theories and Applications
 - **Authors:** Natsuki Ueno, Shoichi Koyama
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2503.10016

 - **Pdf link:** https://arxiv.org/pdf/2503.10016

 - **Abstract**
 The spatial information of sound plays a crucial role in various situations, ranging from daily activities to advanced engineering technologies. To fully utilize its potential, numerous research studies on spatial audio signal processing have been carried out in the literature. Sound field estimation is one of the key foundational technologies that can be applied to a wide range of acoustic signal processing techniques, including sound field reproduction using loudspeakers and binaural playback through headphones. The purpose of this paper is to present an overview of sound field estimation methods. After providing the necessary mathematical background, two different approaches to sound field estimation will be explained. This paper focuses on clarifying the essential theories of each approach, while also referencing state-of-the-art developments. Finally, several acoustic signal processing technologies will be discussed as examples of the application of sound field estimation.
#### Bilingual Dual-Head Deep Model for Parkinson's Disease Detection from Speech
 - **Authors:** Moreno La Quatra, Juan Rafael Orozco-Arroyave, Marco Sabato Siniscalchi
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Artificial Intelligence (cs.AI)
 - **Arxiv link:** https://arxiv.org/abs/2503.10301

 - **Pdf link:** https://arxiv.org/pdf/2503.10301

 - **Abstract**
 This work aims to tackle the Parkinson's disease (PD) detection problem from the speech signal in a bilingual setting by proposing an ad-hoc dual-head deep neural architecture for type-based binary classification. One head is specialized for diadochokinetic patterns. The other head looks for natural speech patterns present in continuous spoken utterances. Only one of the two heads is operative accordingly to the nature of the input. Speech representations are extracted from self-supervised learning (SSL) models and wavelet transforms. Adaptive layers, convolutional bottlenecks, and contrastive learning are exploited to reduce variations across languages. Our solution is assessed against two distinct datasets, EWA-DB, and PC-GITA, which cover Slovak and Spanish languages, respectively. Results indicate that conventional models trained on a single language dataset struggle with cross-linguistic generalization, and naive combinations of datasets are suboptimal. In contrast, our model improves generalization on both languages, simultaneously.
#### Handling Domain Shifts for Anomalous Sound Detection: A Review of DCASE-Related Work
 - **Authors:** Kevin Wilkinghoff, Takuya Fujimura, Keisuke Imoto, Jonathan Le Roux, Zheng-Hua Tan, Tomoki Toda
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2503.10435

 - **Pdf link:** https://arxiv.org/pdf/2503.10435

 - **Abstract**
 When detecting anomalous sounds in complex environments, one of the main difficulties is that trained models must be sensitive to subtle differences in monitored target signals, while many practical applications also require them to be insensitive to changes in acoustic domains. Examples of such domain shifts include changing the type of microphone or the location of acoustic sensors, which can have a much stronger impact on the acoustic signal than subtle anomalies themselves. Moreover, users typically aim to train a model only on source domain data, which they may have a relatively large collection of, and they hope that such a trained model will be able to generalize well to an unseen target domain by providing only a minimal number of samples to characterize the acoustic signals in that domain. In this work, we review and discuss recent publications focusing on this domain generalization problem for anomalous sound detection in the context of the DCASE challenges on acoustic machine condition monitoring.
#### Quantization for OpenAI's Whisper Models: A Comparative Analysis
 - **Authors:** Allison Andreyev
 - **Subjects:** Subjects:
Sound (cs.SD); Computation and Language (cs.CL); Machine Learning (cs.LG); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2503.09905

 - **Pdf link:** https://arxiv.org/pdf/2503.09905

 - **Abstract**
 Automated speech recognition (ASR) models have gained prominence for applications such as captioning, speech translation, and live transcription. This paper studies Whisper and two model variants: one optimized for live speech streaming and another for offline transcription. Notably, these models have been found to generate hallucinated content, reducing transcription reliability. Furthermore, larger model variants exhibit increased latency and pose challenges for deployment on resource-constrained devices. This study analyzes the similarities and differences between three Whisper models, qualitatively examining their distinct capabilities. Next, this study quantifies the impact of model quantization on latency and evaluates its viability for edge deployment. Using the open source LibriSpeech dataset, this paper evaluates the word error rate (WER) along with latency analysis of whispercpp using 3 quantization methods (INT4, INT5, INT8). Results show that quantization reduces latency by 19\% and model size by 45\%, while preserving transcription accuracy. These findings provide insights into the optimal use cases of different Whisper models and edge device deployment possibilities. All code, datasets, and implementation details are available in a public GitHub repository: this https URL
#### Adaptive Inner Speech-Text Alignment for LLM-based Speech Translation
 - **Authors:** Henglyu Liu, Andong Chen, Kehai Chen, Xuefeng Bai, Meizhi Zhong, Yuan Qiu, Min Zhang
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2503.10211

 - **Pdf link:** https://arxiv.org/pdf/2503.10211

 - **Abstract**
 Recent advancement of large language models (LLMs) has led to significant breakthroughs across various tasks, laying the foundation for the development of LLM-based speech translation systems. Existing methods primarily focus on aligning inputs and outputs across modalities while overlooking deeper semantic alignment within model representations. To address this limitation, we propose an Adaptive Inner Speech-Text Alignment (AI-STA) method to bridge the modality gap by explicitly aligning speech and text representations at selected layers within LLMs. To achieve this, we leverage the optimal transport (OT) theory to quantify fine-grained representation discrepancies between speech and text. Furthermore, we utilize the cross-modal retrieval technique to identify the layers that are best suited for alignment and perform joint training on these layers. Experimental results on speech translation (ST) tasks demonstrate that AI-STA significantly improves the translation performance of large speech-text models (LSMs), outperforming previous state-of-the-art approaches. Our findings highlight the importance of inner-layer speech-text alignment in LLMs and provide new insights into enhancing cross-modal learning.
#### MACS: Multi-source Audio-to-image Generation with Contextual Significance and Semantic Alignment
 - **Authors:** Hao Zhou, Xiaobao Guo, Yuzhe Zhu, Adams Wai-Kin Kong
 - **Subjects:** Subjects:
Sound (cs.SD); Computer Vision and Pattern Recognition (cs.CV); Graphics (cs.GR); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2503.10287

 - **Pdf link:** https://arxiv.org/pdf/2503.10287

 - **Abstract**
 Propelled by the breakthrough in deep generative models, audio-to-image generation has emerged as a pivotal cross-model task that converts complex auditory signals into rich visual representations. However, previous works only focus on single-source audio inputs for image generation, ignoring the multi-source characteristic in natural auditory scenes, thus limiting the performance in generating comprehensive visual content. To bridge this gap, a method called MACS is proposed to conduct multi-source audio-to-image generation. This is the first work that explicitly separates multi-source audio to capture the rich audio components before image generation. MACS is a two-stage method. In the first stage, multi-source audio inputs are separated by a weakly supervised method, where the audio and text labels are semantically aligned by casting into a common space using the large pre-trained CLAP model. We introduce a ranking loss to consider the contextual significance of the separated audio signals. In the second stage, efficient image generation is achieved by mapping the separated audio signals to the generation condition using only a trainable adapter and a MLP layer. We preprocess the LLP dataset as the first full multi-source audio-to-image generation benchmark. The experiments are conducted on multi-source, mixed-source, and single-source audio-to-image generation tasks. The proposed MACS outperforms the current state-of-the-art methods in 17 of the 21 evaluation indexes on all tasks and delivers superior visual quality. The code will be publicly available.
#### Whisper Speaker Identification: Leveraging Pre-Trained Multilingual Transformers for Robust Speaker Embeddings
 - **Authors:** Jakaria Islam Emon, Md Abu Salek, Kazi Tamanna Alam
 - **Subjects:** Subjects:
Sound (cs.SD); Artificial Intelligence (cs.AI); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2503.10446

 - **Pdf link:** https://arxiv.org/pdf/2503.10446

 - **Abstract**
 Speaker identification in multilingual settings presents unique challenges, particularly when conventional models are predominantly trained on English data. In this paper, we propose WSI (Whisper Speaker Identification), a framework that repurposes the encoder of the Whisper automatic speech recognition model pre trained on extensive multilingual data to generate robust speaker embeddings via a joint loss optimization strategy that leverages online hard triplet mining and self supervised Normalized Temperature-scaled Cross Entropy loss. By capitalizing on Whisper language-agnostic acoustic representations, our approach effectively distinguishes speakers across diverse languages and recording conditions. Extensive evaluations on multiple corpora, including VoxTube (multilingual), JVS (Japanese), CallHome (German, Spanish, Chinese, and Japanese), and Voxconverse (English), demonstrate that WSI consistently outperforms state-of-the-art baselines, namely Pyannote Embedding, ECAPA TDNN, and Xvector, in terms of lower equal error rates and higher AUC scores. These results validate our hypothesis that a multilingual pre-trained ASR encoder, combined with joint loss optimization, substantially improves speaker identification performance in non-English languages.


by Zyzzyva0381 (Windy). 


2025-03-14
