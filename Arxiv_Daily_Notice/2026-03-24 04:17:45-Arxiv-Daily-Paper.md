# Showing new listings for Tuesday, 24 March 2026
Auto update papers at about 2:30am UTC (10:30am Beijing time) every weekday.


阅读 `Usage.md`了解如何使用此repo实现个性化的Arxiv论文推送

See `Usage.md` for instructions on how to personalize the repo. 


Keyword list: ['text-to-speech', 'text to speech', 'tts', 'LLM-based', 'speech', 'voice']


Excluded: []


### Today: 9papers 
#### OmniCodec: Low Frame Rate Universal Audio Codec with Semantic-Acoustic Disentanglement
 - **Authors:** Jingbin Hu, Haoyu Zhang, Dake Guo, Qirui Zhan, Wenhao Li, Huakang Chen, Guobin Ma, Hanke Xie, Chengyou Wang, Pengyuan Xie, Chuan Xie, Qiang Zhang, Lei Xie
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2603.20638

 - **Pdf link:** https://arxiv.org/pdf/2603.20638

 - **Abstract**
 Large Language Models (LLMs) have advanced audio generation through discrete representation learning. However, most existing neural codecs focus on speech and emphasize reconstruction fidelity, overlooking unified low frame rate modeling across diverse audio domains, including speech, music, and general sound. Moreover, high reconstruction quality does not necessarily yield semantically informative representations, limiting effectiveness in downstream generation tasks. We propose OmniCodec, a universal neural audio codec tailored for low frame rate. It adopts a hierarchical multi-codebook design with semantic-acoustic decoupling by leveraging the audio encoder of the pre-trained understanding model, along with a self-guidance strategy to improve codebook utilization and reconstruction. Compared with the Mimi codec, experiments show that OmniCodec achieves outstanding performance at the same bitrate, delivering superior reconstruction quality while also providing more semantically informative representations that benefit downstream generation tasks. Our model and code will be open-sourced. Our demo page is available.
#### DiT-Flow: Speech Enhancement Robust to Multiple Distortions based on Flow Matching in Latent Space and Diffusion Transformers
 - **Authors:** Tianyu Cao, Helin Wang, Ari Frummer, Yuval Sieradzki, Adi Arbel, Laureano Moro Velazquez, Jesus Villalba, Oren Gal, Thomas Thebaud, Najim Dehak
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Artificial Intelligence (cs.AI); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2603.21608

 - **Pdf link:** https://arxiv.org/pdf/2603.21608

 - **Abstract**
 Recent advances in generative models, such as diffusion and flow matching, have shown strong performance in audio tasks. However, speech enhancement (SE) models are typically trained on limited datasets and evaluated under narrow conditions, limiting real-world applicability. To address this, we propose DiT-Flow, a flow matching-based SE framework built on the latent Diffusion Transformer (DiT) backbone and trained for robustness across diverse distortions, including noise, reverberation, and compression. DiT-Flow operates on compact variational auto-encoders (VAEs)-derived latent features. We validated our approach on StillSonicSet, a synthetic yet acoustically realistic dataset composed of LibriSpeech, FSD50K, FMA, and 90 Matterport3D scenes. Experiments show that DiT-Flow consistently outperforms state-of-the-art generative SE models, demonstrating the effectiveness of flow matching in multi-condition speech enhancement. Despite ongoing efforts to expand synthetic data realism, a persistent bottleneck in SE is the inevitable mismatch between training and deployment conditions. By integrating LoRA with the MoE framework, we achieve both parameter-efficient and high-performance training for DiT-Flow robust to multiple distortions with using 4.9% percentage of the total parameters to obtain a better performance on five unseen distortions.
#### Disentangling Speaker Traits for Deepfake Source Verification via Chebyshev Polynomial and Riemannian Metric Learning
 - **Authors:** Xi Xuan, Wenxin Zhang, Zhiyu Li, Jennifer Williams, Ville Hautamäki, Tomi H. Kinnunen
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Computation and Language (cs.CL); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2603.21875

 - **Pdf link:** https://arxiv.org/pdf/2603.21875

 - **Abstract**
 Speech deepfake source verification systems aims to determine whether two synthetic speech utterances originate from the same source generator, often assuming that the resulting source embeddings are independent of speaker traits. However, this assumption remains unverified. In this paper, we first investigate the impact of speaker factors on source verification. We propose a speaker-disentangled metric learning (SDML) framework incorporating two novel loss functions. The first leverages Chebyshev polynomial to mitigate gradient instability during disentanglement optimization. The second projects source and speaker embeddings into hyperbolic space, leveraging Riemannian metric distances to reduce speaker information and learn more discriminative source features. Experimental results on MLAAD benchmark, evaluated under four newly proposed protocols designed for source-speaker disentanglement scenarios, demonstrate the effectiveness of SDML framework. The code, evaluation protocols and demo website are available at this https URL.
#### Adaptive Federated Fine-Tuning of Self-Supervised Speech Representations
 - **Authors:** Xin Guo, Chunrui Zhao, Hong Jia, Ting Dang, Gongping Huang, Xianrui Zheng, Yan Gao
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2603.21888

 - **Pdf link:** https://arxiv.org/pdf/2603.21888

 - **Abstract**
 Integrating Federated Learning (FL) with self-supervised learning (SSL) enables privacy-preserving fine-tuning for speech tasks. However, federated environments exhibit significant heterogeneity: clients differ in computational capacity, causing straggler effects under unified fine-tuning, while diverse downstream tasks require different representation depths, making full-model updates inefficient. To address these challenges, we propose an adaptive federated fine-tuning framework with early exits. Lightweight prediction heads are inserted at intermediate layers of the SSL backbone, allowing clients to terminate computation based on local constraints and task requirements. We further introduce a layer-wise, depth-aware partial aggregation strategy to better utilize representations from different network depths. Experiments show that the framework reduces edge overhead, supports heterogeneous hardware, and maintains competitive performance in resource-constrained federated environments.
#### SelfTTS: cross-speaker style transfer through explicit embedding disentanglement and self-refinement using self-augmentation
 - **Authors:** Lucas H. Ueda, João G. T. Lima, Pedro R. Corrêa, Flávio O. Simões, Mário U. Neto, Paula D. P. Costa
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2603.22252

 - **Pdf link:** https://arxiv.org/pdf/2603.22252

 - **Abstract**
 This paper presents SelfTTS, a text-to-speech (TTS) model designed for cross-speaker style transfer that eliminates the need for external pre-trained speaker or emotion encoders. The architecture achieves emotional expressivity in neutral speakers through an explicit disentanglement strategy utilizing Gradient Reversal Layers (GRL) combined with cosine similarity loss to decouple speaker and emotion information. We introduce Multi Positive Contrastive Learning (MPCL) to induce clustered representations of speaker and emotion embeddings based on their respective labels. Furthermore, SelfTTS employs a self-refinement strategy via Self-Augmentation, exploiting the model's voice conversion capabilities to enhance the naturalness of synthesized speech. Experimental results demonstrate that SelfTTS achieves superior emotional naturalness (eMOS) and robust stability in target timbre and emotion compared to state-of-the-art baselines.
#### LL-SDR: Low-Latency Speech enhancement through Discrete Representations
 - **Authors:** Jingyi Li, Luca Della Libera, Mirco Ravanelli, Cem Subakan
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2603.20242

 - **Pdf link:** https://arxiv.org/pdf/2603.20242

 - **Abstract**
 Many speech enhancement (SE) methods rely on continuous representations. Recently, discrete audio tokens have been explored to enable autoregressive generation for SE. However, it remains unclear whether discretization itself consistently improves SE performance. In this paper, we introduce LL-SDR, a token-based speech enhancement framework that explicitly leverages discretization to better separate speech and noise. Our first contribution is a Variance-Ordered Residual Vector Quantizer (VO-RVQ), designed to disentangle speech and noise distributions during tokenization. Second, we propose a latent-space discriminator to better align enhanced embeddings with semantic embeddings. Experiments show that LL-SDR outperforms continuous baselines and matches the performance of autoregressive token-based approaches, while enabling lightweight, low-latency speech enhancement in both reverberant and non-reverberant noisy environments. Demos and source code are available at our project websites.
#### Abjad-Kids: An Arabic Speech Classification Dataset for Primary Education
 - **Authors:** Abdul Aziz Snoubara, Baraa Al_Maradni, Haya Al_Naal, Malek Al_Madrmani, Roaa Jdini, Seedra Zarzour, Khloud Al Jallad
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Human-Computer Interaction (cs.HC); Machine Learning (cs.LG); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2603.20255

 - **Pdf link:** https://arxiv.org/pdf/2603.20255

 - **Abstract**
 Speech-based AI educational applications have gained significant interest in recent years, particularly for children. However, children speech research remains limited due to the lack of publicly available datasets, especially for low-resource languages such as this http URL paper presents Abjad-Kids, an Arabic speech dataset designed for kindergarten and primary education, focusing on fundamental learning of alphabets, numbers, and colors. The dataset consists of 46397 audio samples collected from children aged 3 - 12 years, covering 141 classes. All samples were recorded under controlled specifications to ensure consistency in duration, sampling rate, and format. To address high intra-class similarity among Arabic phonemes and the limited samples per class, we propose a hierarchical audio classification based on CNN-LSTM architectures. Our proposed methodology decomposes alphabet recognition into a two-stage process: an initial grouping classification model followed by specialized classifiers for each group. Both strategies: static linguistic-based grouping and dynamic clustering-based grouping, were evaluated. Experimental results demonstrate that static linguistic-based grouping achieves superior performance. Comparisons between traditional machine learning with deep learning approaches, highlight the effectiveness of CNN-LSTM models combined with data augmentation. Despite achieving promising results, most of our experiments indicate a challenge with overfitting, which is likely due to the limited number of samples, even after data augmentation and model regularization. Thus, future work may focus on collecting additional data to address this issue. Abjad-Kids will be publicly available. We hope that Abjad-Kids enrich children representation in speech dataset, and be a good resource for future research in Arabic speech classification for kids.
#### TaigiSpeech: A Low-Resource Real-World Speech Intent Dataset and Preliminary Results with Scalable Data Mining In-the-Wild
 - **Authors:** Kai-Wei Chang, Yi-Cheng Lin, Huang-Cheng Chou, Wenze Ren, Yu-Han Huang, Yun-Shao Tsai, Chien-Cheng Chen, Yu Tsao, Yuan-Fu Liao, Shrikanth Narayanan, James Glass, Hung-yi Lee
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Machine Learning (cs.LG); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2603.21478

 - **Pdf link:** https://arxiv.org/pdf/2603.21478

 - **Abstract**
 Speech technologies have advanced rapidly and serve diverse populations worldwide. However, many languages remain underrepresented due to limited resources. In this paper, we introduce \textbf{TaigiSpeech}, a real-world speech intent dataset in Taiwanese Taigi (aka Taiwanese Hokkien/Southern Min), which is a low-resource and primarily spoken language. The dataset is collected from older adults, comprising 21 speakers with a total of 3k utterances. It is designed for practical intent detection scenarios, including healthcare and home assistant applications. To address the scarcity of labeled data, we explore two data mining strategies with two levels of supervision: keyword match data mining with LLM pseudo labeling via an intermediate language and an audio-visual framework that leverages multimodal cues with minimal textual supervision. This design enables scalable dataset construction for low-resource and unwritten spoken languages. TaigiSpeech will be released under the CC BY 4.0 license to facilitate broad adoption and research on low-resource and unwritten languages. The project website and the dataset can be found on this https URL.
#### TiCo: Time-Controllable Training for Spoken Dialogue Models
 - **Authors:** Kai-Wei Chang, Wei-Chih Chen, En-Pei Hu, Hung-yi Lee, James Glass
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Artificial Intelligence (cs.AI); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2603.22267

 - **Pdf link:** https://arxiv.org/pdf/2603.22267

 - **Abstract**
 We propose TiCo, a simple post-training method for enabling spoken dialogue models (SDMs) to follow time-constrained instructions and generate responses with controllable duration. This capability is valuable for real-world spoken language systems such as voice assistants and interactive agents, where controlling response duration can improve interaction quality. However, despite their strong ability to generate natural spoken responses, existing models lack time awareness and struggle to follow duration-related instructions (e.g., "Please generate a response lasting about 15 seconds"). Through an empirical evaluation of both open-source and commercial SDMs, we show that they frequently fail to satisfy such time-control requirements. TiCo addresses this limitation by enabling models to estimate elapsed speaking time during generation through Spoken Time Markers (STM) (e.g., <10.6 seconds>). These markers help the model maintain awareness of time and adjust the remaining content to meet the target duration. TiCo is simple and efficient: it requires only a small amount of data and no additional question-answer pairs, relying instead on self-generation and reinforcement learning. Experimental results show that TiCo significantly improves adherence to duration constraints while preserving response quality.


by Zyzzyva0381 (Windy). 


2026-03-24
