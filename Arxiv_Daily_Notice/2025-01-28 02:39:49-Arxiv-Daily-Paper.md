# Showing new listings for Monday, 27 January 2025
Auto update papers at about 2:30am UTC (10:30am Beijing time) every weekday.


阅读 `Usage.md`了解如何使用此repo实现个性化的Arxiv论文推送

See `Usage.md` for instructions on how to personalize the repo. 


Keyword list: ['acoustic', 'reinforcement learning', 'reverb', 'meta', 'separate', 'reconstruction', 'noise', 'enhance', 'localization', 'speech']


Excluded: []


### Today: 6papers 
#### Generalizable Audio Deepfake Detection via Latent Space Refinement and Augmentation
 - **Authors:** Wen Huang, Yanmei Gu, Zhiming Wang, Huijia Zhu, Yanmin Qian
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2501.14240

 - **Pdf link:** https://arxiv.org/pdf/2501.14240

 - **Abstract**
 Advances in speech synthesis technologies, like text-to-speech (TTS) and voice conversion (VC), have made detecting deepfake speech increasingly challenging. Spoofing countermeasures often struggle to generalize effectively, particularly when faced with unseen attacks. To address this, we propose a novel strategy that integrates Latent Space Refinement (LSR) and Latent Space Augmentation (LSA) to improve the generalization of deepfake detection systems. LSR introduces multiple learnable prototypes for the spoof class, refining the latent space to better capture the intricate variations within spoofed data. LSA further diversifies spoofed data representations by applying augmentation techniques directly in the latent space, enabling the model to learn a broader range of spoofing patterns. We evaluated our approach on four representative datasets, i.e. ASVspoof 2019 LA, ASVspoof 2021 LA and DF, and In-The-Wild. The results show that LSR and LSA perform well individually, and their integration achieves competitive results, matching or surpassing current state-of-the-art methods.
#### Characteristic-Specific Partial Fine-Tuning for Efficient Emotion and Speaker Adaptation in Codec Language Text-to-Speech Models
 - **Authors:** Tianrui Wang, Meng Ge, Cheng Gong, Chunyu Qiang, Haoyu Wang, Zikang Huang, Yu Jiang, Xiaobao Wang, Xie Chen, Longbiao Wang, Jianwu Dang
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2501.14273

 - **Pdf link:** https://arxiv.org/pdf/2501.14273

 - **Abstract**
 Recently, emotional speech generation and speaker cloning have garnered significant interest in text-to-speech (TTS). With the open-sourcing of codec language TTS models trained on massive datasets with large-scale parameters, adapting these general pre-trained TTS models to generate speech with specific emotional expressions and target speaker characteristics has become a topic of great attention. Common approaches, such as full and adapter-based fine-tuning, often overlook the specific contributions of model parameters to emotion and speaker control. Treating all parameters uniformly during fine-tuning, especially when the target data has limited content diversity compared to the pre-training corpus, results in slow training speed and an increased risk of catastrophic forgetting. To address these challenges, we propose a characteristic-specific partial fine-tuning strategy, short as CSP-FT. First, we use a weighted-sum approach to analyze the contributions of different Transformer layers in a pre-trained codec language TTS model for emotion and speaker control in the generated speech. We then selectively fine-tune the layers with the highest and lowest characteristic-specific contributions to generate speech with target emotional expression and speaker identity. Experimental results demonstrate that our method achieves performance comparable to, or even surpassing, full fine-tuning in generating speech with specific emotional expressions and speaker identities. Additionally, CSP-FT delivers approximately 2x faster training speeds, fine-tunes only around 8% of parameters, and significantly reduces catastrophic forgetting. Furthermore, we show that codec language TTS models perform competitively with self-supervised models in speaker identification and emotion classification tasks, offering valuable insights for developing universal speech processing models.
#### FireRedASR: Open-Source Industrial-Grade Mandarin Speech Recognition Models from Encoder-Decoder to LLM Integration
 - **Authors:** Kai-Tuo Xu, Feng-Long Xie, Xu Tang, Yao Hu
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2501.14350

 - **Pdf link:** https://arxiv.org/pdf/2501.14350

 - **Abstract**
 We present FireRedASR, a family of large-scale automatic speech recognition (ASR) models for Mandarin, designed to meet diverse requirements in superior performance and optimal efficiency across various applications. FireRedASR comprises two variants: FireRedASR-LLM: Designed to achieve state-of-the-art (SOTA) performance and to enable seamless end-to-end speech interaction. It adopts an Encoder-Adapter-LLM framework leveraging large language model (LLM) capabilities. On public Mandarin benchmarks, FireRedASR-LLM (8.3B parameters) achieves an average Character Error Rate (CER) of 3.05%, surpassing the latest SOTA of 3.33% with an 8.4% relative CER reduction (CERR). It demonstrates superior generalization capability over industrial-grade baselines, achieving 24%-40% CERR in multi-source Mandarin ASR scenarios such as video, live, and intelligent assistant. FireRedASR-AED: Designed to balance high performance and computational efficiency and to serve as an effective speech representation module in LLM-based speech models. It utilizes an Attention-based Encoder-Decoder (AED) architecture. On public Mandarin benchmarks, FireRedASR-AED (1.1B parameters) achieves an average CER of 3.18%, slightly worse than FireRedASR-LLM but still outperforming the latest SOTA model with over 12B parameters. It offers a more compact size, making it suitable for resource-constrained applications. Moreover, both models exhibit competitive results on Chinese dialects and English speech benchmarks and excel in singing lyrics recognition. To advance research in speech processing, we release our models and inference code at this https URL.
#### Enhancing Intelligibility for Generative Target Speech Extraction via Joint Optimization with Target Speaker ASR
 - **Authors:** Hao Ma, Rujin Chen, Ruihao Jing, Xiao-Lei Zhang, Ju Liu, Xuelong Li
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2501.14477

 - **Pdf link:** https://arxiv.org/pdf/2501.14477

 - **Abstract**
 Target speech extraction (TSE) isolates the speech of a specific speaker from a multi-talker overlapped speech mixture. Most existing TSE models rely on discriminative methods, typically predicting a time-frequency spectrogram mask for the target speech. However, imperfections in these masks often result in over-/under-suppression of target/non-target speech, degrading perceptual quality. Generative methods, by contrast, re-synthesize target speech based on the mixture and target speaker cues, achieving superior perceptual quality. Nevertheless, these methods often overlook speech intelligibility, leading to alterations or loss of semantic content in the re-synthesized speech. Inspired by the Whisper model's success in target speaker ASR, we propose a generative TSE framework based on the pre-trained Whisper model to address the above issues. This framework integrates semantic modeling with flow-based acoustic modeling to achieve both high intelligibility and perceptual quality. Results from multiple benchmarks demonstrate that the proposed method outperforms existing generative and discriminative baselines. We present speech samples on our demo page.
#### Diffusion based Text-to-Music Generationwith Global and Local Text based Conditioning
 - **Authors:** Jisi Zhang, Pablo Peso Parada, Md Asif Jalal, Karthikeyan Saravanan
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2501.14680

 - **Pdf link:** https://arxiv.org/pdf/2501.14680

 - **Abstract**
 Diffusion based Text-To-Music (TTM) models generate music corresponding to text descriptions. Typically UNet based diffusion models condition on text embeddings generated from a pre-trained large language model or from a cross-modality audio-language representation model. This work proposes a diffusion based TTM, in which the UNet is conditioned on both (i) a uni-modal language model (e.g., T5) via cross-attention and (ii) a cross-modal audio-language representation model (e.g., CLAP) via Feature-wise Linear Modulation (FiLM). The diffusion model is trained to exploit both a local text representation from the T5 and a global representation from the CLAP. Furthermore, we propose modifications that extract both global and local representations from the T5 through pooling mechanisms that we call mean pooling and self-attention pooling. This approach mitigates the need for an additional encoder (e.g., CLAP) to extract a global representation, thereby reducing the number of model parameters. Our results show that incorporating the CLAP global embeddings to the T5 local embeddings enhances text adherence (KL=1.47) compared to a baseline model solely relying on the T5 local embeddings (KL=1.54). Alternatively, extracting global text embeddings directly from the T5 local embeddings through the proposed mean pooling approach yields superior generation quality (FAD=1.89) while exhibiting marginally inferior text adherence (KL=1.51) against the model conditioned on both CLAP and T5 text embeddings (FAD=1.94 and KL=1.47). Our proposed solution is not only efficient but also compact in terms of the number of parameters required.
#### Leveraging Spatial Cues from Cochlear Implant Microphones to Efficiently Enhance Speech Separation in Real-World Listening Scenes
 - **Authors:** Feyisayo Olalere, Kiki van der Heijden, Christiaan H. Stronks, Jeroen Briaire, Johan HM Frijns, Marcel van Gerven
 - **Subjects:** Subjects:
Sound (cs.SD); Artificial Intelligence (cs.AI); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2501.14610

 - **Pdf link:** https://arxiv.org/pdf/2501.14610

 - **Abstract**
 Speech separation approaches for single-channel, dry speech mixtures have significantly improved. However, real-world spatial and reverberant acoustic environments remain challenging, limiting the effectiveness of these approaches for assistive hearing devices like cochlear implants (CIs). To address this, we quantify the impact of real-world acoustic scenes on speech separation and explore how spatial cues can enhance separation quality efficiently. We analyze performance based on implicit spatial cues (inherent in the acoustic input and learned by the model) and explicit spatial cues (manually calculated spatial features added as auxiliary inputs). Our findings show that spatial cues (both implicit and explicit) improve separation for mixtures with spatially separated and nearby talkers. Furthermore, spatial cues enhance separation when spectral cues are ambiguous, such as when voices are similar. Explicit spatial cues are particularly beneficial when implicit spatial cues are weak. For instance, single CI microphone recordings provide weaker implicit spatial cues than bilateral CIs, but even single CIs benefit from explicit cues. These results emphasize the importance of training models on real-world data to improve generalizability in everyday listening scenarios. Additionally, our statistical analyses offer insights into how data properties influence model performance, supporting the development of efficient speech separation approaches for CIs and other assistive devices in real-world settings.


by Zyzzyva0381 (Windy). 


2025-01-28
