# Showing new listings for Thursday, 11 September 2025
Auto update papers at about 2:30am UTC (10:30am Beijing time) every weekday.


阅读 `Usage.md`了解如何使用此repo实现个性化的Arxiv论文推送

See `Usage.md` for instructions on how to personalize the repo. 


Keyword list: ['text-to-speech', 'text to speech', 'tts', 'LLM-based', 'speech', 'voice']


Excluded: []


### Today: 5papers 
#### A Bottom-up Framework with Language-universal Speech Attribute Modeling for Syllable-based ASR
 - **Authors:** Hao Yen, Pin-Jui Ku, Sabato Marco Siniscalchi, Chin-Hui Lee
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2509.08173

 - **Pdf link:** https://arxiv.org/pdf/2509.08173

 - **Abstract**
 We propose a bottom-up framework for automatic speech recognition (ASR) in syllable-based languages by unifying language-universal articulatory attribute modeling with syllable-level prediction. The system first recognizes sequences or lattices of articulatory attributes that serve as a language-universal, interpretable representation of pronunciation, and then transforms them into syllables through a structured knowledge integration process. We introduce two evaluation metrics, namely Pronunciation Error Rate (PrER) and Syllable Homonym Error Rate (SHER), to evaluate the model's ability to capture pronunciation and handle syllable ambiguities. Experimental results on the AISHELL-1 Mandarin corpus demonstrate that the proposed bottom-up framework achieves competitive performance and exhibits better robustness under low-resource conditions compared to the direct syllable prediction model. Furthermore, we investigate the zero-shot cross-lingual transferability on Japanese and demonstrate significant improvements over character- and phoneme-based baselines by 40% error rate reduction.
#### Few-shot Personalization via In-Context Learning for Speech Emotion Recognition based on Speech-Language Model
 - **Authors:** Mana Ihori, Taiga Yamane, Naotaka Kawata, Naoki Makishima, Tomohiro Tanaka, Satoshi Suzuki, Shota Orihashi, Ryo Masumura
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2509.08344

 - **Pdf link:** https://arxiv.org/pdf/2509.08344

 - **Abstract**
 This paper proposes a personalization method for speech emotion recognition (SER) through in-context learning (ICL). Since the expression of emotions varies from person to person, speaker-specific adaptation is crucial for improving the SER performance. Conventional SER methods have been personalized using emotional utterances of a target speaker, but it is often difficult to prepare utterances corresponding to all emotion labels in advance. Our idea to overcome this difficulty is to obtain speaker characteristics by conditioning a few emotional utterances of the target speaker in ICL-based inference. ICL is a method to perform unseen tasks by conditioning a few input-output examples through inference in large language models (LLMs). We meta-train a speech-language model extended from the LLM to learn how to perform personalized SER via ICL. Experimental results using our newly collected SER dataset demonstrate that the proposed method outperforms conventional methods.
#### Joint Learning using Mixture-of-Expert-Based Representation for Enhanced Speech Generation and Robust Emotion Recognition
 - **Authors:** Jing-Tong Tzeng, Carlos Busso, Chi-Chun Lee
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Artificial Intelligence (cs.AI)
 - **Arxiv link:** https://arxiv.org/abs/2509.08470

 - **Pdf link:** https://arxiv.org/pdf/2509.08470

 - **Abstract**
 Speech emotion recognition (SER) plays a critical role in building emotion-aware speech systems, but its performance degrades significantly under noisy conditions. Although speech enhancement (SE) can improve robustness, it often introduces artifacts that obscure emotional cues and adds computational overhead to the pipeline. Multi-task learning (MTL) offers an alternative by jointly optimizing SE and SER tasks. However, conventional shared-backbone models frequently suffer from gradient interference and representational conflicts between tasks. To address these challenges, we propose the Sparse Mixture-of-Experts Representation Integration Technique (Sparse MERIT), a flexible MTL framework that applies frame-wise expert routing over self-supervised speech representations. Sparse MERIT incorporates task-specific gating networks that dynamically select from a shared pool of experts for each frame, enabling parameter-efficient and task-adaptive representation learning. Experiments on the MSP-Podcast corpus show that Sparse MERIT consistently outperforms baseline models on both SER and SE tasks. Under the most challenging condition of -5 dB signal-to-noise ratio (SNR), Sparse MERIT improves SER F1-macro by an average of 12.0% over a baseline relying on a SE pre-processing strategy, and by 3.4% over a naive MTL baseline, with statistical significance on unseen noise conditions. For SE, Sparse MERIT improves segmental SNR (SSNR) by 28.2% over the SE pre-processing baseline and by 20.0% over the naive MTL baseline. These results demonstrate that Sparse MERIT provides robust and generalizable performance for both emotion recognition and enhancement tasks in noisy environments.
#### Accelerating Diffusion Transformer-Based Text-to-Speech with Transformer Layer Caching
 - **Authors:** Siratish Sakpiboonchit
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2509.08696

 - **Pdf link:** https://arxiv.org/pdf/2509.08696

 - **Abstract**
 This paper presents a method to accelerate the inference process of diffusion transformer (DiT)-based text-to-speech (TTS) models by applying a selective caching mechanism to transformer layers. Specifically, I integrate SmoothCache into the F5-TTS architecture, focusing on caching outputs of self-attention and feed-forward network layers to reduce redundant computations during the denoising process. A calibration phase is introduced to analyze L1 relative errors between timesteps, guiding the selection of cache schedules that minimize quality degradation. To address the problem of inter-layer dependency, a unified caching schedule is adopted, applying the cache pattern derived from self-attention layers to both layer types. Experiments on LibriSpeech-PC and Seed-TTS datasets evaluate various cache thresholds and denoising step configurations. Results show that caching at higher denoising steps reduces inference time without compromising output quality, whereas caching at lower steps can negatively impact synthesis quality similarly to reducing the total number of denoising steps. Objective and subjective metrics confirm the effectiveness of SmoothCache in maintaining performance while improving computational efficiency. Comparisons between cached inference and reduced-step inference further highlight the benefits of selective caching, especially under high-step configurations. This work demonstrates that transformer layer caching is a practical solution for optimizing diffusion transformer-based TTS models without requiring architectural changes or retraining. Example inference results can be heard at this https URL .
#### Behind the Scenes: Mechanistic Interpretability of LoRA-adapted Whisper for Speech Emotion Recognition
 - **Authors:** Yujian Ma, Jinqiu Sang, Ruizhe Li
 - **Subjects:** Subjects:
Sound (cs.SD); Machine Learning (cs.LG); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2509.08454

 - **Pdf link:** https://arxiv.org/pdf/2509.08454

 - **Abstract**
 Large pre-trained speech models such as Whisper offer strong generalization but pose significant challenges for resource-efficient adaptation. Low-Rank Adaptation (LoRA) has become a popular parameter-efficient fine-tuning method, yet its underlying mechanisms in speech tasks remain poorly understood. In this work, we conduct the first systematic mechanistic interpretability study of LoRA within the Whisper encoder for speech emotion recognition (SER). Using a suite of analytical tools, including layer contribution probing, logit-lens inspection, and representational similarity via singular value decomposition (SVD) and centered kernel alignment (CKA), we reveal two key mechanisms: a delayed specialization process that preserves general features in early layers before consolidating task-specific information, and a forward alignment, backward differentiation dynamic between LoRA's matrices. Our findings clarify how LoRA reshapes encoder hierarchies, providing both empirical insights and a deeper mechanistic understanding for designing efficient and interpretable adaptation strategies in large speech models.


by Zyzzyva0381 (Windy). 


2025-09-11
