# Showing new listings for Wednesday, 22 October 2025
Auto update papers at about 2:30am UTC (10:30am Beijing time) every weekday.


阅读 `Usage.md`了解如何使用此repo实现个性化的Arxiv论文推送

See `Usage.md` for instructions on how to personalize the repo. 


Keyword list: ['text-to-speech', 'text to speech', 'tts', 'LLM-based', 'speech', 'voice']


Excluded: []


### Today: 5papers 
#### Hearing Health in Home Healthcare: Leveraging LLMs for Illness Scoring and ALMs for Vocal Biomarker Extraction
 - **Authors:** Yu-Wen Chen, William Ho, Sasha M. Vergez, Grace Flaherty, Pallavi Gupta, Zhihong Zhang, Maryam Zolnoori, Margaret V. McDonald, Maxim Topaz, Zoran Kostic, Julia Hirschberg
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2510.18169

 - **Pdf link:** https://arxiv.org/pdf/2510.18169

 - **Abstract**
 The growing demand for home healthcare calls for tools that can support care delivery. In this study, we explore automatic health assessment from voice using real-world home care visit data, leveraging the diverse patient information it contains. First, we utilize Large Language Models (LLMs) to integrate Subjective, Objective, Assessment, and Plan (SOAP) notes derived from unstructured audio transcripts and structured vital signs into a holistic illness score that reflects a patient's overall health. This compact representation facilitates cross-visit health status comparisons and downstream analysis. Next, we design a multi-stage preprocessing pipeline to extract short speech segments from target speakers in home care recordings for acoustic analysis. We then employ an Audio Language Model (ALM) to produce plain-language descriptions of vocal biomarkers and examine their association with individuals' health status. Our experimental results benchmark both commercial and open-source LLMs in estimating illness scores, demonstrating their alignment with actual clinical outcomes, and revealing that SOAP notes are substantially more informative than vital signs. Building on the illness scores, we provide the first evidence that ALMs can identify health-related acoustic patterns from home care recordings and present them in a human-readable form. Together, these findings highlight the potential of LLMs and ALMs to harness heterogeneous in-home visit data for better patient monitoring and care.
#### Diffusion Buffer for Online Generative Speech Enhancement
 - **Authors:** Bunlong Lay, Rostislav Makarov, Simon Welker, Maris Hillemann, Timo Gerkmann
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Machine Learning (cs.LG); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2510.18744

 - **Pdf link:** https://arxiv.org/pdf/2510.18744

 - **Abstract**
 Online Speech Enhancement was mainly reserved for predictive models. A key advantage of these models is that for an incoming signal frame from a stream of data, the model is called only once for enhancement. In contrast, generative Speech Enhancement models often require multiple calls, resulting in a computational complexity that is too high for many online speech enhancement applications. This work presents the Diffusion Buffer, a generative diffusion-based Speech Enhancement model which only requires one neural network call per incoming signal frame from a stream of data and performs enhancement in an online fashion on a consumer-grade GPU. The key idea of the Diffusion Buffer is to align physical time with Diffusion time-steps. The approach progressively denoises frames through physical time, where past frames have more noise removed. Consequently, an enhanced frame is output to the listener with a delay defined by the Diffusion Buffer, and the output frame has a corresponding look-ahead. In this work, we extend upon our previous work by carefully designing a 2D convolutional UNet architecture that specifically aligns with the Diffusion Buffer's look-ahead. We observe that the proposed UNet improves performance, particularly when the algorithmic latency is low. Moreover, we show that using a Data Prediction loss instead of Denoising Score Matching loss enables flexible control over the trade-off between algorithmic latency and quality during inference. The extended Diffusion Buffer equipped with a novel NN and loss function drastically reduces the algorithmic latency from 320 - 960 ms to 32 - 176 ms with an even increased performance. While it has been shown before that offline generative diffusion models outperform predictive approaches in unseen noisy speech data, we confirm that the online Diffusion Buffer also outperforms its predictive counterpart on unseen noisy speech data.
#### ParaStyleTTS: Toward Efficient and Robust Paralinguistic Style Control for Expressive Text-to-Speech Generation
 - **Authors:** Haowei Lou, Hye-Young Paik, Wen Hu, Lina Yao
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2510.18308

 - **Pdf link:** https://arxiv.org/pdf/2510.18308

 - **Abstract**
 Controlling speaking style in text-to-speech (TTS) systems has become a growing focus in both academia and industry. While many existing approaches rely on reference audio to guide style generation, such methods are often impractical due to privacy concerns and limited accessibility. More recently, large language models (LLMs) have been used to control speaking style through natural language prompts; however, their high computational cost, lack of interpretability, and sensitivity to prompt phrasing limit their applicability in real-time and resource-constrained environments. In this work, we propose ParaStyleTTS, a lightweight and interpretable TTS framework that enables expressive style control from text prompts alone. ParaStyleTTS features a novel two-level style adaptation architecture that separates prosodic and paralinguistic speech style modeling. It allows fine-grained and robust control over factors such as emotion, gender, and age. Unlike LLM-based methods, ParaStyleTTS maintains consistent style realization across varied prompt formulations and is well-suited for real-world applications, including on-device and low-resource deployment. Experimental results show that ParaStyleTTS generates high-quality speech with performance comparable to state-of-the-art LLM-based systems while being 30x faster, using 8x fewer parameters, and requiring 2.5x less CUDA memory. Moreover, ParaStyleTTS exhibits superior robustness and controllability over paralinguistic speaking styles, providing a practical and efficient solution for style-controllable text-to-speech generation. Demo can be found at this https URL. Code can be found at this https URL.
#### Bayesian Low-Rank Factorization for Robust Model Adaptation
 - **Authors:** Enes Yavuz Ugan, Ngoc-Quan Pham, Alexander Waibel
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Machine Learning (cs.LG); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2510.18723

 - **Pdf link:** https://arxiv.org/pdf/2510.18723

 - **Abstract**
 Large speech foundation models achieve strong performance across many domains, but they often require adaptation to handle local needs such as code-switching, where speakers mix languages within the same utterance. Direct fine-tuning of these models risks overfitting to the target domain and overwriting the broad capabilities of the base model. To address this challenge, we explore Bayesian factorized adapters for speech foundation models, which place priors near zero to achieve sparser adaptation matrices and thereby retain general performance while adapting to specific domains. We apply our approach to the Whisper model and evaluate on different multilingual code-switching scenarios. Our results show only minimal adaptation loss while significantly reducing catastrophic forgetting of the base model. Compared to LoRA, our method achieves a backward gain of 54% with only a 4% drop on the new domain. These findings highlight the effectiveness of Bayesian adaptation for fine-tuning speech foundation models without sacrificing generalization.
#### Adapting Language Balance in Code-Switching Speech
 - **Authors:** Enes Yavuz Ugan, Ngoc-Quan Pham, Alexander Waibel
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Machine Learning (cs.LG); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2510.18724

 - **Pdf link:** https://arxiv.org/pdf/2510.18724

 - **Abstract**
 Despite achieving impressive results on standard benchmarks, large foundational models still struggle against code-switching test cases. When data scarcity cannot be used as the usual justification for poor performance, the reason may lie in the infrequent occurrence of code-switched moments, where the embedding of the second language appears subtly. Instead of expecting the models to learn this infrequency on their own, it might be beneficial to provide the training process with labels. Evaluating model performance on code-switching data requires careful localization of code-switching points where recognition errors are most consequential, so that the analysis emphasizes mistakes occurring at those moments. Building on this observation, we leverage the difference between the embedded and the main language to highlight those code-switching points and thereby emphasize learning at those locations. This simple yet effective differentiable surrogate mitigates context bias during generation -- the central challenge in code-switching -- thereby improving the model's robustness. Our experiments with Arabic and Chinese-English showed that the models are able to predict the switching places more correctly, reflected by the reduced substitution error.


by Zyzzyva0381 (Windy). 


2025-10-22
