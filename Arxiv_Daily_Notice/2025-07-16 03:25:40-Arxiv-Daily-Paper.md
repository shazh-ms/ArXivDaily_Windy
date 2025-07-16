# Showing new listings for Wednesday, 16 July 2025
Auto update papers at about 2:30am UTC (10:30am Beijing time) every weekday.


阅读 `Usage.md`了解如何使用此repo实现个性化的Arxiv论文推送

See `Usage.md` for instructions on how to personalize the repo. 


Keyword list: ['text-to-speech', 'text to speech', 'tts', 'LLM-based', 'speech', 'voice']


Excluded: []


### Today: 4papers 
#### P.808 Multilingual Speech Enhancement Testing: Approach and Results of URGENT 2025 Challenge
 - **Authors:** Marvin Sach, Yihui Fu, Kohei Saijo, Wangyou Zhang, Samuele Cornell, Robin Scheibler, Chenda Li, Anurag Kumar, Wei Wang, Yanmin Qian, Shinji Watanabe, Tim Fingscheidt
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2507.11306

 - **Pdf link:** https://arxiv.org/pdf/2507.11306

 - **Abstract**
 In speech quality estimation for speech enhancement (SE) systems, subjective listening tests so far are considered as the gold standard. This should be even more true considering the large influx of new generative or hybrid methods into the field, revealing issues of some objective metrics. Efforts such as the Interspeech 2025 URGENT Speech Enhancement Challenge also involving non-English datasets add the aspect of multilinguality to the testing procedure. In this paper, we provide a brief recap of the ITU-T P.808 crowdsourced subjective listening test method. A first novel contribution is our proposed process of localizing both text and audio components of Naderi and Cutler's implementation of crowdsourced subjective absolute category rating (ACR) listening tests involving text-to-speech (TTS). Further, we provide surprising analyses of and insights into URGENT Challenge results, tackling the reliability of (P.808) ACR subjective testing as gold standard in the age of generative AI. Particularly, it seems that for generative SE methods, subjective (ACR MOS) and objective (DNSMOS, NISQA) reference-free metrics should be accompanied by objective phone fidelity metrics to reliably detect hallucinations. Finally, in the accepted version, we will release our localization scripts and methods for easy deployment for new multilingual speech enhancement subjective evaluations according to ITU-T P.808.
#### Towards Reliable Objective Evaluation Metrics for Generative Singing Voice Separation Models
 - **Authors:** Paul A. Bereuter, Benjamin Stahl, Mark D. Plumbley, Alois Sontacchi
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2507.11427

 - **Pdf link:** https://arxiv.org/pdf/2507.11427

 - **Abstract**
 Traditional Blind Source Separation Evaluation (BSS-Eval) metrics were originally designed to evaluate linear audio source separation models based on methods such as time-frequency masking. However, recent generative models may introduce nonlinear relationships between the separated and reference signals, limiting the reliability of these metrics for objective evaluation. To address this issue, we conduct a Degradation Category Rating listening test and analyze correlations between the obtained degradation mean opinion scores (DMOS) and a set of objective audio quality metrics for the task of singing voice separation. We evaluate three state-of-the-art discriminative models and two new competitive generative models. For both discriminative and generative models, intrusive embedding-based metrics show higher correlations with DMOS than conventional intrusive metrics such as BSS-Eval. For discriminative models, the highest correlation is achieved by the MSE computed on Music2Latent embeddings. When it comes to the evaluation of generative models, the strongest correlations are evident for the multi-resolution STFT loss and the MSE calculated on MERT-L12 embeddings, with the latter also providing the most balanced correlation across both model types. Our results highlight the limitations of BSS-Eval metrics for evaluating generative singing voice separation models and emphasize the need for careful selection and validation of alternative evaluation metrics for the task of singing voice separation.
#### Pronunciation Deviation Analysis Through Voice Cloning and Acoustic Comparison
 - **Authors:** Andrew Valdivia, Yueming Zhang, Hailu Xu, Amir Ghasemkhani, Xin Qin
 - **Subjects:** Subjects:
Sound (cs.SD); Artificial Intelligence (cs.AI); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2507.10985

 - **Pdf link:** https://arxiv.org/pdf/2507.10985

 - **Abstract**
 This paper presents a novel approach for detecting mispronunciations by analyzing deviations between a user's original speech and their voice-cloned counterpart with corrected pronunciation. We hypothesize that regions with maximal acoustic deviation between the original and cloned utterances indicate potential mispronunciations. Our method leverages recent advances in voice cloning to generate a synthetic version of the user's voice with proper pronunciation, then performs frame-by-frame comparisons to identify problematic segments. Experimental results demonstrate the effectiveness of this approach in pinpointing specific pronunciation errors without requiring predefined phonetic rules or extensive training data for each target language.
#### FasTUSS: Faster Task-Aware Unified Source Separation
 - **Authors:** Francesco Paissan, Gordon Wichern, Yoshiki Masuyama, Ryo Aihara, François G. Germain, Kohei Saijo, Jonathan Le Roux
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS); Signal Processing (eess.SP)
 - **Arxiv link:** https://arxiv.org/abs/2507.11435

 - **Pdf link:** https://arxiv.org/pdf/2507.11435

 - **Abstract**
 Time-Frequency (TF) dual-path models are currently among the best performing audio source separation network architectures, achieving state-of-the-art performance in speech enhancement, music source separation, and cinematic audio source separation. While they are characterized by a relatively low parameter count, they still require a considerable number of operations, implying a higher execution time. This problem is exacerbated by the trend towards bigger models trained on large amounts of data to solve more general tasks, such as the recently introduced task-aware unified source separation (TUSS) model. TUSS, which aims to solve audio source separation tasks using a single, conditional model, is built upon TF-Locoformer, a TF dual-path model combining convolution and attention layers. The task definition comes in the form of a sequence of prompts that specify the number and type of sources to be extracted. In this paper, we analyze the design choices of TUSS with the goal of optimizing its performance-complexity trade-off. We derive two more efficient models, FasTUSS-8.3G and FasTUSS-11.7G that reduce the original model's operations by 81\% and 73\% with minor performance drops of 1.2~dB and 0.4~dB averaged over all benchmarks, respectively. Additionally, we investigate the impact of prompt conditioning to derive a causal TUSS model.


by Zyzzyva0381 (Windy). 


2025-07-16
