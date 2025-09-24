# Showing new listings for Wednesday, 24 September 2025
Auto update papers at about 2:30am UTC (10:30am Beijing time) every weekday.


阅读 `Usage.md`了解如何使用此repo实现个性化的Arxiv论文推送

See `Usage.md` for instructions on how to personalize the repo. 


Keyword list: ['text-to-speech', 'text to speech', 'tts', 'LLM-based', 'speech', 'voice']


Excluded: []


### Today: 17papers 
#### Automated Analysis of Naturalistic Recordings in Early Childhood: Applications, Challenges, and Opportunities
 - **Authors:** Jialu Li, Marvin Lavechin, Xulin Fan, Nancy L. McElwain, Alejandrina Cristia, Paola Garcia-Perera, Mark Hasegawa-Johnson
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2509.18235

 - **Pdf link:** https://arxiv.org/pdf/2509.18235

 - **Abstract**
 Naturalistic recordings capture audio in real-world environments where participants behave naturally without interference from researchers or experimental protocols. Naturalistic long-form recordings extend this concept by capturing spontaneous and continuous interactions over extended periods, often spanning hours or even days, in participants' daily lives. Naturalistic recordings have been extensively used to study children's behaviors, including how they interact with others in their environment, in the fields of psychology, education, cognitive science, and clinical research. These recordings provide an unobtrusive way to observe children in real-world settings beyond controlled and constrained experimental environments. Advancements in speech technology and machine learning have provided an initial step for researchers to automatically and systematically analyze large-scale naturalistic recordings of children. Despite the imperfect accuracy of machine learning models, these tools still offer valuable opportunities to uncover important insights into children's cognitive and social development. Several critical speech technologies involved include speaker diarization, vocalization classification, word count estimate from adults, speaker verification, and language diarization for code-switching. Most of these technologies have been primarily developed for adults, and speech technologies applied to children specifically are still vastly under-explored. To fill this gap, we discuss current progress, challenges, and opportunities in advancing these technologies to analyze naturalistic recordings of children during early development (<3 years of age). We strive to inspire the signal processing community and foster interdisciplinary collaborations to further develop this emerging technology and address its unique challenges and opportunities.
#### No Verifiable Reward for Prosody: Toward Preference-Guided Prosody Learning in TTS
 - **Authors:** Seungyoun Shin, Dongha Ahn, Jiwoo Kim, Sungwook Jeon
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Artificial Intelligence (cs.AI); Computation and Language (cs.CL); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2509.18531

 - **Pdf link:** https://arxiv.org/pdf/2509.18531

 - **Abstract**
 Recent work reports gains in neural text-to-speech (TTS) with Group Relative Policy Optimization (GRPO). However, in the absence of a verifiable reward for \textit{prosody}, GRPO trained on transcription-oriented signals (CER/NLL) lowers error rates yet collapses prosody into monotone, unnatural speech; adding speaker-similarity further destabilizes training and degrades CER. We address this with an \textit{iterative Direct Preference Optimization (DPO)} scheme that uses only a few hundred human-labeled preference pairs per round to directly optimize prosodic naturalness while regularizing to the current model. On \textbf{KoCC-TTS}, a curated dataset of authentic Korean call center interactions capturing task-oriented dialogues, our method attains the highest human preference (ELO) with competitive CER, outperforming GRPO and strong commercial baselines. These results suggest that when prosody cannot be rewarded automatically, \textit{human preference optimization} offers a practical and data-efficient path to natural and robust TTS. The demo page is available at \href{this https URL}
#### HarmoniFuse: A Component-Selective and Prompt-Adaptive Framework for Multi-Task Speech Language Modeling
 - **Authors:** Yuke Si, Runyan Yang, Yingying Gao, Junlan Feng, Chao Deng, Shilei Zhang
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Computation and Language (cs.CL); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2509.18570

 - **Pdf link:** https://arxiv.org/pdf/2509.18570

 - **Abstract**
 Recent advances in large language models have facilitated the development of unified speech language models (SLMs) capable of supporting multiple speech tasks within a shared architecture. However, tasks such as automatic speech recognition (ASR) and speech emotion recognition (SER) rely on distinct types of information: ASR primarily depends on linguistic content, whereas SER requires the integration of both linguistic and paralinguistic cues. Existing multitask SLMs typically adopt naive parameter sharing or prompt-based conditioning without explicitly modeling the differences in information composition required by each task. Such designs risk task interference and performance degradation, especially under limited data conditions. To address these limitations, we propose HarmoniFuse, a component-selective and prompt-adaptive framework for multi-task speech language modeling. HarmoniFuse is designed to harmonize heterogeneous task demands by selecting and fusing task-relevant components of speech representations. Specifically, it integrates a gated speech encoder to extract task-specific acoustic features and a prompt-adaptive dynamic fusion module to aggregate transformer layers based on task characteristics. In addition, a batch-interleaved training strategy enables leveraging separate ASR and SER datasets without requiring joint annotation. Experimental results demonstrate that HarmoniFuse improves both ASR and SER performance, offering a scalable and robust solution for multitask speech understanding under realistic data constraints.
#### Teaching Audio Models to Reason: A Unified Framework for Source- and Layer-wise Distillation
 - **Authors:** Runyan Yang, Yuke Si, Yingying Gao, Junlan Feng, Chao Deng, Shilei Zhang
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Computation and Language (cs.CL); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2509.18579

 - **Pdf link:** https://arxiv.org/pdf/2509.18579

 - **Abstract**
 While large audio language models excel at tasks like ASR and emotion recognition, they still struggle with complex reasoning due to the modality gap between audio and text as well as the lack of structured intermediate supervision. To address this, we propose a unified knowledge distillation framework to transfer reasoning capabilities from a high-capacity textual teacher model to a student audio models while preserving its acoustic competence. Our method introduces two key dimensions: source-wise distillation, which leverages both textual and acoustic teachers to provide complementary modality-specific supervision; and layer-wise distillation, which aligns teacher signals with appropriate student layers to improve transfer efficiency. This dual-dimensional strategy enables fine-grained control over the distillation process, effectively bridging the gap between symbolic reasoning and speech representations. Experimental results show significant improvements in audio reasoning performance, demonstrating the effectiveness of our framework as a reasoning transfer solution for audio modeling.
#### Group Relative Policy Optimization for Text-to-Speech with Large Language Models
 - **Authors:** Chang Liu, Ya-Jun Hu, Ying-Ying Gao, Shi-Lei Zhang, Zhen-Hua Ling
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2509.18798

 - **Pdf link:** https://arxiv.org/pdf/2509.18798

 - **Abstract**
 This paper proposes a GRPO-based approach to enhance the performance of large language model (LLM)-based text-to-speech (TTS) models by deriving rewards from an off-the-shelf automatic speech recognition (ASR) model. Compared to previous reinforcement learning methods for LLM-based TTS, our method requires no dedicated model for reward computation or training. Moreover, we design a composite reward function that combines character error rate (CER) with negative log-likelihood (NLL) obtained from the ASR model, providing more informative and accurate reward signals. We apply GRPO fine-tuning to pre-trained LLM-based TTS models and evaluate their zero-shot TTS performance. Experimental results show that the proposed method substantially improves both the intelligibility and naturalness of synthesized speech. Ablation studies and further analyses confirm the effectiveness of integrating the two reward components.
#### Towards Evaluating Generative Audio: Insights from Neural Audio Codec Embedding Distances
 - **Authors:** Arijit Biswas, Lars Villemoes
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2509.18823

 - **Pdf link:** https://arxiv.org/pdf/2509.18823

 - **Abstract**
 Neural audio codecs (NACs) achieve low-bitrate compression by learning compact audio representations, which can also serve as features for perceptual quality evaluation. We introduce DACe, an enhanced, higher-fidelity version of the Descript Audio Codec (DAC), trained on diverse real and synthetic tonal data with balanced sampling. We systematically compare Fréchet Audio Distance (FAD) and Maximum Mean Discrepancy (MMD) on MUSHRA tests across speech, music, and mixed content. FAD consistently outperforms MMD, and embeddings from higher-fidelity NACs (such as DACe) show stronger correlations with human judgments. While CLAP LAION Music (CLAP-M) and OpenL3 Mel128 (OpenL3-128M) embeddings achieve higher correlations, NAC embeddings provide a practical zero-shot approach to audio quality assessment, requiring only unencoded audio for training. These results demonstrate the dual utility of NACs for compression and perceptually informed audio evaluation.
#### Influence of Clean Speech Characteristics on Speech Enhancement Performance
 - **Authors:** Mingchi Hou, Ina Kodrasi
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2509.18885

 - **Pdf link:** https://arxiv.org/pdf/2509.18885

 - **Abstract**
 Speech enhancement (SE) performance is known to depend on noise characteristics and signal to noise ratio (SNR), yet intrinsic properties of the clean speech signal itself remain an underexplored factor. In this work, we systematically analyze how clean speech characteristics influence enhancement difficulty across multiple state of the art SE models, languages, and noise conditions. We extract a set of pitch, formant, loudness, and spectral flux features from clean speech and compute correlations with objective SE metrics, including frequency weighted segmental SNR and PESQ. Our results show that formant amplitudes are consistently predictive of SE performance, with higher and more stable formants leading to larger enhancement gains. We further demonstrate that performance varies substantially even within a single speaker's utterances, highlighting the importance of intraspeaker acoustic variability. These findings provide new insights into SE challenges, suggesting that intrinsic speech characteristics should be considered when designing datasets, evaluation protocols, and enhancement models.
#### Generalizability of Predictive and Generative Speech Enhancement Models to Pathological Speakers
 - **Authors:** Mingchi Hou, Ante Jukic, Ina Kodrasi
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2509.18890

 - **Pdf link:** https://arxiv.org/pdf/2509.18890

 - **Abstract**
 State of the art speech enhancement (SE) models achieve strong performance on neurotypical speech, but their effectiveness is substantially reduced for pathological speech. In this paper, we investigate strategies to address this gap for both predictive and generative SE models, including i) training models from scratch using pathological data, ii) finetuning models pretrained on neurotypical speech with additional data from pathological speakers, and iii) speaker specific personalization using only data from the individual pathological test speaker. Our results show that, despite the limited size of pathological speech datasets, SE models can be successfully trained or finetuned on such data. Finetuning models with data from several pathological speakers yields the largest performance improvements, while speaker specific personalization is less effective, likely due to the small amount of data available per speaker. These findings highlight the challenges and potential strategies for improving SE performance for pathological speakers.
#### Direct Preference Optimization for Speech Autoregressive Diffusion Models
 - **Authors:** Zhijun Liu, Dongya Jia, Xiaoqiang Wang, Chenpeng Du, Shuai Wang, Zhuo Chen, Haizhou Li
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2509.18928

 - **Pdf link:** https://arxiv.org/pdf/2509.18928

 - **Abstract**
 Autoregressive diffusion models (ARDMs) have recently been applied to speech generation, achieving state-of-the-art (SOTA) performance in zero-shot text-to-speech. By autoregressively generating continuous speech tokens with next-token diffusion, these models offer a promising alternative to next-token prediction, avoiding the technical complexities associated with discrete speech tokenization. As a relatively new paradigm, research on reinforcement learning (RL)-based fine-tuning of speech ARDMs remains limited. In this paper, we propose Autoregressive Diffusion-Direct Preference Optimization (ARDM-DPO) to advance this research. By fine-tuning the recently proposed zero-shot text-to-speech model DiTAR with DPO, we achieve significant improvements in terms of speech expressiveness and robustness for long texts.
#### HD-PPT: Hierarchical Decoding of Content- and Prompt-Preference Tokens for Instruction-based TTS
 - **Authors:** Sihang Nie, Xiaofen Xing, Jingyuan Xing, Baiji Liu, Xiangmin Xu
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2509.19001

 - **Pdf link:** https://arxiv.org/pdf/2509.19001

 - **Abstract**
 Large Language Model (LLM)-based Text-to-Speech (TTS) models have already reached a high degree of naturalness. However, the precision control of TTS inference is still challenging. Although instruction-based Text-to-Speech (Instruct-TTS) models are proposed, these models still lack fine-grained control due to the modality gap between single-level text instructions and multilevel speech tokens. To address this limitation, we propose HD-PPT, a framework that transforms speech synthesis into a structured, hierarchical task. To enable fine-grained control, we introduce a novel speech codec to extract distinct prompt-preference and content-preference tokens from the complex speech tokens, supervised by automatic speech recognition (ASR) and cross-lingual audio-text pre-training (CLAP) objectives. To bridge the modality gap of these tokens, we propose a hierarchical decoding strategy, where the LLM generates tokens in a structured order: first semantic, then fine-grained style, and finally complete acoustic representation. Extensive experiments demonstrate that this hierarchical paradigm significantly improves instruction adherence and achieves state-of-the-art naturalness, validating our approach for precise and controllable speech synthesis. Audio samples are available at this https URL.
#### Enhancing Noise Robustness for Neural Speech Codecs through Resource-Efficient Progressive Quantization Perturbation Simulation
 - **Authors:** Rui-Chen Zheng, Yang Ai, Hui-Peng Du, Zhen-Hua Ling
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2509.19025

 - **Pdf link:** https://arxiv.org/pdf/2509.19025

 - **Abstract**
 Noise robustness remains a critical challenge for deploying neural speech codecs in real-world acoustic scenarios where background noise is often inevitable. A key observation we make is that even slight input noise perturbations can cause unintended shifts in quantized codewords, thereby degrading the quality of reconstructed speech. Motivated by this finding, we propose a novel and resource-efficient training strategy to enhance the noise robustness of speech codecs by simulating such perturbations directly at the quantization level. Our approach introduces two core mechanisms: (1) a distance-weighted probabilistic top-K sampling strategy that replaces the conventional deterministic nearest-neighbor selection in residual vector quantization (RVQ); and (2) a progressive training scheme that introduces perturbations from the last to the first quantizer in a controlled manner. Crucially, our method is trained exclusively on clean speech, eliminating the need for any paired noisy-clean data. Experiments on two advanced neural speech codecs, Encodec and WavTokenizer, demonstrate that the proposed strategy substantially improves robustness under noisy conditions-for example, boosting UTMOS from 3.475 to 3.586 at 15 dB SNR on Encodec-while also enhancing coding quality for clean speech.
#### Training Flow Matching Models with Reliable Labels via Self-Purification
 - **Authors:** Hyeongju Kim, Yechan Yu, June Young Yi, Juheon Lee
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Artificial Intelligence (cs.AI); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2509.19091

 - **Pdf link:** https://arxiv.org/pdf/2509.19091

 - **Abstract**
 Training datasets are inherently imperfect, often containing mislabeled samples due to human annotation errors, limitations of tagging models, and other sources of noise. Such label contamination can significantly degrade the performance of a trained model. In this work, we introduce Self-Purifying Flow Matching (SPFM), a principled approach to filtering unreliable data within the flow-matching framework. SPFM identifies suspicious data using the model itself during the training process, bypassing the need for pretrained models or additional modules. Our experiments demonstrate that models trained with SPFM generate samples that accurately adhere to the specified conditioning, even when trained on noisy labels. Furthermore, we validate the robustness of SPFM on the TITW dataset, which consists of in-the-wild speech data, achieving performance that surpasses existing baselines.
#### MUSHRA-1S: A scalable and sensitive test approach for evaluating top-tier speech processing systems
 - **Authors:** Laura Lechler, Ivana Balic
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2509.19219

 - **Pdf link:** https://arxiv.org/pdf/2509.19219

 - **Abstract**
 Evaluating state-of-the-art speech systems necessitates scalable and sensitive evaluation methods to detect subtle but unacceptable artifacts. Standard MUSHRA is sensitive but lacks scalability, while ACR scales well but loses sensitivity and saturates at a high quality. To address this, we introduce MUSHRA 1S, a single-stimulus variant that rates one system at a time against a fixed anchor and reference. Across our experiments, MUSHRA 1S matches standard MUSHRA more closely than ACR, including in the high-quality regime, where ACR saturates. MUSHRA 1S also effectively identifies specific deviations and reduces range-equalizing biases by fixing context. Overall, MUSHRA 1S combines MUSHRA level sensitivity with ACR like scalability, making it a robust and scalable solution for benchmarking top-tier speech processing systems.
#### XMUspeech Systems for the ASVspoof 5 Challenge
 - **Authors:** Wangjie Li, Xingjia Xie, Yishuang Li, Wenhao Guan, Kaidi Wang, Pengyu Ren, Lin Li, Qingyang Hong
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2509.18102

 - **Pdf link:** https://arxiv.org/pdf/2509.18102

 - **Abstract**
 In this paper, we present our submitted XMUspeech systems to the speech deepfake detection track of the ASVspoof 5 Challenge. Compared to previous challenges, the audio duration in ASVspoof 5 database has significantly increased. And we observed that merely adjusting the input audio length can substantially improve system performance. To capture artifacts at multiple levels, we explored the performance of AASIST, HM-Conformer, Hubert, and Wav2vec2 with various input features and loss functions. Specifically, in order to obtain artifact-related information, we trained self-supervised models on the dataset containing spoofing utterances as the feature extractors. And we applied an adaptive multi-scale feature fusion (AMFF) method to integrate features from multiple Transformer layers with the hand-crafted feature to enhance the detection capability. In addition, we conducted extensive experiments on one-class loss functions and provided optimized configurations to better align with the anti-spoofing task. Our fusion system achieved a minDCF of 0.4783 and an EER of 20.45% in the closed condition, and a minDCF of 0.2245 and an EER of 9.36% in the open condition.
#### MNV-17: A High-Quality Performative Mandarin Dataset for Nonverbal Vocalization Recognition in Speech
 - **Authors:** Jialong Mai, Jinxin Ji, Xiaofen Xing, Chen Yang, Weidong Chen, Jingyuan Xing, Xiangmin Xu
 - **Subjects:** Subjects:
Sound (cs.SD); Artificial Intelligence (cs.AI); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2509.18196

 - **Pdf link:** https://arxiv.org/pdf/2509.18196

 - **Abstract**
 Mainstream Automatic Speech Recognition (ASR) systems excel at transcribing lexical content, but largely fail to recognize nonverbal vocalizations (NVs) embedded in speech, such as sighs, laughs, and coughs. This capability is important for a comprehensive understanding of human communication, as NVs convey crucial emotional and intentional cues. Progress in NV-aware ASR has been hindered by the lack of high-quality, well-annotated datasets. To address this gap, we introduce MNV-17, a 7.55-hour performative Mandarin speech dataset. Unlike most existing corpora that rely on model-based detection, MNV-17's performative nature ensures high-fidelity, clearly articulated NV instances. To the best of our knowledge, MNV-17 provides the most extensive set of nonverbal vocalization categories, comprising 17 distinct and well-balanced classes of common NVs. We benchmarked MNV-17 on four mainstream ASR architectures, evaluating their joint performance on semantic transcription and NV classification. The dataset and the pretrained model checkpoints will be made publicly available to facilitate future research in expressive ASR.
#### Discrete-time diffusion-like models for speech synthesis
 - **Authors:** Xiaozhou Tan, Minghui Zhao, Mattias Cross, Anton Ragni
 - **Subjects:** Subjects:
Machine Learning (cs.LG); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2509.18470

 - **Pdf link:** https://arxiv.org/pdf/2509.18470

 - **Abstract**
 Diffusion models have attracted a lot of attention in recent years. These models view speech generation as a continuous-time process. For efficient training, this process is typically restricted to additive Gaussian noising, which is limiting. For inference, the time is typically discretized, leading to the mismatch between continuous training and discrete sampling conditions. Recently proposed discrete-time processes, on the other hand, usually do not have these limitations, may require substantially fewer inference steps, and are fully consistent between training/inference conditions. This paper explores some diffusion-like discrete-time processes and proposes some new variants. These include processes applying additive Gaussian noise, multiplicative Gaussian noise, blurring noise and a mixture of blurring and Gaussian noises. The experimental results suggest that discrete-time processes offer comparable subjective and objective speech quality to their widely popular continuous counterpart, with more efficient and consistent training and inference schemas.
#### Explore the Reinforcement Learning for the LLM based ASR and TTS system
 - **Authors:** Changfeng Gao, Yabin Li, Keyu An, Zhifu Gao, Zhihao Du, Han Zhao, Xiangang Li
 - **Subjects:** Subjects:
Sound (cs.SD); Artificial Intelligence (cs.AI); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2509.18569

 - **Pdf link:** https://arxiv.org/pdf/2509.18569

 - **Abstract**
 In recent years, large language models (LLMs) have played an important role in automatic speech recognition (ASR) and text-to-speech (TTS) systems. While reinforcement learning (RL) has significantly enhanced LLM performance in text-based tasks, its application to ASR and TTS remains underexplored due to the complexity of training audio-based models. In this study, we propose a lightweight RL framework tailored for audio-based LLMs that can process audio inputs and generate audio outputs. Based on this framework, we evaluate the effectiveness of reinforcement learning on both ASR and TTS tasks. For the ASR task, we experiment with different rule-based reward functions within the Group Relative Policy Optimization (GRPO) framework and investigate the impact of RL data construction. For the TTS task, we compare GRPO with Differentiable Reward Optimization (DiffRO) and further combine the two approaches to achieve improved performance. Our experiments demonstrate that RL can significantly enhance the performance of both ASR and TTS systems, even with limited training data and a small number of optimization steps.


by Zyzzyva0381 (Windy). 


2025-09-24
