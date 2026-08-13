# Showing new listings for Thursday, 13 August 2026
Auto update papers at about 2:30am UTC (10:30am Beijing time) every weekday.


阅读 `Usage.md`了解如何使用此repo实现个性化的Arxiv论文推送

See `Usage.md` for instructions on how to personalize the repo. 


Keyword list: ['text-to-speech', 'text to speech', 'tts', 'LLM-based', 'speech', 'voice']


Excluded: []


### Today: 8papers 
#### Robust Multi-Tier Infant-Centered Audio Understanding with Whisper via Structured Speaker Conditioning
 - **Authors:** Xulin Fan, Jialu Li, Mohammad Nur Hossain Khan, Kexin Hu, Bashima Islam, Mark Hasegawa-Johnson, Nancy L. McElwain
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Computation and Language (cs.CL); Machine Learning (cs.LG)
 - **Arxiv link:** https://arxiv.org/abs/2608.11587

 - **Pdf link:** https://arxiv.org/pdf/2608.11587

 - **Abstract**
 Recent advances in model design and self-supervised audio representations have improved speech and audio understanding, yet infant-centered naturalistic recordings remain challenging due to limited labeled data, low signal-to-noise ratio, and cross-family domain shifts. We present a family-conditioned, multi-tier audio tagger that combines a LoRA-finetuned Whisper encoder with a lightweight, target-speaker-aware Transformer for long-context inference and framewise prediction across tiers. To improve temporal coherence, we incorporate a simple sequence-level smoothing loss, and to enhance robustness across households, we introduce a factorized speaker-token design with a shared tier token and a learned family-specific offset, reducing family bias and promoting generalizable representations. Together, these choices enable efficient and effective infant-centered audio tagging of daylong audio recordings in home environments.
#### Deep Learning Based Relative Transfer Matrix Estimation for Multiple Sources and Multiple Microphones
 - **Authors:** Oshan A. B. Yalegama, Wageesha N. Manamperi
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Artificial Intelligence (cs.AI); Signal Processing (eess.SP)
 - **Arxiv link:** https://arxiv.org/abs/2608.11627

 - **Pdf link:** https://arxiv.org/pdf/2608.11627

 - **Abstract**
 The Relative Transfer Matrix (ReTM), recently introduced as a generalization of the relative transfer function for multiple receivers and sources, shows promising performance when applied to speech enhancement in noisy environments. Estimating the ReTM of sound sources by exploiting the covariance matrices of multichannel recordings is highly beneficial for practical applications and, to date, remains the only proposed approach. This paper investigates deep learning-based ReTM estimation. We propose three novel supervised learning frameworks using time and short-time frequency transform domain convolutional networks, and a Long Short-Term Memory-based recurrent neural network. Experimental results demonstrate that the proposed models achieve more accurate estimation of the ReTM using five objective metrics compared to the covariance-based method. We also show the effectiveness of the proposed frameworks for speech enhancement, achieving performance on par with the baseline method.
#### MiDashengLM-Gen: Unified Audio Scene Generation via LLM-Driven Autoregressive Flow Matching
 - **Authors:** Xingwei Sun, Heinrich Dinkel, Gang Li, Jiahao Mei, Yadong Niu, Zerui Han, Yuepeng Jiang, Jiahao Zhou, Lichun Fan, Jian Luan
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2608.11804

 - **Pdf link:** https://arxiv.org/pdf/2608.11804

 - **Abstract**
 Generating coherent audio scenes that simultaneously blend speech, music, and sound effects remains a significant challenge. Current approaches typically rely on a disjointed pipeline where a frozen, decoupled text encoder feeds a separate audio decoder, limiting cross-modal optimization and leading to poor speech intelligibility. To overcome these limitations, we introduce MiDashengLM-Gen, an end-to-end framework that couples a pre-trained Large Language Model (LLM) with per-token conditional flow matching for autoregressive, variable-length mixed-audio scene generation. MiDashengLM-Gen represents a first approach for general text-to-audio generation with one end-to-end trained model. Empirical evaluations demonstrate that MiDashengLM-Gen drastically improves speech intelligibility over existing unified models. On the Seed-TTS benchmark, English Word Error Rate (WER) drops from 12.15% to 2.79%, approaching the performance of dedicated Text-to-Speech (TTS) systems (1.24%). Furthermore, the framework extends effectively to multilingual settings, yielding highly competitive multilingual WERs compared to existing baselines. Lastly, the model maintains competitive mixed-audio generation quality on the MECAT benchmark. Code and checkpoints are available at this https URL and this https URL, and the demo page is available at this https URL.
#### On-Policy Self-Distillation for Multi-Dialect ASR: Mastering Dialects, Retaining Mandarin
 - **Authors:** Shuiyuan Wang, Bingshen Mu, Pengshen Zhang, Chengyou Wang, Yujie Liao, Chengdong Liang, Binbin Zhang, Qiangze Feng, Lei Xie
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2608.11898

 - **Pdf link:** https://arxiv.org/pdf/2608.11898

 - **Abstract**
 Recent large-scale ASR models already achieve strong Mandarin recognition accuracy and have some ability to recognize Chinese dialects. However, their dialect recognition accuracy is still limited in real-world speech. Direct dialect adaptation can lower dialect CER, but it may also raise Mandarin CER. We therefore study how to adapt a capable ASR model to improve multi-dialect recognition without degrading Mandarin recognition. We adopt an adaptation pipeline where continual pre-training (CPT) and dialect supervised fine-tuning (SFT) provide a strong foundation, and On-Policy Self-Distillation (OPSD) serves as the final refinement. OPSD addresses the train--test mismatch in autoregressive ASR by training the student model on its own decoded prefixes while a frozen teacher, conditioned on the reference transcript as privileged context, provides soft token-level targets. This replaces hard cross-entropy updates on dialect data with distillation, preserving Mandarin ability while refining dialect recognition. We instantiate the framework with Qwen3-ASR-1.7B and evaluate it on public and internal Mandarin and dialect test sets. Under matched refinement data and schedule, OPSD improves dialect recognition without raising Mandarin CER, whereas continued teacher-forced fine-tuning increases Mandarin CER. We will release the model weights and evaluation scripts.
#### The SLT 2026 SmartGlasses Challenge: Benchmarking Egocentric Multi-Talker Speech Recognition and Understanding with Audio-Language Models
 - **Authors:** Dehui Gao, Zhixian Zhao, Zhennan Lin, Yujie Liao, Yuhang Dai, Yike Zhu, Longshuai Xiao, Hui Bu, Xin Xu, Xie Chen, Shuai Wang, Liumeng Xue, Zhonghua Fu, Jun Du, Eng-Siong Chng, Jun Zhou, Lei Xie
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2608.12034

 - **Pdf link:** https://arxiv.org/pdf/2608.12034

 - **Abstract**
 Recent advances in large language models (LLMs) and multimodal LLMs (MLLMs) have created new opportunities for wearable speech interfaces, with smart glasses providing an egocentric platform for continuous audio sensing and assistance. However, speech recognition and understanding in this setting remain challenging because of dynamic acoustic conditions, speaker overlap, and the spatial ambiguity introduced by wearer-centered recording geometry. To support systematic evaluation in this setting, we introduce the IEEE SLT 2026 SmartGlasses Challenge for egocentric multi-speaker speech processing. The challenge consists of two tracks, Dyadic Dialogue Understanding and Multi-party Meeting Understanding, and jointly evaluates Time-Stamped Speaker-Attributed Automatic Speech Recognition (TSA-ASR) and Spoken Language Understanding (SLU). It is built on a 106-hour four-channel egocentric speech dataset containing 714 sessions collected in real-world scenarios. This paper describes challenge tasks, dataset construction, submissions, and summarizes the main findings from the shared evaluation. The results show that heavy speaker overlap remains a major factor affecting TSA-ASR performance, while paralinguistic acoustic understanding continues to be difficult for current audio-language models in complex SLU settings. Further details can be found on the official challenge website.
#### Rethinking Language Model-Based Generative Speech Enhancement in the Latent Space of a Neural Audio Codec
 - **Authors:** Yihui Fu, Zhengyang Li, Tim Fingscheidt
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2608.12082

 - **Pdf link:** https://arxiv.org/pdf/2608.12082

 - **Abstract**
 Language model (LM)-based speech enhancement (SE) has recently emerged rapidly using latent space features of neural audio codecs (NACs). In this paper, first, we present a unified framework covering six popular LM-based generative SE modeling paradigms based on discrete/continuous latent NAC features: discrete or continuous autoregressive (D/CAR) SE, discrete or continuous non-autoregressive (D/CNAR) SE, discrete diffusion (DDiff) SE, and continuous flow matching (CFM) SE. Second, we are the first to compare their performance in a unified experimental setup and synopsis with diverse intrusive and non-intrusive metrics, enabling a fair and comprehensive evaluation. Third, we propose a fine-tuning strategy with auxiliary losses on reconstructed speech to improve both intrusive and non-intrusive metrics. Trained and evaluated on URGENT 2025 Speech Enhancement Challenge data splits, all continuous-domain paradigms excel their discrete-domain counterparts. The overall best approach turns out to be CNAR. We further show that our proposed auxiliary loss fine-tuning strategy helps to improve DNSMOS, NISQA, PESQ, and POLQA consistently in all six paradigms.
#### Cloud-Boosted Low-Compute Multi-Channel Speech Enhancement
 - **Authors:** Xulin Fan, Juan Azcarreta, Ashutosh Pandey, Jesus Alvarez, Ke Tan, Jacob Donley, Ritwik Giri, Buye Xu
 - **Subjects:** Subjects:
Sound (cs.SD); Machine Learning (cs.LG); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2608.07423

 - **Pdf link:** https://arxiv.org/pdf/2608.07423

 - **Abstract**
 Low-latency, low-compute speech enhancement is essential for wearable devices with real-time communication requirements, but strict computational constraints significantly limit on-device performance. Knowledge Boosting has been proposed as an effective approach to improve edge model performance by leveraging a more capable server-side model, but performance gains for speech enhancement have been limited. We propose a collaborative framework incorporating three techniques: (1) delayed server output as additional input, (2) layerwise feature boosting that transfers intermediate server representations to guide edge inference, and (3) collaborative multichannel Wiener filtering, which fuses weighted covariance matrices estimated from both server and edge models for improved beamforming. Experimental results demonstrate that the proposed collaborative framework significantly outperforms the edge-only baseline with minimal additional computational overhead.
#### Luna-TTS Family Technical Report
 - **Authors:** Feng Yin, Shuai Shi, Junjie Zheng, Kechenying Zhou, Yiqiu Wang, Chenyang He, Qiuhua Jiang, Mengxiao Bi, Yanmin Qian, Mingxin Chen, Xun Gong, Tianteng Gu, Bing Han, Peng Jiang, Chenda Li, Haiyang Sun, Han Wang, Wei Wang, Yi Wang, Leying Zhang, Wangyou Zhang, Chushu Zhou
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2608.11593

 - **Pdf link:** https://arxiv.org/pdf/2608.11593

 - **Abstract**
 Modern text-to-speech (TTS) is dominated by autoregressive (AR) codec language models, whose left-to-right decoding brings latency that grows with utterance length, error accumulation along the committed prefix, and an artificial generation order imposed on the Residual Vector Quantization (RVQ) token grid. We propose Luna-TTS Family, diffusion-language-model-based TTS systems pretrained on 1 million hours of speech across Chinese, English, Japanese, and Korean. The family is built by progressive adaptation of a pretrained AR text LLM, from causal to bidirectional and finally to block-causal attention, and comprises two variants sharing a single tokenizer, data pipeline, and 0.6B backbone lineage. Luna-TTS is fully non-autoregressive: it generates the entire RVQ token grid in a fixed number of parallel refinement steps, with zero-shot voice cloning and speech editing arising natively as infilling. Luna-TTS Realtime, derived by continual training, is autoregressive over blocks of 32 codec frames (1.28s) while denoising each block in parallel; it supports KV-cached blockwise generation and incremental audio delivery, achieving an end-to-end RTF of 0.0240 and 41.6 ms local first-block latency under the warmed serving protocol. An annealed fine-tuning stage adds explicit control over emotion and non-verbal vocalizations (NVVs), and a reinforcement-learning stage applies GRPO with policy ratios computed over the realized denoising trajectory. On Seed-TTS-Eval, Luna-TTS achieves the best results on all four metrics among compared open-source and commercial systems (0.73 CER / 79.7 SIM on test-zh, 1.49 WER / 76.8 SIM on test-en); on the harder in-the-wild CV3-Eval, it posts the lowest Mandarin and English error rates in our comparison. Against leading commercial systems, it achieves the best results on most objective, model-based, and human-rated metrics for NVV and emotion control.


by Zyzzyva0381 (Windy). 


2026-08-13
