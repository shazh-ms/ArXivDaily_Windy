# Showing new listings for Monday, 25 May 2026
Auto update papers at about 2:30am UTC (10:30am Beijing time) every weekday.


阅读 `Usage.md`了解如何使用此repo实现个性化的Arxiv论文推送

See `Usage.md` for instructions on how to personalize the repo. 


Keyword list: ['text-to-speech', 'text to speech', 'tts', 'LLM-based', 'speech', 'voice']


Excluded: []


### Today: 5papers 
#### UniSRM: A Unified Speech Reward Model for Reasoning-Based Fine-grained Assessment
 - **Authors:** Yuanyuan Wang, Dongchao Yang, Yayue Deng, Zhiyong Wu, Yiwen Guo, Helen Meng, Xixin Wu
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2605.23261

 - **Pdf link:** https://arxiv.org/pdf/2605.23261

 - **Abstract**
 Evaluating speech generation still relies heavily on human judgments, such as Mean Opinion Score (MOS), which are expensive, subjective, and difficult to reproduce at scale. While a few recent studies have begun to explore AudioLLM-based judge models, existing efforts typically target only a narrow set of scenarios (e.g., utterance-level quality or single-turn dialogue) and provide limited coverage of diverse speech generation tasks and evaluation dimensions. In this work, we propose UniSRM, a unified speech reward model that can support multi-dimensional, interpretable reward signals with reliable reasoning. To support training and evaluation, we introduce UniSRM-Data and UniSRM-Bench, covering speech evaluation tasks from utterance-level quality to context-level coherence. Based on this dataset, we present the unified speech reward model, UniSRM, with a two-stage pipeline that enables reasoning-based fine-grained assessment. Furthermore, we introduce Reasoning-Consistent Rewards to improve the reliability of the reasoning process. Experiments show that UniSRM delivers more reliable and human-aligned judgments across a broad range of speech evaluation tasks, offering a practical foundation for scalable and unified evaluation of speech quality.
#### StepAudio 2.5 Technical Report
 - **Authors:** Bin Lin, Bo Zhao, Boyong Wu, Chao Yan, Chen Wu, Cheng Yi, Chengyuan Yao, Daijiao Liu, Fei Tian, Feng Tian, Haiyang Sun, Haoyang Zhang, Jiangjie Zhen, Jinglan Gong, Jun Chen, Li Xie, Peilin Li, Peng Yang, Pengfei Tan, Qingjian Lin, Runze Li, Shenghua Hu, Siyi Zhou, Wenwen Qu, Xiangyu Li, Xiangyu Tony Zhang, Xuerui Yang, Yang Yang, Yechang Huang, Yu Fu, Yuchu Luo, Yuxin Li, Yuxin Zhang, Zhengyan Sheng, Brian Li, Chang Zeng, Changlin Zhang, Chen Geng, Chenghao Dong, Chengli Feng, Dan Zhou, Danni Wan, Di Chen, Die Zhang, Dongqing Pang, Guanglong Yang, Guoqiang Hu, Huangxi Zhu, Jianzheng Gao, Jinghua Liang, Jinmei Wan, Junjie Yuan, Kang An, Lei Lei, Limin Zhong, Lun Cai, Mengqiang Ren, Min Xu, Mingliang Li, Mingxiao Li, Na Wang, Qiang Tong, Qiaoling Huang, Qingfu Du, Rui Wang, Shengchen Zhou, Shi Qiu, Shihao Peng, Shiliang Yang, Siqi Tu, Tianjiao Deng, Ting Xu, Tong Wang, WeiMing Niu, Wuxun Xie, Xianwei Zhang, Xianyu Feng, Xiaojia Liu, Xing Chen, Xiongbin Wu, Yan Wu, Yang Li, Yi Liu, Yifan Zhang, Yile Liu, Yongshen Long, Yu Luo, Yuanhao Ding, Yuhao Wang, Yuhe Yin, Yunfang Xu, Yuxiang Yang, Zhiguo Huang, Zhiyue Wu, Zichao Li, Zichao Zhou, Daxin Jiang, Future Li, Gang Yu, Xiangyu Zhang, Yibo Zhu
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2605.23463

 - **Pdf link:** https://arxiv.org/pdf/2605.23463

 - **Abstract**
 Unified audio-language modeling has emerged as a prominent trend in modern speech systems, promising to bring the reasoning capabilities of large language models to auditory tasks. However, existing unified foundations often struggle to match the depth of specialized systems across automatic speech recognition (ASR), text-to-speech synthesis (TTS), and realtime spoken interaction. Bridging this gap remains an open challenge. This report presents StepAudio 2.5, a unified audio-language foundation model that matches or exceeds specialized systems across all three capabilities. Rather than treating these tasks as architecturally distinct, we operate on the premise that once text and audio share a multimodal representational space, task specialization becomes a matter of operational regimes: data construction, optimization targets, and decoding constraints. Guided by this insight, we advance the post-training paradigm from standard supervised learning to task-tailored Reinforcement Learning from Human Feedback (RLHF), using it as the primary mechanism to define complex optimization targets. We leverage this RLHF-centric alignment, alongside specialized decoding, to shape a shared backbone into three distinct operational modes. Concretely, the ASR branch advances transcription efficiency via verifiable multi-token decoding; the TTS branch achieves controllable, expressive synthesis through preference-based RLHF and context-rich supervision; and the Realtime branch realizes low-latency, persona-consistent dialogue via generative reward modeling within an RLHF framework. On standard benchmarks, StepAudio 2.5 achieves state-of-the-art results across ASR, TTS, and Realtime, demonstrating that a singular audio-language foundation can successfully internalize the distinct deployment objectives of speech understanding, generation, and live interaction.
#### Word-Level Modeling with Alignment-Aware Acoustic Fusion for Text-Assisted Intelligibility Prediction in Listeners with Hearing Loss
 - **Authors:** Kazushi Nakazawa
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2605.23604

 - **Pdf link:** https://arxiv.org/pdf/2605.23604

 - **Abstract**
 We address text-assisted speech intelligibility prediction for hearing-impaired listeners in CPC3. Although the target is a sentence-level percentage, it is determined by reference-word recognition outcomes. We formulate prediction as reference-conditioned word-level correctness modeling: a frozen Whisper encoder analyzes degraded speech, a teacher-forced decoder conditions on the canonical transcript, and sentence intelligibility is obtained by averaging predicted correctness probabilities over valid reference words. To complement transcript-conditioned decoder states, we add a word-aligned local acoustic branch based on character-level cross-attention alignment and an utterance-level global acoustic branch for calibration. On the official evaluation set, the decoder baseline obtains RMSE 24.92 and correlation 0.795, while joint fusion improves to incorrect-word F1 0.778, MCC 0.626, correlation 0.806, and RMSE 24.39. A similar trend with Whisper medium suggests that the gain comes from prediction granularity and alignment-aware fusion.
#### Frame-Aligned Fusion of Canary and WavLM for Non-Intrusive Intelligibility Prediction of Hearing-Aid-Processed Speech
 - **Authors:** Kazushi Nakazawa
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2605.23619

 - **Pdf link:** https://arxiv.org/pdf/2605.23619

 - **Abstract**
 Non-intrusive intelligibility prediction estimates how well hearing-impaired listeners understand hearing-aid-processed speech without a clean reference. We study this task in the 3rd Clarity Prediction Challenge using two frozen speech encoders, Canary and WavLM. The central question is not only whether complementary pretrained representations should be combined, but where their interaction should occur. We compare single-backbone baselines, uniform score averaging, pool-late fusion, cross-attention, frame-aligned fusion, and reverse alignment under a shared left/right-preserving binaural framework. Among the compared systems, the best model temporally prepares WavLM with a learnable strided convolution and fuses it with Canary on the coarser Canary timeline before pooling, reaching Eval RMSE 24.96$\pm$0.06 and Eval Corr 0.796$\pm$0.001. Severity, enhancement-system, layer-window, and temporal-shift analyses indicate that coarse local temporal correspondence before pooling is a useful inductive bias for this task.
#### Natural Yet Challenging to Detect: Robust In-the-Wild TTS through EMA and Dual-Scoring Prompt Selection -- Submission for WildSpoof 2026 TTS Track
 - **Authors:** Renhe Sun, Jiayi Zhou, Haolin He, Yueying Feng, Jian Liu
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2605.23859

 - **Pdf link:** https://arxiv.org/pdf/2605.23859

 - **Abstract**
 In this technical report, we describe our submission for the WildSpoof Challenge TTS Track: Text-to-Speech with In-the-Wild Data. We introduce F5-TTS-DPS, a model built upon the F5-TTS architecture. Our approach integrates Exponential Moving Average (EMA) into supervised fine-tuning to stabilize training and improve generalization. To enhance synthesis fidelity, we leverage large language models (LLMs) and large audio language models (LALMs) for dual-scoring prompt selection, filtering reference audio and text prompts to ensure quality while addressing alignment issues in noisy datasets. Experimental evaluation demonstrates that F5-TTS-DPS achieves strong performance with UTMOS of 3.20 and speaker similarity of 0.51 on the development set. More importantly, our model achieves the best a-DCF scores of 0.1582, 0.5233, and 0.2562 across three advanced SASV systems among all submissions, indicating our synthesized speech is the most difficult to detect and exhibits the highest degree of naturalness and authenticity. Combined with competitive WER performance, these results validate the effectiveness of our approach in generating natural-sounding speech with strong spoofing capabilities.


by Zyzzyva0381 (Windy). 


2026-05-25
