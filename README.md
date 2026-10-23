# Showing new listings for Friday, 23 October 2026
Auto update papers at about 2:30am UTC (10:30am Beijing time) every weekday.


阅读 `Usage.md`了解如何使用此repo实现个性化的Arxiv论文推送

See `Usage.md` for instructions on how to personalize the repo. 


Keyword list: ['text-to-speech', 'text to speech', 'tts', 'LLM-based', 'speech', 'voice']


Excluded: []


### Today: 4papers 
#### HC-AINet: An Auditory-Inspired Network with Harmonic Compensation for Low-Complexity Speech Enhancement
 - **Authors:** Nan Li, Xiaoxiong Wu, Huawei Liu, Ziping Zhao
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2610.30818

 - **Pdf link:** https://arxiv.org/pdf/2610.30818

 - **Abstract**
 Speech enhancement on resource-constrained devices requires noise suppression with limited computation and minimal loss of spectral detail. Auditory-inspired subband processing reduces computation, but its smooth gains have limited ability to distinguish harmonic peaks from neighboring noise. We propose a harmonic-compensated auditory-inspired network (HC-AINet) for low-complexity speech enhancement. A triangular equivalent rectangular bandwidth (ERB) filterbank supplies compact features to a lightweight network comprising inverted-residual units, temporal Conformer layers, and parallel frequency-gated convolutions. The network predicts global and harmonic gains together with fusion weights. A harmonic-location map derived from the noisy spectrum confines the harmonic contribution to selected bins on the original frequency grid, while the global branch controls the broader spectral envelope. The compensated magnitude, noisy phase, and a magnitude-derived minimum-phase cue jointly condition a compact phase estimator for waveform reconstruction. On VoiceBank-DEMAND, the 64-band model achieves a PESQ of 3.20 and a STOI of 0.95 with approximately 0.27 million parameters.
#### Lip-Focused Audio-Visual Consistency Detects Speech Content Manipulation beyond Coarse Synchronization
 - **Authors:** Jian Zong, Chenghan Lin, Meng Liu, Zhao Xu, Junyu Wang, Siyuan Zhang, Peiyuan Jiang, Longbiao Wang, Jianwu Dang
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2610.30872

 - **Pdf link:** https://arxiv.org/pdf/2610.30872

 - **Abstract**
 Speech-content manipulation changes what a visible person appears to say by replacing the speech track and, in stronger attacks, regenerating mouth motion to match the forged utterance. Such attacks are difficult to expose with texture artifacts or coarse lip-sync scores because modern retalking models explicitly optimize audio-visual synchrony. We test the narrower hypothesis that speech replacement still perturbs the local relationship between acoustic dynamics and mouth motion after speaker identity or coarse synchrony is controlled. To isolate this signal, we introduce SyncSpeech, an LRS3-derived benchmark containing real videos, audio-splicing cheapfakes and Wav2Lip retalking attacks under same-speaker and different-speaker replacement. We then introduce C-MIND, a lip-focused detector that aligns audio and mouth-motion representations at frame level and models their agreement and disagreement without transcripts, automatic speech recognition features or an unmanipulated reference. On SyncSpeech, pretrained SyncLip-C achieves 95.813% scenario-averaged accuracy and 0.9943 AUC, outperforming sentence-level fusion, full-face inputs and a reproduced SyncNet baseline under the reported protocol. When trained on the corresponding FakeAVCeleb training split, C-MIND correctly classifies 278 of 280 samples in the selected unseen-speaker test subset, whereas accuracy falls to 82.775% on unseen Video-Retalking attacks. These results identify lip-focused speech-articulation consistency as a useful but generator-sensitive forensic signal within the evaluated settings.
#### A corpus-only AUC diagnostic for pooled speech-based depression detection
 - **Authors:** Cezary Polak, Milosz Dudek, Daria Hemmerling
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2610.31247

 - **Pdf link:** https://arxiv.org/pdf/2610.31247

 - **Abstract**
 When clinical speech corpora are pooled for evaluation, corpus membership can itself become predictive if class prevalences differ across sources. We derive the AUC of an oracle predictor that uses only corpus identity and empirical prevalence, a diagnostic computable directly from corpus composition. Across five depression corpora in three languages (590 speakers), the corpus-only AUC is 0.697, and corpus identity is predicted from six acoustic and self-supervised representations with 97.5-99.8% accuracy. Permuting depression labels within each corpus removes label structure while preserving prevalences; the residual pooled-minus-LOCO discrepancy then increases across three nested configurations in the same order as the diagnostic. Pooled 5-fold cross-validation also outscores corpus-held-out evaluation across all six representations. Prevalence equalisation substantially reduces the discrepancy for eGeMAPS and WavLM-Base, whereas a sample-size-matched control does not. Corpus-specific base rates therefore create a measurable shortcut opportunity in pooled clinical-speech evaluation, motivating corpus-aware diagnostics alongside pooled performance.
#### Spark-Audio 1.0 Technical Report
 - **Authors:** Chenyue Zhang, Cong Liu, Dingshu Tian, Dong Sun, Genshun Wan, Guanming Meng, Guolong Zhong, Hang Chen, Hankun Yang, Haoming Wang, Haoyang Wang, Hengquan Mei, Huaxin Wu, Jiahao Li, Jiahuan Fan, Jiamin Ma, Jianqing Gao, Jigang Tang, Jin Liu, Jingchen Li, Jinjie Fu, Kaixuan Guo, Kun Song, Li Xu, Li Zhang, Lijuan Liu, Liyan Luo, Ming Sheng, Mingen Zhao, Mingting Hu, Ningqian Wu, Pan Xu, Pei Xu, Qiangyu Tan, Rui Wang, Ruzi Li, Saier Jin, Shibo Xu, Shifu Xiong, Shihao Jiang, Tan Liu, Wei Wu, Wei Xu, Wenyu Wang, Xiangyu Jia, Xiao Li, Xiaowen Duan, Xiaoxue Guo, Xin Chen, Xinyi Liang, Xu Chen, Yanan Song, Yang Xu, Yexia Huang, Yibo Li, Yifan Xu, Yikang Liu, Yikun Li, Yiting Lin, Yi Fang, Yuxin Wang, Zan Feng, Zhenchuan Li, Zhuo Chen, Zichao Zhang, Ziwen Wang
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2610.31292

 - **Pdf link:** https://arxiv.org/pdf/2610.31292

 - **Abstract**
 This paper introduces Spark-Audio 1.0, a speech foundation model trained on domestically produced computing infrastructure. The model accepts text and speech inputs and generates text outputs. It supports multilingual (99 languages) and multi-dialect (202 dialects) speech recognition, speech translation, audio perception and understanding, and both text- and speech-based dialogue, marking a shift from merely ``hearing clearly'' to genuinely ``understanding'' speech. Architecturally, Spark-Audio 1.0 couples a 0.65B-parameter dense audio encoder with a 30B-A3B Mixture-of-Experts (MoE) large language model (LLM). The audio encoder is pretrained with representation-enhanced BEST-RQ and then refined through multi-task supervised training. The audio--LLM is trained through staged pretraining, followed by post-training that combines supervised fine-tuning with on-policy distillation. In total, the model is trained on 13 million hours of audio data together with large-scale text data. We evaluate Spark-Audio 1.0 on mainstream audio and text benchmarks, as well as on in-house test sets drawn from real-world application scenarios. Despite using substantially less audio training data than Qwen3.5-Omni-Flash, Spark-Audio 1.0 achieves comparable performance on most audio and text tasks, outperforms it in multilingual speech recognition, spoken dialogue, and real-world application scenarios, and approaches the performance of larger closed-source models. Nevertheless, there remains room for improvement in instruction following and complex audio reasoning. Finally, the successful training of Spark-Audio 1.0 on domestic hardware, including Huawei Ascend 910B and Hygon BW1000 chips, demonstrates the feasibility of a domestic technical pathway for building speech foundation models.


by Zyzzyva0381 (Windy). 


2026-10-23
