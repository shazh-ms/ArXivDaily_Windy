# Showing new listings for Monday, 8 June 2026
Auto update papers at about 2:30am UTC (10:30am Beijing time) every weekday.


阅读 `Usage.md`了解如何使用此repo实现个性化的Arxiv论文推送

See `Usage.md` for instructions on how to personalize the repo. 


Keyword list: ['text-to-speech', 'text to speech', 'tts', 'LLM-based', 'speech', 'voice']


Excluded: []


### Today: 12papers 
#### SEAM: Shortcut-Aware Real-Time Detection of Scripted vs. Spontaneous Speech for Interview Guardrails
 - **Authors:** Vsevolod (V.)Kovalev, Pranay Manocha
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Machine Learning (cs.LG)
 - **Arxiv link:** https://arxiv.org/abs/2606.06837

 - **Pdf link:** https://arxiv.org/pdf/2606.06837

 - **Abstract**
 Scripted vs spontaneous speech detection is appealing for interview guardrails, but benchmark performance can be inflated by shortcuts tied to corpus identity, channel conditions, and recording artifacts rather than speaking style itself. We present SEAM, a shortcut-aware framework for real-time scriptedness detection that combines uniform preprocessing, seam-aware sampling, non-speech augmentation, and a compact DistilHuBERT backbone. With 8s windows, the model achieves 0.971 +- 0.004 ROC-AUC on an external interview-domain evaluation set. Removing the shortcut-prevention components improves internal held-out metrics but sharply reduces external performance, indicating shortcut learning. Post-training quantization reduces the model footprint to 41.8MB with little loss in external performance. The results demonstrate that robust real-time scriptedness detection depends not only on the backbone, but on shortcut-aware data design and evaluation. We release code and model checkpoints.
#### SpectCount: Spectrotemporal Counting via Synthetic Signals Improves Large Audio Language Models
 - **Authors:** Seonuk Kim, Yonghyeon Jun, Ju Yeon Kang, Jimin Hong, Yoonhyeong Lee, Nam Soo Kim
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Artificial Intelligence (cs.AI); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2606.06907

 - **Pdf link:** https://arxiv.org/pdf/2606.06907

 - **Abstract**
 Large audio language models (LALMs) extend large language models with an audio encoder and large-scale audio data. However, the scarcity of high-quality annotated audio data remains a fundamental bottleneck for scaling. Through probing signal detectability analysis, we identify fine-grained spectrotemporal perceptual weaknesses in a foundation LALM. To address these challenges, we propose Spectrotemporal Counting (SpectCount), a data-efficient fine-tuning approach based on fully synthetic audio signals generated on-the-fly, without relying on real-world audio, annotations, or pretrained generative models. SpectCount not only resolves the observed weaknesses but also improves performance on diverse auditory benchmarks spanning sound, music, and speech, unseen during fine-tuning. These results suggest that weakness-targeted synthetic signals provide a data-efficient path toward enhanced auditory understanding capabilities in LALMs.
#### FSC-Net: Integrating Fast Fourier Convolutions and Progressive Learning for Speech Bandwidth Extension
 - **Authors:** Xinan Chen, Xiaobin Rong, Qinwen Hu, Kai Chen, Jing Lu
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2606.06962

 - **Pdf link:** https://arxiv.org/pdf/2606.06962

 - **Abstract**
 Speech bandwidth extension (BWE) aims to reconstruct high-fidelity wideband audio from narrowband inputs. While recent approaches have made significant progress, they often struggle to reconstruct realistic high-frequency phase and harmonic structures, leading to perceptual artifacts. In this paper, we propose FSC-Net (Full-Spectrum Context Network), a parameter-efficient architecture designed to explicitly model cross-band harmonic dependencies. By integrating Fast Fourier Convolutions (FFCs) into a complex spectral mapping framework, FSC-Net expands its receptive field to the entire spectrum, capturing long-range frequency interactions effectively. To address the ill-posed nature of high-frequency generation, our novel frequency-progressive learning curriculum guides the network to reconstruct spectral details from coarse to fine. Experimental results on the VCTK and unseen EARS datasets demonstrate that FSC-Net delivers consistently strong reconstruction quality and generalization, particularly in the challenging VCTK 4 kHz-to-48 kHz task. Compared to scaled-up baselines, our model attains leading LSD and PESQ scores while maintaining a highly compact parameter footprint (1.54 M).
#### Assessing True Generalisability of Audio-Visual Speech Recognisers
 - **Authors:** Zhaofeng Lin, Stavros Petridis, Maja Pantic, Naomi Harte
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2606.07259

 - **Pdf link:** https://arxiv.org/pdf/2606.07259

 - **Abstract**
 Current Audio-Visual Speech Recognition (AVSR) models achieve near-perfect performance on the standard LRS3 benchmark, raising concerns of adaptive overfitting. To systematically assess true generalisability, we construct a highly controlled, unseen evaluation set subsampled from the massive MultiVSR dataset. Unlike standard out-of-distribution benchmarks, our subset strictly matches the acoustic, visual, and demographic distributions of the LRS3 test set. Evaluating five state-of-the-art architectures reveals a universal performance collapse, proving that current systems fail to generalise even under strictly aligned conditions. Through a fine-grained attribute analysis across seven factors, we isolate the specific drivers of this degradation. Furthermore, we uncover a profound lexical bias, expose distinct error patterns, and surprisingly reveal that audio-visual performance even lags behind audio-only settings. We release our matched test set for future benchmarking.
#### VISA: A Visual Information Strengthened Audio-Reasoning System for the Interspeech 2026 ARC Agent Track
 - **Authors:** Wenming Tu, Jian Gao, Yanru Huo, Yixuan Wang, Jing Peng, Bohan Li, Ziyang Ma, Tao Liu, Shuai Fan, Kai Yu, Xie Chen, Zilong Zheng
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2606.07264

 - **Pdf link:** https://arxiv.org/pdf/2606.07264

 - **Abstract**
 Audio reasoning requires multi-step, evidence-grounded inference over temporally dynamic and acoustically mixed signals, exceeding conventional perception tasks such as ASR or captioning. We present VISA, our submission to the Interspeech 2026 Audio Reasoning Challenge (Agent Track), evaluated via the MMAR Rubrics for correctness and reasoning quality. Under a "LALM as a Tool" paradigm, VISA strengthens large audio language models with auxiliary multi-modal evidence while avoiding heavy orchestration. The system integrates three components: multi-modal feature extraction for complementary audio and acoustic-visual clues, model-voting inference with consistency checking for stable predictions, and fine-grained category-aware routing to resolve disagreements and select rubric-aligned reasoning chains. On the official Agent Track leaderboard, VISA ranks 2nd overall with a 66.23% Rubrics score. It also achieves 77.40% Accuracy, the highest among all systems listed across both the Single Model and Agent tracks.
#### Geometric Second-Order Feature Correlation Learning for Self-Supervised Speech Emotion Recognition
 - **Authors:** Shuanglin Li, Ruxiao Qian, Siyang Song
 - **Subjects:** Subjects:
Sound (cs.SD); Artificial Intelligence (cs.AI); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2606.06550

 - **Pdf link:** https://arxiv.org/pdf/2606.06550

 - **Abstract**
 Self-supervised learning (SSL) yields powerful, context-rich representations for speech emotion recognition (SER), yet aggregating these representations into holistic descriptors remains a bottleneck. Conventional first-order aggregation implicitly assumes feature independence, which overlooks the latent Riemannian geometry and discards higher-order relationships essential to the representational power of the backbone. To address this problem, this paper proposes a novel Second-Order Correlation (SOC) layer. Instead of treating features in isolation, SOC models feature correlations as covariance descriptors to capture synergistic co-occurrence patterns, which serve as discriminative signatures for robust emotion recognition. By mapping these descriptors from the Riemannian manifold to a Euclidean tangent space through Log-Euclidean mapping (LEM), the proposed method preserves geometric integrity while enabling direct linear discriminative learning. Extensive experiments on the ESD and RAVDESS datasets demonstrate that SOC recovers discriminative information lost in first-order pooling and effectively aggregates high-dimensional SSL features.
#### IRAF: Interference-Resilient Adaptive Fusion for Noise-Robust End-to-End Full-Duplex Spoken Dialogue Systems
 - **Authors:** Tao Zhong, Jiajun Deng, Nikita Kuzmin, Yinke Zhu, Tianxiang Cao, Tristan Tsoi, Zhili Tan, Simon Lui, Xunying Liu
 - **Subjects:** Subjects:
Sound (cs.SD); Artificial Intelligence (cs.AI); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2606.06559

 - **Pdf link:** https://arxiv.org/pdf/2606.06559

 - **Abstract**
 Full-duplex spoken dialogue models allow voice agents to listen and speak concurrently, enabling natural interaction with real-time overlap. However, end-to-end dual-channel models that jointly encode user and agent streams may degrade in realistic acoustic environments: interfering speakers leaking into the user microphone can be encoded as part of the user query, corrupting the LLM's conditioning and causing unstable turn-taking and reduced response quality. We propose Interference-Resilient Adaptive Fusion (IRAF), a lightweight, streaming-compatible module that modulates the contribution of user audio to the LLM frame by frame. IRAF predicts a scalar reliability gate from target-speaker and user audio embeddings and rescales user representations before fusion with agent embeddings. Experiments on MS-MARCO and InstructS2S-200K show consistent gains in response quality and full-duplex interaction under interfering-speaker conditions.
#### Leveraging Soft Distributions of SSL-Derived Discrete Speech Tokens for Downstream Inference
 - **Authors:** Kentaro Onda, Satoru Fukayama, Daisuke Saito, Nobuaki Minematsu
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2606.06806

 - **Pdf link:** https://arxiv.org/pdf/2606.06806

 - **Abstract**
 Discrete speech tokens obtained from self-supervised learning (SSL) models provide efficient data compression while maintaining strong performance, and have been widely used as intermediate representations in various tasks. However, discretization inevitably causes information loss, leading to degraded performance compared with continuous SSL features. In this work, we propose to apply soft token assignment only during downstream inference. This approach preserves the efficiency of hard discretization during training while enhancing the expressiveness of the tokens at inference. The proposed method outperforms conventional hard assignment on both ASR and speech synthesis tasks, and exhibits particularly strong generalizability to out-of-domain data. For ASR of non-native speech, it even surpasses models using continuous SSL features. Moreover, analysis of the resulting representations shows they align more accurately with phonemes compared with conventional hard assignment.
#### VoxCPM2 Technical Report
 - **Authors:** Yixuan Zhou, Guoyang Zeng, Xin Liu, Xiang Li, Renjie Yu, Jiancheng Gui, Jiaheng Wu, Ziyang Wang, Xudong Shen, Runchuan Ye, Zhisheng Zhang, Jiuyang Zhou, Bingsong Bai, Weiyue Sun, Mengyuan Deng, Qundong Shi, Zhiyong Wu, Zhiyuan Liu
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2606.06928

 - **Pdf link:** https://arxiv.org/pdf/2606.06928

 - **Abstract**
 We present VoxCPM2, a https://info.arxiv.org/help/prep#abstractsfully open-source multilingual and controllable speech generation foundation model that extends the hierarchical diffusion-autoregressive modeling paradigm of VoxCPM. VoxCPM2 advances the framework in three key dimensions: (i) capability, by unifying 30 languages, 9 Chinese dialects, natural-language voice design, style-controllable voice cloning, and high-fidelity continuation cloning within a single backbone; (ii) quality, through an asymmetric AudioVAE that encodes at 16 kHz and reconstructs at 48 kHz, enabling implicit super-resolution with high encoding efficiency; and (iii) scale, by jointly scaling the model to 2B parameters and the training data to over 2 million hours of multilingual speech. To support these diverse capabilities within one model, we introduce a unified sequence organization that expresses all generation modes through different arrangements of the same input building blocks, allowing joint training under a single set of parameters and objective. VoxCPM2 achieves state-of-the-art or competitive performance on public zero-shot and instruction-following TTS benchmarks. On our internal 30-language evaluation set, it attains an average WER of 1.68%. These results demonstrate that hierarchical continuous-latent modeling, without relying on any external discrete speech tokenizer, offers a viable and powerful foundation for large-scale multilingual and controllable speech generation. The model weights, fine-tuning code, and inference tools are publicly released under the Apache 2.0 license to foster community research and development.
#### Contrastive Training with LLM-generated Near-Misses for Robust Code-Switching Speech Recognition
 - **Authors:** Tung X. Nguyen, Hieu Minh Truong, Giang-Son Nguyen, Nhu Vo, Wray Buntine, Dung D. Le
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2606.06985

 - **Pdf link:** https://arxiv.org/pdf/2606.06985

 - **Abstract**
 Code-switching (CS), the alternation between multiple languages within a single utterance, remains challenging for Automatic Speech Recognition (ASR). To address this issue, we propose a Point-of-Interest (POI)-aware contrastive training framework that improves recognition at CS-critical regions. We first identify CS spans by adopting POI detection method from literature, then construct acoustically plausible near-miss hypotheses by perturbing POIs in ASR N-best outputs and expanding candidates with a large language model. Hard but plausible negatives are retained through filtering with acoustic, phonemic, and textual constraints. Finally, we fine-tune Whisper-small with LoRA using a POI-weighted cross-entropy anchor objective together with a multi-negative contrastive ranking loss. Experiments on CS-FLEURS (cmn-eng) and ViMedCSS (vie-eng) show consistent reductions of over 2% in both general and CS-aware error rates compared to standard LoRA fine-tuning.
#### dots.tts Technical Report
 - **Authors:** Shi Lian, Changtao Li, Bohan Li, Hankun Wang, Da Zheng, Junfeng Tian, Yufeng Ma, Colin Zhang, Kai Yu
 - **Subjects:** Subjects:
Sound (cs.SD); Artificial Intelligence (cs.AI); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2606.07080

 - **Pdf link:** https://arxiv.org/pdf/2606.07080

 - **Abstract**
 We present this http URL, a 2B-parameter continuous autoregressive text-to-speech (TTS) foundation model that models speech in a continuous latent space. Compared with existing continuous autoregressive models, our key innovations are threefold. First, we train an AudioVAE with multiple objectives to build a semantically structured and prediction-friendly continuous speech space. Second, we use full-history conditioning in the flow-matching head to preserve long-range consistency and reduce drift during generation. Third, we apply reward-free self-corrective post-training to the flow-matching head to further improve robustness and acoustic quality. After being trained on a large-scale multilingual corpus, this http URL achieves the best average performance on Seed-TTS-Eval, with WERs of 0.94%/1.30%/6.60% and SIM scores of 81.0/77.1/79.5 on the zh/en/zh-hard test sets, respectively. Across other benchmarks, this http URL also consistently demonstrates open-source state-of-the-art performance, exhibiting strong generation stability, voice cloning ability, and emotional expressiveness. For efficient inference, we further apply CFG-aware MeanFlow distillation, enabling low-latency speech generation with first-packet latencies of 85/54 ms in output streaming and dual-streaming modes, respectively. To facilitate reproducible research and practical deployment, we release the training and inference code, together with the pretrained, post-trained, and MeanFlow-distilled checkpoints, under the Apache 2.0 license.
#### Mitigating Proxy-to-Wild Domain Gap in Deepfake Speech
 - **Authors:** Xuanjun Chen, Yun-Shing Wu, Wei-Chung Lu, Claire Lin, Haibin Wu, Hung-yi Lee, Jyh-Shing Roger Jang
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2606.07494

 - **Pdf link:** https://arxiv.org/pdf/2606.07494

 - **Abstract**
 Recent neural audio codec-based speech generation (CodecFake) produces highly realistic audio, posing a challenge to existing deepfake countermeasure models. While using codec resynthesized speech (CoRS) as proxy data improves performance, it often suffers from limited generalization. We propose Domain-Shift Feature Augmentation (DSFA), which simulates "in-the-wild" variations by transforming deterministic feature statistics into stochastic distributions during fine-tuning. To evaluate generalization, we further introduce Codec-based Speech Generation Extension Evaluation (CoSG ExtEval) dataset, a more challenging extension of the CoSG Eval (from CodecFake+) dataset, featuring 40 unseen generative models and long-form audio. Experimental results demonstrate that combining a post-trained SSL backbone with DSFA effectively narrows the proxy-to-wild domain gap. This approach achieves state-of-the-art performance across diverse CodecFake attacks in both CoSG Eval and CoSG ExtEval.


by Zyzzyva0381 (Windy). 


2026-06-08
