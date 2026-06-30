# Showing new listings for Tuesday, 30 June 2026
Auto update papers at about 2:30am UTC (10:30am Beijing time) every weekday.


阅读 `Usage.md`了解如何使用此repo实现个性化的Arxiv论文推送

See `Usage.md` for instructions on how to personalize the repo. 


Keyword list: ['text-to-speech', 'text to speech', 'tts', 'LLM-based', 'speech', 'voice']


Excluded: []


### Today: 9papers 
#### Improving Large-Scale Weakly Supervised ASR by Filtering and Selection
 - **Authors:** Kohei Matsuura, Masato Mimura
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Computation and Language (cs.CL)
 - **Arxiv link:** https://arxiv.org/abs/2606.28728

 - **Pdf link:** https://arxiv.org/pdf/2606.28728

 - **Abstract**
 Leveraging large-scale weakly supervised datasets is crucial to train robust end-to-end automatic speech recognition (ASR) models. However, such datasets often contain noisy labels and lack domain specificity, limiting their effectiveness. To address these issues and make better use of weakly supervised datasets, we propose a novel training approach incorporating data filtering and selection. Our approach consists of three steps: pretraining on the entire dataset, continued pretraining on a filtered subset based on character error rate (CER), and fine-tuning on a small number of acoustically similar samples to the target domain, selected from the filtered subset. In experiments with a 90,000-hour weakly supervised Japanese dataset, the proposed filtering and selection methods synergistically reduced CER by up to 6.4% and 4.0%, respectively, even though these steps reused training samples already used in the first pretraining step.
#### CTC-Seeded Token Edit Refinement for Non-Autoregressive Speech Recognition
 - **Authors:** Wanting Huang, Weiran Wang
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2606.28732

 - **Pdf link:** https://arxiv.org/pdf/2606.28732

 - **Abstract**
 Non-autoregressive automatic speech recognition (ASR) enables parallel decoding, but many refinement-based methods begin from random, fully masked, or fixed-length token sequences, requiring multiple iterations to reconstruct the complete transcript. We instead formulate ASR decoding as a variable-length edit refinement of a greedy connectionist temporal classification (CTC) hypothesis. An acoustic-conditioned Edit Flow decoder operates directly on the collapsed CTC hypothesis, predicting insertion, deletion, and substitution operations in parallel. The Edit Flow decoder is jointly trained with a CTC model using a continuous-time discrete diffusion loss. During inference, we find that just two edit steps yield substantial Word Error Rate (WER) reductions, and classifier-free guidance (CFG) further enhances recognition quality by focusing the model on audio features. We also constrain edit proposals using CTC confidence to improve accuracy. Finally, ablation studies validate our design choices, while decoder pretraining and pretrained encoder integration yield significant additional performance gains.
#### GigaSpeechBench: A Real-World Multilingual Speech-to-Text Benchmark
 - **Authors:** Yujie Tu, Yifan Yang, Tianrui Wang, Yanqiao Zhu, Guodong Lin, Mingchen Shao, Haoran Wang, Junzhe Liu, Yuxiang Fu, Yizhou Peng, Changsong Liu, Peng Wang, Zhikang Niu, Yunchong Xiao, Haolong Zheng, Xiuwen Zheng, Xulin Fan, Wei-Qiang Zhang, Lei Xie, Longbiao Wang, Eng-Siong Chng, Jiajun Zhang, Kele Xu, Jianwei Yu, Binbin Zhang, Jiayu Du, Wupeng Wang, Zhigao Chen, Yunlong Wu, Guoguo Chen, Xipeng Qiu, Mark Hasegawa-Johnson, Kai Yu, Zhifu Gao, Xiangang Li, Xie Chen
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2606.28884

 - **Pdf link:** https://arxiv.org/pdf/2606.28884

 - **Abstract**
 While modern ASR systems achieve low error rates on high-resource benchmarks, such performance often overestimates real-world robustness. Existing evaluations address challenges in isolation, lacking a unified benchmark for domain terminology, age variation, dialects, accents, and low-resource languages, particularly across the Middle East and Southeast Asia, representing over one billion under-evaluated speakers. To address this gap, we introduce GigaSpeechBench, a comprehensive multilingual and multidimensional in-the-wild ASR & AST benchmark comprising 680 hours of human-annotated speech. It features five modules: (1) 12 low-resource Middle Eastern and Southeast Asian languages, plus challenging Japanese and Korean; (2) 6 Chinese dialects; (3) 6 English accents; (4) dense terminology across 12 vertical domains for Chinese and English; and (5) older adult and child speech. We further provide human-annotated Chinese and English translations for 11 languages to support AST evaluation. Extensive evaluations of leading foundation models and commercial APIs reveal significant performance degradation in these challenging settings, exposing critical evaluation blind spots.
#### VeRe-Flow: Guiding Flow Matching toward Clean Speech via Velocity Contrastive Regularization and Representation Alignment for Noise-Robust Bandwidth Expansion
 - **Authors:** Sujin Koo, Sangyoon Kim, Ji Sub Um, Hoirin Kim
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2606.29450

 - **Pdf link:** https://arxiv.org/pdf/2606.29450

 - **Abstract**
 Noise-robust bandwidth expansion aims to reconstruct high-fidelity wideband speech from noisy low-resolution inputs. While flow matching has shown strong performance in speech generation, accurately recovering clean speech from noisy inputs remains challenging due to the ambiguity of velocity estimation under noise. In this work, we propose VeRe-Flow, a clean-guided flow matching framework that introduces multi-level clean supervision to guide the generative process toward clean speech. At the velocity level, we introduce velocity contrastive regularization, which attracts the predicted velocity toward the clean trajectory while repelling it from noisy trajectories. At the representation level, we incorporate representation alignment that aligns intermediate features with clean self-supervised learning representations. The results demonstrate that the proposed method achieves the lowest LSD and highest DNSMOS OVRL among all baselines, and the highest MOS among generative baselines.
#### DTM-Codec: Dynamic Token Masking for VFR Speech Coding with Efficient Boundary Selection
 - **Authors:** Hoyeol Sohn, Juhan Nam
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2606.29480

 - **Pdf link:** https://arxiv.org/pdf/2606.29480

 - **Abstract**
 Variable frame rate (VFR) coding has recently emerged in neural speech codecs, allocating fewer frames to redundant regions and more frames to rapidly changing speech. VFR must transmit side information about retained time steps, but prior gains are either not rigorously addressed or often minor once these overhead bits are included in total bitrate. We present Dynamic Token Masking (DTM)-Codec, a neural speech codec that demonstrates clear gains over fixed-frame-rate baselines under a strict matched-total-bitrate protocol. DTM keeps selected encoder tokens, fills masked positions with a learned <MASK> embedding, and transmits a binary keep-mask for position-aware decoding. We further introduce Path Length Equalization (PLE), a linear-time boundary selector for VFR coding that yields well-spread adaptive segments with negligible overhead. Across operating points, DTM-Codec broadly improves reconstruction quality and intelligibility over fixed-frame-rate baselines.
#### VIB-AVSR: Variational Information Bottleneck for Noise-Robust LLM-Based Audio-Visual Speech Recognition
 - **Authors:** Piyush Arora, Navlika Singh, Umberto Cappellazzo, Stavros Petridis, Maja Pantic
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Computer Vision and Pattern Recognition (cs.CV); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2606.29632

 - **Pdf link:** https://arxiv.org/pdf/2606.29632

 - **Abstract**
 Audio-Visual Speech Recognition takes two input modalities, acoustic and visual streams, where visual information from lip movements aids recognition when audio is noisy. Recently, LLM-based AVSR models have emerged as a promising paradigm by connecting pre-trained audio-visual encoders to an LLM, achieving strong results in clean conditions. However, these models are predominantly optimized for clean acoustic conditions, with limited attention to making the LLM backbone robust to noise. No explicit mechanism is employed to produce stable representations under corrupted audio, leading to performance degradation in noisy environments. To address this, we propose VIB-AVSR, which integrates Variational Information Bottleneck layers at targeted positions within the LLM backbone to regularize representations. VIB-AVSR reduces degradation under noisy conditions across multiple SNR levels and noise types, without requiring architectural modifications or additional training data.
#### MeloDISinger: Melody-Aware & Duration-Preserving Singing Voice Editing with Audio Infilling
 - **Authors:** Yoonjeong Park, Jaekwon Im, Juhan Nam
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2606.30580

 - **Pdf link:** https://arxiv.org/pdf/2606.30580

 - **Abstract**
 Text-based singing voice editing (SVE) aims to revise sung lyrics while preserving the original melody, total duration, and non-edited regions. In this paper, we propose MeloDISinger, a flow-matching-based SVE model for melody-aware and duration-preserving editing. Its core module, MeloDRP, predicts fixed-budget duration ratios, enabling explicit span-wise duration control. For melody-aware duration allocation, MeloDRP fuses phonetic cues with pseudo-MIDI melodic context through cross-attention, while temporal-overlap supervision encourages soft phoneme--note correspondences. We further use a flow-matching mel decoder for audio infilling to synthesize edited regions while preserving surrounding context. In addition, we introduce a duration-aware edited-lyric generation pipeline using WhisperX and an LLM to construct feasible evaluation scenarios. Experiments demonstrate state-of-the-art performance in both objective and subjective evaluations.
#### Preference-ASR: A Preference-Aware Test Set for Benchmarking ASR in the Era of Speech LLMs
 - **Authors:** Nithin Rao Koluguri, Sasha Meister, Nikolay Karpov, Piotr Zelasko, Desh Raj, Jagadeesh Balam, Boris Ginsburg
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2606.29534

 - **Pdf link:** https://arxiv.org/pdf/2606.29534

 - **Abstract**
 Popular ASR test sets adopt inconsistent conventions for numbers, disfluencies, entities, and casing, while standard normalizers erase the format distinctions users care about. Current benchmarks therefore cannot measure whether a model follows user preferences for output style. We introduce PreferenceASR, a test set evaluating ASR systems on their ability to follow natural-language preference instructions across four categories: normalization, entities, disfluencies, and case. Built from seven open-source corpora via a two-stage LLM-assisted pipeline with human verification, it is evaluated with a preference-aware normalizer that selectively skips steps matching the active instruction. Benchmarking four models shows rankings shift across preference types, exposing quality differences traditional evaluation obscures. We publicly release the dataset.
#### OLIVE: View-Augmented Latent Prediction with Waveform Reconstruction for Speech SSL
 - **Authors:** Karl El Hajal, Mathew Magimai.-Doss
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Machine Learning (cs.LG); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2606.30356

 - **Pdf link:** https://arxiv.org/pdf/2606.30356

 - **Abstract**
 We propose Online Latent prediction with Invariant Views and rEconstruction (OLIVE), a self-supervised speech representation learning framework that jointly optimizes analysis and synthesis objectives. OLIVE combines view-augmented masked latent prediction with waveform reconstruction under a unified objective. Reconstruction constrains early encoder features to retain signal-level information, while masked latent prediction shapes later contextual representations toward invariance for robust downstream performance. We show that these objectives enable representations that support a broad range of tasks. In particular, OLIVE improves results on generation and speaker tasks, maintains competitive performance on recognition and semantic tasks, and improves waveform reconstruction.


by Zyzzyva0381 (Windy). 


2026-06-30
