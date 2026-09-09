# Showing new listings for Wednesday, 9 September 2026
Auto update papers at about 2:30am UTC (10:30am Beijing time) every weekday.


阅读 `Usage.md`了解如何使用此repo实现个性化的Arxiv论文推送

See `Usage.md` for instructions on how to personalize the repo. 


Keyword list: ['text-to-speech', 'text to speech', 'tts', 'LLM-based', 'speech', 'voice']


Excluded: []


### Today: 7papers 
#### SETEAB: Multiscale approach with Squeeze-and-Excitation Temporal Enhanced Aware Block for Speech Emotion Recognition
 - **Authors:** Duy Vo, Kiet Anh Hoang, Hao Do
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2609.06101

 - **Pdf link:** https://arxiv.org/pdf/2609.06101

 - **Abstract**
 This paper proposes a novel lightweight multiscale architecture for speech emotion recognition (SER) with three key innovations. First, a depthwise convolution-based subsampling module is introduced to reduce model size and computation while preserving salient emotional cues. Second, a Squeeze-and-Excitation block is integrated to enhance channel-wise recalibration and improve representation robustness. Third, a new Temporal Enhanced Aware Block is designed to strengthen temporal dependency modeling and produce more discriminative emotion-aware features. The proposed model is explicitly designed to jointly improve compactness, recognition performance, and generalizability. Experiments on benchmark SER datasets show that our method achieves higher accuracy with reduced computational complexity, while also delivering stronger cross-corpus performance than most recent advanced networks for SER.
#### Semantic Refinement of Universal Audio Representations through Audio-Description Alignment
 - **Authors:** Lejun Min, Junyu Dai, Ruichen Zheng, Xinyue Fan, Yang Xiang, Huaichen Zhang, Xingchen Song, Yufei Shi, Han Zhao, Xiangang Li
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2609.08429

 - **Pdf link:** https://arxiv.org/pdf/2609.08429

 - **Abstract**
 Universal audio representations must preserve acoustic detail while making high-level concepts accessible across speech, music, environmental sound, and downstream models of different capacities. We study semantic refinement of an acoustically pretrained encoder by adding audio-description alignment to a foundation of BEST-RQ, reconstruction, and CTC. We compare matched control, shuffled-description, and correctly paired trajectories to distinguish correct correspondence from an extra contrastive objective. Each endpoint is frozen and evaluated with a temporal-mean linear probe and a sequence-aware LLM readout, testing whether the refined information is directly accessible and remains useful to a stronger model. Across three paired seeds, correct alignment improves domain-balanced classification by 4.66 points with the linear probe and 2.59 points with the sequence-aware LLM, with positive changes in every domain. Correct pairing accounts for 87% of the linear-probe gain, while the LLM shows its clearest correspondence-specific benefit in captioning. Dense acoustic objectives provide complementary gains under both readouts. A separate 24-layer continuation remains competitive with leading public encoders under the shared evaluator, supporting the recipe beyond the controlled study.
#### Spatial Audio Coding Through Relative Room Impulse Response Estimation
 - **Authors:** Nour Bouayed, Adrien Llave, Jérôme Daniel, Pascal Scalart
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2609.08542

 - **Pdf link:** https://arxiv.org/pdf/2609.08542

 - **Abstract**
 Immersive virtual listening relies on spatial audio technologies such as Higher-Order Ambisonics (HOA), which represent sound scenes as multichannel signals. As the desired spatial resolution increases, so does the number of channels, making efficient compression essential for transmission over bandwidth-limited networks. Moreover, to facilitate deployment by network operators, the target bitrate for immersive audio coding should ideally remain close to the 25 kbps currently allocated to VoLTE audio services. State-of-the-art parametric codecs, such as the recently standardized Immersive Voice and Audio Services (IVAS) codec, achieve compression by transmitting spatial metadata together with a reduced number of transport channels. However, recent studies have shown that IVAS performance degrades on reverberant content, particularly at low bitrates, a limitation that suggests its inability to accurately model room acoustics. In this paper, we propose a novel HOA coding scheme based on the explicit and blind estimation of the Relative Spatial Room Impulse Response (ReSRIR), using a beamformed version of the HOA signal as a reference signal. By exploiting the structure and sparsity of the estimated ReSRIR, we derive an efficient parametric representation for immersive audio coding. Experimental evaluations show that the proposed method achieves higher compression than IVAS in the single-transport-channel regime, while maintaining comparable to slightly better quality.
#### Omni Interaction Agent Technical Report
 - **Authors:** Orantqing, Shengpeng Ji, Junlong Tong, Jialong Zuo, Dongjie Fu, Di Cao, Yangzhuo Li, Shangda Wu, Franz, Evan, Theron Veyra, Changhao Pan, Jingyu Lu, Dongchao Yang, Zhifei Xie, Yang Tan, Xiaoyu Shen, Xiaoda Yang, Wenfu Wang, Teddysun, Steveyves, Zhou Zhao, Bryanytian
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Artificial Intelligence (cs.AI); Machine Learning (cs.LG); Multimedia (cs.MM); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2609.08977

 - **Pdf link:** https://arxiv.org/pdf/2609.08977

 - **Abstract**
 In this work, we present Gander, an end-to-end model that unifies omni perception, realtime interaction, and agentic capabilities within a single framework. In contrast to turn-based conventional paradigms, Gander continuously receives streaming inputs across multiple modalities, including video, speech, and text, enabling natural full-duplex interaction in both everyday conversations and complex workflow-oriented agent scenarios. Users can interrupt the model at any time, while the model can also proactively provide intermediate feedback or ask follow up questions. To natively support these capabilities, Gander adopts two key architectural designs: 1) It employs a Cerebellum-Brain collaborative framework, in which the Cerebellum is responsible for realtime interaction and omni conversational capabilities, while the Brain handles complex reasoning and higher-level agentic tasks. The two components interact continuously through tool calling and the agent orchestration runtime. 2) The Cerebellum is built upon a streaming Thinker-Talker architecture, user inputs and model outputs are further flattened into an ordered token stream at the chunk level, providing a unified representation for low latency, continuous interaction. We conduct comprehensive evaluations of Gander across four dimensions: conversational ability, omni understanding, interactive capability, and agentic intelligence. Internal human evaluations demonstrate that Gander maintains the natural and expressive spoken dialogue capabilities of SOTA open source models while achieving competitive performance in omni interaction. Gander also demonstrates robustness in challenging real-world scenarios, including background noise interference, multi-party interactions, and backchannel communication. We release Gander together with its models, code, and data to facilitate further research and development in the community.
#### Lead Vocal Separation from Vocal Ensemble Mixtures Using Phoneme Alignment
 - **Authors:** Yuma Narahata, Tomohiko Nakamura, Yuki Saito, Hiroshi Saruwatari
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2609.06488

 - **Pdf link:** https://arxiv.org/pdf/2609.06488

 - **Abstract**
 Contemporary a cappella singing often has a lead-and-accompaniment texture, where the lead vocal (Vo) part carries the main melody and the remaining vocal parts provide accompaniment. Owing to their distinct roles, separating the Vo part from the remaining vocal parts, referred to as Vo separation, enables downstream applications such as lyric recognition and minus-one accompaniment generation for vocal ensemble music. Despite these potential applications, acoustic cues for this task are limited because the target and interfering sources are all singing voices with similar acoustic characteristics and often overlap in time, making Vo separation challenging. In this paper, we propose a Vo separation model that uses phoneme alignment of the Vo part as auxiliary information. The proposed model is based on band-split RoPE Transformer (BS-RoFormer), a state-of-the-art music source separation model, and introduces frame-level phoneme labels into its intermediate representations using feature-wise linear modulation (FiLM). Experimental results show that phoneme-alignment conditioning improves Vo separation performance over an audio-only baseline and yields larger average gains than conditioning only on Vo singing/silence activity. Further analysis suggests that the advantage of phoneme-label information is larger when fewer remaining vocal parts share the same phoneme as Vo.
#### ConversationalVoice: Full-Duplex Speech Data from Real Conversations through Source-Faithful Reconstruction and Conversation-Grounded Expansion
 - **Authors:** Richard Yucheng He, Baodong Cao, Chen Xu, Yihang Liu, Tairan Chen
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2609.08147

 - **Pdf link:** https://arxiv.org/pdf/2609.08147

 - **Abstract**
 Full-duplex speech models require training data that preserves turn-taking, overlap, interruption, and backchannel behavior, yet these signals are entangled across speakers in noisy real-world recordings. We present Conversational Voice, a pipeline that converts real two-speaker excerpts into three complementary training-data artifacts. (1) Separation recovers speaker-specific tracks with stable speaker assignments, a canonical transcript, and naturally observed interaction timing. (2) Reconstruction generates speech in matched voices from a fixed source transcript, reconstructs the source turn order, pauses, and overlaps, and adds word-level alignment and delivery instructions. (3) Expansion generates new dialogue constrained by the source context, speakers, and observed interaction pattern. Automatic speaker-verification metrics remain strong across stages, with same-speaker similarity of 0.983-0.991 and positive discrimination margins of 0.199-0.209. Predicted speech quality (NISQA MOS) is 3.56 for separation, 4.41 for reconstruction, and 4.61 for expansion. A Gemini-based automatic evaluator assigns expansion mean scores of 4.94/5 for contextual coherence and 4.80/5 for dialogue naturalness. Expansion and reconstruction exhibit broadly similar interaction profiles; expansion's turn, overlap-event, backchannel, and interruption rates are 4.6%, 8.0%, 13.2%, and 16.0% lower, respectively. We evaluate data properties only; downstream gains in full-duplex model training remain for future work.
#### From Scores to Evidence: Auditable Decisions Can Improve Speech Deepfake Detection
 - **Authors:** Mengzhe Geng, Yujia Lu, Patrick Littell, Manuela Kunz, Xie Chen
 - **Subjects:** Subjects:
Sound (cs.SD); Computation and Language (cs.CL); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2609.08899

 - **Pdf link:** https://arxiv.org/pdf/2609.08899

 - **Abstract**
 Speech deepfakes can mimic a speaker's voice convincingly enough to deceive listeners and automated systems. This has driven strong progress in speech deepfake detection, but most detectors still end with one score per utterance. That score is useful for ranking systems, yet it says little about why a borderline item should be trusted, deferred, or reviewed. Two utterances can fall in the same score band for different reasons, for example because passive and retrieval evidence disagree or because the keyed probe is unavailable. We ask whether the final decision can remain scalar without discarding that provenance. We answer this question with an auditable decision record that carries four aligned cues into a late calibration step: a passive detector score, a conditional keyed-probe score on a marked derivative, retrieval support, and a speaker-profile margin, together with explicit disagreement coordinates. On the 4,080-example ASVspoof 5 Track 1 matched subset, the fixed retrieval-augmented rule improves on retrieval-only evidence, from 15.84 percent to 11.91 percent EER, and late calibration over the full record reaches 8.43 percent EER. At a 33.75 percent review budget, the exposed cue union covers 82.85 percent of the calibrated model's errors. The best passive WavLM run still reaches 6.71 percent EER, so we do not present the decision record as a stronger standalone detector. Its contribution is to preserve the evidence behind each surfaced utterance while still producing one operating score for thresholding and review.


by Zyzzyva0381 (Windy). 


2026-09-09
