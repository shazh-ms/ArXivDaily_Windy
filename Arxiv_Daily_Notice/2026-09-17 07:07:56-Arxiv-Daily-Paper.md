# Showing new listings for Thursday, 17 September 2026
Auto update papers at about 2:30am UTC (10:30am Beijing time) every weekday.


阅读 `Usage.md`了解如何使用此repo实现个性化的Arxiv论文推送

See `Usage.md` for instructions on how to personalize the repo. 


Keyword list: ['text-to-speech', 'text to speech', 'tts', 'LLM-based', 'speech', 'voice']


Excluded: []


### Today: 7papers 
#### Encoder Awakening via Adapters: Effective Domain-Adaptive Fine-tuning of Speech-LLMs
 - **Authors:** Mohan Shi, Zilai Wang, Natarajan Balaji Shankar, Kaiyuan Zhang, Eray Eren, Abeer Alwan
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Computation and Language (cs.CL)
 - **Arxiv link:** https://arxiv.org/abs/2609.17981

 - **Pdf link:** https://arxiv.org/pdf/2609.17981

 - **Abstract**
 Speech Large Language Models (Speech-LLMs), typically built from a pre-trained speech encoder, a modality projector, and an LLM fine-tuned with Low-Rank Adapters (LoRA), have shown strong Automatic Speech Recognition (ASR) performance on general-domain speech. However, adapting them to domain-shifted speech, such as child or dialectal speech, remains challenging under limited target-domain data. Given the dominant role of the LLM in Speech-LLMs, with cross-entropy loss applied only at the LLM output, the speech encoder may receive insufficient adaptation to new acoustic conditions. In this paper, we propose Encoder Awakening via Adapters (EAVA), a simple yet effective domain-adaptive fine-tuning method for Speech-LLM-based ASR. First, lightweight adapters are inserted into each encoder layer and trained exclusively, enabling target-domain acoustic knowledge to be incorporated into the encoder while preserving its pre-trained knowledge. Second, the full model is jointly fine-tuned on the target domain with LoRA applied to the LLM. Experiments on three domain-shifted ASR datasets, covering child and dialectal speech, show that EAVA consistently outperforms vanilla fine-tuning and other baselines, achieving new state-of-the-art performance.
#### G-Mamba: Sparse Graph-Guided Mamba for Audio-Visual Speech Enhancement
 - **Authors:** Guo-Ruei Tseng, Hung-Shin Lee, Hsin-Min Wang, Berlin Chen
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Computation and Language (cs.CL); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2609.18009

 - **Pdf link:** https://arxiv.org/pdf/2609.18009

 - **Abstract**
 Lightweight audio-visual speech enhancement (AVSE) models face a critical trade-off between computational efficiency and cross-modal alignment accuracy. While simple concatenation lacks relational expressiveness, dense cross-attention incurs computational overhead and is prone to unreliable cross-modal correspondence under strong acoustic interference. We propose Sparse Graph-Guided Mamba (SG-Mamba), a lightweight AVSE framework that integrates a sparse heterogeneous graph with a linear-complexity Mamba backbone. The graph explicitly models modality-specific relations through content-adaptive attention and cross-frame audio-visual connections, while Mamba captures long-range temporal context. We further introduce an audio skip connection to preserve spectral detail without sacrificing noise suppression. Evaluated on LRS3, SG-Mamba achieves competitive or superior performance against strong lightweight baselines and reaches 13.091 dB SI-SDR under noise-only condition. It also remains robust in cluttered multi-speaker conditions with a competitive cost of 3.45 G MACs (or 6.90 G FLOPs). Results on VoxCeleb2 further suggest that explicit structural priors improve robustness, generalizability, and computational efficiency in lightweight AVSE.
#### Correlation-Guided Encoder Selection for Multi-Encoder Large Audio-Language Models
 - **Authors:** Pei-Jun Liao, Hung-Shin Lee, Wenze Ren, Kuo-Hsuan Hung, Hung-yi Lee, Hsin-Min Wang
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Computation and Language (cs.CL); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2609.18041

 - **Pdf link:** https://arxiv.org/pdf/2609.18041

 - **Abstract**
 Multi-encoder fusion extends Large Audio-Language Models (LALMs) beyond speech-centric recognition, but selecting encoders via intuition or exhaustive search often introduces redundant representations and inflates an already constrained compute budget. We propose CUES (Correlation-gUided Encoder Selection), a lightweight heuristic that estimates complementarity through task- and category-level Pearson correlations between encoders' performance profiles, scoring a candidate set from single-encoder evaluations alone--without fusion training during selection. Evaluated on the XARES-LLM benchmark with a frozen SmolLM2-135M backbone (LoRA-adapted) via five-fold cross-validation, CUES consistently identifies the same configuration per track from held-out development splits alone, without using test data for selection. For the broad Track~A suite, CUES selects a cross-family trio (Whisper-medium, mHuBERT-147, and Dasheng-base), achieving a 4.3% relative gain over Whisper-medium (0.771 vs. 0.739). For Track~B text generation, it re-anchors on a focused, speech-only pair (mHuBERT-147 and WavLM-base-plus) and actively abstains from adding a divergent encoder, outperforming mHuBERT-147 by 6.3% (0.589 vs. 0.554). Rather than a failure to scale, this divergence is consistent with a diversity--interference trade-off that CUES navigates per track from correlation signals alone: across the evaluated pool, added cross-family diversity tends toward an inverted-U on broad audio tasks but toward steady degradation on text generation, which favors a focused, speech-anchored set.
#### Mask-Based Speech Enhancement for Spatial Audio: A Comparison of Ambisonics, Beamforming, and Microphone Channels
 - **Authors:** Sheli Hendel, Boaz Rafaely, Dorothea Kolossa
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2609.18532

 - **Pdf link:** https://arxiv.org/pdf/2609.18532

 - **Abstract**
 Mask-based speech enhancement is widely used for suppressing noise and interference, but its performance in spatial audio algorithms with multichannel output has not been studied extensively. In such settings, speech enhancement must improve speech quality while preserving spatial cues that are essential for localization, spatial awareness, and spatial release from masking. In this work, we systematically compare time frequency masking applied to three signal representations: microphone signals, beamformer outputs, and Ambisonics signals. Performance is evaluated in terms of speech quality, intelligibility, binaural cue preservation, and reverberation preservation. Results reveal a clear trade-off between enhancement and spatial fidelity: beamformer-domain masking achieves the highest speech enhancement scores, while Ambisonics-domain masking better preserves the spatial attributes of the residual interference. All methods preserve the target's localization cues.
#### Absolute Quality Ratings of Speech Enhancement Systems by Listeners of Different Ages and Degrees of Hearing Loss
 - **Authors:** Matteo Torcoli, Chih-Wei Wu, Andrea Esposito, Phillip A. Williams, Katrien Cambier, William Wolcott, Antonio Curci, Nicholas S. Reed, Mark Laureyns
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2609.18714

 - **Pdf link:** https://arxiv.org/pdf/2609.18714

 - **Abstract**
 Speech Enhancement (SE) supports listening, particularly for older adults with age-related hearing loss. Yet, enhanced Speech Quality (SQ) is commonly evaluated by young normal-hearing listeners, and how their ratings translate to older adults remains under-explored. We compared absolute SQ ratings from 40 younger normal-hearing listeners (20-30 years) and 67 older listeners (60-95 years) with diverse audiometric profiles, after screening. Test materials comprised natural dialogues with realistic backgrounds. SQ differences between SE systems that were clear for younger listeners were smaller or inseparable in older groups, regardless of hearing status. Hearing loss severity was associated with lower absolute ratings, but did not strongly modulate the contraction in separable SQ differences. A small, audiometrically mixed subgroup of older listeners showed younger-like rating patterns, suggesting that peripheral audiology alone cannot explain the contraction.
#### GrainSpeech: Less Context, More Detail for Compact Speech Synthesis
 - **Authors:** Zitao Liang, Chang Gao
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Artificial Intelligence (cs.AI); Sound (cs.SD); Signal Processing (eess.SP)
 - **Arxiv link:** https://arxiv.org/abs/2609.18856

 - **Pdf link:** https://arxiv.org/pdf/2609.18856

 - **Abstract**
 Compact acoustic models face a challenging quality-capacity trade-off. We investigate two factors in this regime: encoder context and Mel-spectrogram supervision. A receptive-field-scaling study shows that expanding self-attention beyond 15 phonemes provides no consistent gains in pitch, energy, or duration prediction. Guided by this finding, we introduce a fixed-receptive-field convolutional encoder that reduces the respective prediction errors by 36.0%, 17.3%, and 3.4%. We further show that directly transferring image-domain gradient-variance supervision restores fine-scale variation but degrades predicted quality, motivating a Mel-specific formulation with axis-specific gradients, overlapping local statistics, and log-domain variance matching. GrainSpeech contains only 264.8K parameters and achieves 17.9x real-time Mel generation on a microcontroller (MCU), while attaining UTMOS scores comparable to substantially larger models with less than 1.5% of their parameters. Source code and demos are available at this https URL.
#### VoiceTrace: A Benchmark and Retrieval Framework for Who-Said-What Speech Retrieval
 - **Authors:** Aaron Yee, Fengjie Lu, Jiarui Hai, Chenang Jiang, Helin Wang, Siwei Tu, Weitao You, Lingyun Sun
 - **Subjects:** Subjects:
Sound (cs.SD); Artificial Intelligence (cs.AI); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2609.18521

 - **Pdf link:** https://arxiv.org/pdf/2609.18521

 - **Abstract**
 Speech retrieval has become increasingly important as spoken content continues to grow across meetings, lectures, podcasts, and videos. Existing benchmarks and models have advanced semantic search over spoken content, but largely focus on \emph{what} is said while overlooking \emph{who} says it. In many real-world scenarios, however, users need to retrieve speech based jointly on semantic content and a target speaker, where the speaker may be specified naturally through a reference speech utterance rather than a predefined identity. To address this gap, we introduce \textbf{VoiceTrace-Bench}, a benchmark for hybrid speech retrieval in which each query combines text specifying \emph{what} to retrieve with reference speech specifying \emph{who} to retrieve. This setting requires models to integrate complementary semantic and speaker information directly from heterogeneous query inputs. Motivated by the joint audio-text modeling capabilities of audio-language models (ALMs), we develop \textbf{VoiceTrace}, a two-stage retrieval framework consisting of \textbf{VoiceTrace-Emb}, an embedding model that learns unified representations for efficient large-scale retrieval, and \textbf{VoiceTrace-Reranker}, a reranking model that jointly examines each query--candidate pair for fine-grained relevance estimation. Experiments show that VoiceTrace achieves state-of-the-art performance on established semantic speech retrieval benchmarks, while substantially outperforming cascade-based approaches on VoiceTrace-Bench, demonstrating its effectiveness for both conventional semantic retrieval and the new hybrid retrieval setting.


by Zyzzyva0381 (Windy). 


2026-09-17
