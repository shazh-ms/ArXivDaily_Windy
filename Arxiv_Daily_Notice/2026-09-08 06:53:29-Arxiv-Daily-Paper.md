# Showing new listings for Monday, 7 September 2026
Auto update papers at about 2:30am UTC (10:30am Beijing time) every weekday.


阅读 `Usage.md`了解如何使用此repo实现个性化的Arxiv论文推送

See `Usage.md` for instructions on how to personalize the repo. 


Keyword list: ['text-to-speech', 'text to speech', 'tts', 'LLM-based', 'speech', 'voice']


Excluded: []


### Today: 18papers 
#### Auditing Bias and Safety in Voice AI Customer Care
 - **Authors:** Vignesh Ethiraj, Ashwath David
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Computation and Language (cs.CL); Computers and Society (cs.CY); Human-Computer Interaction (cs.HC); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2609.04206

 - **Pdf link:** https://arxiv.org/pdf/2609.04206

 - **Abstract**
 Voice AI systems increasingly mediate customer care interactions where caller presentation cues such as accent, affect, fluency, and urgency are available alongside the service request. Existing fairness and safety evaluations cover speech recognition disparities, spoken dialogue bias, and voice agent capability, but rarely treat customer care voice agents as stateful, multi turn, tool mediated systems where harm can appear as additional burden before any final denial occurs. We formalize a validation gated audit framework for such systems. The framework (i) separates native speech to speech, cascaded ASR to language model to TTS, and hybrid tool mediated architectures; (ii) uses matched service facts across controlled caller presentation conditions; (iii) validates fact invariance, presentation cues, artifacts, and acoustic measurements before inference; and (iv) records both material outcomes and path to service burden. We define the research problem, methodology, seven validation gates, a six family metric set, and claim boundaries for an active industry evaluation program. We illustrate the framework with a fully synthetic worked example of a refund dispute audit instance. Production system results are excluded from this release; public reporting is gated by the validation protocol.
#### GEPARD - Generative, Prosody-aware, Autoregressive text-to-speech model for Realtime Dialogue
 - **Authors:** Denis Pavlov, Ulanbek Abdurazakov, Nursultan Bakashov
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Computation and Language (cs.CL); Machine Learning (cs.LG); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2609.04222

 - **Pdf link:** https://arxiv.org/pdf/2609.04222

 - **Abstract**
 We present GEPARD (Generative, Prosody-aware, Autoregressive text-to-speech model for Realtime Dialogue), a streaming text-to-speech model for real-time spoken dialogue. GEPARD generates speech autoregressively with an LLM backbone - text and audio embeddings are trained together in a single decoder-only model - and decodes it to a waveform with an FSQ-based neural codec, streaming audio chunk-by-chunk as text arrives. Our central goal is a TTS architecture served by a standard LLM engine (vLLM) without modifying its compute kernels. This defines the overarching design principle: the backbone is a standard full-attention transformer, while all non-trivial auxiliary mechanisms - zero-shot voice cloning, text augmentation, and classifier-free guidance - are moved out of the autoregressive decode loop into prefill, or distilled directly into the weights. On streaming end-to-end inference, a single stream reaches a Real-Time Factor of about 0.067 (roughly 15x faster than real-time); under 256 concurrent streams the system reaches an aggregate speedup of about 204x on a single server-class GPU. We detail: (1) system-level solutions for vLLM-native serving; (2) the "short register" (1-2 word) failure mode of autoregressive speech decoders, with diagnostic probes and a mitigation; and (3) distillation of two-pass classifier-free guidance over text into single-pass weights via Direct Preference Optimization (DPO).
#### The Trade-off Was in the Labels: Causal Supervision for Turn-Aware Streaming ASR
 - **Authors:** Bojie Li, Noah Shi
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2609.04225

 - **Pdf link:** https://arxiv.org/pdf/2609.04225

 - **Abstract**
 A voice agent must decide, moment to moment, whether the user has finished; silence rarely settles it: a caller reading a phone number pauses mid-digits, a one-word "Stop!" ends a turn, a long question carries pauses longer than real turn-gaps. A voice-activity detector plus a silence timeout (the deployed default) cannot separate these, because within-turn pauses routinely exceed between-turn gaps; what distinguishes them is whether the words so far form a complete thought: what a recognizer computes to produce a transcript. We present the first open training recipe and benchmark for turn-aware streaming ASR: a small LoRA adapter on Qwen3-ASR-0.6B, trained in hours on one GPU, that transcribes, detects end-of-turn from meaning and silence, handles dictation, and grounds transcription in context. On a deployment-matched benchmark it reaches 0.97 boundary recall at 0.39 s median latency with 0.3 false fires per speech-minute, replicated on a fresh test set; no silence timeout reaches this point. The recipe rests on one principle: every streaming-decision label must be computable from input up to the decision point. Offline corpora violate it, encoding the future; such clairvoyant labels manufactured oscillation and a phantom recall-versus-precision trade-off, exposed when one appended second of silence raised a "broken" model's end-of-turn recall from 0.10 to 1.00. The same leak recurred with context: an always-matching biasing prefix became a copied shortcut (40% intrusion), and counterfactuals disagreeing with the audio cut this to 0.8% while keeping most of a +28.9 pp entity-recall benefit.
#### EffVOC: Low-Delay Efficient Speech Waveform Reconstruction from Spectral Representations Without Phase
 - **Authors:** Renzheng Shi, Simon Welker, Timo Gerkmann, Tim Fingscheidt
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2609.04226

 - **Pdf link:** https://arxiv.org/pdf/2609.04226

 - **Abstract**
 The Griffin-Lim algorithm has been a seminal contribution for phase reconstruction from amplitude spectrograms, however, requiring (infinitely) high algorithmic delay. Its low-delay variant suffers in speech quality. Recent (generative) neural network methods improve on speech quality still at medium to high algorithmic delay, but often they are complex and optimized only for one specific input representation. We build upon an efficient low-delay speech vocoder and propose EffVOC, which supports synthesis of wideband (WB) or fullband (FB) speech from either amplitude spectrum or Mel coefficient inputs. We evaluate both input representations across multiple model sizes in a unified framework and compare against state of the art. Results show that our proposed low-delay (20 ms vs. 32 ms or more) efficient approach marks a new SOTA by achieving top-ranked subjective MOS scores (WB: 4.17/4.15, FB: 4.14/4.11) for amplitude spectrum/Mel representations, very close to ground truth.
#### Automatic Speech Recognition for Multilingual Oral History Research
 - **Authors:** Sidney Wong, Chelsea Wong She, Eda Tang, Tiana Marshall Wong, Debbie Sew Hoy, Chelsea Wong
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Computation and Language (cs.CL)
 - **Arxiv link:** https://arxiv.org/abs/2609.04232

 - **Pdf link:** https://arxiv.org/pdf/2609.04232

 - **Abstract**
 This paper offers a unique perspective on how speech technologies are being adopted by community-led heritage language preservation and revitalisation initiatives. As a community-led language maintenance strategy, oral histories play a crucial role in Cantonese language revitalisation in New Zealand. The development of Automatic Speech Recognition (ASR) toolkits, such as Whisper, have expedited what has often been a resource and time-intensive process of transcribing oral history collections. However, there is limited research into the effectiveness of ASR toolkits when applied to code-switched language contexts. Based on Word Error Rate (WER), the best performing Whisper model configuration achieved a WER of 12.10 at the expense of accurately transcribing unsupported non-English segments. However, Whisper remains a useful tool by providing a first-pass transcription using only 1% of the estimated time otherwise needed for manual transcription.
#### Robust Speech Emotion Recognition under Tone-Word Conflict: A Benchmark and Framework
 - **Authors:** Xiaojiang Peng, Dawei Huang, Yongjie Lv, Ruijie Xiong, Chunxiang Jin, Bin Li, Xiaohui Wang, Zitong Yu
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2609.04236

 - **Pdf link:** https://arxiv.org/pdf/2609.04236

 - **Abstract**
 Speech emotion recognition (SER) is a crucial component of human-computer interaction, attracting extensive attention from both industry and academia. However, existing SER systems typically assume alignment between vocal tone and lexical semantics, overlooking the real-world scenarios that involve tone-word conflict-where the emotion conveyed by speech contradicts the literal meaning of the words. To bridge this gap, we introduce TWIN-SER (Tone-Word Incongruent SER), a benchmark for systematic evaluation under acoustic-semantic incongruence, and show that state-of-the-art models degrade severely under such incongruence. To address this, we propose DAS (Disentangled Acoustic-Semantic fusion), a framework that mitigates tone-word conflict by explicitly disentangling acoustic and semantic pathways, selecting informative high-energy embeddings, and adaptively fusing them via a lightweight query-based attention mechanism. Specifically, DAS comprises three crucial modules: i) a heterogeneous feature extraction module that separately captures complementary acoustic and semantic representations from raw input; ii) a high-energy embedding selection module that identifies and retains the most discriminative embeddings; and iii) a Q-Former combination module that bridges the two pathways through cross-attention, enabling robust emotion prediction under incongruent conditions. Extensive experiments demonstrate that DAS consistently outperforms existing methods in tone-word conflict scenarios, as well as in standard in-domain and zero-shot settings. Our code and datasets are available at this https URL
#### Rethinking Speech Codecs: From Compression to Autoregressive Generative Modeling
 - **Authors:** Yazheng Yang, Yao Qiu, Hui Su, Qi Liu
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Multimedia (cs.MM); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2609.04237

 - **Pdf link:** https://arxiv.org/pdf/2609.04237

 - **Abstract**
 Recent advances in speech language models leverage discrete speech representations from pretrained codecs to enable scalable training and generation. However, existing codecs are primarily optimized for compression without accounting for the autoregressive nature of language model training, resulting in suboptimal performance when modeling compressed speech tokens. In this work, we revisit speech discretization from a generative modeling perspective and propose a novel framework that explicitly aligns speech tokenization with autoregressive training. Our approach introduces autoregressive-compatible constraints during codec training, encouraging token sequences that exhibit temporal consistency and predictability. In addition, we propose a heterogeneous downsampling strategy for different layers of speech tokens, distinguishing semantic from acoustic layers, to improve the alignment between semantic tokens and corresponding textual content. Extensive experiments across multiple benchmarks demonstrate that our method bridges the gap between speech compression and generative modeling, enabling more effective continued pretraining of existing language models on speech data. The approach consistently improves performance across multiple codecs, validating its generality and applicability to diverse speech modeling scenarios.
#### TurnFSM for Full-Duplex Dialogue System: Internalizing State-Machine Logic for Streaming Semantic Voice Activity Detection and Utterance-Level Rejection
 - **Authors:** Zhiwei Lin, Tianjiao Du, Qiaochu Huang, Zihan Zhang, Naijun Zheng, Longshuai Xiao, Yunfei Lu, Jun Chen, Zhiyong Wu
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2609.04240

 - **Pdf link:** https://arxiv.org/pdf/2609.04240

 - **Abstract**
 Full-duplex voice assistants must continuously listen while speaking, handling user interruptions under low-latency and resource-constrained streaming conditions. Existing end-to-end full-duplex models can compromise reasoning-related capabilities after speech-domain adaptation, whereas cascaded pipelines introduce extra inference overhead and handcrafted control logic. We propose TurnFSM, an LLM-based state prediction framework that internalizes turn control as explicit finite-state transitions, unifying streaming semantic VAD and utterance-level rejection. TurnFSM decomposes submission and rejection into a serial decision process, reducing multi-task interference while maintaining performance comparable to single-task models. We further introduce a first-order state transition mechanism that enforces the dependency on only the previous state during training, enabling compact inference with the standard causal mask and original LLM positional encoding while avoiding historical state-token accumulation and unnecessary step-by-step state generation. Experimental results show that TurnFSM consistently outperforms the binary-head baseline and remains competitive with task-specific models.
#### Training-Free Speech-Centric Omni Understanding with Frozen VLMs
 - **Authors:** Ankan Deria, Hanoona Rasheed, Xilin He, Fahad Shahbaz Khan, Salman Khan
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Computer Vision and Pattern Recognition (cs.CV); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2609.04242

 - **Pdf link:** https://arxiv.org/pdf/2609.04242

 - **Abstract**
 Audio-visual understanding remains challenging because models must jointly interpret spoken content, visual events, and their temporal relationships. Existing omni models typically introduce dedicated audio encoders and rely on expensive audio-video-text training, tightly coupling omni capability to specific VLM backbones and potentially weakening their existing visual and reasoning abilities. This raises three questions: whether native omni training is necessary for every new VLM, whether speech-centric omni capability can be added while preserving the original backbone, and where richer acoustic representations remain essential. We introduce Training-Free Omni (TFO), a plug-and-play framework that converts a frozen VLM into a speech-centric omni model without architectural modification, or multimodal re-alignment. TFO uses Whisper to extract confidence-filtered, timestamped transcripts and routes them through the VLM's existing language interface, while leaving its visual pathway unchanged. Across matched comparisons with native omni models on 56 benchmarks and 21 languages, TFO is competitive on audio-visual understanding, improves average audio-only performance across all five model settings, and achieves substantial multilingual speech gains. Freezing the VLM also generally preserves stronger image/video understanding, visual grounding, coding, mathematical reasoning, and medical question answering than the corresponding native omni checkpoints. These results show that strong speech-centric omni understanding can often be obtained through modular audio-to-language routing rather than costly backbone-specific training.
#### Probing Warmth-Mediated Harm in Speech-Enabled LLMs for Mental-Health Conversations
 - **Authors:** Eugenia Kim, Bolor-Erdene Jagdagdorj, Dina Pekelis, Leah Zulas, Amanda Minnich
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2609.04256

 - **Pdf link:** https://arxiv.org/pdf/2609.04256

 - **Abstract**
 Audio LLM benchmarks measure understanding and dialogue quality, not whether speech-enabled models respond with relational warmth when a vulnerable user discloses a mental-health concern. We introduce a 7-turn scripted-disclosure probe grounded in WHO mental-health clinical guidelines, with each script run on the same model (Azure OpenAI gpt-realtime) in both audio and text-only conditions, and acoustic-prosody analysis of the generated speech. Across 532 responses we identify two audio-specific patterns transcript-only evaluation would miss: at the elicitation turn the model's voice gets shorter, faster, lower-pitched, and quieter rather than warmer (p < .001 for five of seven acoustic features), and the modality gap on relational acceptance, small in aggregate, concentrates in the highest-stakes self-harm/suicide scripts. A two-rater listener study corroborates that perceived warmth is concentrated at specific turns and on bereavement disclosures. Together these patterns indicate that auditing speech-enabled models in mental-health contexts requires evaluating the combined audio-and-text experience the user encounters, not the transcript in isolation. We release the protocol, scoring pipeline, and scripts as a starting point for evaluating speech-enabled models in mental-health contexts.
#### GhostWord: A Fine-Grained Backdoor Attack on Automatic Speech Recognition
 - **Authors:** Mojtaba Nafez, Mobina Poulaei, Kiarash Kiani Feriz, Aref Mousavi, Mohammad Ebrahim Mahdavi, Mohammad Mosayebi, Mohammad Hossein Rohban
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2609.04260

 - **Pdf link:** https://arxiv.org/pdf/2609.04260

 - **Abstract**
 Automatic Speech Recognition (ASR) systems are widely deployed in safety-critical settings but remain vulnerable to data-poisoning backdoor attacks. Existing ASR backdoors typically use phrase-level triggers paired with a fixed target sentence, creating strong artifacts (e.g., repeated transcripts or triggers placed in non-speech regions) that simple preprocessing can mitigate. We propose GhostWord, a word-level, time-localized ASR backdoor that uses codebooks mapping short ($\approx$400\,ms) acoustic triggers to target words. During poisoning, we inject a trigger into the forced-aligned time span of a chosen source word in the audio and replace only that word in the transcript, enabling precise semantic flips and composable sentence manipulation while avoiding many-to-one label artifacts. Across Common Voice (v23 English, v24 Lithuanian) and multiple backbones (Whisper-Small/Medium, MMS, SpeechT5), GhostWord achieves an average attack success rate of 89.3\% and transfers across languages and models. Adapting optimization-based defenses (ABL, ANP, SAU, I-BAU) reveals a sharp robustness--accuracy trade-off: attack success drops from 89.3\% to 29.1\% while clean WER rises from 21.5\% to 45.0\%, consistent with our theoretical analysis showing that, in high-vocabulary models, backdoor suppression structurally tends to degrade clean performance. The source code is publicly available at this https URL
#### Brain2Speech-Net: Intelligible, Real-Time Brain-to-Speech Synthesis Without Text Decoding
 - **Authors:** Shreeram Suresh Chandra, Zexin Cai, Yu Tsao, Simon King, Berrak Sisman
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2609.04455

 - **Pdf link:** https://arxiv.org/pdf/2609.04455

 - **Abstract**
 The loss of speech limits communication for individuals with paralysis. Restoring speech by synthesizing it directly from neural activity is challenging: intracortical data are scarce and lack aligned targets, so most systems rely on cascaded neural-to-text-to-speech pipelines that add latency and propagate errors. We present Brain2Speech-Net, among the first single-stage frameworks to remain intelligible under limited data while removing intermediate text decoding. A differentiable phoneme bottleneck preserves linguistic structure without explicit text decoding. A lightweight deep-HMM aligner then maps this bottleneck to contextual phoneme representations in a TTS latent space. It learns monotonic alignment between neural recordings and phoneme segments without frame-level supervision, inheriting strong acoustic priors for data-efficient training. On an intracortical dataset, Brain2Speech-Net achieves strong intelligibility in objective and listening tests while running faster than real time. Unlike cascaded systems that incur high latency and direct speech-unit models that lack intelligibility, it delivers both intelligible and real-time speech.
#### Discriminative Flow Matching: Beyond Time-Conditioning in Generative Restoration via Flow-State Representations
 - **Authors:** Shrishti Saha Shetu, Emanuël A. P. Habets, Andreas Brendel
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2609.04525

 - **Pdf link:** https://arxiv.org/pdf/2609.04525

 - **Abstract**
 Existing Conditional Flow Matching (CFM) formulations describe transport progress using an explicit interpolation coordinate, commonly interpreted as time, assuming that a single global variable adequately represents a sample's position along the generative trajectory. In restoration tasks, however, transport progress is sample-dependent because the initial distribution may exhibit varying statistical dependencies with the target distribution. Thus, samples at the same interpolation coordinate can differ substantially in degradation level, distance to the target distribution, and restoration difficulty. We investigate whether signal representations learned by discriminatively trained models provide a meaningful description of generative transport state in CFM-based restoration. Through systematic latent-space analysis, we show that discriminative representations organize according to degradation severity and follow a consistent trajectory toward the clean-data manifold during generation. Motivated by these observations, we introduce the Discriminative Flow-State Hypothesis, which posits that discriminative representations encode a transport state governing generative restoration. Based on this hypothesis, we propose Discriminative Flow Matching, which conditions the Flow-Matching velocity field on Discriminative Flow-State Representations rather than explicit time coordinates. Experiments on speech enhancement and image denoising show that these representations characterize restoration progress, enable adaptive inference, and consistently outperform CFM and diffusion-related baselines. Our findings suggest that discriminative representations provide an effective state-aware alternative to explicit time conditioning and offer a novel perspective on the relationship between discriminative and CFM-based generative modeling.
#### Enhancing Neural Speech Coding with Semantic and Visual Cues
 - **Authors:** Yao Guo, Yang Ai, Hui-Peng Du, Xiao-Hang Jiang, Chen-Yuan Ning, Zhen-Hua Ling
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2609.05076

 - **Pdf link:** https://arxiv.org/pdf/2609.05076

 - **Abstract**
 At low bitrates, neural speech codecs have limited capacity to encode all information needed for high-quality re construction, especially when relying solely on speech-derived representations. To address this limitation, this paper proposes a Semantic- and Visual-enhanced Speech Codec (SVSC), which in corporates semantic and visual cues into the neural speech coding process. Specifically, built upon a mainstream neural speech cod ing architecture, SVSC introduces a semantic encoding-decoding branch and an image analysis-synthesis branch. It fuses deep semantic features with visual cues through a cross-attention mech anism, forming an auxiliary high-level representation enriched with contextual and articulatory information. To handle different inference scenarios, SVSC introduces two information-injection strategies based on the availability of auxiliary semantic and vi sual cues. When such cues are available, the fusion mode directly incorporates the auxiliary representations into the speech coding branch through feature concatenation; otherwise, the distillation mode transfers auxiliary information into the speech coding branch through knowledge distillation during training, enabling speech-only inference without additional inputs. Experimental results validate the effectiveness of incorporating semantic and visual cues, improving the ViSQOL score of reconstructed speech from 3.86 to 4.01.
#### Grounded Decoding for Autoregressive Speech Enhancement via Adaptive Code-Space Grounding and Local LLM Refinement
 - **Authors:** Hao Shi, Yuan Gao, Zhaoheng Ni, Junyi Peng, Gongping Huang, Yu Tsao, Xugang Lu
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2609.04245

 - **Pdf link:** https://arxiv.org/pdf/2609.04245

 - **Abstract**
 Large language model (LLM)-based autoregressive speech enhancement (SE) produces natural speech using learned clean-speech priors, but may hallucinate content unsupported by the input. Deterministic SE better preserves observation-coupled evidence, yet often retains residual noise or local distortion. We propose an evidence-grounded generative SE framework that uses a deterministic estimate as imperfect evidence. A Whisper-guided DPRNN produces an enhanced waveform, which is blended with the observation and tokenized into a discrete evidence sequence. The evidence conditions an autoregressive clean-speech token generator and is reused during decoding through Code-Space Grounding (CSG), which penalizes candidates according to their Hamming distance in the factorized finite-scalar-quantized (FSQ) space. Because the appropriate grounding strength depends on acoustic difficulty, we introduce SNR-Conditioned CSG (SNR-CSG), which maps a calibrated residual-SNR estimate to an utterance-level strength and constructs an adaptive grounded anchor. Although grounding improves content fidelity, the anchor may retain local acoustic defects inherited from the evidence. Since such defects are predominantly local in the FSQ space, nearby tokens may provide better acoustic realizations without large departures from the observation-supported trajectory. We therefore propose Grounded Neighborhood Refinement with LLM ranking (GNR-LLM). It performs one additional teacher-forced pass conditioned on the grounded-anchor history, intersects the LLM top-$K$ candidates with a local FSQ Hamming neighborhood. Experiments on in-domain, controlled-SNR, and DNS no-reverb conditions show that SNR-CSG provides robust automatic grounding, while GNR-LLM substantially improves low-SNR perceptual quality without sacrificing content fidelity.
#### Motion-Omni: End-to-End Joint Speech and Full-Body Motion for Spoken Dialogue
 - **Authors:** Chengqian Ma, Wei Tao, Haoyu Zhang, Yiwen Guo
 - **Subjects:** Subjects:
Sound (cs.SD); Computer Vision and Pattern Recognition (cs.CV); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2609.04250

 - **Pdf link:** https://arxiv.org/pdf/2609.04250

 - **Abstract**
 An avatar that holds a conversation should decide what to say and to move while saying it, yet these abilities live in separate model families: spoken dialogue models produce speech without motion, and co-speech motion models produce motion only from audio handed to them. The standard remedy is a cascade that first generates the spoken response and then runs a motion model over the finished audio, which requires a second full inference pass and precludes any joint optimisation between the two. We present Motion-Omni, an end-to-end framework in which a spoken dialogue model natively outputs explicit facial expression together with hand, upper-body and lower-body motion, generated directly from the hidden states that produce the speech. Joint training is not optional here: with the speech pathway frozen, motion remains misaligned with the audio, and co-adapting the LLM, Speech Generator and Motion Generator under both objectives is what recovers alignment while retaining spoken-dialogue ability. Supervision comes from a scalable, model-agnostic pipeline that pseudo-labels consistent-voice speech responses with a replaceable motion teacher, yielding 422,856 quality-ranked pairs (1,402 hours). We further release SwDA-500 and, to our knowledge, the first public evaluation protocol for stochastic open-ended full-body spoken dialogue, matching audio across motion systems while unifying rendering, automatic metrics, human evaluation, and latency measurement. Instantiated with a Qwen2.5-7B-Instruct backbone, Motion-Omni-Q7 matches the same-audio teacher cascade to within 2% on reference-free motion metrics while responding 5.4 x faster (RTF=0.78, faster than real time), surpasses all non-teacher cascades on beat correlation and diversity, and reaches a 2.62% word error rate, the lowest among the omni-modal systems compared.
#### Low-Latency Spell Correction for Japanese Music Search Queries
 - **Authors:** Anshul Garg, Pavni Tandon, Karan Bhukar, Tanmay Khandelwal, Ujjal Kumar Dutta
 - **Subjects:** Subjects:
Sound (cs.SD); Machine Learning (cs.LG); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2609.04262

 - **Pdf link:** https://arxiv.org/pdf/2609.04262

 - **Abstract**
 Spell correction for Japanese search queries presents unique challenges due to the co-existence of four writing scripts (Latin/romaji, hiragana, katakana, and kanji) and the distinct error patterns each script induces. We present a compact BART-based sequence-to-sequence model (3 encoder + 3 decoder layers) designed for low-latency spell correction of Japanese music search queries. The core contribution lies in a script-aware synthetic misspelling generation pipeline that produces realistic training data by combining keyboard-layout models (QWERTY and flick input), phonetic confusion priors mined from real query logs, voiced/unvoiced consonant alternations, and kana case errors. A key design decision is normalizing mixed-script catalog titles to a single canonical script before misspelling synthesis, which we show is critical for reducing model hallucinations. We train a custom byte-level BPE tokenizer on the target music catalog to handle all four scripts in a unified vocabulary. Experiments on a curated evaluation set show that our model achieves an exact-match accuracy of 41.09% and a character error rate (CER) of 11.62%, outperforming edit-distance baselines and achieving the lowest character error rate among all evaluated systems while maintaining sub-4ms inference latency on a single GPU. We further analyze performance across individual scripts and mixed-script queries, demonstrating the effectiveness of script-aware data augmentation through systematic ablation studies.
#### Scalable Context Orchestration for Serving LLMs Over Voice
 - **Authors:** Linyi Jiang, Silvery D. Fu, Yifei Zhu
 - **Subjects:** Subjects:
Sound (cs.SD); Artificial Intelligence (cs.AI); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2609.04288

 - **Pdf link:** https://arxiv.org/pdf/2609.04288

 - **Abstract**
 Voice AI applications are gaining popularity as advances in large language models (LLMs) enable more natural and accessible spoken interactions. Serving these applications requires accounting not only for what users say, but also for how they speak (e.g., speaking rate) and the conditions under which their audio is captured and transmitted (e.g., background noise and packet loss). However, existing LLM systems represent conversation context as a flat, growing sequence of messages, leaving voice-specific context implicit in the audio. As a result, they can generate responses that are poorly aligned with user preferences, degrade interaction quality under adverse environmental conditions, and incur high costs over long voice sessions. We present llmovoice, a context-management middleware that explicitly models voice context and orchestrates its use. At each turn, llmovoice constructs a bounded voice context from the current user input, relevant interaction history, and explicit paralinguistic and environmental states. It then uses the serving LLM to reason over this context and generate runtime directives that guide how the system responds. We evaluate llmovoice on real-world voice applications and benchmarks. It reduces speaking-rate alignment error by 52.4%, lowers the false-interruption rate from 46.0% to 0.9% under packet loss, and reduces model usage cost by 79.2%. For long sessions, llmovoice reduces per-turn cost by up to 24.9 times while retaining up to 98.7% of baseline answer quality.


by Zyzzyva0381 (Windy). 


2026-09-08
