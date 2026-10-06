# Showing new listings for Tuesday, 6 October 2026
Auto update papers at about 2:30am UTC (10:30am Beijing time) every weekday.


阅读 `Usage.md`了解如何使用此repo实现个性化的Arxiv论文推送

See `Usage.md` for instructions on how to personalize the repo. 


Keyword list: ['text-to-speech', 'text to speech', 'tts', 'LLM-based', 'speech', 'voice']


Excluded: []


### Today: 15papers 
#### MOV-AAD: A Large-Scale Multimodal Dataset for Auditory Attention Decoding During Moving Conversations
 - **Authors:** Xiaomin He, Vishal Choudhari, Tristan J. Spratt, Aarya Raghavan, Richard T. Lee, Nima Mesgarani
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2610.04180

 - **Pdf link:** https://arxiv.org/pdf/2610.04180

 - **Abstract**
 Auditory attention decoding (AAD) is often evaluated on static, simplified speech scenes that poorly match everyday listening. We introduce MOV-AAD, a large-scale dataset for studying auditory attention under moving, naturalistic conversations. MOV-AAD combines 64-channel EEG with synchronized physiological recordings, including eye tracking, respiration, galvanic skin response, heart rate, peripheral oxygen saturation, body temperature, body motion, and photoplethysmography, enabling analysis of cross-modal neural and physiological markers of attention and listening effort. This dataset uses a more ecologically valid listening paradigm with dynamically moving conversational speech sources and behavioral measures of attentional engagement. MOV-AAD supports research on robust AAD in realistic spatial dynamics, multimodal attention modeling, listening effort, and intersubject neural responses, providing a resource for benchmarking selective auditory attention in naturalistic listening.
#### Factorized Delayed Streams Modeling for LLM-based Streaming ASR
 - **Authors:** Tatsunari Takagi, Kai Washizaki, Atsushi Kojima, Lianbo Liu, Koki Nikaido, Yui Sudo
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Computation and Language (cs.CL); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2610.04333

 - **Pdf link:** https://arxiv.org/pdf/2610.04333

 - **Abstract**
 Delayed Streams Modeling (DSM) enables LLM-based streaming automatic speech recognition (ASR) by aligning acoustic and text streams on a common timeline. DSM adds the padding token <p> and the word-start token <w> to the LLM vocabulary and predicts them together with normal text tokens using the same softmax. We first show that <w> can be removed while maintaining competitive recognition performance. Based on this result, we propose Factorized DSM (F-DSM), which separates the waiting probability for <p> from the distribution over the original LLM vocabulary. This factorization removes ASR-specific tokens from the text prediction space and allows the large-vocabulary softmax to be skipped on waiting steps. Experiments on the Corpus of Spontaneous Japanese and LibriSpeech show that F-DSM achieves better recognition performance than DSM. It also greatly reduces GPU memory use while maintaining similar training throughput, provides a small inference speed improvement through softmax skipping, and reduces the degradation in text-only perplexity observed with DSM.
#### SepRQ : Self-Supervised Speech Mixture Representation Learning via Mask-Free, Multi-Scale Source Separation
 - **Authors:** Séverin Baroudi, Hervé Bredin, Ricard Marxer
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Computation and Language (cs.CL); Machine Learning (cs.LG)
 - **Arxiv link:** https://arxiv.org/abs/2610.04690

 - **Pdf link:** https://arxiv.org/pdf/2610.04690

 - **Abstract**
 Self-supervised learning (SSL) is standard for speech representation learning, but mainstream models are designed around single-speaker audio, limiting their usefulness in multi-speakers scenarios. We present SepRQ, an open-source SSL framework that replaces masked prediction with a pseudo-source-separation objective over frozen random-projection codebooks. By adopting a novel mask-free, multiresolution approach, SepRQ achieves state-of-the-art performance in Speaker Diarization and Speech Separation on the SUPERB benchmark, surpassing WavLM and other cocktail-party derived SSLs at both Base and Large scales, while requiring only 85.68M inference parameters. SepRQ also demonstrates strong performance across target-speaker tasks requiring enrollment (such as Target-Speaker Automatic Speech Recognition), and on the challenging multi-domain DIHARD 3 diarization dataset. Notably, we report strong separation capabilities on three-speaker mixtures (WSJ0-3Mix), where current SSL literature struggles. While cocktail-party SSLs remain scarce and closed-source, limited to C-HuBERT and the enrollment-based SA-WavLM, we open-source SepRQ to the community.
#### UltraM2M: Leveraging Text Transcripts and Mixture Constraints for Weakly-Supervised Speech Enhancement
 - **Authors:** Liu, Jiachen, Wu, Fulin, Wang, Zhong-Qiu
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2610.05155

 - **Pdf link:** https://arxiv.org/pdf/2610.05155

 - **Abstract**
 We propose UltraM2M, a weakly-supervised speech enhancement algorithm building upon the recent unsupervised mixture-to-mixture (M2M) algorithm. M2M realizes unsupervised speech enhancement by training deep neural networks on a set of real-recorded noisy-reverberant multi-channel mixture signals to estimate target speech and non-target signals. The two estimated signals are penalized by a so-called mixture-constraint (MC) loss, which constrains them to reconstruct the observed mixture signals. Although shown to be effective, the mixture constraint may be too weak to enable sufficient noise reduction. To deal with this, UltraM2M extends M2M by further leveraging text transcripts of real-recorded mixtures to design an automatic speech recognition (ASR) loss to penalize the estimated speech signal. The ASR loss can be viewed as a form of weak supervision that could help unsupervised enhancement. Evaluation results on the CHiME-4 dataset show the effectiveness of UltraM2M.
#### Transferable Adversarial Robustness for Speech Foundation Models via Hierarchical Stabilization
 - **Authors:** Aref Mousavi, Shahab Sherafat, Kiarash Kiani Feriz, Amirparsa Safari, Raoof Zare Moayedi, Mohammad Hossein Rohban, Mohammad Sabokrou
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Machine Learning (cs.LG)
 - **Arxiv link:** https://arxiv.org/abs/2610.05310

 - **Pdf link:** https://arxiv.org/pdf/2610.05310

 - **Abstract**
 Frozen speech foundation models (SFMs) make downstream adaptation efficient: the backbone can stay fixed while a task learns layer fusion and a lightweight classifier. Full adversarial fine-tuning is a standard route to robustness, but generating adversarial examples and updating the backbone for every task sacrifices that efficiency. We ask whether robustness can instead be learned before future tasks are known. For a frozen backbone and linear classifier, robustness can be understood through the interaction between representation stability and decision-boundary margin. This leads directly to our design: we stabilize representations across the hidden layers, rather than only the final layer, while preserving clean representations; after clean adaptation selects the layer mixture, we keep it fixed and enlarge only the classifier margin, without downstream adversarial examples. We evaluate Wav2Vec2, HuBERT, and WavLM Large on four tasks under adaptive 30 dB attacks. Across 12 backbone-task pairs, hierarchical robustification improves robust accuracy by 46.4 pp, while margin refinement adds 4.0 pp for 1.1 pp of clean accuracy. Code and configurations are available at this https URL.
#### Speaker Tracking: Segment-online Multi-talker Organization with a Varying Number of Speakers
 - **Authors:** Vahid A. Kalkhorani, Daniel Wong, Jacob Donley, Ashutosh Pandey, Buye Xu, DeLiang Wang
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2610.05693

 - **Pdf link:** https://arxiv.org/pdf/2610.05693

 - **Abstract**
 Speaker tracking is the task of separating and following multiple speakers over time. It must address overlapped speech, speech onset and offset, talker identity, and time-varying speaker count. We propose a segment-online, modular framework for single- and multi-channel speaker tracking. The proposed system first performs speaker separation in each segment and computes speech activity via voice activity detection (VAD). To generate speaker tracks over time, we introduce a two-stage sequential organization strategy: Overlap-based stitching for continuous grouping and memory-based speaker verification for discontinuous grouping. For speaker separation, we employ complex spectral mapping to estimate the real and imaginary spectrograms of underlying speakers. The proposed system achieves state-of-the-art segment-online tracking performance on the LibriCSS and AMI datasets. Our framework significantly reduces diarization error rate (DER) and concatenated minimum-permutation word error rate (cpWER) compared to other methods.
#### Enhancing Pathological Speech through Articulatory Bottlenecks
 - **Authors:** Ailín Pollio San Pedro, Olivier Perrotin, Thomas Hueber
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2610.05944

 - **Pdf link:** https://arxiv.org/pdf/2610.05944

 - **Abstract**
 Dysarthric speech reconstruction (DSR) typically relies on linguistic or phonetic representations extracted from impaired speech to generate a more intelligible waveform. We investigate a complementary approach that instead intervenes in a representation related to speech production. Starting from a neural analysis--synthesis framework, we introduce a residual mapper that modifies an articulatory-aligned latent space while preserving speaker and prosodic information. The mapper is pretrained on parallel synthetic healthy and artificially dysarthric speech, then adapted to natural dysarthric speech using phoneme-guided and adversarial objectives. We further compare the articulatory bottleneck with a dimension-matched unsupervised representation to assess the benefit of explicit articulatory supervision.
#### Character Identity is not Speaker Identity: KyaraBench and KyaraEmbed for Character Verification
 - **Authors:** Joonyong Park, Jerry Li
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2610.06013

 - **Pdf link:** https://arxiv.org/pdf/2610.06013

 - **Abstract**
 A dubbed character keeps its identity while the voice actor changes, so character identity and speaker identity are distinct properties of one recording, yet speaker verification measures only the latter. To address this gap, we propose KyaraBench, a benchmark that scores character voice directly instead of speaker voice, built from 85 human-audited identities in a dubbed anime corpus. It poses two challenging conditions: one that swaps the performer under a fixed character, and one that fixes the performer under changing characters. Listening studies with 78 participants provide human reference scores for cross-performer verification and same-actor discrimination. Speaker-verification baselines show increased errors under these character-specific conditions. We then train KyaraEmbed, a compact encoder using multilingual character supervision, same-actor negatives, and a language-alignment term. The model achieves the best performance on all character-specific conditions in the main comparison. We release the benchmark, protocol, and encoder publicly.
#### Preemptive defense against re-identification attacks on voice anonymization via adversarial perturbation
 - **Authors:** Michele Panariello, Yibo Bai, Massimiliano Todisco, Nicholas Evans
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2610.06074

 - **Pdf link:** https://arxiv.org/pdf/2610.06074

 - **Abstract**
 Voice anonymization (VA) is used to conceal the voice identities in speech data. Performance is usually estimated using automatic speaker verification (ASV) as a proxy to judge the capability of an attacker to re-identify speakers after anonymization. The defender anonymizes test utterances; the attacker compares them to equally anonymized reference utterances to infer voice identity, with degraded ASV performance indicating successful anonymization. Thus, the defender's protection is one-sided and only applied to test utterances via VA. Though unexplored so far, there is an opportunity to enhance anonymization by protecting reference utterances too. We present a new, preemptive VA paradigm: reference utterances are protected using adversarial noise to degrade their potential to infer voice identity. Preemptive protection does not degrade the perceived quality of reference utterances and is independent of the specific ASV system used for identity inference. By combining preemptive protection and regular test-side VA, ASV equal error rates can be increased from 14\% to 45\% (near-perfect privacy).
#### Lyric: Wave-Domain Computing for Efficient Spoken-Digit Recognition
 - **Authors:** Jeeven Balasubramaniam, Nakul Garg
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2610.06433

 - **Pdf link:** https://arxiv.org/pdf/2610.06433

 - **Abstract**
 Low-power speech recognition requires reducing audio acquisition and feature-processing costs as well as neural inference. We present Lyric, a speech-recognition front end that uses wave-domain computing to extract features before digitization. Our proposed system uses passive acoustic resonators to separate speech by frequency and supplies their slowly varying envelopes to a compact temporal neural network. This moves spectral filtering into the acoustic structure, reducing the data and processing needed for recognition. We build a prototype and evaluate spoken-digit recognition on a speakerdisjoint AudioMNIST split across nine classifier families. The front end acquires 32 times fewer scalar samples than a 16-kHz waveform. In the EdgeSpeechNet-A comparison on a Raspberry Pi 4, we reduce preprocessing-plus-inference latency and estimated processor energy per inference by 98.6%, with a 3.58-percentage-point decrease in accuracy to 95.70%.
#### When Layer Selection Misleads Speech Depression Detection
 - **Authors:** Paula A. Perez-Toro, David Gimeno-Gómez, Daniel Rückert, Andreas Maier
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2610.06465

 - **Pdf link:** https://arxiv.org/pdf/2610.06465

 - **Abstract**
 Pretrained speech representations are increasingly used for depression detection, but selecting the best encoder layer on the same data used for evaluation biases reported performance. On DAIC-WOZ, with repeated nested cross-validation across five deep encoder families, naive best-of-25-layer probing inflates AUC by up to $0.09$. On shuffled labels over the \emph{real} latent representations, naive best-of-25 still reaches AUC $0.59$ versus $0.50$ under nested selection, and this bias grows with the number of probed layers and with smaller samples, isolating selection, not signal, as the cause. The effect replicates on a second clinical corpus and language (Androids, Italian). Under leakage-free selection, a fair comparison across representation families identifies a compact, task-aligned affect--prosody model as competitive with substantially more complex SSL, deep-learning, audio-LLM, and codec-based approaches, while using $\sim\!10^{3}\times$ fewer trainable downstream parameters. Extending the analysis to individual PHQ-8 symptoms shows that the same selection bias persists at this finer-grained level. We argue nested model-selection protocols should be standard when probing encoder layer representations on small clinical speech datasets.
#### Can LLM Agents Automate Reinforcement Learning for Text-to-Speech?
 - **Authors:** Xuanjun Chen, Zixiong Su, Hao Shi, Chang Zeng, Kai Li, Jyh-Shing Roger Jang, Hung-yi Lee
 - **Subjects:** Subjects:
Sound (cs.SD); Artificial Intelligence (cs.AI); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2610.04488

 - **Pdf link:** https://arxiv.org/pdf/2610.04488

 - **Abstract**
 Although reinforcement learning (RL) post-training repairs the localized segmental errors of zero-shot text-to-speech (TTS), arriving at a working recipe still relies on tedious manual tuning, and whether LLM agents can take over this research pipeline is unclear. We investigate this question with AgenticTTS-Forge, a collaborative workflow that structures human guidance and agentic execution around a shared workspace, applied to CosyVoice2-0.5B. To measure what the agent automates, we audit its trajectory stage by stage against the published recipe. To measure what it exploits, we score its policies with held-out observers hidden from the agent. Our results show that the agent recovers an underspecified recipe, improves it, and, when gains stall, surveys the literature unprompted and pivots from the LM carrier to the flow carrier, halving Bad cases. However, its autonomy exposes three traps across the data, proxy, and algorithm axes: the held-out set leaks through a channel the contract never reads, a self-shaped reward inflates the proxy where it is scored, and separately tuned policies do not compose additively. These findings show that the binding constraint is measurement rather than reasoning, and can inform the design of harnesses whose contracts read every channel the agent does.
#### GS-Codec: A Gaussian-Splatting Bottleneck for Neural Audio Coding
 - **Authors:** Ron Aluf, Alon Canfi, Eliya Nachmani
 - **Subjects:** Subjects:
Sound (cs.SD); Artificial Intelligence (cs.AI); Machine Learning (cs.LG); Audio and Speech Processing (eess.AS); Signal Processing (eess.SP)
 - **Arxiv link:** https://arxiv.org/abs/2610.04651

 - **Pdf link:** https://arxiv.org/pdf/2610.04651

 - **Abstract**
 Neural audio codecs compress waveforms into compact discrete tokens that underpin speech language models, real-time communication, and large-scale audio storage. Almost every dominant design, including residual vector quantization, finite scalar quantization, and single-codebook variants, follows the VQ-VAE template by partitioning the encoder latent through a learned codebook or a fixed scalar grid. We ask whether this partition is necessary. We introduce GS-Codec, a neural speech codec whose bottleneck is a parametric signal decomposition rather than a quantizer. We adapt Gaussian splatting from 3D scene reconstruction to one-dimensional latents. An inner optimization loop fits each encoder segment as a weighted sum of 1D Gaussian primitives. The decoder then reconstructs the waveform from the rendered sum. To avoid the cost of this iterative inner loop at inference time, we additionally train a lightweight GS Predictor Net that regresses the primitive parameters in a single forward pass. The encoder and decoder are trained end-to-end through the inner loop, with no quantizer anywhere in the training pipeline: the bottleneck is the decomposition itself, and scalar quantization is applied only post-training to the fitted parameters. Rather than relying on discrete codebook stages for bitrate control, our representation exposes a fine-grained rate-quality tradeoff: a single trained checkpoint supports post-training bitrate control by varying the number of primitives and the per-parameter bit depth, with no retraining required. GS-Codec matches or exceeds well-established open-source codecs such as EnCodec and DAC on speaker similarity (SIM), intelligibility (STOI), and perceptual quality (UTMOS) at comparable bitrates, while achieving comparable semantic performance (WER). Code and audio samples are available at this https URL
#### Steering Speech-Language Models: Training-Free Task Specialization via Contrastive Activation Addition
 - **Authors:** Séverin Baroudi, Yanis Labrak, Pierfrancesco Melucci, Sergio Burdisso, Petr Motlicek, Hervé Bredin, Mirco Ravanelli, Ricard Marxer
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2610.04683

 - **Pdf link:** https://arxiv.org/pdf/2610.04683

 - **Abstract**
 Activation steering has proven effective for controlling the behavior of Large Language Models (LLMs) at inference time, but its application to SpeechLLMs remains new, and training-free steering approaches for such models are still largely unexplored. We propose a training-free Contrastive Activation Addition (CAA) protocol that derives steering vectors for common speech tasks (e.g. transcription) in SpeechLLMs from a small number of labeled utterances. We showcase that adding these vectors in the representation space, at inference time, enforces better the targeted speech task. We further show that, when combined with prompting, these vectors yield to consistent improvement over prompting alone on most evaluated tasks such as Automatic Speech Recognition (ASR) or Emotion Recognition (ER), and transfer to out-of-domain data. We additionally demonstrate the usefulness of script-normalization directions to enforce the target script of a specific language.
#### Paradee: Distilling Kokoro-82M into an 8M-Parameter Single-Voice Text-to-Speech Model
 - **Authors:** Sahil Mahendrakar
 - **Subjects:** Subjects:
Sound (cs.SD); Artificial Intelligence (cs.AI); Computation and Language (cs.CL); Machine Learning (cs.LG); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2610.06817

 - **Pdf link:** https://arxiv.org/pdf/2610.06817

 - **Abstract**
 We distill Kokoro-82M, a widely used open text-to-speech model with 54 voices, into Paradee, an 8.07M-parameter model that speaks one of them. Paradee keeps Kokoro's architecture with much narrower layers, and each of its two halves is trained separately against the frozen teacher. It has 10x fewer parameters and needs 15x less compute. We first synthesize a corpus with the teacher and keep its durations, pitch, energy and phoneme features. We then train a small text side to predict these values, and a small decoder to turn the teacher's saved values into the teacher's audio, first with spectral losses and then adversarially. Finally, we connect the two halves and quantize the weights to int8. It needs no alignment learning and no joint training, and it runs on one laptop. Stored in int8, Paradee is 8.5 MB, runs 25x faster than real time on one CPU thread, and scores 4.41 on UTMOS against the teacher's 4.52. The student initially kept a slight buzz, which we trace to the phase of voiced speech between 2 and 8 kHz. A phase-locking filter applied after synthesis removes most of it, with no training and no extra parameters. Code, model files and audio samples are at this https URL


by Zyzzyva0381 (Windy). 


2026-10-06
