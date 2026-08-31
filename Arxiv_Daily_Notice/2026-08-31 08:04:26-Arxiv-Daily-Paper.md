# Showing new listings for Monday, 31 August 2026
Auto update papers at about 2:30am UTC (10:30am Beijing time) every weekday.


阅读 `Usage.md`了解如何使用此repo实现个性化的Arxiv论文推送

See `Usage.md` for instructions on how to personalize the repo. 


Keyword list: ['text-to-speech', 'text to speech', 'tts', 'LLM-based', 'speech', 'voice']


Excluded: []


### Today: 7papers 
#### SURE-Challenge: Evaluating Speech Evidence Before Speech-LLM Generation
 - **Authors:** Mengzhe Geng
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Computation and Language (cs.CL); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2608.27783

 - **Pdf link:** https://arxiv.org/pdf/2608.27783

 - **Abstract**
 Speech LLMs are usually graded after they answer, although an operating system first has to decide whether a waveform should be sent to the model. We define the Speech-Unsupported Rejection Evaluation Challenge (SURE-Challenge) for this admission step. The benchmark pairs LibriSpeech-derived transcription and first-word question answering with unsupported silence, colored noise, synthetic tones, and source-ambiguous babble under disjoint source splits. Front-end ablations use Qwen2-Audio; the selected energy-plus-Whisper-score rule is then replayed before six speech/audio LLMs. On the 474-row leakage-screened SURE-Extended test set, raw Qwen2-Audio rejects 15/204 unsupported inputs, whereas the fixed rule rejects 196/204 and leaves supported accuracy unchanged. External checks delimit this number: Common Voice retention drops as the Whisper-score threshold is tightened, and no-speed babble gives 18 to 24 rejected clips out of 54 across regenerated seeds. The result identifies a pre-generation error mode missed by answer-only scoring.
#### Effects of HRTF Augmentation on Predicted Spatial Release from Masking in Music
 - **Authors:** Jack Webb, Christophe Lesimple, Volker Kuehnel, Lorenzo Picinali
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2608.28422

 - **Pdf link:** https://arxiv.org/pdf/2608.28422

 - **Abstract**
 Separating individual musical instruments within a complex mixture of sounds poses a persistent challenge for listeners with hearing loss. Although spatial separation of sources improves speech recognition in this population, the potential benefits of spatial cue enhancement for music perception remain largely unexplored. This paper introduces a method to increase spatial cue salience through the augmentation of individual head-related transfer functions (HRTFs). Auditory model analyses indicate that augmented HRTFs may enhance the separability of musical instruments relative to individual HRTFs. Predicted benefits persist when moderate sensorineural hearing loss is modelled, though they are substantially reduced. Simulated hearing aid processing does not restore these benefits to normal-hearing levels.
#### A Mixed-Behavior Vote Model for Multimedia Subjective Quality Votes, Means, and Variances
 - **Authors:** Jaden Pieper, Stephen D. Voran
 - **Subjects:** Subjects:
Multimedia (cs.MM); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2608.27724

 - **Pdf link:** https://arxiv.org/pdf/2608.27724

 - **Abstract**
 The relationship between subjective test vote variance and vote mean (or MOS) is well-studied, and the mathematically admissible vote variance region has been previously defined. We propose a reduced admissible variance region called the Unimodal Variance Region (UVR) that better describes real subjective rating behavior of multimedia. Further, subjective vote variance is often modeled as parabolic. We explain that, in practice, the parabolic model often violates the admissible region in the variance vs. MOS plane and we propose alternatives that respect the admissible region. We also present a parametrized random process to model votes that mixes voting processes and produces a realistic range of vote variances within the UVR at any desired MOS. This process was inspired by and comports with voting behavior that is observed in many subjective tests. By modeling vote variance from a subjective experiment, this vote model offers additional interpretable insights into voting behavior observed in a given experiment. We present example results from 16 datasets spanning speech, image, and video subjective quality experiments.
#### Auditing Generative Audio Calls for Known-Task Audio-LLM Evaluation
 - **Authors:** Mengzhe Geng
 - **Subjects:** Subjects:
Sound (cs.SD); Computation and Language (cs.CL); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2608.27817

 - **Pdf link:** https://arxiv.org/pdf/2608.27817

 - **Abstract**
 Speech and audio LLMs are often evaluated by asking whether a waveform prompt beats an automatic speech recognition (ASR) transcript. For known closed-set tasks, that comparison conflates two factors: access to acoustic evidence and the need to call a generative audio model. We evaluate this distinction as a controlled call-decision problem. For each example, a policy chooses among keeping a transcript label, using encoder evidence from Contrastive Language-Audio Pretraining (CLAP), Audio Spectrogram Transformer (AST), or WavLM, and calling Qwen2-Audio, Qwen2.5-Omni, or MOSS-Audio; the decisive ablation removes all generative actions while keeping the selector and development protocol fixed. On VocalSound, transcripts reach 0.296 accuracy, so waveform information is needed. Yet supervised CLAP and WavLM controls reach 0.850 and 0.854 with no generative audio calls. A selector with generative actions reaches 0.925 accuracy using 12.5% calls, compared with 0.921 for the matched no-call selector (paired difference 0.004; 95% CI [-0.025,0.033]). Agreement and stacking features improve weaker selectors but do not beat the strongest no-call control. For known-task endpoint claims, the relevant quantity is the marginal value of the generative call after transcript and encoder evidence have already been used.
#### Is Prosody Lost in Translation? Fine-Grained Cross-Lingual Prosody Similarity Across Languages
 - **Authors:** Haopeng Xie, Ismail Rasim Ulgen, Sofia Son, Berrak Sisman, Philipp Koehn
 - **Subjects:** Subjects:
Sound (cs.SD); Computation and Language (cs.CL); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2608.27848

 - **Pdf link:** https://arxiv.org/pdf/2608.27848

 - **Abstract**
 Prosody plays an important role in speech translation, conveying information such as emphasis, emotion, and intent beyond lexical content. However, despite recent progress in expressive speech-to-speech translation (S2ST), little is known about how prosodic patterns are similar/different across languages. Understanding these cross-lingual similarities and differences is crucial for effectively incorporating prosody into expressive S2ST systems. In this work, we present the first fine-grained cross-lingual analysis of prosody using multilingual dubbing data across English-German, English-Spanish, and English-French language pairs. We analyze the similarity of pitch, energy, and temporal feature patterns between source and target speech and investigate the linguistic and alignment-related factors affecting this similarity. Our analysis reveals inherent cross-lingual correlations in prosodic structure between certain languages. The findings provide important insights into the transferability of prosody across languages and offer empirical guidance for future expressive speech-to-speech translation systems.
#### Multirate State Space Models for End-to-End Processing of Pulse Density Modulated Speech Signals
 - **Authors:** Ludovic Boulanger, Sean U. N. Wood
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2608.28472

 - **Pdf link:** https://arxiv.org/pdf/2608.28472

 - **Abstract**
 Deep neural networks (DNNs) based on state-space models (SSMs) are increasingly applied to speech processing, but typically operate on pulse-code-modulated (PCM) audio. This constrains deployment on low-power, always-on edge devices, which commonly use single-bit pulse-density-modulated (PDM) micro-electromechanical (MEMS) microphones for their noise robustness, low cost, and variable sampling rates that enable low-power operation. In fact, converting PDM to PCM requires low-pass filtering and decimation, imposing costly overhead on resource-constrained hardware. While prior works have attempted to process PDM signals directly, they require long training times and generalize poorly across sampling rates. In this paper, we show that the SSM has two key properties that remediate these issues: its continuous-time parametrization allows it to produce a consistent representation of the input audio signal, regardless of the modulation strategy and sampling rate, and its long-term memory enables this representation to be aggressively downsampled without needing any anti-aliasing operations. We then propose a novel end-to-end PDM speech processing architecture that uses an SSM to encode the input audio signal into a modulation- and sampling-rate-invariant latent representation. We show that our proposed architecture achieves robust speech classification and enhancement gains at low-power sampling-rates (512 kHz) and similar performance to state-of-the-art algorithms operating on PCM data when tested on standard PDM sampling-rates of 2 MHz. Moreover, we show that the SSM's output can be downsampled by more than 65,000 times, thus significantly reducing the number of processing timesteps in downstream layers.
#### Low-Power End-to-End Cochlear Implant Speech Denoising with Spiking Neural Networks
 - **Authors:** Ludovic Boulanger, Sean U. N. Wood
 - **Subjects:** Subjects:
Sound (cs.SD); Neural and Evolutionary Computing (cs.NE); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2608.28493

 - **Pdf link:** https://arxiv.org/pdf/2608.28493

 - **Abstract**
 Cochlear implants (CI) restore hearing for individuals with severe to profound hearing loss. However, CI users often struggle to understand speech in noisy environments. Deep neural networks (DNN) have shown promise in enhancing speech for CI users, yet their high energy demands make them non-ideal for low-power CI processors. Spiking neural networks (SNN), on the other hand, offer comparable performance with significantly lower energy consumption. Hence, we propose a novel SNN inspired by the Deep ACE architecture that simultaneously performs speech enhancement and CI coding. Our model achieves competitive vocoded short-time objective intelligibility (VSTOI) and signal-to-noise ratio improvement (SNRi) scores compared to Deep ACE, while achieving more than a sixfold reduction in energy consumption.


by Zyzzyva0381 (Windy). 


2026-08-31
