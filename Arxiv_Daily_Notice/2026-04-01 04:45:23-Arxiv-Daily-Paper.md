# Showing new listings for Wednesday, 1 April 2026
Auto update papers at about 2:30am UTC (10:30am Beijing time) every weekday.


阅读 `Usage.md`了解如何使用此repo实现个性化的Arxiv论文推送

See `Usage.md` for instructions on how to personalize the repo. 


Keyword list: ['text-to-speech', 'text to speech', 'tts', 'LLM-based', 'speech', 'voice']


Excluded: []


### Today: 5papers 
#### Asymmetric Encoder-Decoder Based on Time-Frequency Correlation for Speech Separation
 - **Authors:** Ui-Hyeop Shin, Hyung-Min Park
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2603.29097

 - **Pdf link:** https://arxiv.org/pdf/2603.29097

 - **Abstract**
 Speech separation in realistic acoustic environments remains challenging because overlapping speakers, background noise, and reverberation must be resolved simultaneously. Although recent time-frequency (TF) domain models have shown strong performance, most still rely on late-split architectures, where speaker disentanglement is deferred to the final stage, creating an information bottleneck and weakening discriminability under adverse conditions. To address this issue, we propose SR-CorrNet, an asymmetric encoder-decoder framework that introduces the separation-reconstruction (SepRe) strategy into a TF dual-path backbone. The encoder performs coarse separation from mixture observations, while the weight-shared decoder progressively reconstructs speaker-discriminative features with cross-speaker interaction, enabling stage-wise refinement. To complement this architecture, we formulate speech separation as a structured correlation-to-filter problem: spatio-spectro-temporal correlations computed from the observations are used as input features, and the corresponding deep filters are estimated to recover target signals. We further incorporate an attractor-based dynamic split module to adapt the number of output streams to the actual speaker configuration. Experimental results on WSJ0-2/3/4/5Mix, WHAMR!, and LibriCSS demonstrate consistent improvements across anechoic, noisy-reverberant, and real-recorded conditions in both single- and multi-channel settings, highlighting the effectiveness of TF-domain SepRe with correlation-based filter estimation for speech separation.
#### Advancing LLM-based phoneme-to-grapheme for multilingual speech recognition
 - **Authors:** Lukuang Dong, Ziwei Li, Saierdaer Yusuyin, Xianyu Zhao, Zhijian Ou
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Computation and Language (cs.CL); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2603.29217

 - **Pdf link:** https://arxiv.org/pdf/2603.29217

 - **Abstract**
 Phoneme-based ASR factorizes recognition into speech-to-phoneme (S2P) and phoneme-to-grapheme (P2G), enabling cross-lingual acoustic sharing while keeping language-specific orthography in a separate module. While large language models (LLMs) are promising for P2G, multilingual P2G remains challenging due to language-aware generation and severe cross-language data imbalance. We study multilingual LLM-based P2G on the ten-language CV-Lang10 benchmark. We examine robustness strategies that account for S2P uncertainty, including DANP and Simplified SKM (S-SKM). S-SKM is a Monte Carlo approximation that avoids CTC-based S2P probability weighting in P2G training. Robust training and low-resource oversampling reduce the average WER from 10.56% to 7.66%.
#### IQRA 2026: Interspeech Challenge on Automatic Assessment Pronunciation for Modern Standard Arabic (MSA)
 - **Authors:** Yassine El Kheir, Amit Meghanani, Mostafa Shahin, Omnia Ibrahim, Shammur Absar Chowdhury, Nada AlMarwani, Youssef Elshahawy, Ahmed Ali
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2603.29087

 - **Pdf link:** https://arxiv.org/pdf/2603.29087

 - **Abstract**
 We present the findings of the second edition of the IQRA Interspeech Challenge, a challenge on automatic Mispronunciation Detection and Diagnosis (MDD) for Modern Standard Arabic (MSA). Building on the previous edition, this iteration introduces \textbf{Iqra\_Extra\_IS26}, a new dataset of authentic human mispronounced speech, complementing the existing training and evaluation resources. Submitted systems employed a diverse range of approaches, spanning CTC-based self-supervised learning models, two-stage fine-tuning strategies, and using large audio-language models. Compared to the first edition, we observe a substantial jump of \textbf{0.28 in F1-score}, attributable both to novel architectures and modeling strategies proposed by participants and to the additional authentic mispronunciation data made available. These results demonstrate the growing maturity of Arabic MDD research and establish a stronger foundation for future work in Arabic pronunciation assessment.
#### LongCat-AudioDiT: High-Fidelity Diffusion Text-to-Speech in the Waveform Latent Space
 - **Authors:** Detai Xin, Shujie Hu, Chengzuo Yang, Chen Huang, Guoqiao Yu, Guanglu Wan, Xunliang Cai
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2603.29339

 - **Pdf link:** https://arxiv.org/pdf/2603.29339

 - **Abstract**
 We present LongCat-AudioDiT, a novel, non-autoregressive diffusion-based text-to-speech (TTS) model that achieves state-of-the-art (SOTA) performance. Unlike previous methods that rely on intermediate acoustic representations such as mel-spectrograms, the core innovation of LongCat-AudioDiT lies in operating directly within the waveform latent space. This approach effectively mitigates compounding errors and drastically simplifies the TTS pipeline, requiring only a waveform variational autoencoder (Wav-VAE) and a diffusion backbone. Furthermore, we introduce two critical improvements to the inference process: first, we identify and rectify a long-standing training-inference mismatch; second, we replace traditional classifier-free guidance with adaptive projection guidance to elevate generation quality. Experimental results demonstrate that, despite the absence of complex multi-stage training pipelines or high-quality human-annotated datasets, LongCat-AudioDiT achieves SOTA zero-shot voice cloning performance on the Seed benchmark while maintaining competitive intelligibility. Specifically, our largest variant, LongCat-AudioDiT-3.5B, outperforms the previous SOTA model (Seed-TTS), improving the speaker similarity (SIM) scores from 0.809 to 0.818 on Seed-ZH, and from 0.776 to 0.797 on Seed-Hard. Finally, through comprehensive ablation studies and systematic analysis, we validate the effectiveness of our proposed modules. Notably, we investigate the interplay between the Wav-VAE and the TTS backbone, revealing the counterintuitive finding that superior reconstruction fidelity in the Wav-VAE does not necessarily lead to better overall TTS performance. Code and model weights are released to foster further research within the speech community.
#### A Comprehensive Corpus of Biomechanically Constrained Piano Chords: Generation, Analysis, and Implications for Voicing and Psychoacoustics
 - **Authors:** Mahesh Ramani
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2603.29710

 - **Pdf link:** https://arxiv.org/pdf/2603.29710

 - **Abstract**
 I present the generation and analysis of the largest known open-source corpus of playable piano chords (approximately 19.3 million entries). This dataset enumerates the two-handed search space subject to biomechanical constraints (two hands, each with 1.5 octave reach) to an unprecedented extent. To demonstrate the corpus's utility, the relationship between voicing shape and psychoacoustic targets was modeled. Harmonicity proved intrinsic to pitch-class identity: voicing statistics added negligible variance ($\Delta R^2 \approx 0.014\%$, $p \approx 0.13$). Conversely, voicing significantly predicted dissonance ($\Delta R^2 \approx 6.75\%$, $p \approx 0.0008$). Crucially, skewness ($\beta \approx +0.145$) was approximately 5.8$\times$ more effective than spread ($\beta \approx -0.025$) at predicting roughness. The analysis challenges the pedagogical emphasis on ``spread'': skewness is a stronger predictor of dissonance than spread. This suggests that clarity in ``open voicings'' is driven less by width than by negative skewness; achieving lower-register clearance by placing wide gaps at the bottom and allowing tighter clustering in the treble. The results demonstrate the corpus's ability to enable future research, especially in areas such as generative modeling, voice-leading topology, and psychoacoustic analysis.


by Zyzzyva0381 (Windy). 


2026-04-01
