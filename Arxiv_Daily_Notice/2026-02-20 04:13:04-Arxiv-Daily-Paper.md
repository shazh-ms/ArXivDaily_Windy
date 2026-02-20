# Showing new listings for Friday, 20 February 2026
Auto update papers at about 2:30am UTC (10:30am Beijing time) every weekday.


阅读 `Usage.md`了解如何使用此repo实现个性化的Arxiv论文推送

See `Usage.md` for instructions on how to personalize the repo. 


Keyword list: ['text-to-speech', 'text to speech', 'tts', 'LLM-based', 'speech', 'voice']


Excluded: []


### Today: 3papers 
#### CC-G2PnP: Streaming Grapheme-to-Phoneme and prosody with Conformer-CTC for unsegmented languages
 - **Authors:** Yuma Shirahata, Ryuichi Yamamoto
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2602.17157

 - **Pdf link:** https://arxiv.org/pdf/2602.17157

 - **Abstract**
 We propose CC-G2PnP, a streaming grapheme-to-phoneme and prosody (G2PnP) model to connect large language model and text-to-speech in a streaming manner. CC-G2PnP is based on Conformer-CTC architecture. Specifically, the input grapheme tokens are processed chunk by chunk, which enables streaming inference of phonemic and prosodic (PnP) labels. By guaranteeing minimal look-ahead size to each input token, the proposed model can consider future context in each token, which leads to stable PnP label prediction. Unlike previous streaming methods that depend on explicit word boundaries, the CTC decoder in CC-G2PnP effectively learns the alignment between graphemes and phonemes during training, making it applicable to unsegmented languages. Experiments on a Japanese dataset, which has no explicit word boundaries, show that CC-G2PnP significantly outperforms the baseline streaming G2PnP model in the accuracy of PnP label prediction.
#### Speech to Speech Synthesis for Voice Impersonation
 - **Authors:** Bjorn Johnson, Jared Levy
 - **Subjects:** Subjects:
Sound (cs.SD); Machine Learning (cs.LG); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2602.16721

 - **Pdf link:** https://arxiv.org/pdf/2602.16721

 - **Abstract**
 Numerous models have shown great success in the fields of speech recognition as well as speech synthesis, but models for speech to speech processing have not been heavily explored. We propose Speech to Speech Synthesis Network (STSSN), a model based on current state of the art systems that fuses the two disciplines in order to perform effective speech to speech style transfer for the purpose of voice impersonation. We show that our proposed model is quite powerful, and succeeds in generating realistic audio samples despite a number of drawbacks in its capacity. We benchmark our proposed model by comparing it with a generative adversarial model which accomplishes a similar task, and show that ours produces more convincing results.
#### The Cascade Equivalence Hypothesis: When Do Speech LLMs Behave Like ASR$\rightarrow$LLM Pipelines?
 - **Authors:** Jayadev Billa
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Artificial Intelligence (cs.AI); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2602.17598

 - **Pdf link:** https://arxiv.org/pdf/2602.17598

 - **Abstract**
 Current speech LLMs largely perform implicit ASR: on tasks solvable from a transcript, they are behaviorally and mechanistically equivalent to simple Whisper$\to$LLM cascades. We show this through matched-backbone testing across four speech LLMs and six tasks, controlling for the LLM backbone for the first time. Ultravox is statistically indistinguishable from its matched cascade ($\kappa{=}0.93$); logit lens reveals literal text emerging in hidden states; LEACE concept erasure confirms text representations are causally necessary in both architectures tested, collapsing accuracy to near-zero. Qwen2-Audio genuinely diverges, revealing cascade equivalence is architecture-dependent, not universal. For most deployed use cases, current speech LLMs are expensive cascades, and under noise, they are worse ones, with clean-condition advantages reversing by up to 7.6% at 0 dB.


by Zyzzyva0381 (Windy). 


2026-02-20
