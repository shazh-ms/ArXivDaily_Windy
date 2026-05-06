# Showing new listings for Wednesday, 6 May 2026
Auto update papers at about 2:30am UTC (10:30am Beijing time) every weekday.


阅读 `Usage.md`了解如何使用此repo实现个性化的Arxiv论文推送

See `Usage.md` for instructions on how to personalize the repo. 


Keyword list: ['text-to-speech', 'text to speech', 'tts', 'LLM-based', 'speech', 'voice']


Excluded: []


### Today: 3papers 
#### Assessing the Impact of Noise and Speech Enhancement on the Intelligibility of Speech Codecs
 - **Authors:** Lyonel Behringer, Anna Leschanowsky, Anjana Rajasekhar, Emily Kratsch, Guillaume Fuchs
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2605.03776

 - **Pdf link:** https://arxiv.org/pdf/2605.03776

 - **Abstract**
 Preserving speech intelligibility is a minimum requirement for speech codecs in communication. Recently, very low-bitrate neural codecs have gained interest for replacing classical codecs, reinforcing the need to evaluate whether intelligibility is preserved in realistic scenarios. In this paper, we evaluate the intelligibility and listening effort of classical and neural speech codecs in clean and noisy conditions. Further, we assess the impact of speech enhancement (SE) before coding, simulating a possible audio processing pipeline. The results show that classical codecs are more noise robust than neural codecs. Further, SE can lead to significant intelligibility and listening effort improvements for codecs otherwise negatively affected by noise. Listening effort reveals nuanced differences when intelligibility is saturated. Lastly, objective intelligibility based on automatic speech recognition is highly correlated with subjective intelligibility scores averaged per condition.
#### Phoneme-Level Deepfake Detection Across Emotional Conditions Using Self-Supervised Embeddings
 - **Authors:** Vamshi Nallaguntla, Shruti Kshirsagar, Anderson R. Avila
 - **Subjects:** Subjects:
Sound (cs.SD); Machine Learning (cs.LG); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2605.03079

 - **Pdf link:** https://arxiv.org/pdf/2605.03079

 - **Abstract**
 Recent advances in emotional voice conversion (EVC) have enabled the generation of expressive synthetic speech, raising new concerns in audio deepfake detection. Existing approaches treat speech as a homogeneous signal and largely overlook its internal phonetic structure, limiting their interpretability in emotionally conditioned settings. In this work, we propose a phoneme-level framework to analyze emotionally manipulated synthetic speech using real and EVC-generated speech under matched emotional conditions with shared transcripts, phoneme-aligned TextGrids, and WavLM-based embeddings. Our results show that phoneme behavior varies across categories, with complex vowels and fricatives exhibiting higher divergence while simpler phonemes remain more stable. Phonemes with larger distributional differences are also found to be more easily detected, consistently across multiple emotions and synthesis systems. These findings demonstrate that phoneme-level analysis is an effective and interpretable approach for detecting emotionally manipulated synthetic speech.
#### MiniMind-O Technical Report: An Open Small-Scale Speech-Native Omni Model
 - **Authors:** Jingyao Gong
 - **Subjects:** Subjects:
Sound (cs.SD); Multimedia (cs.MM); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2605.03937

 - **Pdf link:** https://arxiv.org/pdf/2605.03937

 - **Abstract**
 MiniMind-O is an open 0.1B-scale omni model built on the MiniMind language model. It accepts text, speech, and image inputs, and returns both text and streaming speech. The release includes model code, checkpoints, and the main Parquet training datasets for text-to-audio, image-to-text, and audio-to-audio training, making the complete interaction loop directly inspectable. The model uses a full MiniMind backbone as the Thinker and an independent four-layer Talker made from MiniMind blocks. Frozen SenseVoice-Small and SigLIP2 encoders provide speech and image features, which are mapped by lightweight MLP projectors and injected at modality-placeholder positions. The Talker reads a middle-layer Thinker state together with an autoregressive eight-layer Mimi-code buffer. Speaker control is handled by a dedicated speaker token, right-aligned reference codec prompts, and precomputed CAM++ speaker embeddings, so voice conditioning remains part of the audio-code context rather than a separate TTS module. With a 768-dimensional Talker, the dense and MoE variants reach average CERs of 0.0897 and 0.0900 in Thinker--Talker consistency evaluation, with overall voice-cloning similarities of 0.5995 and 0.5937. Beyond reporting a working system, the paper identifies three scale-critical design choices for small omni models: middle-layer semantic bridging, a released multimodal sequence format, and a parameter-efficient eight-codebook interface.


by Zyzzyva0381 (Windy). 


2026-05-06
