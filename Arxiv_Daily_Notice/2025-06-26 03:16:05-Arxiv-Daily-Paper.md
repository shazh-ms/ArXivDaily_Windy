# Showing new listings for Thursday, 26 June 2025
Auto update papers at about 2:30am UTC (10:30am Beijing time) every weekday.


阅读 `Usage.md`了解如何使用此repo实现个性化的Arxiv论文推送

See `Usage.md` for instructions on how to personalize the repo. 


Keyword list: ['text-to-speech', 'text to speech', 'tts', 'LLM-based', 'speech', 'voice']


Excluded: []


### Today: 5papers 
#### MATER: Multi-level Acoustic and Textual Emotion Representation for Interpretable Speech Emotion Recognition
 - **Authors:** Hyo Jin Jon, Longbin Jin, Hyuntaek Jung, Hyunseo Kim, Donghun Min, Eun Yi Kim
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Artificial Intelligence (cs.AI); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2506.19887

 - **Pdf link:** https://arxiv.org/pdf/2506.19887

 - **Abstract**
 This paper presents our contributions to the Speech Emotion Recognition in Naturalistic Conditions (SERNC) Challenge, where we address categorical emotion recognition and emotional attribute prediction. To handle the complexities of natural speech, including intra- and inter-subject variability, we propose Multi-level Acoustic-Textual Emotion Representation (MATER), a novel hierarchical framework that integrates acoustic and textual features at the word, utterance, and embedding levels. By fusing low-level lexical and acoustic cues with high-level contextualized representations, MATER effectively captures both fine-grained prosodic variations and semantic nuances. Additionally, we introduce an uncertainty-aware ensemble strategy to mitigate annotator inconsistencies, improving robustness in ambiguous emotional expressions. MATER ranks fourth in both tasks with a Macro-F1 of 41.01% and an average CCC of 0.5928, securing second place in valence prediction with an impressive CCC of 0.6941.
#### An Exploration of ECAPA-TDNN and x-vector Speaker Representations in Zero-shot Multi-speaker TTS
 - **Authors:** Marie Kunešová, Zdeněk Hanzlíček, Jindřich Matoušek
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2506.20190

 - **Pdf link:** https://arxiv.org/pdf/2506.20190

 - **Abstract**
 Zero-shot multi-speaker text-to-speech (TTS) systems rely on speaker embeddings to synthesize speech in the voice of an unseen speaker, using only a short reference utterance. While many speaker embeddings have been developed for speaker recognition, their relative effectiveness in zero-shot TTS remains underexplored. In this work, we employ a YourTTS-based TTS system to compare three different speaker encoders - YourTTS's original H/ASP encoder, x-vector embeddings, and ECAPA-TDNN embeddings - within an otherwise fixed zero-shot TTS framework. All models were trained on the same dataset of Czech read speech and evaluated on 24 out-of-domain target speakers using both subjective and objective methods. The subjective evaluation was conducted via a listening test focused on speaker similarity, while the objective evaluation measured cosine distances between speaker embeddings extracted from synthesized and real utterances. Across both evaluations, the original H/ASP encoder consistently outperformed the alternatives, with ECAPA-TDNN showing better results than x-vectors. These findings suggest that, despite the popularity of ECAPA-TDNN in speaker recognition, it does not necessarily offer improvements for speaker similarity in zero-shot TTS in this configuration. Our study highlights the importance of empirical evaluation when reusing speaker recognition embeddings in TTS and provides a framework for additional future comparisons.
#### Lightweight Target-Speaker-Based Overlap Transcription for Practical Streaming ASR
 - **Authors:** Aleš Pražák, Marie Kunešová, Josef Psutka
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2506.20288

 - **Pdf link:** https://arxiv.org/pdf/2506.20288

 - **Abstract**
 Overlapping speech remains a major challenge for automatic speech recognition (ASR) in real-world applications, particularly in broadcast media with dynamic, multi-speaker interactions. We propose a light-weight, target-speaker-based extension to an existing streaming ASR system to enable practical transcription of overlapping speech with minimal computational overhead. Our approach combines a speaker-independent (SI) model for standard operation with a speaker-conditioned (SC) model selectively applied in overlapping scenarios. Overlap detection is achieved using a compact binary classifier trained on frozen SI model output, offering accurate segmentation at negligible cost. The SC model employs Feature-wise Linear Modulation (FiLM) to incorporate speaker embeddings and is trained on synthetically mixed data to transcribe only the target speaker. Our method supports dynamic speaker tracking and reuses existing modules with minimal modifications. Evaluated on a challenging set of Czech television debates with 16% overlap, the system reduced WER on overlapping segments from 68.0% (baseline) to 35.78% while increasing total computational load by only 44%. The proposed system offers an effective and scalable solution for overlap transcription in continuous ASR services.
#### The role of audio-visual integration in the time course of phonetic encoding in self-supervised speech models
 - **Authors:** Yi Wang, Oli Danyi Liu, Peter Bell
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD); Image and Video Processing (eess.IV)
 - **Arxiv link:** https://arxiv.org/abs/2506.20361

 - **Pdf link:** https://arxiv.org/pdf/2506.20361

 - **Abstract**
 Human speech perception is multimodal. In natural speech, lip movements can precede corresponding voicing by a non-negligible gap of 100-300 ms, especially for specific consonants, affecting the time course of neural phonetic encoding in human listeners. However, it remains unexplored whether self-supervised learning models, which have been used to simulate audio-visual integration in humans, can capture this asynchronicity between audio and visual cues. We compared AV-HuBERT, an audio-visual model, with audio-only HuBERT, by using linear classifiers to track their phonetic decodability over time. We found that phoneme information becomes available in AV-HuBERT embeddings only about 20 ms before HuBERT, likely due to AV-HuBERT's lower temporal resolution and feature concatenation process. It suggests AV-HuBERT does not adequately capture the temporal dynamics of multimodal speech perception, limiting its suitability for modeling the multimodal speech perception process.
#### CBF-AFA: Chunk-Based Multi-SSL Fusion for Automatic Fluency Assessment
 - **Authors:** Papa Séga Wade, Mihai Andries, Ioannis Kanellos, Thierry Moudenc
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2506.20243

 - **Pdf link:** https://arxiv.org/pdf/2506.20243

 - **Abstract**
 Automatic fluency assessment (AFA) remains challenging, particularly in capturing speech rhythm, pauses, and disfluencies in non-native speakers. We introduce a chunk-based approach integrating self-supervised learning (SSL) models (Wav2Vec2, HuBERT, and WavLM) selected for their complementary strengths in phonetic, prosodic, and noisy speech modeling, with a hierarchical CNN-BiLSTM framework. Speech is segmented into breath-group chunks using Silero voice activity detection (Silero-VAD), enabling fine-grained temporal analysis while mitigating over-segmentation artifacts. SSL embeddings are fused via a learnable weighted mechanism, balancing acoustic and linguistic features, and enriched with chunk-level fluency markers (e.g., speech rate, pause durations, n-gram repetitions). The CNN-BiLSTM captures local and long-term dependencies across chunks. Evaluated on Avalinguo and Speechocean762, our approach improves F1-score by 2.8 and Pearson correlation by 6.2 points over single SSL baselines on Speechocean762, with gains of 4.2 F1-score and 4.0 Pearson points on Avalinguo, surpassing this http URL-based segmentation baselines. These findings highlight chunk-based multi-SSL fusion for robust fluency evaluation, though future work should explore generalization to dialects with irregular prosody.


by Zyzzyva0381 (Windy). 


2025-06-26
