# Showing new listings for Monday, 4 May 2026
Auto update papers at about 2:30am UTC (10:30am Beijing time) every weekday.


阅读 `Usage.md`了解如何使用此repo实现个性化的Arxiv论文推送

See `Usage.md` for instructions on how to personalize the repo. 


Keyword list: ['text-to-speech', 'text to speech', 'tts', 'LLM-based', 'speech', 'voice']


Excluded: []


### Today: 4papers 
#### From Birdsong to Rumbles: Classifying Elephant Calls with Out-of-Species Embeddings
 - **Authors:** Christiaan M. Geldenhuys, Thomas R. Niesler
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Machine Learning (cs.LG); Sound (cs.SD); Quantitative Methods (q-bio.QM)
 - **Arxiv link:** https://arxiv.org/abs/2605.00225

 - **Pdf link:** https://arxiv.org/pdf/2605.00225

 - **Abstract**
 We show that pretrained acoustic embeddings classify elephant vocalisations at a level approaching that of end-to-end supervised neural networks, without any fine-tuning of the embedding model. This result is of practical importance because annotated bioacoustic data are scarce and costly to obtain, leaving conventional supervised approaches prone to overfitting and to poor generalisation under domain shift. A broad range of embedding models drawn from general audio, speech, and bioacoustic domains is evaluated, all of which are either out-of-domain (containing no bioacoustic data) or out-of-species (containing no elephant call data). The embedding networks themselves remain fixed; only the lightweight downstream classifiers, which include a linear model and several small neural networks, are trained. Among the models considered, Perch 2.0 achieves the best cross-validated classification performance, attaining AUCs of 0.849 on African bush elephant (Loxodonta africana) calls and 0.936 on Asian elephant (Elephas maximus) calls, with Perch 1.0 close behind. The best-performing system is within 2.2 % of an end-to-end supervised elephant call classification system. A layerwise analysis of pretrained transformer encoders, considered as embedding models, shows that intermediate representations outperform final-layer outputs. The second layer of both wav2vec2.0 and HuBERT encodes sufficient information for effective elephant call classification; truncation at this layer therefore preserves classification performance whilst retaining only approximately 10 % of the parameters of the full network. Such compact embedding networks are well suited to on-device processing where computational resources are limited.
#### Alethia: A Foundational Encoder for Voice Deepfakes
 - **Authors:** Yi Zhu, Brahmi Dwivedi, Jayaram Raghuram, Surya Koppisetti
 - **Subjects:** Subjects:
Sound (cs.SD); Computation and Language (cs.CL); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2605.00251

 - **Pdf link:** https://arxiv.org/pdf/2605.00251

 - **Abstract**
 Existing voice deepfake detection and localization models rely heavily on representations extracted from speech foundation models (SFMs). However, downstream finetuning has now reached a state of diminishing returns. In this paper, we shift the focus to pretraining and propose a novel recipe that combines bottleneck masked embedding prediction with flow-matching based spectrogram reconstruction. The outcome, Alethia, is the first foundational audio encoder for various voice deepfake detection and localization tasks. We evaluate on $5$ different tasks with $56$ benchmark datasets, and note Alethia significantly outperforms state-of-the-art SFMs with superior robustness to real-world perturbations and zero-shot generalization to unseen domains (e.g., singing deepfakes). We also demonstrate the limitation of discrete targets in masked token prediction, and show the importance of continuous embedding prediction and generative pretraining for capturing deepfake artifacts.
#### Beyond Decodability: Reconstructing Language Model Representations with an Encoding Probe
 - **Authors:** Gaofei Shen, Martijn Bentum, Tom Lentz, Afra Alishahi, Grzegorz ChrupaÅ‚a
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2605.00607

 - **Pdf link:** https://arxiv.org/pdf/2605.00607

 - **Abstract**
 Probing is widely used to study which features can be decoded from language model representations. However, the common decoding probe approach has two limitations that we aim to solve with our new encoding probe approach: contributions of different features to model representations cannot be directly compared, and feature correlations can affect probing results. We present an Encoding Probe that reverses this direction and reconstructs internal representations of models using interpretable features. We evaluate this method on text and speech transformer models, using feature sets spanning acoustics, phonetics, syntax, lexicon, and speaker identity. Our results suggest that speaker-related effects vary strongly across different training objectives and datasets, while syntactic and lexical features contribute independently to reconstruction. These results show that the Encoding Probe provides a complementary perspective on interpreting model representations beyond decodability.
#### LASE: Language-Adversarial Speaker Encoding for Indic Cross-Script Identity Preservation
 - **Authors:** Venkata Pushpak Teja Menta
 - **Subjects:** Subjects:
Sound (cs.SD); Computation and Language (cs.CL); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2605.00777

 - **Pdf link:** https://arxiv.org/pdf/2605.00777

 - **Abstract**
 A speaker encoder used in multilingual voice cloning should treat the same speaker identically regardless of which script the audio was uttered in. Off-the-shelf encoders do not, and the failure is accent-conditional. On a 1043-pair Western-accented voice corpus across English, Hindi, Telugu, and Tamil, WavLM-base-plus-sv loses 0.082 absolute cosine similarity when the same voice changes script and ECAPA-TDNN loses 0.105. On a 1369-pair Indian-accented voice corpus, the gap shrinks to 0.006 (WavLM-SV) and 0.044 (ECAPA-TDNN). The leak is largest where it matters most for cross-script TTS: when a system projects a non-Indic-trained voice into Indic scripts. We present LASE (Language-Adversarial Speaker Encoder), a small projection head over frozen WavLM-base-plus trained with two losses: a supervised contrastive loss over voice identity, and a gradient-reversal cross-entropy against a 4-language classifier that pushes the embedding to be language-uninformative while remaining speaker-informative. Trained on 1118 quality-gated cross-script pairs synthesised from 8 commercial multilingual voices, LASE's residual gap is consistent with zero on both corpora (Delta = 0.013 Western, Delta = 0.026 Indian; both bootstrap 95% CIs include zero) and amplifies the cross-script-vs-floor margin 2.4-2.7x over both baselines. An ECAPA+GRL ablation shows the GRL objective improves either backbone but the WavLM choice contributes too. In synthetic multi-speaker diarisation, LASE matches ECAPA-TDNN on cross-script speaker recall (0.788 vs 0.789) with ~100x less training data. We release the r1 checkpoint, both corpora, and the bootstrap recipe.


by Zyzzyva0381 (Windy). 


2026-05-04
