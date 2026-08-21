# Showing new listings for Friday, 21 August 2026
Auto update papers at about 2:30am UTC (10:30am Beijing time) every weekday.


阅读 `Usage.md`了解如何使用此repo实现个性化的Arxiv论文推送

See `Usage.md` for instructions on how to personalize the repo. 


Keyword list: ['text-to-speech', 'text to speech', 'tts', 'LLM-based', 'speech', 'voice']


Excluded: []


### Today: 5papers 
#### Listening Forward: Next Patch Embedding Prediction Enables Scalable Audio Learners
 - **Authors:** Umberto Cappellazzo, Xubo Liu, Stavros Petridis, Maja Pantic
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Artificial Intelligence (cs.AI); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2608.19863

 - **Pdf link:** https://arxiv.org/pdf/2608.19863

 - **Abstract**
 Self-supervised learning (SSL) has driven substantial progress in audio representation learning, though existing methods have increasingly relied on elaborate pre-training recipes to reach competitive performance. A markedly different pre-training philosophy underpins the most influential progress in language modeling and, more recently, in visual representation learning: rather than train encoders as static feature extractors, models are trained to predict the next element, a discrete token or a continuous embedding, from the preceding context. Autoregressive prediction thereby provides a unified pre-training interface that transfers across modalities, compelling the model to learn the underlying data distribution. We ask whether such a simple causal paradigm can yield strong audio learners, given that audio's temporal structure makes autoregressive prediction of patch embeddings a natural fit. We introduce NAPE (Next-Audio-Patch-Embedding prediction), a self-supervised framework in which a causal Transformer predicts each next patch embedding of a log-mel spectrogram from the previous ones, using causal masking and stop-gradient as its sole training signal. The design is intentionally minimalist, avoiding reconstruction decoders, acoustic tokenizers, student-teacher setups, and auxiliary regularization losses. Across six audio and speech benchmarks, NAPE achieves state-of-the-art fine-tuning performance on several tasks, scales consistently across encoder sizes, and yields strong linear-probing results. NAPE also produces structured attention patterns without explicit supervision.
#### Explainability by Design: Structured Kolmogorov-Arnold Networks over Probabilistic Attributes for Speech Deepfake Source Tracing
 - **Authors:** Hoang H. Pham, Manasi Chhibber, Tomi H. Kinnunen
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2608.20213

 - **Pdf link:** https://arxiv.org/pdf/2608.20213

 - **Abstract**
 Modern speech synthesizers can produce highly realistic speech, making source tracing (i.e. identifying the generator behind a spoofed utterance) increasingly important for forensics, online content provenance, and platform accountability. Building on our prior work on transparent probabilistic attributes, which represent utterances as probability distributions over synthesizer sub-components, we extend speech deepfake source tracing with two key ingredients: multi-task training of the probabilistic attribute extractors and a structured Kolmogorov--Arnold Network (KAN) for attack classification. The probabilistic features are estimated jointly with a multi-task learning module built on a shared AASIST or SSL-AASIST countermeasure backbone. The resulting probabilistic feature embedding is classified by a structured KAN whose topology follows known attribute-to-attack relationships. This provides interpretability by construction: the architecture reflects the generative hierarchy of attacks, while KAN feature-importance scores quantify each probabilistic feature's contribution without post-hoc explainers such as SHAP. On ASVspoof2019-attr-17, the extended framework achieves balanced accuracies above 99% for all seven probabilistic feature extractors, with EERs of 0.16% to 0.07%, and 99.64% balanced accuracy with 0.11% EER for 17-class attack classification. Our revised model outperforms the earlier two-stage baselines, in addition to demonstrating reliable interpretability, with importance scores consistent with SHAP values, and stable results across batch sizes. These findings highlight the potential of structured KAN for speech deepfake source tracing that is both accurate and interpretable by design. For transparency and reproducibility, our codebase is publicly available: this https URL.
#### Represented but Ignored: A Causal Account of Prosodic Underuse in Audio-Language Models
 - **Authors:** Linkai Peng, Baorian Nuchged
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Artificial Intelligence (cs.AI); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2608.19211

 - **Pdf link:** https://arxiv.org/pdf/2608.19211

 - **Abstract**
 Human speech is richly expressive, with prosody carrying linguistic and emotional information beyond the lexical content. A capable large audio-language model (audio-LLM) should therefore support expressive speech understanding, not only transcribing what was said but also interpreting how it was said. Yet behavioral evaluations alone cannot reveal why a model fails on prosodic input. An error may reflect loss of acoustic information, incorrect internal interpretation, or failure to use a representation that is already available inside the model. We introduce a stage-specific probe ladder for localizing these failure modes in audio-LLMs. Across four understanding-only audio-LLMs, prosodic information is usually preserved in the audio path and decodable in late LLM states. Yet it is only partially expressed in the model's final response. We test the causal status of this latent representation with targeted hidden-state interventions. Every intervention shifts the answer distribution in the predicted direction, and in most model--task cells a single edit at the relevant layer is sufficient to drive the model toward the suppressed prosodic decision, though this recovery is directional rather than a selective restoration of the correct class. Feature-level analysis further suggests that this recoverable signal can be expressed through a small subspace. Some of the highest-attribution features in this analysis align with acoustic cues known to carry prosodic information. Within the matched-content contrasts we test, these results locate the recurring bottleneck not in perceiving prosody but in using it. Models that hear and correctly represent a prosodic cue can still fail to express it in their answers.
#### A Speech Corpus for Mizo Automatic Speech Recognition: Whisper and SraVaani 1.0 Fine-Tuning with Morphology-Aware Evaluation
 - **Authors:** Priyankoo Sarmah, Sanasam Ranbir Singh, Lalhmingmawia
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2608.19361

 - **Pdf link:** https://arxiv.org/pdf/2608.19361

 - **Abstract**
 This study reports the development of an Automatic Speech Recognition (ASR) system in Mizo, a low-resource language. The development included collecting 17.62 hours of speech data, curating it, and fine-tuning the Mizo ASR system with three Whisper multilingual models and with the SraVaani 1.0 Indic multilingual model. Whisper-large-v3 achieved the lowest conventional WER (18.08%), while morphology-aware evaluation yielded a WER of 7.22%. Zero-shot evaluation of the SraVaani 1.0 Indic multilingual model yielded a WER of 58.27%, while Mizo-specific fine-tuning reduced the conventional WER to 29.45% and the morphology-aware WER to 17.93%. The results demonstrate that the Whisper model can achieve a substantially low WER, even when adapted to an unseen language. In contrast, SraVaani 1.0 supports the Mizo language in its multilingual model; however, fine-tuning with carefully curated Mizo speech data substantially improves its performance.
#### Tracking the Trend in How Speech Synthesizers Deceive People
 - **Authors:** Milan Šalko, Anton Firc, Kamil Malinka, Vojtěch Staněk, Martin Perešini, Filip Pleško, Jakub Reš
 - **Subjects:** Subjects:
Cryptography and Security (cs.CR); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2608.19959

 - **Pdf link:** https://arxiv.org/pdf/2608.19959

 - **Abstract**
 Advances in speech synthesis have made deepfake audio highly realistic. Earlier studies reported 70-80% human detection accuracy, but relied primarily on older synthesizers. We compare human detection for three selected voice synthesis tools released in 2019, 2022, and 2024 with 82 IT professionals, and benchmark humans against six pretrained detectors on the same material. For fully synthetic speech (full spoofs), the F1 score drops from about 90% for RTVC and YourTTS to 48% for ElevenLabs, although listeners were explicitly warned that deepfakes were present. For partial spoofing, where only one sentence of an utterance is altered, strict accuracy falls to 9%, and listeners classify the synthetic sentence as bona fide 77% of the time. Humans and detectors fail in complementary ways, and neither reliably localizes short manipulations. Additionally, listeners increasingly mislabel bona fide speech as fake, eroding trust in unmanipulated audio. These findings show that human perception alone is unreliable for the selected modern and partial-spoof conditions and motivate procedural verification, provenance, watermarking, and segment-level detection.


by Zyzzyva0381 (Windy). 


2026-08-21
