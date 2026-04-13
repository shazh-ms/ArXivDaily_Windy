# Showing new listings for Monday, 13 April 2026
Auto update papers at about 2:30am UTC (10:30am Beijing time) every weekday.


阅读 `Usage.md`了解如何使用此repo实现个性化的Arxiv论文推送

See `Usage.md` for instructions on how to personalize the repo. 


Keyword list: ['text-to-speech', 'text to speech', 'tts', 'LLM-based', 'speech', 'voice']


Excluded: []


### Today: 7papers 
#### Enhancing Conversational TTS with Cascaded Prompting and ICL-Based Online Reinforcement Learning
 - **Authors:** Zhicheng Ouyang, Seong-Gyun Leem, Bach Viet Do, Haibin Wu, Ariya Rastrow, Yuzong Liu, Florian Metze
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2604.08709

 - **Pdf link:** https://arxiv.org/pdf/2604.08709

 - **Abstract**
 Conversational AI has made significant progress, yet generating expressive and controllable text-to-speech (TTS) remains challenging. Specifically, controlling fine-grained voice styles and emotions is notoriously difficult and typically requires massive amounts of heavily annotated training data. To overcome this data bottleneck, we present a scalable, data-efficient cascaded framework that pairs textual style tokens with human-curated, high-quality audio prompts. This approach enables single-shot adaptation to fine-grained speaking styles and character voices. In the context of TTS, this audio prompting acts as In-Context Learning (ICL), guiding the model's prosody and timbre without requiring massive parameter updates or large-scale retraining. To further enhance generation quality and mitigate hallucinations, we introduce a novel ICL-based online reinforcement learning (RL) strategy. This strategy directly optimizes the autoregressive prosody model using subjective aesthetic rewards while being constrained by Connectionist Temporal Classification (CTC) alignment to preserve intelligibility. Comprehensive human perception evaluations demonstrate significant improvements in both the naturalness and expressivity of the synthesized speech, establishing the efficacy of our ICL-based online RL approach.
#### PS-TTS: Phonetic Synchronization in Text-to-Speech for Achieving Natural Automated Dubbing
 - **Authors:** Changi Hong, Yoonah Song, Hwayoung Park, Chaewoon Bang, Dayeon Gu, Do Hyun Lee, Hong Kook Kim
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Artificial Intelligence (cs.AI)
 - **Arxiv link:** https://arxiv.org/abs/2604.09111

 - **Pdf link:** https://arxiv.org/pdf/2604.09111

 - **Abstract**
 Recently, artificial intelligence-based dubbing technology has advanced, enabling automated dubbing (AD) to convert the source speech of a video into target speech in different languages. However, natural AD still faces synchronization challenges such as duration and lip-synchronization (lip-sync), which are crucial for preserving the viewer experience. Therefore, this paper proposes a synchronization method for AD processes that paraphrases translated text, comprising two steps: isochrony for timing constraints and phonetic synchronization (PS) to preserve lip-sync. First, we achieve isochrony by paraphrasing the translated text with a language model, ensuring the target speech duration matches that of the source speech. Second, we introduce PS, which employs dynamic time warping (DTW) with local costs of vowel distances measured from training data so that the target text composes vowels with pronunciations similar to source vowels. Third, we extend this approach to PSComet, which jointly considers semantic and phonetic similarity to preserve meaning better. The proposed methods are incorporated into text-to-speech systems, PS-TTS and PS-Comet TTS. The performance evaluation using Korean and English lip-reading datasets and a voice-actor dubbing dataset demonstrates that both systems outperform TTS without PS on several objective metrics and outperform voice actors in Korean-to-English and English-to-Korean dubbing. We extend the experiments to French, testing all pairs among these languages to evaluate cross-linguistic applicability. Across all language pairs, PS-Comet performed best, balancing lip-sync accuracy with semantic preservation, confirming that PS-Comet achieves more accurate lip-sync with semantic preservation than PS alone.
#### Phonemes vs. Projectors: An Investigation of Speech-Language Interfaces for LLM-based ASR
 - **Authors:** Ziwei Li, Lukuang Dong, Saierdaer Yusuyin, Xianyu Zhao, Zhijian Ou
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2604.09332

 - **Pdf link:** https://arxiv.org/pdf/2604.09332

 - **Abstract**
 Integrating pretrained speech encoders with large language models (LLMs) is promising for ASR, but performance and data efficiency depend on the speech-language interface. A common choice is a learned projector that maps encoder features into the LLM embedding space, whereas an alternative is to expose discrete phoneme sequences to the LLM. Using the same encoder and LLM backbones, we compare phoneme-based and vanilla projector-based interfaces in high-resource English and low-resource Tatar. We also propose a BPE-phoneme interface that groups frequent local phoneme patterns while preserving explicit word-boundary cues for phoneme-to-grapheme generation. On LibriSpeech, the phoneme-based interface is competitive with the vanilla projector, and the BPE-phoneme interface yields further gains. On Tatar, the phoneme-based interface substantially outperforms the vanilla projector. We further find that phoneme supervision yields a phoneme-informed hybrid interface that is stronger than the vanilla projector.
#### Data Selection Effects on Self-Supervised Learning of Audio Representations for French Audiovisual Broadcasts
 - **Authors:** Valentin Pelloin, Lina Bekkali, Reda Dehak, David Doukhan
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2604.09472

 - **Pdf link:** https://arxiv.org/pdf/2604.09472

 - **Abstract**
 Audio and speech self-supervised encoder models are now widely used for a lot of different tasks. Many of these models are often trained on clean segmented speech content such as LibriSpeech. In this paper, we look into how the pretraining datasets of such SSL (Self-Supervised Learning) models impact their downstream results. We build a large pretraining corpus of highly diverse TV and Radio broadcast audio content, which we describe with automatic tools. We use these annotations to build smaller subsets, which we use to train audio SSL models. Then, we evaluate the models on multiple downstream tasks such as automatic speech recognition, voice activity and music detection, or speaker recognition. The results show the potential of pretraining SSL models on diverse audio content without restricting it to speech. We also perform a membership inference attack to evaluate the encoder ability to memorize their training datasets, which highlight the importance of data deduplication. This unified training could bridge speech and music machine learning communities.
#### Neural networks for Text-to-Speech evaluation
 - **Authors:** Ilya Trofimenko, David Kocharyan, Aleksandr Zaitsev, Pavel Repnikov, Mark Levin, Nikita Shevtsov
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Artificial Intelligence (cs.AI); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2604.08562

 - **Pdf link:** https://arxiv.org/pdf/2604.08562

 - **Abstract**
 Ensuring that Text-to-Speech (TTS) systems deliver human-perceived quality at scale is a central challenge for modern speech technologies. Human subjective evaluation protocols such as Mean Opinion Score (MOS) and Side-by-Side (SBS) comparisons remain the de facto gold standards, yet they are expensive, slow, and sensitive to pervasive assessor biases. This study addresses these barriers by formulating, and implementing a suite of novel neural models designed to approximate expert judgments in both relative (SBS) and absolute (MOS) settings. For relative assessment, we propose NeuralSBS, a HuBERT-backed model achieving 73.7% accuracy (on SOMOS dataset). For absolute assessment, we introduce enhancements to MOSNet using custom sequence-length batching, as well as WhisperBert, a multimodal stacking ensemble that combines Whisper audio features and BERT textual embeddings via weak learners. Our best MOS models achieve a Root Mean Square Error (RMSE) of ~0.40, significantly outperforming the human inter-rater RMSE baseline of 0.62. Furthermore, our ablation studies reveal that naively fusing text via cross-attention can degrade performance, highlighting the effectiveness of ensemble-based stacking over direct latent fusion. We additionally report negative results with SpeechLM-based architectures and zero-shot LLM evaluators (Qwen2-Audio, Gemini 2.5 flash preview), reinforcing the necessity of dedicated metric learning frameworks.
#### Script Collapse in Multilingual ASR: Defining and Measuring Script Fidelity Rate
 - **Authors:** Hanif Rahman
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2604.08786

 - **Pdf link:** https://arxiv.org/pdf/2604.08786

 - **Abstract**
 Word error rate (WER) is the dominant metric for automatic speech recognition, yet it cannot detect a systematic failure mode: models that produce fluent output in the wrong writing system. We define Script Fidelity Rate (SFR), the fraction of hypothesis characters in the target script block, computable without reference transcriptions, and report the first systematic measurement of script collapse across six languages spanning four writing systems (Pashto, Urdu, Hindi, Bengali, Malayalam, Somali) and nine ASR models on FLEURS test sets. Across 53 evaluated model-language pairs, 18 (34%; 95% Wilson CI: 23-47%) exhibit script collapse (SFR < 10%); MMS-1B and SeamlessM4T-v2 maintain SFR above 99% on every language evaluated, confirming that SFR correctly identifies high fidelity where it is present. We identify three distinct collapse patterns: Latin phonetic substitution (smaller Whisper on Indic languages), Arabic substitution for Somali's Latin-script orthography, and Devanagari substitution where larger Whisper models treat all Indic audio as Hindi, a failure present even in Whisper large-v3.
#### DialogueSidon: Recovering Full-Duplex Dialogue Tracks from In-the-Wild Dialogue Audio
 - **Authors:** Wataru Nakata, Yuki Saito, Kazuki Yamauchi, Emiru Tsunoo, Hiroshi Saruwatari
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2604.09344

 - **Pdf link:** https://arxiv.org/pdf/2604.09344

 - **Abstract**
 Full-duplex dialogue audio, in which each speaker is recorded on a separate track, is an important resource for spoken dialogue research, but is difficult to collect at scale. Most in-the-wild two-speaker dialogue is available only as degraded monaural mixtures, making it unsuitable for systems requiring clean speaker-wise signals. We propose DialogueSidon, a model for joint restoration and separation of degraded monaural two-speaker dialogue audio. DialogueSidon combines a variational autoencoder (VAE) operates on the speech self-supervised learning (SSL) model feature, which compresses SSL model features into a compact latent space, with a diffusion-based latent predictor that recovers speaker-wise latent representations from the degraded mixture. Experiments on English, multilingual, and in-the-wild dialogue datasets show that DialogueSidon substantially improves intelligibility and separation quality over a baseline, while also achieving much faster inference.


by Zyzzyva0381 (Windy). 


2026-04-13
