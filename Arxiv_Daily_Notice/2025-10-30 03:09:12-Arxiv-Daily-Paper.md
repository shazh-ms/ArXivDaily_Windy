# Showing new listings for Thursday, 30 October 2025
Auto update papers at about 2:30am UTC (10:30am Beijing time) every weekday.


阅读 `Usage.md`了解如何使用此repo实现个性化的Arxiv论文推送

See `Usage.md` for instructions on how to personalize the repo. 


Keyword list: ['text-to-speech', 'text to speech', 'tts', 'LLM-based', 'speech', 'voice']


Excluded: []


### Today: 5papers 
#### Separating peripheral and higher-level effects on speech intelligibility using a hearing loss simulator and an objective intelligibility measure
 - **Authors:** Toshio Irino, Ayako Yamamoto, Fuki Miyazaki
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2510.25235

 - **Pdf link:** https://arxiv.org/pdf/2510.25235

 - **Abstract**
 This paper presents a new method for separating the effects of peripheral hearing loss (HL) and higher-level processes on speech intelligibility (SI). In a previous study, we conducted an SI experiment with 14 older adult (OA) listeners, using speech-in-noise sounds that were either processed with an ideal ratio mask (IRM) enhancement technique or left unprocessed. The current study involved an SI experiment with 15 young, normal-hearing (YNH) listeners. This experiment used simulated HL sounds processed with the WHIS simulator that reflected the hearing level of a specific OA from the previous study. The results showed that the target OA's SI scores were higher than the average YNH scores. This implies that the target OA's higher-level processes may be more effective than those of the average YNH. To understand the characteristics of other OAs, we used the GESI objective intelligibility measure to predict SI. First, we confirmed that GESI could fairly accurately predict the SI scores for both the YNH and OA listeners. Next, we predicted the SI scores of the 14 OA listeners using the parameters estimated in the YNH experiment. The results showed that some OAs had higher SI scores than the average YNH, while one OA had lower scores. These differences in SI scores may reflect variations in the efficiency of higher-level this http URL results imply that WHIS and GESI could facilitate contrastive experiments between YNH and OA listeners, regardless of hearing level. This would allow us to study the effects of higher-level processes in OA listeners individually.
#### PitchFlower: A flow-based neural audio codec with pitch controllability
 - **Authors:** Diego Torres, Axel Roebel, Nicolas Obin
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Machine Learning (cs.LG)
 - **Arxiv link:** https://arxiv.org/abs/2510.25566

 - **Pdf link:** https://arxiv.org/pdf/2510.25566

 - **Abstract**
 We present PitchFlower, a flow-based neural audio codec with explicit pitch controllability. Our approach enforces disentanglement through a simple perturbation: during training, F0 contours are flattened and randomly shifted, while the true F0 is provided as conditioning. A vector-quantization bottleneck prevents pitch recovery, and a flow-based decoder generates high quality audio. Experiments show that PitchFlower achieves more accurate pitch control than WORLD at much higher audio quality, and outperforms SiFiGAN in controllability while maintaining comparable quality. Beyond pitch, this framework provides a simple and extensible path toward disentangling other speech attributes.
#### Lost in Phonation: Voice Quality Variation as an Evaluation Dimension for Speech Foundation Models
 - **Authors:** Harm Lameris, Shree Harsha Bokkahalli Satish, Joakim Gustafson, Éva Székely
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Artificial Intelligence (cs.AI); Computation and Language (cs.CL)
 - **Arxiv link:** https://arxiv.org/abs/2510.25577

 - **Pdf link:** https://arxiv.org/pdf/2510.25577

 - **Abstract**
 Recent advances in speech foundation models (SFMs) have enabled the direct processing of spoken language from raw audio, bypassing intermediate textual representations. This capability allows SFMs to be exposed to, and potentially respond to, rich paralinguistic variations embedded in the input speech signal. One under-explored dimension of paralinguistic variation is voice quality, encompassing phonation types such as creaky and breathy voice. These phonation types are known to influence how listeners infer affective state, stance and social meaning in speech. Existing benchmarks for speech understanding largely rely on multiple-choice question answering (MCQA) formats, which are prone to failure and therefore unreliable in capturing the nuanced ways paralinguistic features influence model behaviour. In this paper, we probe SFMs through open-ended generation tasks and speech emotion recognition, evaluating whether model behaviours are consistent across different phonation inputs. We introduce a new parallel dataset featuring synthesized modifications to voice quality, designed to evaluate SFM responses to creaky and breathy voice. Our work provides the first examination of SFM sensitivity to these particular non-lexical aspects of speech perception.
#### Evaluating Emotion Recognition in Spoken Language Models on Emotionally Incongruent Speech
 - **Authors:** Pedro Corrêa, João Lima, Victor Moreno, Paula Dornhofer Paro Costa
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2510.25054

 - **Pdf link:** https://arxiv.org/pdf/2510.25054

 - **Abstract**
 Advancements in spoken language processing have driven the development of spoken language models (SLMs), designed to achieve universal audio understanding by jointly learning text and audio representations for a wide range of tasks. Although promising results have been achieved, there is growing discussion regarding these models' generalization capabilities and the extent to which they truly integrate audio and text modalities in their internal representations. In this work, we evaluate four SLMs on the task of speech emotion recognition using a dataset of emotionally incongruent speech samples, a condition under which the semantic content of the spoken utterance conveys one emotion while speech expressiveness conveys another. Our results indicate that SLMs rely predominantly on textual semantics rather than speech emotion to perform the task, indicating that text-related representations largely dominate over acoustic representations. We release both the code and the Emotionally Incongruent Synthetic Speech dataset (EMIS) to the community.
#### SFMS-ALR: Script-First Multilingual Speech Synthesis with Adaptive Locale Resolution
 - **Authors:** Dharma Teja Donepudi
 - **Subjects:** Subjects:
Sound (cs.SD); Artificial Intelligence (cs.AI); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2510.25178

 - **Pdf link:** https://arxiv.org/pdf/2510.25178

 - **Abstract**
 Intra-sentence multilingual speech synthesis (code-switching TTS) remains a major challenge due to abrupt language shifts, varied scripts, and mismatched prosody between languages. Conventional TTS systems are typically monolingual and fail to produce natural, intelligible speech in mixed-language contexts. We introduce Script-First Multilingual Synthesis with Adaptive Locale Resolution (SFMS-ALR), an engine-agnostic framework for fluent, real-time code-switched speech generation. SFMS-ALR segments input text by Unicode script, applies adaptive language identification to determine each segment's language and locale, and normalizes prosody using sentiment-aware adjustments to preserve expressive continuity across languages. The algorithm generates a unified SSML representation with appropriate "lang" or "voice" spans and synthesizes the utterance in a single TTS request. Unlike end-to-end multilingual models, SFMS-ALR requires no retraining and integrates seamlessly with existing voices from Google, Apple, Amazon, and other providers. Comparative analysis with data-driven pipelines such as Unicom and Mask LID demonstrates SFMS-ALR's flexibility, interpretability, and immediate deployability. The framework establishes a modular baseline for high-quality, engine-independent multilingual TTS and outlines evaluation strategies for intelligibility, naturalness, and user preference.


by Zyzzyva0381 (Windy). 


2025-10-30
