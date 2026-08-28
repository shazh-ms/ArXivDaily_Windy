# Showing new listings for Friday, 28 August 2026
Auto update papers at about 2:30am UTC (10:30am Beijing time) every weekday.


阅读 `Usage.md`了解如何使用此repo实现个性化的Arxiv论文推送

See `Usage.md` for instructions on how to personalize the repo. 


Keyword list: ['text-to-speech', 'text to speech', 'tts', 'LLM-based', 'speech', 'voice']


Excluded: []


### Today: 6papers 
#### Towards Interpretable Depression Detection: Linking Acoustic Features to DSM-5 Indicators
 - **Authors:** Jonas Länzlinger, Katharina O.E. Müller, Burkhard Stiller, Bruno Rodrigues
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2608.26148

 - **Pdf link:** https://arxiv.org/pdf/2608.26148

 - **Abstract**
 Depression affects millions worldwide, yet diagnosis relies on subjective self-reports that may miss authentic behavior. This paper presents an approach linking speech acoustics to DSM-5 depressive-behavior indicators through a transparent Linkage Framework. Unlike black-box models, the framework explicitly maps acoustic features (pitch variability, pauses, speech tempo) to clinical indicators, enabling interpretable, indicator-level outputs. The system runs locally on commodity hardware (HW) to preserve privacy. Preliminary evaluation on DAIC-WOZ shows directionally consistent associations between acoustic features and DSM-5 indicators for psychomotor change and concentration difficulty, supporting the design rationale. Future work will validate on longitudinal datasets and extend multimodal integration while maintaining edge constraints.
#### Refusal Is Not Robustness: Auditing Confident Fabrication in Large Language Models on a Provably Uninformative Clinical Pain Speech Transcript
 - **Authors:** Sagnik De, Sreenija Pavuluri
 - **Subjects:** Subjects:
Artificial Intelligence (cs.AI); Emerging Technologies (cs.ET); Machine Learning (cs.LG); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2608.26167

 - **Pdf link:** https://arxiv.org/pdf/2608.26167

 - **Abstract**
 Hallucination and abstention benchmarks rarely establish that a model could not have known the correct answer, making it difficult to distinguish appropriate abstention from an unsupported prediction. Seven large language models were evaluated on the TAME Pain speech corpus. Participants read phonetically balanced Harvard Sentences while one hand was immersed in cold or warm water and reported pain only during periodic pain statements. This protocol generated 5,750 no signal Harvard Sentence utterances whose transcripts contained no lexical pain information and 1,294 signal pain statement utterances in which the pain rating was explicitly spoken. In the no signal arm, pain was recoverable from acoustic features (AUC 0.622, 95% CI 0.553 to 0.662), whereas transcript based prediction was near chance (AUC 0.489, 95% CI 0.418 to 0.504). Because automatic speech recognition removes the acoustic pain cues, any pain score inferred solely from the transcript is unsupported by the available evidence. Under cooperative prompting, six models abstained on nearly all no signal transcripts, correctly extracted spoken pain ratings in the positive control task with accuracies ranging from 0.939 to 1.00, and maintained an expected calibration error of at most 0.100. Under authority framed prompts, abstention became prompt dependent, with the same model ranging from 0.18 to 1.00 across equivalent prompt phrasings. Most models produced low confidence estimates when forced to answer, whereas Gemini 2.5 Flash and Llama 3.1 8B consistently generated confident pain scores with confident fabrication rates of 0.53 and 0.76, compared with at most 0.15 for all other models. No significant demographic effects were observed in forced responses, with all $p$ values greater than or equal to 0.20.
#### Attention-Guided Reliability Scaling for Contrastive Decoding in Robust Audio-Visual Speech Recognition
 - **Authors:** YoungChae Kim, Da-Hee Yang, Joon-Hyuk Chang
 - **Subjects:** Subjects:
Sound (cs.SD); Computer Vision and Pattern Recognition (cs.CV); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2608.26213

 - **Pdf link:** https://arxiv.org/pdf/2608.26213

 - **Abstract**
 Large language model (LLM)-based audio-visual speech recognition (AVSR) systems are robust under noise. Contrastive decoding (CD), originally introduced to stabilize LLM generation by contrasting a weaker model against a stronger one at inference time, adjusts predictions without additional training. In this work, we apply CD to AVSR by contrasting audio-only conditioning with full audio-visual conditioning within the same underlying model. However, using a fixed contrastive strength introduces a trade-off across noise levels: stronger intervention helps under severe noise but may over-correct reliable predictions in clean conditions. We propose reliability-aware scaling of CD for AVSR. Instead of using a fixed strength, we adaptively modulate the contrastive influence at each token based on reliability signals derived from attention dynamics and inter-model predictive divergence. Experiments on LRS3 show consistent improvements across clean and low-SNR conditions.
#### AudioSpan: Spanning the Duration and Depth of Audio Comprehension
 - **Authors:** Wen Huang, Yunfei Chu, Meng Gao, Haolin He, Jin Xu
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2608.26431

 - **Pdf link:** https://arxiv.org/pdf/2608.26431

 - **Abstract**
 General audio comprehension now covers speech, sound, and music over durations from seconds to hours, driven by large audio-language models (LALMs) that are increasingly omni-modal. Yet the benchmarks that test them still rely on clips of seconds, where scores saturate and models converge; recent long-form efforts extend duration but evaluate long audio much as short clips are. We introduce AudioSpan, a benchmark that spans both duration and depth: it pairs audio from 10 minutes to over 2 hours with 3,240 questions across three cognitive levels, namely perception, understanding, and reasoning. Two paths supply the questions, differing in how question content is sourced and how ground truth is obtained. Native QA extracts questions from the audio's content, posing each as a multiple-choice item and an open-ended one graded by detailed rubrics. Anchor QA instead injects ground truth, planting acoustic anchors into the audio and building a perception-to-reasoning chain scored only to the first error. A fully automated pipeline constructs every item through structured captioning, QA generation, and adversarial critic feedback. Evaluating 12 LALMs on AudioSpan, we find the hard part comes before reasoning: distilling a few relevant facts from a long, redundant signal. This difficulty grows with audio length and falls hardest on perception, especially temporal grounding. AudioSpan is available at this https URL.
#### Mapping Written Words to Spoken Words in a Different Language Using Only Visual Grounding
 - **Authors:** Gabriel Pirlogeanu, Dan Oneata, Horia Cucu, Herman Kamper
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2608.26925

 - **Pdf link:** https://arxiv.org/pdf/2608.26925

 - **Abstract**
 In many low-resource settings, even just eliciting speech for data collection is difficult. One promising approach has been to ask speakers to describe images. But how do we build models from such visually grounded speech data? Given a dataset of images with Hindi spoken captions, we consider how we can map a written English keyword to spoken realisations of that word in Hindi. Previous work trained end-to-end multimodal neural models. Instead, we explore a simpler alignment-based approach built on self-supervised speech representations. Written English tags are automatically obtained from images using off-the-shelf image captioning systems. Hindi utterances associated with the same keyword are then aligned (using self-supervised features), and alignment evidence is aggregated to identify recurring speech segments corresponding to the target word. Experiments evaluating keyword spotting and localization show that our alignment-based approach outperforms a previous attention-based neural model. We also show the benefit of incorporating negative examples during alignment. Our work demonstrates that cross-lingual word-to-speech mappings can be learned directly from visual grounding without transcriptions or explicit model training.
#### When Text Misleads: Inconsistent-Aware Reasoning for Audio-Grounded Dialogue
 - **Authors:** Yen-Ju Lu, Yuzhe Wang, Yaohan Guan, Xiluo He, Jiarui Hai, Mingrui Liang, Kaavya Chaparala, Thomas Thebaud, Laureano Moro-Velazquez, Najim Dehak, Jesus Villalba
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Artificial Intelligence (cs.AI); Machine Learning (cs.LG); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2608.27176

 - **Pdf link:** https://arxiv.org/pdf/2608.27176

 - **Abstract**
 Understanding spoken dialogue requires joint reasoning over lexical content and paralinguistic acoustic signals such as emotion and conversational intent. However, existing evaluations often allow shortcuts based on transcripts or single-modality solutions, obscuring whether models genuinely ground predictions in speech. We formalize this failure mode as cross-modal disagreement, where transcripts suggest plausible but incorrect surface interpretations while acoustic cues such as prosody or speaking style support different answers. We develop a scalable framework that identifies text-biased surface interpretations and converts disagreement regions into conflict QA examples. We also include consistent cases where transcript-based and speech-grounded interpretations agree, enabling evaluation beyond adversarial audio dependence. This results in ContraTalk, a controlled benchmark containing 501 questions across five discourse dimensions: interaction behavior, emotion state, dialogue act, social stance, and conversational intent. We further develop an agentic-style reasoning framework that converts speech into an Audio Twin, a text-readable representation of localized acoustic cues that exposes acoustic evidence to the reasoning model. Experiments show that strong text-only LLMs exceed 90% accuracy in consistent cases but drop to 33-48% in conflict cases. Direct AudioLLMs provide only partial grounding, still selecting the transcript-biased trap in roughly 30-40% of conflict cases. Our Audio Twin framework improves conflict-case accuracy while reducing trap selection, but its consistent-case behavior remains backbone-dependent. These results identify transcript-based shortcuts as an important failure mode in spoken dialogue understanding and show that explicit acoustic evidence aggregation provides a more controllable interface for diagnosing and improving speech-grounded reasoning.


by Zyzzyva0381 (Windy). 


2026-08-28
