# Showing new listings for Thursday, 25 June 2026
Auto update papers at about 2:30am UTC (10:30am Beijing time) every weekday.


阅读 `Usage.md`了解如何使用此repo实现个性化的Arxiv论文推送

See `Usage.md` for instructions on how to personalize the repo. 


Keyword list: ['text-to-speech', 'text to speech', 'tts', 'LLM-based', 'speech', 'voice']


Excluded: []


### Today: 9papers 
#### End-to-End Voice Intent Recognition for Spontaneous Human-Drone Interaction with Naive Users
 - **Authors:** Allan Henry (GIPSA-COPERNIC, GETALP, LPNC), Solange Rossato (GETALP), Christian Graff (LPNC), Sylvain Huet (GIPSA-COPERNIC), Jose-Ernesto Gomez-Balderas (GIPSA-COPERNIC)
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Artificial Intelligence (cs.AI); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2606.24910

 - **Pdf link:** https://arxiv.org/pdf/2606.24910

 - **Abstract**
 Voice control offers an intuitive alternative to manual drone piloting, yet most existing systems rely on rigid command vocabularies that fail to handle the spontaneous, disfluent speech of naive users. This paper addresses this gap by proposing an End-to-End Spoken Language Understanding architecture for real-time human-drone interaction in French. Our model combines a frozen Self-Supervised Learning acoustic encoder with a lightweight LSTM-based classification head, augmented by a cross-modal knowledge distillation objective that aligns acoustic representations with semantic embeddings from a text teacher, without requiring transcription at inference time. We evaluate our approach on VoiceStick, a novel French corpus of spontaneous speech collected during real teleoperation sessions with 29 nonexpert dyads. On simple voice commands, our best configuration achieves 93% accuracy at 7 ms inference latency, outperforming cascade baselines (79%, 202 ms) with a 29x speedup. On the full spontaneous speech test set, our architecture reaches 82% accuracy, with crossmodal distillation consistently improving robustness across all configurations. These results demonstrate that End-to-End architectures are not only feasible but preferable for spontaneous voice-guided UAV teleoperation, combining semantic robustness, low latency, and calibrated confidence.
#### Phoneme-Level Mispronunciation Screening in Polish-Speaking Children with an Explainable Assistant
 - **Authors:** Milosz Dudek, Daria Hemmerling, Kamil Kwarciak, Maciej Stroinski, Maria Pensko, Mateusz Kowalewski, Leonid Pavlovskyi, Sebastian Jurczak, Anna-Mariia Vitkovska, Zuzanna Miodonska, Natalia Mocko, Michal Krecichwost
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Artificial Intelligence (cs.AI); Multiagent Systems (cs.MA)
 - **Arxiv link:** https://arxiv.org/abs/2606.25181

 - **Pdf link:** https://arxiv.org/pdf/2606.25181

 - **Abstract**
 Early identification of speech sound errors in children is often limited by access to specialists, motivating lightweight screening tools that can operate outside the clinic. We present a screening pipeline for Polish-speaking children focused on sibilant substitutions, coupling a wav2vec2-based CTC token recognizer with alignment-based error typing and a template-grounded caregiver assistant for screening, not diagnosis. On a held-out test set of 10 unseen children comprising 559 utterances, the recognizer achieves 88.7 percent exact sequence match. As a conservative screening proxy, we flag a mismatch when the system emits substitution-evidence bracketed tokens at the target segment, yielding 72.9 percent precision, 61.4 percent recall, F1 = 0.67, and a 2.7 percent false-alarm rate on target-correct items. We describe the assistant's safety boundaries and outline a clinician-in-the-loop validation plan for future deployment.
#### CrossAccent-TTS: Cross-Lingual Accent-Intensity Controllable Text-to-Speech via Disentangled Speaker and Accent Representations
 - **Authors:** Ram Annamdevula, Ankit Tatawat, Ashishkumar P. Gudmalwar, Nirmesh J. Shah, Pankaj Wasnik
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Artificial Intelligence (cs.AI); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2606.25403

 - **Pdf link:** https://arxiv.org/pdf/2606.25403

 - **Abstract**
 Accent conversion and controllability remain fundamental challenges in cross-lingual text-to-speech (TTS), particularly for low-resource and phonetically diverse Indic languages. While recent large language model (LLM)-based TTS systems exhibit strong cross-lingual generalization, they provide limited explicit control over accent characteristics and intensity. In this paper, we propose CrossAccentTTS, a framework that enables both accent control and conversion while preserving speaker identity. Specifically, we introduce an Accent Intensity Controller (AIC) that injects weighted language embeddings into the accent subspace, allowing smooth interpolation between accents and fine-grained modulation of accent strength at inference time. Experiments on the Indic Multilingual and L2-arctic datasets shows that CrossAccent-TTS achieves precise control of accent intensity, outperforming strong baselines in accent similarity and controllability by maintaining speaker similarity and naturalness.
#### Adaptive Oscillatory Inductive Bias for Modeling Sharp Prosodic Dynamics in Diffusion-Based TTS
 - **Authors:** Sandipan Dhar, Nirmesh J. Shah, Ashishkumar P. Gudmalwar, Pankaj Wasnik
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Artificial Intelligence (cs.AI); Computation and Language (cs.CL); Sound (cs.SD); Signal Processing (eess.SP)
 - **Arxiv link:** https://arxiv.org/abs/2606.25424

 - **Pdf link:** https://arxiv.org/pdf/2606.25424

 - **Abstract**
 Diffusion-based text-to-speech (TTS) models have achieved significant improvements in speech quality. However, modeling sharp prosodic transitions and rapid pitch variations in expressive speech remains challenging. Existing diffusion-based TTS decoders commonly utilize periodic nonlinearities such as Snake activation function to capture harmonic structures, but this activation funcation provides limited adaptability when modeling abrupt amplitude and frequency variations. In this paper, we investigate the role of oscillatory inductive bias in diffusion-based TTS decoders and introduce an adaptive oscillatory nonlinearity that enables controllable periodic modulation while maintaining signal stability through a linear bypass component. We refer the resulting TTS system as OscillaTTS. Experiments on the LJSpeech and Emotional Speech Dataset show consistent improvements across objective and subjective evaluations, indicating improved modeling of expressive prosodic dynamics.
#### Evaluating Japanese Dialect Robustness Across Speech and Text-based Large Language Models
 - **Authors:** Tomoya Mizumoto, Yusuke Fujita, Hao Shi, Lianbo Liu, Atsushi Kojima, Yui Sudo
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Computation and Language (cs.CL); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2606.25436

 - **Pdf link:** https://arxiv.org/pdf/2606.25436

 - **Abstract**
 Dialogue systems based on large language models (LLMs) have advanced significantly in recent years. However, dialectal variation remains a major challenge, particularly for systems that process spoken input. LLM-based speech language models (SLMs), which integrate LLMs with speech processing components, show promise for spoken language tasks, yet their ability to comprehend dialects has not been sufficiently studied. Moreover, it remains unclear how the dialectal understanding of the base LLM affects SLM performance. This study investigates the dialectal robustness of both LLMs and SLMs using Japanese dialects as a test case. We define robustness as the ratio of performance on dialectal versus standard inputs, enabling fair comparisons. Our experiments show that SLM robustness correlates with that of their text-based counterparts. Furthermore, training with dialectal data and fine-tuning the speech encoder each improves robustness in SLMs.
#### Does Translation-Enhanced Speech Encoder Pre-training Affect Speech LLMs?
 - **Authors:** Tomoya Mizumoto, Yusuke Fujita
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Computation and Language (cs.CL); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2606.25444

 - **Pdf link:** https://arxiv.org/pdf/2606.25444

 - **Abstract**
 Connecting a pre-trained speech encoder to a Large Language Model (LLM) is the standard architecture for building Speech LLMs. However, a structural misalignment exists between the encoder and the LLM. Unlike encoders based on automatic speech recognition, which often produce representations in separate language-specific spaces, LLMs operate within a unified language-agnostic space. A mechanism is required to align the encoder's language-specific representations with the LLM's shared space. We argue that speech translation provides a principled way to achieve this. Unlike monolingual transcription, translation requires the model to bridge different languages and learn language-agnostic representations. We experimentally evaluate the impact of incorporating translation objectives into speech encoder pre-training. Our results demonstrate that translation-enhanced pre-training improves cross-modal integration and leads to superior performance across downstream Speech LLM tasks.
#### Joint Residual Reweighting for Classifier Free Guidance in Flow-Matching Zero-Shot TTS
 - **Authors:** Runwu Shi, Yujin Wang, Hongjin Song, Chunxiang Jin
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2606.25672

 - **Pdf link:** https://arxiv.org/pdf/2606.25672

 - **Abstract**
 Classifier-free guidance (CFG) is widely used in flow-matching-based zero-shot text-to-speech (TTS), where generation is typically controlled by two conditions: the target text and a prompt speech signal. Standard CFG strengthens these conditions jointly, while recent branch-selective guidance methods attempt to enhance text or speaker conditioning separately, often leading to a trade-off between text correctness and speaker similarity. In this paper, we revisit the CFG under independently masked text and speech-prompt conditions, and decompose the guidance field into text, speaker, and joint residuals. We show that conventional speaker-selective guidance entangles the speaker residual with the joint residual, which may disturb text-related generation. Based on this observation, we propose joint residual reweighting, which independently controls the speaker and joint residuals within the standard CFG framework. Experiments on F5-TTS and CosyVoice2 show that the proposed method improves speaker similarity while maintaining competitive text correctness, demonstrating the usefulness of the joint residual for balancing speaker fidelity and text accuracy in zero-shot TTS.
#### SE-AGCNet: An End-to-End Framework for Joint Speech Enhancement and Loudness Control in Meeting Scenarios
 - **Authors:** Jinming Zhang, Wei Rao, Xionghu Zhong, Eng Siong Chng
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Artificial Intelligence (cs.AI)
 - **Arxiv link:** https://arxiv.org/abs/2606.25959

 - **Pdf link:** https://arxiv.org/pdf/2606.25959

 - **Abstract**
 Conventional audio pipelines typically treat speech enhancement (SE) and automatic gain control (AGC) as discrete modules, which often limits overall performance. For instance, applying AGC before SE may inadvertently amplify background noise, while prioritizing SE tends to over-suppress low-volume speech. To address these limitations, we propose SE-AGCNet, an end-to-end framework that jointly optimizes SE and AGC. Tailored for meeting scenarios with significant volume variations, SE-AGCNet leverages the synergy between the two tasks: SE preserves quiet speech, thereby facilitating effective volume adjustment by the AGC component. Furthermore, we propose a specialized data simulation pipeline, SE-AGC-DataGen, and incorporate standardized loudness evaluation metrics: integrated loudness (LUFS), short-term loudness (St LUFS), and LRA. Experiments show that SE-AGCNet consistently achieves target loudness while improving speech quality and ASR accuracy over competitive baselines.
#### Real-Time Voice AI Hears but Does Not Listen
 - **Authors:** Martijn Bartelds, Federico Bianchi, James Zou
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2606.26083

 - **Pdf link:** https://arxiv.org/pdf/2606.26083

 - **Abstract**
 Speech conveys information through both words and vocal delivery. We evaluate four leading production realtime voice systems-OpenAI's GPT Realtime 2, Google's Gemini 3.1 Flash Live, and Alibaba's Qwen3.5 Omni Plus and Omni Flash-on tasks where the words and the delivery patterns both convey meaningful information. Across three consequential scenarios, all four systems act on the words rather than the voice. They end calls with crying callers who insist nothing is wrong, approve wire transfers authorized in frightened voices, and enroll callers whose agreement is clearly sarcastic. Surprisingly, this is often not a failure of perception. When asked directly, three of the four systems reliably identify the distress, fear, or sarcasm they later ignore when making decisions. We observe a similar pattern when these realtime voice systems estimate accent and age, as their responses frequently follow the biases of the words rather than the acoustic properties of the speaker. We term this disconnect between perception and action the emotional intelligence gap of voice AI. Prompting systems to explicitly attend to vocal delivery improves performance only partially and inconsistently. Our findings show that current realtime voice AI systems often behave as if speech had been reduced to a transcript, suggesting that they should be used with caution in settings where the tone and emotion of delivery convey important information.


by Zyzzyva0381 (Windy). 


2026-06-25
