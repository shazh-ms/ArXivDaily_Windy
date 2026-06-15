# Showing new listings for Monday, 15 June 2026
Auto update papers at about 2:30am UTC (10:30am Beijing time) every weekday.


阅读 `Usage.md`了解如何使用此repo实现个性化的Arxiv论文推送

See `Usage.md` for instructions on how to personalize the repo. 


Keyword list: ['text-to-speech', 'text to speech', 'tts', 'LLM-based', 'speech', 'voice']


Excluded: []


### Today: 4papers 
#### Unsupervised Approaches for Global Prosodic Embedding Extraction
 - **Authors:** Martin Meza, Luciana Ferrer, Pablo Riera
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2606.14004

 - **Pdf link:** https://arxiv.org/pdf/2606.14004

 - **Abstract**
 Prosody is central to oral communication, conveying information like the emotional state of the speaker and cues needed for meaning disambiguation. Many self-supervised models of speech produce embeddings that encode prosodic as well as linguistic, and speaker information. This entanglement of information is problematic in scenarios where prosody is the main distinguishing factor while other factors may vary between training and deployment; in such cases, a purely prosodic representation would be more robust. Such representation could also be used for analyzing the role of prosody in a given task or as input to speech synthesis systems. In this work, we propose a variety of approaches for producing global prosodic embeddings based on auto-encoder models of pitch and energy. We develop a benchmark for assessing the performance of these representations, showing that our embeddings provide competitive or superior performance under challenging conditions, compared to various alternatives.
#### Who Spoke When in Multi-Conversation: Target Speaker Tagging Task and Benchmark
 - **Authors:** Minjae Lee, Hee-Soo Heo, Youngki Kwon, Han-Gyu Kim, You Jin Kim, Bong-Jin Lee
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2606.14091

 - **Pdf link:** https://arxiv.org/pdf/2606.14091

 - **Abstract**
 We present target speaker tagging (TST), a task that integrates speaker diarization, verification, and identification into a unified workflow for multi-speaker conversations. Given long recordings and pre-enrolled speakers, TST detects and labels speech segments of known speakers while rejecting unknown ones. Despite its practical importance, research has been limited by the absence of suitable evaluation resources. To address this, we introduce TST-Bench, a large-scale synthetic benchmark with over 150 enrolled speakers, 300 sessions of 20-60 minutes, and reference annotations with global speaker labels. We define an evaluation protocol encompassing diarization and full-pipeline scenarios. Experiments on both real and synthetic data show that TST poses challenges not captured by conventional benchmarks, and that dedicated system design yields significant gains over naive integration of existing solutions. The benchmark dataset and evaluation protocols are publicly released.
#### HIDVAS: A Hearing Instrument Dataset in Various Acoustical Scenarios for Algorithm Evaluation and Training
 - **Authors:** Arnout Roebben, Giuliano Bernardi, Jan Wouters, Toon van Waterschoot, Marc Moonen
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2606.14175

 - **Pdf link:** https://arxiv.org/pdf/2606.14175

 - **Abstract**
 To evaluate the performance of audio signal processing algorithms and to train data-driven algorithms, e.g., as applied in hearing instruments, either simulated or recorded data can be used. While large batches of simulated data can be generated using mathematical models, recorded data provide a more adequate representation of real-life scenarios. Therefore, in this paper, the Hearing Instrument Dataset in Various Acoustical Scenarios (HIDVAS) is introduced. This dataset consists of both impulse responses and audio recordings using eight external loudspeakers, two external microphones, and a dummy head. On this dummy head behind-the-ear (BTE) hearing instrument shells with two microphones per shell are mounted, and in the dummy head's ears receiver-in-canal (RIC) hearing instrument loudspeakers are inserted. The dummy head also contains microphones located at its eardrum. The impulse responses have been computed from a swept-sine recording for each microphone-loudspeaker pair, and the audio recordings have been obtained by playing back audio (male and female speech, speech shaped noise, singing voice, stringed instrument, wind instrument, and percussion instrument) through each individual loudspeaker and recording simultaneously using all microphones. These recordings have been repeated for four hearing instrument domes (open, semi-open, closed, and no-RIC) in three reverberation conditions in one room (T30 = 0.09 s, T30 = 0.47 s, and T30 = 0.73 s), and in one reverberation condition in a different room (T30 = 1.48 s). The usage of the dataset as a `hearing instrument in a box' is exemplified with three example use cases.
#### BayLing-Duplex: Native Full-Duplex Speech Dialogue with a Single Autoregressive LLM
 - **Authors:** Qingkai Fang, Shoutao Guo, Yang Feng
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2606.14528

 - **Pdf link:** https://arxiv.org/pdf/2606.14528

 - **Abstract**
 Real-time, full-duplex speech interaction is a key feature of next-generation spoken chatbots, allowing the model to listen and speak at the same time and to handle natural phenomena such as overlap, hesitation, and barge-in. Existing speech language models (SpeechLMs) such as LLaMA-Omni and GLM-4-Voice are still turn-based and rely on an external Voice Activity Detection (VAD) module to mark the end of the user's turn, which fundamentally limits their interactive ability. In this paper, we introduce BayLing-Duplex, a native full-duplex SpeechLM where a single autoregressive LLM decides when to listen, when to speak, and when to stop, with no auxiliary turn-taking module. The design adds only a few special tokens to the standard vocabulary, so it transfers across LLMs and reuses existing training and serving stacks with no architectural adaptation. Starting from the public GLM-4-Voice checkpoint and using only 400K full-duplex samples for fine-tuning followed by a lightweight DPO stage, BayLing-Duplex reaches 92% turn-taking success and 100% interruption success on InstructS2S-Eval, while improving the speech-response score from 2.17 to 3.39 over Moshi. BayLing-Duplex also matches or surpasses its turn-based counterpart on Llama Questions, Web Questions, and Alpaca-Eval, showing that simultaneous listen-and-speak modeling does not sacrifice response quality.


by Zyzzyva0381 (Windy). 


2026-06-15
