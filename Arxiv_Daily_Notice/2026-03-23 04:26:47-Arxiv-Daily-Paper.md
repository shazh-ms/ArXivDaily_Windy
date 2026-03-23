# Showing new listings for Monday, 23 March 2026
Auto update papers at about 2:30am UTC (10:30am Beijing time) every weekday.


阅读 `Usage.md`了解如何使用此repo实现个性化的Arxiv论文推送

See `Usage.md` for instructions on how to personalize the repo. 


Keyword list: ['text-to-speech', 'text to speech', 'tts', 'LLM-based', 'speech', 'voice']


Excluded: []


### Today: 4papers 
#### Gesture2Speech: How Far Can Hand Movements Shape Expressive Speech?
 - **Authors:** Lokesh Kumar, Nirmesh Shah, Ashishkumar P. Gudmalwar, Pankaj Wasnik
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Artificial Intelligence (cs.AI); Multimedia (cs.MM)
 - **Arxiv link:** https://arxiv.org/abs/2603.19831

 - **Pdf link:** https://arxiv.org/pdf/2603.19831

 - **Abstract**
 Human communication seamlessly integrates speech and bodily motion, where hand gestures naturally complement vocal prosody to express intent, emotion, and emphasis. While recent text-to-speech (TTS) systems have begun incorporating multimodal cues such as facial expressions or lip movements, the role of hand gestures in shaping prosody remains largely underexplored. We propose a novel multimodal TTS framework, Gesture2Speech, that leverages visual gesture cues to modulate prosody in synthesized speech. Motivated by the observation that confident and expressive speakers coordinate gestures with vocal prosody, we introduce a multimodal Mixture-of-Experts (MoE) architecture that dynamically fuses linguistic content and gesture features within a dedicated style extraction module. The fused representation conditions an LLM-based speech decoder, enabling prosodic modulation that is temporally aligned with hand movements. We further design a gesture-speech alignment loss that explicitly models their temporal correspondence to ensure fine-grained synchrony between gestures and prosodic contours. Evaluations on the PATS dataset show that Gesture2Speech outperforms state-of-the-art baselines in both speech naturalness and gesture-speech synchrony. To the best of our knowledge, this is the first work to utilize hand gesture cues for prosody control in neural speech synthesis. Demo samples are available at this https URL
#### Listen First, Then Answer: Timestamp-Grounded Speech Reasoning
 - **Authors:** Jihoon Jeong, Pooneh Mousavi, Mirco Ravanelli, Cem Subakan
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2603.19468

 - **Pdf link:** https://arxiv.org/pdf/2603.19468

 - **Abstract**
 Large audio-language models (LALMs) can generate reasoning chains for their predictions, but it remains unclear whether these reasoning chains remain grounded in the input audio. In this paper, we propose an RL-based strategy that grounds the reasoning outputs of LALMs with explicit timestamp annotations referring to relevant segments of the audio signal. Our analysis shows that timestamp grounding leads the model to attend more strongly to audio tokens during reasoning generation. Experiments on four speech-based benchmark datasets demonstrate that our approach improves performance compared to both zero-shot reasoning and fine-tuning without timestamp grounding. Additionally, grounding amplifies desirable reasoning behaviors, such as region exploration, audiology verification, and consistency, underscoring the importance of grounding mechanisms for faithful multimodal reasoning.
#### Borderless Long Speech Synthesis
 - **Authors:** Xingchen Song, Di Wu, Dinghao Zhou, Pengyu Cheng, Hongwu Ding, Yunchao He, Jie Wang, Shengfan Shen, Sixiang Lv, Lichun Fan, Hang Su, Yifeng Wang, Shuai Wang, Meng Meng, Jian Luan
 - **Subjects:** Subjects:
Sound (cs.SD); Computation and Language (cs.CL); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2603.19798

 - **Pdf link:** https://arxiv.org/pdf/2603.19798

 - **Abstract**
 Most existing text-to-speech (TTS) systems either synthesize speech sentence by sentence and stitch the results together, or drive synthesis from plain-text dialogues alone. Both approaches leave models with little understanding of global context or paralinguistic cues, making it hard to capture real-world phenomena such as multi-speaker interactions (interruptions, overlapping speech), evolving emotional arcs, and varied acoustic environments. We introduce the Borderless Long Speech Synthesis framework for agent-centric, borderless long audio synthesis. Rather than targeting a single narrow task, the system is designed as a unified capability set spanning VoiceDesigner, multi-speaker synthesis, Instruct TTS, and long-form text synthesis. On the data side, we propose a "Labeling over filtering/cleaning" strategy and design a top-down, multi-level annotation schema we call Global-Sentence-Token. On the model side, we adopt a backbone with a continuous tokenizer and add Chain-of-Thought (CoT) reasoning together with Dimension Dropout, both of which markedly improve instruction following under complex conditions. We further show that the system is Native Agentic by design: the hierarchical annotation doubles as a Structured Semantic Interface between the LLM Agent and the synthesis engine, creating a layered control protocol stack that spans from scene semantics down to phonetic detail. Text thereby becomes an information-complete, wide-band control channel, enabling a front-end LLM to convert inputs of any modality into structured generation commands, extending the paradigm from Text2Speech to borderless long speech synthesis.
#### Audio Avatar Fingerprinting: An Approach for Authorized Use of Voice Cloning in the Era of Synthetic Audio
 - **Authors:** Candice R. Gerstner
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2603.20165

 - **Pdf link:** https://arxiv.org/pdf/2603.20165

 - **Abstract**
 With the advancements in AI speech synthesis, it is easier than ever before to generate realistic audio in a target voice. One only needs a few seconds of reference audio from the target, quite literally putting words in the target person's mouth. This imposes a new set of forensics-related challenges on speech-based authentication systems, videoconferencing, and audio-visual broadcasting platforms, where we want to detect synthetic speech. At the same time, leveraging AI speech synthesis can enhance the different modes of communication through features such as low-bandwidth communication and audio enhancements - leading to ever-increasing legitimate use-cases of synthetic audio. In this case, we want to verify if the synthesized voice is actually spoken by the user. This will require a mechanism to verify whether a given synthetic audio is driven by an authorized identity, or not. We term this task audio avatar fingerprinting. As a step towards audio forensics in these new and emerging situations, we analyze and extend an off-the-shelf speaker verification model developed outside of forensics context for the task of fake speech detection and audio avatar fingerprinting, the first experimentation of its kind. Furthermore, we observe that no existing dataset allows for the novel task of verifying the authorized use of synthetic audio - a limitation which we address by introducing a new speech forensics dataset for this novel task.


by Zyzzyva0381 (Windy). 


2026-03-23
