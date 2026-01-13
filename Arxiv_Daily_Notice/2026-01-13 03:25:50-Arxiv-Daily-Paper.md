# Showing new listings for Monday, 12 January 2026
Auto update papers at about 2:30am UTC (10:30am Beijing time) every weekday.


阅读 `Usage.md`了解如何使用此repo实现个性化的Arxiv论文推送

See `Usage.md` for instructions on how to personalize the repo. 


Keyword list: ['text-to-speech', 'text to speech', 'tts', 'LLM-based', 'speech', 'voice']


Excluded: []


### Today: 4papers 
#### Discriminative-Generative Target Speaker Extraction with Decoder-Only Language Models
 - **Authors:** Bang Zeng, Beilong Tang, Wang Xiang, Ming Li
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2601.06006

 - **Pdf link:** https://arxiv.org/pdf/2601.06006

 - **Abstract**
 Target speaker extraction (TSE) aims to recover the speech signal of a desired speaker from a mixed audio recording, given a short enrollment utterance. Most existing TSE approaches are based on discriminative modeling paradigms. Although effective at suppressing interfering speakers, these methods often struggle to produce speech with high perceptual quality and naturalness. To address this limitation, we first propose LauraTSE, a generative TSE model built upon an auto-regressive decoder-only language model. However, purely generative approaches may suffer from hallucinations, content drift, and limited controllability, which may undermine their reliability in complex acoustic scenarios. To overcome these challenges, we further introduce a discriminative-generative TSE framework. In this framework, a discriminative front-end is employed to robustly extract the target speaker's speech, yielding stable and controllable intermediate representations. A generative back-end then operates in the neural audio codec representation space to reconstruct fine-grained speech details and enhance perceptual quality. This two-stage design effectively combines the robustness and controllability of discriminative models with the superior naturalness and quality enhancement capabilities of generative models. Moreover, we systematically investigate collaborative training strategies for the proposed framework, including freezing or fine-tuning the front-end, incorporating an auxiliary SI-SDR loss, and exploring both auto-regressive and non-auto-regressive inference mechanisms. Experimental results demonstrate that the proposed framework achieves a more favorable trade-off among speech quality, intelligibility, and speaker consistency.
#### CosyEdit: Unlocking End-to-End Speech Editing Capability from Zero-Shot Text-to-Speech Models
 - **Authors:** Junyang Chen, Yuhang Jia, Hui Wang, Jiaming Zhou, Yaxin Han, Mengying Feng, Yong Qin
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2601.05329

 - **Pdf link:** https://arxiv.org/pdf/2601.05329

 - **Abstract**
 Automatic speech editing aims to modify spoken content based on textual instructions, yet traditional cascade systems suffer from complex preprocessing pipelines and a reliance on explicit external temporal alignment. Addressing these limitations, we propose CosyEdit, an end-to-end speech editing model adapted from CosyVoice through task-specific fine-tuning and an optimized inference procedure, which internalizes speech-text alignment while ensuring high consistency between the speech before and after editing. By fine-tuning on only 250 hours of supervised data from our curated GigaEdit dataset, our 400M-parameter model achieves reliable speech editing performance. Experiments on the RealEdit benchmark indicate that CosyEdit not only outperforms several billion-parameter language model baselines but also matches the performance of state-of-the-art cascade approaches. These results demonstrate that, with task-specific fine-tuning and inference optimization, robust and efficient speech editing capabilities can be unlocked from a zero-shot TTS model, yielding a novel and cost-effective end-to-end solution for high-quality speech editing.
#### Closing the Modality Reasoning Gap for Speech Large Language Models
 - **Authors:** Chaoren Wang, Heng Lu, Xueyao Zhang, Shujie Liu, Yan Lu, Jinyu Li, Zhizheng Wu
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2601.05543

 - **Pdf link:** https://arxiv.org/pdf/2601.05543

 - **Abstract**
 Although speech large language models have achieved notable progress, a substantial modality reasoning gap remains: their reasoning performance on speech inputs is markedly weaker than on text. This gap could be associated with representational drift across Transformer layers and behavior deviations in long-chain reasoning. To address this issue, we introduce TARS, a reinforcement-learning framework that aligns text-conditioned and speech-conditioned trajectories through an asymmetric reward design. The framework employs two dense and complementary signals: representation alignment, which measures layer-wise hidden-state similarity between speech- and text-conditioned trajectories, and behavior alignment, which evaluates semantic consistency between generated outputs and reference text completions. Experiments on challenging reasoning benchmarks, including MMSU and OBQA, show that our approach significantly narrows the modality reasoning gap and achieves state-of-the-art performance among 7B-scale Speech LLMs.
#### SPAM: Style Prompt Adherence Metric for Prompt-based TTS
 - **Authors:** Chanhee Cho, Nayeon Kim, Bugeun Kim
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2601.05554

 - **Pdf link:** https://arxiv.org/pdf/2601.05554

 - **Abstract**
 Prompt-based text-to-speech (TTS) aims to generate speech that adheres to fine-grained style cues provided in a text prompt. However, most prior works depend on neither plausible nor faithful measures to evaluate prompt adherence. That is, they cannot ensure whether the evaluation is grounded on the prompt and is similar to a human. Thus, we present a new automatic metric, the Style Prompt Adherence Metric, which explicitly satisfies both plausibility and faithfulness. Inspired by the CLAP, our approach factorizes speech into acoustic attributes and aligns them with the style prompt. Also, we trained the scorer with a supervised contrastive loss, which could provide a clearer distinction between different semantics. We conducted two experiments on two perspectives. The plausibility experiment showed that SPAM achieved a strong correlation with the mean opinion score (MOS). Also, the faithfulness experiment demonstrated that SPAM is successfully grounded to the given style prompt, as it can discriminate different semantics of the prompt. We believe that SPAM can provide a viable automatic solution for evaluating style prompt adherence of synthesized speech.


by Zyzzyva0381 (Windy). 


2026-01-13
