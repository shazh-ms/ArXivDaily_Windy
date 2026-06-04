# Showing new listings for Thursday, 4 June 2026
Auto update papers at about 2:30am UTC (10:30am Beijing time) every weekday.


阅读 `Usage.md`了解如何使用此repo实现个性化的Arxiv论文推送

See `Usage.md` for instructions on how to personalize the repo. 


Keyword list: ['text-to-speech', 'text to speech', 'tts', 'LLM-based', 'speech', 'voice']


Excluded: []


### Today: 6papers 
#### Read What You Hear: Reference-Free Hypotheses Evaluation with Acoustic Discrepancy
 - **Authors:** Zhihan Li, Hankun Wang, Yiwei Guo, Bohan Li, Xie Chen, Kai Yu
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Computation and Language (cs.CL); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2606.04680

 - **Pdf link:** https://arxiv.org/pdf/2606.04680

 - **Abstract**
 Automatic speech recognition systems commonly rely on reference transcriptions for evaluation, while reference-free approaches often depend on internal confidence estimation or auxiliary language models. We propose READ (Reference-free Hypothesis Evaluation with Acoustic Discrepancy), a novel metric that evaluates ASR hypotheses directly from the speech signal. READ emphasizes the acoustic grounding of hypotheses. It uses a pretrained auto-regressive TTS model to compute the conditional likelihood of speech tokens given a text hypothesis, to measure fine-grained acoustic discrepancy between speech and text. Without additional training, READ can be applied for hypothesis refinement. Experiments show that READ correlates with specific recognition errors and improves ASR outputs, achieving up to 20\% relative error rate reduction, with particularly strong gains under noisy conditions.
#### Feasibility of Time-Domain DNN-Based Speech Enhancement on Embedded FPGA for Hearing Aid
 - **Authors:** Feyisayo Olalere, Umut Altin, Kiki van der Heijden, Marcel van Gerven
 - **Subjects:** Subjects:
Sound (cs.SD); Hardware Architecture (cs.AR); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2606.04221

 - **Pdf link:** https://arxiv.org/pdf/2606.04221

 - **Abstract**
 Hearing aids impose strict latency and power constraints that current DNN-based speech enhancement systems struggle to meet on embedded hardware. We characterize this gap by deploying both speech separation and denoising using the lightweight SuDoRM-RF++ architecture on the AMD-Xilinx Kria KV260, evaluated at FP32 and 16-bit fixed-point precision for each task. Across these configurations, first-sample latency tracks with on-chip parameter caching rather than arithmetic throughput, identifying data movement as the primary bottleneck. Precision reduction halves the model memory footprint without compromising objective speech quality. The fixed-point denoising accelerator achieves a first-sample latency of 9.7~ms, meeting the 10~ms clinical threshold, while speech separation reaches 16.0~ms. These measurements establish concrete resource requirements for embedded DNN-based speech enhancement and quantify the remaining gap to hearing aid deployment.
#### CleanCodec: Efficient and Robust Speech Tokenization via Perceptually Guided Encoding
 - **Authors:** Eugene Kwek, Feng Liu, Rui Zhang, Wenpeng Yin
 - **Subjects:** Subjects:
Sound (cs.SD); Computation and Language (cs.CL); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2606.04418

 - **Pdf link:** https://arxiv.org/pdf/2606.04418

 - **Abstract**
 Neural audio codecs are a key component of speech processing pipelines, compressing audio into discrete tokens for downstream modeling. However, existing codecs struggle to balance reconstruction quality with token efficiency, often encoding perceptually irrelevant information such as background noise and recording artifacts at the expense of linguistically and acoustically meaningful content. We reframe audio tokenization as a selective information bottleneck problem and propose CleanCodec, a denoising audio codec which learns to encode only perceptually important features and discard imperceptible information. At just 12.5 tokens per second, CleanCodec achieves state-of-the-art tokenization efficiency, substantially outperforming existing codecs in speaker similarity and speech intelligibility. Evaluations on downstream text-to-speech and voice conversion tasks further demonstrate improved performance and up to 17x faster inference, highlighting significant efficiency gains.
#### Entity Binding Failures in Speech LLM Reasoning: Diagnosis and Chain-of-Thought Intervention
 - **Authors:** Ming-Hao Hsu, Xiaohai Tian, Jun Zhang, Zhizheng Wu
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2606.04474

 - **Pdf link:** https://arxiv.org/pdf/2606.04474

 - **Abstract**
 Speech Large Language Models (SLLMs) underperform their text counterparts on complex reasoning. We reveal that this modality gap is not a uniform cognitive deficit. Evaluating three diverse SLLMs, we show speech-to-text (S2T) matches or exceeds text-to-text (T2T) on spatial, syntactic, and factual tasks. However, on logical tasks requiring entity tracking, S2T accuracy collapses to chance. We diagnose this localized degradation as an entity binding failure: continuous speech features cause models to lose precise entity-property associations during implicit reasoning. To resolve this, we propose Entity-Aware Chain-of-Thought (EA-CoT), forcing SLLMs to explicitly enumerate entities and bind them to claims before reasoning. Strikingly, EA-CoT bridges the gap, even when spoken names are misrecognized, yielding up to a 24.4% absolute accuracy improvement. Ablations confirm these gains stem entirely from explicit semantic binding, reframing the gap as a resolvable bottleneck.
#### Multilingual Long-Form Speech Instruction Following: KIT's Submission to IWSLT 2026
 - **Authors:** Enes Yavuz Ugan, Maike Züfle, Yuka Ko, Supriti Sinhamahapatra, Fabian Retkowski, Seymanur Akti, Jan Niehues, Alexander Waibel
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2606.04730

 - **Pdf link:** https://arxiv.org/pdf/2606.04730

 - **Abstract**
 With the advent of Large Language Models, single-task and token-based multi-task models have evolved into instruction-based systems that infer task and target language implicitly from natural language prompts. This trend is reflected in IWSLT's Instruction Following Track, which this year introduced new tasks including an unknown surprise task, posing a genuine challenge against overfitting to known tasks. We present KIT's submission to the Long and Short Instruction Following tracks in the unconstrained setting. Our approach combines a general data augmentation pipeline that converts short-form corpora into long-form training data through segment concatenation, LLM-based label generation, and cross-lingual translation, yielding over 1M instances across six tasks and four languages. We further show that likelihood-based re-ranking, while highly effective for ASR, systematically degrades semantic tasks by spuriously selecting candidates generated from segmented audio processing rather than holistic long-form inference, a failure mode resolved by combining likelihood with Minimum Bayes Risk decoding.
#### Audio Interaction Model
 - **Authors:** Zhifei Xie, Zihang Liu, Ze An, Xiaobin Hu, Yue Liao, Ziyang Ma, Dongchao Yang, Mingbao Lin, Deheng Ye, Shuicheng Yan, Chunyan Miao
 - **Subjects:** Subjects:
Sound (cs.SD); Artificial Intelligence (cs.AI); Computation and Language (cs.CL); Multimedia (cs.MM); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2606.05121

 - **Pdf link:** https://arxiv.org/pdf/2606.05121

 - **Abstract**
 Audio is an inherently interactive modality, yet today's Large Audio Language Models (LALMs) are offline, and streaming audio models each handle only a single task such as streaming ASR or voice chatting. It is time to unify them into one online LALM: a model that, through an always-on perceive-decide-respond loop, listens to sound, environment, and instructions in real time and reacts on the fly. We formalize this regime as the Audio Interaction Model, and realize it with Audio-Interaction, a unified streaming model that retains offline task execution while adding online general audio instruction following, from dialogue to full voice chatting, deciding when to respond from the semantics of the stream. To enable this, we propose SoundFlow, a framework that instantiates the perceive-decide-respond loop end to end, from data to training to deployment, through streaming-native data construction, comprehension-aware training, and asynchronous low-latency inference for stable real-time interaction. We further construct StreamAudio-2M, a 2.6M-item streaming corpus spanning 7 fundamental abilities and 28 sub-tasks, and Proactive-Sound-Bench for evaluating proactive audio intervention. Across 8 benchmarks, Audio-Interaction preserves competitive performance on mainstream audio tasks while unlocking capabilities inaccessible to offline LALMs, including real-time ASR, streaming audio instruction following, and proactive help.


by Zyzzyva0381 (Windy). 


2026-06-04
