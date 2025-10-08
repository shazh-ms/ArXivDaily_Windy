# Showing new listings for Wednesday, 8 October 2025
Auto update papers at about 2:30am UTC (10:30am Beijing time) every weekday.


阅读 `Usage.md`了解如何使用此repo实现个性化的Arxiv论文推送

See `Usage.md` for instructions on how to personalize the repo. 


Keyword list: ['text-to-speech', 'text to speech', 'tts', 'LLM-based', 'speech', 'voice']


Excluded: []


### Today: 9papers 
#### WaveSP-Net: Learnable Wavelet-Domain Sparse Prompt Tuning for Speech Deepfake Detection
 - **Authors:** Xi Xuan, Xuechen Liu, Wenxin Zhang, Yi-Cheng Lin, Xiaojian Lin, Tomi Kinnunen
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Computation and Language (cs.CL); Signal Processing (eess.SP)
 - **Arxiv link:** https://arxiv.org/abs/2510.05305

 - **Pdf link:** https://arxiv.org/pdf/2510.05305

 - **Abstract**
 Modern front-end design for speech deepfake detection relies on full fine-tuning of large pre-trained models like XLSR. However, this approach is not parameter-efficient and may lead to suboptimal generalization to realistic, in-the-wild data types. To address these limitations, we introduce a new family of parameter-efficient front-ends that fuse prompt-tuning with classical signal processing transforms. These include FourierPT-XLSR, which uses the Fourier Transform, and two variants based on the Wavelet Transform: WSPT-XLSR and Partial-WSPT-XLSR. We further propose WaveSP-Net, a novel architecture combining a Partial-WSPT-XLSR front-end and a bidirectional Mamba-based back-end. This design injects multi-resolution features into the prompt embeddings, which enhances the localization of subtle synthetic artifacts without altering the frozen XLSR parameters. Experimental results demonstrate that WaveSP-Net outperforms several state-of-the-art models on two new and challenging benchmarks, Deepfake-Eval-2024 and SpoofCeleb, with low trainable parameters and notable performance gains. The code and models are available at this https URL.
#### Teaching Machines to Speak Using Articulatory Control
 - **Authors:** Akshay Anand, Chenxu Guo, Cheol Jun Cho, Jiachen Lian, Gopala Anumanchipalli
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2510.05619

 - **Pdf link:** https://arxiv.org/pdf/2510.05619

 - **Abstract**
 Current speech production systems predominantly rely on large transformer models that operate as black boxes, providing little interpretability or grounding in the physical mechanisms of human speech. We address this limitation by proposing a new framework: speech generation through explicit articulatory control. This reframes speech as a motor control task similar to robotic manipulation. Our approach uses reinforcement learning to train a policy that directly controls the movements of vocal tract articulators, such as the tongue, lips, and jaw, to produce syllable-level speech. Specifically, we employ the Proximal Policy Optimization algorithm to learn optimal articulatory movements based on acoustic feedback provided by our audio perceiver, Sylber. The resulting articulatory trajectories are decoded into audio using SPARC, a pre-trained articulatory-to-speech decoder. We train this framework on six target syllables, and it demonstrates successful convergence, with similarity scores between the policy-generated audio and the target syllables exceeding 0.85. Accurate human transcription of the audio for syllables such as "please", "loot", and "cat" demonstrates the intelligibility of this framework.
#### Investigation of perception inconsistency in speaker embedding for asynchronous voice anonymization
 - **Authors:** Rui Wang, Liping Chen, Kong Aik Lee, Zhengpeng Zha, Zhenhua Ling
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2510.05718

 - **Pdf link:** https://arxiv.org/pdf/2510.05718

 - **Abstract**
 Given the speech generation framework that represents the speaker attribute with an embedding vector, asynchronous voice anonymization can be achieved by modifying the speaker embedding derived from the original speech. However, the inconsistency between machine and human perceptions of the speaker attribute within the speaker embedding remains unexplored, limiting its performance in asynchronous voice anonymization. To this end, this study investigates this inconsistency via modifications to speaker embedding in the speech generation process. Experiments conducted on the FACodec and Diff-HierVC speech generation models discover a subspace whose removal alters machine perception while preserving its human perception of the speaker attribute in the generated speech. With these findings, an asynchronous voice anonymization is developed, achieving 100% human perception preservation rate while obscuring the machine perception. Audio samples can be found in this https URL.
#### Neural Forward Filtering for Speaker-Image Separation
 - **Authors:** Jingqi Sun, Shulin He, Ruizhe Pang, Zhong-Qiu Wang
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2510.05757

 - **Pdf link:** https://arxiv.org/pdf/2510.05757

 - **Abstract**
 We address monaural multi-speaker-image separation in reverberant conditions, aiming at separating mixed speakers but preserving the reverberation of each speaker. A straightforward approach for this task is to directly train end-to-end DNN systems to predict the reverberant speech of each speaker based on the input mixture. Although effective, this approach does not explicitly exploit the physical constraint that reverberant speech can be reproduced by convolving the direct-path signal with a linear filter. To address this, we propose CxNet, a two-DNN system with a neural forward filtering module in between. The first DNN is trained to jointly predict the direct-path signal and reverberant speech. Based on the direct-path estimate, the neural forward filtering module estimates the linear filter, and the estimated filter is then convolved with the direct-path estimate to obtain another estimate of reverberant speech, which is utilized as a discriminative feature to help the second DNN better estimate the reverberant speech. By explicitly modeling the linear filter, CxNet could leverage the physical constraint between the direct-path signal and reverberant speech to capture crucial information about reverberation tails. Evaluation results on the SMS-WSJ dataset show the effectiveness of the proposed algorithms.
#### Revisiting MFCCs: Evidence for Spectral-Prosodic Coupling
 - **Authors:** Vitor Magno de O. S. Bezerra, Gabriel F. A. Bastos, Jugurta Montalvão
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2510.05922

 - **Pdf link:** https://arxiv.org/pdf/2510.05922

 - **Abstract**
 Mel-frequency cepstral coefficients (MFCCs) are an important feature in speech processing. A deeper understanding of their properties can contribute to the work that is being done with both classical and deep learning models. This study challenges the long-held assumption that MFCCs lack relevant temporal information by investigating their relationship with speech prosody. Using a null hypothesis significance testing framework, a systematic assessment is made about the statistical independence between MFCCs and the three prosodic features: energy, fundamental frequency (F0), and voicing. The results demonstrate that it is statistically implausible that the MFCCs are independent of any of these three prosodic features. This finding suggests that MFCCs inherently carry valuable prosodic information, which can inform the design of future models in speech analysis and recognition.
#### Revisiting Modeling and Evaluation Approaches in Speech Emotion Recognition: Considering Subjectivity of Annotators and Ambiguity of Emotions
 - **Authors:** Huang-Cheng Chou, Chi-Chun Lee
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2510.05934

 - **Pdf link:** https://arxiv.org/pdf/2510.05934

 - **Abstract**
 Over the past two decades, speech emotion recognition (SER) has received growing attention. To train SER systems, researchers collect emotional speech databases annotated by crowdsourced or in-house raters who select emotions from predefined categories. However, disagreements among raters are common. Conventional methods treat these disagreements as noise, aggregating labels into a single consensus target. While this simplifies SER as a single-label task, it ignores the inherent subjectivity of human emotion perception. This dissertation challenges such assumptions and asks: (1) Should minority emotional ratings be discarded? (2) Should SER systems learn from only a few individuals' perceptions? (3) Should SER systems predict only one emotion per sample? Psychological studies show that emotion perception is subjective and ambiguous, with overlapping emotional boundaries. We propose new modeling and evaluation perspectives: (1) Retain all emotional ratings and represent them with soft-label distributions. Models trained on individual annotator ratings and jointly optimized with standard SER systems improve performance on consensus-labeled tests. (2) Redefine SER evaluation by including all emotional data and allowing co-occurring emotions (e.g., sad and angry). We propose an ``all-inclusive rule'' that aggregates all ratings to maximize diversity in label representation. Experiments on four English emotion databases show superior performance over majority and plurality labeling. (3) Construct a penalization matrix to discourage unlikely emotion combinations during training. Integrating it into loss functions further improves performance. Overall, embracing minority ratings, multiple annotators, and multi-emotion predictions yields more robust and human-aligned SER systems.
#### TokenChain: A Discrete Speech Chain via Semantic Token Modeling
 - **Authors:** Mingxuan Wang, Satoshi Nakamura
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Artificial Intelligence (cs.AI); Computation and Language (cs.CL); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2510.06201

 - **Pdf link:** https://arxiv.org/pdf/2510.06201

 - **Abstract**
 Machine Speech Chain, simulating the human perception-production loop, proves effective in jointly improving ASR and TTS. We propose TokenChain, a fully discrete speech chain coupling semantic-token ASR with a two-stage TTS: an autoregressive text-to-semantic model co-trained with ASR and a masked-generative semantic-to-acoustic model for synthesis only. End-to-end feedback across the text interface is enabled with straight-through argmax/Gumbel-Softmax and balanced with supervised ASR via dynamic weight averaging. Ablations examine optimal temperature schedules for in- and cross-domain transfer. Evaluation reveals TokenChain surpasses baseline accuracy 2-6 epochs earlier and yields 5-13% lower equal-epoch error with stable T2S on LibriSpeech, and reduces relative ASR WER by 56% and T2S WER by 31% on TED-LIUM with minimal forgetting, showing that chain learning remains effective with token interfaces and models.
#### ECTSpeech: Enhancing Efficient Speech Synthesis via Easy Consistency Tuning
 - **Authors:** Tao Zhu, Yinfeng Yu, Liejun Wang, Fuchun Sun, Wendong Zheng
 - **Subjects:** Subjects:
Sound (cs.SD); Artificial Intelligence (cs.AI); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2510.05984

 - **Pdf link:** https://arxiv.org/pdf/2510.05984

 - **Abstract**
 Diffusion models have demonstrated remarkable performance in speech synthesis, but typically require multi-step sampling, resulting in low inference efficiency. Recent studies address this issue by distilling diffusion models into consistency models, enabling efficient one-step generation. However, these approaches introduce additional training costs and rely heavily on the performance of pre-trained teacher models. In this paper, we propose ECTSpeech, a simple and effective one-step speech synthesis framework that, for the first time, incorporates the Easy Consistency Tuning (ECT) strategy into speech synthesis. By progressively tightening consistency constraints on a pre-trained diffusion model, ECTSpeech achieves high-quality one-step generation while significantly reducing training complexity. In addition, we design a multi-scale gate module (MSGate) to enhance the denoiser's ability to fuse features at different scales. Experimental results on the LJSpeech dataset demonstrate that ECTSpeech achieves audio quality comparable to state-of-the-art methods under single-step sampling, while substantially reducing the model's training cost and complexity.
#### Latent Speech-Text Transformer
 - **Authors:** Yen-Ju Lu, Yashesh Gaur, Wei Zhou, Benjamin Muller, Jesus Villalba, Najim Dehak, Luke Zettlemoyer, Gargi Ghosh, Mike Lewis, Srinivasan Iyer, Duc Le
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Artificial Intelligence (cs.AI); Machine Learning (cs.LG); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2510.06195

 - **Pdf link:** https://arxiv.org/pdf/2510.06195

 - **Abstract**
 Auto-regressive speech-text models are typically pre-trained on a large number of interleaved sequences of text tokens and raw speech encoded as speech tokens using vector quantization. These models have demonstrated state-of-the-art performance in speech-to-speech understanding and generation benchmarks, together with promising scaling laws, primarily enabled by the representational alignment between text and speech. Nevertheless, they suffer from shortcomings, partly owing to the disproportionately longer sequences of speech tokens in contrast to textual tokens. This results in a large compute imbalance between modalities during pre-training as well as during inference, and a potential hindrance to effectively aligning speech and text, ultimately translating to several orders of magnitude slower scaling laws. We introduce the Latent Speech-Text Transformer (LST), which makes pre-training speech-text models more data-efficient by dynamically and inexpensively aggregating speech tokens into latent speech patches. These patches serve as higher-level units that can either align with corresponding textual units to aid capability transfer or even encapsulate common speech sequences like silences to be more compute-efficient. We show that LST outperforms vanilla approaches on speech-to-speech as well as text-to-text benchmarks in both data- and compute-controlled settings, the former indicating more effective representational alignment and the latter indicating steeper scaling laws for speech-text models. On HellaSwag story completion, LST achieves 6.5% absolute gain in speech accuracy under compute-controlled training and 5.3% under data-controlled training, while also improving text performance. We will release our models, code, and the evaluation data to facilitate further research.


by Zyzzyva0381 (Windy). 


2025-10-08
