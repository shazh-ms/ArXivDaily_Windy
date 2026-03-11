# Showing new listings for Wednesday, 11 March 2026
Auto update papers at about 2:30am UTC (10:30am Beijing time) every weekday.


阅读 `Usage.md`了解如何使用此repo实现个性化的Arxiv论文推送

See `Usage.md` for instructions on how to personalize the repo. 


Keyword list: ['text-to-speech', 'text to speech', 'tts', 'LLM-based', 'speech', 'voice']


Excluded: []


### Today: 14papers 
#### Universal Speech Content Factorization
 - **Authors:** Henry Li Xinyuan, Zexin Cai, Lin Zhang, Leibny Paola García-Perera, Berrak Sisman, Sanjeev Khudanpur, Nicholas Andrews, Matthew Wiesner
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2603.08977

 - **Pdf link:** https://arxiv.org/pdf/2603.08977

 - **Abstract**
 We propose Universal Speech Content Factorization (USCF), a simple and invertible linear method for extracting a low-rank speech representation in which speaker timbre is suppressed while phonetic content is preserved. USCF extends Speech Content Factorization, a closed-set voice conversion (VC) method, to an open-set setting by learning a universal speech-to-content mapping via least-squares optimization and deriving speaker-specific transformations from only a few seconds of target speech. We show through embedding analysis that USCF effectively removes speaker-dependent variation. As a zero-shot VC system, USCF achieves competitive intelligibility, naturalness, and speaker similarity compared to methods that require substantially more target-speaker data or additional neural training. Finally, we demonstrate that as a training-efficient timbre-disentangled speech feature, USCF features can serve as the acoustic representation for training timbre-prompted text-to-speech models. Speech samples and code are publicly available.
#### Trade-offs Between Capacity and Robustness in Neural Audio Codecs for Adversarially Robust Speech Recognition
 - **Authors:** Jordan Prescott, Thanathai Lertpetchpun, Shrikanth Narayanan
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2603.09034

 - **Pdf link:** https://arxiv.org/pdf/2603.09034

 - **Abstract**
 Adversarial perturbations exploit vulnerabilities in automatic speech recognition (ASR) systems while preserving human perceived linguistic content. Neural audio codecs impose a discrete bottleneck that can suppress fine-grained signal variations associated with adversarial noise. We examine how the granularity of this bottleneck, controlled by residual vector quantization (RVQ) depth, shapes adversarial robustness. We observe a non-monotonic trade-off under gradient-based attacks: shallow quantization suppresses adversarial perturbations but degrades speech content, while deeper quantization preserves both content and perturbations. Intermediate depths balance these effects and minimize transcription error. We further show that adversarially induced changes in discrete codebook tokens strongly correlate with transcription error. These gains persist under adaptive attacks, where neural codec configurations outperform traditional compression defenses.
#### Emotion-Aware Prefix: Towards Explicit Emotion Control in Voice Conversion Models
 - **Authors:** Haoyuan Yang, Mu Yang, Jiamin Xie, Szu-Jui Chen, John H.L. Hansen
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2603.09120

 - **Pdf link:** https://arxiv.org/pdf/2603.09120

 - **Abstract**
 Recent advances in zero-shot voice conversion have exhibited potential in emotion control, yet the performance is suboptimal or inconsistent due to their limited expressive capacity. We propose Emotion-Aware Prefix for explicit emotion control in a two-stage voice conversion backbone. We significantly improve emotion conversion performance, doubling the baseline Emotion Conversion Accuracy (ECA) from 42.40% to 85.50% while maintaining linguistic integrity and speech quality, without compromising speaker identity. Our ablation study suggests that a joint control of both sequence modulation and acoustic realization is essential to synthesize distinct emotions. Furthermore, comparative analysis verifies the generalizability of proposed method, while it provides insights on the role of acoustic decoupling in maintaining speaker identity.
#### Acoustic and Semantic Modeling of Emotion in Spoken Language
 - **Authors:** Soumya Dutta
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2603.09212

 - **Pdf link:** https://arxiv.org/pdf/2603.09212

 - **Abstract**
 Emotions play a central role in human communication, shaping trust, engagement, and social interaction. As artificial intelligence systems powered by large language models become increasingly integrated into everyday life, enabling them to reliably understand and generate human emotions remains an important challenge. While emotional expression is inherently multimodal, this thesis focuses on emotions conveyed through spoken language and investigates how acoustic and semantic information can be jointly modeled to advance both emotion understanding and emotion synthesis from speech. The first part of the thesis studies emotion-aware representation learning through pre-training. We propose strategies that incorporate acoustic and semantic supervision to learn representations that better capture affective cues in speech. A speech-driven supervised pre-training framework is also introduced to enable large-scale emotion-aware text modeling without requiring manually annotated text corpora. The second part addresses emotion recognition in conversational settings. Hierarchical architectures combining cross-modal attention and mixture-of-experts fusion are developed to integrate acoustic and semantic information across conversational turns. Finally, the thesis introduces a textless and non-parallel speech-to-speech framework for emotion style transfer that enables controllable emotional transformations while preserving speaker identity and linguistic content. The results demonstrate improved emotion transfer and show that style-transferred speech can be used for data augmentation to improve emotion recognition.
#### StuPASE: Towards Low-Hallucination Studio-Quality Generative Speech Enhancement
 - **Authors:** Xiaobin Rong, Jun Gao, Zheng Wang, Mansur Yesilbursa, Kamil Wojcicki, Jing Lu
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2603.09234

 - **Pdf link:** https://arxiv.org/pdf/2603.09234

 - **Abstract**
 Achieving high perceptual quality without hallucination remains a challenge in generative speech enhancement (SE). A representative approach, PASE, is robust to hallucination but has limited perceptual quality under adverse conditions. We propose StuPASE, built upon PASE to achieve studio-level quality while retaining its low-hallucination property. First, we show that finetuning PASE with dry targets rather than targets containing simulated early reflections substantially improves dereverberation. Second, to address performance limitations under strong additive noise, we replace the GAN-based generative module in PASE with a flow-matching module, enabling studio-quality generation even under highly challenging conditions. Experiments demonstrate that StuPASE consistently produces perceptually high-quality speech while maintaining low hallucination, outperforming state-of-the-art SE methods. Audio demos are available at: this https URL.
#### End-to-End Direction-Aware Keyword Spotting with Spatial Priors in Noisy Environments
 - **Authors:** Rui Wang, Zhifei Zhang, Yu Gao, Xiaofeng Mou, Yi Xu
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2603.09505

 - **Pdf link:** https://arxiv.org/pdf/2603.09505

 - **Abstract**
 Keyword spotting (KWS) is crucial for many speech-driven applications, but robust KWS in noisy environments remains challenging. Conventional systems often rely on single-channel inputs and a cascaded pipeline separating front-end enhancement from KWS. This precludes joint optimization, inherently limiting performance. We present an end-to-end multi-channel KWS framework that exploits spatial cues to improve noise robustness. A spatial encoder learns inter-channel features, while a spatial embedding injects directional priors; the fused representation is processed by a streaming backbone. Experiments in simulated noisy conditions across multiple signal-to-noise ratios (SNRs) show that spatial modeling and directional priors each yield clear gains over baselines, with their combination achieving the best results. These findings validate end-to-end multi-channel spatial modeling, indicating strong potential for the target-speaker-aware detection in complex acoustic scenarios.
#### A Fast Solver for Interpolating Stochastic Differential Equation Diffusion Models for Speech Restoration
 - **Authors:** Bunlong Lay, Timo Gerkmann
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2603.09508

 - **Pdf link:** https://arxiv.org/pdf/2603.09508

 - **Abstract**
 Diffusion Probabilistic Models (DPMs) are a well-established class of diffusion models for unconditional image generation, while SGMSE+ is a well-established conditional diffusion model for speech enhancement. One of the downsides of diffusion models is that solving the reverse process requires many evaluations of a large Neural Network. Although advanced fast sampling solvers have been developed for DPMs, they are not directly applicable to models such as SGMSE+ due to differences in their diffusion processes. Specifically, DPMs transform between the data distribution and a standard Gaussian distribution, whereas SGMSE+ interpolates between the target distribution and a noisy observation. This work first develops a formalism of interpolating Stochastic Differential Equations (iSDEs) that includes SGMSE+, and second proposes a solver for iSDEs. The proposed solver enables fast sampling with as few as 10 Neural Network evaluations across multiple speech restoration tasks.
#### Speech-Omni-Lite: Portable Speech Interfaces for Vision-Language Models
 - **Authors:** Dehua Tao, Xuan Luo, Daxin Tan, Kai Chen, Lanqing Hong, Jing Li, Ruifeng Xu, Xiao Chen
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2603.09627

 - **Pdf link:** https://arxiv.org/pdf/2603.09627

 - **Abstract**
 While large-scale omni-models have demonstrated impressive capabilities across various modalities, their strong performance heavily relies on massive multimodal data and incurs substantial computational costs. This work introduces Speech-Omni-Lite, a cost-efficient framework for extending pre-trained Visual-Language (VL) backbones with speech understanding and generation capabilities, while fully preserving the backbones' vision-language performance. Specifically, the VL backbone is equipped with two lightweight, trainable plug-and-play modules, a speech projector and a speech token generator, while keeping the VL backbone fully frozen. To mitigate the scarcity of spoken QA corpora, a low-cost data construction strategy is proposed to generate Question-Text Answer-Text-Speech (QTATS) data from existing ASR speech-text pairs, facilitating effective speech generation training. Experimental results show that, even with only thousands of hours of speech training data, Speech-Omni-Lite achieves excellent spoken QA performance, which is comparable to omni-models trained on millions of hours of speech data. Furthermore, the learned speech modules exhibit strong transferability across VL backbones.
#### Finetuning a Text-to-Audio Model for Room Impulse Response Generation
 - **Authors:** Kirak Kim, Sungyoung Kim
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2603.09708

 - **Pdf link:** https://arxiv.org/pdf/2603.09708

 - **Abstract**
 Room Impulse Responses (RIRs) enable realistic acoustic simulation, with applications ranging from multimedia production to speech data augmentation. However, acquiring high-quality real-world RIRs is labor-intensive, and data scarcity remains a challenge for data-driven RIR generation approaches. In this paper, we propose a novel approach to RIR generation by fine-tuning a pre-trained text-to-audio model, demonstrating for the first time that large-scale generative audio priors can be effectively leveraged for the task. To address the lack of text-RIR paired data, we establish a labeling pipeline utilizing vision-language models to extract acoustic descriptions from existing image-RIR datasets. We introduce an in-context learning strategy to accommodate free-form user prompts during inference. Evaluations involving MUSHRA listening tests and downstream ASR performance demonstrate that our model generates plausible RIRs and serves as an effective tool for speech data augmentation.
#### A Semi-spontaneous Dutch Speech Dataset for Speech Enhancement and Speech Recognition
 - **Authors:** Dimme de Groot, Yuanyuan Zhang, Jorge Martinez, Odette Scharenborg
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2603.09725

 - **Pdf link:** https://arxiv.org/pdf/2603.09725

 - **Abstract**
 We present DRES: a 1.5-hour Dutch realistic elicited (semi-spontaneous) speech dataset from 80 speakers recorded in noisy, public indoor environments. DRES was designed as a test set for the evaluation of state-of-the-art (SOTA) automatic speech recognition (ASR) and speech enhancement (SE) models in a real-world scenario: a person speaking in a public indoor space with background talkers and noise. The speech was recorded with a four-channel linear microphone array. In this work we evaluate the speech quality of five well-known single-channel SE algorithms and the recognition performance of eight SOTA off-the-shelf ASR models before and after applying SE on the speech of DRES. We found that five out of the eight ASR models have WERs lower than 22% on DRES, despite the challenging conditions. In contrast to recent work, we did not find a positive effect of modern single-channel SE on ASR performance, emphasizing the importance of evaluating in realistic conditions.
#### Distributed Multichannel Wiener Filtering for Wireless Acoustic Sensor Networks
 - **Authors:** Paul Didier, Toon van Waterschoot, Simon Doclo, Jörg Bitzer, Pourya Behmandpoor, Henri Gode, Marc Moonen
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Information Theory (cs.IT); Signal Processing (eess.SP)
 - **Arxiv link:** https://arxiv.org/abs/2603.09735

 - **Pdf link:** https://arxiv.org/pdf/2603.09735

 - **Abstract**
 In a wireless acoustic sensor network (WASN), devices (i.e., nodes) can collaborate through distributed algorithms to collectively perform audio signal processing tasks. This paper focuses on the distributed estimation of node-specific desired speech signals using network-wide Wiener filtering. The objective is to match the performance of a centralized system that would have access to all microphone signals, while reducing the communication bandwidth usage of the algorithm. Existing solutions, such as the distributed adaptive node-specific signal estimation (DANSE) algorithm, converge towards the multichannel Wiener filter (MWF) which solves a centralized linear minimum mean square error (LMMSE) signal estimation problem. However, they do so iteratively, which can be slow and impractical. Many solutions also assume that all nodes observe the same set of sources of interest, which is often not the case in practice. To overcome these limitations, we propose the distributed multichannel Wiener filter (dMWF) for fully connected WASNs. The dMWF is non-iterative and optimal even when nodes observe different sets of sources. In this algorithm, nodes exchange neighbor-pair-specific, low-dimensional (fused) signals estimating the contribution of sources observed by both nodes in the pair. We formally prove the optimality of dMWF and demonstrate its performance in simulated speech enhancement experiments. The proposed algorithm is shown to outperform DANSE in terms of objective metrics after short operation times, highlighting the benefit of its iterationless design.
#### VoxEmo: Benchmarking Speech Emotion Recognition with Speech LLMs
 - **Authors:** Hezhao Zhang, Huang-Cheng Chou, Shrikanth Narayanan, Thomas Hain
 - **Subjects:** Subjects:
Sound (cs.SD); Artificial Intelligence (cs.AI); Computation and Language (cs.CL); Multimedia (cs.MM); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2603.08936

 - **Pdf link:** https://arxiv.org/pdf/2603.08936

 - **Abstract**
 Speech Large Language Models (LLMs) show great promise for speech emotion recognition (SER) via generative interfaces. However, shifting from closed-set classification to open text generation introduces zero-shot stochasticity, making evaluation highly sensitive to prompts. Additionally, conventional speech LLMs benchmarks overlook the inherent ambiguity of human emotion. Hence, we present VoxEmo, a comprehensive SER benchmark encompassing 35 emotion corpora across 15 languages for Speech LLMs. VoxEmo provides a standardized toolkit featuring varying prompt complexities, from direct classification to paralinguistic reasoning. To reflect real-world perception/application, we introduce a distribution-aware soft-label protocol and a prompt-ensemble strategy that emulates annotator disagreement. Experiments reveal that while zero-shot speech LLMs trail supervised baselines in hard-label accuracy, they uniquely align with human subjective distributions.
#### SPAR-K: Scheduled Periodic Alternating Early Exit for Spoken Language Models
 - **Authors:** Hsiao-Ying Huang, Cheng-Han Chiang, Hung-yi Lee
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2603.09215

 - **Pdf link:** https://arxiv.org/pdf/2603.09215

 - **Abstract**
 Interleaved spoken language models (SLMs) alternately generate text and speech tokens, but decoding at full transformer depth for every step becomes costly, especially due to long speech sequences. We propose SPAR-K, a modality-aware early exit framework designed to accelerate interleaved SLM inference while preserving perceptual quality. SPAR-K introduces a speech alternating-depth schedule: most speech positions exit at a fixed intermediate layer, while periodic full-depth "refresh" steps mitigate distribution shift due to early exit. We evaluate our framework using Step-Audio-2-mini and GLM-4-Voice across four datasets spanning reasoning, factual QA, and dialogue tasks, measuring performance in terms of ASR transcription accuracy and perceptual quality. Experimental results demonstrate that SPAR-K largely preserves question-answering accuracy with a maximum accuracy drop of 0.82\% while reducing average speech decoding depth by up to 11\% on Step-Audio-2-mini and 5\% on GLM-4-Voice, both with negligible changes in MOS and WER and no auxiliary computation overhead. We further demonstrate that confidence-based early exit strategies, widely used in text LLMs, are suboptimal for SLMs, highlighting that the unique statistical nature of speech tokens necessitates a specialized early exit design.
#### MUGEN: Evaluating and Improving Multi-audio Understanding of Large Audio-Language Models
 - **Authors:** Chih-Kai Yang, Yun-Shao Tsai, Yu-Kai Guo, Ping-Le Tsai, Yen-Ting Piao, Hung-Wei Chen, Ting-Lin Hsiao, Yun-Man Hsu, Ke-Han Lu, Hung-yi Lee
 - **Subjects:** Subjects:
Sound (cs.SD); Artificial Intelligence (cs.AI); Computation and Language (cs.CL); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2603.09714

 - **Pdf link:** https://arxiv.org/pdf/2603.09714

 - **Abstract**
 While multi-audio understanding is critical for large audio-language models (LALMs), it remains underexplored. We introduce MUGEN, a comprehensive benchmark evaluating this capability across speech, general audio, and music. Our experiments reveal consistent weaknesses in multi-audio settings, and performance degrades sharply as the number of concurrent audio inputs increases, identifying input scaling as a fundamental bottleneck. We further investigate training-free strategies and observe that Audio-Permutational Self-Consistency, which diversifies the order of audio candidates, helps models form more robust aggregated predictions, yielding up to 6.28% accuracy gains. Combining this permutation strategy with Chain-of-Thought further improves performance to 6.74%. These results expose blind spots in current LALMs and provide a foundation for evaluating complex auditory comprehension.


by Zyzzyva0381 (Windy). 


2026-03-11
