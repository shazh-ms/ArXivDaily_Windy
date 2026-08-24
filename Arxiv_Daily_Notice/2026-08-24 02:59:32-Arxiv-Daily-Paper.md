# Showing new listings for Monday, 24 August 2026
Auto update papers at about 2:30am UTC (10:30am Beijing time) every weekday.


阅读 `Usage.md`了解如何使用此repo实现个性化的Arxiv论文推送

See `Usage.md` for instructions on how to personalize the repo. 


Keyword list: ['text-to-speech', 'text to speech', 'tts', 'LLM-based', 'speech', 'voice']


Excluded: []


### Today: 5papers 
#### Training DeepFilterNet with Accurate Room Acoustic Simulations Improves Single-Channel Speech Enhancement
 - **Authors:** Alessia Milo, Georg Götz, Steinar Guðjónsson, Daniel Gert Nielsen, Jesper Pedersen, Finnur Pind
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Machine Learning (cs.LG); Computational Physics (physics.comp-ph)
 - **Arxiv link:** https://arxiv.org/abs/2608.20971

 - **Pdf link:** https://arxiv.org/pdf/2608.20971

 - **Abstract**
 We investigate how the realism of synthetic room impulse response (RIR) datasets affects the training of DeepFilterNet3 for single-channel speech enhancement. We compare a DNS4 image-source-method (ISM) RIR dataset with a higher-acoustic-fidelity dataset generated using hybrid wave-based and geometrical acoustics simulation. Rather than isolating individual simulation factors, we compare complete RIR generation pipelines while keeping the enhancement model unchanged. Models are evaluated on unseen measured RIRs using objective speech enhancement metrics and downstream automatic speech recognition (ASR). Training with the higher-fidelity dataset consistently yields modest improvements in objective metrics and substantially lower ASR word error rates than the ISM dataset. Although the experiments do not attribute these gains to individual modelling components, they show that increasing the overall realism of synthetic acoustic training data improves the generalization of DeepFilterNet3 to unseen measured environments.
#### μNet: Ultra-Low-Memory and Low-Complexity Speech Enhancement for Embedded Digital Signal Processors
 - **Authors:** Shrishti Saha Shetu, Jose Miguel Martinez Aponte, Nagashree K. S. Rao, Sharvin Vittappan, Oliver Thiergart, Emanuël A. P. Habets
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2608.21155

 - **Pdf link:** https://arxiv.org/pdf/2608.21155

 - **Abstract**
 Speech enhancement on embedded digital signal processors (DSPs) imposes strict constraints on memory footprint, computational complexity, latency, and support for integer operations. Although recent DNN-based approaches have addressed these challenges individually, no unified framework in the literature simultaneously addresses all these requirements for practical deployment. In this work, we propose {\mu}Net, an ultra-low-memory, low-complexity, and low-latency end-to-end DNN model. The proposed method requires only $90$~KB of static memory and $28$~MMACs, while supporting an algorithmic latency as low as $4$~ms with performance comparable to state-of-the-art methods of similar complexity. Our experiments demonstrate that {\mu}Net is compatible with neural accelerators and supports full integer-arithmetic operations on consumer DSP platforms such as Cadence Tensilica HiFi 4/5.
#### SlimDiffuSE: Towards Efficient Diffusion-Based Speech Enhancement using Slimmable Networks
 - **Authors:** Nagashree K. S. Rao, Shrishti Saha Shetu, Mohamed Elminshawi, Emanuël A. P. Habets, Andreas Brendel
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2608.21188

 - **Pdf link:** https://arxiv.org/pdf/2608.21188

 - **Abstract**
 Diffusion-based models are emerging in the speech enhancement domain and are achieving state-of-the-art performance across various benchmark datasets. A major downside of diffusion models is that data generation requires many evaluations of a typically large neural network, which results in high overall complexity. In this work, we propose a slimmable diffusion model that employs adaptive network widths throughout the data generation process to reduce computational cost. By using a greedy search algorithm to optimize the network width schedule, our method achieves performance comparable to baseline diffusion models with significantly reduced computational complexity. Notably, our approach reduces the computational complexity by up to $87.5\%$ without a significant drop in objective metrics, such as perceptual evaluation of speech quality (PESQ) and SI-SDR.
#### TurboBias 2.0: Streaming Context-Biasing for Production-Efficient ASR Systems
 - **Authors:** Vladimir Bataev, Lilit Grigoryan, Andrei Andrusenko, Nikolay Karpov, Vitaly Lavrukhin, Boris Ginsburg
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Artificial Intelligence (cs.AI); Computation and Language (cs.CL); Machine Learning (cs.LG); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2608.21343

 - **Pdf link:** https://arxiv.org/pdf/2608.21343

 - **Abstract**
 Contextualization is essential for production automatic speech recognition (ASR) systems, where user-provided phrases must be recognized accurately under strict latency constraints. Although many context-biasing methods improve recognition accuracy, they often do not address the practical requirements of modern production ASR systems: streaming inference, efficient batched decoding, user-specific context lists, and low runtime overhead. We propose TurboBias 2.0, a production-oriented framework for efficient phrase boosting in Transducer-based ASR systems. The framework extends GPU-accelerated TurboBias with a case-insensitive boosting graph and per-stream batched decoding, allowing each utterance in a batch to use an independent context-biasing configuration. This enables personalized context biasing for multiple simultaneous users without sharing or mixing their context lists. The proposed framework supports both offline and streaming inference and can be used with greedy and beam-search decoding. Experiments show that TurboBias 2.0 improves contextual phrase recognition while preserving low latency and high throughput.
#### Building and Evaluating a Synthetic Bengali Speech Resource for Telecom Customer Care
 - **Authors:** Kawshik Kumar Paul, Md. Nafiul Alam Fuji
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2608.20346

 - **Pdf link:** https://arxiv.org/pdf/2608.20346

 - **Abstract**
 Speech systems used in customer-facing applications often require domain-specific language coverage. We present a synthetic Bengali speech dataset for telecom customer-care scenarios. The dataset contains 10,000 audio-text pairs, approximately 26.82 hours of 24 kHz speech, and predefined train, validation, and test splits of 9,000, 500, and 500 examples. It is publicly released on Hugging Face under the CC-BY-4.0 license. The speech was generated with OmniVoice in voice-cloning mode using a real female reference recording and transcript, with bfloat16 precision, 16 diffusion sampling steps, and a speaking-rate control value of 1.0. Along with the original Bengali text, the dataset provides a normalized transcript field designed for ASR/STT training and evaluation. We report an automatic intelligibility check over all 10,000 samples using a domain-adapted Whisper ASR model fine-tuned from bengaliAI/tugstugi_bengaliai-regional-asr_whisper-medium, along with a manual listening check on selected samples. The evaluation gives an average WER of 2.54%, an average CER of 0.59%, and median WER and CER values of 0.00%. These results suggest strong text-audio consistency under the selected automatic evaluation pipeline, while the paper also discusses the limitations of synthetic speech and STT-based evaluation.


by Zyzzyva0381 (Windy). 


2026-08-24
