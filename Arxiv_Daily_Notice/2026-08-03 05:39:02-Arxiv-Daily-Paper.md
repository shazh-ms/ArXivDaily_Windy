# Showing new listings for Monday, 3 August 2026
Auto update papers at about 2:30am UTC (10:30am Beijing time) every weekday.


阅读 `Usage.md`了解如何使用此repo实现个性化的Arxiv论文推送

See `Usage.md` for instructions on how to personalize the repo. 


Keyword list: ['text-to-speech', 'text to speech', 'tts', 'LLM-based', 'speech', 'voice']


Excluded: []


### Today: 3papers 
#### Cloned Voices, Real Consequences: Evaluating Bias in Political Deepfake Detection for Electoral Integrity in Brazil
 - **Authors:** Lucas Rafael Stefanel Gris, Daniel Casanova, Frederico Santos De Oliveira, Alef Iury Ferreira, Beatriz Almeida Felício, Raul César Reis Mata, Anderson da Silva Soares
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2607.28770

 - **Pdf link:** https://arxiv.org/pdf/2607.28770

 - **Abstract**
 Recent advances in generative artificial intelligence have made it easier to fabricate statements and amplify political disinformation during elections. We introduce ParlaSpoof-BR, an audio deepfake dataset derived from recordings of the Brazilian Chamber of Deputies and expanded with synthetic utterances from diverse text-to-speech and voice conversion models. Using ParlaSpoof-BR, we benchmark state-of-the-art audio deepfake detectors, examine their ability to generalize to Brazilian Portuguese political speech, and investigate potential biases in their predictions. Our analysis reveals that current systems struggle to provide consistent decisions across the diversity represented in the dataset, with methodological factors (synthesis model choice, manipulation extent) dominating over demographic disparities. ParlaSpoof-BR provides a domain-specific benchmark for studying audio deepfake detection in a socially consequential and underrepresented setting, supporting the development of more robust detection systems for electoral integrity in Brazil.
#### Leveraging Beam Search Information for Confidence Estimation in E2E ASR
 - **Authors:** Yichen Jia, Hugo Van hamme
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2607.29299

 - **Pdf link:** https://arxiv.org/pdf/2607.29299

 - **Abstract**
 To estimate confidence for end-to-end Automatic Speech Recognition (ASR) systems, recent research has proposed Confidence Estimation Modules that incorporate features from the backbone ASR model. Most existing approaches, however, are architecture-dependent. In this paper, we propose the Score-Rank Confidence Estimation Module (SR-CEM), a lightweight module that leverages beam search information to generate token- and word-level confidence scores. Specifically, SR-CEM constructs features by combining the scores and ranks of tokens within a hypothesis. Experiments show that SR-CEM achieves effective calibration on both in-domain and out-of-domain English data. On the in-domain testset, it attains a Maximum Calibration Error of 4.50% and an Expected Calibration Error of 0.30% at the token level, significantly outperforming softmax confidence (20.04% and 1.75%, respectively). At the word level, SR-CEM achieves 8.17% and 0.35%, compared to 17.91% and 1.67% from softmax confidence. Furthermore, we demonstrate its robustness across hybrid and transducer ASR architectures with different decoding strategies, as well as on Dutch, noisy and conversational speech conditions. Our main finding is that SR-CEM is particularly effective in reducing Maximum Calibration Error, which is critical for reliable downstream use of ASR outputs, while maintaining architecture independence and generality across diverse evaluation conditions.
#### Stable Autoregressive Speech Generation with Low-Frame-Rate High-Dimensional Continuous Tokens
 - **Authors:** Yi Luo, Rongzhi Gu, Jixun Yao
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Artificial Intelligence (cs.AI); Machine Learning (cs.LG); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2607.29363

 - **Pdf link:** https://arxiv.org/pdf/2607.29363

 - **Abstract**
 Balancing sequence length, representational capacity, and long-horizon stability is a central problem in autoregressive (AR) speech and audio generation. Representations with higher frame rates or greater capacity can preserve more signal detail, but they also make streaming generation more vulnerable to distribution drift and AR error accumulation. Conversely, shorter and more compressed representations simplify AR modeling, but their limited bandwidth may discard important components and constrain the upper bound of reconstruction fidelity and generation quality. We ask whether a low-frame-rate, high-dimensional, high-bandwidth continuous representation can be co-designed with a streaming generation framework to support robust high-fidelity reconstruction, strong single-token predictability, and superior long-horizon stability. We decompose this goal into two coupled problems: what geometric and statistical properties a high-dimensional representation space should have, and how an AR continuous-token generator should be structured to resist error accumulation. Accordingly, we propose Locodec, a locally encoded codec that shapes its representation space to improve the interpolatability of a lower-dimensional core manifold and the identifiability of the native high-dimensional coordinates, thereby improving the predictability of high-dimensional high-bandwidth tokens. We also propose MP-ELD, a single-token AR flow-matching framework that uses multi-path information routing and residual classifier-free guidance to mitigate error accumulation. Experiments with 8-Hz, 768-dimensional tokens show that our design preserves reconstruction quality, improves single-token predictability, achieves competitive WER, and maintains stable long-form synthesis, without using external SSL/ASR models, pretrained text language models, or post-training stages.


by Zyzzyva0381 (Windy). 


2026-08-03
