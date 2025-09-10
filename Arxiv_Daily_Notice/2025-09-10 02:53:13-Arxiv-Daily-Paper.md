# Showing new listings for Wednesday, 10 September 2025
Auto update papers at about 2:30am UTC (10:30am Beijing time) every weekday.


阅读 `Usage.md`了解如何使用此repo实现个性化的Arxiv论文推送

See `Usage.md` for instructions on how to personalize the repo. 


Keyword list: ['text-to-speech', 'text to speech', 'tts', 'LLM-based', 'speech', 'voice']


Excluded: []


### Today: 6papers 
#### Identifying and Calibrating Overconfidence in Noisy Speech Recognition
 - **Authors:** Mingyue Huo, Yuheng Zhang, Yan Tang
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2509.07195

 - **Pdf link:** https://arxiv.org/pdf/2509.07195

 - **Abstract**
 Modern end-to-end automatic speech recognition (ASR) models like Whisper not only suffer from reduced recognition accuracy in noise, but also exhibit overconfidence - assigning high confidence to wrong predictions. We conduct a systematic analysis of Whisper's behavior in additive noise conditions and find that overconfident errors increase dramatically at low signal-to-noise ratios, with 10-20% of tokens incorrectly predicted with confidence above 0.7. To mitigate this, we propose a lightweight, post-hoc calibration framework that detects potential overconfidence and applies temperature scaling selectively to those tokens, without altering the underlying ASR model. Evaluations on the R-SPIN dataset demonstrate that, in the low signal-to-noise ratio range (-18 to -5 dB), our method reduces the expected calibration error (ECE) by 58% and triples the normalized cross entropy (NCE), yielding more reliable confidence estimates under severe noise conditions.
#### Affine Modulation-based Audiogram Fusion Network for Joint Noise Reduction and Hearing Loss Compensation
 - **Authors:** Ye Ni, Ruiyu Liang, Xiaoshuai Hao, Jiaming Cheng, Qingyun Wang, Chengwei Huang, Cairong Zou, Wei Zhou, Weiping Ding, Björn W. Schuller
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2509.07341

 - **Pdf link:** https://arxiv.org/pdf/2509.07341

 - **Abstract**
 Hearing aids (HAs) are widely used to provide personalized speech enhancement (PSE) services, improving the quality of life for individuals with hearing loss. However, HA performance significantly declines in noisy environments as it treats noise reduction (NR) and hearing loss compensation (HLC) as separate tasks. This separation leads to a lack of systematic optimization, overlooking the interactions between these two critical tasks, and increases the system complexity. To address these challenges, we propose a novel audiogram fusion network, named AFN-HearNet, which simultaneously tackles the NR and HLC tasks by fusing cross-domain audiogram and spectrum features. We propose an audiogram-specific encoder that transforms the sparse audiogram profile into a deep representation, addressing the alignment problem of cross-domain features prior to fusion. To incorporate the interactions between NR and HLC tasks, we propose the affine modulation-based audiogram fusion frequency-temporal Conformer that adaptively fuses these two features into a unified deep representation for speech reconstruction. Furthermore, we introduce a voice activity detection auxiliary training task to embed speech and non-speech patterns into the unified deep representation implicitly. We conduct comprehensive experiments across multiple datasets to validate the effectiveness of each proposed module. The results indicate that the AFN-HearNet significantly outperforms state-of-the-art in-context fusion joint models regarding key metrics such as HASQI and PESQ, achieving a considerable trade-off between performance and efficiency. The source code and data will be released at this https URL.
#### Prototype: A Keyword Spotting-Based Intelligent Audio SoC for IoT
 - **Authors:** Huihong Liang, Dongxuan Jia, Youquan Wang, Longtao Huang, Shida Zhong, Luping Xiang, Lei Huang, Tao Yuan
 - **Subjects:** Subjects:
Sound (cs.SD); Hardware Architecture (cs.AR); Human-Computer Interaction (cs.HC); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2509.06964

 - **Pdf link:** https://arxiv.org/pdf/2509.06964

 - **Abstract**
 In this demo, we present a compact intelligent audio system-on-chip (SoC) integrated with a keyword spotting accelerator, enabling ultra-low latency, low-power, and low-cost voice interaction in Internet of Things (IoT) devices. Through algorithm-hardware co-design, the system's energy efficiency is maximized. We demonstrate the system's capabilities through a live FPGA-based prototype, showcasing stable performance and real-time voice interaction for edge intelligence applications.
#### Controllable Singing Voice Synthesis using Phoneme-Level Energy Sequence
 - **Authors:** Yerin Ryu, Inseop Shin, Chanwoo Kim
 - **Subjects:** Subjects:
Sound (cs.SD); Artificial Intelligence (cs.AI); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2509.07038

 - **Pdf link:** https://arxiv.org/pdf/2509.07038

 - **Abstract**
 Controllable Singing Voice Synthesis (SVS) aims to generate expressive singing voices reflecting user intent. While recent SVS systems achieve high audio quality, most rely on probabilistic modeling, limiting precise control over attributes such as dynamics. We address this by focusing on dynamic control--temporal loudness variation essential for musical expressiveness--and explicitly condition the SVS model on energy sequences extracted from ground-truth spectrograms, reducing annotation costs and improving controllability. We also propose a phoneme-level energy sequence for user-friendly control. To the best of our knowledge, this is the first attempt enabling user-driven dynamics control in SVS. Experiments show our method achieves over 50% reduction in mean absolute error of energy sequences for phoneme-level inputs compared to baseline and energy-predictor models, without compromising synthesis quality.
#### The ML-SUPERB 2.0 Challenge: Towards Inclusive ASR Benchmarking for All Language Varieties
 - **Authors:** William Chen, Chutong Meng, Jiatong Shi, Martijn Bartelds, Shih-Heng Wang, Hsiu-Hsuan Wang, Rafael Mosquera, Sara Hincapie, Dan Jurafsky, Antonis Anastasopoulos, Hung-yi Lee, Karen Livescu, Shinji Watanabe
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2509.07139

 - **Pdf link:** https://arxiv.org/pdf/2509.07139

 - **Abstract**
 Recent improvements in multilingual ASR have not been equally distributed across languages and language varieties. To advance state-of-the-art (SOTA) ASR models, we present the Interspeech 2025 ML-SUPERB 2.0 Challenge. We construct a new test suite that consists of data from 200+ languages, accents, and dialects to evaluate SOTA multilingual speech models. The challenge also introduces an online evaluation server based on DynaBench, allowing for flexibility in model design and architecture for participants. The challenge received 5 submissions from 3 teams, all of which outperformed our baselines. The best-performing submission achieved an absolute improvement in LID accuracy of 23% and a reduction in CER of 18% when compared to the best baseline on a general multilingual test set. On accented and dialectal data, the best submission obtained 30.2% lower CER and 15.7% higher LID accuracy, showing the importance of community challenges in making speech technologies more inclusive.
#### Spectral and Rhythm Feature Performance Evaluation for Category and Class Level Audio Classification with Deep Convolutional Neural Networks
 - **Authors:** Friedrich Wolf-Monheim
 - **Subjects:** Subjects:
Sound (cs.SD); Artificial Intelligence (cs.AI); Computer Vision and Pattern Recognition (cs.CV); Machine Learning (cs.LG); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2509.07756

 - **Pdf link:** https://arxiv.org/pdf/2509.07756

 - **Abstract**
 Next to decision tree and k-nearest neighbours algorithms deep convolutional neural networks (CNNs) are widely used to classify audio data in many domains like music, speech or environmental sounds. To train a specific CNN various spectral and rhythm features like mel-scaled spectrograms, mel-frequency cepstral coefficients (MFCC), cyclic tempograms, short-time Fourier transform (STFT) chromagrams, constant-Q transform (CQT) chromagrams and chroma energy normalized statistics (CENS) chromagrams can be used as digital image input data for the neural network. The performance of these spectral and rhythm features for audio category level as well as audio class level classification is investigated in detail with a deep CNN and the ESC-50 dataset with 2,000 labeled environmental audio recordings using an end-to-end deep learning pipeline. The evaluated metrics accuracy, precision, recall and F1 score for multiclass classification clearly show that the mel-scaled spectrograms and the mel-frequency cepstral coefficients (MFCC) perform significantly better then the other spectral and rhythm features investigated in this research for audio classification tasks using deep CNNs.


by Zyzzyva0381 (Windy). 


2025-09-10
