# Showing new listings for Thursday, 17 April 2025
Auto update papers at about 2:30am UTC (10:30am Beijing time) every weekday.


阅读 `Usage.md`了解如何使用此repo实现个性化的Arxiv论文推送

See `Usage.md` for instructions on how to personalize the repo. 


Keyword list: ['acoustic', 'reinforcement learning', 'reverb', 'meta', 'separate', 'reconstruction', 'noise', 'enhance', 'localization', 'speech']


Excluded: []


### Today: 3papers 
#### Making Acoustic Side-Channel Attacks on Noisy Keyboards Viable with LLM-Assisted Spectrograms' "Typo" Correction
 - **Authors:** Seyyed Ali Ayati, Jin Hyun Park, Yichen Cai, Marcus Botacin
 - **Subjects:** Subjects:
Cryptography and Security (cs.CR); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2504.11622

 - **Pdf link:** https://arxiv.org/pdf/2504.11622

 - **Abstract**
 The large integration of microphones into devices increases the opportunities for Acoustic Side-Channel Attacks (ASCAs), as these can be used to capture keystrokes' audio signals that might reveal sensitive information. However, the current State-Of-The-Art (SOTA) models for ASCAs, including Convolutional Neural Networks (CNNs) and hybrid models, such as CoAtNet, still exhibit limited robustness under realistic noisy conditions. Solving this problem requires either: (i) an increased model's capacity to infer contextual information from longer sequences, allowing the model to learn that an initially noisily typed word is the same as a futurely collected non-noisy word, or (ii) an approach to fix misidentified information from the contexts, as one does not type random words, but the ones that best fit the conversation context. In this paper, we demonstrate that both strategies are viable and complementary solutions for making ASCAs practical. We observed that no existing solution leverages advanced transformer architectures' power for these tasks and propose that: (i) Visual Transformers (VTs) are the candidate solutions for capturing long-term contextual information and (ii) transformer-powered Large Language Models (LLMs) are the candidate solutions to fix the ``typos'' (mispredictions) the model might make. Thus, we here present the first-of-its-kind approach that integrates VTs and LLMs for ASCAs. We first show that VTs achieve SOTA performance in classifying keystrokes when compared to the previous CNN benchmark. Second, we demonstrate that LLMs can mitigate the impact of real-world noise. Evaluations on the natural sentences revealed that: (i) incorporating LLMs (e.g., GPT-4o) in our ASCA pipeline boosts the performance of error-correction tasks; and (ii) the comparable performance can be attained by a lightweight, fine-tuned smaller LLM (67 times smaller than GPT-4o), using...
#### Edge Intelligence for Wildlife Conservation: Real-Time Hornbill Call Classification Using TinyML
 - **Authors:** Kong Ka Hing, Mehran Behjati
 - **Subjects:** Subjects:
Sound (cs.SD); Machine Learning (cs.LG); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2504.12272

 - **Pdf link:** https://arxiv.org/pdf/2504.12272

 - **Abstract**
 Hornbills, an iconic species of Malaysia's biodiversity, face threats from habi-tat loss, poaching, and environmental changes, necessitating accurate and real-time population monitoring that is traditionally challenging and re-source intensive. The emergence of Tiny Machine Learning (TinyML) offers a chance to transform wildlife monitoring by enabling efficient, real-time da-ta analysis directly on edge devices. Addressing the challenge of wildlife conservation, this research paper explores the pivotal role of machine learn-ing, specifically TinyML, in the classification and monitoring of hornbill calls in Malaysia. Leveraging audio data from the Xeno-canto database, the study aims to develop a speech recognition system capable of identifying and classifying hornbill vocalizations. The proposed methodology involves pre-processing the audio data, extracting features using Mel-Frequency Energy (MFE), and deploying the model on an Arduino Nano 33 BLE, which is adept at edge computing. The research encompasses foundational work, in-cluding a comprehensive introduction, literature review, and methodology. The model is trained using Edge Impulse and validated through real-world tests, achieving high accuracy in hornbill species identification. The project underscores the potential of TinyML for environmental monitoring and its broader application in ecological conservation efforts, contributing to both the field of TinyML and wildlife conservation.
#### Dysarthria Normalization via Local Lie Group Transformations for Robust ASR
 - **Authors:** Mikhail Osipov
 - **Subjects:** Subjects:
Sound (cs.SD); Computation and Language (cs.CL); Machine Learning (cs.LG); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2504.12279

 - **Pdf link:** https://arxiv.org/pdf/2504.12279

 - **Abstract**
 We present a geometry-driven method for normalizing dysarthric speech using local Lie group transformations of spectrograms. Time, frequency, and amplitude distortions are modeled as smooth, invertible deformations, parameterized by scalar fields and applied via exponential maps. A neural network is trained to infer these fields from synthetic distortions of typical speech-without using any pathological data. At test time, the model applies an approximate inverse to real dysarthric inputs. Despite zero-shot generalization, we observe substantial ASR gains, including up to 16 percentage points WER reduction on challenging TORGO samples, with no degradation on clean speech. This work introduces a principled, interpretable approach for robust speech recognition under motor speech disorders


by Zyzzyva0381 (Windy). 


2025-04-17
