# Showing new listings for Monday, 1 September 2025
Auto update papers at about 2:30am UTC (10:30am Beijing time) every weekday.


阅读 `Usage.md`了解如何使用此repo实现个性化的Arxiv论文推送

See `Usage.md` for instructions on how to personalize the repo. 


Keyword list: ['text-to-speech', 'text to speech', 'tts', 'LLM-based', 'speech', 'voice']


Excluded: []


### Today: 7papers 
#### Benchmarking Large Pretrained Multilingual Models on Québec French Speech Recognition
 - **Authors:** Coralie Serrand, Gilles Boulianne, Amira Morsli
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2508.21193

 - **Pdf link:** https://arxiv.org/pdf/2508.21193

 - **Abstract**
 We evaluate the performance of large pretrained multilingual speech recognition models on a regional variety of French spoken in Québec, Canada, in terms of speed, word error rate and semantic accuracy. To this end we build a benchmark and evaluation pipeline based on the CommissionsQc datasets, a corpus of spontaneous conversations recorded during public inquiries recently held in Québec. Published results for these models on well-known benchmarks such as FLEURS or CommonVoice are not good predictors of the performance we observe on CommissionsQC. Our results should be of interest for practitioners interested in building speech applications for realistic conditions or regional language varieties.
#### Can Layer-wise SSL Features Improve Zero-Shot ASR Performance for Children's Speech?
 - **Authors:** Abhijit Sinha, Hemant Kumar Kathania, Sudarsana Reddy Kadiri, Shrikanth Narayanan
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Artificial Intelligence (cs.AI); Machine Learning (cs.LG); Sound (cs.SD); Signal Processing (eess.SP)
 - **Arxiv link:** https://arxiv.org/abs/2508.21225

 - **Pdf link:** https://arxiv.org/pdf/2508.21225

 - **Abstract**
 Automatic Speech Recognition (ASR) systems often struggle to accurately process children's speech due to its distinct and highly variable acoustic and linguistic characteristics. While recent advancements in self-supervised learning (SSL) models have greatly enhanced the transcription of adult speech, accurately transcribing children's speech remains a significant challenge. This study investigates the effectiveness of layer-wise features extracted from state-of-the-art SSL pre-trained models - specifically, Wav2Vec2, HuBERT, Data2Vec, and WavLM in improving the performance of ASR for children's speech in zero-shot scenarios. A detailed analysis of features extracted from these models was conducted, integrating them into a simplified DNN-based ASR system using the Kaldi toolkit. The analysis identified the most effective layers for enhancing ASR performance on children's speech in a zero-shot scenario, where WSJCAM0 adult speech was used for training and PFSTAR children speech for testing. Experimental results indicated that Layer 22 of the Wav2Vec2 model achieved the lowest Word Error Rate (WER) of 5.15%, representing a 51.64% relative improvement over the direct zero-shot decoding using Wav2Vec2 (WER of 10.65%). Additionally, age group-wise analysis demonstrated consistent performance improvements with increasing age, along with significant gains observed even in younger age groups using the SSL features. Further experiments on the CMU Kids dataset confirmed similar trends, highlighting the generalizability of the proposed approach.
#### Zero-Shot KWS for Children's Speech using Layer-Wise Features from SSL Models
 - **Authors:** Subham Kutum, Abhijit Sinha, Hemant Kumar Kathania, Sudarsana Reddy Kadiri, Mahesh Chandra Govil
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Artificial Intelligence (cs.AI); Human-Computer Interaction (cs.HC); Sound (cs.SD); Signal Processing (eess.SP)
 - **Arxiv link:** https://arxiv.org/abs/2508.21248

 - **Pdf link:** https://arxiv.org/pdf/2508.21248

 - **Abstract**
 Numerous methods have been proposed to enhance Keyword Spotting (KWS) in adult speech, but children's speech presents unique challenges for KWS systems due to its distinct acoustic and linguistic characteristics. This paper introduces a zero-shot KWS approach that leverages state-of-the-art self-supervised learning (SSL) models, including Wav2Vec2, HuBERT and Data2Vec. Features are extracted layer-wise from these SSL models and used to train a Kaldi-based DNN KWS system. The WSJCAM0 adult speech dataset was used for training, while the PFSTAR children's speech dataset was used for testing, demonstrating the zero-shot capability of our method. Our approach achieved state-of-the-art results across all keyword sets for children's speech. Notably, the Wav2Vec2 model, particularly layer 22, performed the best, delivering an ATWV score of 0.691, a MTWV score of 0.7003 and probability of false alarm and probability of miss of 0.0164 and 0.0547 respectively, for a set of 30 keywords. Furthermore, age-specific performance evaluation confirmed the system's effectiveness across different age groups of children. To assess the system's robustness against noise, additional experiments were conducted using the best-performing layer of the best-performing Wav2Vec2 model. The results demonstrated a significant improvement over traditional MFCC-based baseline, emphasizing the potential of SSL embeddings even in noisy conditions. To further generalize the KWS framework, the experiments were repeated for an additional CMU dataset. Overall the results highlight the significant contribution of SSL features in enhancing Zero-Shot KWS performance for children's speech, effectively addressing the challenges associated with the distinct characteristics of child speakers.
#### Cochleagram-based Noise Adapted Speaker Identification System for Distorted Speech
 - **Authors:** Sabbir Ahmed, Nursadul Mamun, Md Azad Hossain
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2508.21347

 - **Pdf link:** https://arxiv.org/pdf/2508.21347

 - **Abstract**
 Speaker Identification refers to the process of identifying a person using one's voice from a collection of known speakers. Environmental noise, reverberation and distortion make the task of automatic speaker identification challenging as extracted features get degraded thus affecting the performance of the speaker identification (SID) system. This paper proposes a robust noise adapted SID system under noisy, mismatched, reverberated and distorted environments. This method utilizes an auditory features called cochleagram to extract speaker characteristics and thus identify the speaker. A $128$ channel gammatone filterbank with a frequency range from $50$ to $8000$ Hz was used to generate 2-D cochleagrams. Wideband as well as narrowband noises were used along with clean speech to obtain noisy cochleagrams at various levels of signal to noise ratio (SNR). Both clean and noisy cochleagrams of only $-5$ dB SNR were then fed into a convolutional neural network (CNN) to build a speaker model in order to perform SID which is referred as noise adapted speaker model (NASM). The NASM was trained using a certain noise and then was evaluated using clean and various types of noises. Moreover, the robustness of the proposed system was tested using reverberated as well as distorted test data. Performance of the proposed system showed a measurable accuracy improvement over existing neurogram based SID system.
#### Fundamentals of Data-Driven Approaches to Acoustic Signal Detection, Filtering, and Transformation
 - **Authors:** Chao Pan
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2508.21470

 - **Pdf link:** https://arxiv.org/pdf/2508.21470

 - **Abstract**
 In recent decades, the field of signal processing has rapidly evolved due to diverse application demands, leading to a rich array of scientific questions and research areas. The forms of signals, their formation mechanisms, and the information extraction methods vary by application, resulting in diverse signal processing techniques. Common techniques can be categorized into three types: transformation, detection, and filtering. Signal transformation converts signals from their original domain to a more suitable target domain for analysis; signal detection aims to identify the existence of relevant information within a signal and its specific time and location; and signal filtering focuses on extracting or separating source signals of interest from observed signals. In acoustic signal processing, techniques include sound source localization, sound event detection, voiceprint extraction and recognition, noise reduction, and source separation, with applications in speech communication, voice interaction, smart healthcare, and industrial diagnostics. Recently, the advancement of deep learning technologies has shifted methodologies in acoustic signal processing from knowledge-driven to data-driven approaches, leading to significant research outcomes. This paper aims to systematically summarize the principles and methods of data-driven acoustic signal processing, providing a comprehensive understanding framework for academic exploration and practical applications.
#### Towards Improved Speech Recognition through Optimized Synthetic Data Generation
 - **Authors:** Yanis Perrin, Gilles Boulianne
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2508.21631

 - **Pdf link:** https://arxiv.org/pdf/2508.21631

 - **Abstract**
 Supervised training of speech recognition models requires access to transcribed audio data, which often is not possible due to confidentiality issues. Our approach to this problem is to generate synthetic audio from a text-only corpus using a state-of-the-art text-to-speech model with voice cloning capabilities. Our goal is to achieve automatic speech recognition (ASR) performance comparable to models trained on real data. We explore ways to optimize synthetic data generation through finetuning, filtering and evaluation, and its use for training an end-to-end encoder-decoder ASR model. Experiments were conducted using two datasets of spontaneous, conversational speech in Québec French. We show that improving data generation leads to large improvements in the final ASR system trained on synthetic data.
#### WaveLLDM: Design and Development of a Lightweight Latent Diffusion Model for Speech Enhancement and Restoration
 - **Authors:** Kevin Putra Santoso, Rizka Wakhidatus Sholikah, Raden Venantius Hari Ginardi
 - **Subjects:** Subjects:
Sound (cs.SD); Artificial Intelligence (cs.AI); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2508.21153

 - **Pdf link:** https://arxiv.org/pdf/2508.21153

 - **Abstract**
 High-quality audio is essential in a wide range of applications, including online communication, virtual assistants, and the multimedia industry. However, degradation caused by noise, compression, and transmission artifacts remains a major challenge. While diffusion models have proven effective for audio restoration, they typically require significant computational resources and struggle to handle longer missing segments. This study introduces WaveLLDM (Wave Lightweight Latent Diffusion Model), an architecture that integrates an efficient neural audio codec with latent diffusion for audio restoration and denoising. Unlike conventional approaches that operate in the time or spectral domain, WaveLLDM processes audio in a compressed latent space, reducing computational complexity while preserving reconstruction quality. Empirical evaluations on the Voicebank+DEMAND test set demonstrate that WaveLLDM achieves accurate spectral reconstruction with low Log-Spectral Distance (LSD) scores (0.48 to 0.60) and good adaptability to unseen data. However, it still underperforms compared to state-of-the-art methods in terms of perceptual quality and speech clarity, with WB-PESQ scores ranging from 1.62 to 1.71 and STOI scores between 0.76 and 0.78. These limitations are attributed to suboptimal architectural tuning, the absence of fine-tuning, and insufficient training duration. Nevertheless, the flexible architecture that combines a neural audio codec and latent diffusion model provides a strong foundation for future development.


by Zyzzyva0381 (Windy). 


2025-09-01
