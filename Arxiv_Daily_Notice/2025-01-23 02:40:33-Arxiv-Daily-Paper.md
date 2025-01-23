# Showing new listings for Thursday, 23 January 2025
Auto update papers at about 2:30am UTC (10:30am Beijing time) every weekday.


阅读 `Usage.md`了解如何使用此repo实现个性化的Arxiv论文推送

See `Usage.md` for instructions on how to personalize the repo. 


Keyword list: ['acoustic', 'reinforcement learning', 'reverb', 'meta', 'separate', 'reconstruction', 'noise', 'enhance', 'localization', 'speech']


Excluded: []


### Today: 6papers 
#### A Domain Adaptation Framework for Speech Recognition Systems with Only Synthetic data
 - **Authors:** Minh Tran, Yutong Pang, Debjyoti Paul, Laxmi Pandey, Kevin Jiang, Jinxi Guo, Ke Li, Shun Zhang, Xuedong Zhang, Xin Lei
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2501.12501

 - **Pdf link:** https://arxiv.org/pdf/2501.12501

 - **Abstract**
 We introduce DAS (Domain Adaptation with Synthetic data), a novel domain adaptation framework for pre-trained ASR model, designed to efficiently adapt to various language-defined domains without requiring any real data. In particular, DAS first prompts large language models (LLMs) to generate domain-specific texts before converting these texts to speech via text-to-speech technology. The synthetic data is used to fine-tune Whisper with Low-Rank Adapters (LoRAs) for targeted domains such as music, weather, and sports. We introduce a novel one-pass decoding strategy that merges predictions from multiple LoRA adapters efficiently during the auto-regressive text generation process. Experimental results show significant improvements, reducing the Word Error Rate (WER) by 10% to 17% across all target domains compared to the original model, with minimal performance regression in out-of-domain settings (e.g., -1% on Librispeech test sets). We also demonstrate that DAS operates efficiently during inference, introducing an additional 9% increase in Real Time Factor (RTF) compared to the original model when inferring with three LoRA adapters.
#### EmoTech: A Multi-modal Speech Emotion Recognition Using Multi-source Low-level Information with Hybrid Recurrent Network
 - **Authors:** Shamin Bin Habib Avro, Taieba Taher, Nursadul Mamun
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2501.12674

 - **Pdf link:** https://arxiv.org/pdf/2501.12674

 - **Abstract**
 Emotion recognition is a critical task in human-computer interaction, enabling more intuitive and responsive systems. This study presents a multimodal emotion recognition system that combines low-level information from audio and text, leveraging both Convolutional Neural Networks (CNNs) and Bidirectional Long Short-Term Memory Networks (BiLSTMs). The proposed system consists of two parallel networks: an Audio Block and a Text Block. Mel Frequency Cepstral Coefficients (MFCCs) are extracted and processed by a BiLSTM network and a 2D convolutional network to capture low-level intrinsic and extrinsic features from speech. Simultaneously, a combined BiLSTM-CNN network extracts the low-level sequential nature of text from word embeddings corresponding to the available audio. This low-level information from speech and text is then concatenated and processed by several fully connected layers to classify the speech emotion. Experimental results demonstrate that the proposed EmoTech accurately recognizes emotions from combined audio and text inputs, achieving an overall accuracy of 84%. This solution outperforms previously proposed approaches for the same dataset and modalities.
#### EmoFormer: A Text-Independent Speech Emotion Recognition using a Hybrid Transformer-CNN model
 - **Authors:** Rashedul Hasan, Meher Nigar, Nursadul Mamun, Sayan Paul
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2501.12682

 - **Pdf link:** https://arxiv.org/pdf/2501.12682

 - **Abstract**
 Speech Emotion Recognition is a crucial area of research in human-computer interaction. While significant work has been done in this field, many state-of-the-art networks struggle to accurately recognize emotions in speech when the data is both speech and speaker-independent. To address this limitation, this study proposes, EmoFormer, a hybrid model combining CNNs (CNNs) with Transformer encoders to capture emotion patterns in speech data for such independent datasets. The EmoFormer network was trained and tested using the Expressive Anechoic Recordings of Speech (EARS) dataset, recently released by META. We experimented with two feature extraction techniques: MFCCs and x-vectors. The model was evaluated on different emotion sets comprising 5, 7, 10, and 23 distinct categories. The results demonstrate that the model achieved its best performance with five emotions, attaining an accuracy of 90%, a precision of 0.92, a recall, and an F1-score of 0.91. However, performance decreased as the number of emotions increased, with an accuracy of 83% for seven emotions compared to 70% for the baseline network. This study highlights the effectiveness of combining CNNs and Transformer-based architectures for emotion recognition from speech, particularly when using MFCC features.
#### Why disentanglement-based speaker anonymization systems fail at preserving emotions?
 - **Authors:** Ünal Ege Gaznepoglu, Nils Peters
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2501.13000

 - **Pdf link:** https://arxiv.org/pdf/2501.13000

 - **Abstract**
 Disentanglement-based speaker anonymization involves decomposing speech into a semantically meaningful representation, altering the speaker embedding, and resynthesizing a waveform using a neural vocoder. State-of-the-art systems of this kind are known to remove emotion information. Possible reasons include mode collapse in GAN-based vocoders, unintended modeling and modification of emotions through speaker embeddings, or excessive sanitization of the intermediate representation. In this paper, we conduct a comprehensive evaluation of a state-of-the-art speaker anonymization system to understand the underlying causes. We conclude that the main reason is the lack of emotion-related information in the intermediate representation. The speaker embeddings also have a high impact, if they are learned in a generative context. The vocoder's out-of-distribution performance has a smaller impact. Additionally, we discovered that synthesis artifacts increase spectral kurtosis, biasing emotion recognition evaluation towards classifying utterances as angry. Therefore, we conclude that reporting unweighted average recall alone for emotion recognition performance is suboptimal.
#### Retrieval-Augmented Neural Field for HRTF Upsampling and Personalization
 - **Authors:** Yoshiki Masuyama, Gordon Wichern, François G. Germain, Christopher Ick, Jonathan Le Roux
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2501.13017

 - **Pdf link:** https://arxiv.org/pdf/2501.13017

 - **Abstract**
 Head-related transfer functions (HRTFs) with dense spatial grids are desired for immersive binaural audio generation, but their recording is time-consuming. Although HRTF spatial upsampling has shown remarkable progress with neural fields, spatial upsampling only from a few measured directions, e.g., 3 or 5 measurements, is still challenging. To tackle this problem, we propose a retrieval-augmented neural field (RANF). RANF retrieves a subject whose HRTFs are close to those of the target subject from a dataset. The HRTF of the retrieved subject at the desired direction is fed into the neural field in addition to the sound source direction itself. Furthermore, we present a neural network that can efficiently handle multiple retrieved subjects, inspired by a multi-channel processing technique called transform-average-concatenate. Our experiments confirm the benefits of RANF on the SONICOM dataset, and it is a key component in the winning solution of Task 2 of the listener acoustic personalization challenge 2024.
#### FlanEC: Exploring Flan-T5 for Post-ASR Error Correction
 - **Authors:** Moreno La Quatra, Valerio Mario Salerno, Yu Tsao, Sabato Marco Siniscalchi
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Artificial Intelligence (cs.AI); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2501.12979

 - **Pdf link:** https://arxiv.org/pdf/2501.12979

 - **Abstract**
 In this paper, we present an encoder-decoder model leveraging Flan-T5 for post-Automatic Speech Recognition (ASR) Generative Speech Error Correction (GenSEC), and we refer to it as FlanEC. We explore its application within the GenSEC framework to enhance ASR outputs by mapping n-best hypotheses into a single output sentence. By utilizing n-best lists from ASR models, we aim to improve the linguistic correctness, accuracy, and grammaticality of final ASR transcriptions. Specifically, we investigate whether scaling the training data and incorporating diverse datasets can lead to significant improvements in post-ASR error correction. We evaluate FlanEC using the HyPoradise dataset, providing a comprehensive analysis of the model's effectiveness in this domain. Furthermore, we assess the proposed approach under different settings to evaluate model scalability and efficiency, offering valuable insights into the potential of instruction-tuned encoder-decoder models for this task.


by Zyzzyva0381 (Windy). 


2025-01-23
