# Showing new listings for Thursday, 15 May 2025
Auto update papers at about 2:30am UTC (10:30am Beijing time) every weekday.


阅读 `Usage.md`了解如何使用此repo实现个性化的Arxiv论文推送

See `Usage.md` for instructions on how to personalize the repo. 


Keyword list: ['text-to-speech', 'text to speech', 'tts', 'LLM-based', 'speech', 'voice']


Excluded: []


### Today: 5papers 
#### Omni-R1: Do You Really Need Audio to Fine-Tune Your Audio LLM?
 - **Authors:** Andrew Rouditchenko, Saurabhchand Bhati, Edson Araujo, Samuel Thomas, Hilde Kuehne, Rogerio Feris, James Glass
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2505.09439

 - **Pdf link:** https://arxiv.org/pdf/2505.09439

 - **Abstract**
 We propose Omni-R1 which fine-tunes a recent multi-modal LLM, Qwen2.5-Omni, on an audio question answering dataset with the reinforcement learning method GRPO. This leads to new State-of-the-Art performance on the recent MMAU benchmark. Omni-R1 achieves the highest accuracies on the sounds, music, speech, and overall average categories, both on the Test-mini and Test-full splits. To understand the performance improvement, we tested models both with and without audio and found that much of the performance improvement from GRPO could be attributed to better text-based reasoning. We also made a surprising discovery that fine-tuning without audio on a text-only dataset was effective at improving the audio-based performance.
#### WavReward: Spoken Dialogue Models With Generalist Reward Evaluators
 - **Authors:** Shengpeng Ji, Tianle Liang, Yangzhuo Li, Jialong Zuo, Minghui Fang, Jinzheng He, Yifu Chen, Zhengqing Liu, Ziyue Jiang, Xize Cheng, Siqi Zheng, Jin Xu, Junyang Lin, Zhou Zhao
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Artificial Intelligence (cs.AI); Machine Learning (cs.LG); Multimedia (cs.MM); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2505.09558

 - **Pdf link:** https://arxiv.org/pdf/2505.09558

 - **Abstract**
 End-to-end spoken dialogue models such as GPT-4o-audio have recently garnered significant attention in the speech domain. However, the evaluation of spoken dialogue models' conversational performance has largely been overlooked. This is primarily due to the intelligent chatbots convey a wealth of non-textual information which cannot be easily measured using text-based language models like ChatGPT. To address this gap, we propose WavReward, a reward feedback model based on audio language models that can evaluate both the IQ and EQ of spoken dialogue systems with speech input. Specifically, 1) based on audio language models, WavReward incorporates the deep reasoning process and the nonlinear reward mechanism for post-training. By utilizing multi-sample feedback via the reinforcement learning algorithm, we construct a specialized evaluator tailored to spoken dialogue models. 2) We introduce ChatReward-30K, a preference dataset used to train WavReward. ChatReward-30K includes both comprehension and generation aspects of spoken dialogue models. These scenarios span various tasks, such as text-based chats, nine acoustic attributes of instruction chats, and implicit chats. WavReward outperforms previous state-of-the-art evaluation models across multiple spoken dialogue scenarios, achieving a substantial improvement about Qwen2.5-Omni in objective accuracy from 55.1$\%$ to 91.5$\%$. In subjective A/B testing, WavReward also leads by a margin of 83$\%$. Comprehensive ablation studies confirm the necessity of each component of WavReward. All data and code will be publicly at this https URL after the paper is accepted.
#### DPN-GAN: Inducing Periodic Activations in Generative Adversarial Networks for High-Fidelity Audio Synthesis
 - **Authors:** Zeeshan Ahmad, Shudi Bao, Meng Chen
 - **Subjects:** Subjects:
Sound (cs.SD); Artificial Intelligence (cs.AI); Computer Vision and Pattern Recognition (cs.CV); Machine Learning (cs.LG); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2505.09091

 - **Pdf link:** https://arxiv.org/pdf/2505.09091

 - **Abstract**
 In recent years, generative adversarial networks (GANs) have made significant progress in generating audio sequences. However, these models typically rely on bandwidth-limited mel-spectrograms, which constrain the resolution of generated audio sequences, and lead to mode collapse during conditional generation. To address this issue, we propose Deformable Periodic Network based GAN (DPN-GAN), a novel GAN architecture that incorporates a kernel-based periodic ReLU activation function to induce periodic bias in audio generation. This innovative approach enhances the model's ability to capture and reproduce intricate audio patterns. In particular, our proposed model features a DPN module for multi-resolution generation utilizing deformable convolution operations, allowing for adaptive receptive fields that improve the quality and fidelity of the synthetic audio. Additionally, we enhance the discriminator network using deformable convolution to better distinguish between real and generated samples, further refining the audio quality. We trained two versions of the model: DPN-GAN small (38.67M parameters) and DPN-GAN large (124M parameters). For evaluation, we use five different datasets, covering both speech synthesis and music generation tasks, to demonstrate the efficiency of the DPN-GAN. The experimental results demonstrate that DPN-GAN delivers superior performance on both out-of-distribution and noisy data, showcasing its robustness and adaptability. Trained across various datasets, DPN-GAN outperforms state-of-the-art GAN architectures on standard evaluation metrics, and exhibits increased robustness in synthesized audio.
#### SingNet: Towards a Large-Scale, Diverse, and In-the-Wild Singing Voice Dataset
 - **Authors:** Yicheng Gu, Chaoren Wang, Junan Zhang, Xueyao Zhang, Zihao Fang, Haorui He, Zhizheng Wu
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2505.09325

 - **Pdf link:** https://arxiv.org/pdf/2505.09325

 - **Abstract**
 The lack of a publicly-available large-scale and diverse dataset has long been a significant bottleneck for singing voice applications like Singing Voice Synthesis (SVS) and Singing Voice Conversion (SVC). To tackle this problem, we present SingNet, an extensive, diverse, and in-the-wild singing voice dataset. Specifically, we propose a data processing pipeline to extract ready-to-use training data from sample packs and songs on the internet, forming 3000 hours of singing voices in various languages and styles. Furthermore, to facilitate the use and demonstrate the effectiveness of SingNet, we pre-train and open-source various state-of-the-art (SOTA) models on Wav2vec2, BigVGAN, and NSF-HiFiGAN based on our collected singing voice data. We also conduct benchmark experiments on Automatic Lyric Transcription (ALT), Neural Vocoder, and Singing Voice Conversion (SVC). Audio demos are available at: this https URL.
#### The Voice Timbre Attribute Detection 2025 Challenge Evaluation Plan
 - **Authors:** Zhengyan Sheng, Jinghao He, Liping Chen, Kong Aik Lee, Zhen-Hua Ling
 - **Subjects:** Subjects:
Sound (cs.SD); Artificial Intelligence (cs.AI); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2505.09382

 - **Pdf link:** https://arxiv.org/pdf/2505.09382

 - **Abstract**
 Voice timbre refers to the unique quality or character of a person's voice that distinguishes it from others as perceived by human hearing. The Voice Timbre Attribute Detection (VtaD) 2025 challenge focuses on explaining the voice timbre attribute in a comparative manner. In this challenge, the human impression of voice timbre is verbalized with a set of sensory descriptors, including bright, coarse, soft, magnetic, and so on. The timbre is explained from the comparison between two voices in their intensity within a specific descriptor dimension. The VtaD 2025 challenge starts in May and culminates in a special proposal at the NCMMSC2025 conference in October 2025 in Zhenjiang, China.


by Zyzzyva0381 (Windy). 


2025-05-15
