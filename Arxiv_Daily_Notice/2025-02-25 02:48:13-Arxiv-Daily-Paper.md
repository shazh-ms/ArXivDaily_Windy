# Showing new listings for Monday, 24 February 2025
Auto update papers at about 2:30am UTC (10:30am Beijing time) every weekday.


阅读 `Usage.md`了解如何使用此repo实现个性化的Arxiv论文推送

See `Usage.md` for instructions on how to personalize the repo. 


Keyword list: ['acoustic', 'reinforcement learning', 'reverb', 'meta', 'separate', 'reconstruction', 'noise', 'enhance', 'localization', 'speech']


Excluded: []


### Today: 7papers 
#### Enhancing Speech Large Language Models with Prompt-Aware Mixture of Audio Encoders
 - **Authors:** Weiqiao Shan, Yuang Li, Yuhao Zhang, Yingfeng Luo, Chen Xu, Xiaofeng Zhao, Long Meng, Yunfei Lu, Min Zhang, Hao Yang, Tong Xiao, Jingbo Zhu
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2502.15178

 - **Pdf link:** https://arxiv.org/pdf/2502.15178

 - **Abstract**
 Connecting audio encoders with large language models (LLMs) allows the LLM to perform various audio understanding tasks, such as automatic speech recognition (ASR) and audio captioning (AC). Most research focuses on training an adapter layer to generate a unified audio feature for the LLM. However, different tasks may require distinct features that emphasize either semantic or acoustic aspects, making task-specific audio features more desirable. In this paper, we propose Prompt-aware Mixture (PaM) to enhance the Speech LLM that uses multiple audio encoders. Our approach involves using different experts to extract different features based on the prompt that indicates different tasks. Experiments demonstrate that with PaM, only one Speech LLM surpasses the best performances achieved by all single-encoder Speech LLMs on ASR, Speaker Number Verification, and AC tasks. PaM also outperforms other feature fusion baselines, such as concatenation and averaging.
#### Fundamental Survey on Neuromorphic Based Audio Classification
 - **Authors:** Amlan Basu, Pranav Chaudhari, Gaetano Di Caterina
 - **Subjects:** Subjects:
Sound (cs.SD); Artificial Intelligence (cs.AI); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2502.15056

 - **Pdf link:** https://arxiv.org/pdf/2502.15056

 - **Abstract**
 Audio classification is paramount in a variety of applications including surveillance, healthcare monitoring, and environmental analysis. Traditional methods frequently depend on intricate signal processing algorithms and manually crafted features, which may fall short in fully capturing the complexities of audio patterns. Neuromorphic computing, inspired by the architecture and functioning of the human brain, presents a promising alternative for audio classification tasks. This survey provides an exhaustive examination of the current state-of-the-art in neuromorphic-based audio classification. It delves into the crucial components of neuromorphic systems, such as Spiking Neural Networks (SNNs), memristors, and neuromorphic hardware platforms, highlighting their advantages in audio classification. Furthermore, the survey explores various methodologies and strategies employed in neuromorphic audio classification, including event-based processing, spike-based learning, and bio-inspired feature extraction. It examines how these approaches address the limitations of traditional audio classification methods, particularly in terms of energy efficiency, real-time processing, and robustness to environmental noise. Additionally, the paper conducts a comparative analysis of different neuromorphic audio classification models and benchmarks, evaluating their performance metrics, computational efficiency, and scalability. By providing a comprehensive guide for researchers, engineers and practitioners, this survey aims to stimulate further innovation and advancements in the evolving field of neuromorphic audio classification.
#### Improving Streaming Speech Recognition With Time-Shifted Contextual Attention And Dynamic Right Context Masking
 - **Authors:** Khanh Le, Duc Chau
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2502.15158

 - **Pdf link:** https://arxiv.org/pdf/2502.15158

 - **Abstract**
 Chunk-based inference stands out as a popular approach in developing real-time streaming speech recognition, valued for its simplicity and efficiency. However, because it restricts the model's focus to only the history and current chunk context, it may result in performance degradation in scenarios that demand consideration of future context. Addressing this, we propose a novel approach featuring Time-Shifted Contextual Attention (TSCA) and Dynamic Right Context (DRC) masking. Our method shows a relative word error rate reduction of 10 to 13.9% on the Librispeech dataset with the inclusion of in-context future information provided by TSCA. Moreover, we present a streaming automatic speech recognition pipeline that facilitates the integration of TSCA with minimal user-perceived latency, while also enabling batch processing capability, making it practical for various applications.
#### ESPnet-SpeechLM: An Open Speech Language Model Toolkit
 - **Authors:** Jinchuan Tian, Jiatong Shi, William Chen, Siddhant Arora, Yoshiki Masuyama, Takashi Maekaku, Yihan Wu, Junyi Peng, Shikhar Bharadwaj, Yiwen Zhao, Samuele Cornell, Yifan Peng, Xiang Yue, Chao-Han Huck Yang, Graham Neubig, Shinji Watanabe
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2502.15218

 - **Pdf link:** https://arxiv.org/pdf/2502.15218

 - **Abstract**
 We present ESPnet-SpeechLM, an open toolkit designed to democratize the development of speech language models (SpeechLMs) and voice-driven agentic applications. The toolkit standardizes speech processing tasks by framing them as universal sequential modeling problems, encompassing a cohesive workflow of data preprocessing, pre-training, inference, and task evaluation. With ESPnet-SpeechLM, users can easily define task templates and configure key settings, enabling seamless and streamlined SpeechLM development. The toolkit ensures flexibility, efficiency, and scalability by offering highly configurable modules for every stage of the workflow. To illustrate its capabilities, we provide multiple use cases demonstrating how competitive SpeechLMs can be constructed with ESPnet-SpeechLM, including a 1.7B-parameter model pre-trained on both text and speech tasks, across diverse benchmarks. The toolkit and its recipes are fully transparent and reproducible at: this https URL.
#### Retrieval-Augmented Speech Recognition Approach for Domain Challenges
 - **Authors:** Peng Shen, Xugang Lu, Hisashi Kawai
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2502.15264

 - **Pdf link:** https://arxiv.org/pdf/2502.15264

 - **Abstract**
 Speech recognition systems often face challenges due to domain mismatch, particularly in real-world applications where domain-specific data is unavailable because of data accessibility and confidentiality constraints. Inspired by Retrieval-Augmented Generation (RAG) techniques for large language models (LLMs), this paper introduces a LLM-based retrieval-augmented speech recognition method that incorporates domain-specific textual data at the inference stage to enhance recognition performance. Rather than relying on domain-specific textual data during the training phase, our model is trained to learn how to utilize textual information provided in prompts for LLM decoder to improve speech recognition performance. Benefiting from the advantages of the RAG retrieval mechanism, our approach efficiently accesses locally available domain-specific documents, ensuring a convenient and effective process for solving domain mismatch problems. Experiments conducted on the CSJ database demonstrate that the proposed method significantly improves speech recognition accuracy and achieves state-of-the-art results on the CSJ dataset, even without relying on the full training data.
#### Advancing User-Voice Interaction: Exploring Emotion-Aware Voice Assistants Through a Role-Swapping Approach
 - **Authors:** Yong Ma, Yuchong Zhang, Di Fu, Stephanie Zubicueta Portales, Danica Kragic, Morten Fjeld
 - **Subjects:** Subjects:
Human-Computer Interaction (cs.HC); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2502.15367

 - **Pdf link:** https://arxiv.org/pdf/2502.15367

 - **Abstract**
 As voice assistants (VAs) become increasingly integrated into daily life, the need for emotion-aware systems that can recognize and respond appropriately to user emotions has grown. While significant progress has been made in speech emotion recognition (SER) and sentiment analysis, effectively addressing user emotions-particularly negative ones-remains a challenge. This study explores human emotional response strategies in VA interactions using a role-swapping approach, where participants regulate AI emotions rather than receiving pre-programmed responses. Through speech feature analysis and natural language processing (NLP), we examined acoustic and linguistic patterns across various emotional scenarios. Results show that participants favor neutral or positive emotional responses when engaging with negative emotional cues, highlighting a natural tendency toward emotional regulation and de-escalation. Key acoustic indicators such as root mean square (RMS), zero-crossing rate (ZCR), and jitter were identified as sensitive to emotional states, while sentiment polarity and lexical diversity (TTR) distinguished between positive and negative responses. These findings provide valuable insights for developing adaptive, context-aware VAs capable of delivering empathetic, culturally sensitive, and user-aligned responses. By understanding how humans naturally regulate emotions in AI interactions, this research contributes to the design of more intuitive and emotionally intelligent voice assistants, enhancing user trust and engagement in human-AI interactions.
#### Audio signal interpolation using optimal transportation of spectrograms
 - **Authors:** David Valdivia, Marien Renaud, Elsa Cazelles, Cédric Févotte
 - **Subjects:** Subjects:
Signal Processing (eess.SP); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2502.15430

 - **Pdf link:** https://arxiv.org/pdf/2502.15430

 - **Abstract**
 We present a novel approach for generating an artificial audio signal that interpolates between given source and target sounds. Our approach relies on the computation of Wasserstein barycenters of the source and target spectrograms, followed by phase reconstruction and inversion. In contrast with previous works, our new method considers the spectrograms globally and does not operate on a temporal frame-to-frame basis. An other contribution is to endow the transportation cost matrix with a specific structure that prohibits remote displacements of energy along the time axis, and for which optimal transport is made possible by leveraging the unbalanced transport framework. The proposed cost matrix makes sense from the audio perspective and also allows to reduce the computation load. Results with synthetic musical notes and real environmental sounds illustrate the potential of our novel approach.


by Zyzzyva0381 (Windy). 


2025-02-25
