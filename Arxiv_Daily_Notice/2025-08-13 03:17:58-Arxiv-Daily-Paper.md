# Showing new listings for Wednesday, 13 August 2025
Auto update papers at about 2:30am UTC (10:30am Beijing time) every weekday.


阅读 `Usage.md`了解如何使用此repo实现个性化的Arxiv论文推送

See `Usage.md` for instructions on how to personalize the repo. 


Keyword list: ['text-to-speech', 'text to speech', 'tts', 'LLM-based', 'speech', 'voice']


Excluded: []


### Today: 8papers 
#### Exploring Disentangled Neural Speech Codecs from Self-Supervised Representations
 - **Authors:** Ryo Aihara, Yoshiki Masuyama, Gordon Wichern, François G. Germain, Jonathan Le Roux
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Signal Processing (eess.SP)
 - **Arxiv link:** https://arxiv.org/abs/2508.08399

 - **Pdf link:** https://arxiv.org/pdf/2508.08399

 - **Abstract**
 Neural audio codecs (NACs), which use neural networks to generate compact audio representations, have garnered interest for their applicability to many downstream tasks -- especially quantized codecs due to their compatibility with large language models. However, unlike text, speech conveys not only linguistic content but also rich paralinguistic features. Encoding these elements in an entangled fashion may be suboptimal, as it limits flexibility. For instance, voice conversion (VC) aims to convert speaker characteristics while preserving the original linguistic content, which requires a disentangled representation. Inspired by VC methods utilizing $k$-means quantization with self-supervised features to disentangle phonetic information, we develop a discrete NAC capable of structured disentanglement. Experimental evaluations show that our approach achieves reconstruction performance on par with conventional NACs that do not explicitly perform disentanglement, while also matching the effectiveness of conventional VC techniques.
#### Joint decoding method for controllable contextual speech recognition based on Speech LLM
 - **Authors:** Yangui Fang, Jing Peng, Yu Xi, Xu Li, Haoyu Li, Chengwei Zhang, Guohui Zhong, Kai Yu
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2508.08585

 - **Pdf link:** https://arxiv.org/pdf/2508.08585

 - **Abstract**
 Contextual speech recognition refers to the ability to identify preferences for specific content based on contextual information. Recently, leveraging the contextual understanding capabilities of Speech LLM to achieve contextual biasing by injecting contextual information through prompts have emerged as a research this http URL, the direct information injection method via prompts relies on the internal attention mechanism of the model, making it impossible to explicitly control the extent of information injection. To address this limitation, we propose a joint decoding method to control the contextual information. This approach enables explicit control over the injected contextual information and achieving superior recognition performance. Additionally, Our method can also be used for sensitive word suppression this http URL, experimental results show that even Speech LLM not pre-trained on long contextual data can acquire long contextual capabilities through our method.
#### MultiAiTutor: Child-Friendly Educational Multilingual Speech Generation Tutor with LLMs
 - **Authors:** Xiaoxue Gao, Huayun Zhang, Nancy F. Chen
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Artificial Intelligence (cs.AI); Computation and Language (cs.CL); Signal Processing (eess.SP)
 - **Arxiv link:** https://arxiv.org/abs/2508.08715

 - **Pdf link:** https://arxiv.org/pdf/2508.08715

 - **Abstract**
 Generative speech models have demonstrated significant potential in personalizing teacher-student interactions, offering valuable real-world applications for language learning in children's education. However, achieving high-quality, child-friendly speech generation remains challenging, particularly for low-resource languages across diverse languages and cultural contexts. In this paper, we propose MultiAiTutor, an educational multilingual generative AI tutor with child-friendly designs, leveraging LLM architecture for speech generation tailored for educational purposes. We propose to integrate age-appropriate multilingual speech generation using LLM architectures, facilitating young children's language learning through culturally relevant image-description tasks in three low-resource languages: Singaporean-accent Mandarin, Malay, and Tamil. Experimental results from both objective metrics and subjective evaluations demonstrate the superior performance of the proposed MultiAiTutor compared to baseline methods.
#### Transient Noise Removal via Diffusion-based Speech Inpainting
 - **Authors:** Mordehay Moradi, Sharon Gannot
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2508.08890

 - **Pdf link:** https://arxiv.org/pdf/2508.08890

 - **Abstract**
 In this paper, we present PGDI, a diffusion-based speech inpainting framework for restoring missing or severely corrupted speech segments. Unlike previous methods that struggle with speaker variability or long gap lengths, PGDI can accurately reconstruct gaps of up to one second in length while preserving speaker identity, prosody, and environmental factors such as reverberation. Central to this approach is classifier guidance, specifically phoneme-level guidance, which substantially improves reconstruction fidelity. PGDI operates in a speaker-independent manner and maintains robustness even when long segments are completely masked by strong transient noise, making it well-suited for real-world applications, such as fireworks, door slams, hammer strikes, and construction noise. Through extensive experiments across diverse speakers and gap lengths, we demonstrate PGDI's superior inpainting performance and its ability to handle challenging acoustic conditions. We consider both scenarios, with and without access to the transcript during inference, showing that while the availability of text further enhances performance, the model remains effective even in its absence. For audio samples, visit: this https URL
#### DeCRED: Decoder-Centric Regularization for Encoder-Decoder Based Speech Recognition
 - **Authors:** Alexander Polok, Santosh Kesiraju, Karel Beneš, Bolaji Yusuf, Lukáš Burget, Jan Černocký
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2508.08938

 - **Pdf link:** https://arxiv.org/pdf/2508.08938

 - **Abstract**
 This paper presents a simple yet effective regularization for the internal language model induced by the decoder in encoder-decoder ASR models, thereby improving robustness and generalization in both in- and out-of-domain settings. The proposed method, Decoder-Centric Regularization in Encoder-Decoder (DeCRED), adds auxiliary classifiers to the decoder, enabling next token prediction via intermediate logits. Empirically, DeCRED reduces the mean internal LM BPE perplexity by 36.6% relative to 11 test sets. Furthermore, this translates into actual WER improvements over the baseline in 5 of 7 in-domain and 3 of 4 out-of-domain test sets, reducing macro WER from 6.4% to 6.3% and 18.2% to 16.2%, respectively. On TEDLIUM3, DeCRED achieves 7.0% WER, surpassing the baseline and encoder-centric InterCTC regularization by 0.6% and 0.5%, respectively. Finally, we compare DeCRED with OWSM v3.1 and Whisper-medium, showing competitive WERs despite training on much less data with fewer parameters.
#### Listen through the Sound: Generative Speech Restoration Leveraging Acoustic Context Representation
 - **Authors:** Soo-Whan Chung, Min-Seok Choi
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2508.08953

 - **Pdf link:** https://arxiv.org/pdf/2508.08953

 - **Abstract**
 This paper introduces a novel approach to speech restoration by integrating a context-related conditioning strategy. Specifically, we employ the diffusion-based generative restoration model, UNIVERSE++, as a backbone to evaluate the effectiveness of contextual representations. We incorporate acoustic context embeddings extracted from the CLAP model, which capture the environmental attributes of input audio. Additionally, we propose an Acoustic Context (ACX) representation that refines CLAP embeddings to better handle various distortion factors and their intensity in speech signals. Unlike content-based approaches that rely on linguistic and speaker attributes, ACX provides contextual information that enables the restoration model to distinguish and mitigate distortions better. Experimental results indicate that context-aware conditioning improves both restoration performance and its stability across diverse distortion conditions, reducing variability compared to content-based methods.
#### Selection of Layers from Self-supervised Learning Models for Predicting Mean-Opinion-Score of Speech
 - **Authors:** Xinyu Liang, Fredrik Cumlin, Victor Ungureanu, Chandan K. A. Reddy, Christian Schuldt, Saikat Chatterjee
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2508.08962

 - **Pdf link:** https://arxiv.org/pdf/2508.08962

 - **Abstract**
 Self-supervised learning (SSL) models like Wav2Vec2, HuBERT, and WavLM have been widely used in speech processing. These transformer-based models consist of multiple layers, each capturing different levels of representation. While prior studies explored their layer-wise representations for efficiency and performance, speech quality assessment (SQA) models predominantly rely on last-layer features, leaving intermediate layers underexamined. In this work, we systematically evaluate different layers of multiple SSL models for predicting mean-opinion-score (MOS). Features from each layer are fed into a lightweight regression network to assess effectiveness. Our experiments consistently show early-layers features outperform or match those from the last layer, leading to significant improvements over conventional approaches and state-of-the-art MOS prediction models. These findings highlight the advantages of early-layer selection, offering enhanced performance and reduced system complexity.
#### DualSpeechLM: Towards Unified Speech Understanding and Generation via Dual Speech Token Modeling with Large Language Models
 - **Authors:** Yuanyuan Wang, Dongchao Yang, Yiwen Shao, Hangting Chen, Jiankun Zhao, Zhiyong Wu, Helen Meng, Xixin Wu
 - **Subjects:** Subjects:
Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2508.08961

 - **Pdf link:** https://arxiv.org/pdf/2508.08961

 - **Abstract**
 Extending pre-trained Large Language Models (LLMs)'s speech understanding or generation abilities by introducing various effective speech tokens has attracted great attention in the speech community. However, building a unified speech understanding and generation model still faces the following challenges: (1) Due to the huge modality gap between speech tokens and text tokens, extending text LLMs to unified speech LLMs relies on large-scale paired data for fine-tuning, and (2) Generation and understanding tasks prefer information at different levels, e.g., generation benefits from detailed acoustic features, while understanding favors high-level semantics. This divergence leads to difficult performance optimization in one unified model. To solve these challenges, in this paper, we present two key insights in speech tokenization and speech language modeling. Specifically, we first propose an Understanding-driven Speech Tokenizer (USTokenizer), which extracts high-level semantic information essential for accomplishing understanding tasks using text LLMs. In this way, USToken enjoys better modality commonality with text, which reduces the difficulty of modality alignment in adapting text LLMs to speech LLMs. Secondly, we present DualSpeechLM, a dual-token modeling framework that concurrently models USToken as input and acoustic token as output within a unified, end-to-end framework, seamlessly integrating speech understanding and generation capabilities. Furthermore, we propose a novel semantic supervision loss and a Chain-of-Condition (CoC) strategy to stabilize model training and enhance speech generation performance. Experimental results demonstrate that our proposed approach effectively fosters a complementary relationship between understanding and generation tasks, highlighting the promising strategy of mutually enhancing both tasks in one unified model.


by Zyzzyva0381 (Windy). 


2025-08-13
