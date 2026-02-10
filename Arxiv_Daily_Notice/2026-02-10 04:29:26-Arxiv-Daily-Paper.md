# Showing new listings for Tuesday, 10 February 2026
Auto update papers at about 2:30am UTC (10:30am Beijing time) every weekday.


阅读 `Usage.md`了解如何使用此repo实现个性化的Arxiv论文推送

See `Usage.md` for instructions on how to personalize the repo. 


Keyword list: ['text-to-speech', 'text to speech', 'tts', 'LLM-based', 'speech', 'voice']


Excluded: []


### Today: 5papers 
#### SoulX-Singer: Towards High-Quality Zero-Shot Singing Voice Synthesis
 - **Authors:** Jiale Qian, Hao Meng, Tian Zheng, Pengcheng Zhu, Haopeng Lin, Yuhang Dai, Hanke Xie, Wenxiao Cao, Ruixuan Shang, Jun Wu, Hongmei Liu, Hanlin Wen, Jian Zhao, Zhonglin Jiang, Yong Chen, Shunshun Yin, Ming Tao, Jianguo Wei, Lei Xie, Xinsheng Wang
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Artificial Intelligence (cs.AI); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2602.07803

 - **Pdf link:** https://arxiv.org/pdf/2602.07803

 - **Abstract**
 While recent years have witnessed rapid progress in speech synthesis, open-source singing voice synthesis (SVS) systems still face significant barriers to industrial deployment, particularly in terms of robustness and zero-shot generalization. In this report, we introduce SoulX-Singer, a high-quality open-source SVS system designed with practical deployment considerations in mind. SoulX-Singer supports controllable singing generation conditioned on either symbolic musical scores (MIDI) or melodic representations, enabling flexible and expressive control in real-world production workflows. Trained on more than 42,000 hours of vocal data, the system supports Mandarin Chinese, English, and Cantonese and consistently achieves state-of-the-art synthesis quality across languages under diverse musical conditions. Furthermore, to enable reliable evaluation of zero-shot SVS performance in practical scenarios, we construct SoulX-Singer-Eval, a dedicated benchmark with strict training-test disentanglement, facilitating systematic assessment in zero-shot settings.
#### Detect, Attend and Extract: Keyword Guided Target Speaker Extraction
 - **Authors:** Haoyu Li, Yu Xi, Yidi Jiang, Shuai Wang, Kate Knill, Mark Gales, Haizhou Li, Kai Yu
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2602.07977

 - **Pdf link:** https://arxiv.org/pdf/2602.07977

 - **Abstract**
 Target speaker extraction (TSE) aims to extract the speech of a target speaker from mixtures containing multiple competing speakers. Conventional TSE systems predominantly rely on speaker cues, such as pre-enrolled speech, to identify and isolate the target speaker. However, in many practical scenarios, clean enrollment utterances are unavailable, limiting the applicability of existing approaches. In this work, we propose DAE-TSE, a keyword-guided TSE framework that specifies the target speaker through distinct keywords they utter. By leveraging keywords (i.e., partial transcriptions) as cues, our approach provides a flexible and practical alternative to enrollment-based TSE. DAE-TSE follows the Detect-Attend-Extract (DAE) paradigm: it first detects the presence of the given keywords, then attends to the corresponding speaker based on the keyword content, and finally extracts the target speech. Experimental results demonstrate that DAE-TSE outperforms standard TSE systems that rely on clean enrollment speech. To the best of our knowledge, this is the first study to utilize partial transcription as a cue for specifying the target speaker in TSE, offering a flexible and practical solution for real-world scenarios. Our code and demo page are now publicly available.
#### Cross-Modal Bottleneck Fusion For Noise Robust Audio-Visual Speech Recognition
 - **Authors:** Seaone Ok, Min Jun Choi, Eungbeom Kim, Seungu Han, Kyogu Lee
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2602.08293

 - **Pdf link:** https://arxiv.org/pdf/2602.08293

 - **Abstract**
 Audio-Visual Speech Recognition (AVSR) leverages both acoustic and visual cues to improve speech recognition under noisy conditions. A central question is how to design a fusion mechanism that allows the model to effectively exploit visual information when the audio signal is degraded, while maintaining strong performance on clean speech. We propose CoBRA (Cross-modal Bottleneck for Robust AVSR), a bottleneck-based fusion framework that introduces a compact set of learnable tokens to mediate cross-modal exchange. By regulating information flow through these tokens, the audio stream can reliably access essential visual cues even under adverse or out-of-domain noise. Despite limited training data, our model surpasses comparable baselines and remains competitive with large-scale systems through noise-adaptive fusion, demonstrating both efficiency and robustness. Ablation studies highlight that the depth of fusion is the most critical factor, underscoring its importance in designing robust AVSR systems.
#### MENASpeechBank: A Reference Voice Bank with Persona-Conditioned Multi-Turn Conversations for AudioLLMs
 - **Authors:** Zien Sheikh Ali, Hunzalah Hassan Bhatti, Rabindra Nath Nandi, Shammur Absar Chowdhury, Firoj Alam
 - **Subjects:** Subjects:
Sound (cs.SD); Artificial Intelligence (cs.AI); Computation and Language (cs.CL); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2602.07036

 - **Pdf link:** https://arxiv.org/pdf/2602.07036

 - **Abstract**
 Audio large language models (AudioLLMs) enable instruction-following over speech and general audio, but progress is increasingly limited by the lack of diverse, conversational, instruction-aligned speech-text data. This bottleneck is especially acute for persona-grounded interactions and dialectal coverage, where collecting and releasing real multi-speaker recordings is costly and slow. We introduce MENASpeechBank, a reference speech bank comprising about 18K high-quality utterances from 124 speakers spanning multiple MENA countries, covering English, Modern Standard Arabic (MSA), and regional Arabic varieties. Building on this resource, we develop a controllable synthetic data pipeline that: (i) constructs persona profiles enriched with World Values Survey-inspired attributes, (ii) defines a taxonomy of about 5K conversational scenarios, (iii) matches personas to scenarios via semantic similarity, (iv) generates about 417K role-play conversations with an LLM where the user speaks as the persona and the assistant behaves as a helpful agent, and (v) synthesizes the user turns by conditioning on reference speaker audio to preserve speaker identity and diversity. We evaluate both synthetic and human-recorded conversations and provide detailed analysis. We will release MENASpeechBank and the generated conversations publicly for the community.
#### Rho-Perfect: Correlation Ceiling For Subjective Evaluation Datasets
 - **Authors:** Fredrik Cumlin
 - **Subjects:** Subjects:
Machine Learning (cs.LG); Audio and Speech Processing (eess.AS); Machine Learning (stat.ML)
 - **Arxiv link:** https://arxiv.org/abs/2602.08552

 - **Pdf link:** https://arxiv.org/pdf/2602.08552

 - **Abstract**
 Subjective ratings contain inherent noise that limits the model-human correlation, but this reliability issue is rarely quantified. In this paper, we present $\rho$-Perfect, a practical estimation of the highest achievable correlation of a model on subjectively rated datasets. We define $\rho$-Perfect to be the correlation between a perfect predictor and human ratings, and derive an estimate of the value based on heteroscedastic noise scenarios, a common occurrence in subjectively rated datasets. We show that $\rho$-Perfect squared estimates test-retest correlation and use this to validate the estimate. We demonstrate the use of $\rho$-Perfect on a speech quality dataset and show how the measure can distinguish between model limitations and data quality issues.


by Zyzzyva0381 (Windy). 


2026-02-10
