# Showing new listings for Thursday, 28 August 2025
Auto update papers at about 2:30am UTC (10:30am Beijing time) every weekday.


阅读 `Usage.md`了解如何使用此repo实现个性化的Arxiv论文推送

See `Usage.md` for instructions on how to personalize the repo. 


Keyword list: ['text-to-speech', 'text to speech', 'tts', 'LLM-based', 'speech', 'voice']


Excluded: []


### Today: 6papers 
#### Audio-Visual Feature Synchronization for Robust Speech Enhancement in Hearing Aids
 - **Authors:** Nasir Saleem, Mandar Gogate, Kia Dashtipour, Adeel Hussain, Usman Anwar, Adewale Adetomi, Tughrul Arslan, Amir Hussain
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2508.19483

 - **Pdf link:** https://arxiv.org/pdf/2508.19483

 - **Abstract**
 Audio-visual feature synchronization for real-time speech enhancement in hearing aids represents a progressive approach to improving speech intelligibility and user experience, particularly in strong noisy backgrounds. This approach integrates auditory signals with visual cues, utilizing the complementary description of these modalities to improve speech intelligibility. Audio-visual feature synchronization for real-time SE in hearing aids can be further optimized using an efficient feature alignment module. In this study, a lightweight cross-attentional model learns robust audio-visual representations by exploiting large-scale data and simple architecture. By incorporating the lightweight cross-attentional model in an AVSE framework, the neural system dynamically emphasizes critical features across audio and visual modalities, enabling defined synchronization and improved speech intelligibility. The proposed AVSE model not only ensures high performance in noise suppression and feature alignment but also achieves real-time processing with minimal latency (36ms) and energy consumption. Evaluations on the AVSEC3 dataset show the efficiency of the model, achieving significant gains over baselines in perceptual quality (PESQ:0.52), intelligibility (STOI:19\%), and fidelity (SI-SDR:10.10dB).
#### FLASepformer: Efficient Speech Separation with Gated Focused Linear Attention Transformer
 - **Authors:** Haoxu Wang, Yiheng Jiang, Gang Qiao, Pengteng Shi, Biao Tian
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2508.19528

 - **Pdf link:** https://arxiv.org/pdf/2508.19528

 - **Abstract**
 Speech separation always faces the challenge of handling prolonged time sequences. Past methods try to reduce sequence lengths and use the Transformer to capture global information. However, due to the quadratic time complexity of the attention module, memory usage and inference time still increase significantly with longer segments. To tackle this, we introduce Focused Linear Attention and build FLASepformer with linear complexity for efficient speech separation. Inspired by SepReformer and TF-Locoformer, we have two variants: FLA-SepReformer and FLA-TFLocoformer. We also add a new Gated module to improve performance further. Experimental results on various datasets show that FLASepformer matches state-of-the-art performance with less memory consumption and faster inference. FLA-SepReformer-T/B/L increases speed by 2.29x, 1.91x, and 1.49x, with 15.8%, 20.9%, and 31.9% GPU memory usage, proving our model's effectiveness.
#### Lightweight speech enhancement guided target speech extraction in noisy multi-speaker scenarios
 - **Authors:** Ziling Huang, Junnan Wu, Lichun Fan, Zhenbo Luo, Jian Luan, Haixin Guan, Yanhua Long
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2508.19583

 - **Pdf link:** https://arxiv.org/pdf/2508.19583

 - **Abstract**
 Target speech extraction (TSE) has achieved strong performance in relatively simple conditions such as one-speaker-plus-noise and two-speaker mixtures, but its performance remains unsatisfactory in noisy multi-speaker scenarios. To address this issue, we introduce a lightweight speech enhancement model, GTCRN, to better guide TSE in noisy environments. Building on our competitive previous speaker embedding/encoder-free framework SEF-PNet, we propose two extensions: LGTSE and D-LGTSE. LGTSE incorporates noise-agnostic enrollment guidance by denoising the input noisy speech before context interaction with enrollment speech, thereby reducing noise interference. D-LGTSE further improves system robustness against speech distortion by leveraging denoised speech as an additional noisy input during training, expanding the dynamic range of noisy conditions and enabling the model to directly learn from distorted signals. Furthermore, we propose a two-stage training strategy, first with GTCRN enhancement-guided pre-training and then joint fine-tuning, to fully exploit model this http URL on the Libri2Mix dataset demonstrate significant improvements of 0.89 dB in SISDR, 0.16 in PESQ, and 1.97% in STOI, validating the effectiveness of our approach. Our code is publicly available at this https URL.
#### Hybrid Decoding: Rapid Pass and Selective Detailed Correction for Sequence Models
 - **Authors:** Yunkyu Lim, Jihwan Park, Hyung Yong Kim, Hanbin Lee, Byeong-Yeol Kim
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2508.19671

 - **Pdf link:** https://arxiv.org/pdf/2508.19671

 - **Abstract**
 Recently, Transformer-based encoder-decoder models have demonstrated strong performance in multilingual speech recognition. However, the decoder's autoregressive nature and large size introduce significant bottlenecks during inference. Additionally, although rare, repetition can occur and negatively affect recognition accuracy. To tackle these challenges, we propose a novel Hybrid Decoding approach that both accelerates inference and alleviates the issue of repetition. Our method extends the transformer encoder-decoder architecture by attaching a lightweight, fast decoder to the pretrained encoder. During inference, the fast decoder rapidly generates an output, which is then verified and, if necessary, selectively corrected by the Transformer decoder. This results in faster decoding and improved robustness against repetitive errors. Experiments on the LibriSpeech and GigaSpeech test sets indicate that, with fine-tuning limited to the added decoder, our method achieves word error rates comparable to or better than the baseline, while more than doubling the inference speed.
#### CAVEMOVE: An Acoustic Database for the Study of Voice-enabled Technologies inside Moving Vehicles
 - **Authors:** Nikolaos Stefanakis, Marinos Kalaitzakis, Andreas Symiakakis, Stefanos Papadakis, Despoina Pavlidi
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2508.19691

 - **Pdf link:** https://arxiv.org/pdf/2508.19691

 - **Abstract**
 In this paper, we present an acoustic database, designed to drive and support research on voiced enabled technologies inside moving vehicles. The recording process involves (i) recordings of acoustic impulse responses, acquired under static conditions to provide the means for modeling the speech and car-audio components (ii) recordings of acoustic noise at a wide range of static and in-motion conditions. Data are recorded with two different microphone configurations, particularly (i) a compact microphone array and (ii) a distributed microphone setup. We briefly describe the conditions under which the recordings were acquired, and we provide insight into a Python API that we designed to support the research and development of voice-enabled technologies inside moving vehicles. The first version of this Python API and part of the described dataset are available for free download.
#### CAMÕES: A Comprehensive Automatic Speech Recognition Benchmark for European Portuguese
 - **Authors:** Carlos Carvalho, Francisco Teixeira, Catarina Botelho, Anna Pompili, Rubén Solera-Ureña, Sérgio Paulo, Mariana Julião, Thomas Rolland, John Mendonça, Diogo Pereira, Isabel Trancoso, Alberto Abad
 - **Subjects:** Subjects:
Computation and Language (cs.CL); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2508.19721

 - **Pdf link:** https://arxiv.org/pdf/2508.19721

 - **Abstract**
 Existing resources for Automatic Speech Recognition in Portuguese are mostly focused on Brazilian Portuguese, leaving European Portuguese (EP) and other varieties under-explored. To bridge this gap, we introduce CAMÕES, the first open framework for EP and other Portuguese varieties. It consists of (1) a comprehensive evaluation benchmark, including 46h of EP test data spanning multiple domains; and (2) a collection of state-of-the-art models. For the latter, we consider multiple foundation models, evaluating their zero-shot and fine-tuned performances, as well as E-Branchformer models trained from scratch. A curated set of 425h of EP was used for both fine-tuning and training. Our results show comparable performance for EP between fine-tuned foundation models and the E-Branchformer. Furthermore, the best-performing models achieve relative improvements above 35% WER, compared to the strongest zero-shot foundation model, establishing a new state-of-the-art for EP and other varieties.


by Zyzzyva0381 (Windy). 


2025-08-28
