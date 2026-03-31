# Showing new listings for Tuesday, 31 March 2026
Auto update papers at about 2:30am UTC (10:30am Beijing time) every weekday.


阅读 `Usage.md`了解如何使用此repo实现个性化的Arxiv论文推送

See `Usage.md` for instructions on how to personalize the repo. 


Keyword list: ['text-to-speech', 'text to speech', 'tts', 'LLM-based', 'speech', 'voice']


Excluded: []


### Today: 5papers 
#### HASS: Hierarchical Simulation of Logopenic Aphasic Speech for Scalable PPA Detection
 - **Authors:** Harrison Li, Kevin Wang, Cheol Jun Cho, Jiachen Lian, Rabab Rangwala, Chenxu Guo, Emma Yang, Lynn Kurteff, Zoe Ezzes, Willa Keegan-Rodewald, Jet Vonk, Siddarth Ramkrishnan, Giada Antonicelli, Zachary Miller, Marilu Gorno Tempini, Gopala Anumanchipalli
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Artificial Intelligence (cs.AI); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2603.26795

 - **Pdf link:** https://arxiv.org/pdf/2603.26795

 - **Abstract**
 Building a diagnosis model for primary progressive aphasia (PPA) has been challenging due to the data scarcity. Collecting clinical data at scale is limited by the high vulnerability of clinical population and the high cost of expert labeling. To circumvent this, previous studies simulate dysfluent speech to generate training data. However, those approaches are not comprehensive enough to simulate PPA as holistic, multi-level phenotypes, instead relying on isolated dysfluencies. To address this, we propose a novel, clinically grounded simulation framework, Hierarchical Aphasic Speech Simulation (HASS). HASS aims to simulate behaviors of logopenic variant of PPA (lvPPA) with varying degrees of severity. To this end, semantic, phonological, and temporal deficits of lvPPA are systematically identified by clinical experts, and simulated. We demonstrate that our framework enables more accurate and generalizable detection models.
#### PHONOS: PHOnetic Neutralization for Online Streaming Applications
 - **Authors:** Waris Quamer, Mu-Ruei Tseng, Ghady Nasrallah, Ricardo Gutierrez-Osuna
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Computation and Language (cs.CL); Machine Learning (cs.LG)
 - **Arxiv link:** https://arxiv.org/abs/2603.27001

 - **Pdf link:** https://arxiv.org/pdf/2603.27001

 - **Abstract**
 Speaker anonymization (SA) systems modify timbre while leaving regional or non-native accents intact, which is problematic because accents can narrow the anonymity set. To address this issue, we present PHONOS, a streaming module for real-time SA that neutralizes non-native accent to sound native-like. Our approach pre-generates golden speaker utterances that preserve source timbre and rhythm but replace foreign segmentals with native ones using silence-aware DTW alignment and zero-shot voice conversion. These utterances supervise a causal accent translator that maps non-native content tokens to native equivalents with at most 40ms look-ahead, trained using joint cross-entropy and CTC losses. Our evaluations show an 81% reduction in non-native accent confidence, with listening-test ratings consistent with this shift, and reduced speaker linkability as accent-neutralized utterances move away from the original speaker in embedding space while having latency under 241 ms on single GPU.
#### VAANI: Capturing the language landscape for an inclusive digital India
 - **Authors:** Sujith Pulikodan, Abhayjeet Singh, Agneedh Basu, Lokesh Rady, Nihar Desai, Pavan Kumar J, Prajjwal Srivastav, Pranav D Bhat, Raghu Dharmaraju, Ritika Gupta, Sathvik Udupa, Saurabh Kumar, Sumit Sharma, Vaibhav Vishwakarma, Visruth Sanka, Dinesh Tewari, Harsh Dhand, Amrita Kamat, Sukhwinder Singh, Shikhar Vashishth, Partha Talukdar, Raj Acharya, Prasanta Kumar Ghosh
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2603.28714

 - **Pdf link:** https://arxiv.org/pdf/2603.28714

 - **Abstract**
 Project VAANI is an initiative to create an India-representative multi-modal dataset that comprehensively maps India's linguistic diversity, starting with 165 districts across the country in its first two phases. Speech data is collected through a carefully structured process that uses image-based prompts to encourage spontaneous responses. Images are captured through a separate process that encompasses a broad range of topics, gathered from both within and across districts. The collected data undergoes a rigorous multi-stage quality evaluation, including both automated and manual checks to ensure highest possible standards in audio quality and transcription accuracy. Following this thorough validation, we have open-sourced around 289K images, approximately 31,270 hours of audio recordings, and around 2,067 hours of transcribed speech, encompassing 112 languages from 165 districts from 31 States and Union territories. Notably, significant of these languages are being represented for the first time in a dataset of this scale, making the VAANI project a groundbreaking effort in preserving and promoting linguistic inclusivity. This data can be instrumental in building inclusive speech models for India, and in advancing research and development across speech, image, and multimodal applications.
#### ParaSpeechCLAP: A Dual-Encoder Speech-Text Model for Rich Stylistic Language-Audio Pretraining
 - **Authors:** Anuj Diwan, Eunsol Choi, David Harwath
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Artificial Intelligence (cs.AI); Computation and Language (cs.CL); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2603.28737

 - **Pdf link:** https://arxiv.org/pdf/2603.28737

 - **Abstract**
 We introduce ParaSpeechCLAP, a dual-encoder contrastive model that maps speech and text style captions into a common embedding space, supporting a wide range of intrinsic (speaker-level) and situational (utterance-level) descriptors (such as pitch, texture and emotion) far beyond the narrow set handled by existing models. We train specialized ParaSpeechCLAP-Intrinsic and ParaSpeechCLAP-Situational models alongside a unified ParaSpeechCLAP-Combined model, finding that specialization yields stronger performance on individual style dimensions while the unified model excels on compositional evaluation. We further show that ParaSpeechCLAP-Intrinsic benefits from an additional classification loss and class-balanced training. We demonstrate our models' performance on style caption retrieval, speech attribute classification and as an inference-time reward model that improves style-prompted TTS without additional training. ParaSpeechCLAP outperforms baselines on most metrics across all three applications. Our models and code are released at this https URL .
#### Multilingual Stutter Event Detection for English, German, and Mandarin Speech
 - **Authors:** Felix Haas, Sebastian P. Bayerl
 - **Subjects:** Subjects:
Sound (cs.SD); Computation and Language (cs.CL); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2603.26939

 - **Pdf link:** https://arxiv.org/pdf/2603.26939

 - **Abstract**
 This paper presents a multi-label stuttering detection system trained on multi-corpus, multilingual data in English, German, and this http URL leveraging annotated stuttering data from three languages and four corpora, the model captures language-independent characteristics of stuttering, enabling robust detection across linguistic contexts. Experimental results demonstrate that multilingual training achieves performance comparable to and, in some cases, even exceeds that of previous systems. These findings suggest that stuttering exhibits cross-linguistic consistency, which supports the development of language-agnostic detection systems. Our work demonstrates the feasibility and advantages of using multilingual data to improve generalizability and reliability in automated stuttering detection.


by Zyzzyva0381 (Windy). 


2026-03-31
