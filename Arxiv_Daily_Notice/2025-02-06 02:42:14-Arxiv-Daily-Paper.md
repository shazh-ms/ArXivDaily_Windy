# Showing new listings for Wednesday, 5 February 2025
Auto update papers at about 2:30am UTC (10:30am Beijing time) every weekday.


阅读 `Usage.md`了解如何使用此repo实现个性化的Arxiv论文推送

See `Usage.md` for instructions on how to personalize the repo. 


Keyword list: ['acoustic', 'reinforcement learning', 'reverb', 'meta', 'separate', 'reconstruction', 'noise', 'enhance', 'localization', 'speech']


Excluded: []


### Today: 7papers 
#### Privacy-Preserving Edge Speech Understanding with Tiny Foundation Models
 - **Authors:** Afsara Benazir, Felix Xiaozhu Lin
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS); Machine Learning (cs.LG); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2502.01649

 - **Pdf link:** https://arxiv.org/pdf/2502.01649

 - **Abstract**
 Robust speech recognition systems rely on cloud service providers for inference. It needs to ensure that an untrustworthy provider cannot deduce the sensitive content in speech. Sanitization can be done on speech content keeping in mind that it has to avoid compromising transcription accuracy. Realizing the under utilized capabilities of tiny speech foundation models (FMs), for the first time, we propose a novel use: enhancing speech privacy on resource-constrained devices. We introduce XYZ, an edge/cloud privacy preserving speech inference engine that can filter sensitive entities without compromising transcript accuracy. We utilize a timestamp based on-device masking approach that utilizes a token to entity prediction model to filter sensitive entities. Our choice of mask strategically conceals parts of the input and hides sensitive data. The masked input is sent to a trusted cloud service or to a local hub to generate the masked output. The effectiveness of XYZ hinges on how well the entity time segments are masked. Our recovery is a confidence score based approach that chooses the best prediction between cloud and on-device model. We implement XYZ on a 64 bit Raspberry Pi 4B. Experiments show that our solution leads to robust speech recognition without forsaking privacy. XYZ with < 100 MB memory, achieves state-of-the-art (SOTA) speech transcription performance while filtering about 83% of private entities directly on-device. XYZ is 16x smaller in memory and 17x more compute efficient than prior privacy preserving speech frameworks and has a relative reduction in word error rate (WER) by 38.8-77.5% when compared to existing offline transcription services.
#### Self-Supervised Convolutional Audio Models are Flexible Acoustic Feature Learners: A Domain Specificity and Transfer-Learning Study
 - **Authors:** Mattson Ogg
 - **Subjects:** Subjects:
Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2502.02366

 - **Pdf link:** https://arxiv.org/pdf/2502.02366

 - **Abstract**
 Self-supervised learning (SSL) algorithms have emerged as powerful tools that can leverage large quantities of unlabeled audio data to pre-train robust representations that support strong performance on diverse downstream tasks. Up to now these have mostly been developed separately for speech and non-speech applications. Here, we explored the domain specificity of a convolutional model's pre-training data relative to different downstream speech and non-speech tasks using a self-supervised pre-training approach (BYOL-A). We found that these pre-trained models (regardless of whether they were pre-trained on speech data, non-speech data or both) enabled good performance on nearly all downstream tasks, beating or nearly matching the performance of popular domain-specific models. Only small domain-specificity advantages were observed between the different pre-training datasets. The popular domain-specific models used as baselines performed very well in their target domains, but generally faltered outside of them. Together, these results demonstrate that SSL methods can be a powerful way to learn flexible representations for domain specific data without labels. These models can be a powerful resource for later transfer learning, fine-tuning or data exploration applications when the downstream data are similar, but also perhaps when there may be a domain mismatch.
#### Automated Extraction of Spatio-Semantic Graphs for Identifying Cognitive Impairment
 - **Authors:** Si-Ioi Ng, Pranav S. Ambadi, Kimberly D. Mueller, Julie Liss, Visar Berisha
 - **Subjects:** Subjects:
Artificial Intelligence (cs.AI); Computation and Language (cs.CL); Computer Vision and Pattern Recognition (cs.CV); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2502.01685

 - **Pdf link:** https://arxiv.org/pdf/2502.01685

 - **Abstract**
 Existing methods for analyzing linguistic content from picture descriptions for assessment of cognitive-linguistic impairment often overlook the participant's visual narrative path, which typically requires eye tracking to assess. Spatio-semantic graphs are a useful tool for analyzing this narrative path from transcripts alone, however they are limited by the need for manual tagging of content information units (CIUs). In this paper, we propose an automated approach for estimation of spatio-semantic graphs (via automated extraction of CIUs) from the Cookie Theft picture commonly used in cognitive-linguistic analyses. The method enables the automatic characterization of the visual semantic path during picture description. Experiments demonstrate that the automatic spatio-semantic graphs effectively differentiate between cognitively impaired and unimpaired speakers. Statistical analyses reveal that the features derived by the automated method produce comparable results to the manual method, with even greater group differences between clinical groups of interest. These results highlight the potential of the automated approach for extracting spatio-semantic features in developing clinical speech models for cognitive impairment assessment.
#### Adapter-Based Multi-Agent AVSR Extension for Pre-Trained ASR Models
 - **Authors:** Christopher Simic, Korbinian Riedhammer, Tobias Bocklet
 - **Subjects:** Subjects:
Sound (cs.SD); Machine Learning (cs.LG); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2502.01709

 - **Pdf link:** https://arxiv.org/pdf/2502.01709

 - **Abstract**
 We present an approach to Audio-Visual Speech Recognition that builds on a pre-trained Whisper model. To infuse visual information into this audio-only model, we extend it with an AV fusion module and LoRa adapters, one of the most up-to-date adapter approaches. One advantage of adapter-based approaches, is that only a relatively small number of parameters are trained, while the basic model remains unchanged. Common AVSR approaches train single models to handle several noise categories and noise levels simultaneously. Taking advantage of the lightweight nature of adapter approaches, we train noise-scenario-specific adapter-sets, each covering individual noise-categories or a specific noise-level range. The most suitable adapter-set is selected by previously classifying the noise-scenario. This enables our models to achieve an optimum coverage across different noise-categories and noise-levels, while training only a minimum number of parameters. Compared to a full fine-tuning approach with SOTA performance our models achieve almost comparable results over the majority of the tested noise-categories and noise-levels, with up to 88.5% less trainable parameters. Our approach can be extended by further noise-specific adapter-sets to cover additional noise scenarios. It is also possible to utilize the underlying powerful ASR model when no visual information is available, as it remains unchanged.
#### CTC-DRO: Robust Optimization for Reducing Language Disparities in Speech Recognition
 - **Authors:** Martijn Bartelds, Ananjan Nandi, Moussa Koulako Bala Doumbouya, Dan Jurafsky, Tatsunori Hashimoto, Karen Livescu
 - **Subjects:** Subjects:
Machine Learning (cs.LG); Computation and Language (cs.CL); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2502.01777

 - **Pdf link:** https://arxiv.org/pdf/2502.01777

 - **Abstract**
 Modern deep learning models often achieve high overall performance, but consistently fail on specific subgroups. Group distributionally robust optimization (group DRO) addresses this problem by minimizing the worst-group loss, but it fails when group losses misrepresent performance differences between groups. This is common in domains like speech, where the widely used connectionist temporal classification (CTC) loss scales with input length and varies with linguistic and acoustic properties, leading to spurious differences between group losses. We present CTC-DRO, which addresses the shortcomings of the group DRO objective by smoothing the group weight update to prevent overemphasis on consistently high-loss groups, while using input length-matched batching to mitigate CTC's scaling issues. We evaluate CTC-DRO on the task of multilingual automatic speech recognition (ASR) across five language sets from the ML-SUPERB 2.0 benchmark. CTC-DRO consistently outperforms group DRO and CTC-based baseline models, reducing the worst-language error by up to 65.9% and the average error by up to 47.7%. CTC-DRO can be applied to ASR with minimal computational costs, and offers the potential for reducing group disparities in other domains with similar challenges.
#### Sound Judgment: Properties of Consequential Sounds Affecting Human-Perception of Robots
 - **Authors:** Aimee Allen (1), Tom Drummond (2), Dana Kulić (1) ((1) Monash University - Australia, (2) University of Melbourne - Australia)
 - **Subjects:** Subjects:
Robotics (cs.RO); Human-Computer Interaction (cs.HC); Sound (cs.SD); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2502.02051

 - **Pdf link:** https://arxiv.org/pdf/2502.02051

 - **Abstract**
 Positive human-perception of robots is critical to achieving sustained use of robots in shared environments. One key factor affecting human-perception of robots are their sounds, especially the consequential sounds which robots (as machines) must produce as they operate. This paper explores qualitative responses from 182 participants to gain insight into human-perception of robot consequential sounds. Participants viewed videos of different robots performing their typical movements, and responded to an online survey regarding their perceptions of robots and the sounds they produce. Topic analysis was used to identify common properties of robot consequential sounds that participants expressed liking, disliking, wanting or wanting to avoid being produced by robots. Alongside expected reports of disliking high pitched and loud sounds, many participants preferred informative and audible sounds (over no sound) to provide predictability of purpose and trajectory of the robot. Rhythmic sounds were preferred over acute or continuous sounds, and many participants wanted more natural sounds (such as wind or cat purrs) in-place of machine-like noise. The results presented in this paper support future research on methods to improve consequential sounds produced by robots by highlighting features of sounds that cause negative perceptions, and providing insights into sound profile changes for improvement of human-perception of robots, thus enhancing human robot interaction.
#### Pruning-aware Loss Functions for STOI-Optimized Pruned Recurrent Autoencoders for the Compression of the Stimulation Patterns of Cochlear Implants at Zero Delay
 - **Authors:** Reemt Hinrichs, Jörn Ostermann
 - **Subjects:** Subjects:
Sound (cs.SD); Machine Learning (cs.LG); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2502.02424

 - **Pdf link:** https://arxiv.org/pdf/2502.02424

 - **Abstract**
 Cochlear implants (CIs) are surgically implanted hearing devices, which allow to restore a sense of hearing in people suffering from profound hearing loss. Wireless streaming of audio from external devices to CI signal processors has become common place. Specialized compression based on the stimulation patterns of a CI by deep recurrent autoencoders can decrease the power consumption in such a wireless streaming application through bit-rate reduction at zero latency. While previous research achieved considerable bit-rate reductions, model sizes were ignored, which can be of crucial importance in hearing-aids due to their limited computational resources. This work investigates maximizing objective speech intelligibility of the coded stimulation patterns of deep recurrent autoencoders while minimizing model size. For this purpose, a pruning-aware loss is proposed, which captures the impact of pruning during training. This training with a pruning-aware loss is compared to conventional magnitude-informed pruning and is found to yield considerable improvements in objective intelligibility, especially at higher pruning rates. After fine-tuning, little to no degradation of objective intelligibility is observed up to a pruning rate of about 55\,\%. The proposed pruning-aware loss yields substantial gains in objective speech intelligibility scores after pruning compared to the magnitude-informed baseline for pruning rates above 45\,\%.


by Zyzzyva0381 (Windy). 


2025-02-06
