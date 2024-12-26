# Showing new listings for Wednesday, 25 December 2024
Auto update papers at about 2:30am UTC (10:30am Beijing time) every weekday.


阅读 `Usage.md`了解如何使用此repo实现个性化的Arxiv论文推送

See `Usage.md` for instructions on how to personalize the repo. 


Keyword list: ['acoustic', 'reinforcement learning', 'reverb', 'meta', 'separate', 'reconstruction']


Excluded: []


### Today: 6papers 
#### CPFI-EIT: A CNN-PINN Framework for Full-Inverse Electrical Impedance Tomography on Non-Smooth Conductivity Distributions
 - **Authors:** Yang Xuanxuan, Zhang Yangming, Chen Haofeng, Ma Gang, Wang Xiaojie
 - **Subjects:** Subjects:
Signal Processing (eess.SP); Machine Learning (cs.LG)
 - **Arxiv link:** https://arxiv.org/abs/2412.17827

 - **Pdf link:** https://arxiv.org/pdf/2412.17827

 - **Abstract**
 This paper introduces a hybrid learning framework that combines convolutional neural networks (CNNs) and physics-informed neural networks (PINNs) to address the challenging problem of full-inverse electrical impedance tomography (EIT). EIT is a noninvasive imaging technique that reconstructs the spatial distribution of internal conductivity based on boundary voltage measurements from injected currents. This method has applications across medical imaging, multiphase flow detection, and tactile sensing. However, solving EIT involves a nonlinear partial differential equation (PDE) derived from Maxwell's equations, posing significant computational challenges as an ill-posed inverse problem. Existing PINN approaches primarily address semi-inverse EIT, assuming full access to internal potential data, which limits practical applications in realistic, full-inverse scenarios. Our framework employs a forward CNN-based supervised network to map differential boundary voltage measurements to a discrete potential distribution under fixed Neumann boundary conditions, while an inverse PINN-based unsupervised network enforces PDE constraints for conductivity reconstruction. Instead of traditional automatic differentiation, we introduce discrete numerical differentiation to bridge the forward and inverse networks, effectively decoupling them, enhancing modularity, and reducing computational demands. We validate our framework under realistic conditions, using a 16-electrode setup and rigorous testing on complex conductivity distributions with sharp boundaries, without Gaussian smoothing. This approach demonstrates robust flexibility and improved applicability in full-inverse EIT, establishing a practical solution for real-world imaging challenges.
#### Noisereduce: Domain General Noise Reduction for Time Series Signals
 - **Authors:** Tim Sainburg, Asaf Zorea
 - **Subjects:** Subjects:
Signal Processing (eess.SP); Sound (cs.SD)
 - **Arxiv link:** https://arxiv.org/abs/2412.17851

 - **Pdf link:** https://arxiv.org/pdf/2412.17851

 - **Abstract**
 Extracting signals from noisy backgrounds is a fundamental problem in signal processing across a variety of domains. In this paper, we introduce Noisereduce, an algorithm for minimizing noise across a variety of domains, including speech, bioacoustics, neurophysiology, and seismology. Noisereduce uses spectral gating to estimate a frequency-domain mask that effectively separates signals from noise. It is fast, lightweight, requires no training data, and handles both stationary and non-stationary noise, making it both a versatile tool and a convenient baseline for comparison with domain-specific applications. We provide a detailed overview of Noisereduce and evaluate its performance on a variety of time-domain signals.
#### EnhancePPG: Improving PPG-based Heart Rate Estimation with Self-Supervision and Augmentation
 - **Authors:** Luca Benfenati, Sofia Belloni, Alessio Burrello, Panagiotis Kasnesis, Xiaying Wang, Luca Benini, Massimo Poncino, Enrico Macii, Daniele Jahier Pagliari
 - **Subjects:** Subjects:
Signal Processing (eess.SP); Machine Learning (cs.LG)
 - **Arxiv link:** https://arxiv.org/abs/2412.17860

 - **Pdf link:** https://arxiv.org/pdf/2412.17860

 - **Abstract**
 Heart rate (HR) estimation from photoplethysmography (PPG) signals is a key feature of modern wearable devices for health and wellness monitoring. While deep learning models show promise, their performance relies on the availability of large datasets. We present EnhancePPG, a method that enhances state-of-the-art models by integrating self-supervised learning with data augmentation (DA). Our approach combines self-supervised pre-training with DA, allowing the model to learn more generalizable features, without needing more labelled data. Inspired by a U-Net-like autoencoder architecture, we utilize unsupervised PPG signal reconstruction, taking advantage of large amounts of unlabeled data during the pre-training phase combined with data augmentation, to improve state-of-the-art models' performance. Thanks to our approach and minimal modification to the state-of-the-art model, we improve the best HR estimation by 12.2%, lowering from 4.03 Beats-Per-Minute (BPM) to 3.54 BPM the error on PPG-DaLiA. Importantly, our EnhancePPG approach focuses exclusively on the training of the selected deep learning model, without significantly increasing its inference latency
#### Underwater Acoustic Reconfigurable Intelligent Surfaces: from Principle to Practice
 - **Authors:** Yu Luo, Lina Pu, Junming Diao, Chun-Hung Liu, Aijun Song
 - **Subjects:** Subjects:
Signal Processing (eess.SP)
 - **Arxiv link:** https://arxiv.org/abs/2412.17865

 - **Pdf link:** https://arxiv.org/pdf/2412.17865

 - **Abstract**
 This article explores the potential of underwater acoustic reconfigurable intelligent surfaces (UA-RIS) for facilitating long-range and eco-friendly communication in marine environments. Unlike radio frequency-based RIS (RF-RIS), which have been extensively investigated in terrestrial contexts, UA-RIS is an emerging field of study. The distinct characteristics of acoustic waves, including their slow propagation speed and potential for noise pollution affecting marine life, necessitate a fundamentally different approach to the architecture and design principles of UA-RIS compared to RF-RIS. Currently, there is a scarcity of real systems and experimental data to validate the feasibility of UA-RIS in practical applications. To fill this gap, this article presents field tests conducted with a prototype UA-RIS consisting of 24 acoustic elements. The results demonstrate that the developed prototype can effectively reflect acoustic waves to any specified directions through passive beamforming, thereby substantially extending the range and data rate of underwater communication systems.
#### Updatable Closed-Form Evaluation of Arbitrarily Complex Multi-Port Network Connections
 - **Authors:** Hugo Prod'homme, Philipp del Hougne
 - **Subjects:** Subjects:
Signal Processing (eess.SP)
 - **Arxiv link:** https://arxiv.org/abs/2412.17884

 - **Pdf link:** https://arxiv.org/pdf/2412.17884

 - **Abstract**
 The design of large complex wave systems (filters, networks, vacuum-electronic devices, metamaterials, smart radio environments, etc.) requires repeated evaluations of the scattering parameters resulting from complex connections between constituent subsystems. Instead of starting each new evaluation from scratch, we propose a computationally efficient method that updates the outcomes of previous evaluations using the Woodbury matrix identity. To enable this method, we begin by identifying a closed-form approach capable of evaluating arbitrarily complex connection schemes of multi-port networks. We pedagogically present unified equivalence principles for interpretations of system connections, as well as techniques to reduce the computational burden of the closed-form approach using these equivalence principles. Along the way, we also achieve the closed-form retrieval of the power waves traveling through connected ports. We illustrate our techniques considering a complex meta-network involving serial, parallel and cyclic connections between multi-port subsystems. We further validate all results with physics-compliant calculations considering graph-based subsystems, and we conduct exhaustive statistical analyses of computational benefits originating from the reducibility and updatability enabled by our approach. Finally, we find that working with scattering parameters (as opposed to impedance or admittance parameters) presents a fundamental advantage regarding an important class of connection schemes whose closed-form analysis requires the treatment of some connections as delayless, lossless, reflectionless and reciprocal two-port scattering systems. We expect our results to benefit the design (and characterization) of large composite (reconfigurable) wave systems.
#### Joint Adaptive OFDM and Reinforcement Learning Design for Autonomous Vehicles: Leveraging Age of Updates
 - **Authors:** Mamady Delamou, Ahmed Naeem, Huseyin Arslan, El Mehdi Amhoud
 - **Subjects:** Subjects:
Signal Processing (eess.SP); Artificial Intelligence (cs.AI)
 - **Arxiv link:** https://arxiv.org/abs/2412.18500

 - **Pdf link:** https://arxiv.org/pdf/2412.18500

 - **Abstract**
 Millimeter wave (mmWave)-based orthogonal frequency-division multiplexing (OFDM) stands out as a suitable alternative for high-resolution sensing and high-speed data transmission. To meet communication and sensing requirements, many works propose a static configuration where the wave's hyperparameters such as the number of symbols in a frame and the number of frames in a communication slot are already predefined. However, two facts oblige us to redefine the problem, (1) the environment is often dynamic and uncertain, and (2) mmWave is severely impacted by wireless environments. A striking example where this challenge is very prominent is autonomous vehicle (AV). Such a system leverages integrated sensing and communication (ISAC) using mmWave to manage data transmission and the dynamism of the environment. In this work, we consider an autonomous vehicle network where an AV utilizes its queue state information (QSI) and channel state information (CSI) in conjunction with reinforcement learning techniques to manage communication and sensing. This enables the AV to achieve two primary objectives: establishing a stable communication link with other AVs and accurately estimating the velocities of surrounding objects with high resolution. The communication performance is therefore evaluated based on the queue state, the effective data rate, and the discarded packets rate. In contrast, the effectiveness of the sensing is assessed using the velocity resolution. In addition, we exploit adaptive OFDM techniques for dynamic modulation, and we suggest a reward function that leverages the age of updates to handle the communication buffer and improve sensing. The system is validated using advantage actor-critic (A2C) and proximal policy optimization (PPO). Furthermore, we compare our solution with the existing design and demonstrate its superior performance by computer simulations.


by Zyzzyva0381 (Windy). 


2024-12-26
