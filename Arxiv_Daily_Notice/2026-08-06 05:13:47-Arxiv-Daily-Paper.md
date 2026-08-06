# Showing new listings for Thursday, 6 August 2026
Auto update papers at about 2:30am UTC (10:30am Beijing time) every weekday.


阅读 `Usage.md`了解如何使用此repo实现个性化的Arxiv论文推送

See `Usage.md` for instructions on how to personalize the repo. 


Keyword list: ['text-to-speech', 'text to speech', 'tts', 'LLM-based', 'speech', 'voice']


Excluded: []


### Today: 1papers 
#### Helping Music Co-Creation Agents 'Listen' Well: Hierarchical Self-Supervised World Models for Understanding and Generation
 - **Authors:** Scott H. Hawley
 - **Subjects:** Subjects:
Sound (cs.SD); Machine Learning (cs.LG); Audio and Speech Processing (eess.AS)
 - **Arxiv link:** https://arxiv.org/abs/2608.04378

 - **Pdf link:** https://arxiv.org/pdf/2608.04378

 - **Abstract**
 Collaborative music agents need internal representations rich enough to support both understanding and generation, yet flexible enough for a workflow where the human retains agency. We present a hierarchical self-supervised ``world model'' for symbolic music: a 2.55M-parameter Swin V2 encoder trained on MIDI piano-roll images with JEPA-style objectives (pitch- and time-shift equivariance, masked embedding prediction, and a distributional regularizer), using no labels and no music-theory vocabulary. Probing the frozen embeddings shows that the level at which a musical property becomes decodable tracks its musical time scale: phrase boundaries are read off the coarsest levels, note density and harmonic detail off the finest. Temporal and phrase structure emerge from the self-supervised objectives alone, while harmonic content must be asked for; a small chord-supervision head raises joint chord recovery from .18 to .54, and key detection, which is never supervised, from .16 to .70. Following the Representation AutoEncoder paradigm, a conditional flow-matching model stands in for a trained decoder, flowing in pixel space from PCA-reduced conditioning: it reproduces a target window at pixel F1 $0.996$, and the same per-level conditioning dropout that controls how far variations stray also enables graphical prompting for masked inpainting with no inpainting-specific sampler. The pipeline runs on CPU producing a suggestion in $2.8$ s, or $0.6$ s on Apple MPS, which we demonstrate in a live interactive demo. In concert with an LLM-based brain, these capabilities supply the core of a collaborative music creation agent in service of, rather than in place of, human agency.


by Zyzzyva0381 (Windy). 


2026-08-06
