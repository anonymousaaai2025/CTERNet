# CTERNet: Counterfactual Thinking Driven Emotion Regulation for Image Sentiment Recognition
## Introduction
Image sentiment recognition (ISR) facilitates the practical application of affective computing on rapidly growing social platforms. Nowadays, region-based ISR methods that use affective regions to guide emotion prediction have gained significant attention. However, existing methods face the main challenge that they cannot guide the generation of affective regions through causality-based mechanisms as humans do. Inspired by the psychological theory of Emotion Regulation, we propose a counterfactual thinking driven emotion regulation network (CTERNet), which simulates the Emotion Regulation theory by modeling the entire process of ISR based on human causality-driven mechanisms. Specifically, we first use multi-scale perception for feature extraction to simulate the stage of situation selection. Next, we combine situation modification, attentional deployment, and cognitive change into a counterfactual thinking based cognitive reappraisal module, which learns both affective regions (factual) and other potential affective regions (counterfactual). In the response modulation stage, we compare the factual and counterfactual outcomes to encourage the network to discover the most emotionally representative regions, thereby quantifying the quality of affective regions for ISR tasks. Experimental results demonstrate that our method outperforms or matches the state-of-the-art approaches, proving its effectiveness in addressing the key challenges of region-based ISR. 

## Architecture
<img src="https://github.com/anonymousaaai2025/CTERNet/blob/main/framework.png" width="800"/>

## Dependencies
- <code>python</code> (tested on python3.7)
- <code>PyTorch</code>  (tested on 1.2.0)
- <code>torchvision</code>  (tested on 0.4.0)

## Installiation
 1. Clone this repository.
 2. <code>pip install -r requirements.txt</code>


## Data Preparation
 1. Download large-scale dataset FI [here](https://drive.google.com/drive/folders/1gz5WhybpFT7F3YJ8Hl-6gxYWq12Gmbax?usp=drive_link), and put the splited dataset into <code>CausVSR/FI</code>.
 2. Download small-scale dataset [Emotion6](http://chenlab.ece.cornell.edu/downloads.html).


## Train
1. Launch training by the command below:
   ```
   $ python train.py
   ```
  
## Visualization
- All Causal Psuedo Sentiment Maps will be released after the acceptance.
<img src="https://github.com/anonymousaaai2025/CTERNet/blob/main/comparison.png" alt="The Visualization in CTERNet." width="500"/>


## TODO (_after acceptance_)
- Release the code of the Counterfactual Thinking based Cognitive Reappraisal Module (C^2RM).
- All training weights of experiments will be available after acceptance of the paper (Now the training weight on the FI dataset can be obtained [here](https://drive.google.com/file/d/19EJN2WhUFCulJ1FgDscRUZyZf_qxxUwS/view?usp=sharing)).
- All training models will be available after acceptance.

  
## References
Our code is developed based on:
- [WSCNet: Weakly Supervised Coupled Networks for Visual Sentiment Classification and Detection.](https://ieeexplore.ieee.org/document/8825564)
- [DCNet: Weakly Supervised Saliency Guided Dual Coding Network for Visual Sentiment Recognition.](https://www.researchgate.net/publication/374300197_DCNet_Weakly_Supervised_Saliency_Guided_Dual_Coding_Network_for_Visual_Sentiment_Recognition)

Thanks for their great work!
