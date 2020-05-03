# LR_GC_OOD
Code [Under Progress] &amp; Data for the AAAI 2020 Paper "Likelihood Ratios and Generative Classifiers For Unsupervised OOD Detection In Task-Based Dialog"


Code [Under Progress]:

Refer to requirements.txt for the python package requirements
For other specifications, refer to other_specifications.txt


Code Structure and TLDR:
code/util.py: Contains most of the argument speficications. Ignore arguments or argument groups with an "IGNORE" comment on top of them
code/train.py: Contains the training and inference mechanism
code/model.py: Specifices architecture for most of the models e.g Discriminative Classifier, Generative Classifier etc
code/oodmetrics.py: Code for computing the ood-related metrics such as AUROC 

Please ignore code/model_gan.py and code/wasserstein.py. They are not really used much for the paper experiments, but we have just retained them to not meddle with the imports.


Shell Scripts:
train_for_fbrelease.sh - Commands for fbrelease i.e ROSTD with its corresponding id training set and validation sets
train_for_fbreleasecoarse.sh - Commands for fbreleasecoarse i.e ROSTD with its corresponding id training set and validation sets, but with labels coarsened
train_for_atis.sh - Commands for atis
train_for_snips.sh - Commands for snips
