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

Dataset Splits:

- For fbrelease and fbreleasecoarse
	You can directly find the ready-to-use dataset splits under code/data/{dataset_name}/unsup/ for dataset_name = fbrelease / fbreleasecoarse
	This already contains the plain id train split and the id-ood mixed dev and test splits
	Note that only the ood part of the fbrelease dev and test splits constitutes our own released data. The rest is formed from existing datasets.
- For atis and snips
        You will need to run some scripts to do random splitting where a fraction of classes are held out as OOD.
        The code/data/{dataset_name}/preprocess_{dataset_name}.sh need to be run for this. (Where dataset_name = atis/snips) 



Shell Scripts:

train_for_fbrelease.sh - Commands for fbrelease i.e ROSTD with its corresponding id training set and validation sets

train_for_fbreleasecoarse.sh - Commands for fbreleasecoarse i.e ROSTD with its corresponding id training set and validation sets, but with labels coarsened

train_for_atis.sh - Commands for atis

train_for_snips.sh - Commands for snips

