# LR_GC_OOD
Code &amp; Data for the AAAI 2020 Paper <a href="https://arxiv.org/pdf/1912.12800.pdf">"Likelihood Ratios and Generative Classifiers For Unsupervised OOD Detection In Task-Based Dialog"</a><br/>


<b> Data: </b> <br/>
The ROSTD dataset of OOD points can be found under <i>data/fbrelease</i> <br/>
This tsv file contains ~4500 OOD examples. The 3rd field of each line contains the sentence. <br/>
This is the only field which could be of interest - the other fields are vestigial and can be ignored. <br/>

Note that this OOD dataset is a companion to the ID dataset released as part of the paper <i>"Cross-lingual transfer learning for multilingual task oriented dialog"</i> by Schuster et al at NAACL 2019.<br/>
This ID dataset can be found in its original form <a href="https://fb.me/multilingual_task_oriented_data">here</a>.<br/>

Alternatively, you can directly use the splits we made (with ID train, and ID-OOD mixed validation and test) as described under the <b>"Dataset Splits"</b> section below. 

<b> Reference: </b> <br/>
If you find our code or data useful, please consider citing our paper: <br/>
```
@article{gangal2019likelihood,
  title={Likelihood Ratios and Generative Classifiers for Unsupervised Out-of-Domain Detection In Task Oriented Dialog},
  author={Gangal, Varun and Arora, Abhinav and Einolghozati, Arash and Gupta, Sonal},
  journal={arXiv preprint arXiv:1912.12800},
  year={2019}
}
```
<br/>


<b> Contact: </b> <br/>
For any questions or issues, either raise an issue here or drop an email at vgangal@andrew.cmu.edu <br/>

<b> Code: [Under Progress] </b> <br/>

Refer to <b>requirements.txt</b> for the python package requirements
For other specifications, refer to <b>other_specifications.txt</b>


<b> Code Structure and TLDR: </b>

<i>code/util.py</i>: Contains most of the argument specifications. Ignore arguments or argument groups with an "IGNORE" comment on top of them <br/>

<i>code/train.py</i>: Contains the training and inference mechanism <br/>

<i>code/model.py</i>: Specifices architecture for most of the models e.g Discriminative Classifier, Generative Classifier etc <br/>

<i>code/oodmetrics.py</i>: Code for computing the ood-related metrics such as AUROC <br/>

Please ignore code/model_gan.py and code/wasserstein.py. They are not really used much for the paper experiments, but we have just retained them to not meddle with the imports. <br/>

<b>Dataset Splits:</b>

- <b>For fbrelease and fbreleasecoarse</b>
	You can directly find the ready-to-use dataset splits under <i>code/data/{dataset_name}/unsup/</i> for dataset_name = fbrelease / fbreleasecoarse <br/>
	This already contains the plain id train split and the id-ood mixed dev and test splits <br/>
	Note that only the ood part of the fbrelease dev and test splits constitutes our own released data. The rest is formed from existing datasets. <br/>
- <b> For atis and snips </b>
        You will need to run some scripts to do random splitting where a fraction of classes are held out as OOD. <br/>
        The <i>code/data/{dataset_name}/preprocess_{dataset_name}.sh</i> needs to be run for this. (Where dataset_name = atis/snips)<br/> 



<b>Shell Scripts:</b> <br/>

train_for_fbrelease.sh - Commands for fbrelease i.e ROSTD with its corresponding id training set and validation sets <br/>

train_for_fbreleasecoarse.sh - Commands for fbreleasecoarse i.e ROSTD with its corresponding id training set and validation sets, but with labels coarsened. <br/>

train_for_atis.sh - Commands for atis <br/>

train_for_snips.sh - Commands for snips <br/>

<b>Notes:</b><br/>
   - In all of these scripts, you will need to set <b>super_root</b> to point to where the repo resides on your system. We need this because we use torchtext to preprocess, create the vocabulary, load and minibatch our datasets, and we could only get it to work with absolute path specifications.
