import os
import time
import glob
import math
import torch
import torch.optim as O
import torch.nn as nn
import torch.nn.functional as F

from torchtext import data
from torchtext import datasets

from model import IntentClassifier, IntentClassifierGenerative, BareLSTMEncoder, VVUL, corrupt
from model_gan import Gen, Disc
from util import get_args, makedirs

import numpy as np

from oodMetrics import oodMetrics
from wasserstein import WassersteinLossVanilla, WassersteinLossStab

import pickle

import copy

def padUpToLength(string,length=22):
    return string+"".join([" ",]*(length-len(string))),

def get_ppnf_disagreements(model,intermediateRepn,label_index_list,answer,numpy=False):
    if not numpy:
        intermediateRepnNumpy = intermediateRepn.detach().cpu().numpy()
    else:
        intermediateRepnNumpy = intermediateRepn
    start_time = time.process_time()
    neigh_dist, neigh_ids = model.neigh.kneighbors(intermediateRepnNumpy)
    print("Time Taken Per NN Query:",time.process_time()-start_time)
    ppnf_distance_mean = np.mean(neigh_dist,axis=1)
    print(np.shape(neigh_ids),np.shape(label_index_list))
    ppnf_labels = label_index_list[neigh_ids]
    predicted_labels = torch.max(answer,1)[1].detach().cpu().numpy()
    disagreements = (ppnf_labels!=predicted_labels[:,None]).astype(int)
    if args.ppnf_w:
        neigh_sims = np.reciprocal(neigh_dist+1e-8)
        disagreements = neigh_sims * disagreements
        disagreements = np.mean(disagreements,axis=1) * np.reciprocal(np.sum(neigh_sims,axis=1))
    else:
        disagreements = np.mean(disagreements,axis=1)
    return disagreements

# Get argparse object
args = get_args()

# Set torch cuda device
torch.cuda.set_device(args.gpu)
device = torch.device('cuda:{}'.format(args.gpu))

import random
#Setting seeds
random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)
np.random.seed(int(args.seed))

# When using linearized sequences line in seql, the tsv field corresponding to utterances
# becomes quite long. To allow this, some default csv/tsv settings have to be altered.
if args.dataset == "seql":
    import csv
    import sys
    print("Increasing csv field size for the seql dataset")
    csv.field_size_limit(sys.maxsize)

import spacy
spacy_en = spacy.load('en')

def tokenizer(text):
    return [tok.text for tok in spacy_en.tokenizer(text)]

if args.unsup:
    answers = data.Field(sequential=False, use_vocab=True, unk_token="outOfDomain")
    #answers = data.Field(sequential=False, use_vocab=True, unk_token=None)
else:
    answers = data.Field(sequential=False, use_vocab=True, unk_token=None)
emptyField = data.Field(sequential=True, lower=args.lower, tokenize='spacy')
inputs = data.Field(sequential=True, lower=args.lower, tokenize='spacy',eos_token='<eos>')
inputsPost = data.Field(sequential=True, lower=args.lower, tokenize='spacy')


if args.dataset == "assistant":
    if args.debug and not args.unsup:
        train, dev, test = data.TabularDataset.splits(path=args.super_root, train="train10K.tsv", validation="eval.tsv", test = "test.tsv", format="tsv", fields=[('label', answers), ('Garbage', emptyField),  ('hypothesis', inputs), ('Text2', inputsPost)])
    elif args.unsup and args.debug:
        unsup_root = "data/assistant/unsup/"
        train, dev, test = data.TabularDataset.splits(path=args.super_root+unsup_root, train="OODRemovedtrain10K.tsv", validation="eval.tsv", test = "test.tsv", format="tsv", fields=[('label', answers), ('Garbage', emptyField),  ('hypothesis', inputs), ('Text2', inputsPost)])
    elif args.unsup:
        unsup_root = "data/assistant/unsup/"
        train, dev, test = data.TabularDataset.splits(path=args.super_root+unsup_root, train="OODRemovedtrain.tsv", validation="eval.tsv", test = "test.tsv", format="tsv", fields=[('label', answers), ('Garbage', emptyField),  ('hypothesis', inputs), ('Text2', inputsPost)])
    else:
        train, dev, test = data.TabularDataset.splits(path=args.super_root, train="train.tsv", validation="eval.tsv", test = "test.tsv", format="tsv", fields=[('label', answers), ('Garbage', emptyField),  ('hypothesis', inputs), ('Text2', inputsPost)])
elif args.dataset == "augmented_assistant":
    print("Here")
    train, dev, test = data.TabularDataset.splits(path=args.super_root, train="data/augmented_assistant/train.tsv", validation="eval.tsv", test = "test.tsv", format="tsv", fields=[('label', answers), ('Garbage', emptyField),  ('hypothesis', inputs), ('Text2', inputsPost)])
elif args.dataset == "emnlp" or args.dataset == "seql" :
    prefix = args.super_root+"data/"+args.dataset+"/"
    train, dev, test = data.TabularDataset.splits(path=prefix, train="train.tsv", validation="eval.tsv", test = "test.tsv", format="tsv", fields=[('label', answers), ('Garbage', emptyField),  ('hypothesis', inputs), ('Text2', inputsPost)])
elif args.dataset == "assistant_frac":
    train, dev, test = data.TabularDataset.splits(path=args.super_root, train="frac_train_sets/train_"+str(args.frac)+"Pc.tsv", validation="eval.tsv", test = "test.tsv", format="tsv", fields=[('label', answers), ('Garbage', emptyField),  ('hypothesis', inputs), ('Text2', inputsPost)])
elif args.dataset == "assistant_pairwise":
    train, dev, test = data.TabularDataset.splits(path=args.super_root, train="IdVsOod_train.tsv", validation="IdVsOod_eval.tsv", test = "IdVsOod_test.tsv", format="tsv", fields=[('label', answers), ('Garbage', emptyField),  ('hypothesis', inputs), ('Text2', inputsPost)])
elif args.dataset == "atis" or args.dataset=="snips" or args.dataset=="fbsemdial" or args.dataset=="fbmlto":
    prefix = args.super_root+"data/"+args.dataset+"/"
    if not args.unsup:
        train, dev, test = data.TabularDataset.splits(path=prefix+"sup/", train="train.tsv", validation="eval.tsv", test = "eval.tsv", format="tsv", fields=[('label', answers), ('Garbage', emptyField),  ('hypothesis', inputs), ('Text2', inputsPost)])
    else:
        split_prefix = prefix+"unsup_"+str(args.id_ratio)+"_"+str(args.split_id)+"/"
        train, dev, test = data.TabularDataset.splits(path=split_prefix, train="OODRemovedtrain.tsv", validation="eval.tsv", test = "test.tsv", format="tsv", fields=[('label', answers), ('Garbage', emptyField),  ('hypothesis', inputs), ('Text2', inputsPost)])
elif args.dataset=="fbrelease" or args.dataset=="fbreleasecoarse":
    prefix = args.super_root+"data/"+args.dataset+"/"
    if not args.unsup:
        train, dev, test = data.TabularDataset.splits(path=prefix+"sup/", train="train.tsv", validation="eval.tsv", test = "eval.tsv", format="tsv", fields=[('label', answers), ('Garbage', emptyField),  ('hypothesis', inputs), ('Text2', inputsPost)])
    else:
        split_prefix = prefix+"unsup/"
        train, dev, test = data.TabularDataset.splits(path=split_prefix, train="OODRemovedtrain.tsv", validation="eval.tsv", test = "test.tsv", format="tsv", fields=[('label', answers), ('Garbage', emptyField),  ('hypothesis', inputs), ('Text2', inputsPost)])

else:
    prefix = args.super_root+"data/"+args.dataset+"/"
    if args.debug and not args.unsup:
        train, dev, test = data.TabularDataset.splits(path=prefix+"sup/", train="train.tsv", validation="eval.tsv", test = "eval.tsv", format="tsv", fields=[('label', answers), ('Garbage', emptyField),  ('hypothesis', inputs), ('Text2', inputsPost)])
    elif args.unsup and args.debug:
        train, dev, test = data.TabularDataset.splits(path=prefix+"unsup/", train="OODRemovedtrain.tsv", validation="OODRemovedeval.tsv", test = "OODRemovedeval.tsv", format="tsv", fields=[('label', answers), ('Garbage', emptyField),  ('hypothesis', inputs), ('Text2', inputsPost)])
    elif args.unsup:
        train, dev, test = data.TabularDataset.splits(path=prefix+"unsup/", train="OODRemovedtrain.tsv", validation="OODRemovedeval.tsv", test = "OODRemovedeval.tsv", format="tsv", fields=[('label', answers), ('Garbage', emptyField),  ('hypothesis', inputs), ('Text2', inputsPost)])
    else:
        train, dev, test = data.TabularDataset.splits(path=prefix+"sup/", train="train.tsv", validation="eval.tsv", test = "eval.tsv", format="tsv", fields=[('label', answers), ('Garbage', emptyField),  ('hypothesis', inputs), ('Text2', inputsPost)])


inputs.build_vocab(train, dev, test)
emptyField.build_vocab(train,dev,test)
inputsPost.build_vocab(train,dev,test)

#if args.word_vectors:
#    if os.path.isfile(args.vector_cache):
#        inputs.vocab.vectors = torch.load(args.vector_cache)
#    else:
#        inputs.vocab.load_vectors(args.word_vectors)
#        makedirs(os.path.dirname(args.vector_cache))
#        torch.save(inputs.vocab.vectors, args.vector_cache)

init_tuple = None
if args.word_vectors:
    glove_vocab, glove_matrix = pickle.load(open(args.word_vectors,"rb"))
    init_matrix = np.zeros((len(inputs.vocab.itos), args.d_embed))
    init_mask = np.zeros((len(inputs.vocab.itos),))
    for vocab_word in inputs.vocab.stoi.keys():
        if vocab_word in glove_vocab:
            init_matrix[inputs.vocab.stoi[vocab_word],:] = glove_matrix[glove_vocab[vocab_word],:]
            init_mask[inputs.vocab.stoi[vocab_word]]=1
    init_tuple = (init_matrix,init_mask)

answers.build_vocab(train)


train_iter, dev_iter, test_iter = data.BucketIterator.splits(
        (train, dev, test), batch_size=args.batch_size, device=device, sort_key = lambda x: len(x.hypothesis))



config = args
config.n_embed = len(inputs.vocab)
config.d_out = len(answers.vocab)
print("Number Of Labels:",config.d_out)
config.n_cells = config.n_layers
config.vocab_obj = inputs.vocab

train_iter_clone,  = data.BucketIterator.splits((train,), batch_size=args.batch_size, device=device, sort_key = lambda x: len(x.hypothesis))

# double the number of cells for bidirectional networks
if config.birnn:
    config.n_cells *= 2

if args.resume_snapshot:
    model = torch.load(args.resume_snapshot, map_location=device)
    if args.back_lm:
        lm_model = torch.load(args.resume_snapshot[:-3]+"_lm"+args.resume_snapshot[-3:], map_location=device)
    if args.back_mirror:
        mirror_model = torch.load(args.resume_snapshot[:-3]+"_mirror"+args.resume_snapshot[-3:], map_location=device)
    if args.seq_gan:
        gen_model = torch.load(args.resume_snapshot[:-3]+"_gen"+args.resume_snapshot[-3:], map_location=device)
        disc_model = torch.load(args.resume_snapshot[:-3]+"_disc"+args.resume_snapshot[-3:], map_location=device)
else:
    if not args.generative:
        model = IntentClassifier(config,init_tuple = init_tuple)
        if args.seq_gan:
            gen_model = Gen(config,init_tuple=init_tuple)
            disc_model = Disc(config,init_tuple=init_tuple)
    else:
        model = IntentClassifierGenerative(config,init_tuple = init_tuple)
        if args.fore_lm:
            model = BareLSTMEncoder(config)
            perplexity = 0.0
            token_count = 0.0
        if args.back_lm:
            if args.vvul:
                lm_model = VVUL(config,args.vvul_size)
            else:
                lm_model = BareLSTMEncoder(config)
        if args.back_mirror:
            mirror_model = IntentClassifierGenerative(config, init_tuple = init_tuple)
    #if args.word_vectors:
    #    model.embed.weight.data.copy_(inputs.vocab.vectors)
    #    model.to(device)


reductionString = 'mean'
if args.no_CE_for_OOD:
    reductionString = 'none'
criterion = nn.CrossEntropyLoss(reduction=reductionString)
if args.multi_margin_softmax:
    criterion = nn.MultiMarginLoss()

opt = O.Adam(model.parameters(), lr=args.lr)
if args.seq_gan:
    opt_gen = O.Adam(gen_model.parameters(), lr=args.gen_lr)
    opt_disc = O.Adam(disc_model.parameters(), lr=args.disc_lr)
if args.back_lm: opt_lm = O.Adam(lm_model.parameters(), lr=args.lr)
if args.back_mirror: opt_mirror = O.Adam(mirror_model.parameters(), lr=args.lr)

if args.corrupt_back and (args.noise_type=="unigram" or args.noise_type=="uniroot"):
    config.freq_list = [inputs.vocab.freqs[inputs.vocab.itos[i]] for i in range(len(inputs.vocab))]
    config.freq_list_tensor = torch.Tensor(np.asarray(config.freq_list[3:]))
    if args.noise_type=="unigram": config.freq_list_tensor = config.freq_list_tensor / torch.sum(config.freq_list_tensor)
    elif args.noise_type=="uniroot": config.freq_list_tensor = torch.sqrt(config.freq_list_tensor) / torch.sum( torch.sqrt(config.freq_list_tensor) )

iterations = 0
start = time.time()
best_dev_acc = -1
best_dev_macro_f1_score = -1
header = '  Time Epoch Iteration Progress    (%Epoch)   Loss   Dev/Loss     Accuracy  Dev/Accuracy'
dev_log_template = ' '.join('{:>6.0f},{:>5.0f},{:>9.0f},{:>5.0f}/{:<5.0f} {:>7.0f}%,{:>8.6f},{:8.6f},{:12.4f},{:12.4f}'.split(','))
log_template =     ' '.join('{:>6.0f},{:>5.0f},{:>9.0f},{:>5.0f}/{:<5.0f} {:>7.0f}%,{:>8.6f},{},{:12.4f},{}'.split(','))
if not args.infer_only:
    makedirs(args.save_path)
makedirs("validPlots/"+args.save_path)
print(header)

if args.generative:
    "Estimating Prior Counts"
    train_iter_clone.init_epoch()
    model.to(device)
    if args.back_mirror:
        mirror_model.to(device)
    if not args.fore_lm:
        with torch.no_grad():
            for batch_idx,batch in enumerate(train_iter_clone):
                model.update_prior_counts(batch)
                if args.back_mirror: mirror_model.update_prior_counts(batch)
        "Done Estimating Prior Counts"
        print(model.get_prior_log_probs())
        if args.back_mirror: print(mirror_model.get_prior_log_probs())

for epoch in range(args.epochs):
    if args.fore_lm:
        perplexity, token_count = 0.0, 0.0
    if args.infer_only and epoch>=1:
        print("Exiting. Done with the 1 epoch needed for inference")
        exit()
    train_iter.init_epoch()
    n_correct, n_total = 0, 0
    model.to(device)
    if args.back_lm: lm_model.to(device)
    if args.back_mirror: mirror_model.to(device)
    if args.seq_gan:
        gen_model.to(device)
        disc_model.to(device)
        if args.seq_gan_class:
            if args.seq_gan_class_zeta>0:
                if args.seq_gan_class_beta < args.seq_gan_class_beta_cap:
                    args.seq_gan_class_beta = min(args.seq_gan_class_beta*args.seq_gan_class_zeta,args.seq_gan_class_beta_cap)
            elif args.seq_gan_class_omega>0:
                if args.seq_gan_class_beta > args.seq_gan_class_beta_cap:
                    args.seq_gan_class_beta = max(args.seq_gan_class_beta*args.seq_gan_class_omega, args.seq_gan_class_beta_cap)

    for batch_idx, batch in enumerate(train_iter):

        # switch model to training mode, clear gradient accumulators
        if not args.infer_only:
            model.train(); opt.zero_grad()
            if args.seq_gan:
                gen_model.train();opt_gen.zero_grad()
                disc_model.train();opt_disc.zero_grad()
            if args.back_lm: lm_model.train(); opt_lm.zero_grad()
            if args.back_mirror: mirror_model.train(); opt_mirror.zero_grad()
        else:
            print("Putting model in eval mode for inference")
            model.eval();
            if args.seq_gan:
                gen_model.eval()
                disc_model.eval()
            if args.back_lm: lm_model.eval()
            if args.back_mirror: mirror_model.eval()
        iterations += 1

        # forward pass
        if not args.generative:
            answer, answerOOD, intermediateRepn = model(batch)
            if args.seq_gan:
                class_model = None
                if args.seq_gan_class: class_model = model
                loss_gen, loss_class_from_gen = gen_model.sample_batch(batch.hypothesis,disc_model=disc_model,reduction=args.gen_reduction,class_model=class_model)
                print("Generator Loss:",loss_gen)
                if args.seq_gan_class:
                    print("Classifier Loss From Generator:",loss_class_from_gen)
                sampled_batch_for_disc = gen_model.sample_batch(batch.hypothesis,just_sample=True,reduction=args.gen_reduction)
                loss_disc_pos = 0.5*disc_model.compute_loss(batch.hypothesis,1,reduction=args.gen_reduction)
                if not args.infer_only:
                    loss_disc_pos.backward()
                loss_disc_neg = 0.5*disc_model.compute_loss(sampled_batch_for_disc,0,reduction=args.gen_reduction)
                if not args.infer_only:
                    loss_disc_neg.backward()
                print("Discriminator Losses:",loss_disc_pos,loss_disc_neg)
        else:
            if not args.fore_lm:
                answer, word_scores  = model(batch,inferAll=False)
                #answer = model.infer_generative(batch,answer)
            else:
                word_scores = model(batch)
            if args.back_lm:
                lm_batch = batch
                if args.corrupt_back:
                    lm_batch = corrupt(batch, config , inputs, noise_level=args.noise_level)
                word_scores_lm = lm_model(lm_batch)
                if args.back_lm_diff:
                    word_scores_lm_diff = lm_model(batch)
            if args.back_mirror:
                mirror_batch = batch
                if args.corrupt_back:
                    mirror_batch = corrupt(batch, config , inputs, noise_level=args.noise_level)
                _ , word_scores_mirror = mirror_model(mirror_batch, inferAll=False)
        #if args.returnIntermediate:
        #    print(intermediateRepn.size(), batch.label.size())

        # calculate accuracy of predictions in the current batch
        if not args.generative:
            n_correct += (torch.max(answer, 1)[1].view(batch.label.size()) == batch.label).sum().item()
        else:
            n_correct += 0
            torch.cuda.empty_cache()
        n_total += batch.batch_size
        train_acc = 100. * n_correct/n_total

        # calculate loss of the network output with respect to training labels
        if args.conf_teach:
            #one_hot_targets = (torch.FloatTensor(batch.label.size(0), answer.size(1)).zero_().scatter_(1, batch.label.unsqueeze(1).data.cpu(), 1))
            #one_hot_targets = one_hot_targets.to(device=answer.device)
            confidence = F.softmax(answerOOD,dim=1)[:,1]
            #confidence = confidence.unsqueeze(1)
            mixed_answer = F.softmax(answer,dim=1)
            mixed_answer_shifted = mixed_answer+1e-8
            mixed_answer_logits = torch.log(mixed_answer_shifted)
            loss = F.nll_loss(mixed_answer_logits, batch.label)
            oodLabel = (batch.label==args.ood_index).long()
            aux_BCE_OOD_loss =  criterion(answerOOD,oodLabel)
            if args.conf_match:
                reg_reward = torch.mean(mixed_answer[:,args.ood_index]*torch.log(confidence+1e-8)+(1-mixed_answer[:,args.ood_index])*torch.log(1-confidence+1e-8) )
            elif args.conf_mse:
                reg_reward = -torch.mean((mixed_answer[:,args.ood_index]-confidence)*(mixed_answer[:,args.ood_index]-confidence))
            else:
                reg_reward = torch.mean(mixed_answer[:,args.ood_index]*confidence)
            loss = loss - args.conf_teach_beta*reg_reward + 0.05*aux_BCE_OOD_loss
            #print(criterion(answer,batch.label))
        elif args.mos:
            loss = F.nll_loss(answer, batch.label)
        elif args.generative:
            #loss = criterion(word_scores, batch.hypothesis)
            if not args.fore_lm:
                loss = F.cross_entropy(word_scores.contiguous().view(-1,args.n_embed),batch.hypothesis.contiguous().view(-1),reduction='none')
                loss = loss.view(batch.hypothesis.size()).sum(dim=0)
                prior_log_probs = (model.get_prior_log_probs()).gather(0,batch.label)
                loss = loss - prior_log_probs.detach()
                loss = loss.mean(dim=0)
            else:
                loss =  F.cross_entropy(word_scores.contiguous().view(-1,args.n_embed),batch.hypothesis.contiguous().view(-1),reduction='none')
                is_not_pad_mask = (batch.hypothesis.contiguous().view(-1)!=args.pad_id).float()
                token_count+=sum(is_not_pad_mask)
                loss = loss * is_not_pad_mask
                loss = loss.view(batch.hypothesis.size()).sum(dim=0)
                perplexity += loss.sum(dim=0).detach().cpu().item()
                loss = loss.mean(dim=0)
            if args.back_lm:
                loss_lm = F.cross_entropy(word_scores_lm.contiguous().view(-1,args.n_embed),lm_batch.hypothesis.contiguous().view(-1),reduction='none')
                is_not_pad_mask = (lm_batch.hypothesis.contiguous().view(-1)!=args.pad_id).float()
                loss_lm = loss_lm * is_not_pad_mask
                loss_lm = loss_lm.view(lm_batch.hypothesis.size()).sum(dim=0)
                loss_lm = loss_lm.mean(dim=0)
                if args.back_lm_diff:
                    loss_lm_diff = F.cross_entropy(word_scores_lm_diff.contiguous().view(-1,args.n_embed),batch.hypothesis.contiguous().view(-1),reduction='none')
                    loss_lm_diff = loss_lm_diff * is_not_pad_mask
                    loss_lm_diff = loss_lm_diff.view(batch.hypothesis.size()).sum(dim=0)
                    loss_lm_diff = loss_lm_diff.mean(dim=0)
                    loss_lm = (loss_lm - loss_lm_diff)
            if args.back_mirror:
                loss_mirror = F.cross_entropy(word_scores_mirror.contiguous().view(-1,args.n_embed),mirror_batch.hypothesis.contiguous().view(-1),reduction='none')
                loss_mirror = loss_mirror.view(mirror_batch.hypothesis.size()).sum(dim=0)
                prior_log_probs_mirror = (mirror_model.get_prior_log_probs()).gather(0,mirror_batch.label)
                loss_mirror = loss_mirror - prior_log_probs_mirror.detach()
                loss_mirror = loss_mirror.mean(dim=0)
        else:
            if args.zero_exclude:
                loss = criterion(answer[:,1:], batch.label-1)
            else:
                loss = criterion(answer, batch.label)
            if args.seq_gan_class:
                print("Classifier Loss Main:",loss)
                if not args.seq_gan_class_frozen:
                    if args.bound_class_from_gen:
                        loss = loss + torch.min(loss_class_from_gen,(1+args.seq_gan_aux_clip)*loss)
                    else:
                        loss = loss + loss_class_from_gen

        if args.cosine_softmax:
            weightNorm=torch.norm(model.out.weight,dim=1)
            weightNormDeviation=torch.mean(weightNorm*weightNorm-1)
            #print("Loss:",loss)
            #print("Penalty:",weightNormDeviation)
            #loss = loss + 1e-8*weightNormDeviation

        if args.no_CE_for_OOD:
            oodLabel = (batch.label==args.ood_index).float()
            loss = torch.sum(loss*(1-oodLabel))
            if torch.sum(1-oodLabel)>1e-2:
                loss = loss * (1.0/torch.sum(1-oodLabel))
        if args.aux_BCE_OOD:
            #print(loss.size())
            #print(answer.size(),answerOOD.size())
            #print(batch.label.size())
            oodLabel = (batch.label==args.ood_index).long()
            #print(batch.label.type(), oodLabel.type())
            #print(batch.label,oodLabel)
            aux_BCE_OOD_loss = criterion(answerOOD,oodLabel)
            if args.additive_lambda:
                loss = loss + (args.aux_BCE_OOD_lambda)*aux_BCE_OOD_loss
            else:
                loss = (1-args.aux_BCE_OOD_lambda)*loss + (args.aux_BCE_OOD_lambda)*aux_BCE_OOD_loss
            #print(loss.detach().cpu().numpy(),aux_BCE_OOD_loss.detach().cpu().numpy())
            #exit()
        elif args.aux_H_OOD:
            #First we need to renormalize the logits and remove the OOD class
            answerClone = answer.clone()
            answerClone[:,args.ood_index]=-1e7

            #Find the entropy across non OOD classes for each example
            answerClone = torch.cat([answerClone[:,:args.ood_index],answerClone[:,args.ood_index+1:]], dim=1)
            answer_log_probs = F.log_softmax(answerClone,dim=1) #128*7 -> 128*7
            aux_H_loss_across_examples = -1 * answer_log_probs.mean(dim=1) #128*7 -> 128
            #print(aux_H_loss_across_examples.size())

            #Retain and take the average only across OOD Examples
            oodLabel = (batch.label==args.ood_index).float()
            aux_H_loss_across_examples = aux_H_loss_across_examples * oodLabel
            aux_H_loss = torch.sum(aux_H_loss_across_examples)
            if torch.sum(oodLabel)>1e-2:
                aux_H_loss = aux_H_loss * (1.0/torch.sum(oodLabel))
            if args.additive_lambda:
                loss = loss + (args.aux_H_OOD_lambda)*aux_H_loss
            else:
                loss = (1-args.aux_H_OOD_lambda)*loss + (args.aux_H_OOD_lambda)*aux_H_loss

            #print(aux_H_loss)
            #exit()

        # backpropagate and update optimizer learning rate unless doing inference
        if not args.infer_only:
            loss.backward(); opt.step()
            if args.back_lm: loss_lm.backward(); opt_lm.step()
            if args.back_mirror: loss_mirror.backward(); opt_mirror.step()
            if args.seq_gan:
                loss_gen.backward(); opt_gen.step()
                opt_disc.step()
        torch.cuda.empty_cache()

        # checkpoint model periodically
        if iterations % args.save_every == 0:
            if not args.infer_only:
                print("Saving Checkpoint")
                snapshot_prefix = os.path.join(args.save_path, 'snapshot')
                snapshot_path = snapshot_prefix + '_acc_{:.4f}_loss_{:.6f}_iter_{}_model.pt'.format(train_acc, loss.item(), iterations)
                torch.save(model, snapshot_path)
                for f in glob.glob(snapshot_prefix + '*'):
                    if f != snapshot_path:
                        os.remove(f)

        # evaluate performance on validation set periodically

        if (iterations % args.dev_every == 0 or args.infer_only) and (not args.no_dev_eval):
            print("Computing Post-Training Statistics")
            if args.ratioKldPrune or args.ratioEmdPrune:
                print("Computing Label Ratio")
                model.eval();train_iter_clone.init_epoch()
                if args.back_lm: lm_model.eval()
                if args.back_mirror: mirror_model.eval()
                for train_batch_idx, train_batch in enumerate(train_iter_clone):
                    for L in range(args.d_out):
                        isThisLabel = (train_batch.label == L).float()
                        isThisLabel = isThisLabel.to(train_batch.label.device)
                        k = torch.sum(isThisLabel)
                        if k.detach().cpu()>0:
                            model.label_ratio[L] = model.label_ratio[L]+k
                Z = sum(model.label_ratio)
                model.label_ratio = model.label_ratio/Z
                print("Finished Computing Label Ratio")

            if args.lof or args.ppnf:
                print("Constructing nearest neighbor index")
                reset_intermediate=False
                if not model.config.returnIntermediate:
                    model.config.returnIntermediate=True
                    reset_intermediate=True
                model.eval();train_iter_clone.init_epoch()
                intermediate_repn_list = []
                if args.ppnf:
                    label_index_list = []
                    answer_list = []

                with torch.no_grad():
                    for train_batch_idx, train_batch in enumerate(train_iter_clone):
                        answer, _, intermediate_repn = model(train_batch)
                        intermediate_repn_list.append(intermediate_repn.detach().cpu().numpy())
                        if args.ppnf:
                            label_index_list.append(train_batch.label.detach().cpu().numpy())
                            answer_list.append(answer.detach().cpu().numpy())

                if args.lof:
                    if len(intermediate_repn_list)>(args.lof_upper_lim+0.0)/args.batch_size:
                        print("Subsampling Because Index Would Be Too Large")
                        batches_to_select = int((args.lof_upper_lim+0.0)/args.batch_size)
                        intermediate_repn_list = random.sample(intermediate_repn_list,batches_to_select)

                if args.ppnf:
                    tupled_list = [(alpha,beta,gamma) for alpha,beta,gamma in zip(intermediate_repn_list,label_index_list,answer_list)]
                    total_example_count = len(tupled_list)
                    calibration_count = int(args.calib_frac * total_example_count)
                    print("Setting Aside ",calibration_count,"Examples For Calibration")
                    random.shuffle(tupled_list)
                    intermediate_repn_list_calibration = [x[0] for x in tupled_list[:calibration_count]]
                    label_index_list_calibration = [x[1] for x in tupled_list[:calibration_count]]
                    answer_list_calibration = [x[2] for x in tupled_list[:calibration_count]]
                    intermediate_repn_list = [x[0] for x in tupled_list[calibration_count:]]
                    label_index_list = [x[1] for x in tupled_list[calibration_count:]]

                intermediate_repn_list = np.concatenate(intermediate_repn_list, axis=0)
                if args.ppnf:
                    intermediate_repn_list_calibration = np.concatenate(intermediate_repn_list_calibration,axis=0)
                    label_index_list_calibration = np.concatenate(label_index_list_calibration, axis=0)
                if args.ppnf:
                    label_index_list = np.concatenate(label_index_list, axis=0)
                if args.ppnf:
                    answer_list_calibration = np.concatenate(answer_list_calibration, axis=0)
                if reset_intermediate: model.config.returnIntermediate=False
                print("Numpy Dump Size:",np.shape(intermediate_repn_list))
                print("Creating and Fitting NN Index")
                start_time=time.process_time()
                if args.ppnf:
                    from sklearn.neighbors import NearestNeighbors
                    model.neigh = NearestNeighbors(n_neighbors=5)
                    model.neigh.fit(intermediate_repn_list)
                elif args.lof:
                    from sklearn.neighbors import LocalOutlierFactor
                    model.neigh = LocalOutlierFactor(n_neighbors=20,contamination=args.contamination,novelty=True)
                    model.neigh.fit(intermediate_repn_list)
                end_time = time.process_time()
                print("Done Fitting Index")
                print("Time Taken To Fit Index:",end_time-start_time)
                if args.ppnf:
                    print("Computing Calibration Non Conformity")
                    disagreements_calibration = get_ppnf_disagreements(model,intermediate_repn_list_calibration,label_index_list,torch.tensor(answer_list_calibration),numpy=True)
                    print("Done Computing Calibration Non Conformity")
                    print(np.shape(disagreements_calibration))

            if args.mahalanobis or args.euclidean or args.manhattan:
                model.eval();train_iter_clone.init_epoch()
                if args.back_lm: lm_model.eval()
                if args.back_mirror: mirror_model.eval()
                print("Estimating Mahalanobis Means")
                for train_batch_idx, train_batch in enumerate(train_iter_clone):
                    answer, answerOOD, intermediateRepn = model(train_batch)
                    #Updating class-conditional mean
                    for L in range(args.d_out):
                        isThisLabel = (train_batch.label == L).float()
                        isThisLabel = isThisLabel.to(train_batch.label.device)
                        k = torch.sum(isThisLabel)
                        if k.detach().cpu()>0:
                            model.mu_maha_count[L] = model.mu_maha_count[L]+k
                            increment = torch.sum(intermediateRepn*isThisLabel.unsqueeze(dim=1),dim=0) - k*model.mu_maha[L,:]
                            increment = increment / model.mu_maha_count[L]
                            model.mu_maha[L,:] = model.mu_maha[L,:] + increment
                print("Estimating Mahalanobis Shared Covariance")
                model.eval(); train_iter_clone.init_epoch()
                N = torch.sum(model.mu_maha_count)
                if args.mahalanobis:
                    for train_batch_idx, train_batch in enumerate(train_iter_clone):
                        answer, answerOOD, intermediateRepn = model(train_batch)
                        respective_means = model.mu_maha[train_batch.label,:].detach()
                        intermediate_mean_shifted = (intermediateRepn - respective_means).detach()
                        a , b = intermediate_mean_shifted.size()[0] , intermediate_mean_shifted.size()[1]
                        intermediate_mean_shifted_outer_product = torch.bmm( intermediate_mean_shifted.view((a,b,1)), intermediate_mean_shifted.view((a,1,b)) )
                        if args.norm_covar_updates:
                            Z = model.mu_maha_count[train_batch.label].detach()
                            Z = args.d_out * (Z - 1)
                            Z = torch.reciprocal(Z)
                            Z = Z.view(-1,1,1)
                            intermediate_mean_shifted_outer_product = Z.expand_as(intermediate_mean_shifted_outer_product) * intermediate_mean_shifted_outer_product
                            intermediate_mean_shifted_outer_product = torch.sum(intermediate_mean_shifted_outer_product, dim=0)
                            model.sigma_maha_shared = model.sigma_maha_shared + intermediate_mean_shifted_outer_product
                        else:
                            intermediate_mean_shifted_outer_product = torch.sum(intermediate_mean_shifted_outer_product, dim=0)
                            model.sigma_maha_shared = model.sigma_maha_shared + (1.0/(N-1))*intermediate_mean_shifted_outer_product
                    model.sigma_maha_shared_inverse = torch.pinverse(model.sigma_maha_shared+args.covar_smooth)
                if args.id_covar:
                    model.sigma_maha_shared_inverse = torch.eye(model.sigma_maha_shared_inverse.size()[0], device=device)
            print("Finished Computing Post-Training Statistics")
            torch.cuda.empty_cache()
            print("Evaluation Validation Performance")
            # switch model to evaluation mode
            model.eval(); dev_iter.init_epoch()
            if args.back_lm: lm_model.eval()
            if args.back_mirror: mirror_model.eval()
            # calculate accuracy on validation set
            n_dev_correct, dev_loss = 0, 0
            if args.fore_lm:
                dev_perplexity, dev_token_count = 0.0, 0.0
            with torch.no_grad():
                dev_batch_pred_list, dev_batch_gt_list = [], []

                if args.hendrycks or args.entPrune or args.kldPrune or args.jsdPrune or args.ratioKldPrune or args.ratioEmdPrune or args.isdPrune or args.mahalanobis or args.euclidean or args.manhattan or args.emdPrune or args.marginal or args.lof or args.ppnf or args.avgY or args.gen_marginal or args.unimod or args.tentropy:
                    dev_batch_maxP_list, dev_batch_maxP_gt_list = [], []

                for dev_batch_idx, dev_batch in enumerate(dev_iter):
                     if args.stabilize:
                         #model.train()
                         answerList = []
                         for trial in range(10):
                             answer, answerOOD, intermediateRepn = model(dev_batch, addNoise=True)
                             answerList.append(answer)
                     else:
                        if args.generative:
                            if not args.fore_lm:
                                answer_old, word_scores = model(dev_batch, scramble=False)
                                del word_scores
                                answer = model.infer_generative(dev_batch,answer_old)
                                del answer_old
                                if args.scramble:
                                    answer_old_scramble, _ = model(dev_batch, scramble=args.scramble)
                                    answer_scramble = model.infer_generative(dev_batch,answer_old_scramble)
                                    del answer_old_scramble
                            else:
                                word_scores = model(dev_batch)
                            if args.back_lm: word_scores_lm = lm_model(dev_batch)
                            if args.back_mirror:
                                answer_old_mirror, word_scores_mirror = mirror_model(dev_batch)
                                del word_scores_mirror
                                answer_mirror = mirror_model.infer_generative(dev_batch,answer_old_mirror)
                                del answer_old_mirror
                        else:
                            if args.lof or args.ppnf:
                                current_return_intermediate = model.config.returnIntermediate
                                model.config.returnIntermediate = True
                            answer, answerOOD, intermediateRepn = model(dev_batch)
                            if args.lof or args.ppnf: model.config.returnIntermediate = current_return_intermediate
                     if args.hendrycks or args.entPrune or args.kldPrune or args.ratioKldPrune or args.ratioEmdPrune or args.jsdPrune  or args.emdPrune or args.isdPrune or args.mahalanobis or args.euclidean or args.manhattan or args.marginal or args.lof or args.ppnf or args.avgY or args.gen_marginal or args.unimod or args.tentropy:
                        T=args.oodT
                        if args.lof:
                            intermediateRepnNumpy = intermediateRepn.detach().cpu().numpy()
                            lof_result = -model.neigh.score_samples(intermediateRepnNumpy)
                            dev_batch_maxP = torch.tensor(lof_result)
                        elif args.ppnf:
                            intermediateRepnNumpy = intermediateRepn.detach().cpu().numpy()
                            start_time = time.process_time()
                            neigh_dist, neigh_ids = model.neigh.kneighbors(intermediateRepnNumpy)
                            print("Time Taken Per NN Query:",time.process_time()-start_time,dev_batch_idx)
                            ppnf_distance_mean = np.mean(neigh_dist,axis=1)
                            ppnf_labels = label_index_list[neigh_ids]
                            predicted_labels = torch.max(answer,1)[1].detach().cpu().numpy()
                            disagreements = (ppnf_labels!=predicted_labels[:,None]).astype(int)
                            if args.ppnf_w:
                                neigh_sims = np.reciprocal(neigh_dist+1e-8)
                                disagreements = neigh_sims * disagreements
                                disagreements = np.mean(disagreements,axis=1) * np.reciprocal(np.sum(neigh_sims,axis=1))
                            else:
                                disagreements = np.mean(disagreements,axis=1)
                            #print(np.shape(disagreements),np.shape(disagreements_calibration))
                            disagreements = np.greater(disagreements.reshape(-1,1),disagreements_calibration.reshape(1,-1)).astype(int)
                            #print(np.shape(disagreements))
                            disagreements = np.mean(disagreements,axis=1)
                            dev_batch_maxP = torch.tensor(disagreements)
                        elif args.gen_marginal:
                            with torch.no_grad():
                                logits = gen_model(dev_batch.hypothesis)/args.oodT
                                #print(logits.size())
                                loss_lm = F.cross_entropy(logits.contiguous().view(-1,args.n_embed),dev_batch.hypothesis.contiguous().view(-1),reduction='none')
                                is_not_pad_mask = (dev_batch.hypothesis.contiguous().view(-1)!=args.pad_id).float()
                                loss_lm = loss_lm * is_not_pad_mask
                                loss_lm = loss_lm.view(dev_batch.hypothesis.size()).sum(dim=0)
                                #print(loss_lm.size())
                                dev_gen_log_prob = loss_lm
                                #print(dev_gen_log_prob)

                                #exit()

                                log_probs_disc = disc_model(dev_batch.hypothesis,log_prob=True,T=args.oodT)
                                log_probs_disc = log_probs_disc[:,:,0]
                                is_not_pad_mask = (dev_batch.hypothesis.contiguous()!=args.pad_id).float()
                                log_probs_disc = log_probs_disc*is_not_pad_mask + (1-is_not_pad_mask)*(-3e5)
                                log_probs_disc = torch.logsumexp(log_probs_disc,dim=0) - torch.log(torch.sum(is_not_pad_mask,dim=0))
                                #log_probs_disc = torch.mean(log_probs_disc,dim=0)
                                #log_probs_disc = logs_prob_disc[-1,:]
                                #dev_prob = F.log_softmax(answer/T+1e-8,dim=1)
                                #dev_prob =  F.softmax(answer/T+1e-8,dim=1)+F.softmax(2*answer/T+1e-8,dim=1)+F.softmax(4*answer/T+1e-8,dim=1)+F.softmax(8*answer/T+1e-8,dim=1)+F.softmax(16*answer/T+1e-8,dim=1)
                                #dev_prob = dev_prob + F.softmax(32*answer/T+1e-8,dim=1)
                                from sparsemax import Sparsemax
                                sparsemax = Sparsemax(dim=1)
                                dev_prob = sparsemax(answer/T+1e-8)

                                dev_batch_maxP = -torch.max(dev_prob,1)[0].view(dev_batch.label.size())
                                dev_prob_complement = torch.log(1-torch.max(F.softmax(answer/T+1e-8,dim=1),1)[0].view(dev_batch.label.size()))

                                print(dev_batch_maxP[0:3])
                                print((1.0/args.n_embed)*dev_gen_log_prob[0:3])
                                print(log_probs_disc[0:3])
                                dev_gen_log_prob =  dev_batch_maxP #2*dev_prob_complement + (1.0/args.n_embed)*dev_gen_log_prob + log_probs_disc

                                dev_batch_maxP = (dev_gen_log_prob).cpu()
                        elif args.marginal:
                            #dev_per_class_terms = torch.exp(answer[:,1:])
                            #dev_prob = torch.sum(dev_per_class_terms,dim=1)
                            #print("Computing Directly")
                            #print(dev_prob[0:2])
                            if args.back_lm:
                                loss_lm = F.cross_entropy(word_scores_lm.contiguous().view(-1,args.n_embed),dev_batch.hypothesis.contiguous().view(-1),reduction='none')
                                is_not_pad_mask = (dev_batch.hypothesis.contiguous().view(-1)!=args.pad_id).float()
                                loss_lm = loss_lm * is_not_pad_mask
                                loss_lm = loss_lm.view(dev_batch.hypothesis.size()).sum(dim=0)
                                dev_back_log_prob = -loss_lm
                            elif args.back_mirror:
                                dev_log_prob_mirror = torch.logsumexp(answer_mirror[:,1:],dim=1)
                            if not args.fore_lm:
                                dev_log_prob = torch.logsumexp(answer[:,1:],dim=1)
                                if args.scramble: dev_log_prob_scramble = torch.logsumexp(answer_scramble[:,1:],dim=1)
                            else:
                                loss = F.cross_entropy(word_scores.contiguous().view(-1,args.n_embed),dev_batch.hypothesis.contiguous().view(-1),reduction='none')
                                is_not_pad_mask = (dev_batch.hypothesis.contiguous().view(-1)!=args.pad_id).float()
                                dev_token_count += sum(is_not_pad_mask)
                                loss = loss * is_not_pad_mask
                                loss = loss.view(dev_batch.hypothesis.size()).sum(dim=0)
                                dev_perplexity += loss.sum(dim=0).detach().cpu().item()
                                dev_log_prob = -loss
                            if args.back_lm:
                                #print(dev_back_log_prob)
                                #print(dev_log_prob)
                                dev_log_prob = dev_log_prob - dev_back_log_prob
                                #print(dev_log_prob)
                            elif args.back_mirror:
                                #print("Mirror Prob:",dev_log_prob_mirror)
                                #print("Original:",dev_log_prob)
                                dev_log_prob = dev_log_prob - dev_log_prob_mirror
                                #print("Updated:",dev_log_prob)
                            if args.scramble:
                                if not (args.back_lm or args.back_mirror):
                                    if args.just_scramble:
                                        dev_log_prob = -dev_log_prob_scramble
                                    else:
                                        dev_log_prob = dev_log_prob - dev_log_prob_scramble
                                elif args.back_lm:
                                    dev_log_prob = dev_log_prob - 0.5*dev_log_prob_scramble + 0.5*dev_back_log_prob
                                elif args.back_mirror:
                                    dev_log_prob = dev_log_prob - 0.5*dev_log_prob_scramble + 0.5*dev_log_prob_mirror
                            dev_batch_maxP = -dev_log_prob.cpu()
                        elif args.euclidean or args.manhattan:
                            batch_size = intermediateRepn.size()[0]
                            label_size = args.d_out
                            intermediateRepn_per_label_mean_shifted = intermediateRepn.unsqueeze(dim=1)-model.mu_maha.unsqueeze(dim=0)
                            if args.euclidean:
                                intermediateRepn_per_label_mean_shifted_collapsed = intermediateRepn_per_label_mean_shifted.view((batch_size*label_size,-1))
                                a, b = batch_size*label_size , intermediateRepn.size()[1]
                                eucl_distances_accumulated = torch.bmm(intermediateRepn_per_label_mean_shifted_collapsed.view(a,1,b),intermediateRepn_per_label_mean_shifted_collapsed.view(a,b,1))
                            elif args.manhattan:
                                eucl_distances_accumulated = torch.norm(intermediateRepn_per_label_mean_shifted,p=1,dim=2)
                            eucl_distances_accumulated = eucl_distances_accumulated.view((batch_size,label_size))
                            eucl_similarities_accumulated = - eucl_distances_accumulated

                            if args.use_dis_prob:
                                dev_prob = F.softmax(answer,dim=1)
                                dev_maxP_index = torch.max(dev_prob,1)[1]
                                dev_batch_maxP = -eucl_similarities_accumulated.gather(1,dev_maxP_index.view(-1,1))
                            else:
                                dev_batch_maxP = -torch.max(eucl_similarities_accumulated, dim=1)[0]
                            dev_batch_maxP = dev_batch_maxP.cpu()
                        elif args.mahalanobis:
                            batch_size = intermediateRepn.size()[0]
                            label_size = args.d_out

                            intermediateRepn_per_label_mean_shifted = intermediateRepn.unsqueeze(dim=1)-model.mu_maha.unsqueeze(dim=0)

                            maha_distances_intermediate = torch.matmul(intermediateRepn_per_label_mean_shifted, model.sigma_maha_shared_inverse)
                            maha_distances_intermediate_collapsed = maha_distances_intermediate.view((batch_size*label_size,-1))
                            intermediateRepn_per_label_mean_shifted_collapsed = intermediateRepn_per_label_mean_shifted.view((batch_size*label_size,-1))
                            a, b = batch_size*label_size , intermediateRepn.size()[1]
                            maha_distances_accumulated = torch.bmm(maha_distances_intermediate_collapsed.view((a,1,b)), intermediateRepn_per_label_mean_shifted_collapsed.view((a,b,1)))
                            maha_distances_accumulated = maha_distances_accumulated.view((batch_size,label_size))

                            maha_similarities_accumulated = - maha_distances_accumulated

                            if args.use_dis_prob:
                                dev_prob = F.softmax(answer,dim=1)
                                dev_maxP_index = torch.max(dev_prob,1)[1]
                                dev_batch_maxP = -maha_similarities_accumulated.gather(1,dev_maxP_index.view(-1,1))
                            else:
                                dev_batch_maxP = -torch.max(maha_similarities_accumulated, dim=1)[0]
                            dev_batch_maxP = dev_batch_maxP.cpu()
                        elif args.hendrycks:
                            #answer = answer[:,1:]
                            if args.stabilize:
                                dev_prob = torch.zeros(answerList[0].size(),device=answerList[0].device)
                                probList = []
                                for answer in answerList:
                                    probList.append(F.softmax(answer/T,dim=1))
                                    dev_prob += 0.1*probList[-1]
                                #variance = torch.zeros(answerList[0].size(),device=answerList[0].device)
                                #for prob in probList:
                                #    variance += 0.1*((prob-dev_prob)*(prob-dev_prob))
                                #dev_prob = dev_prob - variance
                            else:
                                dev_prob = F.log_softmax(answer/T,dim=1)
                            dev_batch_maxP = 1-torch.max(dev_prob,1)[0].view(dev_batch.label.size()).cpu()
                            if args.altCorrection:
                                #dev_batch_second_maxP = torch.topk(dev_prob,k=2)[0][:,1].cpu() / 2.0
                                dev_batch_maxP = dev_batch_maxP + (1-dev_batch_maxP)*(1-dev_batch_maxP)/2.0
                        elif args.tentropy:
                            from tentropy import compute_tentropy
                            tentropies = []
                            for i in range(answer.size()[0]):
                                tentropy_i_base = compute_tentropy(answer[i],0.99)
                                tentropy_i_forward = compute_tentropy(answer[i],0.90)
                                tentropy_i = tentropy_i_forward-tentropy_i_base
                                tentropies.append(tentropy_i)
                            tentropies = torch.tensor(tentropies,device=answer.device)
                            dev_batch_maxP = (-tentropies).detach().cpu()
                        elif args.unimod:
                            dev_prob = F.softmax(answer/T+1e-8,dim=1)
                            dev_batch_maxP = 1-torch.max(dev_prob,1)[0].view(dev_batch.label.size())
                            #new_Z = (1-dev_batch_maxP) + dev_batch_maxP / (args.d_out-2)
                            #dev_batch_maxP = dev_batch_maxP / (args.d_out-2)
                            #dev_batch_maxP = dev_batch_maxP * torch.reciprocal(new_Z)
                            dev_batch_maxP = -dev_batch_maxP*torch.log(dev_batch_maxP+1e-8)-(1-dev_batch_maxP)*torch.log(1-dev_batch_maxP+1e-8)
                            dev_batch_maxP = dev_batch_maxP.cpu()
                        elif args.avgY:
                            dev_average_logit = torch.mean(answer,dim=1) - torch.sum(F.softmax(answer+1e-8,dim=1)*(answer),dim=1)
                            dev_batch_maxP = dev_average_logit.cpu()
                        elif args.entPrune:
                            dev_log_prob = F.log_softmax(answer/T, dim=1)
                            dev_prob = F.softmax(answer/T, dim=1)
                            dev_plogp = dev_prob * dev_log_prob
                            dev_H = torch.sum(-dev_plogp , dim=1)
                            dev_batch_maxP = dev_H.cpu()
                        elif args.kldPrune:
                            dev_log_prob = F.log_softmax(answer[:,1:]/T, dim=1)
                            dev_kld = -torch.mean(dev_log_prob , dim=1) - math.log(args.d_out)
                            dev_batch_maxP = (-dev_kld).cpu()
                        elif args.isdPrune:
                             dev_log_prob = F.log_softmax(answer/T, dim=1)
                             dev_prob = F.softmax(answer/T, dim=1)
                             dev_isd = torch.sum(args.d_out * dev_prob - dev_log_prob, dim=1) - args.d_out*math.log(args.d_out) - args.d_out
                             dev_batch_maxP = (-dev_isd).cpu()
                        elif args.ratioKldPrune:
                             dev_log_prob = F.log_softmax(answer/T, dim=1)
                             dev_kld_trailing_term = -torch.sum(model.label_ratio.squeeze(0).expand_as(dev_log_prob) * dev_log_prob, dim=1)
                             dev_kld_leading_term =  torch.sum(torch.log(model.label_ratio+1e-8)*model.label_ratio)
                             dev_kld = dev_kld_leading_term + dev_kld_trailing_term
                             dev_batch_maxP = (-dev_kld).cpu()
                        elif args.jsdPrune:
                            dev_M_prob = 0.5*F.softmax(answer/T,dim=1)+0.5*(1.0/args.d_out)
                            dev_log_prob = F.log_softmax(answer/T, dim=1)
                            dev_prob = F.softmax(answer/T, dim=1)
                            dev_plogp_sum = torch.sum(dev_prob * dev_log_prob, dim=1)
                            dev_M_log_prob = torch.log(dev_M_prob+1e-8)
                            dev_M_plogp_sum = torch.sum(dev_M_prob * dev_M_log_prob, dim=1)
                            dev_jsd = 0.5*(dev_plogp_sum-math.log(args.d_out)) - dev_M_plogp_sum
                            dev_batch_maxP = (-dev_jsd).cpu()
                        elif args.emdPrune or args.ratioEmdPrune:
                            dev_prob = F.softmax(answer[:,1:]/T+1e-8, dim=1)
                            if args.emdPrune:
                                dev_prob_uniform = (1.0/(args.d_out-1))*torch.ones(dev_prob.size())
                            elif args.ratioEmdPrune:
                                dev_prob_uniform = model.label_ratio[1:].unsqueeze(dim=0).expand_as(dev_prob)
                            if True:
                                max_label = torch.max(dev_prob,1)[1]
                                max_onehot_label = (torch.FloatTensor(max_label.size(0), dev_prob.size(1)).zero_().scatter_(1,max_label.unsqueeze(1).cpu().data,1)).to(max_label.device)
                                dev_prob_uniform = max_onehot_label

                            dev_prob_uniform = dev_prob_uniform.to(dev_prob.device)
                            cost = torch.ones((args.d_out-1,args.d_out-1))-torch.eye(args.d_out-1)
                            cost = cost * (1.0/(args.d_out-1))
                            cost = cost.to(dev_prob.device)
                            emdComputer = WassersteinLossStab(cost,sinkhorn_iter=100)
                            dev_emd = emdComputer(dev_prob,dev_prob_uniform)
                            uniform_distr = (1.0/(args.d_out-1))*torch.ones(dev_prob.size()).to(dev_prob.device)
                            dev_emd += emdComputer(uniform_distr,dev_prob)

                            dev_batch_maxP = (-dev_emd).cpu()

                            dev_batch_maxP = -dev_batch_maxP

                        dev_batch_gt = dev_batch.label.cpu()
                        dev_batch_maxP_list.append(dev_batch_maxP)
                        dev_batch_maxP_gt_list.append(dev_batch_gt)
                     if not args.fore_lm:
                        dev_batch_pred = torch.max(answer, 1)[1].view(dev_batch.label.size()).cpu()
                     else:
                        dev_batch_pred = torch.ones(dev_batch.label.size(),dtype=dev_batch.label.dtype)
                     dev_batch_pred_list.append(dev_batch_pred)
                     dev_batch_gt = dev_batch.label.cpu()
                     dev_batch_gt_list.append(dev_batch_gt)
                     if not args.fore_lm:
                        n_dev_correct += (torch.max(answer, 1)[1].view(dev_batch.label.size()) == dev_batch.label).sum().item()
                     else:
                        n_dev_correct += 0
                     if not args.fore_lm:
                         if args.mos == False:
                            dev_loss = criterion(answer, dev_batch.label)
                         else:
                            dev_loss = F.nll_loss(answer, dev_batch.label)
                dev_pred_list = torch.cat(dev_batch_pred_list).numpy()
                dev_gt_list = torch.cat(dev_batch_gt_list).numpy()
                if args.hendrycks or args.entPrune or args.mahalanobis or args.euclidean or args.manhattan or args.kldPrune or args.jsdPrune or args.emdPrune or args.ratioKldPrune or args.ratioEmdPrune or args.isdPrune or args.marginal or args.lof or args.ppnf or args.avgY or args.gen_marginal or args.unimod or args.tentropy:
                    dev_maxP_list = torch.cat(dev_batch_maxP_list).numpy()
                    dev_maxP_gt_list = torch.cat(dev_batch_maxP_gt_list).numpy()
                    dev_maxP_gt_list = (dev_maxP_gt_list == 0).astype(int)
                    #dev_maxP_gt_list = (dev_maxP_gt_list < 0).astype(int)
                    print("OODCount:",sum(dev_maxP_gt_list))
                    print(dev_maxP_gt_list)

            dev_acc = 100. * n_dev_correct / len(dev)
            if args.fore_lm:
                dev_perplexity = dev_perplexity / dev_token_count
                dev_perplexity = math.exp(dev_perplexity)
                print("Dev Perplexity:",dev_perplexity)

            #Threshold class comes in here
            if args.hendrycks or args.entPrune or args.kldPrune or args.jsdPrune or args.emdPrune or args.ratioKldPrune or args.ratioEmdPrune or args.isdPrune or args.mahalanobis or args.euclidean or args.manhattan or args.marginal or args.lof or args.ppnf or args.avgY or args.gen_marginal or args.unimod or args.tentropy:
                #TODO: Ship this to the module oodMetrics
                oodMetricsObj = oodMetrics()
                oodMetricsObj.compute_all_metrics(dev_maxP_gt_list, dev_maxP_list, choose_thresholds=True)
                oodMetricsObj.pretty_print_metrics()
                oodMetricsObj.plot_PR_curve("validPlots/"+args.save_path+".png")
                print("Saving OOD Scores Dump")
                ood_scores_path = os.path.join(args.save_path,"ood_scores.p")
                pickle.dump([dev_maxP_gt_list,dev_maxP_list],open(ood_scores_path,"wb"))

            torch.cuda.empty_cache()
            if not args.fore_lm:
                test_iter.init_epoch()
                n_test_correct, test_loss = 0, 0
                with torch.no_grad():
                    test_batch_pred_list, test_batch_gt_list = [], []
                    for test_batch_idx, test_batch in enumerate(test_iter):
                         if args.generative:
                            answer_old, word_scores = model(test_batch)
                            del word_scores
                            answer = model.infer_generative(test_batch,answer_old)
                            del answer_old
                         else:
                            answer, answerOOD, intermediateRepn = model(test_batch)
                         test_batch_pred = torch.max(answer, 1)[1].view(test_batch.label.size()).cpu()
                         test_batch_pred_list.append(test_batch_pred)
                         test_batch_gt = test_batch.label.cpu()
                         test_batch_gt_list.append(test_batch_gt)
                         n_test_correct += (torch.max(answer, 1)[1].view(test_batch.label.size()) == test_batch.label).sum().item()
                         if args.mos == False:
                            test_loss = criterion(answer, test_batch.label)
                         else:
                            test_loss = F.nll_loss(answer, test_batch.label)
                    test_pred_list=torch.cat(test_batch_pred_list).numpy()
                    test_gt_list=torch.cat(test_batch_gt_list).numpy()
                test_acc = 100. * n_test_correct / len(test)

            from sklearn.metrics import f1_score, precision_recall_fscore_support

            print("Label Space:",answers.vocab.itos)

            dev_macro_f1_score=f1_score(dev_gt_list, dev_pred_list, average="macro")
            print("Accuracy:",dev_acc)
            print("Macro F1 Score [Single Number]:",dev_macro_f1_score)
            precision_recall_fscore_support_statistics = precision_recall_fscore_support(dev_gt_list, dev_pred_list , average=None)

            precision_recall_fscore_support_statistics_dict = {}
            for key,val in enumerate(answers.vocab.itos):
                precision_recall_fscore_support_statistics_dict[val]={}
                precision_recall_fscore_support_statistics_dict[val]["Precision"]=precision_recall_fscore_support_statistics[0][key]
                precision_recall_fscore_support_statistics_dict[val]["Recall"]=precision_recall_fscore_support_statistics[1][key]
                precision_recall_fscore_support_statistics_dict[val]["F1 Score"]=precision_recall_fscore_support_statistics[2][key]
                precision_recall_fscore_support_statistics_dict[val]["Support"]=precision_recall_fscore_support_statistics[3][key]

            print("Per Class F1 Statistics")
            print(padUpToLength("Class"),"\t","Precision","\t","Recall","\t","F1 Score","\t","Support")
            for key,val in precision_recall_fscore_support_statistics_dict.items():
                print(padUpToLength(key),"\t",val["Precision"],"\t",val["Recall"],"\t",val["F1 Score"],"\t",val["Support"])

            if not args.fore_lm:
                print(dev_log_template.format(time.time()-start,
                    epoch, iterations, 1+batch_idx, len(train_iter),
                    100. * (1+batch_idx) / len(train_iter), loss.item(), dev_loss.item(), train_acc, dev_acc))

                test_macro_f1_score=f1_score(test_gt_list, test_pred_list, average="macro")
                print("Test Accuracy:",test_acc)
                print("Test Macro F1 Score:",test_macro_f1_score)
            #precision_recall_fscore_support_statistics = precision_recall_fscore_support(dev_gt_list, dev_pred_list, average=None)
            if args.infer_only:
                print("Inference Concluded. Exiting Process")
                exit()
            # update best valiation set accuracy
            #if dev_acc > best_dev_acc:
            if dev_macro_f1_score > best_dev_macro_f1_score and not args.fore_lm:

                # found a model with better validation set accuracy

                #best_dev_acc = dev_acc
                best_dev_macro_f1_score = dev_macro_f1_score
                if not args.infer_only:
                    print("Updating Best Snapshot")
                    snapshot_prefix = os.path.join(args.save_path, 'best_snapshot')
                    snapshot_path = snapshot_prefix+".pt"
                    #snapshot_path = snapshot_prefix + '_devacc_{}_devloss_{}__iter_{}_model.pt'.format(dev_acc, dev_loss.item(), iterations)

                    # save model, delete previous 'best_snapshot' files
                    torch.save(model, snapshot_path)

                    for f in glob.glob(snapshot_prefix + '*'):
                        if f != snapshot_path:
                            os.remove(f)
                    if args.back_mirror:
                        snapshot_path_mirror =  snapshot_path[:-3]+"_mirror"+snapshot_path[-3:]
                        torch.save(mirror_model,snapshot_path_mirror)
                    if args.back_lm:
                        snapshot_path_lm =  snapshot_path[:-3]+"_lm"+snapshot_path[-3:]
                        torch.save(lm_model,snapshot_path_lm)
                    if args.seq_gan:
                        snapshot_path_gen =  snapshot_path[:-3]+"_gen"+snapshot_path[-3:]
                        torch.save(gen_model,snapshot_path_gen)
                        snapshot_path_disc =  snapshot_path[:-3]+"_disc"+snapshot_path[-3:]
                        torch.save(disc_model,snapshot_path_disc)

        elif iterations % args.log_every == 0:

            # print progress message
            print(log_template.format(time.time()-start,
                epoch, iterations, 1+batch_idx, len(train_iter),
                100. * (1+batch_idx) / len(train_iter), loss.item(), ' '*8, n_correct/n_total*100, ' '*12))


    if args.fore_lm:
        perplexity = perplexity/token_count
        perplexity = math.exp(perplexity)
        print("Perplexity:",perplexity)
