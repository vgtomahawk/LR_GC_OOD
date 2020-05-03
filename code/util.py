import os
from argparse import ArgumentParser

def makedirs(name):
    """helper function for python 2 and 3 to call os.makedirs()
       avoiding an error if the directory to be created already exists"""

    import os, errno

    try:
        os.makedirs(name)
    except OSError as ex:
        if ex.errno == errno.EEXIST and os.path.isdir(name):
            # ignore existing directory
            pass
        else:
            # a different error happened
            raise


def get_args():
    parser = ArgumentParser(description='PyTorch/torchtext SNLI example')
    #Number of epochs to train the model for at the maximum
    parser.add_argument('--epochs', type=int, default=5)
    #Batch_size to use. Generally, 128 for the discriminative case and 32 for the generative case. On small datasets like atis and snips. always use 8
    parser.add_argument('--batch_size', type=int, default=128)
    #Per-vector dimension  of the embedding matrix
    parser.add_argument('--d_embed', type=int, default=100)
    parser.add_argument('--d_proj', type=int, default=300)
    #Dimension of the encoder lstm
    parser.add_argument('--d_hidden', type=int, default=300)
    #Number of layers in the LSTM encoder
    parser.add_argument('--n_layers', type=int, default=1)
    #Random seed to use with random, numpy, pytorch, pytorchCuda
    parser.add_argument('--seed', type=int, default=4242)
    #Ids used for pad and eos in the torchtext vocabularies
    parser.add_argument('--pad_id', type=int,default=1)
    parser.add_argument('--eos_id',type=int,default=2)
    #Learning rate for optimizer [we almost always have ADAM]
    parser.add_argument('--lr', type=float, default=.001)
    #Logging and checkpointing frequencies
    parser.add_argument('--log_every', type=int, default=50)
    parser.add_argument('--dev_every', type=int, default=1000)
    parser.add_argument('--save_every', type=int, default=1000)
    parser.add_argument('--dp_ratio', type=int, default=0.2)
    #Use unidirectional LSTM. Note that this is compulsory when --generative is on, since you can't have a bidirectional encoder giving P(x|y)
    parser.add_argument('--no-bidirectional', action='store_false', dest='birnn')
    parser.add_argument('--preserve-case', action='store_false', dest='lower')
    parser.add_argument('--no-projection', action='store_false', dest='projection')
    parser.add_argument('--train_embed', action='store_false', dest='fix_emb')
    parser.add_argument('--aux_BCE_OOD', action='store_true')
    parser.add_argument('--aux_BCE_OOD_lambda', type=float, default=0.1)
    parser.add_argument('--no_CE_for_OOD',action='store_true')
    parser.add_argument('--aux_H_OOD', action='store_true')
    parser.add_argument('--aux_H_OOD_lambda', type=float, default=0.1)

    #Arguments related to cosine_softmax
    parser.add_argument('--margin', type=float, default=0.15)
    parser.add_argument('--cosine_softmax', action='store_true')


    
    #IGNORE Please ignore the arguments in the small group below
    parser.add_argument('--inhibited_softmax', action='store_true')
    parser.add_argument('--mos', action='store_true')
    parser.add_argument('--no_ood_in_half', action='store_true')
    parser.add_argument('--mos_k', type=int, default=3)
    parser.add_argument('--multi_margin_softmax', action='store_true')
    parser.add_argument('--additive_lambda', action='store_true')
    #IGNORE Turn off bias in softmax layer
    parser.add_argument('--bias_false',action='store_true')
    parser.add_argument('--conf_teach',action='store_true')
    parser.add_argument('--conf_match',action='store_true')
    parser.add_argument('--conf_mse',action='store_true')
    parser.add_argument('--embed_out_shared',action='store_true')
    parser.add_argument('--short_circuit',action='store_true')
    parser.add_argument('--short_circuit_in_half',action='store_true')
    parser.add_argument('--conf_teach_beta', type=float, default=0.01)
    #IGNORE Not used
    parser.add_argument('--ood_index',type=int, default=3)


    #Arguments related to loading and unloading of models, checkpoints, devices to use etc
    #Which gpu device to use
    parser.add_argument('--gpu', type=int, default=0)
    #Location at which checkpoint is saved as best_snapshot.pt
    parser.add_argument('--save_path', type=str, default='resultsPlain')
    # Cached word vectors to prevent repeated loading
    parser.add_argument('--vector_cache', type=str, default=os.path.join(os.getcwd(), '.vector_cache/input_vectors.pt'))
    # Whether to use word vectors. With the default empty string, this option doesn't come into play.
    parser.add_argument('--word_vectors', type=str, default='')
    #Start with the model already trained and saved at the .pt file present at this location.
    parser.add_argument('--resume_snapshot', type=str, default='')



    #IGNORE Uses the hidden state directly without using additional intermediate linear+non-linear projection layers
    parser.add_argument('--short_circuit_main',action='store_true')
    #Return the intermediate representation from before the linear layer of the softmax. This is required for the --euclidean, --manhattan, --mahalanobis methods
    parser.add_argument('--returnIntermediate', action='store_true')

    #Trains the generative classifier instead of a discriminative one
    parser.add_argument('--generative',action='store_true')
    #Concatenate label embedding to the hidden state
    parser.add_argument('--at_start', action='store_true')
    #Concatenate label embediding to the LSTM hidden state for the generative model's LSTM [while modelling P(x|y)]
    parser.add_argument('--at_hidden',action='store_true')
    #Use a background model which is a LM to correct the marginal likelihood of the forward model used, be it a generative classifier or a LM itself.
    parser.add_argument('--back_lm',action='store_true')
    parser.add_argument('--back_lm_diff',action='store_true')
    #Use a language model as the forward model instead of a generative or discriminative classifier. No checkpoint is saved in this case because its not clear how to do model selection for this.
    parser.add_argument('--fore_lm',action='store_true')
    #Use a mirror [of the generative model] as the background model. Note that this only makes sense in conjunction with --generative. Also, --corrupt_back should be on when this is on for it to be sensible.
    parser.add_argument('--back_mirror',action='store_true')
    #Corrupt the background model. Used in conjunction with --back_lm and --back_mirror. Note that --back_mirror makes no sense when used without this.
    parser.add_argument('--corrupt_back',action='store_true')
    # Very Very Undercapacity LM. Note that this is not trained using noised inputs; rather its own under-capacity is expected to lead a more "spread out" distribution which upweights noise
    parser.add_argument('--vvul',action='store_true')
    # Size of the Very Very Undercapacity LM. Note that this should typically not exceed 50
    parser.add_argument('--vvul_size',type=int,default=10)
    #Size of background LM and forward LM hidden states/embeddings. Useful only in the case when either --fore_lm OR --back_lm [or both] are on.
    parser.add_argument('--back_input_size',type=int,default=128)
    # Other options: "unigram" and "uniroot". Generally both of these work better than uniform
    # Noise distribution to be used to sample words to replace and perturb training inputs, on which the background model is trained
    # Note that using this makes sense only when --back_lm or --back_mirror are on
    parser.add_argument('--noise_type',type=str,default="uniform")
    # Probability with which input token is replaced with randomly chosen word from noise distribution.
    parser.add_argument('--noise_level',type=float,default=0.5)

    #Arguments related to the dataset
    parser.add_argument('--unsup', action='store_true')             #Unsupervised mode is turned on. This argument should always be given for the experiments in the paper.
    parser.add_argument('--dataset', type=str, default="assistant") #Options: fbrelease, fbreleasecoarse, snips, atis
    parser.add_argument('--super_root', type=str, default="pytorchHack/") #Path of the home directory
    parser.add_argument('--frac', type=int, default=50)             #This argument is not used. Fraction of OOD training points to use. Only used when args.dataset=="assistant_frac"
    parser.add_argument('--id_ratio', type=float, default=0.75)     #In-domain ratio. This is used mainly as an input for atis and snips since we have data splits with varying in-domain ratios
    parser.add_argument('--split_id', type=int, default=0)          #Split Id .  This is used for atis and snips since there are multiple ways of holding some classes as out of domain through the experiment
    #Arguments related to inference
    parser.add_argument('--infer_only', action='store_true')        #The given argument to resume_snapshot is used to load a model, and inference performed over the validation set. NO Training if this switch is on.
    #Defunct argument. Run in --debug mode. DO NOT USE THIS ARGUMENT
    parser.add_argument('--debug', action='store_true')


    #Arguments related to options for unsupervised OOD detection on the validation set.
    #Note that this will only work when the flag --unsup is True
    #Validation set is expected to contain an "outOfDomain" label for such instances

    #Method of using 1-maxP as OOD Score
    parser.add_argument('--hendrycks', action='store_true')
    parser.add_argument('--unimod', action='store_true')
    #IGNORE
    parser.add_argument('--tentropy', action='store_true')
    #Temperature to use to compute softmax probabilities at inference time. Mainly used to increase temperature in conjunction with Hendrycks method
    parser.add_argument('--oodT', type=float, default=1000)
    #IGNORE Temperature used to compute softmax probabilities at traing time. This basically matters only in the GAN setting if at all.
    parser.add_argument('--genT', type=float, default=1)
    #The below two args are not shown to be useful. Refrain from using them.
    #IGNORE Keeps the train mode on and does inference 10 times per example. This mainly is a smoothing over different dropout probabilities
    parser.add_argument('--stabilize', action='store_true')
    #IGNORE Correction to 1-p, 1-p+p**2/2 ... ;
    parser.add_argument('--altCorrection', action='store_true')

    #Uses Euclidean distance to closest class conditional Gaussian mean [from the training set] as the OOD Score
    parser.add_argument('--euclidean', action='store_true')
    #Uses Manhattan distance to closest class conditional Gaussian mean [from the training set] as the OOD Score
    parser.add_argument('--manhattan', action='store_true')
    #Uses Mahalanobis distance to closest class conditional Gaussian mean [from the training set] as the OOD Score
    parser.add_argument('--mahalanobis', action='store_true')
    #Normalize updates to the shared covariance matrix being estimated at run-time, based on class size. This would ensure classes influence this in a more equitable way
    parser.add_argument('--norm_covar_updates', action='store_true')


    parser.add_argument('--entThreshold', action='store_true')

    #Use entropy as the OOD Score
    parser.add_argument('--entPrune', action='store_true')
    #Use KLD between U(y) and P(y|x) as the OOD Score. U(y) is the uniform distribution over labels
    parser.add_argument('--kldPrune', action='store_true')
    #Use ISD between U(y) and P(y|x) as the OOD Score. ISD = Itikura Saito Divergence
    parser.add_argument('--isdPrune', action='store_true')
    #Use KLD between R(y) and P(y|x) as the OOD Score. R(y) is the label frequency distribution over the training set.
    parser.add_argument('--ratioKldPrune', action='store_true')
    #Use EMD betwen R(y) and P(y|x) as the OOD Score. R(y) is the label frequency distribution over the training set.
    parser.add_argument('--ratioEmdPrune', action='store_true')
    #Use Jensen Shannon Divergence between U(y) and P(y|x) as the OOD Score
    parser.add_argument('--jsdPrune', action='store_true')
    #Use Earth Mover's Distance between U(y) and P(y|x) as the OOD Score
    parser.add_argument('--emdPrune', action='store_true')

    #Use 1 - P_{X}(x) as the OOD Score. P here is the marginal likelihood of either the generative classifier [if --generative is on] or the LM [if --fore_lm is on]
    #In the latter case of course, the term marginal is overloaded [there is nothing to marginalize over], but using the term nevertheless
    parser.add_argument('--marginal', action='store_true')
    parser.add_argument('--gen_marginal',action='store_true')
    parser.add_argument('--scramble',action='store_true')
    parser.add_argument('--just_scramble',action='store_true')
    parser.add_argument('--avgY',action='store_true')
    parser.add_argument('--zero_exclude',action='store_true')

    #Local Outlier Factor
    parser.add_argument('--lof',action='store_true')
    parser.add_argument('--contamination',type=float,default=0.1)
    parser.add_argument('--ppnf',action='store_true')
    parser.add_argument('--ppnf_w', action='store_true')
    parser.add_argument('--lof_upper_lim', type=int, default=50000)
    parser.add_argument('--calib_frac', type=float, default=0.1)
    parser.add_argument('--use_dis_prob', action='store_true')
    parser.add_argument('--id_covar', action='store_true')
    parser.add_argument('--covar_smooth',type=float, default=0.0)

    # IGNORE Generator arguments
    parser.add_argument('--seq_gan',action='store_true')
    parser.add_argument('--seq_gan_class',action='store_true')
    parser.add_argument('--seq_gan_class_frozen',action='store_true')
    parser.add_argument('--bound_class_from_gen',action='store_true')
    parser.add_argument('--seq_gan_class_beta',type=float,default=0.07)
    parser.add_argument('--seq_gan_aux_clip',type=float,default=0.0)
    parser.add_argument('--seq_gan_class_zeta',type=float,default=-1)   #Growth rate
    parser.add_argument('--seq_gan_class_omega',type=float,default=-1)   #Decay rate
    parser.add_argument('--seq_gan_class_beta_cap',type=float,default=1.0) #Max limit
    parser.add_argument('--gen_reverse_H',action='store_true')
    parser.add_argument('--gen_square_H',action='store_true')
    parser.add_argument('--gen_emd_H',action='store_true')
    parser.add_argument('--gen_d_embed',type=int,default=256)
    parser.add_argument('--gen_init_from_embed',action='store_true')
    parser.add_argument('--gen_hidden_size',type=int,default=256)
    parser.add_argument('--gen_n_layers',type=int,default=1)
    parser.add_argument('--gen_enc_type',type=str,default="plain")
    parser.add_argument('--gen_dp_ratio',type=float,default=0.2)
    parser.add_argument('--gen_share_out_embed',action='store_true')
    parser.add_argument('--max_gen_sample_L',type=int,default=55)
    parser.add_argument('--gen_gamma',type=float,default=0.91)
    parser.add_argument('--gen_init_type',type=str,default="random_smoothed")
    parser.add_argument('--gen_reduction',type=str,default="full_mean")
    parser.add_argument('--gen_base_type',type=str,default="batch_wise") #Alternatively moving
    parser.add_argument('--gen_lambda',type=float,default=0.2) #Weight to old baseline when computing moving average in case gen_base_type is moving
    #IGNORE Generator optimizer arguments
    parser.add_argument('--gen_lr',type=float,default=2e-4)
    #IGNORE Discriminator arguments
    parser.add_argument('--disc_d_embed',type=int,default=256)
    parser.add_argument('--disc_init_from_embed',action='store_true')
    parser.add_argument('--disc_hidden_size',type=int,default=256)
    parser.add_argument('--disc_n_layers',type=int,default=1)
    parser.add_argument('--disc_enc_type',type=str,default="plain")
    parser.add_argument('--disc_dp_ratio',type=float,default=0.2)
    #IGNORE Discriminator optimizer arguments
    parser.add_argument('--disc_lr',type=float,default=3e-3)

    parser.add_argument('--no_dev_eval',action='store_true')
    args = parser.parse_args()
    return args
