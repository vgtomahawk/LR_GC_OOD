import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import torchtext
from torchtext.data.batch import Batch

class Bottle(nn.Module):

    def forward(self, input):
        if len(input.size()) <= 2:
            return super(Bottle, self).forward(input)
        size = input.size()[:2]
        out = super(Bottle, self).forward(input.view(size[0]*size[1], -1))
        return out.view(size[0], size[1], -1)


class Linear(Bottle, nn.Linear):
    pass


class Encoder(nn.Module):

    def __init__(self, config):
        super(Encoder, self).__init__()
        self.config = config
        input_size = config.d_proj if config.projection else config.d_embed
        dropout = 0 if config.n_layers == 1 else config.dp_ratio
        self.rnn = nn.LSTM(input_size=input_size, hidden_size=config.d_hidden,
                        num_layers=config.n_layers, dropout=dropout,
                        bidirectional=config.birnn)

    def forward(self, inputs):
        batch_size = inputs.size()[1]
        state_shape = self.config.n_cells, batch_size, self.config.d_hidden
        h0 = c0 =  inputs.new_zeros(state_shape)
        outputs, (ht, ct) = self.rnn(inputs, (h0, c0))
        return ht[-1] if not self.config.birnn else ht[-2:].transpose(0, 1).contiguous().view(batch_size, -1)

class LSTMEncoder(nn.Module):

    def __init__(self, config):
        super(LSTMEncoder, self).__init__()
        self.config = config
        input_size = config.d_proj if config.projection else config.d_embed
        dropout = 0 if config.n_layers == 1 else config.dp_ratio
        self.rnn = nn.LSTM(input_size=input_size, hidden_size=config.d_hidden,
                        num_layers=config.n_layers, dropout=dropout,
                        bidirectional=config.birnn)

    def forward(self, inputs, inputs_labels):
        batch_size = inputs.size()[1]
        state_shape = self.config.n_cells, batch_size, self.config.d_hidden
        h0 = c0 =  inputs.new_zeros(state_shape)
        inputs_labels = inputs_labels.unsqueeze(0)

        if self.config.at_start:
            _ , (h1,c1) = self.rnn(inputs_labels, (h0,c0) )
        else:
            (h1,c1) = (h0,c0)

        outputs, (ht, ct) = self.rnn(inputs, (h1, c1))


        shifted_outputs = torch.cat((h1,outputs[:-1,:,:]),dim=0)
        if self.config.at_hidden:
            shifted_outputs  = torch.cat( ( shifted_outputs, inputs_labels.expand( shifted_outputs.size()[0], -1, -1) ) , dim = 2 )
        return shifted_outputs

class BareLSTMEncoder(nn.Module):

    def __init__(self, config,disc=False):
        super(BareLSTMEncoder, self).__init__()
        #self.config = config
        self.input_size = input_size = config.back_input_size #config.d_proj if config.projection else config.d_embed
        dropout = 0 if config.n_layers == 1 else config.dp_ratio
        self.rnn = nn.LSTM(input_size=input_size, hidden_size=input_size,
                        num_layers=1, dropout=dropout,
                        bidirectional=False)
        self.embed = nn.Embedding(config.n_embed,self.input_size)
        if disc:
            #No sharing of embedding matrices in this case
            self.out = Linear(self.input_size,2)
        else:
            self.out = Linear(self.input_size,config.n_embed,bias=False)
            self.out.weight.data = self.embed.weight.data

    def forward(self, batch):
        inputs=self.embed(batch.hypothesis)
        batch_size = inputs.size()[1]
        state_shape = 1, batch_size, self.input_size
        h0 = c0 =  inputs.new_zeros(state_shape)
        outputs, (ht, ct) = self.rnn(inputs, (h0, c0))
        shifted_outputs = torch.cat((h0,outputs[:-1,:,:]),dim=0)
        logits = self.out(shifted_outputs)
        return logits

class VVUL(nn.Module):

    def __init__(self, config, h_size):
        super(VVUL, self).__init__()
        #self.config = config
        self.h_size = h_size
        input_size = self.h_size #config.d_proj if config.projection else config.d_embed
        dropout = 0 if config.n_layers == 1 else config.dp_ratio
        self.rnn = nn.LSTM(input_size=input_size, hidden_size=input_size,
                        num_layers=1, dropout=dropout,
                        bidirectional=False)
        self.embed = nn.Embedding(config.n_embed,self.h_size)
        self.out = Linear(self.h_size,config.n_embed,bias=False)
        self.out.weight.data = self.embed.weight.data

    def forward(self, batch):
        inputs=self.embed(batch.hypothesis)
        batch_size = inputs.size()[1]
        state_shape = 1, batch_size, self.h_size
        h0 = c0 =  inputs.new_zeros(state_shape)
        outputs, (ht, ct) = self.rnn(inputs, (h0, c0))
        shifted_outputs = torch.cat((h0,outputs[:-1,:,:]),dim=0)
        logits = self.out(shifted_outputs)
        return logits


def corrupt(batch,config,inputs,eos_id=2,pad_id=1, noise_level=0.5):
    #print(batch.hypothesis.size())
    #Create random tensor of ints from inputs.vocab
    #Sample random tensor of 0-1, with probability given by corruption factor
    #mix batch.hypothesis with this random tensor [Except at EOS and Pad Positions]
    #Return an updated "batch" object with batch.hypothesis replaced by this new hypothesis. batch.label remains the same
    is_eos_or_pad_mask =  (batch.hypothesis == eos_id).long() + (batch.hypothesis == pad_id).long()
    bernoulli_mask = (torch.rand(batch.hypothesis.size(), device=batch.hypothesis.device) < noise_level ).long()
    if config.noise_type == "uniform":
        replacement_indices = torch.randint(low=3,high=len(inputs.vocab),size=tuple(batch.hypothesis.size()),dtype=batch.hypothesis.dtype, device=batch.hypothesis.device).long()
    elif config.noise_type == "unigram" or config.noise_type=="uniroot":
        replacement_indices = torch.multinomial(config.freq_list_tensor.unsqueeze(dim=0).expand((batch.hypothesis.size()[0],-1)),batch.hypothesis.size()[1],replacement=True)+3
        device = torch.device('cuda:{}'.format(config.gpu))
        replacement_indices = replacement_indices.to(device)
    corrupted_hypothesis = (1-bernoulli_mask)*batch.hypothesis + bernoulli_mask*replacement_indices
    corrupted_hypothesis = is_eos_or_pad_mask*batch.hypothesis + (1-is_eos_or_pad_mask)*corrupted_hypothesis
    #print(sum(sum(corrupted_hypothesis!=batch.hypothesis)))
    #print(type(batch))
    corrupted_batch = Batch.fromvars(batch.dataset,batch.batch_size)
    #Create and return corrupted batch using the corrupted hypothesis
    corrupted_batch.label = batch.label
    corrupted_batch.hypothesis = corrupted_hypothesis
    return corrupted_batch

class IntentClassifierGenerative(nn.Module):

    def __init__(self, config, init_tuple=None):
        super(IntentClassifierGenerative, self).__init__()

        self.config = config
        self.label_embed = nn.Embedding(config.d_out, config.d_embed)
        self.embed = nn.Embedding(config.n_embed, config.d_embed)
        self.projection = Linear(config.d_embed, config.d_proj)
        self.encoder = LSTMEncoder(config)
        self.dropout = nn.Dropout(p=config.dp_ratio)
        self.relu = nn.ReLU()
        if self.config.at_hidden:
            self.out = Linear(config.d_hidden+config.d_embed,config.n_embed)
        else:
            self.out = Linear(config.d_hidden,config.n_embed)

        if self.config.word_vectors:
            init_matrix, init_mask = init_tuple[0], init_tuple[1]
            init_matrix, init_mask = torch.FloatTensor(init_matrix), torch.FloatTensor(init_mask).unsqueeze(dim=1)
            modified_init_matrix = (1-init_mask) * self.embed.weight.data.detach()+init_mask*init_matrix
            self.embed.weight.data.copy_(modified_init_matrix)
            self.embed.weight.requires_grad = False

        #Priors
        self.device = torch.device('cuda:{}'.format(self.config.gpu))
        #self.register_buffer("device",self.device)
        self.prior_counts = torch.zeros((self.config.d_out,), device=self.device)
        #self.register_buffer("prior_counts",self.prior_counts)
        self.Z = 0.0
        #self.register_buffer("Z",self.Z)
        self.prior_log_probs = None
        #self.register_buffer("prior_log_probs",self.prior_log_probs)

    def update_prior_counts(self,batch):
        categorical_counts = torch.zeros((batch.label.size()[0],self.config.d_out),device=self.device).scatter_(1,batch.label.unsqueeze(1),1.)
        self.prior_counts = self.prior_counts + torch.sum(categorical_counts,dim=0).detach()
        self.Z+=batch.label.size()[0]

    def get_prior_log_probs(self,reestimate=False):
        if reestimate or self.prior_log_probs is None:
            self.prior_log_probs = torch.log(self.prior_counts/self.Z + 1e-10).detach()
        return self.prior_log_probs

    def forward(self,batch,addNoise=False, inferAll=True, scramble=False):
        hypo_label_embed = self.label_embed(batch.label)
        hypo_embed = self.embed(batch.hypothesis)
        if addNoise:
            hypo_embed = hypo_embed + 1e-1*torch.randn(hypo_embed.size(),dtype=hypo_embed.dtype, device=hypo_embed.device)
        if self.config.fix_emb:
            hypo_embed =hypo_embed.detach()
        if self.config.projection:
            hypo_embed = self.relu(self.projection(hypo_embed))
            if self.config.at_start:
                hypo_label_embed = self.relu(self.projection(hypo_label_embed))
        hypothesis = self.encoder(hypo_embed,hypo_label_embed)
        logits = self.out(hypothesis)
        with torch.no_grad():
            inf_logits_list = None
            if inferAll:
                inf_logits_list = []
                if not scramble:
                    for inf_label_index in range(self.config.d_out):
                        inf_label = inf_label_index * torch.ones(batch.label.size(),device=batch.label.device,dtype=batch.label.dtype)
                        inf_hypo_label_embed = self.label_embed(inf_label).detach()
                        if self.config.projection:
                            if self.config.at_start:
                                inf_hypo_label_embed = self.relu(self.projection(inf_hypo_label_embed))
                        inf_hypothesis = self.encoder(hypo_embed,inf_hypo_label_embed).detach()
                        inf_logits = self.out(inf_hypothesis)
                        inf_logits_list.append(inf_logits.contiguous().detach())
                else:
                    inf_hypo_label_embed_scrambled = 0
                    for inf_label_index in range(self.config.d_out):
                        inf_label = inf_label_index * torch.ones(batch.label.size(),device=batch.label.device,dtype=batch.label.dtype)
                        inf_hypo_label_embed = self.label_embed(inf_label).detach()
                        if self.config.projection:
                            if self.config.at_start:
                                inf_hypo_label_embed = self.relu(self.projection(inf_hypo_label_embed))
                        inf_hypo_label_embed_scrambled = inf_hypo_label_embed_scrambled + inf_hypo_label_embed.detach()
                    inf_hypo_label_embed_scrambled = inf_hypo_label_embed_scrambled / self.config.d_out
                    inf_hypothesis = self.encoder(hypo_embed,inf_hypo_label_embed_scrambled).detach()
                    inf_logits_scrambled = self.out(inf_hypothesis)
                    for inf_label_index in range(self.config.d_out):
                        inf_logits_list.append(inf_logits_scrambled.contiguous().detach())
                inf_logits_list = torch.stack(inf_logits_list,dim=0).detach()
                torch.cuda.empty_cache()

        return inf_logits_list, logits

    def infer_generative(self,batch,answer):
        expanded_hypothesis = batch.hypothesis.unsqueeze(0).expand_as(answer[:,:,:,0]).detach()
        expanded_hypothesis_original_size = expanded_hypothesis.size()
        answer_original_size = answer.size()

        infer_loss = F.cross_entropy(answer.contiguous().view(-1,self.config.n_embed),expanded_hypothesis.contiguous().view(-1),reduction='none').detach()
        infer_loss = infer_loss.view(expanded_hypothesis_original_size).sum(dim=1).transpose(0,1).detach()
        answer = None
        answer = -infer_loss.detach()
        infer_loss = None
        #print(answer.size())
        prior_factor = self.prior_log_probs.unsqueeze(dim=0).expand_as(answer).detach()
        #print(prior_factor.size())
        answer = answer+prior_factor
        return answer

class IntentClassifier(nn.Module):

    def __init__(self, config, init_tuple=None):
        super(IntentClassifier, self).__init__()
        self.config = config
        self.embed = nn.Embedding(config.n_embed, config.d_embed)

        if self.config.word_vectors:
            init_matrix, init_mask = init_tuple[0], init_tuple[1]
            init_matrix, init_mask = torch.FloatTensor(init_matrix), torch.FloatTensor(init_mask).unsqueeze(dim=1)
            modified_init_matrix = (1-init_mask) * self.embed.weight.data.detach()+init_mask*init_matrix
            self.embed.weight.data.copy_(modified_init_matrix)
            #self.embed = nn.Embedding.from_pretrained(torch.FloatTensor(text_field.vocab.vectors))
            self.embed.weight.requires_grad = False

        self.projection = Linear(config.d_embed, config.d_proj)
        self.encoder = Encoder(config)
        self.dropout = nn.Dropout(p=config.dp_ratio)
        self.relu = nn.ReLU()

        seq_in_size = config.d_hidden
        if self.config.birnn:
            seq_in_size *= 2
        lin_config = [seq_in_size]*2

        if not self.config.short_circuit_main:
            self.intermediate = nn.Sequential(
                Linear(*lin_config),
                self.relu,
                self.dropout,
                Linear(*lin_config),
                self.relu,
                self.dropout,
                Linear(*lin_config),
                self.relu,
                self.dropout)

        if self.config.embed_out_shared:
            self.pre_embedding_layer =  nn.Sequential(Linear(seq_in_size,config.n_embed),self.relu,self.dropout)
            self.shared_embedding_layer = Linear(config.n_embed,config.d_embed,bias=False)
            print(self.shared_embedding_layer.weight.data.size())
            print(self.embed.weight.data.size())
            self.shared_embedding_layer.weight.data = self.embed.weight.data.transpose(0,1)
            #self.on_top = nn.Sequential(self.relu,self.dropout)

        if self.config.mos:
            self.outSeq=[]
            for k in range(self.config.mos_k):
                if not self.config.embed_out_shared:
                    self.outSeq.append(Linear(seq_in_size, config.d_out, bias=(not self.config.bias_false)))
                else:
                    self.outSeq.append(Linear(config.d_embed, config.d_out, bias=(not self.config.bias_false)))
            self.mixLayer = Linear(seq_in_size, self.config.mos_k)
            self.outSeq = nn.ModuleList(self.outSeq)
        else:
            if not self.config.embed_out_shared:
                self.out = Linear(seq_in_size, config.d_out, bias=(not self.config.bias_false))
            else:
                self.out = Linear(config.d_embed, config.d_out, bias=(not self.config.bias_false))


        if self.config.aux_BCE_OOD or self.config.conf_teach:
            self.oodOut = Linear(seq_in_size, 2)
        self.inhibitor = nn.Parameter(torch.rand(1))

        #Post-Training Statistics
        if self.config.mahalanobis or self.config.euclidean or self.config.manhattan:
            device = torch.device('cuda:{}'.format(self.config.gpu))
            self.mu_maha_count = torch.zeros((self.config.d_out,), device=device)
            if not self.config.embed_out_shared:
                space_size = seq_in_size
            else:
                space_size = config.d_embed
            self.mu_maha = torch.zeros((self.config.d_out, space_size), device=device)
            self.sigma_maha_shared = torch.zeros((space_size, space_size), device=device)
            self.sigma_maha_shared_inverse = None
        #if self.config.ratioKldPrune or self.config.ratioEmdPrune:
        device = torch.device('cuda:{}'.format(self.config.gpu))
        self.label_ratio = torch.zeros((self.config.d_out,), device=device)


    def forward(self, batch, addNoise=False, direct_hypothesis=None):
        if direct_hypothesis is not None:
            hypo_embed = self.embed(direct_hypothesis)
        else:
            hypo_embed = self.embed(batch.hypothesis)

        if addNoise:
            hypo_embed = hypo_embed + 1e-1*torch.randn(hypo_embed.size(),dtype=hypo_embed.dtype, device=hypo_embed.device)
        if self.config.fix_emb:
            hypo_embed =hypo_embed.detach()
        if self.config.projection:
            hypo_embed = self.relu(self.projection(hypo_embed))
        hypothesis = self.encoder(hypo_embed)

        if not self.config.short_circuit_main:
            intermediateRepn = self.intermediate(hypothesis)

        if self.config.short_circuit_main and self.config.embed_out_shared:
            #hypothesis = self.on_top(self.shared_embedding_layer(self.pre_embedding_layer(hypothesis)))
            hypothesis = self.shared_embedding_layer(self.pre_embedding_layer(hypothesis))
        elif (not self.config.short_circuit_main) and self.config.embed_out_shared:
            #intermediateRepn = self.on_top(self.shared_embedding_layer(self.pre_embedding_layer(intermediateRepn)))
            intermediateRepn = self.shared_embedding_layer(self.pre_embedding_layer(intermediateRepn))

        # Mixture of softmaxes - we need to take the mixture in probability space
        # Hence, we cannot pass out logits directly like in the typical forward pass.
        if self.config.mos:
            scoresSeq = []
            for k in range(self.config.mos_k):
                if (not self.config.short_circuit_in_half) or (k%2==0):
                    scoresSeq.append(F.softmax((self.outSeq[k])(intermediateRepn),dim=1))
                else:
                    scoresSeq.append(F.softmax((self.outSeq[k])(hypothesis),dim=1))
                if self.config.no_ood_in_half and k%2==0:
                    inputMask = torch.ones(scoresSeq[k].size(), device=scoresSeq[k].device)
                    inputMask[:,self.config.ood_index] = 1e-8
                    scoresSeq[k] = scoresSeq[k] * inputMask
                    Zsum = torch.sum(scoresSeq[k],dim=1)
                    Z = torch.reciprocal(Zsum)
                    scoresSeq[k] = scoresSeq[k] * Z.unsqueeze(1)
            mixCoeffs = F.softmax(self.mixLayer(intermediateRepn),dim=1)
            scoresSeq = torch.stack(scoresSeq,dim=1)
            scores = (scoresSeq * mixCoeffs.unsqueeze(2)).sum(1)
            scores = torch.log(scores.add_(1e-8))
        elif self.config.cosine_softmax:
            intermediateRepn = hypothesis
            scoresNorm = torch.reciprocal(1e-8+torch.norm(intermediateRepn,dim=1)).unsqueeze(dim=1)
            intermediateRepnNormalized = intermediateRepn * scoresNorm
            scores = self.out(intermediateRepnNormalized) - self.config.margin
            scores = scores * scoresNorm
        else:
            if self.config.short_circuit_main:
                scores = self.out(hypothesis)
            else:
                scores = self.out(intermediateRepn)

        if self.config.cosine_softmax:
            weightNorm = torch.reciprocal(torch.norm(self.out.weight,dim=1))
            scores = scores * weightNorm.unsqueeze(dim=0)


        if self.config.inhibited_softmax or self.config.conf_teach:
            onesVector = (torch.ones(scores.size()[0], device=scores.device)*self.inhibitor).unsqueeze(dim=1)
            scores = torch.cat((scores[:,:self.config.ood_index],onesVector,scores[:,self.config.ood_index+1:]),dim=1)

        #if self.config.zero_exclude:
        #    scores = scores[:,1:]

        oodScores = None
        if self.config.aux_BCE_OOD or self.config.conf_teach:
            if self.config.short_circuit:
                oodScores = self.oodOut(hypothesis)
            else:
                oodScores = self.oodOut(intermediateRepn)

        if self.config.returnIntermediate:
            if self.config.short_circuit_main:
                return (scores,oodScores,hypothesis.clone().detach())
            else:
                return (scores,oodScores,intermediateRepn.clone().detach())
        else:
            intermediateRepn=None
            return (scores,oodScores,intermediateRepn)
