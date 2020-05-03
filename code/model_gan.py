import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from torch.distributions import Categorical
import random
from wasserstein import WassersteinLossStab

class Gen(nn.Module):

    def __init__(self, config, init_tuple=None):
        super(Gen, self).__init__()
        #Initialize self from config. Do not refer to config beyond this point.
        #Kept as general as possible to allow initialization from other config schemas
        #A: Copy config
        self.config = config
        #B: Embedding specs
        self.d_embed = config.gen_d_embed
        self.n_embed = config.n_embed
        self.init_from_embed = config.gen_init_from_embed
        #C: Encoder specs
        self.hidden_size = config.gen_hidden_size
        self.n_layers = config.gen_n_layers
        self.enc_type = config.gen_enc_type
        self.dp_ratio = config.gen_dp_ratio
        self.enc_dropout = 0 if self.n_layers == 1 else self.dp_ratio
        self.device = torch.device('cuda:{}'.format(self.config.gpu))
        #D: Out specs
        self.share_out_embed = config.gen_share_out_embed

        self.dropout = nn.Dropout(p=self.dp_ratio)
        self.relu = nn.ReLU()

        if self.config.gen_init_type == "random_smoothed":
            self.input_smoother = nn.Sequential(nn.Linear(self.hidden_size ,self.hidden_size),self.relu,self.dropout, nn.Linear(self.hidden_size,self.hidden_size),self.relu,self.dropout)
        #Now initialize layers
        # A: Initialize Embeddings
        self.embed = nn.Embedding(self.n_embed,self.d_embed)
        self.start_embed = nn.Embedding(1,self.d_embed)
        if self.init_from_embed:
            init_matrix, init_mask = init_tuple[0], init_tuple[1]
            init_matrix, init_mask = torch.FloatTensor(init_matrix), torch.FloatTensor(init_mask).unsqueeze(dim=1)
            modified_init_matrix = (1-init_mask) * self.embed.weight.data.detach()+init_mask*init_matrix
            self.embed.weight.data.copy_(modified_init_matrix)
            self.embed.weight.requires_grad = False
        # B: Initialize Encoder
        if self.enc_type == "plain":
            self.encoder = nn.LSTM(input_size = self.d_embed, hidden_size = self.hidden_size, num_layers = self.n_layers, dropout = self.enc_dropout, bidirectional=False)
        # C: Initialize Out. Bias turned off if output embedding matrix is shared
        self.out = nn.Linear(self.hidden_size,config.n_embed,bias=(not self.share_out_embed))
        if self.share_out_embed:
            if self.hidden_size != self.d_embed:
                print("Hidden State Size Differs From Embedding Size. You can't share output embeddings unless you project the Hidden State to your Embedding Size")
            self.out.weight.data = self.embed.weight.data
        # D: Other parameters such as baseline
        if self.config.gen_base_type == "moving":
            self.global_baseline = 0.0

    def feed_start_token(self,batch_size,h0,c0):
        start_input = self.start_embed( torch.zeros(batch_size,device=self.device).long()  ) # (B,E)
        start_input = start_input.unsqueeze(dim=0) # (1,B,E)
        _, (h1,c1)  = self.encoder(start_input, (h0,c0))
        return (h1,c1)

    def forward(self,batch,log_prob=False):
        inputs=self.embed(batch) # (L,B) -> (L,B,E)
        batch_size = inputs.size()[1]
        #state_shape = 1, batch_size, self.hidden_size
        state_shape = self.n_layers, batch_size, self.hidden_size

        (h0,c0) = self.init_hidden(batch_size,init_type="zero") #(1,B,H)
        (h1,c1) = self.feed_start_token(batch_size,h0,c0) # (1,B,H)

        outputs, (ht, ct) = self.encoder(inputs, (h1, c1))
        shifted_outputs = torch.cat((h1,outputs[:-1,:,:]),dim=0) #(L,B,H) -> (L,B,H)
        logits = self.out(shifted_outputs) # (L,B,H) -> (L,B,V)
        if log_prob:
            log_probs = F.log_softmax(logits,dim=2) # (L,B,V) -> (L,B.V)
            return log_probs
        else:
            return logits

    """
    def forward(self,batch,log_prob=False,no_shift=True,prob=False):
        inputs=self.embed(batch) # (L,B) -> (L,B,E)
        batch_size = inputs.size()[1]
        #state_shape = 1, batch_size, self.hidden_size
        state_shape = self.n_layers, batch_size, self.hidden_size

        (h0,c0) = self.init_hidden(batch_size,init_type="zero") #(1,B,H)
        (h1,c1) = self.feed_start_token(batch_size,h0,c0) # (1,B,H)

        outputs, (ht, ct) = self.encoder(inputs, (h1, c1))
        if no_shift:
            shifted_outputs = outputs
        else:
            shifted_outputs = torch.cat((h1,outputs[:-1,:,:]),dim=0) #(L,B,H) -> (L,B,H)
        logits = self.out(shifted_outputs) # (L,B,H) -> (L,B,V)
        if log_prob:
            log_probs = F.log_softmax(logits,dim=2) # (L,B,V) -> (L,B.V)
            return log_probs
        elif prob:
            probs = F.softmax(logits+1e-8,dim=2)
            return probs
        else:
            return logits
    """

    def init_hidden(self,batch_size,init_type="zero"):
        #state_shape = 1, batch_size, self.hidden_size
        state_shape = self.n_layers, batch_size, self.hidden_size
        if init_type == "zero":
            h0 = c0 = torch.zeros(state_shape,device=self.device)
        elif init_type == "random":
            h0 = c0 = torch.zeros(state_shape,device=self.device).normal_()
        elif init_type == "random_smoothed":
            h0 = c0 = self.input_smoother(torch.zeros(state_shape,device=self.device).normal_())
        return (h0,c0)

    def sample_batch(self,batch,disc_model=None,class_model=None,custom=False,sample_size=None,just_sample=False,reduction="full_mean"):
        #batch is used here only to get an estimate about batch statistics
        batch_size = batch.size(1) if not custom else sample_size
        all_ended = torch.zeros((batch_size),device=self.device).long() #(B,1)

        (h0,c0) = self.init_hidden(batch_size,init_type=self.config.gen_init_type)
        (h1,c1) = self.feed_start_token(batch_size,h0,c0)

        (ht,ct) = (h1,c1) #(1,B,H),(1,B,H) -> (1,B,H),(1,B,H)

        actions, log_probs, rewards = [], [], []

        all_pad_ids = torch.ones((batch_size,),device=self.device).long()* self.config.pad_id
        all_eos_ids = torch.ones((batch_size,),device=self.device).long()* self.config.eos_id

        gen_sample_L = 0

        chosen_limit = random.randint(self.config.max_gen_sample_L-5, self.config.max_gen_sample_L+5)

        while torch.sum(all_ended)<batch_size and gen_sample_L <= chosen_limit:
            out_logits = self.out(ht) # (1,B,V)
            out_logits = out_logits.squeeze(dim=0) #(1,B,V) -> (B,V)
            out_probs = F.softmax(out_logits+1e-8,dim=1) # (B,V) -> (B,V)
            if gen_sample_L == chosen_limit:
                print(torch.sum(all_ended))
                action = (1-all_ended)*all_eos_ids   + all_ended*(all_pad_ids)
            else:
                #out_probs[:,self.config.pad_id]=0       # (B,V) -> inplace
                action = torch.multinomial(out_probs,1) # (B,V) -> (B,1)
            del out_probs
            #Update only with pad-ids for those that have already ended
            action = (1-all_ended)*action.view(-1) + all_ended*all_pad_ids #(B,1) -> (B,)
            out_log_probs = F.log_softmax(out_logits,dim=1)  # (B,V) -> (B,V)
            out_log_prob_action = out_log_probs.gather(1,action.view(-1,1)) # (B,V) , (1,B) -> (B,1)

            out_log_prob_action = out_log_prob_action.view(-1) # (B,1) -> (B,)
            out_log_prob_action = (1-all_ended.float())*out_log_prob_action + (all_ended.float())*torch.zeros((batch_size,),device=self.device) # (B,)
            #Update all_ended
            all_ended = all_ended + (action==self.config.eos_id).long()
            #print(all_ended.size())
            #Update final state
            action_inputs = self.embed(action.view(1,-1))  #(1,B,E)
            #print(ht.size(),ct.size())
            _, (ht,ct) = self.encoder(action_inputs, (ht,ct)) #(1,B,E),(1,B,H) -> (1,B,H)
            #print(ht.size(),ct.size())
            #Update the episode records
            actions.append(action.detach())
            log_probs.append(out_log_prob_action)
            gen_sample_L += 1

        print("Generated Sample Length:",gen_sample_L)
        #Convert "actions" to a batch of longs to feed into the discriminator
        sampled_batch = torch.stack(actions,dim=0)
        del actions

        first_sample = sampled_batch[:,0].detach().cpu().numpy().tolist()
        print("Generated Sample:",[self.config.vocab_obj.itos[word_id] for word_id in first_sample])

        if just_sample:
            return sampled_batch
        else:
            #Compute return
            class_H = None
            if class_model is not None:
                class_logits,_,_ = class_model(None,direct_hypothesis=sampled_batch)
                class_logits = class_logits[:,1:]
                class_log_prob = F.log_softmax(class_logits/self.config.genT,dim=1)
                if self.config.gen_reverse_H:
                    class_H = self.config.seq_gan_class_beta * torch.mean(class_log_prob,dim=1)
                elif self.config.gen_square_H:
                    label_size = class_logits.size(1)
                    class_prob = F.softmax(class_logits/self.config.genT,dim=1)
                    class_prob_diff_with_uniform_squared = torch.pow(class_prob - 1.0/label_size,2)
                    class_H = self.config.seq_gan_class_beta * -torch.mean(class_prob_diff_with_uniform_squared,dim=1)
                elif self.config.gen_emd_H:
                    label_size = class_logits.size(1) # |L|
                    class_prob = F.softmax(class_logits/self.config.genT,dim=1)
                    class_prob_uniform = (1.0/label_size) * torch.ones(class_prob.size(),device=class_prob.device)
                    cost = torch.ones((label_size,label_size))-torch.eye(label_size)
                    cost = cost.to(class_prob.device)
                    emd_computer = WassersteinLossStab(cost)
                    emd_distance = emd_computer(class_prob,class_prob_uniform)
                    class_H = self.config.seq_gan_class_beta * (-emd_distance)
                else:
                    class_prob = F.softmax(class_logits/self.config.genT,dim=1)
                    class_H = self.config.seq_gan_class_beta * torch.sum(-class_prob*class_log_prob,dim=1)
                #This negative sign is because we want to convert the reward to a loss to pass on this term to a classifier
                loss_class = -torch.mean(class_H)
                #Henceforth, we detach the "entropy" reward since we will mix it in with the main reward
                class_H = class_H.detach()

            with torch.no_grad():

                disc_probs = disc_model(sampled_batch,prob=True)
                rewards = disc_model.get_rewards_from_probs(disc_probs)
                if class_model is not None:
                    print("Entropy Contribution:",torch.sum(class_H))
                    rewards[-1,:] = rewards[-1,:] + class_H
                #pad_mask = (sampled_batch!=self.config.pad_id).float()
                #rewards = rewards*pad_mask
                #del pad_mask
                baseline_reward = torch.mean(rewards)
                if self.config.gen_base_type == "moving":
                    temp = baseline_reward.detach().item()
                    self.global_baseline = self.config.gen_lambda * self.global_baseline + (1-self.config.gen_lambda) * temp
                    baseline_reward = self.global_baseline
                self.compute_returns_from_rewards(rewards,gamma=self.config.gen_gamma,inplace=True)
                shifted_rewards = (rewards - baseline_reward)
            log_probs_tensor = torch.stack(log_probs,dim=0)
            del log_probs
            #pad_mask = (sampled_batch!=self.config.pad_id).float()
            #shifted_rewards = shifted_rewards*pad_mask
            #del pad_mask
            if reduction=="batch_mean":
                overall_reward = torch.mean(torch.sum(shifted_rewards * log_probs_tensor,dim=0),dim=0)
            elif reduction=="full_mean":
                overall_reward = torch.mean(shifted_rewards * log_probs_tensor)

            overall_loss = (-overall_reward)

        if class_model is None:
            loss_class = None

        return overall_loss, loss_class

    def compute_returns_from_rewards(self,rewards,inplace=True,gamma=0.79):
        L = rewards.size()[0]

        if not inplace:
            returns = torch.zeros(rewards.size(),device=self.device)
            returns[L-1,:] = rewards[L-1,:]

        for i in range(1,L):
            current_index = L-1-i
            if inplace:
                rewards[current_index,:] = rewards[current_index,:] + gamma*rewards[current_index+1,:]
            else:
                returns[current_index,:] = rewards[current_index,:] + gamma*returns[current_index+1,:]

        if inplace:
            return None
        else:
            return returns


class Disc(nn.Module):

    def __init__(self, config, init_tuple=None):
        super(Disc, self).__init__()
        #Initialize self from config. Do not refer to config beyond this point.
        #Kept as general as possible to allow initialization from other config schemas
        #A: Copy config
        self.config = config
        #B: Embedding specs
        self.d_embed = config.disc_d_embed
        self.n_embed = config.n_embed
        self.init_from_embed = config.disc_init_from_embed
        #C: Encoder specs
        self.hidden_size = config.disc_hidden_size
        self.n_layers = config.disc_n_layers
        self.enc_type = config.disc_enc_type
        self.dp_ratio = config.disc_dp_ratio
        self.enc_dropout = 0 if self.n_layers == 1 else self.dp_ratio
        self.device = torch.device('cuda:{}'.format(self.config.gpu))
        #D: Out specs
        #self.share_out_embed = config.disc_share_out_embed

        #Now initialize layers
        # A: Initialize Embeddings
        self.embed = nn.Embedding(self.n_embed,self.d_embed)
        self.start_embed = nn.Embedding(1,self.d_embed)
        if self.init_from_embed:
            init_matrix, init_mask = init_tuple[0], init_tuple[1]
            init_matrix, init_mask = torch.FloatTensor(init_matrix), torch.FloatTensor(init_mask).unsqueeze(dim=1)
            modified_init_matrix = (1-init_mask) * self.embed.weight.data.detach()+init_mask*init_matrix
            self.embed.weight.data.copy_(modified_init_matrix)
            self.embed.weight.requires_grad = False
        # B: Initialize Encoder
        if self.enc_type == "plain":
            self.encoder = nn.LSTM(input_size = self.d_embed, hidden_size = self.hidden_size, num_layers = self.n_layers, dropout = self.enc_dropout, bidirectional=False)
        # C: Initialize Out. Bias turned off if output embedding matrix is shared
        self.out = nn.Linear(self.hidden_size,2)


    def feed_start_token(self,batch_size,h0,c0):
        start_input = self.start_embed( torch.zeros(batch_size,device=self.device).long()  ) # (B,E)
        start_input = start_input.unsqueeze(dim=0) # (1,B,E)
        _, (h1,c1)  = self.encoder(start_input, (h0,c0))
        return (h1,c1)

    def forward(self,batch,log_prob=False,no_shift=True,prob=False,T=1):
        inputs=self.embed(batch) # (L,B) -> (L,B,E)
        batch_size = inputs.size()[1]
        #state_shape = 1, batch_size, self.hidden_size
        state_shape = self.n_layers, batch_size, self.hidden_size

        (h0,c0) = self.init_hidden(batch_size,init_type="zero") #(1,B,H)
        (h1,c1) = self.feed_start_token(batch_size,h0,c0) # (1,B,H)

        outputs, (ht, ct) = self.encoder(inputs, (h1, c1))
        if no_shift:
            shifted_outputs = outputs
        else:
            shifted_outputs = torch.cat((h1,outputs[:-1,:,:]),dim=0) #(L,B,H) -> (L,B,H)
        logits = self.out(shifted_outputs) # (L,B,H) -> (L,B,V)
        if log_prob:
            log_probs = F.log_softmax(logits/T,dim=2) # (L,B,V) -> (L,B.V)
            return log_probs
        elif prob:
            probs = F.softmax(logits/T+1e-8,dim=2)
            return probs
        else:
            return logits

    def compute_loss(self,batch,broad_label,reduction="full_mean"):
        logits = self.forward(batch)
        #print(logits.size())
        L=logits.size(0)
        B=logits.size(1)
        logits=logits.view(L*B,2)
        #print(logits.size())
        target = torch.ones(L*B,device=self.device).long()*broad_label
        #print(target.size())
        #pad_mask = (batch!=self.config.pad_id).float()
        if reduction=="full_mean":
            loss_value = F.cross_entropy(logits,target,reduction='none')
            loss_value = loss_value.view(L,B)
            #loss_value = loss_value * pad_mask
            loss_value = torch.mean(loss_value)
        elif reduction=="batch_mean":
            loss_value = F.cross_entropy(logits,target,reduction='none')
            loss_value = loss_value.view(L,B)
            #loss_value = loss_value * pad_mask
            loss_value = (loss_value.sum(dim=0)).mean(dim=0)
        return loss_value

    def get_rewards_from_probs(self,probs):
        reward_tensor = probs[:,:,1]
        reward_tensor = 2*reward_tensor-1
        return reward_tensor

    def init_hidden(self,batch_size,init_type="zero"):
        #state_shape = 1, batch_size, self.hidden_size
        state_shape = self.n_layers, batch_size, self.hidden_size
        if init_type == "zero":
            h0 = c0 = torch.zeros(state_shape,device=self.device)
        elif init_type == "random":
            h0 = c0 = torch.zeros(state_shape,device=self.device).normal_()
        return (h0,c0)

