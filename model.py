
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn import CrossEntropyLoss, MSELoss
from transformers import BertPreTrainedModel, RobertaModel, BertModel
from sklearn.covariance import EmpiricalCovariance
import random
import math

 
class RobertaClassificationHead(nn.Module): 
    def __init__(self, config):
        super().__init__()
        self.dense0 = nn.Linear(config.hidden_size, config.hidden_size) 
        
        self.dense = nn.Linear(config.hidden_size, config.hidden_size) 
        self.layer_norm = nn.LayerNorm(config.hidden_size) 
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels) 
        self.tanh = nn.Tanh()

    def forward(self, features):
        x = features
        x = self.dropout(x)
        x0 = self.dense0(x) 
        x0 = self.tanh(x0) 
         
        x = self.dropout(x0)
        x = self.dense(x) 
        x = pooled = torch.tanh(x)  
        
        x = self.dropout(x)
        x = self.out_proj(x)
        return x, pooled

class UnMaskedSRLHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense0 = nn.Linear(config.hidden_size, int(config.hidden_size)) 
        
        self.dense = nn.Linear(int(config.hidden_size), int(config.hidden_size) )
        self.layer_norm = nn.LayerNorm(config.hidden_size) 
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, 3) 
        self.tanh = nn.Tanh()

    def forward(self, SRL_means_concat):
        x = SRL_means_concat
        x = self.dropout(x)
        x0 = self.dense0(x)  
        x0 = self.tanh(x0)
         
        
        x = self.dropout(x0)
        x = self.dense(x)
        pooled = x = self.tanh(x) 
        x = self.dropout(x)
        x = self.out_proj(x) 
        return x 



class SelfSupervisedTransformer(nn.Module):
    def __init__(self, config):
        super().__init__() 
        self.encoder_layer =  nn.TransformerEncoderLayer(d_model=1024, nhead=16)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=3) 
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
    def forward(self, _last_hidden_states_ , SRL_verb, SRL_arg0, SRL_arg1, mask_labels, truncate_len):        
        if mask_labels!=[]:
            last_hidden_states_= _last_hidden_states_.clone()
         
        else: 
            last_hidden_states_ = _last_hidden_states_
        max_len=last_hidden_states_.shape[1] 
        new_SRL_verb=[]
        for verb_indices in SRL_verb:
            if len(verb_indices)>truncate_len:
                verb_indices = random.sample(verb_indices, truncate_len)
                new_SRL_verb.append(verb_indices)
            else: 
                new_SRL_verb.append(verb_indices)
                
        new_SRL_arg0=[]
        for arg0_indices in SRL_arg0:
            if len(arg0_indices)>truncate_len:
                arg0_indices = random.sample(arg0_indices, truncate_len)
                new_SRL_arg0.append(arg0_indices)
            else:
                new_SRL_arg0.append(arg0_indices)
                
        new_SRL_arg1=[]
        for arg1_indices in SRL_arg1:
            if len(arg1_indices)>truncate_len:
                arg1_indices = random.sample(arg1_indices, truncate_len)
                new_SRL_arg1.append(arg1_indices)
            else:
                new_SRL_arg1.append(arg1_indices)
 
        SRLs =[new_SRL_verb, new_SRL_arg0, new_SRL_arg1]
         
        if mask_labels!=[]:
            for batch_ind in range(last_hidden_states_.shape[0]):
 
                SRL_indices_to_be_masked=SRLs[ mask_labels[ batch_ind ]  ][batch_ind]
                srl_inds = torch.LongTensor(SRL_indices_to_be_masked).cuda()
                if torch.sum(srl_inds>=(max_len-1))>0:
                    new_s_i = []
                    for ind in srl_inds:
                        if ind <=(max_len-1):
                            new_s_i.append(ind)
                    srl_inds = torch.LongTensor(new_s_i)
                last_hidden_states_[batch_ind , srl_inds , :] =last_hidden_states_[batch_ind , srl_inds , :]*0
       
        
        if mask_labels!=[]: 
            last_hidden_states_ = last_hidden_states_.detach()

        transformer_input = last_hidden_states_
        transformer_output = self.transformer_encoder(transformer_input)
        
        return transformer_output
    
     
        
class RobertaForSequenceClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config) 
        self.config = config
        self.num_labels = config.num_labels
        self.masking_probability_train = config.masking_probability_train
        self.masking_probability = config.masking_probability     
        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.classifier = RobertaClassificationHead(config)
        self.SelfSuperT = SelfSupervisedTransformer(config)
        self.unmaskhead = UnMaskedSRLHead(config)
        self.init_weights()
        self.beta = config.beta   
        self.alpha = config.alpha
        self.theta=config.theta
        self.ssLoss = 0
        self.iter_count=0
        seed = self.config.seed 
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        
 
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
        SRL_verb=None,
        SRL_arg0=None,
        SRL_arg1=None
    ):
        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
             return_dict = True,
        )
 
        roberta_last_hidden_states =outputs['last_hidden_state']
        roberta_sigmoid_last_hidden_states = torch.sigmoid(roberta_last_hidden_states)
        roberta_last_hidden_states_copy=outputs['last_hidden_state']
        roberta_sigmoid_last_hidden_states_copy = torch.sigmoid(roberta_last_hidden_states_copy) 
        roberta_sigmoid_last_hidden_states_copy = roberta_sigmoid_last_hidden_states_copy.clone()

        if True:      
            transformer_output = self.SelfSuperT( roberta_sigmoid_last_hidden_states, SRL_verb, SRL_arg0, SRL_arg1, [] ,10)

        logits, pooled = self.classifier(transformer_output[:,0,:])  
        last_hidden_states = transformer_output
        
        v_avg_list=[]
        a0_avg_list=[]
        a1_avg_list=[]
        max_len=last_hidden_states.shape[1]
        if not ( SRL_verb == None): 
            for v in range(len(SRL_verb)):
                verb_indices = torch.tensor(SRL_verb[v]).int().cuda() 
                if torch.sum(verb_indices>=(max_len-1))>0: 
                    new_v_i = []
                    for ind in SRL_verb[v]:
                        if ind <(max_len-1):
                            new_v_i.append(ind)
                    verb_indices = torch.tensor(new_v_i).int().cuda()###
                        
                verbs = torch.index_select(torch.squeeze(last_hidden_states[v,:,:]), 0, verb_indices)  
                verbs_avg = torch.mean(verbs,0)
                v_avg_list.append(verbs_avg)
            v_avg = torch.stack(v_avg_list) 
            if torch.sum( v_avg.isnan()) >0: 
                    v_avg = torch.nan_to_num(v_avg, nan=0.0 , posinf=10, neginf =-10)

            pooledv=v_avg  
            for a0 in range(len(SRL_arg0)):
                _indices = torch.tensor(SRL_arg0[a0]).int().cuda()# 
                if torch.sum(_indices>=(max_len-1))>0: 
                    new_v_i = []
                    for ind in SRL_arg0[a0]:
                        if ind <(max_len-1):
                            new_v_i.append(ind) 
                    _indices = torch.LongTensor(new_v_i).cuda()# 

                        
                arg0s = torch.index_select(torch.squeeze(last_hidden_states[a0,:,:]), 0, _indices) 
                if torch.sum( arg0s.isnan()) >0: 
                    arg0s = torch.nan_to_num(arg0s, nan=0.0, posinf=10, neginf =-10) 
                arg0s_avg = torch.mean(arg0s,0)
                a0_avg_list.append(arg0s_avg)
            a0_avg = torch.stack(a0_avg_list)
            if torch.sum( a0_avg.isnan()) >0: 
                    a0_avg = torch.nan_to_num(a0_avg, nan=0.0 , posinf=10, neginf =-10)  
            pooleda0=a0_avg 
            for a1 in range(len(SRL_arg1)):
                _indices = torch.tensor(SRL_arg1[a1]).int().cuda()# 
                if torch.sum(_indices>=(max_len-1))>0: 
                    new_v_i = []
                    for ind in SRL_arg1[a1]:
                        if ind <(max_len-1):
                            new_v_i.append(ind)
                    _indices = torch.tensor(new_v_i).int().cuda()# 
                        
                arg1s = torch.index_select(torch.squeeze(last_hidden_states[a1,:,:]), 0, _indices)#  
                arg1s_avg = torch.mean(arg1s,0)
                a1_avg_list.append(arg1s_avg)
            a1_avg = torch.stack(a1_avg_list)
            if torch.sum( a1_avg.isnan()) >0: 
                    a1_avg = torch.nan_to_num(a1_avg, nan=0.0 , posinf=10, neginf =-10) 
            pooleda1=a1_avg 

            features_ = torch.cat([transformer_output[:,0,:],pooleda0,pooledv,pooleda1], dim=1) 
            mask_len = roberta_sigmoid_last_hidden_states_copy.shape[0] 
            mask_label_ = list(np.random.randint(3, size=( mask_len))) 
            truncate_length =512  
            max_len=transformer_output.shape[1]
            mask_v_avg_list=[]
            mask_a0_avg_list=[]
            mask_a1_avg_list=[]
            ss_va0a1=[]

            mask_label_ = torch.LongTensor(mask_label_).to(0) 

            new_SRL_verb=[]
            new_SRL_arg0=[]
            new_SRL_arg1=[]
            Union=[]
            if not ( SRL_verb == None):
                new_SRL_verb=[]
                new_SRL_arg0=[]
                new_SRL_arg1=[]
                Union=[]
                for bs in range(len( SRL_verb)):
                    verb_inds = SRL_verb[bs]
                    arg0_inds = SRL_arg0[bs]
                    arg1_inds = SRL_arg1[bs] 
                    Vs=verb_inds
                    A0s=arg0_inds
                    A1s=arg1_inds

                    num_probabilistic_selection_Vs = math.ceil(len(Vs)*self.masking_probability)
                    Vs_probabilistic_selection = random.sample(Vs, num_probabilistic_selection_Vs)
                    new_SRL_verb.append( Vs_probabilistic_selection )

                    num_probabilistic_selection_A0s = math.ceil(len(A0s)*self.masking_probability)
                    A0s_probabilistic_selection = random.sample(A0s, num_probabilistic_selection_A0s)
                    new_SRL_arg0.append( A0s_probabilistic_selection )

                    num_probabilistic_selection_A1s = math.ceil(len(A1s)*self.masking_probability)
                    A1s_probabilistic_selection = random.sample(A1s, num_probabilistic_selection_A1s)
                    new_SRL_arg1.append( A1s_probabilistic_selection )
                    
                    Union.append(Vs_probabilistic_selection+A0s_probabilistic_selection+A1s_probabilistic_selection)
            if True:        
                transformer_output_masked = self.SelfSuperT( roberta_sigmoid_last_hidden_states, Union, Union, Union,mask_label_,truncate_length ) 
            mask_v_avg_list=[]
            mask_a0_avg_list=[]
            mask_a1_avg_list=[]
            ss_va0a1=[]
            for batch_ind in range(transformer_output_masked.shape[0]):
 
                V_indices = torch.tensor(new_SRL_verb[batch_ind]).int().cuda()
                A0_indices = torch.tensor(new_SRL_arg0[batch_ind]).int().cuda()
                A1_indices = torch.tensor(new_SRL_arg1[batch_ind]).int().cuda()
                if torch.sum(V_indices>=(max_len-1))>0: 
                    new_v_i = []
                    for ind in new_SRL_verb[batch_ind]:
                        if ind <=(max_len-1):
                            new_v_i.append(ind)
                    V_indices = torch.tensor(new_v_i).int().cuda()# 
                if torch.sum(A0_indices>=(max_len-1))>0: 
                    new_v_i = []
                    for ind in new_SRL_arg0[batch_ind]:
                        if ind <=(max_len-1):
                            new_v_i.append(ind)
                    A0_indices = torch.tensor(new_v_i).int().cuda()# 
                if torch.sum(A1_indices>=(max_len-1))>0: 
                    new_v_i = []
                    for ind in new_SRL_arg1[batch_ind]:
                        if ind <=(max_len-1):
                            new_v_i.append(ind)
                    A1_indices = torch.tensor(new_v_i).int().cuda()###

                verbs = torch.index_select(torch.squeeze(transformer_output_masked[batch_ind,:,:]), 0, V_indices)
                verbs_isnan = torch.sum( verbs.isnan())
                if verbs_isnan >0: 
                    verbs = torch.nan_to_num(verbs, nan=0.0 , posinf=10, neginf =-10) 
                verbs_avg = torch.mean(verbs,0)
                mask_v_avg_list.append(verbs_avg)


                arg0s = torch.index_select(torch.squeeze(transformer_output_masked[batch_ind,:,:]), 0, A0_indices)
                arg0s_isnan = torch.sum( arg0s.isnan())
                if arg0s_isnan >0: 
                    arg0s = torch.nan_to_num(arg0s, nan=0.0 , posinf=10, neginf =-10) 
                arg0s_avg = torch.mean(arg0s,0)
                mask_a0_avg_list.append(arg0s_avg)

                arg1s = torch.index_select(torch.squeeze(transformer_output_masked[batch_ind,:,:]), 0, A1_indices)
                arg1s_isnan = torch.sum( arg1s.isnan())
                if arg1s_isnan >0: 
                    arg1s = torch.nan_to_num(arg1s, nan=0.0 , posinf=10, neginf =-10) 
                arg1s_avg = torch.mean(arg1s,0)
                mask_a1_avg_list.append(arg1s_avg)

                va0a1_label = mask_label_[batch_ind]
                avg_list = [verbs_avg, arg0s_avg,arg1s_avg]
                ss_srl = avg_list[va0a1_label]
                ss_va0a1.append(ss_srl)
 
            ss_va0a1 = torch.stack(ss_va0a1)
            if torch.sum( ss_va0a1.isnan()) >0:  
                ss_va0a1 = torch.nan_to_num(ss_va0a1, nan=0.0 , posinf=10, neginf =-10)    

            self_supervise_task_labels_masked = self.maskhead(ss_va0a1)

        loss = None
        cos_lossv=None
        if labels != None:
            pooledv = features_
            if self.config.loss == 'margin' : 
                
                
                dist = ((pooledv.unsqueeze(1) - pooledv.unsqueeze(0)) ** 2).mean(-1) 
                
                mask = (labels.unsqueeze(1) == labels.unsqueeze(0)).float()
                mask = mask - torch.diag(torch.diag(mask))
                neg_mask = (labels.unsqueeze(1) != labels.unsqueeze(0)).float()
                max_dist = (dist * mask).max()
                cos_lossv = (dist * mask).sum(-1) / (mask.sum(-1) + 1e-3) + (F.relu(max_dist - dist) * neg_mask).sum(-1) / (neg_mask.sum(-1) + 1e-3)
                cos_lossv = cos_lossv.mean() 

                
            elif self.config.loss == 'scl':
                norm_pooledv = F.normalize(pooledv, dim=-1)
                cosine_scorev = torch.exp(norm_pooledv @ norm_pooledv.t() / 0.3)
                maskv = (labels.unsqueeze(1) == labels.unsqueeze(0)).float()
                cosine_scorev = cosine_scorev - torch.diag(torch.diag(cosine_scorev))
                maskv = maskv - torch.diag(torch.diag(maskv))
                cos_lossv = cosine_scorev / cosine_scorev.sum(dim=-1, keepdim=True)
                cos_lossv = -torch.log(cos_lossv + 1e-5)
                cos_lossv = (mask * cos_lossv).sum(-1) / (mask.sum(-1) + 1e-3)
                cos_lossv = cos_lossv.mean() 
   
            if self.num_labels == 1:
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
                
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))   
     
            
            self_super_loss_fct3way = CrossEntropyLoss()
            self_super_loss =self_super_loss_fct3way(self_supervise_task_labels_masked.view(-1,3), mask_label_.view(-1))
            cos_losses=cos_lossv
   
            L= loss.item()
            CL=cos_losses.item()
            SL = self_super_loss.item() 
            self.ssLoss=SL
            loss = self.beta*loss + cos_losses + self.theta*self_super_loss

        output = (logits,) + outputs[2:]
        output = output + (pooled,)
        return ((loss,  cos_losses) + output) if (not(loss == None)) else output





    def compute_ood(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
        SRL_verb=None,
        SRL_arg0=None,
        SRL_arg1=None
    ):

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            return_dict = True,
        )
        roberta_last_hidden_states =outputs['last_hidden_state']
        roberta_sigmoid_last_hidden_states = torch.sigmoid(roberta_last_hidden_states)
        
        truncate_length = 512
        ood_keys = None
        roberta_sigmoid_last_hidden_states = roberta_sigmoid_last_hidden_states.clone().detach()
        if not ( SRL_verb == None ):
            transformer_output = self.SelfSuperT( roberta_sigmoid_last_hidden_states, SRL_verb, SRL_arg0, SRL_arg1, [] ,10)

        logits, pooled = self.classifier(transformer_output[:,0,:])
        
        mask_len = transformer_output.shape[0]
        mask_labelz = list(np.random.randint(3, size=(mask_len)))
   
        if not ( SRL_verb == None):
            new_SRL_verb=[]
            new_SRL_arg0=[]
            new_SRL_arg1=[]
            Union=[]
            for bs in range(len( SRL_verb)):
                Vs = SRL_verb[bs]
                A0s= SRL_arg0[bs]
                A1s= SRL_arg1[bs]
                num_probabilistic_selection_Vs = math.ceil(len(Vs)*self.masking_probability)
                Vs_probabilistic_selection = random.sample(Vs, num_probabilistic_selection_Vs)
                new_SRL_verb.append( Vs_probabilistic_selection )

                num_probabilistic_selection_A0s = math.ceil(len(A0s)*self.masking_probability)
                A0s_probabilistic_selection = random.sample(A0s, num_probabilistic_selection_A0s)
                new_SRL_arg0.append( A0s_probabilistic_selection )

                num_probabilistic_selection_A1s = math.ceil(len(A1s)*self.masking_probability)
                A1s_probabilistic_selection = random.sample(A1s, num_probabilistic_selection_A1s)
                new_SRL_arg1.append( A1s_probabilistic_selection )
                    
                Union.append(Vs_probabilistic_selection+A0s_probabilistic_selection+A1s_probabilistic_selection)
            transformer_output = self.SelfSuperT( roberta_sigmoid_last_hidden_states, Union, Union, Union,[],truncate_length )
            
            pooled_A0=[]
            pooled_V=[]
            pooled_A1=[]

            max_len=transformer_output.shape[1]
            for ci in range(len(mask_labelz)):
                masked_SRLs_batchA0 =new_SRL_arg0 
                masked_SRLs_batchV = new_SRL_verb
                masked_SRLs_batchA1 =new_SRL_arg1 
                masked_SRL_seq_listA0 = masked_SRLs_batchA0[ci]
                masked_SRL_seq_listV =  masked_SRLs_batchV[ci]
                masked_SRL_seq_listA1 = masked_SRLs_batchA0[ci]
                masked_SRL_seqA0 = torch.tensor(masked_SRL_seq_listA0).int().cuda()
                if torch.sum(masked_SRL_seqA0>=(max_len-1))>0:
                    new_iA0 = []
                    for ind in masked_SRL_seq_listA0:
                        if ind <(max_len-1):
                            new_iA0.append(ind)
                    masked_SRL_seqA0 = torch.tensor(new_iA0).int().cuda()
                    
                masked_SRL_seqV = torch.tensor(masked_SRL_seq_listV).int().cuda()
                if torch.sum(masked_SRL_seqV>=(max_len-1))>0:
                    new_iV = []
                    for ind in masked_SRL_seq_listV:
                        if ind <(max_len-1):
                            new_iV.append(ind)
                    masked_SRL_seqV = torch.tensor(new_iV).int().cuda()
                    
                masked_SRL_seqA1 = torch.tensor(masked_SRL_seq_listA1).int().cuda()
                if torch.sum(masked_SRL_seqA1>=(max_len-1))>0:
                    new_iA1 = []
                    for ind in masked_SRL_seq_listA1:
                        if ind <(max_len-1):
                            new_iA1.append(ind)
                    masked_SRL_seqA1 = torch.tensor(new_iA1).int().cuda()
     
                orig_tokensA0 = torch.index_select(torch.squeeze(transformer_output[ci,:,:]), 0, masked_SRL_seqA0)
                if torch.sum( orig_tokensA0.isnan()) >0:
                    orig_tokensA0 = torch.nan_to_num(orig_tokensA0, nan=0.0, posinf=10, neginf =-10)
                
                orig_tokensV = torch.index_select(torch.squeeze(transformer_output[ci,:,:]), 0, masked_SRL_seqV)
                if torch.sum( orig_tokensV.isnan()) >0:
                    orig_tokensV = torch.nan_to_num(orig_tokensV, nan=0.0, posinf=10, neginf =-10 )
                
                orig_tokensA1 = torch.index_select(torch.squeeze(transformer_output[ci,:,:]), 0, masked_SRL_seqA1)
                if torch.sum( orig_tokensA1.isnan()) >0:
                    orig_tokensA1 = torch.nan_to_num(orig_tokensA1, nan=0.0, posinf=10, neginf =-10)

                tokens_avgA0 = torch.mean(orig_tokensA0,0)
                tokens_avgA0 = tokens_avgA0.unsqueeze(0)
                
                tokens_avgV = torch.mean( orig_tokensV,0 )
                tokens_avgV = tokens_avgV.unsqueeze(0)

                tokens_avgA1 = torch.mean(orig_tokensA1,0)
                tokens_avgA1 = tokens_avgA1.unsqueeze(0)

                if torch.sum( tokens_avgA0.isnan()) >0:
                    tokens_avgA0 = torch.nan_to_num(tokens_avgA0, nan=0.0, posinf=10, neginf =-10)
                if torch.sum( tokens_avgV.isnan()) >0 :
                    tokens_avgV = torch.nan_to_num(tokens_avgV , nan=0.0, posinf=10, neginf =-10 )
                if torch.sum( tokens_avgA1.isnan()) >0:
                    tokens_avgA1 = torch.nan_to_num(tokens_avgA1, nan=0.0, posinf=10, neginf =-10)
                    
                pooled_A0.append(tokens_avgA0)
                pooled_V.append( tokens_avgV )
                pooled_A1.append(tokens_avgA1)
                
            pooled_A0=torch.cat(pooled_A0,dim=0)
            pooled_V =torch.cat(pooled_V ,dim=0)
            pooled_A1=torch.cat(pooled_A1,dim=0)
            pooledAVG = torch.cat([transformer_output[:,0,:],pooled_A0,pooled_V,pooledA1], dim=1)

            ood_keys = None
            
            maha_score = []
            for c in self.all_classes:
                centered_pooledAVG = pooledAVG- self.class_meanAVG[c].unsqueeze(0)
                msAVG = torch.diag(centered_pooledAVG @ self.class_varAVG @ centered_pooledAVG.t())
                maha_score.append( msAVG)

            maha_score = torch.stack(maha_score, dim=-1)
            maha_score = maha_score.min(-1)[0]
            maha_score = -maha_score
            maha_score=maha_score.cpu()  
            norm_pooledAVG = F.normalize(pooledAVG, dim=-1)
            cosine_scoreAVG = norm_pooledAVG @ self.norm_bankAVG.t()
            cosine_score_total = cosine_scoreAVG 
            cosine_score  = cosine_score_total.max(-1)[0]

        energy_score = torch.logsumexp(logits, dim=-1)

        softmax_score = F.softmax(logits, dim=-1).max(-1)[0]
        
        ood_keys = {
            'maha': maha_score.tolist(),
            'cosine': cosine_score.tolist(),
            'softmax': softmax_score.tolist(),
            'energy': energy_score.tolist(),
        }  
        return ood_keys

    def prepare_ood(self, dataloader=None):
        self.bank = None
        self.label_bank = None
        self.bankAVG = None
        
        for batch in dataloader:
            self.eval()
            for key, value in batch.items():
                if not 'SRL' in key.split('_'):
                    batch[key]=value.to(0)

            labels = batch['labels']
            
            SRL_verb=batch['SRL_verb']
            SRL_arg0=batch['SRL_arg0']
            SRL_arg1=batch['SRL_arg1']
            
            outputs = self.roberta(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                return_dict = True,
            )
            roberta_last_hidden_states =outputs['last_hidden_state']
            roberta_sigmoid_last_hidden_states = torch.sigmoid(roberta_last_hidden_states)
            truncate_length =512 
   
            transformer_output = self.SelfSuperT( roberta_sigmoid_last_hidden_states, SRL_verb, SRL_arg0, SRL_arg1, [] ,10)
            last_hidden_states=transformer_output
            logits, pooled_ = self.classifier(transformer_output[:,0,:]) 
            mask_len = transformer_output.shape[0]
            mask_labelz = list(np.random.randint(3, size=( mask_len))) 
            
            if not ( SRL_verb == None):
                new_SRL_verb=[]
                new_SRL_arg0=[]
                new_SRL_arg1=[]
                Union=[]
                for bs in range(len( SRL_verb)):
                    Vs = SRL_verb[bs]
                    A0s= SRL_arg0[bs]
                    A1s= SRL_arg1[bs]
                    num_probabilistic_selection_Vs = math.ceil(len(Vs)*self.masking_probability)
                    Vs_probabilistic_selection = random.sample(Vs, num_probabilistic_selection_Vs)
                    new_SRL_verb.append( Vs_probabilistic_selection )

                    num_probabilistic_selection_A0s = math.ceil(len(A0s)*self.masking_probability)
                    A0s_probabilistic_selection = random.sample(A0s, num_probabilistic_selection_A0s)
                    new_SRL_arg0.append( A0s_probabilistic_selection )

                    num_probabilistic_selection_A1s = math.ceil(len(A1s)*self.masking_probability)
                    A1s_probabilistic_selection = random.sample(A1s, num_probabilistic_selection_A1s)
                    new_SRL_arg1.append( A1s_probabilistic_selection )

                    Union.append(Vs_probabilistic_selection+A0s_probabilistic_selection+A1s_probabilistic_selection)
                transformer_output = self.SelfSuperT( roberta_sigmoid_last_hidden_states, Union, Union, Union,[],truncate_length )
                
                masked_A0=[]
                masked_V=[]
                masked_A1=[]
                max_len=transformer_output.shape[1]
                for ci in range(len(mask_labelz)):
                    masked_SRLs_batchA0 =new_SRL_arg0
                    masked_SRLs_batchV = new_SRL_verb
                    masked_SRLs_batchA1 =new_SRL_arg1 

                    masked_SRL_seq_listA0 = masked_SRLs_batchA0[ci]
                    masked_SRL_seq_listV = masked_SRLs_batchV[ci]
                    masked_SRL_seq_listA1 = masked_SRLs_batchA0[ci]

                    masked_SRL_seqA0 = torch.tensor(masked_SRL_seq_listA0).int().cuda()
                    if torch.sum(masked_SRL_seqA0>=(max_len-1))>0:
                        new_iA0 = []
                        for ind in masked_SRL_seq_listA0:
                            if ind <(max_len-1):
                                new_iA0.append(ind)
                        masked_SRL_seqA0 = torch.tensor(new_iA0).int().cuda()
                        
                    masked_SRL_seqV = torch.tensor(masked_SRL_seq_listV).int().cuda()
                    if torch.sum(masked_SRL_seqV>=(max_len-1))>0:
                        new_iV = []
                        for ind in masked_SRL_seq_listV:
                            if ind <(max_len-1):
                                new_iV.append(ind)
                        masked_SRL_seqV = torch.tensor(new_iV).int().cuda()
                        
                    masked_SRL_seqA1 = torch.tensor(masked_SRL_seq_listA1).int().cuda()

                    if torch.sum(masked_SRL_seqA1>=(max_len-1))>0:
                        new_iA1 = []
                        for ind in masked_SRL_seq_listA1:
                            if ind <(max_len-1):
                                new_iA1.append(ind)
                        masked_SRL_seqA1 = torch.tensor(new_iA1).int().cuda()

                    masked_tokensA0 = torch.index_select(torch.squeeze(transformer_output[ci,:,:]), 0, masked_SRL_seqA0)
                    if torch.sum( masked_tokensA0.isnan()) >0:
                        masked_tokensA0 = torch.nan_to_num(masked_tokensA0, nan=0.0, posinf=10, neginf =-10)  

                    masked_tokensV = torch.index_select(torch.squeeze(transformer_output[ci,:,:]), 0, masked_SRL_seqV)
                    if torch.sum( masked_tokensV.isnan()) >0:
                        masked_tokensV = torch.nan_to_num(masked_tokensV, nan=0.0, posinf=10, neginf =-10)

                    masked_tokensA1 = torch.index_select(torch.squeeze(transformer_output[ci,:,:]), 0, masked_SRL_seqA1)
                    if torch.sum( masked_tokensA1.isnan()) >0:
                        masked_tokensA1 = torch.nan_to_num(masked_tokensA1, nan=0.0, posinf=10, neginf =-10)

                    tokens_avgA0 = torch.mean(masked_tokensA0,0)
                    tokens_avgA0 = tokens_avgA0.unsqueeze(0)

                    tokens_avgV = torch.mean(masked_tokensV,0)
                    tokens_avgV = tokens_avgV.unsqueeze(0)

                    tokens_avgA1 = torch.mean(masked_tokensA1,0)
                    tokens_avgA1 = tokens_avgA1.unsqueeze(0)

                    if torch.sum( tokens_avgA0.isnan()) >0:
                        tokens_avgA0 = torch.nan_to_num(tokens_avgA0, nan=0.0, posinf=10, neginf =-10)
                    if torch.sum( tokens_avgV.isnan()) >0:
                        tokens_avgV = torch.nan_to_num(tokens_avgV, nan=0.0, posinf=10, neginf =-10)
                    if torch.sum( tokens_avgA1.isnan()) >0:
                        tokens_avgA1 = torch.nan_to_num(tokens_avgA1, nan=0.0, posinf=10, neginf =-10)
        
                    pooled_A0.append(tokens_avgA0)
                    pooled_V.append( tokens_avgV )
                    pooled_A1.append(tokens_avgA1)
  
                pooled_A0=torch.cat(pooled_A0,dim=0)
                pooled_V =torch.cat(pooled_V ,dim=0)
                pooled_A1=torch.cat(pooled_A1,dim=0)
                pooledAVG = torch.cat([transformer_output[:,0,:],pooled_A0,pooled_V,pooled_A1], dim=1) 

            if self.bankAVG is None:
                self.bankAVG = pooledAVG.clone().detach()
                self.label_bank = labels.clone().detach()
            else:
                bankAVG = pooledAVG.clone().detach()
                label_bank = labels.clone().detach()
                self.bankAVG = torch.cat([bankAVG, self.bankAVG], dim=0)
                self.label_bank = torch.cat([label_bank, self.label_bank], dim=0)
                          
        self.norm_bankAVG = F.normalize(self.bankAVG , dim=-1)
        d_avg =self.bankAVG.size()[-1] 
        self.all_classes = list(set(self.label_bank.tolist()))
        self.class_meanAVG = torch.zeros(max(self.all_classes) + 1, d_avg).cuda()
        for c in self.all_classes:
            self.class_meanAVG[c] = (self.bankAVG[self.label_bank == c].mean(0))
            
        centered_bankAVG = self.bankAVG - self.class_meanAVG[self.label_bank]
    
        centered_bankAVG = torch.nan_to_num(centered_bankAVG, nan=0.0).detach().cpu().numpy()
        
        precisionAVG = EmpiricalCovariance().fit(centered_bankAVG).precision_.astype(np.float32)

        self.class_varAVG = torch.from_numpy(precisionAVG).float().cuda()

