import torch
import random
import numpy as np

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0 and torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)


def collate_fn(batch):
    max_len = max([len(f["input_ids"]) for f in batch])
    input_ids = [f["input_ids"] + [0] * (max_len - len(f["input_ids"])) for f in batch]
    input_mask = [[1.0] * len(f["input_ids"]) + [0.0] * (max_len - len(f["input_ids"])) for f in batch]
    labels = [f["labels"] for f in batch]
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    input_mask = torch.tensor(input_mask, dtype=torch.float)
    labels = torch.tensor(labels, dtype=torch.long)
    outputs = {
        "input_ids": input_ids,
        "attention_mask": input_mask,
        "labels": labels,
    }
    return outputs

def pad0s(lis ,tolen):
    len_ = len(lis)
    if len_<tolen:
        lis+=[0]*(tolen-len_)
    return lis

def pad1s(lis ,tolen):
    len_ = len(lis)
    if len_<tolen:
        lis+=[1]*(tolen-len_)
    return lis

def CDataLoader(dataset, batch_size, collate_fn, shuffle, drop_last=True):
    loader=[]
    if shuffle:
        random.shuffle(dataset)
    length = len(dataset)
    num_b = length//batch_size
    last_num_b=length%batch_size
    k=0
    if last_num_b>0:
        k=1
    for i in range(num_b)
        
        if i!=num_b:
            batch = dataset[i*batch_size : (i+1)*batch_size]
        else:
            batch = dataset[-last_num_b : ]
        input_ids=[]
        attention_mask=[]
        labels=[]
        SRL_verb=[]
        SRL_arg0=[]
        SRL_arg1=[]
        all_len=[]
        for entry in batch:
            all_len.append(len(entry['attention_mask']))
        max_len = max(all_len)
        if max_len>=254:
            max_len=254
        
        for dic in batch:
            selflen = len(dic['input_ids'])
            if  selflen>max_len or selflen>255:
                more = selflen-max_len
                inputs = dic['input_ids'][:-more]
                input_ids.append( pad1s( inputs,max_len))
                attentions = dic['attention_mask'][:-more]
                attention_mask.append(pad0s( attentions,max_len))
            else:
                inputs = dic['input_ids']
                input_ids.append( pad1s( inputs,max_len))
                attentions = dic['attention_mask']
                attention_mask.append(pad0s( attentions,max_len))
                
            
            labels.append(  dic['labels'] )
            V=dic['SRL_verb']
            newV=[]
            for v in V:
                if v>selflen or v>max_len:
                    pass
                else:
                    newV.append(v)
            SRL_verb.append( newV )
            
            A0 = dic['SRL_arg0']
            newA0=[]
            for a in A0:
                if a>selflen or a>max_len:
                    pass
                else:
                    newA0.append(a)
            SRL_arg0.append(A0)
            
            A1 = dic['SRL_arg1']
            newA1=[]
            for a in A1:
                if a>selflen or a>max_len:
                    pass
                else:
                    newA1.append(a)
            SRL_arg1.append( A1 )
        input_ids = torch.LongTensor(input_ids)
        attention_mask= torch.LongTensor(attention_mask)
        labels= torch.LongTensor(labels)
        Batch_is_dic = {'input_ids': input_ids, 'attention_mask':attention_mask,'labels':labels,'SRL_verb':SRL_verb,'SRL_arg0':SRL_arg0, 'SRL_arg1':SRL_arg1}
        loader.append(Batch_is_dic)
    return loader
        
def CDataLoaderDev(dataset, batch_size, collate_fn, shuffle, drop_last=True):
    loader=[]
    if shuffle:
        random.shuffle(dataset)
    length = len(dataset)
    num_b = length//batch_size
    last_num_b=length%batch_size
    k=0
    if last_num_b>0:
        k=1
    for i in range(num_b):
        if i!=num_b:
            batch = dataset[i*batch_size : (i+1)*batch_size]
        else:
            batch = dataset[-last_num_b : ]
        input_ids=[]
        attention_mask=[]
        labels=[]
        SRL_verb=[]
        SRL_arg0=[]
        SRL_arg1=[]
        all_len=[]
        for entry in batch:
            all_len.append(len(entry['attention_mask']))
        max_len = max(all_len)
        for dic in batch:
            input_ids.append( pad0s( dic['input_ids'],max_len))
            attention_mask.append(pad0s( dic['attention_mask'],max_len))
            labels.append(  dic['labels'] )
            SRL_verb.append( dic['SRL_verb'][0][0])
            SRL_arg0.append( dic['SRL_arg0'][0][0])
            SRL_arg1.append( dic['SRL_arg1'][0][0])
        input_ids = torch.Tensor(input_ids)
        attention_mask= torch.Tensor(attention_mask)
        labels= torch.Tensor(labels)
        Batch_is_dic = {'input_ids': input_ids, 'attention_mask':attention_mask,'labels':labels,'SRL_verb':SRL_verb,'SRL_arg0':SRL_arg0, 'SRL_arg1':SRL_arg1}
        loader.append(Batch_is_dic)
    return loader    