import random
import pickle as pkl

task_to_keys = {
    "mnli": ("premise", "hypothesis"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    '20ng': ("text", None),
    'trec': ("text", None),
    'imdb': ("text", None),
    'wmt16': ("en", None),
    'multi30k': ("text", None),
}

task_to_keys = {"mnli": ("premise", "hypothesis"),"rte": ("sentence1", "sentence2"),"sst2": ("sentence", None),'20ng': ("text", None),'trec': ("text", None),'imdb': ("text", None),'wmt16': ("en", None),'multi30k': ("text", None),}



def load(task_name, tokenizer, max_seq_length=256, is_id=False):
    sentence1_key, sentence2_key = task_to_keys[task_name]
    print("Loading {}".format(task_name))
    if  task_name =='mnli':
        with open('_DSs/_MNLI.pkl','rb') as f:
            datasets = pkl.load(f)
    elif task_name =='rte':
        with open('_DSs/_RTE.pkl','rb') as f:
            datasets = pkl.load(f)
    elif task_name == 'sst2':
        with open('_DSs/_SST2.pkl','rb') as f:
            datasets = pkl.load(f)
    elif task_name == '20ng':
        with open('_DSs/_20NG.pkl','rb') as f:
            datasets = pkl.load(f)
    elif task_name == 'trec':
        with open('_DSs/_TREC.pkl','rb') as f:
            datasets = pkl.load(f)
    elif task_name == 'imdb':
        with open('_DSs/_IMDB.pkl','rb') as f:
            datasets = pkl.load(f)
    elif task_name == 'wmt16':
        with open('_DSs/_WMT16.pkl','rb') as f:
            datasets = pkl.load(f)
    elif task_name == 'multi30k':
        with open('_DSs/_MULTI30K.pkl','rb') as f:
            datasets = pkl.load(f)

    train_dataset =datasets['train']
    train_dataset = check_srl(train_dataset)
    dev_dataset  =datasets['validation']
    dev_dataset=check_srl(dev_dataset)
    test_dataset = datasets['test']
    test_dataset=check_srl(test_dataset)
    if task_name == 'imdb': 
        IMDB_og = train_dataset+dev_dataset
        IMDB_0 = []
        IMDB_1=[]
        inds = list(range(len(IMDB_og)-1))
        new_val_ind = random.sample(inds,2500)
        new_train_ind = list(set(inds)-set(new_val_ind))    
        new_val=[]
        for j in new_val_ind:
            new_val.append(IMDB_og[j])
        new_train=[]
        for i in new_train_ind:
            new_train.append(IMDB_og[i])

        train_dataset=new_train
        dev_dataset=new_val
    return train_dataset, dev_dataset, test_dataset

def load_ood(task_name, tokenizer, max_seq_length=256, is_id=False): 
    sentence1_key, sentence2_key = task_to_keys[task_name]
    print("Loading {}".format(task_name))
    if  task_name =='mnli':
        with open('_DSs/_MNLI.pkl','rb') as f:
            datasets = pkl.load(f)
    elif task_name =='rte':
        with open('_DSs/_RTE.pkl','rb') as f:
            datasets = pkl.load(f)
    elif task_name == 'sst2':
        with open('_DSs/_SST2.pkl','rb') as f:
            datasets = pkl.load(f)
    elif task_name == '20ng':
        with open('_DSs/_20NG.pkl','rb') as f:
            datasets = pkl.load(f)
    elif task_name == 'trec':
        with open('_DSs/_TREC.pkl','rb') as f:
            datasets = pkl.load(f)
    elif task_name == 'imdb':
        with open('_DSs/_IMDB.pkl','rb') as f:
            datasets = pkl.load(f)
    elif task_name == 'wmt16':
        with open('_DSs/_WMT16.pkl','rb') as f:
            datasets = pkl.load(f)
    elif task_name == 'multi30k':
        with open('_DSs/_MULTI30K.pkl','rb') as f:
            datasets = pkl.load(f)


    test_dataset = datasets['test']
    test_dataset = check_srl(test_dataset)
    return  test_dataset