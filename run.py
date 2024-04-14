import argparse
import torch
from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader
from transformers import RobertaConfig, RobertaTokenizer, BertConfig, BertTokenizer
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from utils import set_seed, collate_fn, pad0s, CDataLoader,CDataLoaderDev 
from datasets import load_metric
from model import RobertaForSequenceClassification
from evaluation import evaluate_ood 
import warnings
from data import load , load_ood
import time
task_to_labels = {
    'sst2': 2,
    'imdb': 2,
    '20ng': 20,
    'trec': 6,
}

task_to_metric = {
    'sst2': 'sst2',
    'imdb': 'sst2',
    '20ng': 'mnli',
    'trec': 'mnli',
}


def train(args, model, train_dataset, dev_dataset, test_dataset, benchmarks):
    train_dataloader = CDataLoader(train_dataset, batch_size=args.batch_size, collate_fn=collate_fn, shuffle=True, drop_last=True)
    dev_dataloader = CDataLoader(dev_dataset, batch_size=args.batch_size, collate_fn=collate_fn,shuffle=False, drop_last=False)
    total_steps = int(len(train_dataloader) * args.num_train_epochs)
    warmup_steps = int(total_steps * args.warmup_ratio)
    no_decay = ["LayerNorm.weight", "bias"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
    ood_minfpr=100#
    ood_maxaoc=0
    results_dic={}
    score_dic={'maha':[], 'cosine':[], 'softmax':[], 'energy':[]}
    def detect_ood():
        model.prepare_ood(dev_dataloader)
        
        for tag, ood_features in benchmarks:
            print('..Evaluating ood for DataSet ', tag)
            results = evaluate_ood(args, model, test_dataset, ood_features, tag=tag)
            print("detect_ood: ", results)
        

    num_steps = 0
    results_list=[]
    ood_results_dic={}
    for epoch in range(int(args.num_train_epochs)):
        model.zero_grad()
        for step, batch in enumerate(tqdm(train_dataloader)):
            model.train()
            for key, value in batch.items():
                if not ('SRL' in key.split('_')):
                    batch[key]=value.to(0)

                else:
                    batch[key]=value
                    
            outputs = model(**batch)
            loss, cos_loss = outputs[0], outputs[1]
            loss.backward()
            num_steps += 1
            optimizer.step()
            scheduler.step()
            model.zero_grad()

        results = evaluate(args, model, dev_dataset, tag="dev")
        results = evaluate(args, model, test_dataset, tag="test")
        detect_ood()

def evaluate(args, model, eval_dataset, tag="train"):
    metric_name = task_to_metric[args.task_name]
    metric = load_metric("glue", metric_name)

    def compute_metrics(preds, labels):
        preds = np.argmax(preds, axis=1)
        result = metric.compute(predictions=preds, references=labels)
        if len(result) > 1:
            result["score"] = np.mean(list(result.values())).item()
        return result
    dataloader = CDataLoader(eval_dataset, batch_size=args.batch_size, collate_fn=collate_fn,shuffle=False, drop_last=False)

    label_list, logit_list = [], []
    for step, batch in enumerate(tqdm(dataloader)):
        model.eval()
        labels = batch["labels"].detach().cpu().numpy()
        for key, value in batch.items():
            if not ('SRL' in key.split('_')):
                batch[key]=value.to(0)
            else:
                batch[key]=value
        outputs = model(**batch)
        logits = outputs[2].detach().cpu().numpy()
        label_list.append(labels)
        logit_list.append(logits)
    preds = np.concatenate(logit_list, axis=0)
    labels = np.concatenate(label_list, axis=0)
    results = compute_metrics(preds, labels)
    print("evaluate::",results)
    results = {"{}_{}".format(tag, key): value for key, value in results.items()}
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", default="roberta-large", type=str)
    parser.add_argument("--max_seq_length", default=256, type=int)
    parser.add_argument("--task_name", default="sst2", type=str)

    parser.add_argument("--batch_size", default=12, type=int)
    parser.add_argument("--learning_rate", default=1e-5, type=float)
    parser.add_argument("--adam_epsilon", default=1e-6, type=float)
    parser.add_argument("--warmup_ratio", default=0.06, type=float)###
    parser.add_argument("--weight_decay", default=0.01, type=float)
    parser.add_argument("--num_train_epochs", default=10.0, type=float)
    parser.add_argument("--seed", type=int, default=900)
    parser.add_argument("--project_name", type=str, default="ood")
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--alpha", type=float, default=3.0)
    parser.add_argument("--theta", type=float, default=1.0)
    parser.add_argument("--loss", type=str, default="margin")
    parser.add_argument("--cuda", type=str, default="0")
    parser.add_argument("--masking_probability_train", type=float, default=0.30)
    parser.add_argument("--masking_probability", type=float, default=0.30)
    

    args = parser.parse_args()
    args.n_gpu = torch.cuda.device_count()
    args.device = device                                                
    set_seed(args)

    num_labels = task_to_labels[args.task_name]
    if args.model_name_or_path.startswith('roberta'):
        model_class = 'roberta'    
        config = RobertaConfig.from_pretrained(args.model_name_or_path, num_labels=num_labels)
        config.gradient_checkpointing = True
        config.alpha = args.alpha
        config.beta = args.beta
        config.theta = args.theta
        config.loss = args.loss
        config.cuda = args.cuda
        config.masking_probability   = args.masking_probability
        config.masking_probability_train   = args.masking_probability_train
        config.seed = args.seed
        
        tokenizer = RobertaTokenizer.from_pretrained(args.model_name_or_path)
        model = RobertaForSequenceClassification.from_pretrained(
            args.model_name_or_path, config=config,
        )
        model.to(0)
   
    datasets = ['rte', 'sst2', 'mnli', '20ng', 'trec', 'imdb', 'wmt16', 'multi30k'] 
    benchmarks = ()

    for dataset in datasets:
        if dataset == args.task_name:
            print("Loading ", dataset, " as ID")
            train_dataset, dev_dataset, test_dataset = load(dataset, tokenizer, max_seq_length=args.max_seq_length, is_id=True)
        else:
            if args.task_name=='sst2' and dataset=='imdb':
                pass
            elif args.task_name=='imdb' and dataset=='sst2':
                pass
            else:
                print("Loading ", dataset, " as OOD")
                ood_dataset = load_ood(dataset, tokenizer, max_seq_length=args.max_seq_length)
                benchmarks = (('ood_' + dataset, ood_dataset),) + benchmarks

    train(args, model, train_dataset, dev_dataset, test_dataset, benchmarks)

if __name__ == "__main__":
    main()
