import os
import wandb
import torch
import pandas as pd

from tqdm.auto import tqdm
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
from torch.utils.data.dataloader import DataLoader
from torch.cuda.amp import GradScaler, autocast


# from local file
from utils import *
from filtering import *
from preprocess import preprocess_function, group_texts

# from huggingface🤗
from datasets import Dataset
from transformers import (AutoTokenizer,
                          AutoModelForCausalLM,
                          get_scheduler) 

from datasets.utils.logging import set_verbosity_error
set_verbosity_error()

from transformers.utils import logging
logging.set_verbosity_error()


def main(CONFIG):
    # logger
    mkdir_result_dir(CONFIG)
    sys.stdout = Logger(CONFIG)
    
    # device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device: ', device)

    # seed
    seed_everything(CONFIG['seed'])
    
    # gather all the datasets
    csv_list = get_all_datapaths(CONFIG)
    df_dataset = get_all_datasets(CONFIG, csv_list)

    # DataFrame -> huggingface🤗 Dataset 
    hf_dataset = Dataset.from_pandas(df_dataset)
    hf_dataset = hf_dataset.train_test_split(test_size=0.2, shuffle=True, seed=CONFIG['seed'])

    # tokenize dataset
    tokenized_dataset = hf_dataset.map(
        preprocess_function,
        batched=True,
        num_proc=4,
        remove_columns=hf_dataset['train'].column_names
    )

    # group dataset
    grouped_dataset = tokenized_dataset.map(group_texts, batched=True, num_proc=4)
    grouped_dataset.set_format('torch')
    
    # dataloader
    train_dataloader = DataLoader(grouped_dataset['train'], batch_size=CONFIG['batch_size'], shuffle=True)
    eval_dataloader = DataLoader(grouped_dataset['test'], batch_size=CONFIG['batch_size'], shuffle=False)

    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(CONFIG['checkpoint'], truncation=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    # model
    model = AutoModelForCausalLM.from_pretrained(CONFIG['model_path'])

    # optimizer & scheduler
    optimizer = AdamW(get_grouped_params(model, CONFIG['weight_decay']),
                      lr=CONFIG['lr'], eps=1e-8)

    num_train_epochs = CONFIG['epoch']
    num_update_steps_per_epoch = len(train_dataloader)
    num_training_steps = num_train_epochs * num_update_steps_per_epoch

    lr_scheduler = get_scheduler(
        name=CONFIG['scheduler_name'],
        optimizer=optimizer,
        num_warmup_steps=CONFIG['num_warmup_steps'],
        num_training_steps=num_training_steps,
    )
    
    train(
        CONFIG=CONFIG,
        tokenizer=tokenizer,
        model=model,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        num_train_epochs=num_train_epochs,
        num_training_steps=num_training_steps,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        device=device
    )
    
    model.save_pretrained(CONFIG['result_path'])


def get_grouped_params(model, weight_decay, no_decay=['bias', 'LayerNorm.weight']):
    """
    이 함수는 huggingface 에서 쓰는 함수를 그대로 차용한 것인데,
    weight decay 의 적용유무에 따라 params 를 분리해서 저장하도록 해줍니다.
    """
    params_with_wd, params_without_wd = [], []
    for n, p in model.named_parameters():
        if any(nd in n for nd in no_decay):
            params_without_wd.append(p)
        else:
            params_with_wd.append(p)
    return [
        {'params': params_with_wd, 'weight_decay': weight_decay},
        {'params': params_without_wd, 'weight_decay': 0.0},
    ]


def ce_loss(inputs, logits, device):
    """
    for your customizing the loss
    """
    # S: 1023, H: 51200
    # 여기서 각각 앞, 뒤를 1칸씩 삭제하는 이유는 다음 단어를 예측하도록 하기 위함입니다.
    shift_labels = inputs[:, 1:].to(device) # B X S
    shift_logits = logits[:, :-1, :].permute(0, 2, 1).to(device) # B X S X H

    # Calculate per-token loss
    loss = CrossEntropyLoss().to(device)
    return loss(shift_logits, shift_labels)


def train(CONFIG, tokenizer, model,
          train_dataloader, eval_dataloader,
          num_train_epochs, num_training_steps,
          optimizer, lr_scheduler, device):

    # wandb 를 사용한다면, generate 방식에 따른 내용을 모두 저장할 수 있도록 했습니다.
    if CONFIG['wandb']:
        wandb.init(project='bookathon', config=CONFIG)
        text_table = wandb.Table(columns=['epoch', 'method', 'prompt', 'output'])

    gradient_accumulation_steps = 8 # GPU 크기가 작아서 활용합니다.
    samples_per_step = CONFIG['batch_size']

    scaler = GradScaler(enabled=CONFIG['fp16']) # AMP 역시 GPU 와 학습 속도를 위해 사용합니다.

    model.to(device)
    model.train()
    completed_steps = 0

    for epoch in tqdm(range(num_train_epochs), desc='epoch', total=num_train_epochs, file=sys.stdout):
        print(f'\n==== Epoch: {epoch} ====\n')
        for step, batch in tqdm(
            enumerate(train_dataloader, start=1),
            total=len(train_dataloader),
            desc='batch'
        ):
            input_ids = batch['input_ids'].to(device)

            with autocast(enabled=CONFIG['fp16']):
                logits = model(input_ids).logits
                loss = ce_loss(input_ids, logits, device)
                loss = loss / gradient_accumulation_steps

            scaler.scale(loss).backward()

            if step % gradient_accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                lr_scheduler.step()
                optimizer.zero_grad()
                completed_steps += 1

            if step % 200 == 0:    
                print('\n',
                    {
                        'lr': lr_scheduler.get_last_lr(),
                        'samples': step * samples_per_step,
                        'steps': completed_steps,
                        'loss / train': loss.item() * gradient_accumulation_steps,
                    },
                    '\n',
                    sep=''
                )
                
        eval_loss, perplexity = evaluate(model, eval_dataloader, device)
        print('\n', {'loss/eval': eval_loss, 'perplexity': perplexity}, '\n', sep='')
        if CONFIG['wandb']:
            wandb.log({'eval_loss': eval_loss, 'perplexity': perplexity})

        model.train()

        generating_methods = ['greedy', 'beam_search', 'top_k', 'top_p', 'contrastive_search']
        
        for method in generating_methods:
            prompt, output = generate_with_prompt(model=model, tokenizer=tokenizer, 
                                                  prompt=CONFIG['prompt'], method=method,
                                                  device=device)
            
            print('\n', f'epoch: {epoch}, prompt: {prompt}, method: {method} \n model: {output}', '\n', sep='')

            if CONFIG['wandb']:
                text_table.add_data(epoch, method, prompt, output)

        model.train()
    
    if CONFIG['wandb']:
        wandb.log({'result': text_table})


def evaluate(model, eval_dataloader, device):
    model.eval()
    losses = []
    for step, batch in enumerate(eval_dataloader):
        input_ids = batch['input_ids'].to(device)
        with torch.no_grad():
            outputs = model(input_ids, labels=input_ids)

        losses.append(outputs.loss)

    loss = torch.mean(torch.stack(losses))

    try:
        perplexity = torch.exp(loss)

    except OverflowError:
        perplexity = float('inf')

    return loss.item(), perplexity.item()


def generate_with_prompt(model, tokenizer, prompt, method, device, max_length=256):
    """
    If you want to study more about decoding methods,
    You'd better find 'inference.py'.
    """
    model.eval()

    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)

    if method == 'greedy':
        output = model.generate(input_ids,
                                max_length=max_length,
                                no_repeat_ngram_size=2,                                
                                early_stopping=True)
    
    elif method == 'beam_search':
        output = model.generate(input_ids,
                                max_length=max_length,
                                num_beams=5,
                                no_repeat_ngram_size=2,
                                do_sample=True,
                                early_stopping=True)

    elif method == 'top_k':
        output = model.generate(input_ids,
                                max_length=max_length,
                                do_sample=True,
                                top_k=30)

    elif method == 'top_p':
        output = model.generate(input_ids,
                                max_length=max_length,
                                do_sample=True,
                                top_k=30,
                                top_p=0.9)
        
    elif method == 'contrastive_search':
        output = model.generate(input_ids,
                                max_length=max_length,
                                penalty_alpha=0.5,
                                top_k=5)
        

    output = tokenizer.decode(output[0], skip_special_tokens=True)
    output = output[len(prompt):]

    return prompt, output


if __name__ == '__main__':
    # get config
    CONFIG = get_config()
    main(CONFIG)
