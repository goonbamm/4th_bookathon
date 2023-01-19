import os
import re
import sys
import json
import torch
import argparse

# from kss
from kss import split_sentences

# from huggingface🤗
from datasets import Dataset
from transformers import (AutoTokenizer,
                          AutoModelForCausalLM,
                          get_scheduler) 

from datasets.utils.logging import set_verbosity_error
set_verbosity_error()

from transformers.utils import logging
logging.set_verbosity_error()


def get_argments():
    parser = argparse.ArgumentParser(description='Argparser for inference')
    
    parser.add_argument('--seed', type=int, default=42, help='seed for everything')
        
    parser.add_argument('--model_path', type=str, help='where the model you want to use')
    parser.add_argument('--checkpoint', type=str, default='', help='for tokenizer')

    parser.add_argument('--fp16', action='store_true', help='')
    parser.add_argument('--result_path', type=str, default='./book', help='')

    parser.add_argument('--wandb', action='store_true', help='')
    
    args = parser.parse_args()
    
    return args


def inference(args):
    mkdir_result_dir(args)
    sys.stdout = TestLogger(args)    
    
    # device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device: ', device)

    # prompt
    # prompt = input("Please give me the 1st sentence: ")
    prompt = '마지막으로 네 운명을 사랑하라. 비록 삶이 불행하게 느껴지더라도 반드시 사랑해야 한다.'

    # seed
    seed_everything(args.seed)
    
    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, truncation_side='left')
    tokenizer.pad_token = tokenizer.eos_token
    
    # model
    model = AutoModelForCausalLM.from_pretrained(args.model_path).to(device)
    model.eval()

    # book
    book = [prompt]

    while True:
        print('-' * 5 + 'BOOK' + '-' * 5)
        for i, s in enumerate(book):
            print(f'{i}: {s}')
        
        print('-' * 10 + '\n')

        candidates = generate_next_sentence(prompt=prompt,
                                            tokenizer=tokenizer,
                                            model=model,
                                            device=device)
        
        candidates['enter'] = '\n'
        
        num2key = dict()
        for i, item in enumerate(candidates.items()):
            k, v = item            
            num2key[i+1] = k
            
            print(f'[{i+1}] - {k}')
            print(f': {v}\n')
        
        print('Choose one sentence. Give me the number: ')
        
        while True:
            sent_num = input()
            
            try:
                sent_num = int(sent_num)

                if sent_num in range(len(candidates) + 1):
                    if sent_num != 0:
                        book.append(candidates[num2key[sent_num]])
                        break

                    else:
                        print('Okay, our book is complete.')
                        return ' '.join(book)
                
                else:
                    print('Something Wrong. Please input again.\n')
                    continue

            except:
                print('Something Wrong. Please input again.\n')
                continue
        
        prompt = book[-1]


def generate_next_sentence(prompt, tokenizer, model, device, max_length=100):
    candidates = dict()
    
    input_ids = tokenizer.encode(prompt, return_tensors='pt', truncation=True, max_length=256).to(device)
    
    # 1. greedy search
    output = model.generate(input_ids,
                            max_new_tokens=max_length,
                            no_repeat_ngram_size=2,
                            early_stopping=True)
    
    candidates['greedy_search'] = output
    
    # 2. beam search
    output = model.generate(input_ids,
                            max_new_tokens=max_length,
                            num_beams=5,
                            num_return_sequences=2,
                            no_repeat_ngram_size=2,
                            do_sample=False,
                            early_stopping=True)

    candidates['beam_search'] = output

    # 3. beam sample
    output = model.generate(input_ids,
                            max_new_tokens=max_length,
                            num_beams=5,
                            num_return_sequences=2,
                            no_repeat_ngram_size=2,
                            do_sample=True,
                            early_stopping=True)

    candidates['beam_search'] = output
    
    # 4. top_k & top_p
    output = model.generate(input_ids,
                            max_new_tokens=max_length,
                            do_sample=True,
                            top_k=30,
                            top_p=0.9,
                            no_repeat_ngram_size=2,
                            num_return_sequences=3)    

    candidates['top_k_&_p_sampling'] = output
    
    # 5. contrastive search - 1
    output = model.generate(input_ids,
                            max_new_tokens=max_length,
                            penalty_alpha=0.25,
                            no_repeat_ngram_size=2,
                            top_k=5)

    candidates['contrastive_search_0.25'] = output

    # 6. contrastive search - 2
    output = model.generate(input_ids,
                            max_new_tokens=max_length,
                            penalty_alpha=0.5,
                            no_repeat_ngram_size=2,
                            top_k=5)

    candidates['contrastive_search_0.5'] = output

    # 7. contrastive search - 3
    output = model.generate(input_ids,
                            max_new_tokens=max_length,
                            penalty_alpha=0.75,
                            no_repeat_ngram_size=2,                            
                            top_k=5)

    candidates['contrastive_search_0.75'] = output
    
    # 8. locally typical sampling - 1
    output = model.generate(input_ids,
                            max_new_tokens=max_length,
                            no_repeat_ngram_size=2,
                            typical_p=0.2)

    candidates['locally_typical_sampling_0.2'] = output

    # 9. locally typical sampling - 2
    output = model.generate(input_ids,
                            max_new_tokens=max_length,
                            no_repeat_ngram_size=2,
                            typical_p=0.5)

    candidates['locally_typical_sampling_0.5'] = output    
    
    # 10. group beam search (diverse beam search)
    output = model.generate(input_ids,
                            max_new_tokens=max_length,
                            num_beams=3,
                            num_return_sequences=2,
                            no_repeat_ngram_size=2,
                            do_sample=False,
                            early_stopping=True,
                            num_beam_groups=3)

    candidates['group_beam_search'] = output

    next_sentences = dict()

    for k, v in candidates.items():
        for i in range(v.size()[0]):
            output = tokenizer.decode(v[i], skip_special_tokens=True)
            output = output[len(prompt):]
            output = re.sub('[\n]+', '\n', output)

            
            try:
                output = ' '.join([s for s in split_sentences(output)][:-1])
            
            except:
                pass

            next_sentences[f'{k}_{i}'] = output

    return next_sentences


class TestLogger(object):
    def __init__(self, args):
        self.terminal = sys.stdout
        self.log = open(os.path.join(args.result_path, 'console.log'), 'w', encoding='utf-8')
            
   
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass


def mkdir_result_dir(args):
    file_number = 0
    
    if not os.path.exists(args.result_path):
        os.mkdir(args.result_path)
    
    else:
        file_number = len(os.listdir(args.result_path))
        
    args.result_path = os.path.join(args.result_path, str(file_number))
    os.mkdir(os.path.join(args.result_path))
    
    with open(os.path.join(args.result_path, 'hyperparameter.json'), 'w', encoding='utf-8') as f:
        json.dump(vars(args), f, indent=4)


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


if __name__ == '__main__':
    args = get_argments()
    book = inference(args)
    
    with open(os.path.join(args.result_path, 'book.txt'), 'w', encoding='utf-8') as f:
        f.write(book)
