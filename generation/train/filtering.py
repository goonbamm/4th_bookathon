import re
import unicodedata

import pandas as pd

from datetime import datetime
from kss import split_sentences

FORBIDDEN_WORD_LIST =[]

def has_forbidden_word(s):
    for fw in FORBIDDEN_WORD_LIST:
        if fw in s:
            return True
        
    return False


def filtering(df):
    def num(x):
        if not isinstance(x, int):
            x = x.replace(',', '')
            if '만' in x:
                x = float(x.replace('만', '')) * 10000
            x = int(x)
            
        return x
    
    
    def get_threshold(df, c):
        mean, std = df[c].mean(), df[c].std()
        
        # 정규분포에서 '평균 - 표준편차'에 속할 확률은 약 68% 입니다.
        threshold = mean - std
        
        return threshold
    
    print('\n' + '=' * 20)
    print('FILTERING STARTS')
    print(f'==> {len(df)}')

    # 1. eliminate NaN values
    df.dropna(axis=0, inplace=True)
    print('\nAfter eliminating NaN values')
    print(f'==> {len(df)}')
    
    # 2-1. value -> int
    try:
        df['like'] = df['like'].apply(num)
        df['comment'] = df['comment'].apply(num)
        df['subscribe'] = df['subscribe'].apply(num)
            
        # 2-2. filter by threshold
        like_threshold = get_threshold(df, 'like')
        comment_threshold = get_threshold(df, 'comment')
        subscribe_threshold = get_threshold(df, 'subscribe')

        df = df[
            (df['like'] > like_threshold) &
            (df['comment'] > comment_threshold) &
            (df['subscribe'] > subscribe_threshold)
            ]
    
    except:
        pass

    print('\nAfter filtering by several thresholds')
    print(f'==> {len(df)}')

    # 3. filter by keyword
    forbidden_list = [i for i, row in df.iterrows() if has_forbidden_word(row['content'])]
    forbidden_list += [i for i, row in df.iterrows() if has_forbidden_word(row['title'])]    
    df.drop(index=forbidden_list, inplace=True)

    print('\nAfter filtering by keywords like \'구독과 좋아요\'')
    print(f'==> {len(df)}')
    print('=' * 20 + '\n')
        
    return df


def clear_text(text):
    # full-width charater -> half-width charater
    text = unicodedata.normalize('NFC', text)

    # e-mail
    pattern = '([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)' 
    text = re.sub(pattern=pattern, repl='', string=text)

    # url
    pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    text = re.sub(pattern=pattern, repl='', string=text)
    text = re.sub(r"(http|https)?:\/\/\S+\b|www\.(\w+\.)+\S*", "", text).strip()
    text = re.sub(r"/", "", text).strip()

    # "#char"
    text = re.sub(r"#\S+", "", text).strip()
    
    # "@char"
    text = re.sub(r"@\w+", "", text).strip()

    # (content)
    pattern = r'\([^)]*\)'
    text = re.sub(pattern=pattern, repl='', string=text).strip()


    punct_mapping = {'\u200b': ' ', '…': '...', "‘": "'", "’": "'", "´": "'", "×": "x", "`": "'", '“': '"', '”': '"', '•': '.', "<":"'", ">":"'", "[":"'", "]":"'", "{":"'", "}":"'"} 

    for p in punct_mapping:
        text = text.replace(p, punct_mapping[p])

    text = re.sub(pattern='[\n]+', repl=' ', string=text)

    # remain only Korean + 문장부호 + alphabet
    hangul = re.compile('[^ a-zA-Z0-9가-힣.,?!\'\"]')
    text = hangul.sub('', text).strip()

    text = ' '.join(text.split())

    return text


def doc2sent(doc_df):
    def useful(sent):
        if len(sent) > 500: # too much long line
            return False
        
        elif sent.startswith('#'): # hash tag line
            return False

        return True
        
    """
    이 함수는 kss.split_sentences 를 사용하기 때문에
    약 1 ~ 30분 정도 소요될 수 있습니다.
    """
    
    print('\n' + '=' * 20)
    print('[def doc2sent] takes several minutes, please wait.')
    begins = datetime.now()
    print(f'begins at: {begins}\n')
    
    doc_df['content'] = doc_df['content'].apply(
        lambda x: re.sub(pattern='[\n]+', repl=' ', string=x)
        )
    
    pair_list = list()

    for i, row in doc_df.iterrows():
        # 2 sentences must be input. It means pair.
        sent_list = [s for s in split_sentences(row['content']) if useful(s)]
        pair_list += [f'{first} {second}' for first, second in zip(sent_list[:-1], sent_list[1:])]
    
    sent_df = pd.DataFrame(pair_list, columns=['content'])

    print('Thanks for waiting for me :)')
    ends = datetime.now()
    print(f'ends at: {ends}\n')
    print(f'=> TIME TAKEN: {ends - begins}')
    print('=' * 20, '\n')
        
    return sent_df
