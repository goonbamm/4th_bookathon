from utils import get_config
from transformers import AutoTokenizer

CONFIG = get_config()
TOKENIZER = AutoTokenizer.from_pretrained(CONFIG['checkpoint'])


def preprocess_function(example):
    """
    문자열을 토큰화시켜줍니다.
    truncation 같은 조건을 넣지 않았기 때문에, 문장이 잘리지 않고 모두 토큰화됩니다.
    예시: 9823개로 이뤄진 한글 문자열 -> 4443개의 토큰
    """
    return TOKENIZER(example['content'], truncation=True)


def group_texts(examples, block_size=1024):
    """
    데이터의 길이가 천차만별이기 때문에
    GPT2 입력 길이에 맞춰서 토큰을 나누는 함수입니다.
    """
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    total_length = (total_length // block_size) * block_size
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result['labels'] = result['input_ids'].copy()
    return result
