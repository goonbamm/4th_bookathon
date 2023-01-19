## Model Candidates 🔥

<br>

||checkpoint|model structure|size|characteristic|License|
|:-:|:-:|:-:|:-:|:-:|:-:|
|1|[skt/kogpt2-base-v2](https://huggingface.co/skt/kogpt2-base-v2)|GPT2|Base (110M)|KoGPT2 은 대부분 이걸 finetuning 함.|CC BY-NC-SA 4.0 |
|2|[kykim/gpt3-kor-small_based_on_gpt2](https://huggingface.co/kykim/gpt3-kor-small_based_on_gpt2)|GPT2<br>(reflects GPT3 structure)|Base (110M)|KoGPT3 의 구조를 반영하고, 70GB(나무위키, 블로그 글 등) 더 많은 데이터셋으로 사전학습한 모델|Apache License 2.0|
|3|[ttop324/kogpt2novel](https://huggingface.co/ttop324/kogpt2novel)|GPT2|Base (110M)|1번 모델에 소설로 finetuning 함.|-|
|4|[ttop324/kogpt2jnovel](https://huggingface.co/ttop324/kogpt2jnovel)|GPT2|Base (110M)|일본 소설을 한글로 번역하여 finetuning 한 모델이라고 함.|-|
|5|[gogamza/kobart-base-v2](https://huggingface.co/gogamza/kobart-base-v2)|BART|Base (130M)|40GB 이상의 한국어 텍스트에 대해서 학습한 한국어 encoder-decoder 언어 모델|modified MIT|
|6|[kykim/bertshared-kor-base](https://huggingface.co/kykim/bertshared-kor-base)|encoder-decoder|Base (130M)|seq2seq모델로 encoder와 decoder를 bert-kor-base로 초기화한 다음 training을 한 것. Encoder와 decoder가 파라미터를 공유하게 함으로써 하나의 bert 모델 용량으로 seq2seq를 구현할 수 있게 되었음.|Apache License 2.0|
|7|[monologg/kobigbird-bert-base](https://huggingface.co/monologg/kobigbird-bert-base)|BigBird|Base (110M)|최대 512개의 token을 다룰 수 있는 BERT의 8배인 최대 4096개의 token을 다룸.  Full attention이 아닌 Sparse Attention을 이용하여 O(n2)에서 O(n)으로 개선함.|Apache License 2.0|
|8|[paust/pko-t5-base](https://huggingface.co/paust/pko-t5-base)|T5|Base (250M)|한국어 데이터 (나무위키, 위키피디아, 모두의말뭉치 등..) 를 T5 의 span corruption task 를 사용해서 unsupervised learning 만 적용하여 학습을 진행함.|MIT license|
