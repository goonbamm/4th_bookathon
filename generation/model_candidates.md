## Model Candidates ๐ฅ

<br>

||checkpoint|model structure|size|characteristic|License|
|:-:|:-:|:-:|:-:|:-:|:-:|
|1|[skt/kogpt2-base-v2](https://huggingface.co/skt/kogpt2-base-v2)|GPT2|Base (110M)|KoGPT2 ์ ๋๋ถ๋ถ ์ด๊ฑธ finetuning ํจ.|CC BY-NC-SA 4.0 |
|2|[kykim/gpt3-kor-small_based_on_gpt2](https://huggingface.co/kykim/gpt3-kor-small_based_on_gpt2)|GPT2<br>(reflects GPT3 structure)|Base (110M)|KoGPT3 ์ ๊ตฌ์กฐ๋ฅผ ๋ฐ์ํ๊ณ , 70GB(๋๋ฌด์ํค, ๋ธ๋ก๊ทธ ๊ธ ๋ฑ) ๋ ๋ง์ ๋ฐ์ดํฐ์์ผ๋ก ์ฌ์ ํ์ตํ ๋ชจ๋ธ|Apache License 2.0|
|3|[ttop324/kogpt2novel](https://huggingface.co/ttop324/kogpt2novel)|GPT2|Base (110M)|1๋ฒ ๋ชจ๋ธ์ ์์ค๋ก finetuning ํจ.|-|
|4|[ttop324/kogpt2jnovel](https://huggingface.co/ttop324/kogpt2jnovel)|GPT2|Base (110M)|์ผ๋ณธ ์์ค์ ํ๊ธ๋ก ๋ฒ์ญํ์ฌ finetuning ํ ๋ชจ๋ธ์ด๋ผ๊ณ  ํจ.|-|
|5|[gogamza/kobart-base-v2](https://huggingface.co/gogamza/kobart-base-v2)|BART|Base (130M)|40GB ์ด์์ ํ๊ตญ์ด ํ์คํธ์ ๋ํด์ ํ์ตํ ํ๊ตญ์ด encoder-decoder ์ธ์ด ๋ชจ๋ธ|modified MIT|
|6|[kykim/bertshared-kor-base](https://huggingface.co/kykim/bertshared-kor-base)|encoder-decoder|Base (130M)|seq2seq๋ชจ๋ธ๋ก encoder์ decoder๋ฅผ bert-kor-base๋ก ์ด๊ธฐํํ ๋ค์ training์ ํ ๊ฒ. Encoder์ decoder๊ฐ ํ๋ผ๋ฏธํฐ๋ฅผ ๊ณต์ ํ๊ฒ ํจ์ผ๋ก์จ ํ๋์ bert ๋ชจ๋ธ ์ฉ๋์ผ๋ก seq2seq๋ฅผ ๊ตฌํํ  ์ ์๊ฒ ๋์์.|Apache License 2.0|
|7|[monologg/kobigbird-bert-base](https://huggingface.co/monologg/kobigbird-bert-base)|BigBird|Base (110M)|์ต๋ 512๊ฐ์ token์ ๋ค๋ฃฐ ์ ์๋ BERT์ 8๋ฐฐ์ธ ์ต๋ 4096๊ฐ์ token์ ๋ค๋ฃธ.  Full attention์ด ์๋ Sparse Attention์ ์ด์ฉํ์ฌ O(n2)์์ O(n)์ผ๋ก ๊ฐ์ ํจ.|Apache License 2.0|
|8|[paust/pko-t5-base](https://huggingface.co/paust/pko-t5-base)|T5|Base (250M)|ํ๊ตญ์ด ๋ฐ์ดํฐ (๋๋ฌด์ํค, ์ํคํผ๋์, ๋ชจ๋์๋ง๋ญ์น ๋ฑ..) ๋ฅผ T5 ์ span corruption task ๋ฅผ ์ฌ์ฉํด์ unsupervised learning ๋ง ์ ์ฉํ์ฌ ํ์ต์ ์งํํจ.|MIT license|
