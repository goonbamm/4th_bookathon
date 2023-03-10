## 4th_bookathon ๐

<br>

๋ถ์ปคํค์ ์ฐธ๊ฐํ 'Hey, Shakesby(ํค์ด, ์ฐ์ต์ค๋น)' ํ์๋๋ค. ์ ํฌ๋ ์ด ์ด์ผ๊ธฐ๋ ์๋์ ๊ทธ๋ฆผ๊ณผ ํจ๊ป ์ ์์ฝ๋์ด ์์ต๋๋ค. ๊ทธ๋ฆผ์ openAI ์์ ์ ๊ณตํ๋ [DALLE2](https://openai.com/dall-e-2/)๋ฅผ ํ์ฉํ์ฌ ์ ์ํ์์ต๋๋ค.

<br>

![](image/logo.png)

<br>

Hey, Shakesby: [๊น๋ํ](https://github.com/brianzkim), [๋ฐ์ง์ด](https://github.com/goonbamm), [๋ณ์งํ](https://github.com/Quswlghks), [์ ์๋ฆฐ](https://github.com/seolinj), [์ ํจ๋ฆผ](https://github.com/jeonhyolim)

<br>

## Story ๐

<br>

![](image/story_abstraction.png)

<br>

<details>
<summary>์์ธํ ์ด์ผ๊ธฐ ํผ์น๊ธฐ/์ ๊ธฐ</summary>
<div markdown="1">

<br>

### ์ ๋ชฉ: ๋ํ

<br>

์ด ์ํ์ ์ ๋ชฉ์ ์ ์ถํ์จ๋์? ์๋์ด ๋ด๋ํจ์ ์ฐพ์๊ฐ๋ ๊ณผ์ ์ ์ฐ๋ฆฌ๊ฐ ์ธ์์ ์ด์๊ฐ๋ฉฐ ๊นจ๋ซ๋ ๊ฒ๊ณผ ๋ฎ์์๋ค๊ณ  ์๊ฐํ์ต๋๋ค. ๊ทธ๋์ ์ ๋ชฉ์ '๋ํ'์ด๋ผ๊ณ  ์ง์์ต๋๋ค.

<br>

### ใฑ. ์ธ์์ ์ฌํ์ผ๋ก ๋ฐ์๋ค์ธ ์ฌํ๊ฐ

<br>

![](image/traveler.png)

<br>

### ใด. ๊ฐ๋์ ์ํด ๋์์ด ๊ณ ๋ฏผํ๋ ์์ ๊ฐ

<br>

![](image/artist.png)

<br>

### ใท. ์ฌํ์ ์ฑ์ฅ์ ํฌ๋งํ๋ ๊ต์ก์

<br>

![](image/educator.png)

<br>

### ใน. ๋ฏธ์๋ฐ๋ ๊ฑธ ๋๋ ค์ํ์ง ์๋ ๋ญ๋ง์ฃผ์์

<br>

![](image/lover.png)

<br>

### ใ. ์์ ์ ์ด๋ช์ ์ฌ๋ํ๊ณ  ๊ทธ ํํ ์์์ ๋์๊ฐ๋ ์ฒ ํ๊ฐ

<br>

![](image/philosopher.png)

<br>

### ๊ฒฐ๋ง: ์์ ๋ง์ ๋ด๋ํจ์ ๊นจ๋ฌ์ ์๋

<br>

![](image/ending.png)


</div>
</details>

<br>

๊ธ์ ์๋ฌธ์ [์ฌ๊ธฐ](book/final_output/๋ํ.pdf)์์ ํ์ธํ  ์ ์์ต๋๋ค.

<br>

## Review ๐

์์ธํ [ํ๊ธฐ](https://heygeronimo.tistory.com/42)๋ ์ฌ๊ธฐ์ ๋ณด์ค ์ ์์ต๋๋ค.

<br>

## Requirements ๐

~~~
pip install numpy pandas torch
pip install kss
pip install selenium
pip install transformers datasets
pip install wandb (optional)
~~~

- [kss](https://github.com/hyunwoongko/kss)
- [huggingface๐ค](https://github.com/huggingface/transformers)
- [wandb](https://github.com/wandb/wandb)

<br>

## Usage ๐ป

<br>

### Train

~~~
python generation/train/main.py
~~~

: ์ฌ์ฉํ์ค ๋, config file ์ json ํํ๋ก ๋ง๋ค์ด์ฃผ์ธ์. ์ด๋, ์ ์ฅ ๊ฒฝ๋ก ๋ฑ์ ์ ๊ฒฝ์จ์ฃผ์ธ์.

<br>

### Test

~~~
python generation/test/inference.py --model_path your_model_path
~~~

: ์ฌ์ฉํ์ค ๋, args ์์ ์ฌ์ฉํ  ๋ชจ๋ธ ๊ฒฝ๋ก๋ฅผ ํ์ธํด์ฃผ์ธ์. ๊ตณ์ด ๋ถ๋ฆฌํ์ฌ ์์ฑํ ์ด์ ๋ ๋ํ ๋น์์ ๋์์ 5๋ช์ด์ inference ๋ฅผ ์งํํด์ผ ํด์ ๋ถ๋์ดํ๊ฒ ๋ณ๋ ์์ฑํ์์ต๋๋ค.

<br>

### Crawling

- python ์ผ๋ก ์คํํ์๋ฉด ๋ฉ๋๋ค.
- ๋ค๋ง ์ด์์ฒด์ ์ Chrome Version ์ ์ค์ํ์ฌ, driver ๋ฅผ ์ค์นํ์๊ธธ ๋ฐ๋๋๋ค.
- ์ฝ๋๋ง๋ค ๋๋ฌธ์๋ก ์ ํ ๋ณ์์ ๋ง์ถฐ ์๋ ฅํ์๋ฉด ๋ฉ๋๋ค.