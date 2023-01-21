## 4th_bookathon 📚

<br>

북커톤에 참가한 'Hey, Shakesby(헤이, 셰익스비)' 팀입니다. 저희는 쓴 이야기는 아래에 그림과 함께 잘 요약되어 있습니다. 그림은 openAI 에서 제공하는 [DALLE2](https://openai.com/dall-e-2/)를 활용하여 제작하였습니다.

<br>

![](image/logo.png)

<br>

Hey, Shakesby: [김도현](https://github.com/brianzkim), [박지열](https://github.com/goonbamm), [변지환](https://github.com/Quswlghks), [전서린](https://github.com/seolinj), [전효림](https://github.com/jeonhyolim)

<br>

## Story 🌈

<br>

![](image/story_abstraction.png)

<br>

<details>
<summary>자세한 이야기 펼치기/접기</summary>
<div markdown="1">

<br>

### 제목: 동행

<br>

이 작품의 제목을 유추하셨나요? 소년이 담대함을 찾아가는 과정은 우리가 인생을 살아가며 깨닫는 것과 닮아있다고 생각했습니다. 그래서 제목은 '동행'이라고 지었습니다.

<br>

### ㄱ. 인생을 여행으로 받아들인 여행가

<br>

![](image/traveler.png)

<br>

### ㄴ. 감동을 위해 끝없이 고민하는 예술가

<br>

![](image/artist.png)

<br>

### ㄷ. 사회의 성장을 희망하는 교육자

<br>

![](image/educator.png)

<br>

### ㄹ. 미움받는 걸 두려워하지 않는 낭만주의자

<br>

![](image/lover.png)

<br>

### ㅁ. 자신의 운명을 사랑하고 그 풍파 속에서 나아가는 철학가

<br>

![](image/philosopher.png)

<br>

### 결말: 자신만의 담대함을 깨달은 소년

<br>

![](image/ending.png)


</div>
</details>

<br>

글의 원문은 [여기](book/final_output/동행.pdf)에서 확인할 수 있습니다.

<br>

## Review 📝

자세한 [후기](https://heygeronimo.tistory.com/42)는 여기서 보실 수 있습니다.

<br>

## Requirements 🔑

~~~
pip install numpy pandas torch
pip install kss
pip install selenium
pip install transformers datasets
pip install wandb (optional)
~~~

- [kss](https://github.com/hyunwoongko/kss)
- [huggingface🤗](https://github.com/huggingface/transformers)
- [wandb](https://github.com/wandb/wandb)

<br>

## Usage 💻

<br>

### Train

~~~
python generation/train/main.py
~~~

: 사용하실 때, config file 을 json 형태로 만들어주세요. 이때, 저장 경로 등을 신경써주세요.

<br>

### Test

~~~
python generation/test/inference.py --model_path your_model_path
~~~

: 사용하실 때, args 에서 사용할 모델 경로를 확인해주세요. 굳이 분리하여 작성한 이유는 대회 당시에 동시에 5명이서 inference 를 진행해야 해서 부득이하게 별도 작성하였습니다.

<br>

### Crawling

- python 으로 실행하시면 됩니다.
- 다만 운영체제와 Chrome Version 에 준수하여, driver 를 설치하시길 바랍니다.
- 코드마다 대문자로 적힌 변수에 맞춰 입력하시면 됩니다.