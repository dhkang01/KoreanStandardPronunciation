# Korean Standard Pronunciation Prediction

한국어 표준 발음 예측 모델입니다. KoCharELECTRA 기반의 RNN 구조를 활용하여 텍스트로부터 표준 발음을 예측합니다.

## 모델 아키텍처

본 프로젝트는 다음과 같은 구조로 구성되어 있습니다:

- **Base Model**: KoCharELECTRA-small-discriminator
- **Fine-tuning**: LoRA (Low-Rank Adaptation) 적용
- **Architecture**: 
  - ElectraWithCharEmbedding: 문자 임베딩과 발음 정보(41차원)를 결합
  - PronunciationTaggerRNN: RNN 기반 발음 태거
  - Context-aware prediction: 이전/현재/다음 음절 정보를 활용한 발음 예측

## 결과

### 평가 지표: PER (Phoneme Error Rate)

| 모델 | PER |
|------|-----|
| Dummy Model | 29.5% |
| **My Model** | **0.63%** |

Dummy Model은 입력 텍스트를 그대로 발음으로 출력하는 베이스라인 모델입니다.

## 데이터셋

- **Dataset**: KMA_dataset (`dhkang01/KMA_dataset`)
- **Split**: Train (90%) / Validation (5%) / Test (5%)
- **Training samples**: 120,000
- **Validation samples**: 1,000
- **Test samples**: 100

## 설치 및 사용

### 요구사항

```bash
pip install transformers>=4.38.0 datasets>=2.18.0 peft>=0.11.0 accelerate huggingface_hub evaluate
```

### 모델 학습

학습은 `KMA_Train.ipynb` 노트북을 참고하세요.

주요 단계:
1. 데이터셋 로드 및 전처리
2. KoCharELECTRA 토크나이저 및 모델 로드
3. LoRA 설정 및 적용
4. 발음 정보 임베딩 생성 (41차원)
5. RNN 기반 발음 태거 모델 구성
6. 학습 및 평가

### 추론

학습된 모델을 사용하여 발음을 예측할 수 있습니다:

```python
from KoCharELECTRA.tokenization_kocharelectra import KoCharElectraTokenizer
from safetensors.torch import load_file
import json

# 모델 및 토크나이저 로드
tokenizer = KoCharElectraTokenizer.from_pretrained(save_dir)
pron_model_rnn.load_state_dict(state, strict=False)

# 발음 예측
text = "안녕하세요"
inputs = tokenizer(text, return_tensors="pt")
output = pron_model_rnn.forward(**inputs)
pred_ids = pron_model_rnn.predict_chars(output["logits"])
```

## 발음 특징 벡터

모델은 각 음절을 41차원 벡터로 표현합니다:
- **초성 (Onset)**: 15차원 (14개 기본 자음 + 쌍자음 플래그)
- **중성 (Nucleus)**: 10차원 (10개 기본 모음)
- **종성 (Coda)**: 16차원 (14개 기본 자음 + 쌍자음 플래그 + 종성 없음 플래그)

## 라이선스

KoCharELECTRA는 Apache-2.0 라이선스를 따릅니다.

## 참고

- Base Model: [monologg/kocharelectra-small-discriminator](https://huggingface.co/monologg/kocharelectra-small-discriminator)
- Dataset: [dhkang01/KMA_dataset](https://huggingface.co/datasets/dhkang01/KMA_dataset)

