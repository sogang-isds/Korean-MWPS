# 인공지능 그랜드 챌린지 5차 대회 2단계

그랜드 챌린드 5차대회 2단계에서 수행했던 모델과 데이터셋을 공개합니다.


## 팀명 : ISDSMATH 

## 프로젝트 소개
인공지능을 이용하여 수학문제를 AI 모델을 통해 전처리

풀이과정과 답을 파이썬 코드로 제공

- ai_challenge_2step_data : 데이터 전처리

- Graph2Tree_kor : Ko-Graph2Tree 모델링

- main.py : 학습한 모델로 추론

## 예시
``python main.py``
```
#문제샘플 : "4, 2, 1 중에서 서로 다른 숫자 2개를 뽑아 만들 수 있는 가장 큰 두 자리 수를 구하시오."

#출력 : 
seq0 = [4, 2, 1]
num0 = 2
num1 = 2
import itertools
result0 = [int(''.join(map(str, a))) for a in itertools.permutations(seq0, 2) if a[0] != 0]
result1 = sorted(result0, reverse=True)[1 - 1]
output = None
if type(result1) == int or type(result1) == float:
    input_int = int(result1)
    if input_int != result1:
        output = round(result1, 2)
    else:
        output = input_int
else:
    output = result1
final_result = output

print(final_result)
```
```
### Execution Result ###
42
```


## Requirements
``Python 3.7``

``>= PyTorch 1.0.0``

``transformers==3.0.2``


자세한 사항은 requirements.txt 참조.
