# 인공지능 그랜드 챌린지 5차 대회

본 저장소는 인공지능 그랜드 챌린지 5차 대회를 위한 모델링이 포함되어 있습니다.

데이터셋 정의, 데이터 전처리, 토크나이저에 대한 구현은 [aichallenge-5th-model](https://github.com/sogang-isds/aichallenge-5th-model) 에서 확인하실 수 잇습니다.


## 설명


### Ko-Graph2Tree

본 모델은 MATH23K, MAWPS 데이터에서 SOTA를 달성한 Graph2Tree 모델을 한국어 데이터셋에 맞게 수정한 모델입니다.

### GloVe 추가

[한국어 임베딩 튜토리얼](https://github.com/ratsgo/embedding/releases) 사이트에서 **[word-embeddings.zip](https://drive.google.com/open?id=1yHGtccC2FV3_d6C6_Q4cozYSOgA7bG-e)** 파일을 받아  `glove.txt` 파일을 `mawps/data` 디렉토리에 추가합니다.


## Training

cross-validation setting :
* ``cd mawps``
* ``python cross_valid_mawps.py``

## 성능

![image](https://user-images.githubusercontent.com/86343047/155657065-3788672d-0041-4a57-8d65-484b5f8ea26a.png)

## Acknowledgements
We use [Graph2Tree](https://github.com/2003pro/Graph2Tree) as reference code.
