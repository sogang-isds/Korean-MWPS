# 인공지능 그랜드 챌린지 5차 대회

본 저장소는 인공지능 그랜드 챌린지 5차 대회를 위한 데이터셋 정의, 데이터 전처리, 토크나이저에 대한 구현이 포함되어 있습니다.

데이터를 학습해볼 수 있는 모델은 [Graph2Tree_kor](https://github.com/sogang-isds/MathAI/tree/main/Graph2Tree_kor) 에서 확인하실 수 잇습니다.

## 설명

### 데이터 변환


```bash
cd data
python convert_data.py
```

### 파일 설명

- functions.py : 수학문제를 코드로 풀이하는데 필요한 함수 정의
- data_utils.py : 수학 equation을 python code로 변환하는데 필요한 함수 정의
- utils.py : 토크나이저 및 기타 유틸리티 코드 정의
- names.txt : 한국인의 이름 통계 서비스 등의 출처로부터 수집한 한국인 이름 데이터


### 테스트 코드

```bash
python test_binarytree.py
python test_codeconv.py
python test_tokenization.py
```

### 사용자 사전

Komoran 형태소 분석기에서 사람 이름 처리를 위해 사용자 사전을 정의하였습니다. `userdict.txt`에 정의되어 있습니다.

### 문제 유형

![image](https://user-images.githubusercontent.com/86343047/155657012-cb53f862-30c7-4bbc-893c-a080cba626e5.png)


## Copyright

본 프로젝트에 대한 저작권은 서강대학교 지능형 음성대화 인터페이스 연구실에 있습니다.

본 저장소는 비공개 저장소로 외부 유출을 금합니다.

