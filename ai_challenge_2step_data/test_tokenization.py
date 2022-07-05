import unittest

from utils import *


class TestTokenizationMethods(unittest.TestCase):
    def test_convert_num_int(self):
        tokenizer = CustomTokenizer()

        #
        # Sample 1
        #
        text = '상자안에 9개의 공이 있습니다. 석진이가 5개의 공을 상자 안에 더 넣었습니다. 상자 안에 있는 공은 모두 몇 개입니까?'
        output_text, data = tokenizer.tokenize(text)

        expected_text = '상자안에 num0 개의 공이 있습니다 nae0 이가 num1 개의 공을 상자 안에 더 넣었습니다 상자 안에 있는 공은 모두 몇 개입니까'
        expected_data = {'nae0': '석진', 'num0': 9, 'num1': 5}

        self.assertEqual(expected_text, output_text)
        self.assertEqual(expected_data, data)

        #
        # Sample 2
        #
        text = '지민, 정국, 태형이의 수학 점수는 각각 94점, 82점, 88점입니다. 이 셋을 제외한 학급의 수학 점수 평균은 78점입니다. 지민이네 학급 인원수가 30명일 때, 학급 수학 평균 점수는 몇 점입니까?'
        output_text, data = tokenizer.tokenize(text)

        expected_text = 'nae0 nae1 nae2 이의 sbj0 점수는 각각 num0 점 num1 점 num2 점입니다 이 셋을 제외한 학급의 sbj0 점수 평균은 num3 점입니다 nae0 이네 학급 인원수가 num4 명일 때 학급 sbj0 평균 점수는 몇 점입니까'
        expected_data = {'nae0': '지민', 'nae1': '정국', 'nae2': '태형', 'num0': 94, 'num1': 82, 'num2': 88, 'num3': 78, 'num4': 30, 'sbj0': '수학'}

        self.assertEqual(expected_text, output_text)
        self.assertEqual(expected_data, data)

        #
        # Sample 3
        #
        # text = '4개의 수 53, 98, 69, 84가 있습니다. 그 중에서 가장 큰 수와 가장 작은 수의 차는 얼마입니까?'
        # output_text, data = tokenizer.tokenize(text)
        #
        # expected_text = 'num0 개의 수 num1 num2 num3 num4 가 있습니다 그 중에서 가장 큰 수와 가장 작은 수의 차는 얼마입니까 ?'
        # expected_data = {'num0': 4, 'num1': 53, 'num2': 98, 'num3': 69, 'num4': 84}
        #
        # self.assertEqual(expected_text, output_text)
        # self.assertEqual(expected_data, data)

    def test_convert_num_float(self):
        tokenizer = CustomTokenizer()
        #
        # Sample 1
        #
        text = '어떤 소수의 소수점을 오른쪽으로 한 자리 옮기면 원래보다 2.7만큼 커집니다. 원래의 소수를 구하시오.'
        output_text, data = tokenizer.tokenize(text)

        expected_text = '어떤 소수의 소수점을 오른쪽으로 dig0 자리 옮기면 원래보다 num0 만큼 커집니다 원래의 소수를 구하시오'
        expected_data = {'dig0': 1, 'num0': 2.7}

        self.assertEqual(expected_text, output_text)
        self.assertEqual(expected_data, data)

    def test_convert_num_hangul(self):
        tokenizer = CustomTokenizer()

        text = '정국이는 중간고사에서 국어, 수학, 영어의 점수가 각각 90점, 80점, 100점이다. 정국이의 세 과목의 점수의 평균을 구하시오.'
        output_text, data = tokenizer.tokenize(text)

        expected_text = 'nae0 이는 중간고사에서 sbj0 sbj1 sbj2 의 점수가 각각 num0 점 num1 점 num2 점이다 nae0 이의 num3 과목의 점수의 평균을 구하시오'
        expected_data = 3

        self.assertEqual(expected_text, output_text)
        self.assertEqual(expected_data, data['num3'])

        #
        # 2
        #
        text = '석진과 유정이는 가지고 있던 돈을 합하여 7300원짜리 조립장난감을 샀습니다. 석진이가 낸 돈은 유정이가 낸 돈의 2배보다 700원이 더 많습니다. 유정이가 낸 돈은 얼마일까요?'
        output_text, data = tokenizer.tokenize(text)

        expected_text = 'nae0 과 nae1 이는 가지고 있던 돈을 합하여 num0 원짜리 조립장난감을 샀습니다 nae0 이가 낸 돈은 nae1 이가 낸 돈의 num1 배보다 num2 원이 더 많습니다 nae1 이가 낸 돈은 얼마일까요'
        expected_data = {
            'nae0':'석진',
            'nae1':'유정',
            'num0':7300,
            'num1':2,
            'num2':700
        }

        self.assertEqual(expected_text, output_text)
        self.assertEqual(expected_data, data)

        #
        # 3
        #
        text = '무게가 같은 9개의 추의 무게가 4kg 500g입니다. 이 추 2개와 무게가 850g인 필통 몇 개를 양팔 저울의 한 쪽에 올려놓고, 다른 쪽에는 무게가 1050g인 사전을 5권 올려놓았더니 수평을 이루었습니다. 올려놓은 필통은 몇 개일까요?'
        output_text, data = tokenizer.tokenize(text)

        expected_text = '무게가 같은 num0 개의 추의 무게가 num1 kg num2 g 입니다 이 추 num3 개와 무게가 num4 g 인 필통 몇 개를 양팔 저울의 한 쪽에 올려놓고 다른 쪽에는 무게가 num5 g 인 사전을 num6 권 올려놓았더니 수평을 이루었습니다 올려놓은 필통은 몇 개일까요'
        expected_data = {'num0': 9, 'num1': 4, 'num2': 500, 'num3': 2, 'num4': 850, 'num5': 1050, 'num6': 5}

        self.assertEqual(expected_text, output_text)
        self.assertEqual(expected_data, data)

        #
        # 4
        #
        text = '봉사 동아리 회원 4명의 평균 나이는 14살이다. 새로운 회원 한 명이 더 들어와서 나이의 평균이 1살 늘어났습니다. 회원이 더 들어오고 나서 전체 회원 나이의 총합은 몇 살 늘어났습니까?'
        output_text, data = tokenizer.tokenize(text)

        expected_text = '봉사 동아리 회원 num0 명의 평균 나이는 num1 살이다 새로운 회원 num2 명이 더 들어와서 나이의 평균이 num3 살 늘어났습니다 회원이 더 들어오고 나서 전체 회원 나이의 총합은 몇 살 늘어났습니까'
        expected_data = {'num0': 4, 'num1': 14, 'num2': 1, 'num3': 1}

        self.assertEqual(expected_text, output_text)
        self.assertEqual(expected_data, data)

        #
        # 5
        #
        text = '어떤 수를 5로 나눈 후 12를 더해야 할 것을 잘못하여 5를 곱한 후 12를 빼었더니 113이 되었습니다. 처음 구하려고 한 답을 구해 보세요.'
        output_text, data = tokenizer.tokenize(text)

        expected_text = '어떤 수를 num0 로 나눈 후 num1 를 더해야 할 것을 잘못하여 num2 를 곱한 후 num3 를 빼었더니 num4 이 되었습니다 처음 구하려고 한 답을 구해 보세요'
        expected_data = {'num0': 5, 'num1': 12, 'num2': 5, 'num3': 12, 'num4': 113}

        self.assertEqual(expected_text, output_text)
        self.assertEqual(expected_data, data)

        #
        # 6
        #
        text = '5개의 수 151,130,26,156,37가 있습니다. 내림차순으로 나열했을 때 세 번째 수는?'
        output_text, data = tokenizer.tokenize(text)

        expected_text = 'num0 개의 수 seq0 가 있습니다 내림차순으로 나열했을 때 ord0 번째 수는'
        expected_data = {'num0': 5, 'seq0': [151, 130, 26, 156, 37], 'ord0': 3}

        self.assertEqual(expected_text, output_text)
        self.assertEqual(expected_data, data)

        #
        # 7
        #
        # text = '4개의 수 165,57,108,118가 있습니다. 가장 작은 수를 뺀 뒤 남은 수 중에서 가장 큰 수의 1의 자리 수와 10의자리 수의 합은?'
        text = '4개의 수 165,57,108,118가 있습니다. 가장 작은 수를 뺀 뒤 남은 수 중에서 가장 큰 수의 일의 자리 수와 십의 자리 수의 합은?'
        output_text, data = tokenizer.tokenize(text)

        expected_text = 'num0 개의 수 seq0 가 있습니다 가장 작은 수를 뺀 뒤 남은 수 중에서 가장 큰 수의 num1 의 자리 수와 num2 의 자리 수의 합은'
        expected_data = {'num0': 4, 'seq0': [165, 57, 108, 118], 'num1': 1, 'num2': 10}

        self.assertEqual(expected_text, output_text)
        self.assertEqual(expected_data, data)

    def test_convert_num_ord(self):
        tokenizer = CustomTokenizer()

        text = '놀이 공원에 들어가려고 친구들이 줄을 서 있습니다. 남준이는 앞에서 4번째, 뒤에서 5번째에 서 있습니다. 줄을 서 있는 어린이는 모두 몇 명입니까?'
        output_text, data = tokenizer.tokenize(text)

        expected_text = '놀이 공원에 들어가려고 친구들이 줄을 서 있습니다 nae0 이는 앞에서 ord0 번째 뒤에서 ord1 번째에 서 있습니다 줄을 서 있는 어린이는 모두 몇 명입니까'
        expected_data = {'nae0': '남준', 'ord0': 4, 'ord1': 5}

        self.assertEqual(expected_text, output_text)
        self.assertEqual(expected_data, data)

    def test_convert_num_seq(self):
        tokenizer = CustomTokenizer()

        #
        # 1
        #
        text = '5개의 수 105,62,116,89,43가 있습니다. 내림차순으로 나열했을 때 마지막으로 오는 수는 얼마입니까?'
        output_text, data = tokenizer.tokenize(text)

        expected_text = 'num0 개의 수 seq0 가 있습니다 내림차순으로 나열했을 때 마지막으로 오는 수는 얼마입니까'
        expected_data = {'num0': 5, 'seq0': [105, 62, 116, 89, 43]}

        self.assertEqual(expected_text, output_text)
        self.assertEqual(expected_data, data)

        #
        # 2
        #
        text = 'A, 821, 721, 621, 521 수들의 규칙을 찾고, A 안에 알맞은 수를 써넣으세요'
        output_text, data = tokenizer.tokenize(text)

        expected_text = 'seq0 수들의 규칙을 찾고 unk0 안에 알맞은 수를 써넣으세요'
        expected_data = {'seq0': ['A', 821, 721, 621, 521], 'unk0': 'A'}

        self.assertEqual(expected_text, output_text)
        self.assertEqual(expected_data, data)

        #
        # 2.2
        #
        text = '821, 721, A, 621, 521 수들의 규칙을 찾고, A에 알맞은 수를 써넣으세요'
        output_text, data = tokenizer.tokenize(text)

        expected_text = 'seq0 수들의 규칙을 찾고 unk0 에 알맞은 수를 써넣으세요'
        expected_data = {'seq0': [821, 721, 'A', 621, 521], 'unk0': 'A'}

        self.assertEqual(expected_text, output_text)
        self.assertEqual(expected_data, data)


    def test_convert_num_dig(self):
        tokenizer = CustomTokenizer()

        text = '어떤 소수의 소수점을 오른쪽으로 한 자리 옮기면 원래보다 2.7만큼 커집니다. 원래의 소수를 구하시오.'
        output_text, data = tokenizer.tokenize(text)

        expected_text = '어떤 소수의 소수점을 오른쪽으로 dig0 자리 옮기면 원래보다 num0 만큼 커집니다 원래의 소수를 구하시오'
        expected_data = {'dig0': 1, 'num0': 2.7}

        self.assertEqual(expected_text, output_text)
        self.assertEqual(expected_data, data)

    def test_convert_num_rep(self):
        tokenizer = CustomTokenizer()

        #
        # 1
        #
        text = '100개의 사탕을 태형, 남준, 윤기 3명에게 순서대로 2개씩 나누어 줍니다. 91번째 사탕을 받는 사람은 누구입니까?'
        output_text, data = tokenizer.tokenize(text)

        expected_text = 'num0 개의 foo0 을 nae0 nae1 nae2 num1 명에게 순서대로 rep0 개씩 나누어 줍니다 ord0 번째 foo0 을 받는 사람은 누구입니까'
        self.assertEqual(expected_text, output_text)

    def test_convert_color(self):
        tokenizer = CustomTokenizer()

        text = '왼쪽부터 흰색 공 1개, 노란색 공 2개, 빨간색 공 3개가 반복되어 놓여 있습니다. 58번째 공의 색깔을 쓰시오.'
        output_text, data = tokenizer.tokenize(text)

        expected_text = '왼쪽부터 col0 공 num0 개 col1 공 num1 개 col2 공 num2 개가 반복되어 놓여 있습니다 ord0 번째 공의 색깔을 쓰시오'
        self.assertEqual(expected_text, output_text)

        text = '왼쪽부터 빨간 모자를 쓴 사람 3명, 노란 모자를 쓴 사람 3명이 규칙을 가지고 있습니다. 이때 46째 색깔을 구하시오.'
        output_text, data = tokenizer.tokenize(text)
        expected_data = {'ord0': 46, 'col0': '빨간', 'col1': '노란', 'num0': 3, 'num1': 3}

        expected_text = '왼쪽부터 col0 모자를 쓴 사람 num0 명 col1 모자를 쓴 사람 num1 명이 규칙을 가지고 있습니다 이때 ord0 째 색깔을 구하시오'
        self.assertEqual(expected_text, output_text)
        self.assertEqual(expected_data, data)

    def test_convert_name(self):
        tokenizer = CustomTokenizer()

        #
        # 1
        #
        text = '100개의 사탕을 태형, 남준, 윤기, 석진 4명에게 순서대로 2 개씩 나누어 줍니다. 59번째 사탕을 받는 사람은 누구입니까?'
        output_text, data = tokenizer.tokenize(text)

        expected_text = 'num0 개의 foo0 을 nae0 nae1 nae2 nae3 num1 명에게 순서대로 rep0 개씩 나누어 줍니다 ord0 번째 foo0 을 받는 사람은 누구입니까'
        self.assertEqual(expected_text, output_text)

        #
        # 2
        #
        text = '달리기 대회에서 은지는 5번째로 들어 왔습니다. 은지의 바로 뒤에 들어온 사람이 윤기라면 윤기는 몇 등으로 들어왔습니까?'
        output_text, data = tokenizer.tokenize(text)

        expected_text = 'spt0 대회에서 nae0 는 ord0 번째로 들어 왔습니다 nae0 의 바로 뒤에 들어온 사람이 nae1 라면 nae1 는 몇 등으로 들어왔습니까'
        self.assertEqual(expected_text, output_text)


    def test_convert_unk(self):
        tokenizer = CustomTokenizer()

        text = '821, 721, A, 621, 521 수들의 규칙을 찾고, A 안에 알맞은 수를 써넣으세요'
        output_text, data = tokenizer.tokenize(text)

        expected_text = 'seq0 수들의 규칙을 찾고 unk0 안에 알맞은 수를 써넣으세요'
        expected_data = {'seq0': [821, 721, 'A', 621, 521], 'unk0': 'A'}

        self.assertEqual(expected_text, output_text)
        self.assertEqual(expected_data, data)

        text = '1, 17, 33, 49, 65와 같은 규칙에서 25번째 놓일 수와 40번째 놓일 수를 각각 A와 B라 할 때, B-A를 구하시오.'
        output_text, data = tokenizer.tokenize(text)

        expected_text = 'seq0 와 같은 규칙에서 ord0 번째 놓일 수와 ord1 번째 놓일 수를 각각 unk0 와 unk1 라 할 때 unk1 opr0 unk0 를 구하시오'
        self.assertEqual(expected_text, output_text)

        text = '어떤 값을 다음과 같이 2, 6, 10, 14, A 를 정렬하였습니다. A와 3, 5, 7, 9, 11, B를 정렬 하였을 때 B를 구하여 A-B 값은?'
        output_text, data = tokenizer.tokenize(text)

        expected_text = '어떤 값을 다음과 같이 seq0 를 정렬하였습니다 unk0 와 seq1 를 정렬 하였을 때 unk1 를 구하여 unk0 opr0 unk1 값은'
        self.assertEqual(expected_text, output_text)

    def test_preprocessing(self):
        tokenizer = CustomTokenizer()

        file_name = 'data/1.json'
        # file_name = 'data/2.json'
        # file_name = 'data/3.json'
        # file_name = 'data/3_ext.json'
        # file_name = 'data/4.json'
        # file_name = 'data/4_ext.json'
        # file_name = 'data/6.json'
        # file_name = 'data/6_ext.json'
        # file_name = 'data/7.json'
        # file_name = 'data/8_ext.json'

        data_list = load_json(file_name)

        fail_count = 0

        outputs = []
        for elem in data_list:
            question = elem['Question']
            question_conv = elem['QuestionConv']
            var_dict = elem['Numbers']

            print(f'\n{question}')

            print(var_dict)

            output_text, data = tokenizer.tokenize(question)
            print(data)

            expected = question_conv.replace('.', '')
            expected = expected.replace(',', '')
            expected = expected.replace('?', '')


            if expected != output_text:
                fail_count += 1
                output = {
                    'Expected': expected,
                    'Converted': output_text
                }
                outputs.append(output)

                print(f'\nFail Expected\t: {expected}')
                print(f'Fail Actual\t: {output_text}')
            # self.assertEqual(expected, output_text)

        if fail_count > 0:
            output_file = file_name.replace('.json', '_error.json')
            save_to_json(outputs, output_file)
            print(f'Saved error log : {output_file}')

        self.assertEqual(0, fail_count)

    def test_full_tokenizer(self):
        tokenizer = CustomTokenizer()

        # text = '원의 지름이 8 m입니다. 넓이를 구하세요. 원주율 : 3.14'
        text = '원의 지름이 8 m입니다. 3.14의 넓이를 구하세요.'
        output_text, data = tokenizer.tokenize(text)

        print(f'{output_text}')

        self.assertEqual(3.14, data['num1'])

        # output_text = tokenizer.tokenize_for_train(output_text)
        output_text = tokenizer.tokenize_for_train_v2(output_text)

        text = '정국이는 중간고사에서 국어, 수학, 영어의 점수가 각각 90점, 80점, 100점이다. 정국이의 세 과목의 점수의 평균을 구하시오.'

    def test_basic_tokenizer(self):
        text = '길이가 5m인 리본 중에서 1m 86cm를 잘라 꽃바구니를 장식했습니다. 남은 끈의 길이는 몇 cm인가요?'
        result = basic_tokenizer(text)
        expected = '길이가 5 m 인 리본 중에서 1 m 86 cm 를 잘라 꽃바구니를 장식했습니다 . 남은 끈의 길이는 몇 cm 인가요 ?'
        self.assertEqual(expected, result)

        text = '상자안에 num0 개의 공이 있습니다. 석진이가 num1 개의 공을 상자 안에 더 넣었습니다. 상자 안에 있는 공은 모두 몇 개입니까?'
        result = basic_tokenizer(text)
        expected = '상자안에 num0 개의 공이 있습니다 . 석진이가 num1 개의 공을 상자 안에 더 넣었습니다 . 상자 안에 있는 공은 모두 몇 개입니까 ?'
        self.assertEqual(expected, result)

        text = '왼쪽부터 흰색 공1개, 노란색 공23개, 빨간색 공 321개가 반복되어 놓여 있습니다. 58번째 공의 색깔을 쓰시오.'
        result = basic_tokenizer(text)
        expected = '왼쪽부터 흰색 공 1 개 , 노란색 공 23 개 , 빨간색 공 321 개가 반복되어 놓여 있습니다 . 58 번째 공의 색깔을 쓰시오 .'
        self.assertEqual(expected, result)

        text = '남준, 은지, 유나의 영어 점수는 각각 90점, 80점, 88점입니다. 이 3을 제외한 학급의 영어 점수 평균은 67점입니다. 남준이네 학급 인원수가 30명일 때, 학급 영어 평균 점수는 몇 점입니까?'
        result = basic_tokenizer(text)
        expected = '남준 , 은지 , 유나의 영어 점수는 각각 90 점 , 80 점 , 88 점입니다 . 이 3 을 제외한 학급의 영어 점수 평균은 67 점입니다 . 남준이네 학급 인원수가 30 명일 때 , 학급 영어 평균 점수는 몇 점입니까 ?'
        self.assertEqual(expected, result)









if __name__ == '__main__':
    unittest.main()
