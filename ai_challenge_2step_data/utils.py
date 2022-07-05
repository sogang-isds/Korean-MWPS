import json
import pickle
import re
from konlpy.tag import Komoran
import linecache
import sys
import pandas as pd
import os
import pprint

import torch
import torch.nn as nn
from transformers import (
        ElectraTokenizer,
        ElectraConfig,
        ElectraForSequenceClassification,
)
#koelec_tokenizer = ElectraTokenizer.from_pretrained("monologg/koelectra-base-v3-discriminator",
#                                             additional_special_tokens=["[NUM]",'num0','num1','num2','num3','num4','num5','num6','num7','num8','num9'])


def print_exception(logger=None, exit=False):
    exc_type, exc_obj, tb = sys.exc_info()
    f = tb.tb_frame
    lineno = tb.tb_lineno
    filename = f.f_code.co_filename
    linecache.checkcache(filename)
    line = linecache.getline(filename, lineno, f.f_globals)
    line = line.strip()

    msg = 'EXCEPTION IN ({}, LINE: {}, CODE: {}): {} {}'.format(filename, lineno, line, exc_type, exc_obj)

    if logger:
        logger.error(msg)
    else:
        print(msg)

    if exit:
        sys.exit(1)

    return msg


def save_to_json(data, filename='data.json'):
    if filename[-4:] != 'json':
        filename += '.json'
    with open(f'{filename}', 'w', encoding='utf-8') as fw:
        json.dump(data, fw, indent=4, ensure_ascii=False)


def load_json(data_file):
    with open(data_file, 'r', encoding='utf-8-sig') as f:
        data = json.load(f)
    return data


def save_vocab(vocab, path):
    output = open(path, 'wb')
    pickle.dump(vocab, output)
    output.close()


def load_vocab(path):
    import pickle

    with open(path, 'rb') as f:
        vocab = pickle.load(f)

    return vocab


def is_hangul(c):
    if ord('가') <= ord(c) <= ord('힣'):
        return True

    return False

def is_eng(c):
    if ord('A') <= ord(c) <= ord('Z'):
        return True

    return False
def get_key(value, data_dict):
    for key, val in data_dict.items():
        if value == val:
            return key

    return None


def basic_tokenizer(text):
    text = text.replace(',', ' , ')
    text = text.replace('+', ' + ')
    text = text.replace('-', ' - ')
    text = text.replace('*', ' * ')
    text = text.replace('=', ' = ')
    #text = text.replace('/', ' / ')

    def fraction(m):  # 매개변수로 매치 객체를 받음
        a, b = m.group().split(sep='/')
        return str(int(a) / int(b))
    text= re.sub('\d+/\d+', fraction,text)

    text = ' '.join(text.split())
    prev_hangul=False
    prev_num = False
    prev_eng = False

    output = []
    for char in text:
        if is_hangul(char):
            if not prev_hangul:
                output.append(' ')
            output.append(char)

            prev_hangul = True
            prev_num = False
        elif is_eng(char):
            output.append(' ')
            output.append(char)

            prev_eng = True
            prev_num = False
        else:
            if char.isdigit():
                if prev_hangul or prev_eng:
                    output.append(' ')
                prev_num = True

            else:
                if prev_num and (char != '.' and char != ']'):
                    output.append(' ')
                    prev_num = False

                if not prev_num and char in ['.', '?']:
                    output.append(' ')

            output.append(char)
            prev_hangul = False
            prev_eng = False

    result = ' '.join(''.join(output).split())
    if re.search('( [가-하])', result):
        result = result.replace('( ', '(')
    return result


def convert_num(text):
    """
    숫자인 부분을 num 태그로 변환

    :param text:
    :return:
    """
    idx = 0

    words = text.split()
    data = {}
    outputs = []

    for word in words:
        try:
            value = float(word)

            if '.' not in word:
                value = int(word)

            key = f'num{idx}'
            data[key] = value
            outputs.append(key)

            idx += 1

        except ValueError:
            outputs.append(word)

    return ' '.join(outputs), data


class CustomTokenizer():
    def __init__(self, name_file='ai_challenge_2step_data/names.txt', verbose=True):
        self.komoran = Komoran()
        self.verbose = verbose
        self.name_file = name_file

        # CustomTokenizer.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #
        # self.config = ElectraConfig.from_pretrained("monologg/koelectra-base-v3-discriminator",num_labels=8)
        #
        # self.model = ElectraForSequenceClassification.from_pretrained("monologg/koelectra-base-v3-discriminator",config=self.config)
        #
        # self.model.to(CustomTokenizer.device)
        #
        # try:
        #     self.model.load_state_dict(torch.load('ai_challenge_2step_data/classifier/checkpoint.pt'))
        # except:
        #     self.model.load_state_dict(torch.load('classifier/checkpoint.pt'))
        # self.criterion = nn.CrossEntropyLoss()
        #
        # self.electra_tokenizer = ElectraTokenizer.from_pretrained('monologg/koelectra-base-v3-discriminator',config=self.config)


    def get_num_hanguls(self, text):
        """
        PoS Tagger를 이용하여 관형사 부분을 추출
        :param text:
        :return:
        """
        pos_results = self.komoran.pos(text)
        output = []
        for i, (key, val) in enumerate(pos_results):
            # print(key, val)

            if val == 'MM':
                next_val = pos_results[i + 1][1]
                if 'NN' in next_val :#and pos_results[i + 1][0] not in ['봉지']:
                    output.append((pos_results[i][0], pos_results[i + 1][0]))

            elif val =='NR':
                next_val = pos_results[i + 1][1]
                if 'JK' in next_val or 'NNB' in next_val: #JK 조사 :다섯이, NNB 의존명사 : 다섯번
                    output.append((pos_results[i][0], pos_results[i + 1][0]))
        return output


    def replace_hangul_to_num(self, text, output):
        """
        추출된 관형사 목록을 이용하여 한글을 숫자로 변환
        :param text:
        :param output:
        :return:
        """
        convert_map = {
            '일의 자리': '1의 자리',
            '십의 자리': '10의 자리',
            '백의 자리': '100의 자리',
            '천의 자리': '1000의 자리',
            '만의 자리': '10000의 자리',
        }

        for key, val in convert_map.items():
            text = text.replace(key, val)

        num_hanguls = {
            '1': ['한','첫'],
            '2': ['두', '둘'],
            '3': ['세', '셋'],
            '4': ['네', '넷'],
            '5':['다섯'],
            '6': ['여섯'],
            '7': ['일곱'],
            '8': ['여덟'],
            '9': ['아홉'],
            '10': ['열']

        }

        for i, elem in enumerate(output):
            num = None
            # print(elem)
            if elem[0] == '세':
                if elem[1] == '진' or elem[1] == '현':
                    continue
            for key, hanguls in num_hanguls.items():
                if elem[0] in hanguls:
                    try:
                        if elem[0] == '한' and elem[1] == '답':  # 예외 : 처음 구하려고 한 답을 구해 보세요.
                            break
                        #if elem[0] == '한' and output[i + 1][0] == '다른':  # 예외 : 필통 몇 개를 양팔 저울의 한 쪽에 올려놓고, 다른 쪽에는 -> 한 대각선, 다른 대각선
                        #    break

                    except IndexError:
                        pass

                    num = key
                    break

            if num is None:
                continue

            #old_text = f'{elem[0]} {elem[1]}'
            #repl_text = f'{num} {elem[1]}'
            old_text = f'{elem[0]}'
            repl_text = f'{num}'
            text = text.replace(old_text, repl_text, 1)

        output_text = text

        return output_text

    def replace_fig_to_num(self, text):
        """
        추출된 관형사 목록을 이용하여 도형속 한글을 숫자로 변환
        :param text:
        :param :
        :return:
        """

        number = '일이삼사오육칠팔구'
        for word in self.komoran.nouns(text):
            if '각' in word or '면체' in word or '변형' in word or '다리꼴' in word:
                fig = word.translate(str.maketrans(number,'123456789'))
                if '십' in fig:
                    idx = fig.find('십')
                    pre = fig[idx - 1].isdigit()
                    nex = fig[idx + 1].isdigit()
                    if pre and nex:
                        fig = fig.replace(fig[idx - 1:idx + 2], str(10 * int(fig[idx - 1]) + int(fig[idx + 1])))
                    elif pre:
                        fig = fig.replace('십', '0')
                    elif nex:
                        fig = fig.replace('십', '1')
                    else:
                        fig = fig.replace('십', '10')
                text = text.replace(word,fig)
        for x in re.findall('[가-힣]각기둥', text):
            fig = x.translate(str.maketrans(number,'123456789'))
            text = text.replace(x, fig)
        return text


    def convert_name(self, text):
        # with open('./names.txt','r') as f:
        try:
            with open(self.name_file,'rt',encoding='UTF8' ) as f:
                names = re.split('[, \n]',f.read())
                names = [n for n in names if n != '']
                names = sorted(list(set(names)))
        except:
            with open('../names.txt','rt',encoding='UTF8' ) as f:
                names = re.split('[, \n]',f.read())
                names = [n for n in names if n != '']
                names = sorted(list(set(names)))

        words = text.split()

        for i, word in enumerate(words):
            for name in names:
                if word.startswith(name):
                    if name != word:
                        if word == '지우개':
                            continue
                        words[i] = word.replace(name, f'{name} ')

        text = ' '.join(words)

        words = text.split()

        idx = 0
        data = {}


        for i, word in enumerate(words):
            if word in names:
                key = get_key(word, data)

                if key is None:
                    key = f'nae{idx}'
                    data[key] = word
                    idx += 1

                words[i] = key

        return ' '.join(words), data

    def convert_unk_opr(self, text):
        """
        미지수 및 연산기호 변환
        :param text:
        :return:
        """
        from string import ascii_uppercase

        unks = list(ascii_uppercase)
        oprs = ['+', '-', '*']

        words = text.split()

        idx = 0
        idx2 = 0
        data = {}

        for i, word in enumerate(words):
            if word in unks:
                key = get_key(word, data)

                if key is None:
                    key = f'unk{idx}'
                    data[key] = word
                    idx += 1

                words[i] = key

            if word in oprs:
                key = get_key(word, data)

                if key is None:
                    key = f'opr{idx2}'
                    data[key] = word
                    idx2 += 1

                words[i] = key

        return ' '.join(words), data

    def convert_seq(self, text):
        """
        수열을 변환
        :param text:
        :return:
        """
        # pattern = r'\d+[ ]*[,]+[,\d ]+\d+'
        pattern = r'[A-Z.\d]+[ ]*[,][A-Z,.\d ]+[A-Z.\d]+'
        pattern2 = r'[()가-힣]*[ ]*[,][()가-힣, ]+[,][ ]*[()가-힣]+'
        pattern3 = r'[가-힣]+[ ]*[,][ ]*[가-힣]+[ ]*[중]'
        #수열이 3개 이상 원소로 구성

        results = re.findall(pattern, text)
        results += re.findall(pattern2, text)
        results += re.findall(pattern3, text)
        idx = 0
        data = {}

        for result in results:
            key = f'seq{idx}'

            flag = False
            output = []
            for x in result.split(' , '):
                if '.' in x:
                    try:
                        output.append(float(x))
                    except ValueError:
                        if x.count(' ') > 1:
                            if x.endswith(('때','이고')):
                                flag = False
                                output.append(x.split(' ')[0])
                            else:
                                flag = True
                            break
                        output.append(x)
                else:
                    try:
                        output.append(int(x))
                    except ValueError:
                        if x.count(' ') > 1:
                            if x.endswith(('때','이고')):
                                flag = False
                                output.append(x.split(' ')[0])
                            else:
                                flag=True
                            break
                        elif x.count(' ') == 1:
                            if x.strip().endswith('중'):
                                flag = False
                                output.append(x.split(' ')[0])
                                break
                        output.append(x)

            if flag == True:
                continue

            if type(output[0]) == str:
                if output[0].endswith(('때','지만','이고')):
                    continue

            if len(output) == 2:
                if type(output[0]) == int and type(output[1]) == str:
                    continue
                elif type(output[0]) == str and type(output[1]) == int:
                    continue

            if str(output[-1]).endswith('이가'):
                output[-1] = output[-1][:-2]

            if str(output[-1]).endswith(('을','가','의','는')):
                output[-1] = output[-1][:-1]

            data[key] = output

            #text = text.replace(result, key, 1)
            try:
                text = text.replace(' , '.join(output), key+' ', 1)
            except:
                output_str = [str(n) for n in output]
                text = text.replace(' , '.join(output_str),key+' ',1)
            #text = re.sub(f'{output[0]}[ ].+?{output[-1]}','seq0',text)

            idx += 1

        return text, data

    def convert_unk(self, text):
        """
        미지수 포함 수를 변환
        'A 12','23 B 1'
        :param text:
        :return:
        """
        pattern = r'[A-Z\d ]{2,}'
        results =[]
        for word in re.findall(pattern, text):
            if re.search("[A-Z]",word) and re.search("\d",word):
                results.append(word)

        idx = 0
        data = {}

        for result in results:
            key = f'unk{idx}'

            output = []
            for x in result.split():
                try:
                    output.append(int(x))
                except ValueError:
                    output.append(x)

            data[key] = output
            text = text.replace(result, ' '+key+' ')
            idx += 1

        return text, data

    def tokenize(self, text):
        # print(f'\noriginal : {text}') if self.verbose else None
        #
        # # 유형 분류
        # data = self.electra_tokenizer.encode(
        #         text,
        #         max_length = 512,
        #         padding='max_length',
        #         truncation=True
        # )
        #
        # self.model.eval()
        #
        # output = self.model(torch.tensor([data]).to(self.device))
        # _, pred = torch.max(output[0],1)
        #
        # cls = pred.tolist()[0]+1
        #
        # text = f'[cls{cls}] '+text
        # print(text)

        # 관형사를 한글로 변환
        output = self.get_num_hanguls(text)
        text = self.replace_hangul_to_num(text, output)

        #text = self.replace_place_name(text)
        text = self.replace_fig_to_num(text)
        text = basic_tokenizer(text)
        print(f'basic tokenizer : {text}') if self.verbose else None

        text, data_seq = self.convert_seq(text)  # 수열을 seq로 변환
        #text, data_dig = self.convert_dig(text)  # 자릿수 표시를 dig로 변환
        #text, data_rep = self.convert_rep(text)  # 반복의미를 rep로 변환
        #text, data_ord = self.convert_ord(text)  # 순서표시를 ord로 변환
        #text, data_col = self.convert_color(text)
        #text, data_sha = self.convert_shape(text)
        text, data_nae = self.convert_name(text)
        text, data_unk = self.convert_unk(text)

        #text, data_matched = self.convert_matched_value(text)  # 단순매칭 변환
        #text, data_unk = self.convert_unk_opr(text)  # 미지수(unk) 및 연산기호(opr) 변환

        text, data_num = convert_num(text)  # 숫자를 num 태그로 변환
        print(f'convert num : {text}') if self.verbose else None

        output_text = convert_text_final(text)  # 불필요한 문자 제거
        print(f'convert final : {output_text}') if self.verbose else None

        data = {}
        data.update(data_seq)
        #data.update(data_dig)
        #data.update(data_rep)
        #data.update(data_ord)
        #data.update(data_col)
        #data.update(data_sha)
        data.update(data_nae)
        #data.update(data_matched)
        data.update(data_unk)
        data.update(data_num)

        return output_text, data

    def tokenize_for_train(self, text):
        words = text.split()
        space_token = '_'

        output = []
        for word in words:
            result = re.findall(r'\d+', word)
            if len(result) == 0:
                for char in word:
                    output.append(char)
                output.append(space_token)
            else:
                output.append(word)

        if output[-1] == space_token:
            output = output[:-1]

        return ' '.join(output)

    def tokenize_for_train_v2(self, text):
        pos = self.komoran.pos(text)

        for key, tag in pos:
            if tag.startswith('NN'):
                text = text.replace(key, f' {key} ')
            elif tag.startswith('VV'):
                if len(key) > 1:
                    text = text.replace(key, f' {key} ')

        return ' '.join(text.split())

    def tokenize_for_train_v3(self, text):
        result = []
        for word in text.split():
            nouns = self.komoran.nouns(word)
            if len(nouns) > 0:
                for noun in nouns:
                    word = word.replace(noun, f' {noun} ')

            result.append(word)
        return ' '.join(result)

    # def tokenize_for_train_v4(self, text):
    #     result=koelec_tokenizer.tokenize(text)
    #     return ' '.join(result)

    def get_group_num_v2(self, text):
        words = text.split()
        group_num=[]
        leng=len(words)
        notmain=['##에','##의','##을','##를','##은','##는','##가']
        for idx, word in enumerate(words):
            if word[:3]=='num':
                for i in range(idx-4,idx+3):
                    if words[i] not in notmain:
                        group_num.append(idx)
            if idx+7>leng and '##' not in word:
                group_num.append(idx)

        group_num=sorted([x for x in set(group_num) if x>=0])
        return group_num

    def get_group_num(self, text):
        words = text.split()
        group_num = []
        main_words = ['더','까지','부터','앞', '뒤','처음','마지막','가운데', '나중','자연수','주사위','않','없','뽑','홀수', '짝수', '소수','평균','먼저','늦게','째','꼴',
                    '모서리','넓이','잘못', '가장','보다','합','차','곱','더한','빼','나누','나눈','어떤','모두','적어도','순서','몫','나머지','오른','왼','씩',
                    '제대로','바르게','원래','반복','차례','소수점','중복','약수','배','원래','바르게','실수','결과','제일','둘레','각형','면체',
                    '변','가로','세로','자리','경우','주사위','길이','맨','높','낮','중간','오름차순','내림차순','큰','작','처음','옮','커','대각선','마름모'
                      ,'평행','꼭짓점','제곱','센티미터','각','형','각뿔','각기둥','부피','모서리','밑','정','최대한','전체','나머지','남은','동안','맨','앞','뒤','사이',
                      '바로','번째','만큼','잘','못','이미','아직','아무','요일','씩','가장','뺀','연속','빨리','빠르게','차례','그러나','㎠','㎡','반지름',
                      '원주율','수평','최대공약수','최소공배수','약수','배수','미터','리터','킬로미터'
                    ]
        
        for idx, word in enumerate(words):
            if re.match('[a-z]+\d*',word):
                group_num.append(idx)
            elif word in main_words:
                group_num.append(idx)
            elif word not in ['은','는','이','가','와']: # '(주어) + 가' 에서 '가'가 동사로 인식됨
                pos_results = self.komoran.pos(word)
                for key, tag in pos_results:
                    if tag == 'SL': # 영어(cm, kg 단위)
                        group_num.append(idx)
                        break
                    elif tag=='VV': # 서술어
                        group_num.append(idx)
                        break
                    elif tag=='VA': # 형용사 : 빠른 느린 큰 작은 ...
                        group_num.append(idx)
                        break
                    elif tag.startswith('MA'): # 부사
                        group_num.append(idx)
                        break
                if word.startswith(tuple(main_words)) or word.endswith(tuple(main_words)):
                    group_num.append(idx)
        return list(set(group_num))


def convert_text_final(text):
    """
    불필요한 문자 제거
    :param text:
    :return:
    """
    output_text = text.replace('#', '')
    output_text = output_text.replace('.', '')
    output_text = output_text.replace(',', '')
    output_text = output_text.replace('?', '')
    output_text = ' '.join(output_text.split())

    return output_text

def check_group_num(filename = './data/train.json'):
    tokenizer = CustomTokenizer()

    data = load_json(filename)
    questions = []

    for d in data:
        questions.append(d['Question'])

    for q in questions:
        text, data = tokenizer.tokenize(q)
        group_num = tokenizer.get_group_num(text)
        words = text.split()
        for g in group_num:
            print(f'{g}: {words[g]}')

def generate_data_from_excel(sheet,num = 1,filename = 'raw_data.xlsx',output_file=None):
    df = pd.read_excel(filename, engine = 'openpyxl', sheet_name=f'유형{sheet}')
    df = df[df['번호'] >= num]
    pair = df[['문제','정답']]

    questions = pair['문제'].values
    answers = pair['정답'].values

    print(questions)
    print(len(questions),len(answers))

    if output_file == None:
        output_file = f'./data/{sheet}.json'

    data_list = []

    tokenizer = CustomTokenizer(name_file='./names.txt')

    for i in range(len(questions)):
        q = questions[i]
        a = answers[i]
        
        conv, data = tokenizer.tokenize(q)
        data_dict = {
                'Question': q,
                'QuestionConv': conv,
                'Numbers': data,
                'Answer': a
        }

        flag = False

        pprint.pprint(data_dict)
        func = input('Equation: ')

        while True:
            ans = input('continue to the next question? [y/n] ')
            if ans == 'y' or ans == 'Y':
                break
            else:
                ans = input('want to rewrite the equation? [y/n] ')
                if ans == 'y' or ans == 'Y':
                    continue
                else:
                    print('stop generating data.')
                    flag = True
                    break

        data_dict['Equation'] = func
        data_list.append(data_dict)

        if flag == True:
            break

    if os.path.exists(output_file):
        exist_list = load_json(output_file)
        exist_list += data_list
        save_to_json(exist_list,output_file)
    else:
        save_to_json(data_list,output_file)

    print(f'data saved into {output_file}.')

def check_convert_seq(filename='./data/all_preprocessed.json'):
    data = load_json(filename)

    tokenizer = CustomTokenizer(name_file='./names.txt')
    
    for question in data:
        print("Original Question:")
        print(question['Question'])
        print()
        print("Converted Question:")
        conv, data_dict = tokenizer.tokenize(question['Question'])
        print(conv)
        print()
        print("Data Dictionary:")
        pprint.pprint(data_dict)

        for key in data_dict.keys():
            if key.startswith('seq'):
                res = input("continue? (y/n)")
                if res == 'n':
                    return
                else:
                    break

if __name__ == "__main__":
    generate_data_from_excel(sheet=2,num=74,output_file='./data/classifier_test.json')
    # check_convert_seq('./data/7.json')
