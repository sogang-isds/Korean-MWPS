import random
import sys
sys.path.append('../../ai_challenge_2step_data')
from utils import CustomTokenizer, save_to_json, load_json
from data_utils import convert_func_to_binary,generate_code_from_binary

tokenizer = CustomTokenizer()
random.seed(42)

files = {
    '1': ['1.json'],
    '2': ['2.json'],
    '3': ['3.json'],
    '4': ['4.json'],
    '5':['5.json'],
    '6': ['6.json'],
    '7':['7.json'],
    '8': ['8.json']
}

def split_data(train_data):
    data_list = []

    count = round(len(train_data) * 0.8)
    for elem in train_data:
        output = {}

        question = elem['Question']
        #question_conv = elem['QuestionConv']
        #var_dict = elem['Numbers']
        equation = elem['Equation']
        answer = elem['Answer']
        # new_elem = {}
        elem['trg'] = convert_func_to_binary(equation)

        output_text, data = tokenizer.tokenize(question)
        # for num in var_dict.values():
        #     if num not in data.values():
        #         print("\norigin:", var_dict, "tokened", data)
        #         break
        elem['Numbers']=data

        output_codes = generate_code_from_binary(convert_func_to_binary(equation).split(), data)
        run_code = '\n'.join(output_codes)
        exec_vars = {}
        exec(run_code, None, exec_vars)
        if exec_vars['final_result']!=answer:
            print("error!!!!!!",question)
            continue


        elem['src']  = tokenizer.tokenize_for_train_v3(output_text)
        elem['group_num']  = tokenizer.get_group_num(elem['src'])


        data_list.append(elem)

    random.shuffle(data_list)
    train_list = data_list[:count]
    test_list = data_list[count:]

    #return train_list,test_list
    return data_list


def save_data(data_list, type='train'):
    with open(f'{type}.json', 'w', encoding='utf-8') as f:
        for line in data_list:
            f.write(line)


if __name__ == '__main__':
    train_list = []
    test_list = []
    data = []
    for key, file_list in files.items():
        data_list = []
        for file in file_list:
            data_list += load_json(file)

        print(f'Chapter {key} : {len(data_list)}')
        data += split_data(data_list)
        #train,test = split_data(data_list)
        #train_list += train
        #test_list += test

        # 챕터별로 테스트 데이터 저장
        #save_to_json(test, f'test_{key}.json')

    # train, test 저장
    print(f'len(train_list) : {len(train_list)}')
    print(f'len(test_list) : {len(test_list)}')
    print(f'len(data) : {len(data)}')
    #save_to_json(train_list, 'train.json')
    #save_to_json(test_list, 'test.json')
    save_to_json(data, f'all_preprocessed.json')

    #
    # test sheet 형식의 파일 만들기

    output = {}
    for i, test_elem in enumerate(test_list):
        elem = {'question': test_elem['Question']}
        output[i + 1] = elem

    save_to_json(output, 'test_sheet.json')
