import unittest
import inspect
import json
from data_utils import *
from utils import *
import functions

class TestCodeConversionMethods(unittest.TestCase):
    def test_convert_code(self):
        var_dict = {
            'num0': 3,
            'num1': 5,
            'num2': 7
        }

        output_codes = []
        var_codes = []

        for key, val in var_dict.items():
            var_codes.append(f'{key} = {val}')

        output_codes.extend(var_codes)
        output_codes.append('')

        #
        # func_multiply(func_add(num0, num1), num2)
        #
        code = inspect.getsource(getattr(functions, 'func_add'))
        output_code, arg0, arg1 = convert_func_to_code(code)
        output_code = output_code.replace(arg0, 'num0')
        output_code = output_code.replace(arg1, 'num1')

        output_codes.append(output_code)

        code = inspect.getsource(getattr(functions, 'func_multiply'))
        output_code, arg0, arg1 = convert_func_to_code(code)

        output_code = output_code.replace(arg0, 'result')
        output_code = output_code.replace(arg1, 'num2')
        output_codes.append(output_code)

        output_codes.append('')
        output_codes.append('print(result)')

        print('\n### Final Code ###')
        run_code = '\n'.join(output_codes)
        print(run_code)

        print('\n### Run Code ###')
        exec_vars = {}
        exec(run_code, None, exec_vars)

        print('\n### Execution Result ###')
        print(exec_vars['result'])

        output_dict = {}
        output_dict['equation'] = run_code
        output_dict['answer'] = exec_vars['result']
        print(json.dumps(output_dict))

        expected_value = 56
        actual_value = exec_vars['result']

        self.assertEqual(expected_value, actual_value)

    def test_convert_code1_1(self):
        var_dict = {
            "num0": 4,
            "num1": 14,
            "num2": 1,
        }

        predicted = 'func_multiply(func_add(num0,1),num2)'

        output_codes = generate_code(predicted, var_dict=var_dict)

        print('\n### Final Code ###')
        run_code = '\n'.join(output_codes)
        print(run_code)

        print('\n### Run Code ###')
        exec_vars = {}
        exec(run_code, None, exec_vars)

        print('\n### Execution Result ###')
        print(exec_vars['final_result'])

        expected_value = 5
        actual_value = exec_vars['final_result']

        self.assertEqual(expected_value, actual_value)


    def test_convert_code2(self):
        var_dict = {
            'num0': 5,
            'num1': 6,
            'num2': 3,
            'num3': 6,
            'num4': 3,
            'num5': 7,
        }

        predicted = 'func_multiply(func_add(num0, func_divide(num1, num2)), func_minus(func_divide(num3, num4), num5))'

        output_codes = generate_code(predicted, var_dict=var_dict)

        print('\n### Final Code ###')
        run_code = '\n'.join(output_codes)
        print(run_code)

        print('\n### Run Code ###')
        exec_vars = {}
        exec(run_code, None, exec_vars)

        print('\n### Execution Result ###')
        print(exec_vars['final_result'])

        expected_value = -35
        actual_value = exec_vars['final_result']

        self.assertEqual(expected_value, actual_value)

    def test_convert_code3(self):
        var_dict = {
            'seq': [1, 2, 3, 4, 5, 6, 7, 1, 2, 3, 4, 5, 6, 7],
            'ord': 57,
            'none': None
        }

        predicted = "func_findord(func_findbasicseq(seq, none), ord)"

        output_codes = generate_code(predicted, var_dict=var_dict)

        print('\n### Final Code ###')
        run_code = '\n'.join(output_codes)
        print(run_code)

        print('\n### Run Code ###')
        exec_vars = {}
        exec(run_code, None, exec_vars)

        print('\n### Execution Result ###')
        print(exec_vars['final_result'])

        expected_value = 1
        actual_value = exec_vars['final_result']

        self.assertEqual(expected_value, actual_value)

    def test_convert_code4(self):
        # 사과, 복숭아, 배, 참외 중에서 2가지의 과일을 골라서 사는 경우는 모두 몇 가지입니까?
        # foo0, foo1, foo2, foo3 중에서 num0 가지의 과일을 골라서 사는 경우는 모두 몇 가지입니까?

        foo0 = '사과'
        foo1 = '복숭아'
        foo2 = '배'
        foo3 = '참외'

        var_dict = {
            'obj_list': [foo0, foo1, foo2, foo3],
            'num0': 2
        }

        predicted = "func_combinations(obj_list, num0)"

        output_codes = generate_code(predicted, var_dict=var_dict)

        print('\n### Final Code ###')
        run_code = '\n'.join(output_codes)
        print(run_code)

        print('\n### Run Code ###')
        exec_vars = {}
        exec(run_code, None, exec_vars)

        print('\n### Execution Result ###')
        print(exec_vars['final_result'])

        expected_value = 6
        actual_value = exec_vars['final_result']

        self.assertEqual(expected_value, actual_value)

    def test_convert_code6_1(self):
        # 어떤 수에서 36을 빼야 하는데 잘못하여 63을 뺀 결과가 8이 나왔습니다. 바르게 계산한 결과를 구하시오.'
        # 어떤 수에서 num0 을 opr0 하는데 잘못하여 num1 을 opr1 결과가 num2 이 나왔습니다. 바르게 계산한 결과를 구하시오.

        var_dict = {
            'num0':36,
            'num1':63,
            'num2':8,
            'opr0':'-',
            'opr1':'-'
        }

        predicted = "func_rightoperator(func_transform_list(func_inverseoperator(func_transform_list(num1, num2), opr1),num0),opr0)"

        output_codes = generate_code(predicted, var_dict=var_dict)

        print('\n### Final Code ###')
        run_code = '\n'.join(output_codes)
        print(run_code)

        print('\n### Run Code ###')
        exec_vars = {}
        exec(run_code, None, exec_vars)

        print('\n### Execution Result ###')
        print(exec_vars['final_result'])

        expected_value = 35
        actual_value = exec_vars['final_result']

        self.assertEqual(expected_value, actual_value)

    def test_convert_code3_1(self):
        # 4개의 숫자 7,2,5,9를 한 번씩만 사용하여 4 자리 수를 만들려고 합니다. 만들 수 있는 4 자리 수는 모두 몇 개입니까?

        var_dict = {'num': [7, 2, 5, 9], 'num5': 4, 'none': None}

        predicted = "func_len(func_permutation_list(num,num5),none)"

        output_codes = generate_code(predicted, var_dict=var_dict)

        print('\n### Final Code ###')
        run_code = '\n'.join(output_codes)
        print(run_code)

        print('\n### Run Code ###')
        exec_vars = {}
        exec(run_code, None, exec_vars)

        print('\n### Execution Result ###')
        print(exec_vars['final_result'])

        expected_value = 24
        actual_value = exec_vars['final_result']

        self.assertEqual(expected_value, actual_value)

    def test_run_code_from_binary(self):
        # 사과, 복숭아, 배, 참외 중에서 2가지의 과일을 골라서 사는 경우는 모두 몇 가지입니까?
        # foo0, foo1, foo2, foo3 중에서 num0 가지의 과일을 골라서 사는 경우는 모두 몇 가지입니까?

        foo0 = '사과'
        foo1 = '복숭아'
        foo2 = '배'
        foo3 = '참외'

        var_dict = {
            'foo': [foo0, foo1, foo2, foo3],
            'num0': 2
        }

        predicted = "func_combinations foo num0"
        binary_list = predicted.split()
        output_codes = generate_code_from_binary(binary_list, var_dict=var_dict)

        print('\n### Final Code ###')
        run_code = '\n'.join(output_codes)
        print(run_code)

        print('\n### Run Code ###')
        exec_vars = {}
        exec(run_code, None, exec_vars)

        print('\n### Execution Result ###')
        print(exec_vars['final_result'])

        expected_value = 6
        actual_value = exec_vars['final_result']

        self.assertEqual(expected_value, actual_value)

    def test_from_json(self):
        data_list = load_json('data/1.json')
        # data_list = load_json('data/2.json')
        # data_list = load_json('data/3.json')
        # data_list = load_json('data/3_ext.json')
        # data_list = load_json('data/4.json')
        # data_list = load_json('data/6.json')
        # data_list = load_json('data/6_ext.json')
        # data_list = load_json('data/7.json')
        # data_list = load_json('data/8.json')
        # data_list = load_json('data/8_ext.json')


        for elem in data_list:
            question = elem['Question']
            var_dict = elem['Numbers']
            equation = elem['Equation']
            answer = elem['Answer']

            print(question)
            print(var_dict)
            print(equation)

            output_codes = generate_code(equation, var_dict=var_dict)

            print('\n### Final Code ###')
            run_code = '\n'.join(output_codes)

            for i, line in enumerate(run_code.split('\n')):
                print(f'{i+1} : {line}')
            # print(run_code)

            print('\n### Run Code ###')
            exec_vars = {}
            exec(run_code, None, exec_vars)

            print('\n### Execution Result ###')
            print(exec_vars['final_result'])

            expected_value = answer
            actual_value = exec_vars['final_result']

            # if type(actual_value) == int:
            #     expected_value = int(expected_value)

            self.assertEqual(expected_value, actual_value)

    def test_code_to_binary(self):
        func_text = 'func_rightoperator(func_transform_list(func_FinddigitNumber(func_SplitNumber(func_FindFirstnumber(func_removeelement(num, dig0),dig1), none),func_SplitNumber(num1,none)),func_FinddigitNumber(func_SplitNumber(func_FindFirstnumber(func_removeelement(num, dig0),dig1), none),func_SplitNumber(num2,none))), "+")'
        binary_exp = convert_func_to_binary(func_text)
        print(binary_exp)

    def test_generate_code(self):
        #
        # 최소 입력 개수인 3개보다 짧은 길이인 경우
        #
        binary_text = 'ord0'
        binary_list = binary_text.split()
        output_codes = generate_code_from_binary(binary_list)

        self.assertEqual(4, len(output_codes))

        #
        # operator만 있는 경우
        #
        binary_text = "func_combinations"
        binary_list = binary_text.split()
        output_codes = generate_code_from_binary(binary_list)

        self.assertEqual(4, len(output_codes))

        #
        # operator 가 없는 경우
        #
        binary_text = "foo num0 num1"
        binary_list = binary_text.split()
        output_codes = generate_code_from_binary(binary_list)

        self.assertEqual(4, len(output_codes))




    def test_code_run(self):
        tokenizer = CustomTokenizer()

        file = 'data/1.json'
        # file = 'data/2.json'
        # file = 'data/3.json'
        # file = 'data/3_ext.json'
        # file = 'data/4.json'
        # file = 'data/4_ext.json'
        # file = 'data/6.json'
        # file = 'data/6_ext.json'
        # file = 'data/7.json'
        # file = 'data/8.json'
        # file = 'data/8_ext.json'
        data_list = load_json(file)

        outputs = []

        for elem in data_list:
            question = elem['Question']
            var_dict = elem['Numbers']
            equation = elem['Equation']
            answer = elem['Answer']

            print(question)
            print(var_dict)
            print(equation)

            output_text, data = tokenizer.tokenize(question)

            binary_exp = convert_func_to_binary(equation)
            binary_list = binary_exp.split()

            arg_list = []
            for node in binary_list:
                if not node.startswith('func_') and node not in reserved_args:
                    arg_list.append(node)

            my_args = set(arg_list)
            print(f'my_args : {my_args}')
            print(f'data : {data}')

            for my_arg in my_args:
                if my_arg not in data:
                    print(f'finding... "{my_arg}"')
                    if my_arg == 'rep':
                        value = find_arg_rep(data)
                    else:
                        value = find_arg(my_arg, data)
                    print(f'got : {value}')
                    data[my_arg] = value

            for my_arg in my_args:
                if type(data[my_arg]) == list:
                    if 'rep' in data:
                        if type(data['rep']) != list:
                            data['rep'] = [data['rep']] * len(data[my_arg])

            print(f'data(converted) : {data}')

            output_codes = generate_code_from_binary(binary_list, var_dict=data)

            print('\n### Final Code ###')
            run_code = '\n'.join(output_codes)
            # print(run_code)

            print('\n### Run Code ###')
            exec_vars = {}
            exec(run_code, None, exec_vars)

            print('\n### Execution Result ###')
            print(exec_vars['final_result'])

            elem = {}
            elem['equation'] = run_code
            elem['answer'] = f"{exec_vars['final_result']}"

            outputs.append(elem)

            if answer == exec_vars['final_result']:
                print('Success!')
            else:
                print(f'Fail. The answer is "{answer}"')

if __name__ == '__main__':
    unittest.main()