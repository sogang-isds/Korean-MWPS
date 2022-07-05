import re
import inspect
try:
    from ai_challenge_2step_data import functions 
except:
    import functions
from utils import load_json, save_to_json
from translation import translate

reserved_args = ['none', 'add', 'sub', 'mul', 'div', 'mod',
                         'asc', 'desc', 'first', 'last', 'middle',
                         'left', 'right', 'largest', 'smallest',
                 'one', 'two', 'three']


def find_arg(arg, data):
    key_list = data.keys()

    count = 0
    for key in key_list:
        if key.startswith(arg):
            count += 1

    if count == 1:
        return data[f'{arg}0']
    elif count == 0:
        return None
    else:
        output = []
        for i in range(count):
            try:
                output.append(data[f'{arg}{i}'])
            except KeyError:
                continue

        return output


def find_arg_rep(data):
    arg = find_arg('rep', data)

    if not arg:
        arg = find_arg('num', data)

    return arg


def convert_func_to_binary(string):
    string = string.replace('(', ' ( ')
    string = string.replace(')', ' ) ')
    string = string.replace(',', ' ')

    elems = string.split()

    results = []

    for i, elem in enumerate(elems):
        if elem != ')':
            results.append(elem)
        else:
            right = results.pop()
            left = results.pop()
            left_paren = results.pop()
            operator = results.pop()

            output = f'{operator} {left} {right}'
            results.append(output)

    return results[0]


def convert_math_to_binary(string):
    string = string.replace('(', ' ( ')
    string = string.replace(')', ' ) ')
    string = string.replace('+', ' + ')
    string = string.replace('-', ' - ')
    string = string.replace('*', ' * ')
    string = string.replace('/', ' / ')

    elems = string.split()

    results = []

    for i, elem in enumerate(elems):
        if elem != ')':
            results.append(elem)
        else:
            right = results.pop()
            operator = results.pop()
            left = results.pop()
            left_paren = results.pop()

            output = f'{operator} {left} {right}'
            results.append(output)

    while len(results) >= 3:
        left = results.pop(0)
        operator = results.pop(0)
        right = results.pop(0)

        output = f'{operator} {left} {right}'
        results.insert(0, output)

    return results[0]



def convert_func_to_code(function_code):
    results = re.findall(r'\((.+)\,(.+)\)', function_code)

    if len(results) == 0:
        raise Exception("Can't find argument")
    else:
        result = results[0]
        arg0 = result[0].strip()
        arg1 = result[1].strip()

    # print('arg0 :', arg0)
    # print('arg1 :', arg1)

    # print(code)

    code_lines = function_code.split('\n')

    if len(code_lines) < 4:
        raise Exception("Lines must be at least 4 lines")

    if code_lines[-2].strip() != 'return result':
        raise Exception("Last line must be 'return result'")

    code_lines = function_code.split('\n')[1:-2]
    # print(code_lines)

    new_lines = []
    for code_line in code_lines:
        new_lines.append(code_line[4:])

    # print(new_lines)

    new_code = '\n'.join(new_lines)
    # print(new_code)

    return new_code, arg0, arg1


def exec_operator(elems, counter, operation_dict):
    # for i, elem in enumerate(elems):
    i = 0
    outputs = []

    #
    # operator가 없는 경우 처리
    #
    has_operator = False
    for elem in elems:
        if elem.startswith('func_'):
            has_operator = True
            break

    if not has_operator:
        return

    #
    # elems 갯수는 최소 3개 이상이어야 함
    #
    if len(elems) < 3:
        return

    while True:
        if i > len(elems) - 1:
            i = 0

        elem = elems[i]

        if elem.startswith('func_'):
            outputs.append(elem)

            next_elem = elems[i + 1]

            if next_elem.startswith('func_'):
                elems = elems[:i + 1] + exec_operator(elems[i + 1:], counter, operation_dict)
                counter()
            else:
                next_elem2 = elems[i + 2]

                if not next_elem2.startswith('func_'):
                    # Calculate
                    print(f'{elem}, {next_elem}, {next_elem2} - result{counter.count}')

                    result = f'result{counter.count}'

                    operation_dict[result] = (elem, next_elem, next_elem2)

                    if len(elems) <= 3:
                        return result
                    else:
                        results = elems[:i] + [result] + elems[i + 3:]
                        return results
                else:
                    elems = elems[:i + 1] + exec_operator(elems[i + 1:], counter, operation_dict)
                    counter()
        else:

            i += 1
            continue

        i = 0


class Count():
    count = 0

    def __call__(self, *args, **kwargs):
        self.count += 1


def generate_code_from_binary(binary_list, var_dict={},do_translate=True):
    count_obj = Count()

    output_codes = []
    operation_dict = {}

    used_functions = []

    if len(var_dict) > 0:
        var_codes = []

        for key, val in var_dict.items():
            if type(val) != str:
                var_codes.append(f'{key} = {val}')
            else:
                var_codes.append(f'{key} = "{val}"')

        output_codes.extend(var_codes)
        output_codes.append('')

    #
    # 예약된 변수명 처리
    #
    if 'none' in binary_list and 'none' not in var_dict:
        output_codes.append(f'none = None')
    if 'add' in binary_list and 'add' not in var_dict:
        output_codes.append(f'add = "+"')
    if 'sub' in binary_list and 'sub' not in var_dict:
        output_codes.append(f'sub = "-"')
    if 'mul' in binary_list and 'mul' not in var_dict:
        output_codes.append(f'mul = "*"')
    if 'div' in binary_list and 'div' not in var_dict:
        output_codes.append(f'div = "/"')
    if 'mod' in binary_list and 'mod' not in var_dict:
        output_codes.append(f'mod = "%"')
    if 'asc' in binary_list and 'asc' not in var_dict:
        output_codes.append(f'asc = "asc"')
    if 'desc' in binary_list and 'desc' not in var_dict:
        output_codes.append(f'desc = "desc"')
    if 'first' in binary_list and 'first' not in var_dict:
        output_codes.append(f'first = "first"')
    if 'last' in binary_list and 'last' not in var_dict:
        output_codes.append(f'last = "last"')
    if 'middle' in binary_list and 'middle' not in var_dict:
        output_codes.append(f'middle = "middle"')
    if 'left' in binary_list and 'left' not in var_dict:
        output_codes.append(f'left = "left"')
    if 'right' in binary_list and 'right' not in var_dict:
        output_codes.append(f'right = "right"')
    if 'largest' in binary_list and 'largest' not in var_dict:
        output_codes.append(f'largest = "largest"')
    if 'smallest' in binary_list and 'smallest' not in var_dict:
        output_codes.append(f'smallest = "smallest"')
    if 'one' in binary_list and 'one' not in var_dict:
        output_codes.append(f'one = 1')
    if 'two' in binary_list and 'two' not in var_dict:
        output_codes.append(f'two = 2')
    if 'three' in binary_list and 'three' not in var_dict:
        output_codes.append(f'three = 3')

    output_codes.append('')
    exec_operator(binary_list, count_obj, operation_dict)

    if do_translate:
        output_codes.append('solution = open("solution.txt","w")')

    last_output = ''
    for output, (func_name, arg_value0, arg_value1) in operation_dict.items():
        code = inspect.getsource(getattr(functions, func_name))
        output_code, arg0, arg1 = convert_func_to_code(code)
        output_code = output_code.replace(arg0, arg_value0)
        output_code = output_code.replace(arg1, arg_value1)
        output_code = output_code.replace('result =', f'{output} =')

        output_codes.append(output_code)
        # print()
        # print(output, func_name, arg0, arg1)
        # print(arg_value0, arg_value1)
        # print(output_code)
        used_functions = [output,func_name,arg_value0,arg_value1]
        
        # line added for translation
        if do_translate:
            output_codes.append(translate(used_functions))

        last_output = output

    # 최종 출력 형식 조정
    code = inspect.getsource(getattr(functions, 'func_final'))
    output_code, arg0, arg1 = convert_func_to_code(code)
    output_code = output_code.replace(arg0, last_output)
    output_codes.append(output_code)

    output_codes[-1] = output_codes[-1].replace(f'result =', f'final_result =')

    output_codes.append('')
    output_codes.append(f'print(final_result)')
    
    if do_translate:
        output_codes.append('solution.close()')

    return output_codes


def generate_code(func_text, var_dict=None):
    '''
    input: function text
    output: output code with translation(to execute), output code without translation (to display)
    '''
    output_codes = []
    output_codes_for_display = []

    if var_dict:
        var_codes = []

        for key, val in var_dict.items():
            if type(val) != str:
                var_codes.append(f'{key} = {val}')
            else:
                var_codes.append(f'{key} = "{val}"')

        output_codes.extend(var_codes)
        output_codes_for_display.extend(var_codes)
        output_codes.append('')
        output_codes_for_display.append('')

    binary_list = func_text
    #binary_exp = convert_func_to_binary(func_text)
    #binary_list = binary_exp.split()
    #functions = [f for f in binary_list if f.startswith('func')]

    output_codes += generate_code_from_binary(binary_list,do_translate=True)
    #output_codes_for_display += generate_code_from_binary(binary_list,do_translate=False)

    return  output_codes
    #return output_codes_for_display, output_codes, functions

def test_run_per_question(data):
    print('\n## Original Question')
    print(data['Question'])
    print(data['QuestionConv'])

    print('\n## Equation')
    print(data['Equation'])

    print('\n## Generate Python Code & Run ##')

    #output_codes_for_display, output_codes = generate_code(data['Equation'], var_dict=data['Numbers'])

    output_codes_for_display, output_codes, functions = generate_code(data['Equation'], var_dict=data['Numbers'])
    #function_set = list(set(function_set) | set(functions))

    print('\n### Final Code ###')
    run_code = '\n'.join(output_codes)
    display_code = '\n'.join(output_codes_for_display)
    print(display_code)
    print(run_code)
    display_code_list = display_code.replace('\t','    ').split('\n')
    print(display_code_list)

    print('\n### Run Code ###')
    exec_vars = {}
    exec(run_code, None, exec_vars)

    print('\n### Korean Solution ###')
    with open('solution.txt','r') as f:
        solution = f.read()
        print(solution)
    solution_list = solution.split('\n')
    print(solution_list)

    print('\n### Execution Result ###')
    print(exec_vars['final_result'])

    if exec_vars['final_result'] == data['Answer']:
        return True
        #return True, function_set
    else:
        return (run_code, exec_vars)
        # return (run_code, exec_vars), function_set

def test_run(file_name = './data/1_ny_new.json'):
    data_list = load_json(file_name)
    error_list = []
    function_set = []

    for data in data_list:
        ret = test_run_per_question(data)
        # ret,function_set = test_run_per_question(data,function_set)
        # print(function_set)
        if ret != True:
            error_dict = {}
            error_dict['data'] = data
            error_dict['run_code'] = ret[0]
            error_dict['exec_vars'] = ret[1]
            error_list.append(error_dict)

    """
    if len(error_list) > 0:
        print(error_list)
        save_to_json(error_list,f'{file_name[:-5]}_error.json')
    else:
        print("No ERROR!@!!!!")
    """

    print('\n## Errors\n')
    for data in error_list:
        print()
        print()
        print(data['data']['Question'])
        print()
        print(data['data']['QuestionConv'])
        print()
        print(data['data']['Equation'])
        print()
        print(f'Real Answer:',data['data']['Answer'])
        print('Predicted Answer:',data['exec_vars']['final_result'])
        print(data['run_code'])

    """
    print('\n## Used Functions\n')
    for f in function_set:
        print(f)
    """

if __name__ == '__main__':
    # test_run('./data_easy/6_easy.json')
    test_run('./data/2.json')
