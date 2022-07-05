def translate(used_function):
    result = used_function[0]
    function_name = used_function[1]
    arg0 = used_function[2]
    arg1 = used_function[3]

    #
    # 4
    #

    if function_name == 'func_find_bigorder':
        if arg1 != '0':
            text = "f'{%s}을 내림차순으로 나열하면{sorted(%s,reverse=True)}와 같다. 해당 수열에서의 {int(%s)}번째 수는 {%s}이다.'" % (
            arg0, arg0, arg1, result)
        else:
            text = "f'{%s}을 내림차순으로 나열하면{sorted(%s,reverse=True)}와 같다. 해당 수열에서의 마지막에서 {int(%s)+1}번째 수는 {%s}이다.'" % (
            arg0, arg0, arg1, result)

        return f'solution.write("\\n"+{text})'

    elif function_name == 'func_find_smallorder':
        if arg1 != '0':
            text = "f'{%s}을 오름차순으로 나열하면{sorted(%s)}와 같다. 해당 수열에서의 {int(%s)}번째 수는 {%s}이다.'" % (arg0, arg0, arg1, result)
        else:
            text = "f'{%s}을 오름차순으로 나열하면{sorted(%s)}와 같다. 해당 수열에서의 마지막에서 {int(%s)+1}번째 수는 {%s}이다.'" % (
            arg0, arg0, arg1, result)

        return f'solution.write("\\n"+{text})'

    elif function_name == 'func_middle_idx':
        text = "f'주어진 수열의 중간에 있는 수는 {%s}번째 수이다.'" % (result)

        return f'solution.write("\\n"+{text})'

    elif function_name == 'func_movepoint':
        text = "f'어떤 소수를 {%s}으로 {%s}자리 옮기면 {%s[1]}배가 된다.'" % (arg0, arg1, result)

        return f'solution.write("\\n"+{text}.replace("left","왼쪽").replace("right","오른쪽"))'

    elif function_name == 'func_OriginNumber':
        text = "f'원래 소수에서 소수점을 {%s[0]}으로 옮겨 {%s[1]}배가 되었을 때의 변화량이 {%s}이라면, 원래 소수는'" % (arg0, arg0, arg1)
        code = """
if %s[0] == "left":
    equation = f"{%s} / (1 - {%s[1]}) = {%s}"
else:
    equation = f"{%s} / ({%s[1]} - 1) = {%s}"
""" % (arg0, arg1, arg0, result, arg1, arg0, result)
        return code + f'solution.write("\\n"+{text}.replace("left","왼쪽").replace("right","오른쪽"))\nsolution.write("\\n"+equation+"이다.")'

    elif function_name == 'func_diff_Numbers':
        text = "f'원래 소수 {%s}에서 소수점을 {%s[0]}으로 옮겨 {%s[1]}배가 되었을 때, 원래 소수와의 차는 |{%s} - ({%s}*{%s[1]})| = {%s}이다.'" % (
        arg1, arg0, arg0, arg1, arg1, arg0, result)
        return f'solution.write("\\n"+{text}.replace("left","왼쪽").replace("right","오른쪽"))'

    elif function_name == 'func_difference':
        text = "f'{%s}와 {%s}의 차는 |{%s} - {%s}| = {%s}이다.'" % (arg0, arg1, arg0, arg1, result)
        return f'solution.write("\\n"+{text})'

    elif function_name == 'func_permutation_list':
        text = "f'{%s} 중 {%s}개를 뽑아 조합하여 만들 수 있는 {%s}자리 수는 {%s} 이다.'" % (arg0, arg1, arg1, result)
        return f'solution.write("\\n"+{text})'

    elif function_name == 'func_remove_smallest':
        text = "f'{%s} 에서 가장 작은 수를 빼면, {%s} 의 수들만 남게 된다.'" % (arg0, result)
        return f'solution.write("\\n"+{text})'

    elif function_name == 'func_remove_largest':
        text = "f'{%s} 에서 가장 큰 수를 빼면, {%s} 의 수들만 남게 된다.'" % (arg0, result)
        return f'solution.write("\\n"+{text})'

    elif function_name == 'func_FindDivisor':
        text = "f'그 수들 중 {%s} 의 수로 나누어 떨어질 수 있는 수는 {%s} 와 같다.'" % (arg1, result)
        return f'solution.write("\\n"+{text})'

    elif function_name == 'func_MakeSeq':
        text = "f'만들 수 있는 {%s}자리 수는 [{%s[0]}, {%s[1]},..., {%s[-1]}] 이다.'" % (arg0, result, result, result)
        return f'solution.write("\\n"+{text})'

    elif function_name == 'func_len':
        text = "f'{%s} 의 원소의 개수는 총 {%s}개 이다.'" % (arg0, result)
        return f'solution.write("\\n"+{text})'

    elif function_name == 'func_lcm':
        code = """
if type(%s) == list:
    text = f"{%s} 의 최소공배수는 {%s}이다."
else:
    text = f"{%s} & {%s}의 최소공배수는 {%s}이다."
solution.write("\\n"+text)
""" % (arg0, arg0, result, arg0, arg1, result)
        return code

    elif function_name == 'func_gcd':
        text = "f'{%s}와 {%s}의 최대공약수는 {%s}이다.'" % (arg0, arg1, result)
        return f'solution.write("\\n"+{text})'

    elif function_name == 'func_add':
        text = "f'{%s} + {%s} = {%s}'" % (arg0, arg1, result)
        return f'solution.write("\\n"+{text})'

    elif function_name == 'func_multiply':
        text = "f'{%s} * {%s} = {%s}'" % (arg0, arg1, result)
        return f'solution.write("\\n"+{text})'

    elif function_name == 'func_minus':
        text = "f'{%s} - {%s} = {%s}'" % (arg0, arg1, result)
        return f'solution.write("\\n"+{text})'

    elif function_name == 'func_divide':
        text = "f'{%s} / {%s} = {%s}'" % (arg0, arg1, result)
        return f'solution.write("\\n"+{text})'

    elif function_name == 'func_mod':
        text = "f'{%s}를 {%s}로 나누었을 때의 나머지는 {%s}이다.'" % (arg0, arg1, result)
        return f'solution.write("\\n"+{text})'

    #
    # 1
    #

    elif function_name == 'func_kg2g':
        code = """
if %s == None:
    text = f"{%s}kg은 총 {%s}g이다."
else:
    text = f"{%s}kg {%s}g은 총 {%s}g이다."
solution.write("\\n"+text)
""" % (arg1, arg0, result, arg0, arg1, result)
        return code

    elif function_name == 'func_rangelist':
        text = "f'{%s}부터 {%s}까지 수만을 고려한다고 하자.'" % (arg0, arg1)
        return f'solution.write("\\n"+{text})'

    elif function_name == 'func_min2hr':
        text = "f'{%s}분은 {%s}시간 이다.'" % (arg0, result)
        return f'solution.write("\\n"+{text})'

    elif function_name == 'func_hr2min':
        code = """
if %s == None:
    text = f"{%s}시간은 총 {%s}분이다."
else:
    text = f"{%s}시간 {%s}분은 총 {%s}분이다."
solution.write("\\n"+text)
""" % (arg1, arg0, result, arg0, arg1, result)
        return code

    elif function_name == 'func_m2cm':
        code = """
if %s == None:
    text = f"{%s}m는 총 {%s}cm이다."
else:
    text = f"{%s}m {%s}cm는 총 {%s}cm이다."
solution.write("\\n"+text)
""" % (arg1, arg0, result, arg0, arg1, result)
        return code

    elif function_name == 'func_circlearea':
        text = "f'반지름이 {%s}인 원의 넓이는 {%s} * {%s} * π = {%s}'" % (arg0, arg0, arg0, result)
        return f'solution.write("\\n"+{text})'

    #
    # 3
    #

    elif function_name == 'func_combination':
        code = "combinations = list(itertools.combinations(%s,%s))" % (arg0, arg1)
        text = "f'{%s} 중 {%s}개를 조합하여 만들 수 있는 조합은 {combinations}로 {len(%s)}C{%s} = {%s}개 이다.'" % (
        arg0, arg1, arg0, arg1, result)
        return f'{code}\nsolution.write("\\n"+{text})'

    elif function_name == 'func_permutation':
        code = "permutations = list(itertools.permutations(%s,%s))" % (arg0, arg1)
        text = "f'{%s} 중 {%s}개를 순서를 고려하여 중복없이 뽑아서 만들 수 있는 경우는 {permutations}로 {len(%s)}P{%s} = {%s}개 이다.'" % (
        arg0, arg1, arg0, arg1, result)
        return f'{code}\nsolution.write("\\n"+{text})'

    elif function_name == 'func_product':
        code = "product = list(itertools.product(%s,repeat=%s))" % (arg0, arg1)
        text = "f'{%s} 중 중복을 허용하여 {%s}개를 순서를 고려하여 뽑아서 만들 수 있는 경우는 {product}로 {len(%s)}Π{%s} = {%s}개 이다.'" % (
        arg0, arg1, arg0, arg1, result)
        return f'{code}\nsolution.write("\\n"+{text})'

    elif function_name == 'func_combinations_with_replacement':
        code = "combinations = list(itertools.combinations_with_replacement(%s,%s))" % (arg0, arg1)
        text = "f'{%s} 중 중복을 허용하여 {%s}개를 조합하여 만들 수 있는 조합은 {combinations}로 {len(%s)}H{%s} = {%s}개 이다.'" % (
        arg0, arg1, arg0, arg1, result)
        return f'{code}\nsolution.write("\\n"+{text})'

    elif function_name == 'func_range_list':
        text = "f'{%s}보다 작은 수만을 고려한다고 하자.'" % (arg0)
        return f'solution.write("\\n"+{text})'

    elif function_name == 'func_find_range':
        text = "f'{%s} 중 {%s[0]}보다 크고 {%s[1]}보다 작은 수는 총 {%s}개이다.'" % (arg0, arg1, arg1, result)
        return f'solution.write("\\n"+{text})'

    elif function_name == 'func_find_even':
        text = "f'{%s} 중 짝수의 개수는 {%s}개이다.'" % (arg0, result)
        return f'solution.write("\\n"+{text})'

    elif function_name == 'func_find_odd':
        text = "f'{%s} 중 홀수의 개수는 {%s}개이다.'" % (arg0, result)
        return f'solution.write("\\n"+{text})'

    elif function_name == 'func_product_list':
        text = "f'{%s} 중 중복을 허용하여 만들 수 있는 {%s}자리 숫자는 {%s}이다.'" % (arg0, arg1, result)
        return f'solution.write("\\n"+{text})'

    elif function_name == 'func_find_mul':
        text = "f'{%s} 중 {%s}의 배수는 {output}로 총 {%s}개이다.'" % (arg0, arg1, result)
        return f'solution.write("\\n"+{text})'

    elif function_name == 'func_multiple':
        code = """
if type(%s) == int:
    text = f"{%s}자리 수 중 {%s}의 배수는 {%s}이다."
else:
    text = f"{%s} 중 {%s}의 배수는 {%s}이다."
solution.write("\\n"+text)
""" % (arg0,arg0,arg1,result,arg0,arg1,result)
        return code

    elif function_name == 'func_smallnum':
        text = "f'{%s} 중 {%s}보다 작은 수는 {%s}이다.'" % (arg0, arg1, result)
        return f'solution.write("\\n"+{text})'

    elif function_name == 'func_largenum':
        text = "f'{%s} 중 {%s}보다 큰 수는{%s}이다.'" % (arg0, arg1, result)
        return f'solution.write("\\n"+{text})'

    elif function_name == 'func_combination_sum':
        text = "f'{%s} 중 서로 다른 {%s}개를 뽑아 만들 수 있는 조합은 {list(itertools.combinations(%s,%s))}이다.\\n각 조합을 합한 값들은 각각 {%s}이다.'" % (
        arg0, arg1, arg0, arg1, result)
        return f'solution.write("\\n"+{text})'

    elif function_name == 'func_product_sum':
        text = "f'{%s} 중 중복을 허용하여 순서를 고려하여 {%s}개를 뽑아 만들 수 있는 순열은 {list(itertools.product(%s,repeat=%s))}이다.\\n각 조합을 합한 값들은 각각 {%s}이다.'" % (
        arg0, arg1, arg0, arg1, result)
        return f'solution.write("\\n"+{text})'

    elif function_name == 'func_combination_diff':
        text = "f'{%s} 중 서로 다른 {%s}개를 뽑아 만들 수 있는 조합은 {list(itertools.combinations(%s,%s))}이다.\\n각 조합의 차는 각각 {%s}이다.'" % (
        arg0, arg1, arg0, arg1, result)
        return f'solution.write("\\n"+{text})'

    elif function_name == 'func_combination_mul':
        text = "f'{%s} 중 서로 다른 {%s}개를 뽑아 만들 수 있는 조합은 {list(itertools.combinations(%s,%s))}이다.\\n각 조합을 곱한 값들은 각각 {%s}이다.'" % (
        arg0, arg1, arg0, arg1, result)
        return f'solution.write("\\n"+{text})'

    elif function_name == 'func_numfreq':
        text = "f'{%s} 중 {%s}는 총 {%s}번 사용된다.'" % (arg0, arg1, result)
        return f'solution.write("\\n"+{text})'

    elif function_name == 'func_makedivisor':
        text = "f'{%s}의 약수는 {%s}이다.'" % (arg0, result)
        return f'solution.write("\\n"+{text})'

    elif function_name == 'func_listsearch':
        text = "f'{%s} 중 {%s}은/는 총 {%s}번 나타난다.'" % (arg0, arg1, result)
        return f'solution.write("\\n"+{text})'

    #
    # 6
    #

    elif function_name == 'func_whichsmall':
        text = "f'{%s}와 {%s} 중 더 작은 수는 {%s}이다.'" % (arg0,arg1,result)
        return f'solution.write("\\n"+{text})'

    elif function_name == 'func_whichbig':
        text = "f'{%s}와 {%s} 중 더 큰 수는 {%s}이다.'" % (arg0,arg1,result)
        return f'solution.write("\\n"+{text})'

    elif function_name == 'func_square':
        text = "f'({%s})² = {%s}'" % (arg0,result)
        return f'solution.write("\\n"+{text})'

    elif function_name == 'func_raise':
        text = "f'{%s}를 올림하면 {%s}이다.'" % (arg0,result)
        return f'solution.write("\\n"+{text})'

    elif function_name == 'func_inverseoperator':
        text = "f'x {%s} {%s[0]} = {%s[1]}\\nx = {%s[1]} {fault[%s]} {%s[0]}\\n∴ x = {%s}'" % (arg1,arg0,arg0,arg0,arg1,arg0,result)
        return f'solution.write("\\n"+{text})'

    else:
        return ""
