# -*- coding: utf-8 -*-

def func_final(input_value):
    output = None
    if type(input_value) == int or type(input_value) == float:
        input_int = int(input_value)
        if input_int != input_value:
            output = round(input_value, 2)
        else:
            output = input_int
    else:
        output = input_value
    result = output
    return result


def func_add(arg0, arg1):
    # Function : add
    result = arg0 + arg1
    return result


def func_minus(arg0, arg1):
    # Function : minus
    result = arg0 - arg1
    return result


def func_multiply(arg0, arg1):
    # Function : multiply
    result = arg0 * arg1
    return result


def func_divide(arg0, arg1):
    # Function : divide
    result = arg0 / arg1
    return result

def func_square(arg0, none):
    # Function : divide
    result = arg0**2
    return result

def func_findbasicseq(rep_seq_list, rep_num_list):
    seq_list = list()
    if rep_num_list is None:
        cut_num = len(rep_seq_list) // 2
        for i in range(2, cut_num + 1):
            seq_num = i
            seq_a = rep_seq_list[:seq_num]
            seq_b = rep_seq_list[seq_num:seq_num + seq_num]
            if seq_a == seq_b:
                seq_list = seq_a
                break
            seq_list = rep_seq_list
    else:
        for i in range(len(rep_num_list)):
            dis_rep_seq = rep_seq_list[i]
            dis_rep_num = rep_num_list[i]
            for j in range(dis_rep_num):
                seq_list.append(rep_seq_list[i])

    result = seq_list
    return result


def func_findord(basic_seq_list, order):
    result = basic_seq_list[order % len(basic_seq_list) - 1]
    return result


def func_drop(arg0, _):
    result = int(arg0)
    return result

def func_raise(arg0, _):
    from math import ceil
    result = ceil(arg0)
    return result

#
# 6
#


def func_inverseoperator(number_list, operator0):
    opers = {"+": (lambda x, y: x + y), "-": (lambda x, y: x - y), "*": (lambda x, y: x * y), "/": (lambda x, y: x / y)}
    fault = {'+': '-', '-': '+', '*': '/', '/': '*'}
    result = opers[fault[operator0]](number_list[1], number_list[0])
    return result


def func_rightoperator(number_list, operator0):
    opers = {"+": (lambda x,y: x+y), "-": (lambda x,y: x-y), "*": (lambda x,y: x*y),"/": (lambda x,y: x/y),"%": (lambda x,y: x%y)}
    answer = opers[operator0](number_list[0], number_list[1])
    result = answer
    return result


def func_list_divide(num_list, seq_list):
    answer = []
    for i in range(len(num_list)):
        x = num_list[i]/seq_list[i]
        answer.append(x)
    result = answer
    return result

def func_list_index(arg0,arg1):
    answer = arg0[arg1]
    result = answer
    return result

def func_whichsmall(arg0,arg1):
    if arg0 <= arg1:
        answer = arg0
    else:
        answer = arg1
    result = answer
    return result

def func_whichbig(arg0,arg1):
    if arg0 >= arg1:
        answer = arg0
    else:
        answer = arg1
    result = answer
    return result
#
# 1
#
def func_sum(arg0, _):
    # arg0 (datatype: list()): list of numbers to be summed
    # return: sum of given numbers
    answer = sum(arg0)
    result = answer
    return result


def func_odd(arg0, _):
    # arg0 (datatype: list()): list of numbers to find odd numbers
    # return: list of odd numbers from given list
    result = [i for i in arg0 if i % 2 == 1]
    return result

def func_mod(arg0, arg1):
    result = arg0%arg1
    return result

def func_quotient(arg0,arg1):
    result = arg0//arg1
    return result

def func_circlearea(arg0, arg1):
    # arg0 (datatype: int/float): radius of circle
    # arg1 (datatype: float): pi
    # return: area of circle
    from math import pi
    if arg1 == None:
        output = pi * arg0 * arg0
    else:
        output = arg1 * arg0 * arg0
    result = round(output, 2)
    return result



def func_m2cm(arg0, arg1):
    # arg0: m
    # arg1: cm (if given, else None)
    # return: _m _cm converted into _cm
    if arg1 == None:
        result = arg0 * 100
    else:
        result = arg0 * 100 + arg1
    return result


def func_cm2m(arg0, _):
    # arg0: cm
    # return: _m
    result = arg0 / 100
    return result

def func_kg2g(arg0, arg1):
    # arg0: kg
    # arg1: g (if given)
    # return: _kg _g to _g
    if arg1 == None:
        result = arg0 * 1000
    else:
        result = arg0 * 1000 + arg1
    return result
    
def func_hr2min(arg0,arg1):
    if arg1 == None:
        result = arg0 * 60
    else:
        result = arg0 * 60 + arg1
    return result

def func_min2hr(arg0,_):
    result = arg0/60
    return result

def func_even(arg0, _):
    # arg0: list of numbers
    result = [i for i in arg0 if i % 2 == 0]
    return result


def func_multiple(arg0, arg1):
    # arg0: list of numbers or digit
    # arg_1: we need to find multiple of arg_1
    # return: list of numbers that are multiple of arg_1
    output = []
    if type(arg0) == int:
        num_list = range(10**(arg0-1),10**arg0)
        for i in num_list:
            if i % arg1 == 0:
                output.append(i)
    else:
        for i in arg0:
            if i % arg1 == 0:
                output.append(i)
    result = output
    return result

def func_find_mul(elems, number):
    output = []
    if type(number)==list:
        for x in elems:
            for y in number:
                if x % y == 0:
                    output.append(x)
        output = set(output)
    else:
        for x in elems:
            if x % number == 0:
                output.append(x)

    result = len(output)
    return result


def func_between(arg0, arg1):
    # arg0: left num
    # arg1: right num
    # reutrn: center num
    if arg0 <= arg1:
        answer = range(arg0, arg1)[1]
    else:
        answer = range(arg1, arg0)[1]
    result = answer
    return result

def func_listindex(arg0,arg1):
    # arg0: list of numbers
    result = arg0[arg1 - 1]
    return result

def func_same(arg0,_):
    # returns the exact same number
    result = arg0
    return result
    
def func_exp(arg0,arg1):
    # returns arg0 ^ arg1
    result = arg0 ** arg1
    return result
    
def func_floor(arg0,_):
    # returns floor(arg0)
    import math
    result = math.floor(arg0)
    return result


#
# 3
#
def func_product(elems, number):
    import itertools
    result = len(list(itertools.product(elems, repeat=number)))
    return result


def func_permutation(elems, number):
    import itertools
    result = len(list(itertools.permutations(elems, number)))
    return result


def func_combination(elems, number):
    import itertools
    result = len(list(itertools.combinations(elems, number)))
    return result


def func_combinations_with_replacement(elems, number):
    import itertools
    result = len(list(itertools.combinations_with_replacement(elems, number)))
    return result

def func_combination_list(elems, number):
    import itertools
    result = [int(''.join(map(str, a))) for a in itertools.combinations(elems, number) if a[0] != 0]
    return result

def func_combination_sum(elems, number):
    answer = []
    import itertools
    comb_list = [int(''.join(map(str, a))) for a in itertools.combinations(elems, number) if 0 not in a]
    for i in comb_list:
        value = [int(j) for j in str(i)]
        sumvalue = sum(value)
        answer.append(sumvalue)
    result = answer
    return result
    
def func_product_sum(elems, number):
    answer = []
    import itertools
    comb_list = [''.join(map(str, a)) for a in itertools.product(elems, repeat=number) if 0 not in a]
    for i in comb_list:
        value = [int(j) for j in i]
        sumvalue = sum(value)
        answer.append(sumvalue)
    result = answer
    return result

def func_combination_diff(elems, number):
    answer = []
    import itertools
    for a in itertools.combinations(elems, number):
        if a[0] != 0:
            cum = a[-1] * 2 - sum(a)
            answer.append(abs(cum))
    result = answer
    return result

def func_combination_mul(elems, number):
    answer = []
    import itertools
    for a in itertools.combinations(elems, number):
        cum = 1
        if a[0] != 0:
            for elem in a:
                cum *= elem
        answer.append(cum)
    result = answer
    return result

def func_product_list(elems, number):
    import itertools
    output = []
    for a in itertools.product(elems, repeat=number):
        if a[0] != 0:
            output.append(int(''.join(map(str, a))))

    result = output
    return result


def func_permutation_list(elems, number):
    import itertools
    result = [int(''.join(map(str, a))) for a in itertools.permutations(elems, number) if a[0] != 0]
    return result


def func_len(elems, none):
    result = len(elems)
    return result

def func_find_bigorder(elems, number):
    result = sorted(elems, reverse=True)[number - 1]
    return result

def func_find_smallorder(elems, number):
    result = sorted(elems)[number- 1]
    return result

def func_find_odd(elems, none):
    result = len([x for x in elems if x % 2 == 1 and x > 0])
    return result


def func_find_even(elems, none):
    result = len([x for x in elems if x % 2 == 0 and x > 0])
    return result


def func_find_range(elems, number):
    output = []
    for x in elems:
        if x <= number[1] and x >= number[0]:
            output.append(x)

    result = len(output)
    return result

def func_range_list(number, none):
    result = [i for i in range(number)]
    return result

def func_expandlist(arg0,arg1):
    arg0.append(arg1)
    result = arg0
    return result

def func_makelist(arg0, arg1):
    result = [arg0, arg1]
    return result

def func_rangelist(arg0,arg1):
    result = [i for i in range(arg0,arg1+1)]
    return result

def func_numfreq(num_list,dir0):
    count = 0
    if type(dir0) == list:
        for i in num_list:
            for arg in dir0:
                count += str(i).count(str(arg))
    else:
        for i in num_list:
            count += str(i).count(str(dir0))
    result = count
    return result

def func_makedivisor(arg0,_):
    answer = []
    for i in range(1, arg0+1):
        if i==arg0 or arg0%i==0:
            answer.append(i)
    result = answer
    return result
#
# 4
#

def func_smaller(arg0,arg1):
    if arg0 < arg1:
        result = arg0
    else:
        result = arg1
    return result

def func_bigger(arg0,arg1):
    if arg0 > arg1:
        result = arg0
    else:
        result = arg1
    return result

def func_middle_idx(elems, _):
    result = int((len(elems)+1)/2)
    return result

def func_remove_smallest(elems, _):
    smallest = sorted(elems)[0]
    answer = elems.copy()
    answer.remove(smallest)
    result = answer
    return result

def func_remove_largest(elems, _):
    largest = sorted(elems)[-1]
    answer = elems.copy()
    answer.remove(largest)
    result = answer
    return result

def func_MakeSeq(elems, none):
    answer = list()
    seq_1 = range(10**elems)
    for i in range(len(seq_1)):
        if seq_1[i] >= 10**(elems-1):
            answer.append(seq_1[i])
    result = answer
    return result


def func_FindDivisor(elems, numbers):
    answer = list()
    for elem in elems:
        flag = True
        for elem2 in numbers:
            if elem%elem2!=0:
                flag = False
        if flag:
            answer.append(elem)
    result = answer
    return result



def func_movepoint(dir0, dig0):
    answer = list()
    if dir0 == 'right':
        answer.append(dir0)
        answer.append(10**dig0)
    elif dir0 == 'left':
        answer.append(dir0)
        answer.append(1/(10**dig0))
    else:
        answer.append(dir0)
        answer.append(0)
    result = answer
    return result


def func_OriginNumber(dir_list, num0):
    answer = 0
    if dir_list[0] == 'right':
        answer = round(num0 / (dir_list[1] - 1), 2)
    elif dir_list[0] == 'left':
        answer = round(num0 / (1 - dir_list[1]), 2)
    else:
        answer = 0

    result = answer
    return result

def func_SplitNumber(elems, none):
    answer = [int(x) for x in str(elems)]
    result = answer
    return result

def func_FinddigitNumber(number_0, number_1):
    digit_index = len(number_1)
    answer = number_0[-1*digit_index]
    result = answer
    return result


def func_diff_Numbers(dir_list, num0):
    answer = 0
    if dir_list[0] == 'right':
        answer = round((num0*dir_list[1])-num0,2)
    elif dir_list[0] == 'left':
        answer = round(num0-(num0*dir_list[1]),2)
    else:
        answer = 0
    result = answer
    return result

def func_largenum(num_list,arg0):
    answer = []
    if arg0==None:
        for i in num_list:
            if(isinstance(i,list)):
                maxi = max(i)
                answer.append(maxi)
            else:
                break
        answer = max(num_list)
    else:
        for i in num_list:
            if i>arg0:
                answer.append(i)
    result = answer
    return result

def func_smallnum(num_list, arg0):
    answer = []
    if arg0==None:
        answer = min(num_list)
    else:
        for i in num_list:
            if i < arg0:
                answer.append(i)
    result = answer
    return result

def func_seq_mul(arg0, arg1):
    flag = False
    for i in range(arg0):
        mul_res = i
        for j in range(1, arg1):
            mul_res *= (i+j)
        if mul_res == arg0:
            answer = []
            for j in range(arg1):
                answer.append(i+j)
            flag = True
            break
        if flag == True:
            break
    result = answer
    return result

def func_seq_add(added, cnt):
    flag = False
    for i in range(added):
        add_res = i
        for j in range(1, cnt):
            add_res += (i+j)
        if add_res == added:
            answer = []
            for j in range(cnt):
                answer.append(i+j)
            flag = True
            break
        if flag == True:
            break
    result = answer
    return result

def func_dual_eq(arg0,arg1):
    # x + y = arg1[0]
    # arg0[0]*x + arg0[1]*y = arg1[1]
    x = (arg1[1] - arg0[1]*arg1[0])/(arg0[0]-arg0[1])
    y = arg1[0] - x
    result = [int(x), int(y)]
    return result

#
#5
#

def func_permutation_list2list(elems, number):
    ans = []
    import itertools
    for i in elems:
        answer = [int(''.join(map(str, a))) for a in itertools.permutations(i, number) if a[0] != 0]
        ans.append(answer)
    result = ans
    return result

def func_make_unknown(num_list, none):
    x = ''
    answer = []
    for a in num_list:
        x += str(a)
    for number in range(10**(len(x)-1),10**len(x)):
        flag=[]
        for idx,digit in enumerate(x):
            if ord('0')<=ord(digit)<=ord('9'):
                if x[idx]==str(number)[idx]:
                    flag.append(True)
                else:
                    flag.append(False)

        if all(flag):
            answer.append(number)   
    result = answer
    return result

def func_make_unknown_list(num_list, none):
    x = ''
    answer2 = []
    length = len(num_list)
    for a in num_list:
        x += str(a)
    for number in range(10**(len(x)-1),10**len(x)):
        flag = []
        answer1 = []
        for idx,digit in enumerate(x):
            if ord('0')<=ord(digit)<=ord('9'):
                if x[idx]==str(number)[idx]:
                    flag.append(True)
                else:
                    flag.append(False)
                    
        if all(flag):
            number_list=str(number)
            for i in range(0,length):
                answer1.append(number_list[i])
            answer2.append(answer1)
    result = answer2
    return result

def func_list_samenum(num_list1, num_list2):
    answer = []
    for i in num_list1:
        for j in i:
            for k in num_list2:
                if j==k:
                    answer.append(k)
    result = answer
    return result


def func_roundlist(num_list, dir0):
    answer = []
    dir0 = dir0*10
    for i in num_list:
        i = i/dir0
        i = round(i)
        i = i*dir0
        answer.append(i)
    result = answer
    return result

def func_droplist(num_list, dir0):
    from math import floor
    answer = []
    dir1 = 10 * dir0
    for i in num_list:
        i = i/dir1
        i = floor(i)
        i = i * dir1
        answer.append(i)
    result = answer
    return result

def func_raiselist(num_list, dir0):
    from math import ceil
    answer = []
    dir0 = dir0*10
    for i in num_list:
        i=i/dir0
        i=ceil(i)
        i=i*dir0
        answer.append(i)
    result = answer
    return result

def func_listsearch(num_list, dir0):
    count = 0
    if type(dir0) == list:
        for i in num_list:
            if i in dir0:
                count += 1
    else:
        for i in num_list:
            if i == dir0:
                count += 1
    result = count
    return result

def func_listcheckint(num_list, none=None):
    count = 0
    for i in num_list:
        if (i).is_integer():
            count = count+1
    result = count
    return result

def func_listadd(num_list,arg0):
    answer = []
    for i in range(len(num_list)):
        x = num_list[i]+arg0
        answer.append(x)
    result = answer
    return result

def func_listminus(num_list,arg0):
    answer = []
    for i in range(len(num_list)):
        x = num_list[i]-arg0
        answer.append(x)
    result = answer
    return result

def func_listdivide(num_list,arg0):
    answer = []
    for i in range(len(num_list)):
        x = num_list[i]/arg0
        answer.append(x)
    result = answer
    return result

def func_listmultiply(num_list,arg0):
    answer = []
    for i in range(len(num_list)):
        x = num_list[i]*arg0
        answer.append(x)
    result = answer
    return result

def func_gcd(arg0,arg1):
    from math import gcd
    answer = gcd(arg0,arg1)
    result = answer
    return result

def func_divisorset(arg0, _):
    arg1 = []
    for i in range(1, arg0 + 1):
        answer = []
        if i * i > arg0:
            break
        if arg0 % i:
            continue 
        answer.append(i)
        answer.append(arg0//i)
        arg1.append(answer)
    result = arg1
    return result

def func_divisorlist(arg0, _):
    answer = []
    for i in range(1, arg0+1):
        if arg0%i==0:
            answer.append(i)
    result = answer
    return result

def func_listdifference(num_list, arg0):
    answer = []
    for i in num_list:
        minvalue = min(i)
        maxvalue = max(i)
        arg1 = maxvalue-minvalue
        if arg1 == arg0:
            answer.append(i)
    result = answer
    return result

def func_remove_num(num_list, arg0):
    answer = []
    for i in num_list:
        if i!=arg0:
            answer.append(i)
    result = answer
    return result

def func_merge(list1,list2):
    answer = list1 + list2
    result = answer
    return result

def func_add_digit2(equa_list,elem): 
    for arg1 in range(10):
        for arg2 in range(10):
            if isinstance(equa_list[0],int):
                if 10*equa_list[0]+arg1+10*arg2+equa_list[-1]==elem:
                    answer = {equa_list[1]:arg1,equa_list[-2]:arg2}
            else:
                if 10*arg1+equa_list[1]+10*equa_list[-2]+arg2==elem:
                    answer = {equa_list[0]:arg1,equa_list[-1]:arg2}
    result = answer
    return result

def func_sub_digit2(equa_list,elem): 
    for arg1 in range(10):
        for arg2 in range(10):
            if isinstance(equa_list[0],int):
                if 10*equa_list[0]+arg1-10*arg2-equa_list[-1]==elem:
                    answer = {equa_list[1]:arg1,equa_list[-2]:arg2}
            else:
                if 10*arg1+equa_list[1]-10*equa_list[-2]-arg2==elem:
                    answer = {equa_list[0]:arg1,equa_list[-1]:arg2}
    result = answer
    return result

def func_mul_digit2(equa_list,elem):
    for arg1 in range(10):
        for arg2 in range(10):
            if isinstance(equa_list[0],int):
                if (10*equa_list[0]+arg1)*(10*arg2+equa_list[-1])==elem:
                    answer = {equa_list[1]:arg1,equa_list[-2]:arg2}
            else:
                if (10*arg1+equa_list[1])*(10*equa_list[-2]+arg2)==elem:
                    answer = {equa_list[0]:arg1,equa_list[-1]:arg2}
    result = answer
    return result

def func_add_digit3(equa_list,final_list): 
    for arg1 in range(10):
        for arg2 in range(10):
            for arg3 in range(10):
                if isinstance(equa_list[0],int):
                    if equa_list[0]>=10:
                        if isinstance(equa_list[2],int):
                            if 10*equa_list[0]+arg1+100*equa_list[2]+10*arg2+equa_list[-1]==100*arg3+final_list[-1]:
                                answer={equa_list[1]:arg1,equa_list[-2]:arg2,final_list[0]:arg3}
                        else:
                            if 10*equa_list[0]+arg1+100*arg2+equa_list[-1]==100*final_list[0]+10*arg3+final_list[-1]:
                                answer={equa_list[1]:arg1,equa_list[-2]:arg2,final_list[1]:arg3}
                    else:
                        if isinstance(equa_list[-1],int):
                            if 100*equa_list[0]+10*arg1+equa_list[2]+100*arg2+equa_list[-1]==10*final_list[0]+arg3:
                                answer={equa_list[1]:arg1,equa_list[-2]:arg2,final_list[-1]:arg3}
                        else:
                            if 100*equa_list[0]+10*arg1+equa_list[2]+10*equa_list[-2]+arg2==100*arg3+final_list[-1]:
                                answer={equa_list[1]:arg1,equa_list[-1]:arg2,final_list[0]:arg3}
                else:
                    if isinstance(equa_list[-1],int):
                        if 100*arg1+equa_list[1]+100*equa_list[2]+10*arg2+equa_list[-1]==10*final_list[0]+arg3:
                                answer={equa_list[0]:arg1,equa_list[-2]:arg2,final_list[-1]:arg3}
                    else:
                        if 100*arg1+equa_list[1]+10*equa_list[2]+arg2==100*final_list[0]+10*arg3+final_list[-1]:
                                answer={equa_list[0]:arg1,equa_list[-1]:arg2,final_list[1]:arg3}
    result = answer
    return result

def func_sub_digit3(equa_list,final_list):  
    for arg1 in range(10):
        for arg2 in range(10):
            for arg3 in range(10):
                if isinstance(equa_list[0],int):
                    if equa_list[0]>=10:
                        if isinstance(equa_list[2],int):
                            if 10*equa_list[0]+arg1-(100*equa_list[2]+10*arg2+equa_list[-1])==100*arg3+final_list[-1]:
                                answer={equa_list[1]:arg1,equa_list[-2]:arg2,final_list[0]:arg3}
                        else:
                            if 10*equa_list[0]+arg1-(100*arg2+equa_list[-1])==100*final_list[0]+10*arg3+final_list[-1]:
                                answer={equa_list[1]:arg1,equa_list[-2]:arg2,final_list[1]:arg3}
                    else:
                        if isinstance(equa_list[-1],int):
                            if 100*equa_list[0]+10*arg1+equa_list[2]-(100*arg2+equa_list[-1])==10*final_list[0]+arg3:
                                answer={equa_list[1]:arg1,equa_list[-2]:arg2,final_list[-1]:arg3}
                        else:
                            if 100*equa_list[0]+10*arg1+equa_list[2]-(10*equa_list[-2]+arg2)==100*arg3+final_list[-1]:
                                answer={equa_list[1]:arg1,equa_list[-1]:arg2,final_list[0]:arg3}
                else:
                    if isinstance(equa_list[-1],int):
                        if 100*arg1+equa_list[1]-(100*equa_list[2]+10*arg2+equa_list[-1])==10*final_list[0]+arg3:
                                answer={equa_list[0]:arg1,equa_list[-2]:arg2,final_list[-1]:arg3}
                    else:
                        if 100*arg1+equa_list[1]-(10*equa_list[2]+arg2)==100*final_list[0]+10*arg3+final_list[-1]:
                                answer={equa_list[0]:arg1,equa_list[-1]:arg2,final_list[1]:arg3}
    result = answer
    return result

def func_find_answ(equa_list,key):
    result = equa_list[key]
    return result

def func_findlargeindex(rep_seq_list, arg0):
    for i in range(len(rep_seq_list)):
        i = len(rep_seq_list)-i-1
        if rep_seq_list[i]==arg0:
            answer = i
            break
    result = answer
    return result

def func_continuenum3(arg0, _):
    answer = []
    for i in range(1,8):
        x = i*100+(i+1)*10+(i+2)
        answer.append(x)
    result = answer
    return result

def func_listint(num_list, none=None):
    for i in range(len(num_list)):
        if (num_list[i]).is_integer():
            answer = i
    result = answer
    return result

def func_gcdlcm2num(arg0,arg1):
    answer = []
    from math import gcd
    for i in range(1,arg1+1):
        for j in range(1,arg1+1):
            if i>=j:
                if gcd(i,j)==arg0:
                    if i*j//gcd(i,j)==arg1:
                        answer.append(i)
                        answer.append(j)
    result = answer
    return result

def func_listsamenumindex(arg0,arg1):
    if len(arg0)==len(arg1):
        for i in range(len(arg0)):
            if arg0[i]==arg1[i]:
                answer = i
    result = answer
    return result

#
# 7
#

def func_difference(arg0,arg1):
    if arg0 >= arg1:
        result = arg0 - arg1
    else:
        result = arg1 - arg0
    return result

def func_findindex(rep_seq_list, text):
    if text==None:
        answer = len(rep_seq_list) + 1
    else:
        answer = int(rep_seq_list.index(text))
    result = answer
    return result


def func_operator(opr, number):
    if (opr == '-'):
        if (number[0] > number[1]):
            answer = number[0] - number[1]
        else:
            answer = number[1] - number[0]
    elif (opr == '+'):
        answer = number[0] + number[1]
    elif (opr == '*'):
        answer = number[0] * number[1]
    elif (opr == '/'):
        if (number[0] > number[1]):
            answer = number[0] / number[1]
        else:
            answer = number[1] / number[0]

    result = answer
    return result

def func_putseq(num_list, arg0):
    answer = []
    for i in num_list:
        answer.append(i)
    answer.append(arg0)
    result = answer
    return result

def func_findlargestidx(num_list,none):
    largest = max(num_list)
    answer = num_list.index(largest)
    result = answer
    return result

def func_printseqidx(num_list,dir0):
    result = num_list[dir0]
    return result

#
#8
#

def func_triangle_angle_minus(arg0, _):
    answer = 180-arg0
    result = answer
    return result

def func_righttriangle_angle_minus(arg0, _):
    answer = 90-arg0
    result = answer
    return result

def func_lcm(arg0,arg1):
    from math import gcd
    if type(arg0) == list:
        for i in range(len(arg0)-1):
            arg0[i+1] = arg0[i] * arg0[i+1] // gcd(arg0[i],arg0[i+1])
        result = arg0[-1]
    else:
      result = arg0*arg1//gcd(arg0,arg1)
    return result

def func_factorial(arg0, _):
    import math
    result = math.factorial(arg0)
    return result

def func_roundareatoedge(arg0,arg1):
    for i in range(1, arg1 + 1):
        if i * i > arg1:
            break
        if arg1 % i:
            continue 
        if i+(arg1//i)==arg0//2:
            answer = arg1//i
    result = answer
    return result

def func_sqrt(arg0, _):
    import math
    answer = math.sqrt(arg0)
    result = answer
    return result

def func_cuberoot(arg0, _):
    answer = arg0**(1/3)
    result = answer
    return result

def func_cube(arg0, _):
    answer = arg0**(3)
    result = answer
    return result

def func_km2m(arg0, arg1):
    # arg0: km
    # arg1: m (if given, else None)
    # return: _m _cm converted into _cm
    if arg1 == None:
        result = arg0 * 1000
    else:
        result = arg0 * 1000 + arg1
    return result

def func_edge(arg0,arg1):
    if arg1==None:
        if arg0==4:
            answer=6
        elif arg0==6:
            answer=12
        elif arg0==8:
            answer=12
        elif arg0==12:
            answer=30
        elif arg0==20:
            answer=30
    else:
        if arg0==4:
            answer=3*arg1
        elif arg0==6:
            answer=4*arg1
        elif arg0==8:
            answer=3*arg1
        elif arg0==12:
            answer=5*arg1
        elif arg0==20:
            answer=3*arg1
    result = answer
    return result

def func_vertex(arg0,arg1):
    if arg1==None:
        if arg0==4:
            answer=4
        elif arg0==6:
            answer=8
        elif arg0==8:
            answer=6
        elif arg0==12:
            answer=20
        elif arg0==20:
            answer=12
    else:
        if arg0==4:
            answer=3*arg1
        elif arg0==6:
            answer=4*arg1
        elif arg0==8:
            answer=3*arg1
        elif arg0==12:
            answer=5*arg1
        elif arg0==20:
            answer=3*arg1
    result = answer
    return result

def func_plane(arg0,_):
    answer = arg0
    result = answer
    return result

def func_cubearea(elems,number):
    answer = (elems[0]*elems[1]+number*elems[1]+number*elems[0])*2
    result = answer
    return result

def func_findlargeordidx(num_list,arg0):
    listset = set(num_list)
    sorted_list = sorted(listset, reverse=True)
    x = sorted_list[arg0 - 1]
    for i in range(len(num_list)):
        if num_list[i] == x:
            answer = i
    result = answer
    return result


def func_findsmallestidx(num_list, none):
    smallest = min(num_list)
    answer = num_list.index(smallest)
    result = answer
    return result
    
# NOT USED
def func_listindexright(arg0,arg1):
    # arg0: list of numbers
    result = arg0[-1 * arg1]
    return result


def func_range(arg0, arg1):
    # arg0 (datatype: int): start of list
    # arg1 (datatype: int): end of list (to be included)
    # return: list of numbers from start to end
    result = range(arg0, arg1 + 1)
    return result

