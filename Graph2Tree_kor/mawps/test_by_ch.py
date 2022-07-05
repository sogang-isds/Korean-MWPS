from src.train_and_evaluate import *
from src.models import *
import time
import torch.optim
from src.expressions_transfer import *
import json
import warnings
import random
import pickle
from timeout_timer import timeout, TimeoutInterrupt
import time
import logging
from ai_challenge_2step_data.data_utils import generate_code_from_binary,generate_code
import pandas as pd

path = '../../ai_challenge_2step_data/data/'
model_path = 'model_traintest/'
logger = logging.getLogger(__name__)
warnings.filterwarnings(action='ignore')
def read_json(path):
    with open(path,'r') as f:
        file = json.load(f)
    return file

batch_size = 32
embedding_size = 128
hidden_size = 512
n_epochs = 80
learning_rate = (1e-3)
weight_decay = 1e-5
beam_size = 5
n_layers = 2

def read_pickle(data_file):
    with open(model_path+data_file, 'rb') as f:
        data = pickle.load(f)
    return data


test1=load_kor_data(path+"train_tunip.json")

(generate_nums, copy_nums) = read_pickle('generate_nums.p')
input_lang = read_pickle('input_lang.p')
output_lang = read_pickle('output_lang.p')

pairs_tested = transfer_num_test(test1)
test_pairs = prepare_data_for_test(pairs_tested, input_lang, output_lang, tree=True)

encoder = EncoderSeq(input_size=input_lang.n_words, embedding_size=embedding_size, hidden_size=hidden_size,
                     n_layers=n_layers)
predict = Prediction(hidden_size=hidden_size, op_nums=output_lang.n_words - copy_nums - 1 - len(generate_nums),
                     input_size=len(generate_nums))
generate = GenerateNode(hidden_size=hidden_size, op_nums=output_lang.n_words - copy_nums - 1 - len(generate_nums),
                        embedding_size=embedding_size)
merge = Merge(hidden_size=hidden_size, embedding_size=embedding_size)
# 모델 초기화
encoder.load_state_dict(torch.load(model_path + "encoder"))
predict.load_state_dict(torch.load(model_path + "predict"))
generate.load_state_dict(torch.load(model_path + "generate"))
merge.load_state_dict(torch.load(model_path + "merge"))
generate_num_ids = []

for num in generate_nums:
    generate_num_ids.append(output_lang.word2index[num])

if USE_CUDA:
    encoder.cuda()
    predict.cuda()
    generate.cuda()
    merge.cuda()

value_ac = 0
equation_ac = 0
eval_total = 0
for test_batch in test_pairs:
    # print(test_batch)
    batch_graph = get_single_example_graph(test_batch[0], test_batch[1], test_batch[7], test_batch[4], test_batch[5])
    test_res = evaluate_tree(test_batch[0], test_batch[1], generate_num_ids, encoder, predict, generate,
                             merge, output_lang, test_batch[5], batch_graph, beam_size=beam_size)
    val_ac, equ_ac, _, _ = compute_prefix_tree_result(test_res, test_batch[2], output_lang, test_batch[4],
                                                      test_batch[6])
    eval_total += 1

    prediction = out_expression_list(test_res, output_lang, test_batch[4], test_batch[6])
    output_codes = generate_code(prediction, var_dict=test_batch[8])

    run_code = '\n'.join(output_codes)
    exec_vars = {}
    try:
        with timeout(30):
            exec(run_code, None, exec_vars)
            if abs(exec_vars['final_result']- float(test_batch[9]))<1e-4:
                value_ac += 1
    except:
        pass


print( value_ac, eval_total)
print("test_answer_acc",  float(value_ac) / eval_total)
print("------------------------------------------------------")