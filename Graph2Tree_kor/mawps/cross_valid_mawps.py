# coding: utf-8
import random
import pickle
from Graph2Tree_kor.mawps.src.pre_data import *
from Graph2Tree_kor.mawps.src.train_and_evaluate import *
from Graph2Tree_kor.mawps.src.models import *
import time
import torch.optim
from Graph2Tree_kor.mawps.src.expressions_transfer import *
import json
import warnings
from torchtext.vocab import Vectors

warnings.filterwarnings(action='ignore')

def read_json(path):
    with open(path,'r', encoding='utf-8-sig') as f:
        file = json.load(f)
    return file
def save_pickle(data,name):
    with open(f'model_traintest/{name}.p', 'wb') as file:
        pickle.dump(data, file)

batch_size = 32
embedding_size = 128
hidden_size = 512
n_epochs = 75
learning_rate = (1e-3)
weight_decay = 1e-5
beam_size = 5
n_layers = 2

path = '../../ai_challenge_2step_data/data/'
#prefix = '23k_processed.json'
#data = load_mawps_data("data/mawps_combine.json")

#train = load_kor_data(path+"train.json")
#data = load_kor_data(path+"test.json")
data= load_kor_data(path+"all_preprocessed.json")

size = len(data)
#data += train
pairs, generate_nums, copy_nums = transfer_kor_num(data)
pairs_trained = pairs
pairs_tested = pairs[:size]

best_acc_fold = []


# random.seed(42)
# data = load_kor_data(path+"1_preprocessed.json")
# pairs, generate_nums, copy_nums = transfer_kor_num(data)
# random.shuffle(pairs)
# size = int(len(pairs)*0.8)
# pairs_trained = pairs[:size]
# pairs_tested = pairs[size:]

fold_size = int(len(pairs) * 0.2)
fold_pairs = []

input_lang, output_lang, train_pairs, test_pairs = prepare_data(pairs_trained, pairs_tested, 1, generate_nums,
                                                        copy_nums,  tree=True)
#print('train_pairs[0]')
#print(train_pairs[0])
#exit()
# Initialize models
pair_nums=(generate_nums, copy_nums)

save_pickle(pair_nums,'generate_nums')
save_pickle(input_lang,'input_lang')
save_pickle(output_lang,'output_lang')

#weights_matrix = read_vector('./data/glove.txt', word2idx=input_lang.word2index)

#encoder = EncoderSeq(input_size=input_lang.n_words, embedding_size=embedding_size, hidden_size=hidden_size,
#                     n_layers=n_layers,weights_matrix=weights_matrix)
encoder = EncoderSeq(input_size=input_lang.n_words, embedding_size=embedding_size, hidden_size=hidden_size,
                     n_layers=n_layers)
predict = Prediction(hidden_size=hidden_size, op_nums=output_lang.n_words - copy_nums - 1 - len(generate_nums),
             input_size=len(generate_nums))
generate = GenerateNode(hidden_size=hidden_size, op_nums=output_lang.n_words - copy_nums - 1 - len(generate_nums),
                embedding_size=embedding_size)
merge = Merge(hidden_size=hidden_size, embedding_size=embedding_size)
# the embedding layer is  only for generated number embeddings, operators, and paddings

encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=learning_rate, weight_decay=weight_decay)
predict_optimizer = torch.optim.Adam(predict.parameters(), lr=learning_rate, weight_decay=weight_decay)
generate_optimizer = torch.optim.Adam(generate.parameters(), lr=learning_rate, weight_decay=weight_decay)
merge_optimizer = torch.optim.Adam(merge.parameters(), lr=learning_rate, weight_decay=weight_decay)

encoder_scheduler = torch.optim.lr_scheduler.StepLR(encoder_optimizer, step_size=20, gamma=0.5)
predict_scheduler = torch.optim.lr_scheduler.StepLR(predict_optimizer, step_size=20, gamma=0.5)
generate_scheduler = torch.optim.lr_scheduler.StepLR(generate_optimizer, step_size=20, gamma=0.5)
merge_scheduler = torch.optim.lr_scheduler.StepLR(merge_optimizer, step_size=20, gamma=0.5)

# Move models to GPU
if USE_CUDA:
    encoder.cuda()
    predict.cuda()
    generate.cuda()
    merge.cuda()

generate_num_ids = []
for num in generate_nums:
    generate_num_ids.append(output_lang.word2index[num])

for epoch in range(n_epochs):
    encoder_scheduler.step()
    predict_scheduler.step()
    generate_scheduler.step()
    merge_scheduler.step()
    loss_total = 0
    input_batches, input_lengths, output_batches, output_lengths, nums_batches, \
    num_stack_batches, num_pos_batches, num_size_batches, num_value_batches, graph_batches = prepare_train_batch(train_pairs, batch_size)
    # input_batches : padding 한 input
    # input_lengths : input 문장 토큰개수
    # output_batches : padding 한 output
    # output_lengths : output 문장 토큰개수
    # nums_batches : len(num)
    # num_stack_batches
    # num_pos_batches : 숫자 위치 index
    # num_size_batches : len(num_pos)
    # num_value_batches : 숫자 값
    # graph_batches
    # get_quantity_cell_graph : 숫자가 아닌 group num 중에 num_pos – group num <4 이면 1
    # get_greater_num_graph : 숫자끼리 값 비교해서 a>b 이면 graph[a][b]=1 , 나머지 0
    # get_lower_num_graph
    # get_quantity_between_graph :숫자 가 아닌 group num 중에 num_pos - group num <4 이면 1  or 숫자인 성분 1
    # get_attribute_between_graph: 숫자가 아닌 group num 중에 num_pos - group num <4 이면 1 or group num 이 같은 단어를 가리키고 있으면 1
    print("epoch:", epoch + 1)
    start = time.time()
    for idx in range(len(input_lengths)):
        loss = train_tree(
            input_batches[idx], input_lengths[idx], output_batches[idx], output_lengths[idx],
            num_stack_batches[idx], num_size_batches[idx], generate_num_ids, encoder, predict, generate, merge,
            encoder_optimizer, predict_optimizer, generate_optimizer, merge_optimizer, output_lang, num_pos_batches[idx], graph_batches[idx])
        loss_total += loss
    print("loss:", loss_total / len(input_lengths))
    print("training time", time_since(time.time() - start))
    print("--------------------------------")
    # if epoch > n_epochs - 10 or epoch % 10 ==0:
    #     value_ac = 0
    #     equation_ac = 0
    #     eval_total = 0
    #     start = time.time()
    #     for test_batch in test_pairs:
    #         #print(test_batch)
    #         batch_graph = get_single_example_graph(test_batch[0], test_batch[1], test_batch[7], test_batch[4], test_batch[5])
    #         test_res = evaluate_tree(test_batch[0], test_batch[1], generate_num_ids, encoder, predict, generate,
    #                                  merge, output_lang, test_batch[5], batch_graph, beam_size=beam_size)
    #         val_ac, equ_ac, _, _ = compute_prefix_tree_result(test_res, test_batch[2], output_lang, test_batch[4], test_batch[6])
    #         #test_batch[8]: NUM_DICT
    #         if val_ac:
    #             value_ac += 1
    #
    #         eval_total += 1
    #     print( value_ac, eval_total)
    #     print("test_answer_acc", float(value_ac) / eval_total)
    #     print("testing time", time_since(time.time() - start))
    #     print("------------------------------------------------------")
torch.save(encoder.state_dict(), "model_traintest/encoder")
torch.save(predict.state_dict(), "model_traintest/predict")
torch.save(generate.state_dict(), "model_traintest/generate")
torch.save(merge.state_dict(), "model_traintest/merge")

