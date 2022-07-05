from Graph2Tree_kor.mawps.src.train_and_evaluate import *
from Graph2Tree_kor.mawps.src.models import *
from Graph2Tree_kor.mawps.src.expressions_transfer import *
from Graph2Tree_kor.mawps.src.pre_data import *
import torch.optim
import warnings
import logging
import sys
sys.path.append('./ai_challenge_2step_data')
print(sys.path.append)
from ai_challenge_2step_data.utils import CustomTokenizer, save_to_json
from ai_challenge_2step_data.data_utils import generate_code_from_binary,generate_code
import pickle


warnings.filterwarnings(action='ignore')
path = 'ai_challenge_2step_data/data/'
# if __name__ == "__main__":
#     model_path = 'web/model_traintest/'
# else:
model_path = 'Graph2Tree_kor/mawps/model_traintest/'
logger = logging.getLogger(__name__)
def read_pickle(data_file):
    with open(model_path+data_file, 'rb') as f:
        data = pickle.load(f)
    return data

class graph2tree():
    def __init__(self):
        (self.generate_nums, self.copy_nums) = read_pickle('generate_nums.p')
        self.input_lang = read_pickle('input_lang.p')
        self.output_lang = read_pickle('output_lang.p')
        self.batch_size = 32
        self.embedding_size = 128
        self.hidden_size = 512
        self.weight_decay = 1e-5
        self.beam_size = 5
        self.n_layers = 2
        self.encoder = EncoderSeq(input_size=self.input_lang.n_words, embedding_size=self.embedding_size, hidden_size=self.hidden_size,
                             n_layers=self.n_layers)
        self.predict = Prediction(hidden_size=self.hidden_size, op_nums=self.output_lang.n_words - self.copy_nums - 1 - len(self.generate_nums),
                             input_size=len(self.generate_nums))
        self.generate = GenerateNode(hidden_size=self.hidden_size,
                                op_nums=self.output_lang.n_words - self.copy_nums - 1 - len(self.generate_nums),
                                embedding_size=self.embedding_size)
        self.merge = Merge(hidden_size=self.hidden_size, embedding_size=self.embedding_size)
        # 모델 초기화
        if torch.cuda.is_available():
            self.encoder.load_state_dict(torch.load(model_path + "encoder"))
            self.predict.load_state_dict(torch.load(model_path + "predict"))
            self.generate.load_state_dict(torch.load(model_path + "generate"))
            self.merge.load_state_dict(torch.load(model_path + "merge"))
            self.encoder.cuda()
            self.predict.cuda()
            self.generate.cuda()
            self.merge.cuda()
        else:
            device = torch.device('cpu')
            self.encoder.load_state_dict(torch.load(model_path + "encoder", map_location=device))
            self.predict.load_state_dict(torch.load(model_path + "predict", map_location=device))
            self.generate.load_state_dict(torch.load(model_path + "generate", map_location=device))
            self.merge.load_state_dict(torch.load(model_path + "merge", map_location=device))

        self.tokenizer = CustomTokenizer()

    def model_input_preprocess(self, question):
        test = []
        output_text, data = self.tokenizer.tokenize(question)
        output_text = self.tokenizer.tokenize_for_train_v3(output_text)
        elem = {}
        elem['src'] = output_text
        elem['Numbers'] = data
        elem['group_num'] = self.tokenizer.get_group_num(output_text)
        elem['src_split'] = elem['src'].split()
        test.append(elem)
        pairs_tested = transfer_num_new(test)
        test_pairs = prepare_data_for_new(pairs_tested, self.input_lang, self.output_lang, tree=True)

        return test_pairs

    def solve(self, question):
        test_batch = self.model_input_preprocess(question)[0]
        encoder = self.encoder
        predict = self.predict
        generate = self.generate
        merge = self.merge
        generate_num_ids = []

        for num in self.generate_nums:
            generate_num_ids.append(self.output_lang.word2index[num])

        batch_graph = get_single_example_graph(test_batch[0], test_batch[1], test_batch[7], test_batch[4],
                                               test_batch[5])
        test_res = evaluate_tree(test_batch[0], test_batch[1], generate_num_ids, encoder, predict, generate,
                                 merge, self.output_lang, test_batch[5], batch_graph, beam_size=self.beam_size)

        prediction = out_expression_list(test_res, self.output_lang, test_batch[4],test_batch[6])

        print('\n## Equation')
        print(prediction)

        print('\n## Generate Python Code & Run ##')

        #output_codes_for_display, output_codes = generate_code(prediction, var_dict=test_batch[8])
        output_codes = generate_code_from_binary(prediction, var_dict=test_batch[8],do_translate=False)

        print('\n### Final Code ###')
        run_code = '\n'.join(output_codes)
        #display_code = '\n'.join(output_codes_for_display)
        #print(display_code)
        print(run_code)

        #display_code_list = display_code.replace('\t','    ').split('\n')
        #print(display_code_list)

        

        print('\n### Run Code ###')
        exec_vars = {}
        exec(run_code, None, exec_vars)

        print('\n### Execution Result ###')
        print(exec_vars['final_result'])

        # print('\n### Korean Solution ###')
        # # with open('solution.txt','r') as f:
        # try:
        #     with open('ai_challenge_2step_data/solution.txt','r') as f:
        #         solution = f.read()
        #         print(solution)
        # except:
        #     try:
        #         with open('solution.txt','r') as f:
        #             solution = f.read()
        #             print(solution)
        #     except:
        #         with open('../solution.txt','r') as f:
        #             solution = f.read()
        #             print(solution)
        #
        # solution_list = solution.split('\n')
        # print(solution_list)


        # return (run_code, display_code)
        #return display_code_list, solution_list
        return run_code

if __name__ == '__main__':
    g2t = graph2tree()
    g2t.solve("4, 2, 1 중에서 서로 다른 숫자 2개를 뽑아 만들 수 있는 가장 큰 두 자리 수를 구하시오.")
