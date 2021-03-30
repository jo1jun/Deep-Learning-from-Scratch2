# coding: utf-8
import os, sys
os.chdir(os.path.dirname(os.path.abspath(__file__))) 
sys.path.append(os.pardir)
from common.np import *
from rnnlm_gen import BetterRnnlmGen
from dataset import ptb


corpus, word_to_id, id_to_word = ptb.load_data('train')
vocab_size = len(word_to_id)
corpus_size = len(corpus)


model = BetterRnnlmGen()
model.load_params('../ch06/BetterRnnlm.pkl')

# start 문자와 skip 문자 설정
start_word = 'you'
start_id = word_to_id[start_word]
skip_words = ['N', '<unk>', '$']
skip_ids = [word_to_id[w] for w in skip_words]
# 문장 생성
word_ids = model.generate(start_id, skip_ids)
txt = ' '.join([id_to_word[i] for i in word_ids])
txt = txt.replace(' <eos>', '.\n')

print(txt)

model.reset_state() # TimeLSTM 의 함수를 호출해 은닉벡터, 기억 셀 초기화

start_words = 'the meaning of life is'
start_ids = [word_to_id[w] for w in start_words.split(' ')]

for x in start_ids[:-1]: # 'the meaning of life'로 차례대로 순전파를 수행 -> LSTM 계층에 이 단어열 정보(은닉 벡터, 기억 셀)가 유지된다. 
    x = np.array(x).reshape(1, 1)
    model.predict(x)

word_ids = model.generate(start_ids[-1], skip_ids) # 순전파 진행 후 'is' 를 start_id로 주면 'the meaning of life is ~~~' 문장 생성.
word_ids = start_ids[:-1] + word_ids # start_ids 와 'is' 포함 새롭게 생성한 word_ids 를 이어 붙여 'the meaning of life is ~~~' 문장 생성.
txt = ' '.join([id_to_word[i] for i in word_ids])
txt = txt.replace(' <eos>', '.\n')
print('-' * 50)
print(txt)