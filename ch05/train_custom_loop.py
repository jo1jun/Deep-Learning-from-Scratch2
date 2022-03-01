# coding: utf-8
import os, sys
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.append('..')       
import matplotlib.pyplot as plt
import numpy as np
from common.optimizer import SGD
from dataset import ptb
from simple_rnnlm import SimpleRnnlm

# 하이퍼파라미터 설정
batch_size = 10 # 배치 개수를 의미. (각 배치안의 원소 수는 전체입력 999개 중 10개로 나누었으므로 99개)
wordvec_size = 100 # 단어의 분산 표현 벡터의 원소 수
hidden_size = 100 # RNN의 은닉 상태 벡터의 원소 수
time_size = 5     # Truncated BPTT가 한 번에 펼치는 시간 크기 (batch 마다 99개의 원소가 있고 5개 원소를 가진 block 19개로 구성됨)
lr = 0.1
max_epoch = 100

# 학습 데이터 읽기(전체 중 1000개만)
corpus, word_to_id, id_to_word = ptb.load_data('train')
corpus_size = 1000
corpus = corpus[:corpus_size]
vocab_size = int(max(corpus) + 1)


xs = corpus[:-1]  # 입력 (마지막 단어 다음은 없으므로 예측 x 그래서 :-1) 999개
ts = corpus[1:]   # 출력(정답 레이블) (다음에 출연할 단어가 정답이다. 그래서 1:) 999개
data_size = len(xs)
print('말뭉치 크기: %d, 어휘 수: %d' % (corpus_size, vocab_size))

# 학습 시 사용하는 변수
max_iters = data_size // (batch_size * time_size) # 999 / 50 = 19 (block 수)
time_idx = 0
total_loss = 0
loss_count = 0
ppl_list = []

# 모델 생성
model = SimpleRnnlm(vocab_size, wordvec_size, hidden_size)
optimizer = SGD(lr)

# 미니배치의 각 샘플의 읽기 시작 위치를 계산
jump = (corpus_size - 1) // batch_size # 입력이 999개 이므로 배치 개수인 10으로 나누고 소수점 버려서 99개씩이 batch 안의 데이터 수.
offsets = [i * jump for i in range(batch_size)]

for epoch in range(max_epoch):
    
    # time_idx = 0 # batch 끼리는 순전파, 역전파 둘 다 안 이어지고 독립적으로 학습한다.
                   # 따라서 각각의 배치가 동일한 원소 범위를 계속 보면서 학습해도 큰 차이가 없다. (에포크 마다 time_idx = 0 설정)
                   # 그런데 숫자가 딱 떨어지지 않아서 계속해서 못 보는 데이터가 있다. (배치1 : 0~95, 배치2 : 99~194 ...)
                   # 숫자가 딱 안 떨어지기 때문에 계속해서 보지 않는 원소까지 학습하기 위해서 time_idx 를 0으로 매번 초기화 안 하고
                   # (offset + time_idx) % data_size 로 각각의 배치가 보는 원소 범위를 이동시키는 것 같다. 그리고 이게 합리적.
                   # 배치가 보는 범위를 고정하든 안 하든 성능에서 큰 차이는 없다. (데이터 수가 작아서 그런듯??)
                   
                   # 그냥 위 내용은 무시해도 된다. 이해도가 낮은 상태에서 질문을 던졌던 것들.
                   # 앞으로 배치가 보는 원소 범위도 이동시키는 것으로 이해하자. (맨 처음 시작위치만 고정하고 순회.)
                   
    for iter in range(max_iters):
        # 미니배치 취득
        batch_x = np.empty((batch_size, time_size), dtype='i') # (batch 개수, block 안의 데이터 개수) = (10, 5)
        batch_t = np.empty((batch_size, time_size), dtype='i') # 행렬(사각형) 을 생각하면 됨.
        for t in range(time_size): # block 크기인 5 만큼 반복해서 batch 를 채운다. (옆으로 이동)
            for i, offset in enumerate(offsets): # 10개의 배치에 각각의 위치마다의 원소 삽입 (10 * 5 행렬에서 열단위로 원소를 채운다.)
                batch_x[i, t] = xs[(offset + time_idx) % data_size] # 말뭉치 크기를 넘어가면 다시 처음부터.
                batch_t[i, t] = ts[(offset + time_idx) % data_size]
                
            time_idx += 1 # 각 배치가 보는 원소의 범위를 계속해서 이동시킨다. (말뭉치 크기 넘어가면 다시 처음부터 본다.)
        
        # 기울기를 구하여 매개변수 갱신
        # batch 들에 block 크기만큼 원소들을 채워서 forward 시키고 이를 반복.
        loss = model.forward(batch_x, batch_t)
        model.backward() # block 을 담고있는 batch 들을 채우고 backward 하는 데, Truncated BPTT 이므로 각 block 에서만 역전파하고
        optimizer.update(model.params, model.grads) # 갱신한다.
        total_loss += loss
        loss_count += 1
    # 에폭마다 퍼플렉서티 평가
    ppl = np.exp(total_loss / loss_count) # 반복 수 만큼나눠서 평균손실 구함.
    print('| 에폭 %d | 퍼플렉서티 %.2f'
          % (epoch+1, ppl))
    ppl_list.append(float(ppl))
    total_loss, loss_count = 0, 0

# 그래프 그리기
x = np.arange(len(ppl_list))
plt.plot(x, ppl_list, label='train')
plt.xlabel('epochs')
plt.ylabel('perplexity')
plt.show()
