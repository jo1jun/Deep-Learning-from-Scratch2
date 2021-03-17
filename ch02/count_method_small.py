# coding: utf-8
import os, sys
os.chdir(os.path.dirname(os.path.abspath(__file__))) 
sys.path.append(os.pardir)
import numpy as np
import matplotlib.pyplot as plt
from common.util import preprocess, create_co_matrix, ppmi


text = 'You say goodbye and I say hello.'
corpus, word_to_id, id_to_word = preprocess(text)
vocab_size = len(id_to_word)
C = create_co_matrix(corpus, vocab_size, window_size=1) # 동시발생 행렬
W = ppmi(C) # PPMI 행렬

# SVD
U, S, V = np.linalg.svd(W)  # SVD 를 통해 희소벡터(W)를 밀집벡터(U)로 변환

np.set_printoptions(precision=3)  # 유효 자릿수를 세 자리로 표시
print(C[0])
print(W[0])
print(U[0])

# 플롯
for word, word_id in word_to_id.items():
    # 차원 축소 : 단순히 U에서 순차적으로 원하는 차원(column) 수 만큼 꺼내면 된다. 2차원으로 줄이므로 0, 1 을 꺼낸다.
    plt.annotate(word, (U[word_id, 0], U[word_id, 1]))  # annotate(word, x, y) : x, y 좌표에 word에 담긴 텍스트를 그린다.
plt.scatter(U[:,0], U[:,1], alpha=0.5)  # alpha : 투명도 (0 ~ 1)
plt.show()
