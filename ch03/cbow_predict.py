# coding: utf-8
import os, sys
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.append('..')                           

import numpy as np
from common.layers import MatMul


# 샘플 맥락 데이터 (원 핫 표현)
c0 = np.array([[1, 0, 0, 0, 0, 0, 0]])
c1 = np.array([[0, 0, 1, 0, 0, 0, 0]])

# 가중치 초기화
W_in = np.random.randn(7, 3)
W_out = np.random.randn(3, 7)

# 계층 생성 (bias 생략하므로 Affine 대신 MatMul 로 사용 가능.)
in_layer0 = MatMul(W_in)    # 원하는 맥락 수만큼 input MatMul 계층을 만든다. (가중치는 W_in 으로 공유.)
in_layer1 = MatMul(W_in)
out_layer = MatMul(W_out)   # output MatMul 계층은 한 개만.

# 순전파
h0 = in_layer0.forward(c0)
h1 = in_layer1.forward(c1)
h = 0.5 * (h0 + h1) # 입력층이 N 개면 은닉층 결과를 총합 후 N 으로 나눈다. (평균)
s = out_layer.forward(h)
print(s)
