# coding: utf-8
import os, sys
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.append('..')                           

import numpy as np
from common.layers import MatMul, SoftmaxWithLoss


class SimpleSkipGram:
    def __init__(self, vocab_size, hidden_size):
        V, H = vocab_size, hidden_size

        # 가중치 초기화
        W_in = 0.01 * np.random.randn(V, H).astype('f')
        W_out = 0.01 * np.random.randn(H, V).astype('f')

        # 계층 생성
        self.in_layer = MatMul(W_in)            # input MatMul 계층은 한 개만.
        self.out_layer = MatMul(W_out)
        self.loss_layer1 = SoftmaxWithLoss()    # 원하는 맥락 수만큼 output loss 계층을 만든다.
        self.loss_layer2 = SoftmaxWithLoss()

        # 모든 가중치와 기울기를 리스트에 모은다.
        layers = [self.in_layer, self.out_layer]
        self.params, self.grads = [], []
        for layer in layers:
            self.params += layer.params
            self.grads += layer.grads

        # 인스턴스 변수에 단어의 분산 표현(단어의 밀집벡터)을 저장한다.
        self.word_vecs = W_in

    def forward(self, contexts, target):
        h = self.in_layer.forward(target)
        s = self.out_layer.forward(h)
        l1 = self.loss_layer1.forward(s, contexts[:, 0])    # 맥락마다 개별적으로 loss 를 구한다.
        l2 = self.loss_layer2.forward(s, contexts[:, 1])
        loss = l1 + l2                                      # 개별 loss들을 모두 더한 값을 최종 손실로 한다.
        return loss

    def backward(self, dout=1):
        dl1 = self.loss_layer1.backward(dout)
        dl2 = self.loss_layer2.backward(dout)
        ds = dl1 + dl2  # forward 할 때 분기했으므로 backward 할 때는 더해준다.
        dh = self.out_layer.backward(ds)
        self.in_layer.backward(dh)
        return None
