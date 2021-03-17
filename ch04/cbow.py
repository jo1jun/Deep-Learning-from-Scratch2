# coding: utf-8
import os, sys
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.append('..')
from common.np import *  # import numpy as np
from common.layers import Embedding
from ch04.negative_sampling_layer import NegativeSamplingLoss


class CBOW:
    def __init__(self, vocab_size, hidden_size, window_size, corpus):  # vocab_size : 어휘 수, hidden_size : 은닉층 뉴런 수, window_size : 맥락의 크기
        V, H = vocab_size, hidden_size # 단어의 종류, 각 단어의 분산표현 dimension

        # 가중치 초기화
        W_in = 0.01 * np.random.randn(V, H).astype('f')
        W_out = 0.01 * np.random.randn(V, H).astype('f')   # 둘 다 Embedding 계층을 사용하기 때문에 형상이 같다.

        # 계층 생성
        self.in_layers = []
        for i in range(2 * window_size):  # 맥락의 수 만큼 Embedding layer 생성 (W_in) 각 Embedding layer 는 맥락 단어 1개를 batch 크기만큼 처리한다.
            layer = Embedding(W_in)  # Embedding 계층 사용
            self.in_layers.append(layer)
        self.ns_loss = NegativeSamplingLoss(W_out, corpus, power=0.75, sample_size=5)  # 1 + negative sample 수 만큼 Embedding(+dot) lyaer 생성 (W_out)

        # 모든 가중치와 기울기를 배열에 모은다.
        layers = self.in_layers + [self.ns_loss]
        self.params, self.grads = [], []
        for layer in layers:
            self.params += layer.params
            self.grads += layer.grads

        # 인스턴스 변수에 단어의 분산 표현을 저장한다.
        self.word_vecs = W_in

    def forward(self, contexts, target):  # contexts.shape = (batch_size, 맥락 수) / target.shape = (batch_size, 1)
        h = 0
        for i, layer in enumerate(self.in_layers):
            h += layer.forward(contexts[:, i])
        h *= 1 / len(self.in_layers)  # 평균
        loss = self.ns_loss.forward(h, target)
        return loss

    def backward(self, dout=1):
        dout = self.ns_loss.backward(dout)
        dout *= 1 / len(self.in_layers)
        for layer in self.in_layers:
            layer.backward(dout)  # layer 가 직렬이 아닌 병렬적으로 결합되어서 역순으로 backward 할 필요가 없다.
        return None
