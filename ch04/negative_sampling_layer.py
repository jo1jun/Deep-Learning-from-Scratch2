# coding: utf-8
import os, sys
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.append('..')
from common.np import *  # import numpy as np
from common.layers import Embedding, SigmoidWithLoss
import collections


class EmbeddingDot:                             # embedding(idx 에 해당하는 가중치 추출) 와 dot product 를 결합시킨 layer
    def __init__(self, W):
        self.embed = Embedding(W)               # Embedding layer
        self.params = self.embed.params
        self.grads = self.embed.grads
        self.cache = None                       # 순전파 시의 계산 결과를 잠시 유지하기 위한 변수

    def forward(self, h, idx):
        target_W = self.embed.forward(idx)      # embedding layer 의 forward 진행하여 target W 추출
        out = np.sum(target_W * h, axis=1)      # target W 와 은닉층 h 를 dot product

        self.cache = (h, target_W)              # h 와 target_W 를 임시로 저장하여 backward 때 사용한다.
        return out

    def backward(self, dout):
        h, target_W = self.cache                # 임시 저장했던 h 와 target_W 를 불러오고
        dout = dout.reshape(dout.shape[0], 1)
        dtarget_W = dout * h                    # dtarget_W 로 가는 dot product 에 대한 역전파        
        self.embed.backward(dtarget_W)          # embed layer backward 로 dW 를 계산.
        dh = dout * target_W                    # dh 로 가는 dot product 에 대한 역전파
        return dh


class UnigramSampler:                               # 단어의 확률분포를 만들고 0.75제곱 후, negative sampling
    def __init__(self, corpus, power, sample_size): # corpus(단어 ID list), power(제곱할 지수), sample_size
        self.sample_size = sample_size
        self.vocab_size = None
        self.word_p = None

        counts = collections.Counter()              # 단어 등장 횟수 세기
        for word_id in corpus:
            counts[word_id] += 1

        vocab_size = len(counts)                    # 중복 제외 단어 수
        self.vocab_size = vocab_size

        self.word_p = np.zeros(vocab_size)
        for i in range(vocab_size):
            self.word_p[i] = counts[i]              # 단어 빈도수

        self.word_p = np.power(self.word_p, power)  # 0.75 제곱
        self.word_p /= np.sum(self.word_p)          # 총합으로 나누어 확률분포로 변환

    def get_negative_sample(self, target):
        batch_size = target.shape[0]

        if not GPU:
            negative_sample = np.zeros((batch_size, self.sample_size), dtype=np.int32)

            for i in range(batch_size):
                p = self.word_p.copy()
                target_idx = target[i]
                p[target_idx] = 0                   # target 제외
                p /= p.sum()
                negative_sample[i, :] = np.random.choice(self.vocab_size, size=self.sample_size, replace=False, p=p)
        else:
            # GPU(cupy）로 계산할 때는 속도를 우선한다.
            # 부정적 예에 타깃이 포함될 수 있다.
            negative_sample = np.random.choice(self.vocab_size, size=(batch_size, self.sample_size),
                                               replace=True, p=self.word_p)

        return negative_sample


class NegativeSamplingLoss:                                         # negative sampling + loss calculating layer
    def __init__(self, W, corpus, power=0.75, sample_size=5):       # W(출력층 가중치), corpus(단어 ID list), power(제곱할 지수), sample_size
        self.sample_size = sample_size
        self.sampler = UnigramSampler(corpus, power, sample_size)   # 위에서 구현한 sampler 클래스를 생성하여 인스턴스 변수 저장
        self.loss_layers = [SigmoidWithLoss() for _ in range(sample_size + 1)]      # 1 + negative sample 의 숫자 만큼 loss layer 생성
        self.embed_dot_layers = [EmbeddingDot(W) for _ in range(sample_size + 1)]   # 1 + negative sample 의 숫자 만큼 embed & dot layer 생성
                                                                                    # 1 은 positive sample 즉, target 을 의미한다.
        self.params, self.grads = [], []
        for layer in self.embed_dot_layers:
            self.params += layer.params
            self.grads += layer.grads

    def forward(self, h, target):
        batch_size = target.shape[0]
        negative_sample = self.sampler.get_negative_sample(target)  # negative sampling

        # 긍정적 예 순전파
        score = self.embed_dot_layers[0].forward(h, target)
        correct_label = np.ones(batch_size, dtype=np.int32)
        loss = self.loss_layers[0].forward(score, correct_label)

        # 부정적 예 순전파
        negative_label = np.zeros(batch_size, dtype=np.int32)
        for i in range(self.sample_size):
            negative_target = negative_sample[:, i]
            score = self.embed_dot_layers[1 + i].forward(h, negative_target)
            loss += self.loss_layers[1 + i].forward(score, negative_label)

        return loss # positive sample 의 loss + negetive sample 의 loss 총합이 최종 loss

    def backward(self, dout=1):
        dh = 0
        for l0, l1 in zip(self.loss_layers, self.embed_dot_layers):
            dscore = l0.backward(dout)  # loss layer backprop 이후 아래에서 embed & dot layer backprop.
            dh += l1.backward(dscore)   # 은닉층 뉴런은 forward 할 때 여러개로 분기(복사)되었으므로 baward 시 기울기들을 더해준다.

        return dh
