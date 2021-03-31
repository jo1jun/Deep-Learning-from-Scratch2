# coding: utf-8
import os, sys
os.chdir(os.path.dirname(os.path.abspath(__file__))) 
sys.path.append(os.pardir)
from common.time_layers import *
from seq2seq import Seq2seq, Encoder


class PeekyDecoder:
    def __init__(self, vocab_size, wordvec_size, hidden_size):
        V, D, H = vocab_size, wordvec_size, hidden_size # (13, 16, 128)
        rn = np.random.randn

        embed_W = (rn(V, D) / 100).astype('f')
        lstm_Wx = (rn(H + D, 4 * H) / np.sqrt(H + D)).astype('f') # encoder 에서 전파된 h 를 모든 LSTM 에 concat. (D -> H + D)
        lstm_Wh = (rn(H, 4 * H) / np.sqrt(H)).astype('f')
        lstm_b = np.zeros(4 * H).astype('f')
        affine_W = (rn(H + H, V) / np.sqrt(H + H)).astype('f') # encoder 에서 전파된 h 를 모든 affine 에 concat. (H -> H + H)
        affine_b = np.zeros(V).astype('f')

        self.embed = TimeEmbedding(embed_W)
        self.lstm = TimeLSTM(lstm_Wx, lstm_Wh, lstm_b, stateful=True)
        self.affine = TimeAffine(affine_W, affine_b)

        self.params, self.grads = [], []
        for layer in (self.embed, self.lstm, self.affine):
            self.params += layer.params
            self.grads += layer.grads
        self.cache = None

    def forward(self, xs, h):
        N, T = xs.shape # (128, 4) (batch_size, block_size)
        N, H = h.shape # (128, 128) (batch_size, hidden_size)

        self.lstm.set_state(h)

        out = self.embed.forward(xs) # out.shape = (128, 4, 16) (batch_size, block_size, wordvec_size)
        # 윗 방향으로 forward pass 되는 out(x) 를 encoder 에서 나온 h 와 concat 후, 윗 방향(LSTM) pass.
        hs = np.repeat(h, T, axis=0).reshape(N, T, H) # hs.shape = (128, 4, 128)
        out = np.concatenate((hs, out), axis=2) # out.shape = (128, 4, 144)

        out = self.lstm.forward(out) # out.shape = (128, 4, 128)
        
        # 윗 방향으로 forward pass 되는 out(h) 를 encoder 에서 나온 h 와 concat 후, 윗 방향(Affine) pass.
        out = np.concatenate((hs, out), axis=2) # out.shape = (128, 4, 256)

        score = self.affine.forward(out)
        self.cache = H
        return score

    def backward(self, dscore):
        H = self.cache

        dout = self.affine.backward(dscore)
        dout, dhs0 = dout[:, :, H:], dout[:, :, :H] # forward 때 concat & pass 했으므로 backward 때 split & pass
        dout = self.lstm.backward(dout)
        dembed, dhs1 = dout[:, :, H:], dout[:, :, :H]
        self.embed.backward(dembed)

        dhs = dhs0 + dhs1
        dh = self.lstm.dh + np.sum(dhs, axis=1)
        return dh

    def generate(self, h, start_id, sample_size):
        sampled = []
        char_id = start_id
        self.lstm.set_state(h)

        H = h.shape[1]
        peeky_h = h.reshape(1, 1, H)
        for _ in range(sample_size):
            x = np.array([char_id]).reshape((1, 1))
            out = self.embed.forward(x)

            out = np.concatenate((peeky_h, out), axis=2)
            out = self.lstm.forward(out)
            out = np.concatenate((peeky_h, out), axis=2)
            score = self.affine.forward(out)

            char_id = np.argmax(score.flatten())
            sampled.append(char_id)

        return sampled


class PeekySeq2seq(Seq2seq):
    def __init__(self, vocab_size, wordvec_size, hidden_size):
        V, D, H = vocab_size, wordvec_size, hidden_size
        self.encoder = Encoder(V, D, H)
        self.decoder = PeekyDecoder(V, D, H) # 기존 Seq2seq 에서 초기화 부분의 dcoder 만 replace.
        self.softmax = TimeSoftmaxWithLoss()

        self.params = self.encoder.params + self.decoder.params
        self.grads = self.encoder.grads + self.decoder.grads
