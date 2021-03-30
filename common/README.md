## trainer.py
  여러 하이퍼파라미터를 입력받아 신경망 학습을 해주는 class 구현

## utils.py
  각종 전처리 함수들
  
  question1 : analogy() 함수에서 단어간 유사도로 dot product 만 한 이유?? cos 유사도로 측정하면 안 되는 것인가?
  answer1 : 내적값이 크다면 유사도가 높다고 할 수 있다. convoluion 생각하면 됨.

## layers.py
  ### MatMul, Affine, Softmax, SoftmaxWithLoss, Sigmoid, SigmoidWithLoss, Dropout, Embedding
