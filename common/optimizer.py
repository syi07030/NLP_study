class SGD:
    def __init__(self, lr = 0.01) -> None:
        self.lr = lr #lr은 학습률
        
    def update(self, params, grads): #매개변수 갱신
        for i in range(len(params)):
            params[i] -= self.lr * grads[i]