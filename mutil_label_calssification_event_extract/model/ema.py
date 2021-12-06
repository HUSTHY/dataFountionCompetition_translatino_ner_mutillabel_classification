class EMA():
    """
    通过滑动平均来更新模型的参数，增加模型的鲁棒性
    示例如下
    ....
    model = Model()
    ema = Ema(model,decay = 0.999)
    ema.register()
    #训练阶段
    for epoch in epochs:
        for batch in dataloader:
            ....
            loss.backforward()
            optimizer.zero_grad()
            optimizer.step()
            ema.update()

        ema.apply_shadow()
        model.eval()
        #模型评估
        evaluate()
        #保存最佳的那个模型
        model.save()——————这里模型就是保存的影子权重
        #ema.restore()————————这里又恢复模型参数，这一步做了效果好一些，没有做导致模型参数变化过大

    ......
    #模型推理——由于保存的是影子权重所以没有问题
    model.eval()
    test(model)
    """
    def __init__(self, model, decay):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}