
class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0
        self.count = 0

    # def update(self, x):
    #     self.sum += x.sum()
    #     self.count += x.size(0)

    def update(self, x, n=1):
        self.sum += (x.sum() * n)
        try:
            size = x.size(0)
        except:
            size = 1
        self.count += (size * n)

    def avg(self):
        return self.sum / self.count

    def __str__(self) -> str:
        return str(self.avg())
