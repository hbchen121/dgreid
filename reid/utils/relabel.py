import torch


class ReLabel:
    def __init__(self, nclass):
        self.nclass = nclass
        self.idxSum = [0]
        sum = 0
        for num in nclass:
            sum += num
            self.idxSum.append(sum)
        self.pids = sum
        self.idxSum = torch.tensor(self.idxSum, dtype=torch.long)

    def __call__(self, idx, label):
        assert idx < len(self.nclass)
        if torch.is_tensor(label):
            self.idxSum.to(label.device)
        return self.idxSum[idx] + label


class ReLabel_identity:
    def __init__(self, nclass):
        self.nclass = nclass

    def __call__(self, idx, label):
        return label