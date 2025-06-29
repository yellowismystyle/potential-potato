import torch.nn as nn
from models.llmesr import LLMESR_SASRec  # 可根据使用的具体模型修改

class LLMESRActor(nn.Module):
    def __init__(self, args, user_num, item_num, device):
        super().__init__()
        self.model = LLMESR_SASRec(user_num=user_num, item_num=item_num, device=device, args=args)

    def forward(self, **kwargs):
        return self.model(**kwargs)