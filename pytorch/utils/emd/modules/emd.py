from torch.nn.modules.module import Module
from functions.emd import EMDFunction

class EMDModule(Module):
    def forward(self, input1, input2):
        return EMDFunction()(input1, input2)
