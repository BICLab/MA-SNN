import torch
from utils.util import *


class IFCell(nn.Module):
    def __init__(self,
                 inputSize,
                 hiddenSize,
                 spikeActFun,#放电函数
                 scale=0.3,#
                 pa_dict=None,
                 pa_train_self=False,
                 bias=True,
                 p=0,
                 mode_select='spike',#模式设置，spike：脉冲放电，mem：膜电位放电
                 mem_act=torch.relu,
                 TR_model='NTR',#膜电位是否使用时间常数规则，NTR：不使用，TR：使用
                 ):
        super().__init__()
        self.inputSize = inputSize
        self.hiddenSize = hiddenSize
        self.bias = bias
        self.spikeActFun = spikeActFun

        self.UseDropOut = True
        self.batchSize = None
        self.scale = scale
        self.pa_dict = pa_dict
        self.pa_train_self = pa_train_self
        self.p = p

        # LIAF
        self.TR_model = TR_model
        # lif
        self.mode_select = mode_select
        self.mem_act = mem_act

        if not pa_train_self and pa_dict is None:
            pa_dict = {'alpha': 0.3, 'beta': 0., 'Vreset': 0., 'Vthres': 0.6}
        self.pa_dict = pa_dict

        if self.pa_train_self:

            self.alpha = nn.Parameter(torch.Tensor(1, hiddenSize))
            self.beta = nn.Parameter(torch.Tensor(1, hiddenSize))
            self.Vreset = nn.Parameter(torch.Tensor(1, hiddenSize))
            self.Vthres = nn.Parameter(torch.Tensor(1, hiddenSize))

            nn.init.uniform_(self.alpha, 0, 1)
            nn.init.uniform_(self.beta, 0, 1)
            nn.init.uniform_(self.Vreset, 0, 1)
            nn.init.uniform_(self.Vthres, 0, 1)
        else:
            try:
                self.alpha = self.pa_dict['alpha']
                self.beta = self.pa_dict['beta']
                self.Vreset = self.pa_dict['Vreset']
                self.Vthres = self.pa_dict['Vthres']
            except KeyError:
                print('Please set right LIF hyper-parameter, Now use default setting！！')
                self.pa_dict = {'alpha': 0.3, 'beta': 0., 'Vreset': 0., 'Vthres': 0.3}
                self.alpha = self.pa_dict['alpha']
                self.beta = self.pa_dict['beta']
                self.Vreset = self.pa_dict['Vreset']
                self.Vthres = self.pa_dict['Vthres']
        if 0 < self.p < 1:
            self.dropout = nn.Dropout(p=self.p)

        self.h = None

    def forward(self, input, init_v=None):
        self.batchSize = input.size()[0]

        input = input.reshape([self.batchSize, -1])

        if self.h is None:
            if init_v is None:
                self.h = torch.zeros(self.batchSize,
                                     self.hiddenSize,
                                     device=input.device)

            else:
                self.h = init_v.clone()
        if input.device != self.h.device:
            input = input.to(self.h.device)

        # 整合放电过程
        # Step 1: accumulate and reset,spike used as forgetting gate
        u = self.h + input

        x_ = u - self.Vthres
        x = self.spikeActFun(x_)
        self.h = x * self.Vthres + (1 - x) * u
        self.h = self.h * self.alpha + self.beta

        # step 4:
        if self.mode_select == 'spike':
            x = x
        elif self.mode_select == 'mem':# 膜电位模式

            # TR
            if self.TR_model == 'TR':
                if not self.mem_act:
                    x = x_
                else:
                    x = self.mem_act(x_)
            else:
                if not self.mem_act:
                    x = u
                else:
                    x = self.mem_act(u)

        if 1 > self.p > 0:
            x = self.dropout(x)
        return x

    def reset(self):
        self.h = None


class ConvIFCell(nn.Module):
    def __init__(self,
                 inputSize,
                 hiddenSize,
                 kernel_size,
                 spikeActFun,
                 padding=1,
                 scale=0.02,
                 pa_dict=None,
                 pa_train_self=False,
                 bias=True,
                 p=0,
                 mode_select='spike',
                 mem_act=torch.relu,
                 TR_model='NTR',
                 ):
        super().__init__()
        self.inputSize = inputSize
        self.bias = bias
        self.hiddenSize = hiddenSize
        self.spikeActFun = spikeActFun

        self.batchSize = None
        self.scale = scale
        self.pa_dict = pa_dict
        self.pa_train_self = pa_train_self
        self.kernel_size = kernel_size
        self.padding = padding
        self.p = p

        # LIAF
        self.TR_model = TR_model
        # lif 
        self.mode_select = mode_select
        self.mem_act = mem_act

        if not self.pa_train_self and self.pa_dict is None:
            self.pa_dict = {'alpha': 0.3, 'beta': 0., 'Vreset': 0., 'Vthres': 0.3}

        if self.pa_train_self:
            self.alpha = nn.Parameter(torch.Tensor(self.hiddenSize, 1, 1))
            self.beta = nn.Parameter(torch.Tensor(self.hiddenSize, 1, 1))
            self.Vreset = nn.Parameter(torch.Tensor(self.hiddenSize, 1, 1))
            self.Vthres = nn.Parameter(torch.Tensor(self.hiddenSize, 1, 1))
            nn.init.uniform_(self.alpha, 0, 1)
            nn.init.uniform_(self.beta, 0, 1)
            nn.init.uniform_(self.Vreset, 0, 1)
            nn.init.uniform_(self.Vthres, 0, 1)
        else:
            try:
                self.alpha = self.pa_dict['alpha']
                self.beta = self.pa_dict['beta']
                self.Vreset = self.pa_dict['Vreset']
                self.Vthres = self.pa_dict['Vthres']
            except KeyError:
                print('Please set Full LIF hyper-parameter, Now use default setting！！')
                self.pa_dict = {'alpha': 0.3, 'beta': 0., 'Vreset': 0., 'Vthres': 0.3}
                self.alpha = self.pa_dict['alpha']
                self.beta = self.pa_dict['beta']
                self.Vreset = self.pa_dict['Vreset']
                self.Vthres = self.pa_dict['Vthres']

        if 0 < self.p < 1:
            self.dropout = nn.Dropout2d(p=self.p)
        self.c = None

    def forward(self, input, init_v=None):

        self.batchSize = input.size()[0]

        if self.h is None:
            if init_v is None:
                self.h = torch.zeros(self.batchSize,
                                     self.hiddenSize,
                                     input.size()[-2],
                                     input.size()[-1],
                                     device=input.device)

            else:
                self.h = init_v.clone()

        if input.device != self.h.device:
            input = input.to(self.h.device)

        u = self.h + input

        x_ = u - self.Vthres
        x = self.spikeActFun(x_)

        self.h = x * self.Vthres + (1 - x) * u
        self.h = self.h * self.alpha + self.beta

        # step 4:
        if self.mode_select == 'spike':

            x = x
        elif self.mode_select == 'mem':

            if self.TR_model == 'TR':
                if not self.mem_act:
                    x = x_
                else:
                    x = self.mem_act(x_)
            else:
                if not self.mem_act:
                    x = u
                else:
                    x = self.mem_act(u)

        if 1 > self.p > 0:
            x = self.dropout(x)

        return x

    def reset(self):
        self.h = None
