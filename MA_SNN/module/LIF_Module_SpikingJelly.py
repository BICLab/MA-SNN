import torch
from torch import nn
from spikingjelly.activation_based.neuron import IFNode,surrogate
from module.Attention import *
from typing import Callable

# 模型权重初始化
def paramInit(model, method='xavier'):
    scale = 0.05
    for name, w in model.named_parameters():
        if 'weight' in name:
            if method == 'xavier':
                nn.init.xavier_normal_(w)
            elif method == 'kaiming':
                nn.init.kaiming_normal_(w)
            else:
                nn.init.normal_(w)
                w *= scale
        elif 'bias' in name:
            nn.init.constant_(w, 0)
        else:
            pass

class AttLIF(nn.Module):
    def __init__(
        self,
        inputSize:int,
        hiddenSize:int,
        attention:str="TA",# 是否应用注意力
        useBatchNorm:bool=False,
        init_method:str=None,# 模型权重初始化方式
        pa_dict:dict=None,# 电压参数字典
        bias:bool=True,
        track_running_stats:bool=False,# batchnorm参数
        step_mode:str='m',# multi_step or single_step
        surrogate_function:Callable=surrogate.Sigmoid(),
        onlyLast:bool=False,# 是否选择仅保留最后时刻的电压值
        backend:str='cupy',# IFNode是否使用cupy进行后端
        T:int=60,
        t_ratio:int=16
    ) -> None:
        super().__init__()
        self.store_v_req = not onlyLast
        self.step_mode = step_mode
        self.useBatchNorm = useBatchNorm
        self.surrogate_function = surrogate_function
        self.backend = backend

        self.network = nn.Sequential()
        self.attention_flag = attention
        self.linear = nn.Linear(
            in_features=inputSize,
            out_features=hiddenSize,
            bias=bias,
        )

        if self.useBatchNorm:
            self.BNLayer = nn.BatchNorm1d(
                num_features=hiddenSize, track_running_stats=track_running_stats
            )

        if init_method is not None:
            paramInit(model=self.linear, method=init_method)
        if self.attention_flag == "TA":
            self.attention = TA(T, hiddenSize, t_ratio=t_ratio, fc=True)
        elif self.attention_flag == "no":
            pass

        if pa_dict is None:
            pa_dict={'Vreset': 0., 'Vthres': 0.6}
        self.v_threshold = float(pa_dict["Vthres"]) if pa_dict["Vthres"] is not None else 0.6
        self.v_reset = float(pa_dict["Vreset"]) if pa_dict["Vreset"] is not None else None
        
        self.network.add_module(
            "IFNode",
            IFNode(v_threshold=self.v_threshold,v_reset=self.v_reset,surrogate_function=self.surrogate_function,
                   step_mode=self.step_mode,backend=self.backend,store_v_seq=self.store_v_req)
            )
    
    def forward(self,data:torch.Tensor) -> torch.Tensor:
        #*对data shape的操作应该视具体问题进行调整*
        for layer in self.network:
            layer.reset()

        b, t, _ = data.size()
        output = self.linear(data.reshape(b * t, -1))

        if self.useBatchNorm:
            output = self.BNLayer(output)

        outputsum = output.reshape(b, t, -1)
        
        if self.attention_flag == "no":
            data=outputsum
        else:
            data=self.attention(outputsum)
        
        #*单独使用此神经元时建议取消本行代码的注释*
        # data.reshape(b, t, c, h, w)
        
        output=self.network(data.transpose(0,1))

        return output.transpose(0,1)

class ConvAttLIF(nn.Module):
    def __init__(
        self,
        inputSize:int,
        hiddenSize:int,
        kernel_size:tuple,
        attention:str="TA",
        onlyLast:bool=False,
        padding:int=1,
        useBatchNorm:bool=False,
        init_method:str=None,
        pa_dict:dict=None,
        step_mode:str="m",
        surrogate_function:Callable=surrogate.Sigmoid(),
        backend:str="cupy",
        T:int=60,
        stride:int=1,
        pooling_kernel_size:int=1,
        p:float=0,#dropout几率
        track_running_stats:int=False,
        c_ratio=16,
        t_ratio=16
    ) -> None:
        super().__init__()
        self.store_v_req = not onlyLast
        self.step_mode = step_mode
        self.useBatchNorm = useBatchNorm
        self.surrogate_function = surrogate_function
        self.backend = backend
        self.attention_flag = attention
        self.p=p

        self.conv2d = nn.Conv2d(
            in_channels=inputSize,
            out_channels=hiddenSize,
            kernel_size=kernel_size,
            bias=True,
            padding=padding,
            stride=stride,
        )

        if init_method is not None:
            paramInit(model=self.conv2d, method=init_method)

        self.useBatchNorm = useBatchNorm

        if self.useBatchNorm:
            self.BNLayer = nn.BatchNorm2d(
                hiddenSize, track_running_stats=track_running_stats
            )

        self.pooling_kernel_size = pooling_kernel_size
        if self.pooling_kernel_size > 1:
            self.pooling = nn.AvgPool2d(kernel_size=pooling_kernel_size)

        if self.attention_flag == "TCSA":
            self.attention = TCSA(T, hiddenSize, c_ratio=c_ratio, t_ratio=t_ratio)
        elif self.attention_flag == "TSA":
            self.attention = TSA(T, hiddenSize, t_ratio=t_ratio)
        elif self.attention_flag == "TCA":
            self.attention = TCA(T, hiddenSize, c_ratio=c_ratio, t_ratio=t_ratio)
        elif self.attention_flag == "CSA":
            self.attention = CSA(T, hiddenSize, c_ratio=c_ratio)
        elif self.attention_flag == "TA":
            self.attention = TA(T, hiddenSize, t_ratio=t_ratio)
        elif self.attention_flag == "CA":
            self.attention = CA(T, hiddenSize, c_ratio=c_ratio)
        elif self.attention_flag == "SA":
            self.attention = SA(T, hiddenSize)
        elif self.attention_flag == "no":
            pass
        
        if pa_dict is None:
            pa_dict={'alpha': 0.3, 'beta': 0., 'Vreset': 0., 'Vthres': 0.6}
        self.v_threshold = pa_dict["Vthres"]
        self.v_reset = pa_dict["Vreset"]  

        self.network = nn.Sequential()
        self.network.add_module(
            "ConvIF",
            IFNode(v_threshold=self.v_threshold,v_reset=self.v_reset,surrogate_function=self.surrogate_function,
                   step_mode=self.step_mode,backend=self.backend,store_v_seq=self.store_v_req)
        )
        if 0 < self.p < 1:
            self.network.add_module(
                "ConvIF_Dropout",
                nn.Dropout2d(p=self.p)
            )

    def forward(self,data:torch.Tensor) -> torch.Tensor:

        for layer in self.network:
            layer.reset()

        b, t, c, h, w = data.size()
        out = data.reshape(b * t, c, h, w)
        output = self.conv2d(out)

        if self.useBatchNorm:
            output = self.BNLayer(output)

        if self.pooling_kernel_size > 1:
            output = self.pooling(output)

        _, c, h, w = output.size()
        outputsum = output.reshape(b, t, c, h, w)

        if self.attention_flag == "no":
            data = outputsum
        else:
            data = self.attention(outputsum)

        output=self.network(data.transpose(0,1))

        return output.transpose(0,1)