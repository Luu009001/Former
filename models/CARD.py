import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange
import numpy as np
from layers.SEAttention import SEAttention
from layers.SKAttention import SKAttention
from layers.CBAM import CBAMBlock
from layers.ECANet import ECAAttention
from layers.MGTU import Multi_GTU
from layers.AxialAttention import AxialAttention
from layers.CR_MSA import CrossRegionAttntion
from layers.Trend_Forecast import RobTF
from layers.RevIN import RevIN, ResidualMLP
from layers.koopa import KoopmanDecomposition


class Transpose(nn.Module):
    def __init__(self, *dims, contiguous=False): 
        super().__init__()
        self.dims, self.contiguous = dims, contiguous
    def forward(self, x):
        if self.contiguous: return x.transpose(*self.dims).contiguous()
        else: return x.transpose(*self.dims)

# class moving_avg(nn.Module):
#     """
#     Moving average block to highlight the trend of time series
#     """

#     def __init__(self, kernel_size, stride):
#         super(moving_avg, self).__init__()
#         self.kernel_size = kernel_size
#         self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)
#     def forward(self, x):
#         # padding on the both ends of time series
#         front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
#         end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
#         x = torch.cat([front, x, end], dim=1)
#         x = self.avg(x.permute(0, 2, 1))
#         x = x.permute(0, 2, 1)


#         return x


# class series_decomp(nn.Module):
#     """
#     Series decomposition block
#     """

#     def __init__(self, kernel_size):
#         super(series_decomp, self).__init__()
#         self.moving_avg = moving_avg(kernel_size, stride=1)

#     def forward(self, x):
#         moving_mean = self.moving_avg(x) # 提取的趋势
#         res = x - moving_mean # 季节项
#         return res, moving_mean


# class Model(nn.Module):  #add
#     def __init__(self, config, **kwargs):
        
#         super().__init__()    
#         self.model = CARDformer(config)
#         self.task_name = config.task_name
#         self.decomp = series_decomp(kernel_size=config.moving_avg)   #add
#         # self.decomp = KoopmanDecomposition(dynamic_dim=128)   #add4
#         self.robTF = RobTF(thetas_dim1 = 512, thetas_dim2 = 512, seq_len = config.seq_len, pred_len = config.pred_len)  #add
#         self.residual_mlp = ResidualMLP(config.seq_len, config.d_model, config.pred_len)  #add3
#         self.mlp = nn.Sequential(
#             nn.Linear(config.seq_len, config.d_model),
#             nn.ReLU(),
#             nn.Linear(config.d_model, config.d_model),
#             nn.ReLU(),
#             nn.Linear(config.d_model, config.pred_len)
#         )
#         self.revin_trend = RevIN(config.enc_in)  #add3
#     def forward(self, x, *args, **kwargs): 

#         seasonal_init, trend_init = self.decomp(x)  #add
#         seasonal_init=seasonal_init.permute(0,2,1) #add  #[32, 96, 7]-->[32, 7, 96]
#         # trend_init=trend_init.permute(0,2,1) #add      #[32, 96, 7]-->[32, 7, 96]

#         trend_init = self.revin_trend(trend_init, 'norm')  #add3
#         trend_output = self.residual_mlp(trend_init.permute(0, 2, 1)).permute(0, 2, 1) #add3
#         # trend_output = self.mlp(trend_init.permute(0, 2, 1)).permute(0, 2, 1) #add3

#         trend_output = self.revin_trend(trend_output, 'denorm') #add3

#         # trend_output = self.robTF(trend_init) #add
        
#         mask = args[-1]            
#         seasonal_out= self.model(seasonal_init,mask = mask).permute(0,2,1)  #[32, 7, 96]-->[32, 96, 7]
#         # trend_out= self.model(trend_init,mask = mask).permute(0,2,1)   #[32, 7, 96]-->[32, 96, 7]

#         x=seasonal_out+trend_output #[32, 96, 7]

#         return x
    
class Model(nn.Module):
    def __init__(self, config, **kwargs):
        
        super().__init__()    
        self.model = CARDformer(config)
        self.task_name = config.task_name
        self.cluster_prob = None#add
    def forward(self, x, *args, **kwargs): 

        x = x.permute(0,2,1)    #[32, 96, 7]-->[32, 7, 96]

        # mask = args[-1]            
        x= self.model(x)
        if self.task_name != 'classification':
            x = x.permute(0,2,1)    #[32, 7, 96]-->[32, 96, 7]
        return x    

def FFT_for_Period(x, k=2):   ##add2
    # [B, T, C]
    xf = torch.fft.rfft(x, dim=1)
    # find period by amplitudes
    frequency_list = abs(xf).mean(0).mean(-1)
    frequency_list[0] = 0
    _, top_list = torch.topk(frequency_list, k)
    top_list = top_list.detach().cpu().numpy()
    period = x.shape[1] // top_list
    return period, abs(xf).mean(-1)[:, top_list]


    
class CARDformer(nn.Module):
    def __init__(self, 
                 config,**kwargs):
        
        super().__init__()
        
        self.patch_len  = config.patch_len
        self.seq_len = config.seq_len
        self.pred_len = config.pred_len
        self.stride = config.stride
        self.d_model = config.d_model
        self.task_name = config.task_name
        patch_num = int((config.seq_len - self.patch_len)/self.stride + 1)
        self.patch_num = patch_num
        self.W_pos_embed = nn.Parameter(torch.randn(patch_num,config.d_model) *1e-2)
        self.model_token_number = 0
        
        if self.model_token_number > 0:
            self.model_token = nn.Parameter(torch.randn(config.enc_in,self.model_token_number,config.d_model)*1e-2)
        
        
        self.total_token_number = (self.patch_num  + self.model_token_number + 1)
        config.total_token_number = self.total_token_number
             
        self.W_input_projection = nn.Linear(self.patch_len, config.d_model)  
        self.input_dropout  = nn.Dropout(config.dropout)  # dropout 层，用于在训练过程中随机地将一部分输入置为零，以防止过拟合
        
                
        self.use_statistic = config.use_statistic
        self.W_statistic = nn.Linear(2,config.d_model) 
        self.cls = nn.Parameter(torch.randn(1,config.d_model)*1e-2) #生成一个形状为 [1, config.d_model] 的张量
        
        
        
        if config.task_name == 'long_term_forecast' or config.task_name == 'short_term_forecast':
            self.W_out = nn.Linear((patch_num+1+self.model_token_number)*config.d_model, config.pred_len) 
        elif config.task_name == 'imputation' or config.task_name == 'anomaly_detection':
            self.W_out = nn.Linear((patch_num+1+self.model_token_number)*config.d_model, config.seq_len) 
        elif config.task_name == 'classification':
            self.W_out = nn.Linear(config.d_model*config.enc_in, config.num_class)

     
        
        
        self.Attentions_over_token = nn.ModuleList([Attenion(config) for i in range(config.e_layers)])
        self.Attentions_over_channel = nn.ModuleList([Attenion(config,over_hidden = True) for i in range(config.e_layers)])
        self.Attentions_mlp = nn.ModuleList([nn.Linear(config.d_model,config.d_model)  for i in range(config.e_layers)])
        self.Attentions_dropout = nn.ModuleList([nn.Dropout(config.dropout)  for i in range(config.e_layers)])
        self.Attentions_norm = nn.ModuleList([nn.Sequential(Transpose(1,2), nn.BatchNorm1d(config.d_model,momentum = config.momentum), Transpose(1,2)) for i in range(config.e_layers)])       
            
        # self.robTF = RobTF(thetas_dim1 = 512, thetas_dim2 = 512, seq_len = config.seq_len, pred_len = config.pred_len)
        # self.decomp = series_decomp(kernel_size=config.moving_avg)  

    def forward(self, z,*args, **kwargs):  

        b,c,s = z.shape

        # seasonal_init, trend_init = self.decomp(z)  #add
        # z=seasonal_init #add
        # z1=trend_init #add
        

        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast' or self.task_name == 'anomaly_detection':
            z_mean = torch.mean(z,dim = (-1),keepdims = True)
            z_std = torch.std(z,dim = (-1),keepdims = True)
            z =  (z - z_mean)/(z_std + 1e-4)
            # z1_mean = torch.mean(z1,dim = (-1),keepdims = True) #add
            # z1_std = torch.std(z1,dim = (-1),keepdims = True) #add
            # z1 =  (z1 - z1_mean)/(z1_std + 1e-4) #add
        
        elif self.task_name == 'imputation':     
            mask = kwargs['mask'].permute(0,2,1) 
            z_mean = torch.sum(z, dim=-1) / torch.sum(mask == 1, dim=-1)
            z_mean = z_mean.unsqueeze(-1)
            z = z - z_mean
            z = z.masked_fill(mask == 0, 0)
            z_std = torch.sqrt(torch.sum(z * z, dim=-1) /
                           torch.sum(mask == 1, dim=-1) + 1e-5)
            z_std = z_std.unsqueeze(-1)
            z /= z_std + 1e-4

       
        zcube = z.unfold(dimension=-1, size=self.patch_len, step=self.stride)           #[32, 7, 96]->[32, 7, 11, 16] stride=8
        z_embed = self.input_dropout(self.W_input_projection(zcube))+ self.W_pos_embed  #将每个块映射到模型的嵌入维度，并加上位置嵌入 [32, 7, 11, 512]
                                                                                        #W_pos_embed 通过广播变为 [1, 1, 11, 512]，然后与 [32, 7, 11, 512] 相加，最终得到的 z_embed 形状是 [32, 7, 11, 512]。         
                                                                                        #广播用于使不同形状的张量能够进行算术运算。W_pos_embed 的形状为 [11, 512]，当它与 self.W_input_projection(zcube) 的输出形状 [32, 7, 11, 512] 相加时，会发生广播。
        if self.use_statistic:
            
            z_stat = torch.cat((z_mean,z_std),dim = -1)
            if z_stat.shape[-2]>1:
                z_stat = (z_stat - torch.mean(z_stat,dim =-2,keepdims = True))/( torch.std(z_stat,dim =-2,keepdims = True)+1e-4)
            z_stat = self.W_statistic(z_stat)
            z_embed = torch.cat((z_stat.unsqueeze(-2),z_embed),dim = -2) 
        else:
            cls_token = self.cls.repeat(z_embed.shape[0],z_embed.shape[1],1,1) #torch.Size([32, 7, 1, 512])
            z_embed = torch.cat((cls_token,z_embed),dim = -2) #（cls_token）添加到嵌入张量 z_embed 中，以便于模型进行分类任务。torch.Size([32, 7, 12, 512])
                                                                # cls_token 添加到 z_embed 的前面
        inputs = z_embed  #[32, 7, 12, 512]
        b,c,t,h = inputs.shape   #b（批次大小）、c（特征数量）、t（时间步数或标记数量）、h（特征维度）
        for a_2,a_1,mlp,drop,norm in zip(self.Attentions_over_token, self.Attentions_over_channel,self.Attentions_mlp ,self.Attentions_dropout,self.Attentions_norm ):
            output_1 = a_1(inputs.permute(0,2,1,3)).permute(0,2,1,3)  #torch.Size([32, 7, 12, 512])
            output_2 = a_2(output_1)   #torch.Size([32, 7, 12, 512])
            outputs = drop(mlp(output_1+output_2))+inputs
            outputs = norm(outputs.reshape(b*c,t,-1)).reshape(b,c,t,-1) 
            inputs = outputs  #[32, 7, 12, 512]

        # predict_init=z1 #add
        # trend_output = self.robTF(predict_init) #add
        # z1=trend_output #add
        

        if self.task_name != 'classification':
            z_out = self.W_out(outputs.reshape(b,c,-1))   #[32, 7, 12, 512]->[32, 7, 12*512]->W_out[32, 7, 96]
            z = z_out *(z_std+1e-4)  + z_mean 
            # z1_out = self.W_out(outputs.reshape(b,c,-1))   #add
            # z1 = z1_out *(z1_std+1e-4)  + z1_mean  #add
        else:
            z = self.W_out(torch.mean(outputs[:,:,:,:],dim = -2).reshape(b,-1))
        # return z+z1  #add
        return z

    

class Attenion(nn.Module):
    def __init__(self,config, over_hidden = False,trianable_smooth = False,untoken = False, *args, **kwargs):
        super().__init__()

        
        self.over_hidden = over_hidden
        self.untoken = untoken
        self.n_heads = config.n_heads
        self.c_in = config.enc_in
        self.qkv = nn.Linear(config.d_model, config.d_model * 3, bias=True)
        
        
    
        self.attn_dropout = nn.Dropout(config.dropout)
        self.head_dim = config.d_model // config.n_heads
        

        self.dropout_mlp = nn.Dropout(config.dropout)
        self.mlp = nn.Linear( config.d_model,  config.d_model)
        
        

        self.norm_post1  = nn.Sequential(Transpose(1,2), nn.BatchNorm1d(config.d_model,momentum = config.momentum), Transpose(1,2))
        self.norm_post2  = nn.Sequential(Transpose(1,2), nn.BatchNorm1d(config.d_model,momentum = config.momentum), Transpose(1,2))
        
        self.norm_attn = nn.Sequential(Transpose(1,2), nn.BatchNorm1d(config.d_model,momentum = config.momentum), Transpose(1,2))
        
        
        self.dp_rank = config.dp_rank
        self.dp_k = nn.Linear(self.head_dim, self.dp_rank)
        self.dp_v = nn.Linear(self.head_dim, self.dp_rank)

        #通道注意力
        # self.attn = SEAttention(channel=config.d_model, reduction=8) 
        # self.attn = SKAttention(channel=config.d_model,reduction=8)
            # (B,C,H,W)  注意: 因为在模型中需要将HW和C拼接起来,所在在输入到模型的时候,最好把通道C和HW做个降维(池化、下采样均可),然后在输入到模型中去,输出之后再恢复shape就可以了！
        # self.attn=CBAMBlock(channel=config.d_model,reduction=16,kernel_size=7)
        # self.attn=ECAAttention(kernel_size=3)
        # self.attn1 = Multi_GTU(num_of_timesteps=config.total_token_number, in_channels=config.d_model, time_strides=1, kernel_size=[3,5,7], pool=True)  #1
        # self.attn1 = Multi_GTU(num_of_timesteps=config.d_model, in_channels=self.c_in, time_strides=1, kernel_size=[3,5,7], pool=True)  #2 gd
        # self.attn1= CrossRegionAttntion(dim=config.d_model, num_heads=config.n_heads, region_size=5)  #1
        # self.attn1= CrossRegionAttntion(dim=self.c_in, num_heads=self.c_in, region_size=5)   #2 gd
        # self.attn1 = AxialAttention(in_planes=config.d_model, out_planes=config.d_model, groups=1, kernel_size=self.c_in, stride=1, bias=False, width=False)
            #in_planes、out_planes和channel一致， kernel_size和h,w一致
        
        
        self.ff_1 = nn.Sequential(nn.Linear(config.d_model, config.d_ff, bias=True),
                        nn.GELU(),
                        nn.Dropout(config.dropout),
                        nn.Linear(config.d_ff, config.d_model, bias=True)
                       )
        
        self.ff_2= nn.Sequential(nn.Linear(config.d_model, config.d_ff, bias=True),
                        nn.GELU(),
                        nn.Dropout(config.dropout),
                        nn.Linear(config.d_ff, config.d_model, bias=True)
                                )     
        self.merge_size = config.merge_size

        ema_size = max(config.enc_in,config.total_token_number,config.dp_rank)
        ema_matrix = torch.zeros((ema_size,ema_size))
        alpha = config.alpha
        ema_matrix[0][0] = 1
        for i in range(1,config.total_token_number):
            for j in range(i):
                ema_matrix[i][j] =  ema_matrix[i-1][j]*(1-alpha)
            ema_matrix[i][i] = alpha
        self.register_buffer('ema_matrix',ema_matrix)
 
           

       
    def ema(self,src):
        return torch.einsum('bnhad,ga ->bnhgd',src,self.ema_matrix[:src.shape[-2],:src.shape[-2]]) #[32, 12, 8,7,64] [7,7]-->[32,12,7,7,64]
                                                                                                   #[32, 7, 8,12,64] [12,12]-->[32,7,12,12,64]
        
    def ema_trianable(self,src):
        alpha = F.sigmoid(self.alpha)
        
        weights = alpha * (1 - alpha) ** self.arange[-src.shape[-2]:]
 

        w_f = torch.fft.rfft(weights,n = src.shape[-2]*2)
        src_f = torch.fft.rfft(src.float(),dim = -2,n = src.shape[-2]*2)    
        src_f = (src_f.permute(0,1,2,4,3)*w_f)
        src1 =torch.fft.irfft(src_f.float(),dim = -1,n=src.shape[-2]*2)[...,:src.shape[-2]].permute(0,1,2,4,3)#.half()
        return src1



    def dynamic_projection(self,src,mlp):
        src_dp = mlp(src) #[32, 12, 8, 7, 64]-->[32, 12, 8, 7, 8] 降维的过程，为了减少后续计算的复杂度和内存消耗。
        src_dp = F.softmax(src_dp,dim = -1)  #最后一维（即 8）进行归一化
        src_dp = torch.einsum('bnhef,bnhec -> bnhcf',src,src_dp) #[32, 12, 8, 7,64][32, 12, 8, 7, 8]-->[32, 12, 8, 8, 64]将每个时间步的原始特征（src）加权平均，权重由 src_dp 提供。
        return src_dp
        

        
        
    def forward(self, src, *args,**kwargs):


        B,nvars, H, C, = src.shape #[32,12,7,512]  a2 [32,7,12,512]
        


        
        
        qkv = self.qkv(src).reshape(B,nvars, H, 3, self.n_heads, C // self.n_heads).permute(3, 0, 1,4, 2, 5)
   

        q, k, v = qkv[0], qkv[1], qkv[2]  #torch.Size([32, 12, 8, 7, 64]) a2 [32, 7, 8, 12, 64])
    
        if not self.over_hidden:  #over token attn 2
        
            attn_score_along_token = torch.einsum('bnhed,bnhfd->bnhef', self.ema(q), self.ema(k))/ self.head_dim ** -0.5   #[32,7,8,12,64][32,7,8,12,64]-->[32,7,8,12,12]
            attn_along_token = self.attn_dropout(F.softmax(attn_score_along_token, dim=-1) )
            output_along_token = torch.einsum('bnhef,bnhfd->bnhed', attn_along_token, v)   #[32,7,8,12,12][32,7,8,12,64]-->[32,7,8,12,64]

            # (B,C,N,T)  N:序列的个数  T:序列的长度  更改输入的时候记得把64行对应的时间步和输入通道数改整
            # B C H W
            # X = src.permute(0,3,1,2)  #[32,7,12,512]->[32,512,7,12] #1
            # X = src  #2 gd
            # output = self.attn1(X) #[32,7,12,512] B=32 C=7 N=12 T=512
            # output_along_token=output.reshape(B,nvars, self.n_heads,H, C // self.n_heads)

            # 输入张量的形状为 (batch_size, sequence_length, embedding_dim)  CR-MSA
            # X=src.reshape(B*nvars,H,C)  #1
            # X=src.reshape(B*C,H,nvars,)  #2 gd
            # output = self.attn1(X)
            # output_along_token=output.reshape(B,nvars, self.n_heads,H, C // self.n_heads)


            
        else:  #over channel attn 1

            v_dp,k_dp = self.dynamic_projection(v,self.dp_v) , self.dynamic_projection(k,self.dp_k)  #K,V进行变换转换为低维表示64-->8，[32, 12, 8, 8, 64]
            attn_score_along_token = torch.einsum('bnhed,bnhfd->bnhef', self.ema(q), self.ema(k_dp))/ self.head_dim ** -0.5  #[32,12,8,7,64][32,12,8,8,64]-->[32,12,8,7,8]
            attn_along_token = self.attn_dropout(F.softmax(attn_score_along_token, dim=-1) ) #[32, 12, 8, 7, 8]
            output_along_token = torch.einsum('bnhef,bnhfd->bnhed', attn_along_token, v_dp)#[32,12,8,7,8][32,12,8,8,64]--> [32, 12, 8, 7, 64]

            # x1=src
            # x1 = x1.permute(0,3,1,2)  #[32,12,7,512]->[32,512,12,7]

            # (B,C,H,W)
            # input1 = q.reshape(B,self.n_heads*(C // self.n_heads),nvars,H)  #[32,512,12,7]
            # input2 = k.reshape(B, self.n_heads*(C // self.n_heads),nvars,H)
            # input3 = v.reshape(B, self.n_heads*(C // self.n_heads),nvars,H)
            # # 定义通道注意力SEAttn
            # output = self.attn(input1,input2,input3).permute(0,2,3,1)#[32, 512, 12, 7]->[32,12,7,512] 
            # output_along_token=output.reshape(B,nvars, self.n_heads,H, C // self.n_heads) #[32, 12, 8, 7, 64]

            # # 定义通道注意力SKAttn\CBAM\ECANet
            # output=self.attn(x1).permute(0,2,3,1)
            # output_along_token=output.reshape(B,nvars, self.n_heads,H, C // self.n_heads)
            
        
        attn_score_along_hidden = torch.einsum('bnhae,bnhaf->bnhef', q,k)/ q.shape[-2] ** -0.5 #torch.Size([32, 12, 8, 64, 64])
        attn_along_hidden = self.attn_dropout(F.softmax(attn_score_along_hidden, dim=-1) )    
        output_along_hidden = torch.einsum('bnhef,bnhaf->bnhae', attn_along_hidden, v) #torch.Size([32, 12, 8, 7, 64])



        merge_size = self.merge_size
        if not self.untoken:
            output1 = rearrange(output_along_token.reshape(B*nvars,-1,self.head_dim),  #[384, 7, 512] a2[224, 12, 512]
                            'bn (hl1 hl2 hl3) d -> bn  hl2 (hl3 hl1) d', 
                            hl1 = self.n_heads//merge_size, hl2 = output_along_token.shape[-2] ,hl3 = merge_size
                            ).reshape(B*nvars,-1,self.head_dim*self.n_heads)
        
        
            output2 = rearrange(output_along_hidden.reshape(B*nvars,-1,self.head_dim),  #torch.Size([384, 7, 512])  a2[224, 12, 512]
                            'bn (hl1 hl2 hl3) d -> bn  hl2 (hl3 hl1) d', 
                            hl1 = self.n_heads//merge_size, hl2 = output_along_token.shape[-2] ,hl3 = merge_size
                            ).reshape(B*nvars,-1,self.head_dim*self.n_heads)
        

        output1 = self.norm_post1(output1)  #[384, 7, 512]
        output1 = output1.reshape(B,nvars, -1, self.n_heads * self.head_dim)  #[32,12, 7, 512]  a2[32,7,12, 512]
        output2 = self.norm_post2(output2)
        output2 = output2.reshape(B,nvars, -1, self.n_heads * self.head_dim)





        src2 =  self.ff_1(output1)+self.ff_2(output2)  #[32,12, 7, 512]  a2[32,7,12, 512]
        
        
        src = src + src2
        src = src.reshape(B*nvars, -1, self.n_heads * self.head_dim)  #[32, 12, 7, 512]->[384, 7, 512]
        src = self.norm_attn(src)

        src = src.reshape(B,nvars, -1, self.n_heads * self.head_dim)   #[384, 7, 512]->[32, 12, 7, 512]
        return src