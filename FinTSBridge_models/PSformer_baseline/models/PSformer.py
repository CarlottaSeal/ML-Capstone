from torch import nn
from einops import rearrange
import torch.nn.functional as F
from utils.positional_embedding import PositionalEmbedding




class ps_block(nn.Module):

    def __init__(self, input_dim, output_dim, hidden_dim=512):
        super(ps_block, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.gelu = nn.GELU()
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):

        x = self.fc1(x)  # x-> hidden_dim 
        res = self.gelu(x)
        res = self.fc2(res)  # hidden_dim
        x = x + res
        x = self.fc3(x) # 

        return x


class ps_encoder(nn.Module):  
    #
    def __init__(self, configs):
        super(ps_encoder, self).__init__()
        self.num_seg = configs.num_seg
        self.parameter_share = configs.parameter_share
        self.use_pos_emb = configs.use_pos_emb
        if self.parameter_share:
            self.ps_block = ps_block(self.num_seg, self.num_seg, hidden_dim=self.num_seg) 
        else:
            self.ps_block1 = ps_block(self.num_seg, self.num_seg, hidden_dim=self.num_seg) 
            self.ps_block2 = ps_block(self.num_seg, self.num_seg, hidden_dim=self.num_seg) 
            self.ps_block3 = ps_block(self.num_seg, self.num_seg, hidden_dim=self.num_seg) 
            self.ps_block4 = ps_block(self.num_seg, self.num_seg, hidden_dim=self.num_seg) 
            self.ps_block5 = ps_block(self.num_seg, self.num_seg, hidden_dim=self.num_seg) 
            self.ps_block6 = ps_block(self.num_seg, self.num_seg, hidden_dim=self.num_seg) 
            self.ps_block7 = ps_block(self.num_seg, self.num_seg, hidden_dim=self.num_seg) 

        
        self.gelu = nn.GELU()
        self.relu = nn.ReLU()
        self.qkv_1 = None
        self.cal_att_map = configs.cal_att_map

        if self.use_pos_emb:
            self.num_channels = configs.num_channels
            self.embed_dim = configs.seq_len//configs.num_seg
            self.pos_emb = PositionalEmbedding(self.num_seg, self.num_channels*self.embed_dim)


    def forward(self, x):

        B, M, L = x.shape # B M L

        x = rearrange(x, 'b m (n p) -> b (p m) n', b=B, m=M, n=self.num_seg) # B M (N P) -> B (P M) N
        # x = rearrange(x, 'b m (n p) -> b n (p m)', b=B, m=M, n=self.num_seg)
        if self.use_pos_emb:
            pos_emb = self.pos_emb(x) # L M
            x = x + pos_emb # B L M
        # x = rearrange(x, 'b n (p m) -> b (p m) n', b=B, m=M, n=self.num_seg)

        if self.parameter_share:
            qkv = self.ps_block(x)
        else:
            q = self.ps_block1(x)  # B (P M) N 
            k = self.ps_block2(x)  # B (P M) N 
            v = self.ps_block3(x)

        if self.cal_att_map:
            self.q_1 = q.clone() #计算att map
            self.k_1 = k.clone()

        if self.parameter_share:
            att_x = F.scaled_dot_product_attention(qkv, qkv, qkv) # B (P M) N
        else:
            att_x = F.scaled_dot_product_attention(q, k, v) 
        
        att_x = self.relu(att_x) 

        if self.parameter_share:
            qkv = self.ps_block(att_x)
        else:
            q = self.ps_block4(att_x)  # # B (P M) N
            k = self.ps_block5(att_x)  # B (P M) N
            v = self.ps_block6(att_x)

        if self.cal_att_map:
            self.qkv_2 = qkv.clone()
            self.q_2 = q.clone() #计算att map
            self.k_2 = k.clone()

        if self.parameter_share:
            att_x = F.scaled_dot_product_attention(qkv, qkv, qkv)  
        else:
            att_x = F.scaled_dot_product_attention(q, k, v) 

        # self.att_x = att_x.clone()
        x = x + att_x  # B (P M) N

        if self.parameter_share:
            x = self.ps_block(x)  # B (P M) N
        else:
            x = self.ps_block7(x)

        x = rearrange(x, 'b (p m) n -> b m (n p)', b=B, m=M, n=self.num_seg)

        return x





class Model(nn.Module):
    
    def __init__(self, configs): 
        super().__init__()
        self.hid_dim = configs.seq_len
        self.pred_horizon = configs.pred_len
        self.seq_len = configs.seq_len
        self.num_seg = configs.num_seg  # segment number
        self.num_encoder = configs.num_encoder  # encoder number
        self.hidden_dim = configs.hidden_dim # psblock hidden dim
        self.cl_ps = configs.cl_ps

        if self.cl_ps:
            self.ps_encoder = ps_encoder(configs)
        else:
            self.ps_encoder = nn.ModuleList([
            ps_encoder(configs)
            for _ in range(self.num_encoder)
            ])
            
        self.dropout = nn.Dropout(configs.dropout)
        self.norm_window = configs.norm_window
        self.fc = nn.Linear(self.seq_len, self.pred_horizon)
        self.gelu = nn.GELU()

        # self.pos_emb = configs.pos_emb


        
    def forward(self, x, batch_x_mark, dec_inp, batch_y_mark, flatten_output=True):


        # Rearrange: B L M -> B M L
        x = rearrange(x, 'b l m -> b m l') 

        # RevIN Normalization
        mean = x[:,:,-self.norm_window:].mean(dim=2, keepdim=True) # B M L
        std = x[:,:,-self.norm_window:].std(dim=2, keepdim=True) # B M L
        x = (x - mean) / (std+1e-3) # B M L


        if self.cl_ps:
            #cross encoder ps
            for _ in range(self.num_encoder):
                res = self.ps_encoder(x)
                x = x + res

        else:
            # Encoder
            if len(self.ps_encoder) == 1:
                x = self.ps_encoder[0](x) 
            else:
                for layer in self.ps_encoder:
                    res = layer(x) 
                    x = x + res  # B M L

        x = self.dropout(x)

        # Mapping
        x = self.fc(x)  # B M L -> B M F

        # RevIN Denormalization
        x = x * (std+1e-3) + mean # B M F

        # Rearrange: B M F -> B F M
        x = rearrange(x, 'b m f -> b f m') 

        return x




