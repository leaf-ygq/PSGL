import torch
import torch.nn as nn
import torch_geometric as tg
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, SAGPooling, TopKPooling, ASAPooling
from torch_geometric.utils import add_self_loops, degree
from torch.nn import init
import pdb
from args import *
import numpy as np
import os

from torch_geometric.nn import GCNConv, SAGEConv, GATConv, ChebConv, GraphConv
####################################################################
args = make_args()
device = torch.device('cuda:'+str(args.cuda) if args.gpu else 'cpu')

torch.manual_seed(2020) # seed for reproducible numbers


class PGNN_layer(nn.Module):
    def __init__(self, input_dim, output_dim,dist_trainable=True):
        super(PGNN_layer, self).__init__()
        self.input_dim = input_dim
        self.dist_trainable = dist_trainable

        if self.dist_trainable:
            self.dist_compute = Nonlinear(1, output_dim, 1)

        self.linear_hidden = nn.Linear(input_dim*2, output_dim)
        self.linear_out_position = nn.Linear(output_dim,1)
        self.act = nn.ReLU()

        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data = init.xavier_uniform_(m.weight.data, gain=nn.init.calculate_gain('relu'))
                if m.bias is not None:
                    m.bias.data = init.constant_(m.bias.data, 0.0)

    def forward(self, feature, dists_max, dists_argmax, score):
        if self.dist_trainable:
            dists_max = self.dist_compute(dists_max.unsqueeze(-1)).squeeze()

        # print('score:', score.unsqueeze(0).shape)
        # print(score)
        # print(dists_max[:2])
        
        dists_max = dists_max * score.unsqueeze(0)
        # print('dists_max:', dists_max.shape) #[400, 100]

        # print(dists_max[:2])
        # print(dists_max.shape)
        # pdb.set_trace()

        # print('feature: ', feature.shape) #[400, 32]

        subset_features = feature[dists_argmax.flatten(), :]

        # print('subset_features:', subset_features.shape) #[40000, 32]

        subset_features = subset_features.reshape((dists_argmax.shape[0], dists_argmax.shape[1],
                                                   feature.shape[1]))
        # print('subset_features:', subset_features.shape) #[400, 100, 32]
        messages = subset_features * dists_max.unsqueeze(-1) 
        # print('messages:',messages.shape) #[400, 100, 32]
        
        self_feature = feature.unsqueeze(1).repeat(1, dists_max.shape[1], 1)
        # print('self_feature:', self_feature.shape) #[400, 100, 32]

        
        # pdb.set_trace()
        
        messages = torch.cat((messages, self_feature), dim=-1)
        # print(messages.shape)
        messages = self.linear_hidden(messages).squeeze()
        messages = self.act(messages) # n*m*d
        # print(messages.shape)
        out_position = self.linear_out_position(messages).squeeze(-1)  # n*m_out
        out_structure = torch.mean(messages, dim=1)  # n*d

        # print(out_position.shape)
        # print(out_structure.shape)
        
        # pdb.set_trace()

        return out_position, out_structure

class PGNN_layer_original(nn.Module):
    def __init__(self, input_dim, output_dim,dist_trainable=True):
        super(PGNN_layer, self).__init__()
        self.input_dim = input_dim
        self.dist_trainable = dist_trainable

        if self.dist_trainable:
            self.dist_compute = Nonlinear(1, output_dim, 1)

        self.linear_hidden = nn.Linear(input_dim*2, output_dim)
        self.linear_out_position = nn.Linear(output_dim,1)
        self.act = nn.ReLU()

        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data = init.xavier_uniform_(m.weight.data, gain=nn.init.calculate_gain('relu'))
                if m.bias is not None:
                    m.bias.data = init.constant_(m.bias.data, 0.0)

    def forward(self, feature, dists_max, dists_argmax):
        if self.dist_trainable:
            dists_max = self.dist_compute(dists_max.unsqueeze(-1)).squeeze()

        subset_features = feature[dists_argmax.flatten(), :]
        subset_features = subset_features.reshape((dists_argmax.shape[0], dists_argmax.shape[1],
                                                   feature.shape[1]))
        messages = subset_features * dists_max.unsqueeze(-1)

        self_feature = feature.unsqueeze(1).repeat(1, dists_max.shape[1], 1)
        messages = torch.cat((messages, self_feature), dim=-1)

        messages = self.linear_hidden(messages).squeeze()
        messages = self.act(messages) # n*m*d

        out_position = self.linear_out_position(messages).squeeze(-1)  # n*m_out
        out_structure = torch.mean(messages, dim=1)  # n*d

        return out_position, out_structure



class Attention(nn.Module):
    def __init__(self, in_size, hidden_size=16):
        super(Attention, self).__init__()
        # print('in, hidden', in_size, hidden_size)
        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )

    def forward(self, z):
        # print('z:', z.shape)
        w = self.project(z)
        # print('w:', w.shape)
        # print(w[0])
        beta = torch.softmax(w, dim=1)
        # print('beta:', beta.shape)
        # print(beta[0])
        # print('alpha:', (beta * z).sum(1).shape)
        # pdb.set_trace()
        return (beta * z).sum(1), beta

class P_A_GIN(torch.nn.Module):
    
    def __init__(self, num_class, device, final_emb_size, input_dim, feature_dim, hidden_dim, output_dim,
                 feature_pre=True, layer_num=2, dropout=True, **kwargs):
        super(P_A_GIN, self).__init__()
        self.feature_pre = feature_pre
        self.layer_num = layer_num 
        self.dropout = dropout
        
        self.attention = Attention(hidden_dim)

        if feature_pre:
            self.linear_pre = nn.Linear(input_dim, feature_dim)
            self.conv_first_nn = nn.Linear(feature_dim, hidden_dim)
            self.conv_gcn_first = tg.nn.GINConv(self.conv_first_nn)

            self.conv_first = PGNN_layer(feature_dim, hidden_dim)
            # self.conv_first = GraphReach_Layer(feature_dim, hidden_dim)
        else:
            self.conv_first = PGNN_layer(input_dim, hidden_dim)
            # self.conv_first = GraphReach_Layer(input_dim, hidden_dim)

            self.conv_first_nn = nn.Linear(input_dim, hidden_dim)
            self.conv_gcn_first = tg.nn.GINConv(self.conv_first_nn)


        self.conv_hidden = nn.ModuleList([PGNN_layer(hidden_dim, hidden_dim) for i in range(layer_num - 2)])
        self.conv_out = PGNN_layer(hidden_dim, output_dim)
        # self.conv_out = GraphReach_Layer(hidden_dim, output_dim)
        # self.conv_hidden = nn.ModuleList([GraphReach_Layer(hidden_dim, hidden_dim) for i in range(layer_num - 2)])
        

        self.conv_hidden_nn = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for i in range(layer_num - 2)])
        self.conv_gcn_hidden = nn.ModuleList([tg.nn.GINConv(self.conv_hidden_nn[i]) for i in range(layer_num - 2)])

        self.conv_out_nn = nn.Linear(hidden_dim, output_dim)
        self.conv_gcn_out = tg.nn.GINConv(self.conv_out_nn)

        # self.pool = TopKPooling(hidden_dim, ratio=0.25)
        self.pool = SAGPooling(hidden_dim, ratio=0.5, GNN=GCNConv)
        # self.pool = SAGPooling(hidden_dim)

        self.Conv1 = GCNConv(feature_dim, hidden_dim)

        # self.Conv1 = GCNConv(input_dim, feature_dim)

        self.device = device

        self.lin2 = nn.Linear(final_emb_size + output_dim, num_class)

    def get_our_dist(self, anchorset_id, dists, device):

        # anchor to node similarity
        dist_max1 = torch.zeros((dists.shape[0],len(anchorset_id))).to(device)
        dist_argmax1 = torch.zeros((dists.shape[0],len(anchorset_id))).long().to(device)

        # node to anchor similarity

        dist = torch.from_numpy(dists).float()


        for i in range(len(anchorset_id)):
            temp_id = anchorset_id[i]
            dist_temp = dist[:, temp_id]
            
            dist_max_temp, dist_argmax_temp = torch.max(dist_temp, dim=-1)
            
            dist_max1[:,i]=dist_max_temp
            dist_argmax1[:,i]=torch.tensor(anchorset_id[i])

        return dist_max1, dist_argmax1

    def forward(self, data, dists):
        # x = data.x #我删掉的
        x, edge_index = data.x, data.edge_index
        # x, edge_index = data.x, data.all_edges
        # print(x.shape)

        if self.feature_pre:
            x = self.linear_pre(x)

        x2 = self.Conv1(x, edge_index) #这一步考虑是否删除
        # x = self.Conv1(x, edge_index)

        _, _, _, _, perm, score = self.pool(x2, edge_index, None, None)   

        score = torch.sigmoid(score)
        # print(score)
        # pdb.set_trace()

        m = int(np.log2(x.shape[0]))
        anchor_num = m * m

        # anchor_num = 128
        # print(anchor_num)

        if len(perm) >= anchor_num:
            tmp = perm.cpu()[:anchor_num]
            anchorset_id = []
            for v in tmp:
                anchorset_id.append(np.array([v]))
            score = score[:anchor_num]
        else:
            # np.random.seed(10)
            anchorset_id = []
            index = np.arange(len(perm))
            tmp = np.random.choice(index, anchor_num)
            # score = torch.cat((score, score[tmp]), axis=0)
            score = score[tmp]
            for v in tmp:
                anchorset_id.append(np.array([v]))
       
        data.dists_max, data.dists_argmax = self.get_our_dist(anchorset_id, dists, self.device)

        # print(data.dists_max)
        # print(data.dists_argmax)
        x_position, x_ = self.conv_first(x, data.dists_max, data.dists_argmax, score)
        # print(x_position.shape)
        # pdb.set_trace()

        x_gcn = self.conv_gcn_first(x, edge_index)
        x_gcn = F.relu(x_gcn)

        x_ = F.relu(x_) #optional

        add = torch.stack([x_, x_gcn], dim = 1)
        add, att = self.attention(add)

        if self.dropout:
            add = F.dropout(add, training=self.training)

        x_gcn = add 
        x_ = add
        # if self.dropout:
        #     x_ = F.dropout(x_, training=self.training)
        #     x_gcn = F.dropout(x_gcn, training=self.training)
        
            
        for i in range(self.layer_num - 2):
            # _, x_ = self.conv_hidden[i](x_, data.dists_max, data.dists_argmax)
            
            x_position, x_ = self.conv_hidden[i](x_, data.dists_max, data.dists_argmax, score)
            x_gcn = self.conv_gcn_hidden[i](x_gcn, edge_index)
            x_gcn = F.relu(x_gcn)

            x_ = F.relu(x_) #optional
               
            
            add = torch.stack([x_, x_gcn], dim = 1)
            add, att = self.attention(add)
            if self.dropout:
                add = F.dropout(add, training=self.training)
            x_ = add
            x_gcn = add

        # x_position, x_ = self.conv_out(x_, data.dists_max, data.dists_argmax)
        x_position, x_ = self.conv_out(x_, data.dists_max, data.dists_argmax, score)
        x_gcn = self.conv_gcn_out(x_gcn, edge_index)

        add = torch.cat((x_position, x_gcn), dim=-1)
        x_position = F.normalize(add, p=2, dim=-1)

        # x_position = F.normalize(x_position, p=2, dim=-1)

        x_position = F.dropout(x_position, training=self.training)
        x_position = self.lin2(x_position)

        return F.log_softmax(x_position, dim=1)



class P_A_GCN(torch.nn.Module):
    
    def __init__(self, num_class, device, final_emb_size, input_dim, feature_dim, hidden_dim, output_dim,
                 feature_pre=True, layer_num=2, dropout=True, **kwargs):
        super(P_A_GCN, self).__init__()
        self.feature_pre = feature_pre
        self.layer_num = layer_num 
        self.dropout = dropout
        
        self.attention = Attention(hidden_dim)

        if feature_pre:
            self.linear_pre = nn.Linear(input_dim, feature_dim)
            # self.conv_first_nn = nn.Linear(feature_dim, hidden_dim)
            # self.conv_gcn_first = tg.nn.GINConv(self.conv_first_nn)

            self.conv_gcn_first = tg.nn.GCNConv(feature_dim, hidden_dim)

            self.conv_first = PGNN_layer(feature_dim, hidden_dim)
            # self.conv_first = GraphReach_Layer(feature_dim, hidden_dim)
        else:
            self.conv_first = PGNN_layer(input_dim, hidden_dim)
            # self.conv_first = GraphReach_Layer(input_dim, hidden_dim)

            # self.conv_first_nn = nn.Linear(input_dim, hidden_dim)
            # self.conv_gcn_first = tg.nn.GINConv(self.conv_first_nn)
            self.conv_gcn_first = tg.nn.GCNConv(input_dim, hidden_dim)


        self.conv_hidden = nn.ModuleList([PGNN_layer(hidden_dim, hidden_dim) for i in range(layer_num - 2)])
        self.conv_out = PGNN_layer(hidden_dim, output_dim)
        # self.conv_out = GraphReach_Layer(hidden_dim, output_dim)
        # self.conv_hidden = nn.ModuleList([GraphReach_Layer(hidden_dim, hidden_dim) for i in range(layer_num - 2)])
        

        # self.conv_hidden_nn = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for i in range(layer_num - 2)])
        # self.conv_gcn_hidden = nn.ModuleList([tg.nn.GINConv(self.conv_hidden_nn[i]) for i in range(layer_num - 2)])

        self.conv_gcn_hidden = nn.ModuleList([tg.nn.GCNConv(hidden_dim, hidden_dim) for i in range(layer_num - 2)])
        
        self.conv_gcn_out = tg.nn.GCNConv(hidden_dim, output_dim)

        # self.conv_out_nn = nn.Linear(hidden_dim, output_dim)
        # self.conv_gcn_out = tg.nn.GINConv(self.conv_out_nn)

        # self.pool = TopKPooling(hidden_dim, ratio=0.25)
        self.pool = SAGPooling(hidden_dim, ratio=0.5, GNN=GCNConv)
        # self.pool = SAGPooling(hidden_dim)

        self.Conv1 = GCNConv(feature_dim, hidden_dim)

        # self.Conv1 = GCNConv(input_dim, feature_dim)

        self.device = device

        self.lin2 = nn.Linear(final_emb_size + output_dim, num_class)

    def get_our_dist(self, anchorset_id, dists, device):

        # anchor to node similarity
        dist_max1 = torch.zeros((dists.shape[0],len(anchorset_id))).to(device)
        dist_argmax1 = torch.zeros((dists.shape[0],len(anchorset_id))).long().to(device)

        # node to anchor similarity

        dist = torch.from_numpy(dists).float()


        for i in range(len(anchorset_id)):
            temp_id = anchorset_id[i]
            dist_temp = dist[:, temp_id]
            
            dist_max_temp, dist_argmax_temp = torch.max(dist_temp, dim=-1)
            
            dist_max1[:,i]=dist_max_temp
            dist_argmax1[:,i]=torch.tensor(anchorset_id[i])

        return dist_max1, dist_argmax1

    def forward(self, data, dists):
        # x = data.x #我删掉的
        x, edge_index = data.x, data.edge_index
        # x, edge_index = data.x, data.all_edges
        # print(x.shape)

        if self.feature_pre:
            x = self.linear_pre(x)

        x2 = self.Conv1(x, edge_index) #这一步考虑是否删除
        # x = self.Conv1(x, edge_index)

        _, _, _, _, perm, score = self.pool(x2, edge_index, None, None)   

        score = torch.sigmoid(score)
        # print(score)
        # pdb.set_trace()

        m = int(np.log2(x.shape[0]))
        anchor_num = m * m

        # anchor_num = 128
        # print(anchor_num)

        if len(perm) >= anchor_num:
            tmp = perm.cpu()[:anchor_num]
            anchorset_id = []
            for v in tmp:
                anchorset_id.append(np.array([v]))
            score = score[:anchor_num]
        else:
            # np.random.seed(10)
            anchorset_id = []
            index = np.arange(len(perm))
            tmp = np.random.choice(index, anchor_num)
            # score = torch.cat((score, score[tmp]), axis=0)
            score = score[tmp]
            for v in tmp:
                anchorset_id.append(np.array([v]))
       
        data.dists_max, data.dists_argmax = self.get_our_dist(anchorset_id, dists, self.device)

        # print(data.dists_max)
        # print(data.dists_argmax)
        x_position, x_ = self.conv_first(x, data.dists_max, data.dists_argmax, score)
        # print(x_position.shape)
        # pdb.set_trace()

        x_gcn = self.conv_gcn_first(x, edge_index)
        x_gcn = F.relu(x_gcn)

        x_ = F.relu(x_) #optional

        add = torch.stack([x_, x_gcn], dim = 1)
        add, att = self.attention(add)

        if self.dropout:
            add = F.dropout(add, training=self.training)

        x_gcn = add 
        x_ = add
        # if self.dropout:
        #     x_ = F.dropout(x_, training=self.training)
        #     x_gcn = F.dropout(x_gcn, training=self.training)
        
            
        for i in range(self.layer_num - 2):
            # _, x_ = self.conv_hidden[i](x_, data.dists_max, data.dists_argmax)
            
            x_position, x_ = self.conv_hidden[i](x_, data.dists_max, data.dists_argmax, score)
            x_gcn = self.conv_gcn_hidden[i](x_gcn, edge_index)
            x_gcn = F.relu(x_gcn)

            x_ = F.relu(x_) #optional
               
            
            add = torch.stack([x_, x_gcn], dim = 1)
            add, att = self.attention(add)
            if self.dropout:
                add = F.dropout(add, training=self.training)
            x_ = add
            x_gcn = add

        # x_position, x_ = self.conv_out(x_, data.dists_max, data.dists_argmax)
        x_position, x_ = self.conv_out(x_, data.dists_max, data.dists_argmax, score)
        x_gcn = self.conv_gcn_out(x_gcn, edge_index)

        add = torch.cat((x_position, x_gcn), dim=-1)
        x_position = F.normalize(add, p=2, dim=-1)

        # x_position = F.normalize(x_position, p=2, dim=-1)

        x_position = F.dropout(x_position, training=self.training)
        x_position = self.lin2(x_position)

        return F.log_softmax(x_position, dim=1)

class GAT_Attention(nn.Module):
    def __init__(self,orignal_features, anchor_features, hidden_features, dropout=0.5, alpha=0.2, nheads=1):
        super(GAT_Attention, self).__init__()
        self.dropout = dropout

        self.attentions = [GATLayer(orignal_features,anchor_features, hidden_features, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]

    def forward(self,orignal_x, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(orignal_x,x, adj) for att in self.attentions], dim=1)
        #x = torch.sum(torch.stack([att(x, edge_list) for att in self.attentions]), dim=0) / len(self.attentions)
        x = F.dropout(x, self.dropout, training=self.training)

        return x

class GATLayer(nn.Module):
    def __init__(self,orignal_features, in_features, out_features, dropout=0.5, alpha=0.2, concat=True):
        super(GATLayer, self).__init__()
        self.dropout             = dropout       
        self.orignal_features    = orignal_features 
        self.in_features         = in_features    # 
        self.out_features        = out_features   # 
        self.alpha               = alpha          # LeakyReLU with negative input slope, alpha = 0.2
        self.concat              = concat         # Always Set to True
        
        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features))).to(device)
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

        self.a = nn.Parameter(torch.zeros(size=(2*out_features, 1))).to(device)
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        
        # LeakyReLU
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self,orignal_input, input, adj):

        h = torch.matmul(orignal_input, self.W) #orignal_input(N*feature_size)
        N = h.size()[0] # Num of nodes
        f = h.size()[1] #features
        k = adj.size()[1] # Num of anchors

        h_i=h.repeat(1, k).view(N,k, f)

        input_reshape=input.view(N*k,f)
        h_j = torch.matmul(input_reshape, self.W)
        h_j=h_j.view(N,k,f)
        # Attention Mechanism
        a_input = torch.cat([h_i,h_j],dim=2)

        e       = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))

        # Masked Attention
        zero_vec  = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        
        attention = F.softmax(attention, dim=1)
        
        attention = F.dropout(attention, self.dropout, training=self.training)
        
        atten=attention.reshape(N,k,1)
        

        h_prime   = atten* h_j
        
        h_prime = torch.sum(h_prime, dim=1)  # n*d
        if args.attentionAddSelf:
            h_prime = (h_prime+h)
        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

###############################################################
class GraphReach_Layer(nn.Module):
    def __init__(self, input_dim, output_dim,dist_trainable=True):
        super(GraphReach_Layer, self).__init__()
        self.input_dim = input_dim
        self.dist_trainable = dist_trainable

        if self.dist_trainable:
            self.dist_compute = Nonlinear(1, output_dim, 1)
        
        self.linear_hidden = nn.Linear(input_dim*2, output_dim)

        if args.attention:
            self.linear_hidden_orignal_features = nn.Linear(input_dim, output_dim)
            self.gat_layer = GAT_Attention(output_dim,output_dim, output_dim)#,0.5,0.2)

        self.linear_out_position = nn.Linear(output_dim,1)
        self.act = nn.ReLU()

        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data = init.xavier_uniform_(m.weight.data, gain=nn.init.calculate_gain('relu'))
                if m.bias is not None:
                    m.bias.data = init.constant_(m.bias.data, 0.0)

        

    def forward(self, feature, dists_max, dists_argmax,dists_max2):
        if self.dist_trainable:
            
            dists_max = self.dist_compute(dists_max.unsqueeze(-1)).squeeze()
            dists_max2 = self.dist_compute(dists_max2.unsqueeze(-1)).squeeze()
            
        subset_features = feature[dists_argmax.flatten(), :]
        subset_features = subset_features.reshape((dists_argmax.shape[0], dists_argmax.shape[1],
                                                   feature.shape[1]))
        messages = subset_features * dists_max.unsqueeze(-1)        
        
        self_feature = feature.unsqueeze(1).repeat(1, dists_max.shape[1], 1)* dists_max2.unsqueeze(-1)
        
        messages = torch.cat((messages, self_feature), dim=-1)
        

        messages = self.linear_hidden(messages).squeeze()
        
        messages = self.act(messages) # n*m*d
        
        out_position = self.linear_out_position(messages).squeeze(-1)  # n*m_out embedding
        
        if args.attention:
            feature=self.linear_hidden_orignal_features(feature)
            out_structure = self.gat_layer(feature,messages,dists_max).squeeze()
        else:
            
            out_structure = torch.mean(messages, dim=1)  # n*d

        return out_position, out_structure


### Non linearity
class Nonlinear(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Nonlinear, self).__init__()

        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, output_dim)

        self.act = nn.ReLU()
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data = init.xavier_uniform_(m.weight.data, gain=nn.init.calculate_gain('relu'))
                if m.bias is not None:
                    m.bias.data = init.constant_(m.bias.data, 0.0)

        
    def forward(self, x):
        x = self.linear1(x)
        x = self.act(x)
        x = self.linear2(x)
        return x


class GraphReach(torch.nn.Module):
    def __init__(self, num_class, final_emb_size, input_dim, feature_dim, hidden_dim, output_dim,
                 feature_pre=True, layer_num=1, dropout=True, **kwargs):
        super(GraphReach, self).__init__()
        self.feature_pre = feature_pre
        self.layer_num = layer_num
        self.dropout = dropout
        if layer_num == 1:
            hidden_dim = output_dim
        
        if feature_pre:
            self.linear_pre = nn.Linear(input_dim, feature_dim)
            self.conv_first = GraphReach_Layer(feature_dim, hidden_dim)
        else:
            self.conv_first = GraphReach_Layer(input_dim, hidden_dim)
        if layer_num>1:
            self.conv_hidden = nn.ModuleList([GraphReach_Layer(hidden_dim, hidden_dim) for i in range(layer_num - 2)])
            self.conv_out = GraphReach_Layer(hidden_dim, output_dim)

        self.lin2 = nn.Linear(final_emb_size, num_class)


    def forward(self, data):
        x = data.x
        if self.feature_pre:
            x = self.linear_pre(x)

        x_position, x = self.conv_first(x, data.dists_max, data.dists_argmax,data.dists_max2)
        if self.layer_num == 1:
            return x_position
        # x = F.relu(x) # Note: optional!
        if self.dropout:
            x = F.dropout(x, training=self.training)

        for i in range(self.layer_num-2):
            _, x = self.conv_hidden[i](x, data.dists_max, data.dists_argmax,data.dists_max2)
            # x = F.relu(x) # Note: optional!
            if self.dropout:
                x = F.dropout(x, training=self.training)

        x_position, x = self.conv_out(x, data.dists_max, data.dists_argmax,data.dists_max2)
        x_position = F.normalize(x_position, p=2, dim=-1)
        x_position = F.dropout(x_position, training=self.training)
        x_position = self.lin2(x_position)
        return F.log_softmax(x_position, dim=1)

####################### NNs #############################

class MLP(torch.nn.Module):
    def __init__(self, num_class, final_emb_size, input_dim, feature_dim, hidden_dim, output_dim,
                 feature_pre=True, layer_num=2, dropout=True, **kwargs):
        super(MLP, self).__init__()
        self.feature_pre = feature_pre
        self.layer_num = layer_num
        self.dropout = dropout
        if feature_pre:
            self.linear_pre = nn.Linear(input_dim, feature_dim)
            self.linear_first = nn.Linear(feature_dim, hidden_dim)
        else:
            self.linear_first = nn.Linear(input_dim, hidden_dim)
        self.linear_hidden = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for i in range(layer_num - 2)])
        self.linear_out = nn.Linear(hidden_dim, output_dim)

        self.lin2 = nn.Linear(output_dim, num_class)


    def forward(self, data):
        x = data.x
        if self.feature_pre:
            x = self.linear_pre(x)
        x = self.linear_first(x)
        x = F.relu(x)
        if self.dropout:
            x = F.dropout(x, training=self.training)
        for i in range(self.layer_num - 2):
            x = self.linear_hidden[i](x)
            x = F.relu(x)
            if self.dropout:
                x = F.dropout(x, training=self.training)
        x = self.linear_out(x)
        x = F.normalize(x, p=2, dim=-1)
        ######
        x = F.dropout(x, training=self.training)
        x = self.lin2(x)
        return F.log_softmax(x, dim=-1)


class GCN(torch.nn.Module):
    def __init__(self, num_class, device, final_emb_size, input_dim, feature_dim, hidden_dim, output_dim,
                 feature_pre=True, layer_num=2, dropout=True, **kwargs):
        super(GCN, self).__init__()
        self.feature_pre = feature_pre
        self.layer_num = layer_num
        self.dropout = dropout
        if feature_pre:
            self.linear_pre = nn.Linear(input_dim, feature_dim)
            self.conv_first = tg.nn.GCNConv(feature_dim, hidden_dim)
        else:
            self.conv_first = tg.nn.GCNConv(input_dim, hidden_dim)
        self.conv_hidden = nn.ModuleList([tg.nn.GCNConv(hidden_dim, hidden_dim) for i in range(layer_num - 2)])
        self.conv_out = tg.nn.GCNConv(hidden_dim, output_dim)


        self.lin2 = nn.Linear(output_dim, num_class)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        if self.feature_pre:
            x = self.linear_pre(x)
        x = self.conv_first(x, edge_index)
        x = F.relu(x)
        if self.dropout:
            x = F.dropout(x, training=self.training)
        for i in range(self.layer_num-2):
            x = self.conv_hidden[i](x, edge_index)
            x = F.relu(x)
            if self.dropout:
                x = F.dropout(x, training=self.training)
        x = self.conv_out(x, edge_index)
        x = F.normalize(x, p=2, dim=-1)

        x = F.dropout(x, training=self.training)
        x = self.lin2(x)
        return F.log_softmax(x, dim=-1)
        

class SAGE(torch.nn.Module):
    def __init__(self, num_class, device, final_emb_size, input_dim, feature_dim, hidden_dim, output_dim,
                 feature_pre=True, layer_num=2, dropout=True, **kwargs):
        super(SAGE, self).__init__()
        self.feature_pre = feature_pre
        self.layer_num = layer_num
        self.dropout = dropout
        if feature_pre:
            self.linear_pre = nn.Linear(input_dim, feature_dim)
            self.conv_first = tg.nn.SAGEConv(feature_dim, hidden_dim)
        else:
            self.conv_first = tg.nn.SAGEConv(input_dim, hidden_dim)
        self.conv_hidden = nn.ModuleList([tg.nn.SAGEConv(hidden_dim, hidden_dim) for i in range(layer_num - 2)])
        self.conv_out = tg.nn.SAGEConv(hidden_dim, output_dim)

        self.lin2 = nn.Linear(output_dim, num_class)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        if self.feature_pre:
            x = self.linear_pre(x)
        x = self.conv_first(x, edge_index)
        x = F.relu(x)
        if self.dropout:
            x = F.dropout(x, training=self.training)
        for i in range(self.layer_num-2):
            x = self.conv_hidden[i](x, edge_index)
            x = F.relu(x)
            if self.dropout:
                x = F.dropout(x, training=self.training)
        x = self.conv_out(x, edge_index)
        x = F.normalize(x, p=2, dim=-1)

        x = F.dropout(x, training=self.training)
        x = self.lin2(x)
        return F.log_softmax(x, dim=-1)

class GAT(torch.nn.Module):
    def __init__(self, num_class, device, final_emb_size, input_dim, feature_dim, hidden_dim, output_dim,
                 feature_pre=True, layer_num=2, dropout=True, **kwargs):
        super(GAT, self).__init__()
        self.feature_pre = feature_pre
        self.layer_num = layer_num
        self.dropout = dropout
        if feature_pre:
            self.linear_pre = nn.Linear(input_dim, feature_dim)
            self.conv_first = tg.nn.GATConv(feature_dim, hidden_dim)
        else:
            self.conv_first = tg.nn.GATConv(input_dim, hidden_dim)
        self.conv_hidden = nn.ModuleList([tg.nn.GATConv(hidden_dim, hidden_dim) for i in range(layer_num - 2)])
        self.conv_out = tg.nn.GATConv(hidden_dim, output_dim)

        self.lin2 = nn.Linear(output_dim, num_class)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        if self.feature_pre:
            x = self.linear_pre(x)
        x = self.conv_first(x, edge_index)
        x = F.relu(x)
        if self.dropout:
            x = F.dropout(x, training=self.training)
        for i in range(self.layer_num-2):
            x = self.conv_hidden[i](x, edge_index)
            x = F.relu(x)
            if self.dropout:
                x = F.dropout(x, training=self.training)
        x = self.conv_out(x, edge_index)
        x = F.normalize(x, p=2, dim=-1)

        x = F.dropout(x, training=self.training)
        x = self.lin2(x)
        return F.log_softmax(x, dim=-1)

class GIN(torch.nn.Module):
    def __init__(self, num_class, device, final_emb_size, input_dim, feature_dim, hidden_dim, output_dim,
                 feature_pre=True, layer_num=2, dropout=True, **kwargs):
        super(GIN, self).__init__()
        self.feature_pre = feature_pre
        self.layer_num = layer_num
        self.dropout = dropout
        if feature_pre:
            self.linear_pre = nn.Linear(input_dim, feature_dim)
            self.conv_first_nn = nn.Linear(feature_dim, hidden_dim)
            self.conv_first = tg.nn.GINConv(self.conv_first_nn)
        else:
            self.conv_first_nn = nn.Linear(input_dim, hidden_dim)
            self.conv_first = tg.nn.GINConv(self.conv_first_nn)
        self.conv_hidden_nn = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for i in range(layer_num - 2)])
        self.conv_hidden = nn.ModuleList([tg.nn.GINConv(self.conv_hidden_nn[i]) for i in range(layer_num - 2)])

        self.conv_out_nn = nn.Linear(hidden_dim, output_dim)
        self.conv_out = tg.nn.GINConv(self.conv_out_nn)

        self.lin2 = nn.Linear(output_dim, num_class)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        if self.feature_pre:
            x = self.linear_pre(x)
        x = self.conv_first(x, edge_index)
        x = F.relu(x)
        if self.dropout:
            x = F.dropout(x, training=self.training)
        for i in range(self.layer_num-2):
            x = self.conv_hidden[i](x, edge_index)
            x = F.relu(x)
            if self.dropout:
                x = F.dropout(x, training=self.training)
        x = self.conv_out(x, edge_index)
        x = F.normalize(x, p=2, dim=-1)
        x = F.dropout(x, training=self.training)
        x = self.lin2(x)
        return F.log_softmax(x, dim=-1)

class PGNN(torch.nn.Module):
    def __init__(self, num_class, final_emb_size, input_dim, feature_dim, hidden_dim, output_dim,
                 feature_pre=True, layer_num=2, dropout=True, **kwargs):
        super(PGNN, self).__init__()
        self.feature_pre = feature_pre
        self.layer_num = layer_num
        self.dropout = dropout
        if layer_num == 1:
            hidden_dim = output_dim
        if feature_pre:
            self.linear_pre = nn.Linear(input_dim, feature_dim)
            self.conv_first = PGNN_layer(feature_dim, hidden_dim)
        else:
            self.conv_first = PGNN_layer(input_dim, hidden_dim)
        if layer_num>1:
            self.conv_hidden = nn.ModuleList([PGNN_layer(hidden_dim, hidden_dim) for i in range(layer_num - 2)])
            self.conv_out = PGNN_layer(hidden_dim, output_dim)

        self.lin2 = nn.Linear(final_emb_size, num_class)

    def forward(self, data):
        x = data.x
        if self.feature_pre:
            x = self.linear_pre(x)
        x_position, x = self.conv_first(x, data.dists_max, data.dists_argmax)
        if self.layer_num == 1:
            return x_position
        # x = F.relu(x) # Note: optional!
        if self.dropout:
            x = F.dropout(x, training=self.training)
        for i in range(self.layer_num-2):
            _, x = self.conv_hidden[i](x, data.dists_max, data.dists_argmax)
            # x = F.relu(x) # Note: optional!
            if self.dropout:
                x = F.dropout(x, training=self.training)
        # x_position, x = self.conv_out(x, data.dists_max, data.dists_argmax)
        # x_position = F.normalize(x_position, p=2, dim=-1)
        # return x_position

        x_position, x = self.conv_out(x, data.dists_max, data.dists_argmax)
        x_position = F.normalize(x_position, p=2, dim=-1)
        x_position = F.dropout(x_position, training=self.training)
        x_position = self.lin2(x_position)
        return F.log_softmax(x_position, dim=1)

