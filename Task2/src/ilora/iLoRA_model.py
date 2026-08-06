import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_sequence
from torch_geometric.nn import RGCNConv, GraphConv, GCNConv
import numpy as np, itertools, random, copy, math
from npn import *

class Attention(nn.Module):
    def __init__(self, embed_dim, hidden_dim=None, out_dim=None, n_head=1, score_function='dot_product', dropout=0):
        ''' Attention Mechanism
        :param embed_dim:
        :param hidden_dim:
        :param out_dim:
        :param n_head: num of head (Multi-Head Attention)
        :param score_function: scaled_dot_product / mlp (concat) / bi_linear (general dot)
        :return (?, q_len, out_dim,)
        '''
        super(Attention, self).__init__()
        if hidden_dim is None:
            hidden_dim = embed_dim // n_head
        if out_dim is None:
            out_dim = embed_dim
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.n_head = n_head
        self.score_function = score_function
        self.w_k = nn.Linear(embed_dim, n_head * hidden_dim)
        self.w_q = nn.Linear(embed_dim, n_head * hidden_dim)
        self.proj = nn.Linear(n_head * hidden_dim, out_dim)
        self.dropout = nn.Dropout(dropout)
        if score_function == 'mlp':
            self.weight = nn.Parameter(torch.Tensor(hidden_dim * 2))
        elif self.score_function == 'bi_linear':
            self.weight = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        else:  # dot_product / scaled_dot_product
            self.register_parameter('weight', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.hidden_dim)
        if self.weight is not None:
            self.weight.data.uniform_(-stdv, stdv)

    def forward(self, k, q):
        if len(q.shape) == 2:  # q_len missing
            q = torch.unsqueeze(q, dim=1)
        if len(k.shape) == 2:  # k_len missing
            k = torch.unsqueeze(k, dim=1)
        mb_size = k.shape[0]  # ?
        k_len = k.shape[1]
        q_len = q.shape[1]
        # k: (?, k_len, embed_dim,)
        # q: (?, q_len, embed_dim,)
        # kx: (n_head*?, k_len, hidden_dim)
        # qx: (n_head*?, q_len, hidden_dim)
        # score: (n_head*?, q_len, k_len,)
        # output: (?, q_len, out_dim,)
        kx = self.w_k(k).view(mb_size, k_len, self.n_head, self.hidden_dim)
        kx = kx.permute(2, 0, 1, 3).contiguous().view(-1, k_len, self.hidden_dim)
        qx = self.w_q(q).view(mb_size, q_len, self.n_head, self.hidden_dim)
        qx = qx.permute(2, 0, 1, 3).contiguous().view(-1, q_len, self.hidden_dim)
        if self.score_function == 'dot_product':
            kt = kx.permute(0, 2, 1)
            score = torch.bmm(qx, kt)
        elif self.score_function == 'scaled_dot_product':
            kt = kx.permute(0, 2, 1)
            qkt = torch.bmm(qx, kt)
            score = torch.div(qkt, math.sqrt(self.hidden_dim))
        elif self.score_function == 'mlp':
            kxx = torch.unsqueeze(kx, dim=1).expand(-1, q_len, -1, -1)
            qxx = torch.unsqueeze(qx, dim=2).expand(-1, -1, k_len, -1)
            kq = torch.cat((kxx, qxx), dim=-1)  # (n_head*?, q_len, k_len, hidden_dim*2)
            # kq = torch.unsqueeze(kx, dim=1) + torch.unsqueeze(qx, dim=2)
            score = torch.tanh(torch.matmul(kq, self.weight))
        elif self.score_function == 'bi_linear':
            qw = torch.matmul(qx, self.weight)
            kt = kx.permute(0, 2, 1)
            score = torch.bmm(qw, kt)
        else:
            raise RuntimeError('invalid score_function')
        # score = F.softmax(score, dim=-1)
        score = F.softmax(score, dim=0)
        # print (score)
        # print (sum(score))
        output = torch.bmm(score, kx)  # (n_head*?, q_len, hidden_dim)
        output = torch.cat(torch.split(output, mb_size, dim=0), dim=-1)  # (?, q_len, n_head*hidden_dim)
        output = self.proj(output)  # (?, q_len, out_dim)
        output = self.dropout(output)
        return output, score

class MatchingAttention(nn.Module):

    def __init__(self, mem_dim, cand_dim, alpha_dim=None, att_type='general'):
        super(MatchingAttention, self).__init__()
        assert att_type != 'concat' or alpha_dim != None
        assert att_type != 'dot' or mem_dim == cand_dim
        self.mem_dim = mem_dim
        self.cand_dim = cand_dim
        self.att_type = att_type
        if att_type == 'general':
            self.transform = nn.Linear(cand_dim, mem_dim, bias=False)
        if att_type == 'general2':
            self.transform = nn.Linear(cand_dim, mem_dim, bias=True)
            # torch.nn.init.normal_(self.transform.weight,std=0.01)
        elif att_type == 'concat':
            self.transform = nn.Linear(cand_dim + mem_dim, alpha_dim, bias=False)
            self.vector_prod = nn.Linear(alpha_dim, 1, bias=False)

    def forward(self, M, x, mask=None):
        """
        M -> (seq_len, batch, mem_dim)
        x -> (batch, cand_dim)
        mask -> (batch, seq_len)
        """
        if type(mask) == type(None):
            mask = torch.ones(M.size(1), M.size(0)).type(M.type())

        if self.att_type == 'dot':
            # vector = cand_dim = mem_dim
            M_ = M.permute(1, 2, 0)  # batch, vector, seqlen
            x_ = x.unsqueeze(1)  # batch, 1, vector
            alpha = F.softmax(torch.bmm(x_, M_), dim=2)  # batch, 1, seqlen
        elif self.att_type == 'general':
            M_ = M.permute(1, 2, 0)  # batch, mem_dim, seqlen
            x_ = self.transform(x).unsqueeze(1)  # batch, 1, mem_dim
            alpha = F.softmax(torch.bmm(x_, M_), dim=2)  # batch, 1, seqlen
        elif self.att_type == 'general2':
            M_ = M.permute(1, 2, 0)  # batch, mem_dim, seqlen
            x_ = self.transform(x).unsqueeze(1)  # batch, 1, mem_dim
            mask_ = mask.unsqueeze(2).repeat(1, 1, self.mem_dim).transpose(1, 2)  # batch, seq_len, mem_dim
            M_ = M_ * mask_
            alpha_ = torch.bmm(x_, M_) * mask.unsqueeze(1)
            alpha_ = torch.tanh(alpha_)
            alpha_ = F.softmax(alpha_, dim=2)
            # alpha_ = F.softmax((torch.bmm(x_, M_))*mask.unsqueeze(1), dim=2) # batch, 1, seqlen
            alpha_masked = alpha_ * mask.unsqueeze(1)  # batch, 1, seqlen
            alpha_sum = torch.sum(alpha_masked, dim=2, keepdim=True)  # batch, 1, 1
            alpha = alpha_masked / alpha_sum  # batch, 1, 1 ; normalized
            # import ipdb;ipdb.set_trace()
        else:
            M_ = M.transpose(0, 1)  # batch, seqlen, mem_dim
            x_ = x.unsqueeze(1).expand(-1, M.size()[0], -1)  # batch, seqlen, cand_dim
            M_x_ = torch.cat([M_, x_], 2)  # batch, seqlen, mem_dim+cand_dim
            mx_a = F.tanh(self.transform(M_x_))  # batch, seqlen, alpha_dim
            alpha = F.softmax(self.vector_prod(mx_a), 1).transpose(1, 2)  # batch, 1, seqlen

        attn_pool = torch.bmm(alpha, M.transpose(0, 1))[:, 0, :]  # batch, mem_dim
        return attn_pool, alpha

class encode_mean_std(nn.Module):
    def __init__(self, graph_node_dim, dropout=0.1):
        super(encode_mean_std, self).__init__()

        self.graph_node_dim = graph_node_dim

        self.dropout = dropout

        self.enc = nn.Sequential(
            nn.Linear(graph_node_dim, graph_node_dim),
            nn.ReLU(),
            nn.Dropout(dropout))
        self.enc_g = nn.Sequential(
            nn.Linear(graph_node_dim, graph_node_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.enc_b = nn.Sequential(
            nn.Linear(graph_node_dim, graph_node_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.enc_mean = nn.Linear(graph_node_dim, graph_node_dim)
        self.enc_std = nn.Sequential(
            nn.Linear(graph_node_dim, graph_node_dim),
            nn.Softplus())

    def forward(self, x):
        enc_ = self.enc(x)
        enc_g = self.enc_g(enc_)
        enc_b = self.enc_b(enc_)
        mean = self.enc_mean(enc_g)
        std = self.enc_std(enc_g)
        return mean, std, enc_b


class GraphNN(torch.nn.Module):
    def __init__(self, num_features, hidden_size=64, dropout=0.5):
        """
        The Speaker-level context encoder in the form of a 2 layer GCN.
        """
        super(GraphNN, self).__init__()

        self.conv1 = GraphConv(num_features, hidden_size)
        self.conv2 = GraphConv(hidden_size, hidden_size)
        # self.conv1 = GCNConv(num_features, hidden_size)
        # self.conv2 = GCNConv(hidden_size, hidden_size)


    def forward(self, x, edge_index, edge_weight):
        x = x.squeeze(1)
        out = self.conv1(x, edge_index, edge_weight)
        out = self.conv2(out, edge_index, edge_weight)
        out = out.unsqueeze(1)

        return out

class ILoRAGCNModel(nn.Module):
    def __init__(self, base_model, graph_node_dim, D_m, D_e, D_h, dropout_rec=0.5, dropout=0.5):

        super(ILoRAGCNModel, self).__init__()
        self.base_model = base_model

        if self.base_model == 'LSTM':
            self.lstm = nn.LSTM(input_size=D_m, hidden_size=D_e, num_layers=2, bidirectional=True, dropout=dropout)

        elif self.base_model == 'GRU':
            self.gru = nn.GRU(input_size=D_m, hidden_size=D_e, num_layers=2, bidirectional=True, dropout=dropout)

        elif self.base_model == 'None':
            self.base_linear = nn.Linear(D_m, 2 * D_e)

        else:
            print('Base model must be one of LSTM/GRU')
            raise NotImplementedError

        self.graph_node_dim = graph_node_dim
        self.dropout_rec = nn.Dropout(dropout + 0.15)
        self.dropout = nn.Dropout(dropout)

        self.matchatt = MatchingAttention(2*D_e+D_e, 2*D_e+D_e, att_type='general2')

        # # prior
        # self.prior_enc = encode_mean_std(self.graph_node_dim, dropout)
        # self.prior_mij = nn.Linear(self.graph_node_dim, self.graph_node_dim)

        # # post
        # self.post_enc = encode_mean_std(self.graph_node_dim, dropout)
        # self.post_mean_approx_g = nn.Linear(self.graph_node_dim, self.graph_node_dim)
        # self.post_std_approx_g = nn.Sequential(
        #     nn.Linear(self.graph_node_dim, self.graph_node_dim),
        #     nn.Softplus())
        # prior
        self.prior_enc = encode_mean_std(self.graph_node_dim, dropout)
        self.prior_mij = nn.Linear(self.graph_node_dim, 1)

        # post
        self.post_enc = encode_mean_std(self.graph_node_dim, dropout)
        self.post_mean_approx_g = nn.Linear(self.graph_node_dim, self.graph_node_dim)
        self.post_std_approx_g = nn.Sequential(
            nn.Linear(self.graph_node_dim, self.graph_node_dim),
            nn.Softplus())

        # graph
        self.node_emb = nn.Sequential(
            nn.Linear(2*D_e, self.graph_node_dim),
            nn.ReLU())
        self.gen_edge_emb = nn.Sequential(
            nn.Linear(self.graph_node_dim*2, self.graph_node_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.graph_node_dim, self.graph_node_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.graph_node_dim, self.graph_node_dim)
        )
        self.transform = nn.Sequential(
            nn.Linear(self.graph_node_dim, self.graph_node_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.graph_node_dim, 1),
            nn.ReLU())

        self.graph_model = GraphNN(2*D_e, D_e, dropout)
        self.linear = nn.Linear(2*D_e+D_e, D_h)

        self.prior_b0 = nn.Sequential(
            nn.Linear(self.graph_node_dim, self.graph_node_dim),
            nn.Softplus()
        )
        self.npn_trans = nn.Sequential(
            NPNLinear(self.graph_node_dim, self.graph_node_dim),
            NPNRelu(),
            # NPNDropout(self.dropout_rate),
            NPNLinear(self.graph_node_dim, self.graph_node_dim),
            NPNRelu()
        )

    def forward(self, textf, qmask, umask, att2=True):
        if textf.dim() != 3:
            raise ValueError("textf must be a 3D tensor [seq_len, batch, dim].")
        batch_size = textf.shape[1]
        if batch_size == 1:
            return self._forward_single(textf, qmask, umask, att2=att2)

        hidden_list = []
        kl_l_total = None
        kl_b_total = None
        relation_vals = []

        for idx in range(batch_size):
            hidden, kl_l, kl_b, relation_val = self._forward_single(
                textf[:, idx : idx + 1, :],
                qmask[:, idx : idx + 1, :],
                umask[:, idx : idx + 1],
                att2=att2,
            )
            hidden_list.append(hidden)
            kl_l_total = kl_l if kl_l_total is None else kl_l_total + kl_l
            kl_b_total = kl_b if kl_b_total is None else kl_b_total + kl_b
            relation_vals.append(relation_val)

        hidden = torch.cat(hidden_list, dim=1)
        relation_val_cat = torch.cat(relation_vals, dim=0)
        return hidden, kl_l_total, kl_b_total, relation_val_cat

    def _forward_single(self, textf, qmask, umask, att2=True):
        seq_len, batch_size = textf.shape[0], textf.shape[1]
        total_seq_len = (seq_len-1)*seq_len
        # encode the context sentence
        if self.base_model == 'LSTM':
            sentence_emb, _ = self.lstm(textf)
        elif self.base_model == 'GRU':
            sentence_emb, _ = self.gru(textf)
        elif self.base_model == 'None':
            sentence_emb = self.base_linear(textf)

        sentence_emb = self.dropout_rec(sentence_emb)
        node_emb = self.node_emb(sentence_emb)

        # construct node pairs
        node_pairs = torch.zeros(total_seq_len, batch_size, 2*self.graph_node_dim).cuda()
        one_node = []
        two_node = []
        for i in range(seq_len-1):
            start = int((seq_len-i-2)*(seq_len-i-1)/2)
            end = int((seq_len-i)*(seq_len-i-1)/2)
            one = node_emb[seq_len-i-1].unsqueeze(0).repeat(seq_len-i-1, 1, 1)
            two = node_emb[0:seq_len-i-1]
            node_pairs[start:end] = torch.cat([one, two], dim=2)
            one_node += (seq_len-i-1)*[seq_len-i-1]
            two_node += [i for i in range(seq_len-i-1)]

        edge_index = torch.tensor([one_node+two_node, two_node+one_node], dtype=torch.long).cuda()
        node_pairs[int(total_seq_len/2):] = node_pairs[:int(total_seq_len/2)]

        # node2edge
        edge_emb = self.gen_edge_emb(node_pairs)

        input4prior = edge_emb.clone()
        input4post = edge_emb.clone()

        # prior
        prior_mean_g, prior_std_g, prior_b = self.prior_enc(input4prior)
        # estimate prior mij of Binomial Dis
        prior_mij = self.prior_mij(prior_b)
        b0 = self.prior_b0(prior_b)
        prior_mij = 0.4*torch.sigmoid(prior_mij)

        # post
        post_mean_g, post_std_g, post_b = self.post_enc(input4post)
        post_mean_approx_g = self.post_mean_approx_g(post_b)
        post_std_approx_g = self.post_std_approx_g(post_b)
        # estimate post mij for Binomial Dis
        nij = F.softplus(post_mean_approx_g) + 0.01
        nij_ = 2.0*nij*post_std_approx_g.pow(2)
        post_mij = 0.5*(1.0 + nij_ - torch.sqrt(nij_.pow(2) + 1))
        var_ij = post_mij*(1.0-post_mij)

        # post_mij and var_ij are mu and var for alpha_tilde
        # print(post_mij.shape, var_ij.shape)
        post_mij_edges = post_mij
        var_ij_edges = var_ij

        post_mij = post_mij_edges.permute(1, 0, 2).contiguous()
        var_ij = var_ij_edges.permute(1, 0, 2).contiguous()
        bs, l, h = post_mij.shape
        post_mij_flat = post_mij.reshape(bs * l, h)
        var_ij_flat = var_ij.reshape(bs * l, h)
        npn_mu, npn_var = self.npn_trans((post_mij_flat, var_ij_flat))
        npn_mu = npn_mu.reshape(bs, l, h)
        npn_var = npn_var.reshape(bs, l, h)
        pi = torch.tensor(np.pi)
        # b = (torch.sqrt(2*pi)*npn_mu+torch.sqrt(2*pi*npn_mu*npn_mu+32*npn_var))/8.0
        b = (-torch.sqrt(2*pi)*npn_mu+torch.sqrt(2*pi*npn_mu*npn_mu+8*(4-pi)*(npn_mu*npn_mu+npn_var)))/(2.0*(4-pi))
        alpha_bar = self.sample_laplace(b, post_mij)
        alpha_bar = alpha_bar.permute(1, 0, 2).contiguous()
        b_for_kl = b.permute(1, 0, 2).contiguous()
        post_mij_for_kl = post_mij_edges

        alpha_bar = torch.relu(alpha_bar)
        ei = torch.mul(alpha_bar, edge_emb)
        # print(ei[:int(ei.shape[0]/2)], ei[int(ei.shape[0]/2):])
        # print(ei.shape)
        transformed_ei = self.transform(ei).reshape(-1)
        #print(transformed_ei[:int(len(transformed_ei)/2)], transformed_ei[int(len(transformed_ei)/2):])
        # fully connected graph -> edge_inx, edge_weight

        sentence_emb_new = self.graph_model(sentence_emb, edge_index=edge_index, edge_weight=transformed_ei)
        emotions = torch.cat([sentence_emb, sentence_emb_new], dim=2)

        att_mask = umask.transpose(0, 1)

        if att2:
            att_emotions = []
            alpha = []
            for t in emotions:
                att_em, alpha_ = self.matchatt(emotions, t, mask=att_mask)
                att_emotions.append(att_em.unsqueeze(0))
                alpha.append(alpha_[:, 0, :])
            att_emotions = torch.cat(att_emotions, dim=0)
            hidden = F.relu(self.linear(att_emotions))
        else:
            hidden = F.relu(self.linear(emotions))
        hidden = self.dropout(hidden)

        # kl_g = self.kld_loss_gauss(alpha_tilde*post_mean_g, torch.sqrt(alpha_tilde)*post_std_g,
        #                             alpha_tilde*prior_mean_g, torch.sqrt(alpha_tilde)*prior_std_g)
        kl_l = self.kld_loss_laplace(b_for_kl, b0)
        kl_b = self.kld_loss_binomial_upper_bound(post_mij_for_kl, prior_mij)

        # kl_g = torch.tensor(0)
        # kl_b = torch.tensor(0)

        return hidden, kl_l, kl_b, transformed_ei


    def sample_laplace(self, b, mij):

        pi = torch.tensor(np.pi)
        mu = torch.sqrt(pi/2)*b
        std_1 = torch.sqrt((4-pi)/2)*b
        eps_1 = torch.FloatTensor(b.size()).normal_().cuda()
        std_2 = std_1*eps_1+mu

        eps_2 = torch.FloatTensor(mij.size()).normal_().cuda()
        alpha_bar = eps_2*std_2
        return alpha_bar

    def sample_repara(self, mean, std, mij):

        mean = mean[:int(len(mean)/2)]
        std = std[:int(len(std)/2)]
        mij = mij[:int(len(mij)/2)]
        mean_alpha = mij
        std_alpha = torch.sqrt(mij*(1.0 - mij))
        eps = torch.FloatTensor(std.size()).normal_().cuda()
        alpha_tilde = eps*std_alpha+mean_alpha
        alpha_tilde = F.softplus(alpha_tilde)

        mean_sij = alpha_tilde*mean
        std_sij = torch.sqrt(alpha_tilde)*std
        eps_2 = torch.FloatTensor(std.size()).normal_().cuda()
        s_ij = eps_2*std_sij+mean_sij

        alpha_bar = s_ij*alpha_tilde
        alpha_bar = torch.cat([alpha_bar, alpha_bar], dim=0)
        alpha_tilde = torch.cat([alpha_tilde, alpha_tilde], dim=0)

        # mean_alpha = mij
        # std_alpha = torch.sqrt(mij*(1.0 - mij))
        # eps = torch.FloatTensor(std.size()).normal_().cuda()
        # alpha_tilde = eps*std_alpha+mean_alpha
        # alpha_tilde = F.softplus(alpha_tilde)

        # mean_sij = alpha_tilde*mean
        # std_sij = torch.sqrt(alpha_tilde)*std
        # eps_2 = torch.FloatTensor(std.size()).normal_().cuda()
        # s_ij = eps_2*std_sij+mean_sij

        # alpha_bar = s_ij*alpha_tilde

        return alpha_bar, alpha_tilde

    def kld_loss_gauss(self, mean_post, std_post, mean_prior, std_prior):
        eps = 1e-6
        kld_element = (2*torch.log(std_prior+eps) - 2*torch.log(std_post+eps) +
                       ((std_post).pow(2) + (mean_post-mean_prior).pow(2)) /
                       (std_prior+eps).pow(2) - 1)

        return 0.5 * torch.sum(torch.abs(kld_element))

    def kld_loss_binomial_upper_bound(self, mij_post, mij_prior):
        eps = 1e-6
        first_item = mij_post*(torch.log(mij_post+eps)-torch.log(mij_prior+eps))
        second_item = (1-mij_post)*(torch.log(1-mij_post+0.5*mij_post.pow(2)+eps)-torch.log(1-mij_prior+0.5*mij_prior.pow(2)+eps))
        kld_element_term1 = first_item + second_item

        return torch.sum(torch.abs(kld_element_term1))

    def kld_loss_laplace(self, b, b0):
        eps = 1e-6
        loss = torch.log(b0+eps) - torch.log(b+eps) + b/(b0+eps) - 1
        return torch.sum(torch.abs(loss))


class ILoRAMatrix(nn.Module):
    def __init__(self, args,input_embedding_dim: int, llm_embedding_dim: int):
        super(ILoRAMatrix, self).__init__()
        self.args = args
        self.D_h = 100
        self.graph_emb_dim = 128
        self.llm_embedding_dim = llm_embedding_dim
        self.little_model = "LSTM"

        self.structured_model = ILoRAGCNModel(
            self.little_model,
            self.graph_emb_dim,
            D_m=input_embedding_dim,
            D_e = 100,
            D_h = self.D_h,
        )
        print("ILoRAMatrix initialized")

        self.projection_head = nn.Sequential(
            nn.Linear(self.D_h, self.D_h * 2),
            nn.ReLU(),
            nn.Linear(self.D_h * 2, self.args.lora_r * self.llm_embedding_dim)
        )

    def forward(self, textf, qmask, umask):
        structuredEmbed, kl_g, kl_b, relation_val = self.structured_model(textf, qmask, umask)
        context_per_item = structuredEmbed.mean(dim=0)
        lora_A_flat = self.projection_head(context_per_item)
        lora_A = lora_A_flat.view(-1, self.args.lora_r, self.llm_embedding_dim)

        return lora_A, kl_g, kl_b, relation_val
