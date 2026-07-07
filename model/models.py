import torch as th
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from .torch_gcn import GCN
from .torch_gat import GAT

class BertClassifier(th.nn.Module):
    def __init__(self, pretrained_model='roberta_base', nb_class=2):
        super(BertClassifier, self).__init__()
        self.nb_class = nb_class
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
        self.bert_model = AutoModel.from_pretrained(pretrained_model)
        self.feat_dim = list(self.bert_model.modules())[-2].out_features
        self.classifier = th.nn.Linear(self.feat_dim, nb_class)

    def forward(self, input_ids, attention_mask):
        cls_feats = self.bert_model(input_ids, attention_mask)[0][:, 0]
        cls_logit = self.classifier(cls_feats)
        return cls_logit

class BertGCN(th.nn.Module):
    def __init__(self, pretrained_model='roberta_base', nb_class=20, m=0.7, gcn_layers=2, n_hidden=200, dropout=0.5):
        super(BertGCN, self).__init__()
        self.m = m
        self.nb_class = nb_class
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
        self.bert_model = AutoModel.from_pretrained(pretrained_model)
        self.feat_dim = list(self.bert_model.modules())[-2].out_features
        self.classifier = th.nn.Linear(self.feat_dim, nb_class)
        self.gcn = GCN(
            in_feats=self.feat_dim,
            n_hidden=n_hidden,
            n_classes=nb_class,
            n_layers=gcn_layers-1,
            activation=F.elu,
            dropout=dropout
        )

    def forward(self, g, idx):
        input_ids, attention_mask = g.ndata['input_ids'][idx], g.ndata['attention_mask'][idx]
        if self.training:
            cls_feats = self.bert_model(input_ids, attention_mask)[0][:, 0]
            g.ndata['cls_feats'][idx] = cls_feats
        else:
            cls_feats = g.ndata['cls_feats'][idx]
        cls_logit = self.classifier(cls_feats)
        cls_pred = th.nn.Softmax(dim=1)(cls_logit)
        gcn_logit = self.gcn(g.ndata['cls_feats'], g, g.edata['edge_weight'])[idx]
        gcn_pred = th.nn.Softmax(dim=1)(gcn_logit)
        pred = (gcn_pred+1e-10) * self.m + cls_pred * (1 - self.m)
        pred = th.log(pred)
        return pred

class BertGCN_gated_fusion(th.nn.Module):
    def __init__(self, pretrained_model='/geode3/projects/IN-REGI-PDM/Delirium/Delirium-Workspace/ara-lena/new_algorithm/paper_3_again/saint/models_huggingface/bioformer-16L', nb_class=20, m=0.7, gcn_layers=2, n_hidden=200, dropout=0.5):
        super(BertGCN_gated_fusion, self).__init__()
        self.m = m
        self.nb_class = nb_class
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
        self.bert_model = AutoModel.from_pretrained(pretrained_model)

        self.feat_dim = list(self.bert_model.modules())[-2].out_features
        self.classifier = th.nn.Linear(self.feat_dim, nb_class)
        self.tabular_dim = 39

        self.tabular_proj = th.nn.Sequential(
            th.nn.Linear(39,128),
            th.nn.ReLU(),
            th.nn.Linear(128,self.feat_dim),
            th.nn.LayerNorm(self.feat_dim)
        )

        self.gate_net = th.nn.Sequential(
            th.nn.Linear(self.feat_dim * 2, 256),
            th.nn.ReLU(),
            th.nn.Linear(256, 1)
        )
        #self.alpha = th.nn.Parameter(th.tensor(-1.5))#Check: ~0.18 -> 82%CLS, 18%tabular
        self.gcn = GCN(
            in_feats=self.feat_dim,
            n_hidden=n_hidden,
            n_classes=nb_class,
            n_layers=gcn_layers-1,
            activation=F.elu,
            dropout=dropout
        )

        self.fusion_norm = th.nn.LayerNorm(self.feat_dim)

    def forward(self, g, idx):
        device = next(self.parameters()).device
        input_ids, attention_mask = g.ndata['input_ids'][idx].to(device), g.ndata['attention_mask'][idx].to(device)
        if self.training:
            cls_feats = self.bert_model(input_ids, attention_mask)[0][:, 0]

        else:
            cls_feats = g.ndata['cls_feats'][idx].to(device)

        cls_all = g.ndata['cls_feats'].to(device).detach().clone()#prevents gradients from trying to flow through cached CLS embeddings.

        cls_all[idx] = cls_feats


        tab_orig = g.ndata['tab_feats_orig'].to(device)
        tab_projected = self.tabular_proj(tab_orig)


        fusion_input = th.cat([cls_all, tab_projected], dim=1)


        gate = th.sigmoid(
            self.gate_net(fusion_input)
        )
        #print('checking the value of gate', gate)
        #print('checking the type and size of gate', type(gate), gate.size())
        fused_all = self.fusion_norm(cls_all + gate * tab_projected)

        cls_logit = self.classifier(fused_all[idx])#(cls_feats)

        cls_pred = F.softmax(cls_logit, dim=1)
        gcn_logit = self.gcn(fused_all, g, g.edata['edge_weight'])[idx] #g.ndata['cls_feats']
        
        gcn_pred = F.softmax(gcn_logit, dim=1)
        pred = (gcn_pred+1e-10) * self.m + cls_pred * (1 - self.m)
        pred = th.log(pred)
        return pred

class BertGCN_fusion(th.nn.Module):
    def __init__(self, pretrained_model='/path_to_Bioformer/bioformer-16L', nb_class=2, m=0.7, gcn_layers=2, n_hidden=200, dropout=0.5):
        super(BertGCN_fusion, self).__init__()
        self.m = m
        self.nb_class = nb_class
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
        self.bert_model = AutoModel.from_pretrained(pretrained_model)
        self.feat_dim = list(self.bert_model.modules())[-2].out_features
        self.classifier = th.nn.Linear(self.feat_dim, nb_class)
        #new start
        self.tabular_dim = 39   # e.g., 4 + 31 + 4 = 39
        self.tabular_proj = th.nn.Linear(self.tabular_dim, self.feat_dim)
        self.fusion_proj = th.nn.Linear(self.feat_dim + self.feat_dim, self.feat_dim)

        #end
        self.gcn = GCN(
            in_feats=self.feat_dim,
            n_hidden=n_hidden,
            n_classes=nb_class,
            n_layers=gcn_layers-1,
            activation=F.elu,
            dropout=dropout
        )

    def forward(self, g, idx):
        device = next(self.parameters()).device
        input_ids = g.ndata['input_ids'][idx].to(device)
        attention_mask = g.ndata['attention_mask'][idx].to(device)
        if self.training:
            cls_batch= self.bert_model(input_ids, attention_mask)[0][:, 0]
        else:
            cls_batch = g.ndata['cls_feats'][idx].to(device)

         # --- tabular fusion ---

        #Compute gradients for the idx batch while using detached precomputed bg features for rest of the graph's message passing
        if 'tab_feats_orig' in g.ndata:
            tab_batch = g.ndata['tab_feats_orig'][idx].to(device)
            tab_projected_batch = self.tabular_proj(tab_batch)

            fused_batch = self.fusion_proj(th.cat([cls_batch, tab_projected_batch], dim=1))

        else:
            fused_batch = cls_batch

        fused_all = g.ndata['fused_feats'].to(device).clone()
        fused_all[idx] = fused_batch
        cls_logit = self.classifier(fused_batch)
        cls_pred = th.nn.Softmax(dim=1)(cls_logit)

        gcn_logit_all = self.gcn(fused_all, g, g.edata['edge_weight'])
        gcn_logit = gcn_logit_all[idx]
        gcn_pred = th.nn.Softmax(dim=1)(gcn_logit)
        pred = (gcn_pred+1e-10) * self.m + cls_pred * (1 - self.m)
        pred = th.log(pred)
        return pred

class BertGAT(th.nn.Module):
    def __init__(self, pretrained_model='/path_to_LLM/models_huggingface/biobert', nb_class=2, m=0.7, gcn_layers=2, heads=8, n_hidden=32, dropout=0.5):
        super(BertGAT, self).__init__()
        self.m = m
        self.nb_class = nb_class
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
        self.bert_model = AutoModel.from_pretrained(pretrained_model)
        self.feat_dim = list(self.bert_model.modules())[-2].out_features
        self.classifier = th.nn.Linear(self.feat_dim, nb_class)
        self.gcn = GAT(
                 num_layers=gcn_layers-1,
                 in_dim=self.feat_dim,
                 num_hidden=n_hidden,
                 num_classes=nb_class,
                 heads=[heads] * (gcn_layers-1) + [1],
                 activation=F.elu,
                 feat_drop=dropout,
                 attn_drop=dropout,
        )

    def forward(self, g, idx):
        input_ids, attention_mask = g.ndata['input_ids'][idx], g.ndata['attention_mask'][idx]
        if self.training:
            cls_feats = self.bert_model(input_ids, attention_mask)[0][:, 0]
            g.ndata['cls_feats'][idx] = cls_feats
        else:
            cls_feats = g.ndata['cls_feats'][idx]
        cls_logit = self.classifier(cls_feats)
        cls_pred = th.nn.Softmax(dim=1)(cls_logit)
        gcn_logit = self.gcn(g.ndata['cls_feats'], g)[idx]
        gcn_pred = th.nn.Softmax(dim=1)(gcn_logit)
        pred = (gcn_pred+1e-10) * self.m + cls_pred * (1 - self.m)
        pred = th.log(pred)
        return pred
