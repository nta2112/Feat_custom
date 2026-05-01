import torch
import torch.nn as nn
import numpy as np

class FewShotModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        if args.backbone_class == 'ConvNet':
            from model.networks.convnet import ConvNet
            self.encoder = ConvNet()
        elif args.backbone_class == 'Res12':
            hdim = 640
            from model.networks.res12 import ResNet
            self.encoder = ResNet()
        elif args.backbone_class == 'Res18':
            hdim = 512
            from model.networks.res18 import ResNet
            self.encoder = ResNet()
        elif args.backbone_class == 'WRN':
            hdim = 640
            from model.networks.WRN28 import Wide_ResNet
            self.encoder = Wide_ResNet(28, 10, 0.5)  # we set the dropout=0.5 directly here, it may achieve better results by tunning the dropout
        else:
            raise ValueError('')

    def split_instances(self, data):
        args = self.args
        nb = data.shape[0]
        ni = data.shape[1]
        if self.training:
            way, shot, query = args.way, args.shot, args.query
        else:
            way, shot, query = args.eval_way, args.eval_shot, args.eval_query
            
        s_idx = torch.arange(way * shot).view(1, shot, way)
        q_idx = torch.arange(way * shot, way * (shot + query)).view(1, query, way)
        
        offsets = torch.arange(nb).view(nb, 1, 1) * ni
        s_idx = (s_idx + offsets).long()
        q_idx = (q_idx + offsets).long()
        
        if data.is_cuda:
            s_idx = s_idx.cuda()
            q_idx = q_idx.cuda()
            
        return s_idx, q_idx

    def forward(self, x, get_feature=False):
        if get_feature:
            # get feature with the provided embeddings
            return self.encoder(x)
        else:
            # feature extraction
            if x.dim() == 4:
                x = x.unsqueeze(0)

            nb, ni, c, h, w = x.shape
            instance_embs = self.encoder(x.view(-1, c, h, w))
            
            # split support query set for few-shot data
            support_idx, query_idx = self.split_instances(x)
            if self.training:
                logits, logits_reg = self._forward(instance_embs, support_idx, query_idx)
                return logits, logits_reg
            else:
                logits = self._forward(instance_embs, support_idx, query_idx)
                return logits

    def _forward(self, x, support_idx, query_idx):
        raise NotImplementedError('Suppose to be implemented by subclass')