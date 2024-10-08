"""
Reference: https://github.com/google-research/simclr

# Note

## Keypoints in SimCLR

- data augmentation: random cropping and random color distortion stand out.
- Global BN (SyncBN): This operation aggregates BN mean and variance over all devices during the training.
- Projector: a MLP with BN. By leveraging the nonlinear transformation g(·), more information can be formed and maintained in h. 2048-2048-256
- Batch size: it is crucial for improving performance. BS=4096 achieves good results.
- Epoch: Contrastive learning benefits (more) from larger batch sizes and longer training. At least 400 epochs.
# Result


"""
import torch
from torch import nn
import gin
import timm
from .head import MLPHead, build_backbone, build_head
from .operation import contrastive_loss
import torch.nn.functional as F

@gin.configurable
class SimCLR(nn.Module):
    def __init__(self, 
                 out_dim=256,
                 embed_dim=2048,
                 mlp_dim=2048, 
                 temperature=0.5):
        super(SimCLR, self).__init__()
        self.temperature = temperature
        self.out_dim = out_dim
        backbone = build_backbone()
        embed_dim = backbone(torch.randn(10,3,224,224)).shape[1]
        self.embed_dim = embed_dim
        self.backbone = backbone
        self.projector = MLPHead(embed_dim, out_dim, mlp_dim, False)

    @torch.no_grad()
    def representation(self, x):
        if isinstance(x, list) or isinstance(x, tuple):
            x = x[0]
        latent = self.backbone(x)
        proj = self.projector(latent)
        rep = dict(latent=latent,proj=proj)
        return rep

    def forward(self, samples, **kwargs):
        self.log = {}
        x1,x2 = samples[:2]

        z1 = self.projector(self.backbone(x1))
        z2 = self.projector(self.backbone(x2))
        
        loss = (contrastive_loss(z1,z2,self.temperature) + 
                contrastive_loss(z2,z1,self.temperature))/2

        self.log['z@sim'] = F.cosine_similarity(z1,z2).mean().item()

        return loss, self.log