from abc import abstractmethod
import math
from torch import nn
import torch.nn.functional as F
import torch, gin
from timm.models.layers import trunc_normal_
from timm import create_model

@gin.configurable
class ClassificationHead(nn.Module):
    """ Head for classification
    """
    def __init__(self, embed_dim, num_classes,head_type="CLS",head_net=nn.Linear, loss_fn=nn.CrossEntropyLoss):
        super().__init__()
        assert head_type in ["CLS","AVG","Hyber"]
        self.head_type = head_type
        if head_type == "Hyber":
            self.head = head_net(embed_dim*2, num_classes)
        else:
            self.head = head_net(embed_dim, num_classes)
        
    def forward(self, x):
        if self.head_type == "CLS":
            x=(x[:, 0])
        elif self.head_type == "AVG":
            x=(x[:,1:].mean(1))
        elif self.head_type == "Hyber":
            x=(torch.cat([x[:, 0], x[:,1:].mean(1)],dim=-1))
        return self.head(x)
    
@gin.configurable()
class SegmentationHead(nn.Module):
    def __init__(self, embed_dim, output_shape,head_net=nn.Linear) -> None:
        super().__init__()
        patch_size = output_shape[1]
        self.fpn = nn.ConvTranspose2d(embed_dim, output_shape[0], kernel_size=patch_size, stride=patch_size)
    
    def forward(self, x):
        return self.fpn(x)
    
    


@gin.configurable
class RECHead(nn.Module):
    def __init__(self, in_dim, in_chans=3, patch_size=16):
        super().__init__()

        layers = [nn.Linear(in_dim, in_dim)]
        layers.append(nn.GELU())
        layers.append(nn.Linear(in_dim, in_dim))
        layers.append(nn.GELU())
        layers.append(nn.Linear(in_dim, in_dim))
        layers.append(nn.GELU())

        self.mlp = nn.Sequential(*layers)
        self.apply(self._init_weights)
        
        self.convTrans = nn.ConvTranspose2d(in_dim, in_chans, kernel_size=(patch_size, patch_size), 
                                                stride=(patch_size, patch_size))


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.mlp(x)
        x_rec = x.transpose(1, 2)
        out_sz = tuple( (  int(math.sqrt(x_rec.size()[2]))  ,   int(math.sqrt(x_rec.size()[2])) ) )
        x_rec = self.convTrans(x_rec.unflatten(2, out_sz))
                
        return x_rec

# copy from https://github.com/facebookresearch/dino/blob/main/vision_transformer.py#L257
@gin.configurable()
class DINOHead(nn.Module):
    def __init__(self, in_dim, out_dim, use_bn=False, norm_last_layer=True, nlayers=3, hidden_dim=2048, bottleneck_dim=256):
        super().__init__()

        nlayers = max(nlayers, 1)
        if nlayers == 1:
            self.mlp = nn.Linear(in_dim, bottleneck_dim)
        else:
            layers = [nn.Linear(in_dim, hidden_dim)]
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
            for _ in range(nlayers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if use_bn:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            self.mlp = nn.Sequential(*layers)
        self.apply(self._init_weights)
        self.last_layer = nn.utils.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1)
        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.mlp(x)
        x = nn.functional.normalize(x, dim=-1, p=2)
        x = self.last_layer(x)
        return x

@gin.configurable()
class MLPHead(nn.Module):
    def __init__(self, in_dim, out_dim,hidden_dim=8124, use_bn=False,nlayers=3):
        super().__init__()
        
        
        nlayers = max(nlayers, 1)
        if nlayers == 1:
            self.mlp = nn.Linear(in_dim, out_dim)
        else:
            layers = [nn.Linear(in_dim, hidden_dim)]
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
            for _ in range(nlayers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if use_bn:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, out_dim))
            self.mlp = nn.Sequential(*layers)
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
            
    def forward(self, x):
        x = self.mlp(x)
        return x
                 
@gin.configurable(allowlist=['head_fn'])
def build_head(*args,head_fn=ClassificationHead,**kwargs):
    return head_fn(*args,**kwargs)


@gin.configurable()
def build_backbone(model_name="resnet50",pretrained=False, **kwargs):
    return create_model(model_name,pretrained=pretrained,**kwargs)
