# Timm 要求

## Timm data transform

```python
data_cfg = timm.data.resolve_data_config(model.pretrained_cfg)
transform = timm.data.create_transform(**data_cfg)
transform
```

## Timm model

```python
import timm

m = timm.create_model('mobilenetv3_large_100', pretrained=True)
m.eval()
```

## Timm feature extractor

Without modifying the network, one can call model.forward_features(input) on any model instead of the usual model(input). This will bypass the head classifier and global pooling for networks.

```python
import timm
import torch
x = torch.randn(1, 3, 224, 224)
model = timm.create_model('mobilenetv3_large_100', pretrained=True)
features = model.forward_features(x)
print(features.shape) # torch.Size([1, 960, 7, 7])
```
