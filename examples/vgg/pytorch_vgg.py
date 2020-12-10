import jax.numpy as jn
import torch
import torchvision

from objax.zoo import vgg

mo = vgg.VGG16()
vgg.load_pretrained_weights_from_pytorch(mo)
print(mo.vars())

mt = torchvision.models.vgg16(pretrained=True)
mt.eval()  # Wow that's error prone
x = torch.randn((4, 3, 224, 224))
yt = mt(x)  # (4, 1000)

for name, param in mt.state_dict().items():
    print(f'{name:40s} {tuple(param.shape)}')

yo = mo(x.numpy(), training=False)
print('Max difference:', jn.abs(yt.detach().numpy() - yo).max())
