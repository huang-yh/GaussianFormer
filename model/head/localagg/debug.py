import torch
from local_aggregate import LocalAggregator

P = 25600
N = 80000

a = LocalAggregator(3, 200, 200, 16, [-40.0, -40.0, -1.0], 0.4).cuda()
s = torch.tensor([80.0, 80.0, 6.4]).cuda()
o = torch.tensor([-40.0, -40.0, -1.0]).cuda()

pts = torch.rand(1, N, 3).cuda() * s + o
means3D = torch.rand(1, P, 3).cuda() * s + o
opas = torch.rand(1, P).cuda()
semantics = torch.rand(1, P, 18).softmax(dim=-1).cuda()
scales = torch.rand(1, P, 3).cuda()
cov3D = torch.rand(1, P, 3, 3).cuda()

means3D.requires_grad_(True)
opas.requires_grad_(True)
semantics.requires_grad_(True)
cov3D.requires_grad_(True)

out = a(pts, means3D, opas, semantics, scales, cov3D)
import pdb; pdb.set_trace()