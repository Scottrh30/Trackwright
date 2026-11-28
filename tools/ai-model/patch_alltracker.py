"""Make the AllTracker code exportable to ONNX and runnable without a GPU.

Run from the root of a checkout of https://github.com/aharley/alltracker (commit
e7553135e7b361590dbccd10e2b274b024f41cd6). The changes don't change the results:
- 'area' downsampling of even-sized maps by 2 is the same as 2x2 average pooling,
  which exports with fixed shapes;
- the correlation lookup's dilation is always 1: skip the multiply, which otherwise
  bakes a huge constant into the graph;
- no forced .cuda() calls.
"""
import re

p = "nets/blocks.py"
s = open(p).read()
a = "            fmap2 = F.interpolate(fmap2, scale_factor=0.5, mode='area')"
assert a in s, "blocks.py: interpolate line not found"
s = s.replace(a, "            fmap2 = F.avg_pool2d(fmap2, 2, ceil_mode=True)")
a = "            delta_lvl = delta_lvl * dilation.view(batch * h1 * w1, 1, 1, 1)"
assert a in s, "blocks.py: dilation line not found"
s = s.replace(a, "            if dilation is not None:\n                delta_lvl = delta_lvl * dilation.view(batch * h1 * w1, 1, 1, 1)")
a = "        if dilation is None:\n            dilation = torch.ones(batch, 1, h1, w1, device=coords.device)\n"
assert a in s, "blocks.py: dilation default not found"
s = s.replace(a, "")
open(p, "w").write(s)

p = "nets/alltracker.py"
s = open(p).read()
s = s.replace("images_chunk = images_chunk.cuda()", "images_chunk = images_chunk")
s = s.replace("images_ = images_.cuda()", "images_ = images_")
open(p, "w").write(s)
print("patched")
