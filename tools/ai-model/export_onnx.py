"""Export AllTracker to the two ONNX files Trackwright runs.

usage: python export_onnx.py ALLTRACKER_CHECKOUT alltracker.pth OUT_DIR

at_encoder.onnx: frames (n x 3 x 256 x 384, ImageNet-normalised) -> feature maps.
at_window.onnx:  one 16-frame tracking window (4 refinement steps), as in the
                 authors' forward_window, returning the full-resolution flow and
                 visibility/confidence plus the 1/8 state for the next window.
"""
import sys, torch
sys.path.insert(0, sys.argv[1])
from nets.alltracker import Net

torch.set_grad_enabled(False)
m = Net(16, init_weights=False)
sd = torch.load(sys.argv[2], map_location="cpu", weights_only=False)
m.load_state_dict(sd["model"], strict=True)
m.eval()
out = sys.argv[3]
H, W, S = 256, 384, 16
H8, W8, C = H // 8, W // 8, m.dim * 2


class Enc(torch.nn.Module):
    def __init__(s, m):
        super().__init__()
        s.m = m

    def forward(s, x):
        return s.m.dot_conv(s.m.cnn(x))


class Win(torch.nn.Module):
    def __init__(s, m):
        super().__init__()
        s.m = m

    def forward(s, fa, f2, fl8, vc8):
        fp, vp, fl8o, vc8o, _ = s.m.forward_window(fa, f2, vc8, iters=4, flowfeat=None, flows8=fl8, is_training=False)
        return fp[-1], torch.sigmoid(vp[-1]), fl8o, vc8o


torch.onnx.export(Enc(m).eval(), (torch.randn(2, 3, H, W),), f"{out}/at_encoder.onnx",
                  input_names=["images"], output_names=["fmaps"],
                  dynamic_axes={"images": {0: "n"}, "fmaps": {0: "n"}}, opset_version=17, dynamo=False)
torch.onnx.export(Win(m).eval(),
                  (torch.randn(1, C, H8, W8), torch.randn(1, S, C, H8, W8), torch.zeros(S, 2, H8, W8), torch.zeros(S, 2, H8, W8)),
                  f"{out}/at_window.onnx",
                  input_names=["fmap_anchor", "fmaps", "flows8", "visconfs8"],
                  output_names=["flows", "visconfs", "flows8_out", "visconfs8_out"], opset_version=17, dynamo=False)
print("exported to", out)
