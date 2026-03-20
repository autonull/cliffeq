import torch
from torch import nn
from cliffeq.training.ff_engine import FFEngine
from cliffeq.algebra.utils import clifford_norm_sq, embed_vector, scalar_part
from cliffordlayers.signature import CliffordSignature

def run_ff_baseline():
    g = torch.tensor([1.0, 1.0])
    sig = CliffordSignature(g)

    def get_data(n=512):
        x = torch.randn(n, 2)
        y = (torch.norm(x, dim=1) > 1.0).float()
        x_pos = x.clone()
        x_pos[:, 0] = y
        x_neg = x.clone()
        x_neg[:, 0] = 1.0 - y
        return x_pos, x_neg

    x_pos, x_neg = get_data()
    x_pos_mv = embed_vector(x_pos.unsqueeze(1), sig) # (B, 1, 4)
    x_neg_mv = embed_vector(x_neg.unsqueeze(1), sig)

    threshold = 1.0

    def goodness_a(h):
        if h.dim() == 2:
            h = h.view(-1, 16, 4)
        return clifford_norm_sq(h, sig).mean(dim=-1)

    class LearnableGoodness(nn.Module):
        def __init__(self):
            super().__init__()
            self.w = nn.Parameter(torch.randn(16 * 4) * 0.01) # Smaller init
        def forward(self, h):
            if h.dim() == 3:
                h = h.reshape(h.shape[0], -1)
            return (h @ self.w)**2

    for name, goodness_fn_class in [("CliffordNorm", None), ("Learnable", LearnableGoodness)]:
        if goodness_fn_class is None:
            goodness_fn = goodness_a
            params = []
        else:
            obj = goodness_fn_class()
            goodness_fn = obj
            params = list(obj.parameters())

        engine = FFEngine(goodness_fn, threshold)
        layer = nn.Linear(4, 16 * 4)
        optimizer = torch.optim.Adam(list(layer.parameters()) + params, lr=0.01)

        print(f"\nTraining FF - {name}")
        h_pos, h_neg = engine.train_layer(layer, x_pos_mv.reshape(-1, 4), x_neg_mv.reshape(-1, 4), optimizer, n_epochs=50)

        with torch.no_grad():
            g_pos = goodness_fn(h_pos).mean()
            g_neg = goodness_fn(h_neg).mean()
        print(f"Final Goodness - Pos: {g_pos:.4f}, Neg: {g_neg:.4f}")

if __name__ == "__main__":
    run_ff_baseline()
