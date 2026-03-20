import torch
from torch import nn
from cliffeq.training.ff_engine import FFEngine
from cliffeq.algebra.utils import clifford_norm_sq, embed_vector, scalar_part
from cliffordlayers.signature import CliffordSignature

class LearnableGoodness(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.w = nn.Parameter(torch.randn(in_dim) * 0.01)
    def forward(self, h):
        if h.dim() == 3:
            h = h.reshape(h.shape[0], -1)
        return (h @ self.w)**2

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
    x_pos_mv = embed_vector(x_pos.unsqueeze(1), sig).reshape(-1, 4)
    x_neg_mv = embed_vector(x_neg.unsqueeze(1), sig).reshape(-1, 4)

    threshold = 1.0

    def goodness_norm(h):
        if h.dim() == 2:
            h = h.view(-1, 16, 4)
        return clifford_norm_sq(h, sig).mean(dim=-1)

    print("Forward-Forward Clifford Baselines:")

    # Scalar FF Baseline (using only scalar part of MV)
    layer_scalar = nn.Linear(4, 16 * 4)
    opt_scalar = torch.optim.Adam(layer_scalar.parameters(), lr=0.01)
    engine_scalar = FFEngine(lambda h: torch.norm(h, dim=-1)**2, threshold)
    print("\nTraining Scalar FF...")
    engine_scalar.train_layer(layer_scalar, x_pos_mv, x_neg_mv, opt_scalar, n_epochs=50)

    # Clifford FF - A: Norm
    layer_a = nn.Linear(4, 16 * 4)
    opt_a = torch.optim.Adam(layer_a.parameters(), lr=0.01)
    engine_a = FFEngine(goodness_norm, threshold)
    print("Training Clifford-FF-A (Clifford Norm)...")
    engine_a.train_layer(layer_a, x_pos_mv, x_neg_mv, opt_a, n_epochs=50)

    # Clifford FF - B: Learnable
    layer_b = nn.Linear(4, 16 * 4)
    good_b = LearnableGoodness(16 * 4)
    opt_b = torch.optim.Adam(list(layer_b.parameters()) + list(good_b.parameters()), lr=0.01)
    engine_b = FFEngine(good_b, threshold)
    print("Training Clifford-FF-B (Learnable Geometric)...")
    engine_b.train_layer(layer_b, x_pos_mv, x_neg_mv, opt_b, n_epochs=50)

    with torch.no_grad():
        for name, layer, engine in [("Scalar", layer_scalar, engine_scalar),
                                    ("Clifford-A", layer_a, engine_a),
                                    ("Clifford-B", layer_b, engine_b)]:
            h_pos = layer(x_pos_mv)
            h_neg = layer(x_neg_mv)
            g_pos = engine.goodness_fn(h_pos).mean()
            g_neg = engine.goodness_fn(h_neg).mean()
            print(f"Results {name:10}: Pos Goodness {g_pos:8.4f}, Neg Goodness {g_neg:8.4f}")

if __name__ == "__main__":
    run_ff_baseline()
