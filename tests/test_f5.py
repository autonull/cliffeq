import torch
from torch import nn
from cliffeq.training.ff_engine import FFEngine
from cliffeq.algebra.utils import clifford_norm_sq
from cliffordlayers.signature import CliffordSignature

def test_f5_ff():
    g = torch.tensor([1.0, 1.0])
    sig = CliffordSignature(g)
    threshold = 1.0
    def goodness_fn(h):
        return clifford_norm_sq(h, sig)
    engine = FFEngine(goodness_fn, threshold)
    layer = nn.Linear(4, 4)
    optimizer = torch.optim.Adam(layer.parameters(), lr=0.1)
    positive_data = torch.randn(10, 4) * 2.0
    negative_data = torch.randn(10, 4) * 0.5
    engine.train_layer(layer, positive_data, negative_data, optimizer, n_epochs=20)
    with torch.no_grad():
        h_pos = layer(positive_data)
        h_neg = layer(negative_data)
        g_pos_final = goodness_fn(h_pos).mean()
        g_neg_final = goodness_fn(h_neg).mean()
    assert g_pos_final > threshold
    assert g_neg_final < threshold
    print("test_f5_ff passed")

if __name__ == "__main__":
    test_f5_ff()
