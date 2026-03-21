import torch
from cliffeq.models.sparse import CliffordISTA, CliffordLISTA
from cliffeq.models.pc import CliffordPC
from cliffeq.models.tp import CliffordTP

def check_sparse():
    print("Checking Clifford-ISTA/LISTA...")
    sig_g = torch.tensor([1.0, 1.0])
    A = torch.randn(8, 4, 4)
    lambdas = {0: 0.1, 1: 0.1, 2: 0.1}
    ista = CliffordISTA(A, sig_g, lambdas, n_iter=10)
    y = torch.randn(2, 8, 4)
    out = ista(y)
    print(f"  ISTA output shape: {out.shape}")

    lista = CliffordLISTA(in_nodes=8, hidden_nodes=4, sig_g=sig_g, n_layers=2)
    out = lista(y)
    print(f"  LISTA output shape: {out.shape}")

def check_pc():
    print("Checking Clifford-PC...")
    sig_g = torch.tensor([1.0, 1.0])
    layer_dims = [4, 8, 16] # input, hidden1, hidden2
    pc = CliffordPC(layer_dims, sig_g)
    x = torch.randn(2, 4, 4)
    states = pc(x, n_iter=5)
    print(f"  PC states count: {len(states)}")
    for i, s in enumerate(states):
        print(f"    Layer {i} shape: {s.shape}")

def check_tp():
    print("Checking Clifford-TP...")
    sig_g = torch.tensor([1.0, 1.0])
    layer_dims = [4, 8, 2] # input, hidden, output
    tp = CliffordTP(layer_dims, sig_g)
    x = torch.randn(2, 4, 4)
    activations = tp(x)
    global_target = torch.randn(2, 2, 4)
    targets = tp.compute_targets(activations, global_target)
    print(f"  TP targets count: {len(targets)}")
    for i, t in enumerate(targets):
        print(f"    Layer {i} target shape: {t.shape}")

if __name__ == "__main__":
    check_sparse()
    check_pc()
    check_tp()
