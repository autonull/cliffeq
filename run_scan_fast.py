import experiments.p3_l2_scan_compositional as scan
import torch
import json
import os

def run_fast():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Only use a subset of the data for faster testing
    train_loader, test_loader, w2i_c, w2i_a, mlc, mla = scan.load_scan_data('length', 64)

    # Take first 500 samples for training
    train_subset = torch.utils.data.Subset(train_loader.dataset, range(500))
    train_loader = torch.utils.data.DataLoader(train_subset, batch_size=64, shuffle=True)

    # Take first 100 samples for testing
    test_subset = torch.utils.data.Subset(test_loader.dataset, range(100))
    test_loader = torch.utils.data.DataLoader(test_subset, batch_size=64)

    sos_idx = w2i_a['<SOS>']
    eos_idx = w2i_a['<EOS>']
    pad_idx = w2i_a['<PAD>']

    results = {}

    # Reduced number of layers and hidden dim for speed
    # Clifford dim 8 means 8 features per head, Cl(2,0) means 4 blades.
    # d_model = nhead * clifford_dim * n_blades = 4 * 8 * 4 = 128
    d_model = 128
    nhead = 4
    clifford_dim = 8
    num_layers = 1

    configs = [
        ("Standard", scan.StandardScanModel(len(w2i_c), len(w2i_a), d_model=d_model, nhead=nhead, num_layers=num_layers)),
        ("Clifford-NoBias", scan.CliffordScanModel(len(w2i_c), len(w2i_a), d_model=d_model, nhead=nhead, clifford_dim=clifford_dim, use_orientation_bias=False, num_layers=num_layers)),
        ("Clifford-Bias", scan.CliffordScanModel(len(w2i_c), len(w2i_a), d_model=d_model, nhead=nhead, clifford_dim=clifford_dim, use_orientation_bias=True, num_layers=num_layers))
    ]

    n_epochs = 20
    for name, model in configs:
        print(f"\nTraining model: {name}")
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = torch.nn.CrossEntropyLoss(ignore_index=pad_idx)

        for epoch in range(n_epochs):
            loss = scan.train_epoch(model, train_loader, optimizer, criterion, device)
            if (epoch + 1) % 5 == 0:
                print(f"  Epoch {epoch+1}, Loss: {loss:.4f}")

        acc = scan.evaluate(model, test_loader, mla, sos_idx, eos_idx, pad_idx, device)
        print(f"  Final Test Accuracy: {acc:.4f}")
        results[name] = acc

    os.makedirs("results", exist_ok=True)
    with open("results/p3_l2_scan_results.json", "w") as f:
        json.dump({"length": results}, f)

if __name__ == "__main__":
    run_fast()
