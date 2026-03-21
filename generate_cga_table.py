import clifford
import torch
import numpy as np
import os

def generate_clifford_multiplication_table(p, q):
    layout, blades = clifford.Cl(p, q)
    dim = layout.gaDims
    table = np.zeros((dim, dim, dim))

    # Construct basis elements
    basis = []
    for i in range(dim):
        coeffs = np.zeros(dim)
        coeffs[i] = 1.0
        basis.append(layout.MultiVector(coeffs))

    for i in range(dim):
        for j in range(dim):
            prod = basis[i] * basis[j]
            table[i, j, :] = prod.value

    return table

if __name__ == "__main__":
    # Cl(4, 1) -> 5D, 32 blades
    table = generate_clifford_multiplication_table(4, 1)
    # Save as torch tensor
    output_path = "cliffeq/algebra/cl41_table.pt"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(torch.from_numpy(table).float(), output_path)
    print(f"Cl(4,1) multiplication table saved to {output_path}")
    print(f"Shape: {table.shape}")
