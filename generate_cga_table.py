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
    table_cl41 = generate_clifford_multiplication_table(4, 1)
    output_path_cl41 = "cliffeq/algebra/cl41_table.pt"
    os.makedirs(os.path.dirname(output_path_cl41), exist_ok=True)
    torch.save(torch.from_numpy(table_cl41).float(), output_path_cl41)
    print(f"Cl(4,1) multiplication table saved to {output_path_cl41}")
    print(f"Shape: {table_cl41.shape}")

    # Cl(1, 3) -> 4D, 16 blades
    table_cl13 = generate_clifford_multiplication_table(1, 3)
    output_path_cl13 = "cliffeq/algebra/cl13_table.pt"
    torch.save(torch.from_numpy(table_cl13).float(), output_path_cl13)
    print(f"Cl(1,3) multiplication table saved to {output_path_cl13}")
    print(f"Shape: {table_cl13.shape}")
