import numpy as np
from scipy import sparse
from scipy.sparse import linalg as sla

N_base = 25       # Tamaño de la base (N x N en 2D)
alpha_val = 0.5   # Fuerza del acoplamiento 
n_estados = 9     # Cantidad de estados a calcular

#####################FUNCIONES#####################

# Creación de matrices en 1D (Base del oscilador armónico)
def bloques_1d(N):
    n_idx = np.arange(1, N)
    a = np.diag(np.sqrt(n_idx), k=1)
    ad = a.conj().T
    x = (1/np.sqrt(2)) * (ad + a)
    p = (1j/np.sqrt(2)) * (ad - a)
    return sparse.csr_matrix(x), sparse.csr_matrix(p)

# Expansión a 2D usando Producto de Kronecker
def hamiltoniano_2d(x_s, p_s, alpha, N):
    I = sparse.eye(N)

    # Proyectando a 2 dimensiones
    x1 = sparse.kron(x_s, I)
    x2 = sparse.kron(I, x_s)
    p1 = sparse.kron(p_s, I)
    p2 = sparse.kron(I, p_s)
    
    # Construcción del Hamiltoniano H = T + V
    T = (p1**2 + p2**2) / 2
    V = (x1**2 + x2**2) / 2 + alpha * (x1**2 @ x2**2)
    return T + V

#####################CODIGO#####################

print(f"--- Iniciando Solver Numérico ---")
print(f"Parámetros: N={N_base}, alpha={alpha_val}")

# 1. Construir operadores
x_s, p_s = bloques_1d(N_base)

# 2. Armar el Hamiltoniano
H_total = hamiltoniano_2d(x_s, p_s, alpha_val, N_base)
print(f"Matriz del Hamiltoniano construida con tamaño: {H_total.shape}")

# 3. Diagonalización numérica (Resolviendo la ecuación de Schrödinger) Y eigsh extrae los 'k' autovalores más bajos 
evals, evecs = sla.eigsh(H_total, k=n_estados, which='SA')

# 4. Ordenar los resultados de menor a mayor energía
idx = evals.argsort()
E = evals[idx]
Psi_v = evecs[:, idx] 

#####################RESULTADOS#####################

print("Primeros niveles de energía calculados:")
for i in range(n_estados):
    print(f"Estado {i}: Energía = {E[i]:.4f}")
