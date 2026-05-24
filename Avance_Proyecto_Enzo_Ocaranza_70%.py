import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.sparse import linalg as sla
from scipy.special import eval_hermite
from math import factorial


N_base = 25         # Tamaño de la base (N x N en 2D)
alpha_val = 0.5     # Fuerza del acoplamiento 
n_estados = 9       # Cantidad de estados a calcular
limite_espacial = 4 # Rango del gráfico 

# Creación de matrices en 1D (Base del oscilador armónico)
def obtener_bloques_1d(N):
    n_idx = np.arange(1, N)
    a = np.diag(np.sqrt(n_idx), k=1)
    ad = a.conj().T
    x = (1/np.sqrt(2)) * (ad + a)
    p = (1j/np.sqrt(2)) * (ad - a)
    return sparse.csr_matrix(x), sparse.csr_matrix(p)

# Expansión a 2D usando Producto de Kronecker
def armar_hamiltoniano_2d(x_s, p_s, alpha, N):
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

# Calcula la base del oscilador armónico
def phi_n_normalizada(n, z):
    n_f = float(factorial(n))
    norm = 1.0 / np.sqrt((2**n) * n_f * np.sqrt(np.pi))
    return norm * eval_hermite(n, z) * np.exp(-z**2 / 2)

#####################CODIGO#####################

print(f"--- Iniciando Solver Numérico ---")
print(f"Parámetros: N={N_base}, alpha={alpha_val}")

# 1. Construir operadores
x_s, p_s = obtener_bloques_1d(N_base)

# 2. Armar el Hamiltoniano
H_total = armar_hamiltoniano_2d(x_s, p_s, alpha_val, N_base)
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

#####################GRAFICOS#####################
cols = 3
filas = int(np.ceil(n_estados / cols))

fig, axes = plt.subplots(filas, cols, figsize=(5*cols, 5*filas))
axes = np.atleast_1d(axes).flatten() 

x_grid = np.linspace(-limite_espacial, limite_espacial, 100)
X, Y = np.meshgrid(x_grid, x_grid)

for i in range(n_estados):
    C_nm = Psi_v[:, i].reshape(N_base, N_base)
    
    psi_xy = np.zeros_like(X, dtype=complex)
    for n in range(N_base):
        phi_nx = phi_n_normalizada(n, X)
        for m in range(N_base):
            if abs(C_nm[n, m]) > 1e-4:
                psi_xy += C_nm[n, m] * phi_nx * phi_n_normalizada(m, Y)
    
    z_limit = np.max(np.abs(psi_xy.real))
    im = axes[i].pcolormesh(X, Y, psi_xy.real, cmap='RdBu_r', 
                            vmin=-z_limit, vmax=z_limit, shading='auto')
    
    axes[i].set_title(f"Estado {i}: E={E[i]:.3f}")
    axes[i].set_aspect('equal')
    plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)

for j in range(i + 1, len(axes)):
    axes[j].axis('off')

plt.tight_layout()
plt.show()