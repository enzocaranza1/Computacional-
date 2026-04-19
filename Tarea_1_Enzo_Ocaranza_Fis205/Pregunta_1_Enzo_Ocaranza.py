import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.linalg import expm, eigh

#CONSTRUCCIÓN SIMBÓLICA DEL HAMILTONIANO 
def build_symbolic_hamiltonian(N):

    J, B = sp.symbols('J B', real=True)
    
    sigma_x = sp.Matrix([[0, 1], [1, 0]])
    sigma_z = sp.Matrix([[1, 0], [0, -1]])
    identity = sp.eye(2)
    
    def tensor_operator_sym(op, position):
        op_list = [identity] * N
        op_list[position] = op
        result = op_list[0]
        for i in range(1, N):
            result = sp.kronecker_product(result, op_list[i])
        return result

    dim = 2**N
    H = sp.zeros(dim, dim)
    
    for i in range(N - 1):
        op1 = tensor_operator_sym(sigma_x, i)
        op2 = tensor_operator_sym(sigma_x, i + 1)
        H += J * (op1 * op2)
        
    for i in range(N):
        H += B * tensor_operator_sym(sigma_z, i)
        
    return H


#CONSTRUCCIÓN NUMÉRICA DEL HAMILTONIANO
sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)
identity = np.eye(2, dtype=complex)

def tensor_operator(op, position, N):
    op_list = [identity] * N
    op_list[position] = op
    result = op_list[0]
    for i in range(1, N):
        result = np.kron(result, op_list[i])
    return result

def build_hamiltonian(N, J, B):
    dim = 2**N
    H = np.zeros((dim, dim), dtype=complex)
    
    for i in range(N - 1):
        op1 = tensor_operator(sigma_x, i, N)
        op2 = tensor_operator(sigma_x, i + 1, N)
        H += J * (op1 @ op2)
        
    for i in range(N):
        H += B * tensor_operator(sigma_z, i, N)
        
    return H


#EVOLUCIÓN TEMPORAL
def simulate_evolution(N=4, J=1.0, B_values=[0.1, 1.0, 10.0], t_max=10.0, dt=0.05):
    t_steps = np.arange(0, t_max, dt)
    plt.figure(figsize=(10, 6))

    state_0 = np.array([1, 0], dtype=complex)
    Psi_0 = state_0
    for _ in range(1, N):
        Psi_0 = np.kron(Psi_0, state_0)

    for B in B_values:
        H = build_hamiltonian(N, J, B)
        U = expm(-1j * H * dt)
        
        p_t = []
        Psi_t = Psi_0.copy()
        
        for t in t_steps:
            prob = np.abs(np.vdot(Psi_0, Psi_t))**2
            p_t.append(prob)
            Psi_t = U @ Psi_t
            
        plt.plot(t_steps, p_t, label=f'B/J = {B/J}')

    plt.xlabel('Tiempo t')
    plt.ylabel('Probabilidad de retorno p(t)')
    plt.title(f'Evolución temporal del estado inicial (N={N})')
    plt.legend()
    plt.grid(True)
    plt.show()


#TIEMPOS Y ESTIMACIONES 
def measure_plot_and_estimate(N_list=[4, 5, 6, 7, 8], J=1.0, B=1.0, realizations=5):
    average_times = []
    
    #Medición de los tiempos
    for N in N_list:
        times_N = []
        for _ in range(realizations):
            start_time = time.time()
            H = build_hamiltonian(N, J, B)
            eigenvalues, eigenvectors = eigh(H)
            end_time = time.time()
            times_N.append(end_time - start_time)
            
        avg_time = np.mean(times_N)
        average_times.append(avg_time)
        print(f"N = {N} -> Tiempo promedio calculado: {avg_time:.6f} segundos")

    N_vals = np.array(N_list)
    t_vals = np.array(average_times)

    #Gráfico de los tiempos obtenidos
    plt.figure(figsize=(8, 5))
    plt.plot(N_vals, t_vals, marker='o', linestyle='-', color='r')
    plt.xlabel('Número de espines N')
    plt.ylabel('Tiempo de ejecución (segundos)')
    plt.title('Tiempo de construcción y diagonalización vs N')
    plt.grid(True)
    
    print("\n[NOTA: Cierra la ventana del gráfico para ver las estimaciones]")
    plt.show() 

    #Ajuste lineal y proyecciones automáticas
    print("\nEstimaciones de tiempo para N grandes")
    log_t = np.log(t_vals)
    m, c = np.polyfit(N_vals, log_t, 1)

    def estimar_tiempo(N_val):
        return np.exp(m * N_val + c)

    N_estimar = [20, 50, 100]
    tiempos_estimados = []

    for N in N_estimar:
        t_est = estimar_tiempo(N)
        tiempos_estimados.append(t_est)
        print(f"Para N = {N:3d}: {t_est:.2e} segundos")

    #Comparación con la edad del universo
    print("\nComparación con la edad del Universo")
    edad_universo = 4.3e17 # en segundos

    for i, N in enumerate(N_estimar):
        t_est = tiempos_estimados[i]
        proporcion = t_est / edad_universo
        
        if proporcion < 1:
            print(f"Para N = {N:3d}: Tomaría una fracción de {proporcion:.2e} de la edad del universo.")
        else:
            print(f"Para N = {N:3d}: Tomaría {proporcion:.2e} veces la edad del universo.")


if __name__ == "__main__":
    print("==================================================")
    print("Matriz del Hamiltoniano para N=2")
    H_simbolico = build_symbolic_hamiltonian(2)
    sp.pprint(H_simbolico)
    
    print("\n==================================================")
    print("Evolución Temporal")
    simulate_evolution()
    
    print("\n==================================================")
    print("Medición y Estimaciones")
    measure_plot_and_estimate()
    print("==================================================")