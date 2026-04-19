import numpy as np
import matplotlib.pyplot as plt
import time

# =====================================================================
# FUNCIÓN PRINCIPAL: DFT MANUAL (Se define una sola vez)
# =====================================================================
def dft_manual(x):
    """Calcula la DFT usando la definición estricta (dos ciclos for)."""
    N = len(x)
    X_k = np.zeros(N, dtype=complex)
    for k in range(N):
        for n in range(N):
            exponente = -1j * 2 * np.pi * k * n / N
            X_k[k] += x[n] * np.exp(exponente)
    return X_k

# =====================================================================
# INCISOS (A), (B) y (C): GENERACIÓN DE SEÑAL Y ESPECTRO
# =====================================================================
print("==================================================")
print("1. GENERANDO SEÑAL Y CALCULANDO ESPECTRO (Incisos a, b, c)")
print("==================================================")

N_puntos = 100        
f_muestreo = 100      
f1, f2 = 10, 25               

t = np.arange(N_puntos) / f_muestreo
x_n = np.sin(2 * np.pi * f1 * t) + 0.5 * np.sin(2 * np.pi * f2 * t)

# Cálculo del espectro
X_k_manual = dft_manual(x_n)
magnitud_X = np.abs(X_k_manual)
frecuencias = np.arange(N_puntos) * (f_muestreo / N_puntos)

# Gráfico del espectro
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(t, x_n, marker='.', linestyle='-', color='b')
plt.title('Señal Original en el Tiempo')
plt.xlabel('Tiempo $t_n$ (s)')
plt.ylabel('Amplitud $x_n$')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.stem(frecuencias, magnitud_X, basefmt="k-")
plt.title('Espectro de Frecuencias $|X_k|$')
plt.xlabel('Frecuencia (Hz)')
plt.ylabel('Magnitud $|X_k|$')
plt.xlim(0, f_muestreo / 2) 
plt.grid(True)
plt.tight_layout()

print(">> Mostrando gráfico del espectro. CIERRA LA VENTANA para continuar con la simulación...")
plt.show()

# =====================================================================
# INCISOS (D), (E) y (F): MEDICIÓN DE TIEMPOS Y GRÁFICO LINEAL
# =====================================================================
print("\n==================================================")
print("2. INICIANDO COMPETENCIA DE ALGORITMOS (Incisos d, e, f)")
print("==================================================")

# Usamos valores intermedios para un buen ajuste sin esperar horas
N_values = np.array([100, 500, 1000, 2000, 5000, 10000])
tiempos_dft = []
tiempos_fft = []

for N in N_values:
    print(f"Procesando señal de tamaño N = {N}...")
    t_n = np.linspace(0, 1, N, endpoint=False)
    x_test = np.sin(2 * np.pi * f1 * t_n) + 0.5 * np.sin(2 * np.pi * f2 * t_n)
    
    # Tiempo FFT (NumPy) - USANDO PERF_COUNTER
    start_fft = time.perf_counter()
    np.fft.fft(x_test)
    tiempos_fft.append(time.perf_counter() - start_fft)
    
    # Tiempo DFT (Manual) - USANDO PERF_COUNTER
    start_dft = time.perf_counter()
    dft_manual(x_test)
    tiempos_dft.append(time.perf_counter() - start_dft)

tiempos_fft = np.array(tiempos_fft)
tiempos_dft = np.array(tiempos_dft)

# Gráfico Lineal (Inciso f)
plt.figure(figsize=(9, 6))
plt.plot(N_values, tiempos_dft, marker='o', linestyle='-', color='red', label='DFT Manual $\mathcal{O}(N^2)$')
plt.plot(N_values, tiempos_fft, marker='s', linestyle='-', color='blue', label='FFT NumPy $\mathcal{O}(N \log N)$')
plt.xlabel('Tamaño de la señal $N$')
plt.ylabel('Tiempo de ejecución (segundos)')
plt.title('Comparación de Rendimiento Lineal: DFT vs FFT')
plt.legend()
plt.grid(True)

print("\n>> Mostrando gráfico lineal de tiempos. CIERRA LA VENTANA para ver el ajuste logarítmico...")
plt.show()

# =====================================================================
# INCISOS (G) y (H): AJUSTE LOG-LOG Y PREDICCIONES
# =====================================================================
print("\n==================================================")
print("3. ANÁLISIS DE ESCALAMIENTO Y PREDICCIÓN (Incisos g, h)")
print("==================================================")

log_N = np.log(N_values)
m_dft, c_dft = np.polyfit(log_N, np.log(tiempos_dft), 1)
m_fft, c_fft = np.polyfit(log_N, np.log(tiempos_fft), 1)

print(f"Exponente empírico DFT: {m_dft:.2f} (Teórico es ~2.00)")
print(f"Exponente empírico FFT: {m_fft:.2f} (Teórico es ~1.00 - 1.20)")

# Gráfico Log-Log (Inciso g)
plt.figure(figsize=(9, 6))
plt.loglog(N_values, tiempos_dft, marker='o', linestyle='', color='red', label=f'Datos DFT')
plt.loglog(N_values, tiempos_fft, marker='s', linestyle='', color='blue', label=f'Datos FFT')

N_linea = np.linspace(100, 100000, 100)
plt.loglog(N_linea, np.exp(m_dft * np.log(N_linea) + c_dft), 'r--', label=f'Ajuste DFT (pendiente={m_dft:.2f})')
plt.loglog(N_linea, np.exp(m_fft * np.log(N_linea) + c_fft), 'b--', label=f'Ajuste FFT (pendiente={m_fft:.2f})')
plt.xlabel('Tamaño de la señal $N$ (escala log)')
plt.ylabel('Tiempo de ejecución (segundos) (escala log)')
plt.title('Escalamiento de Algoritmos (Gráfico Log-Log)')
plt.legend()
plt.grid(True, which="both", ls="--", alpha=0.5)

# Cálculo crítico (Inciso h)
ln_N_critico = (np.log(100) + c_fft - c_dft) / (m_dft - m_fft)
N_critico = np.exp(ln_N_critico)

print(f"\nLa FFT se vuelve al menos 100 veces más rápida que la DFT directa")
print(f"para un tamaño de señal aproximado de N = {int(N_critico)} puntos.")

print("\n>> Mostrando gráfico log-log final. ¡Trabajo completado!")
plt.show()

