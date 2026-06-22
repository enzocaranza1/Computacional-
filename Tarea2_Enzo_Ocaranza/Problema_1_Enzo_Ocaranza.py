import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error




np.random.seed(42)

# Parámetros base
N_senales = 3000
N_t = 1000
t = np.linspace(0, 10, N_t)

# Generación de parámetros físicos (Etiquetas)
gamma = np.random.uniform(0.05, 1.0, size=N_senales)
k = np.random.uniform(1.0, 5.0, size=N_senales)
y = np.column_stack((gamma, k))

# Cálculo vectorizado de la señal pura (sin ruido)
w_d = np.sqrt(k - (gamma**2) / 4.0)
gamma_ext = gamma[:, np.newaxis]
w_d_ext = w_d[:, np.newaxis]
t_ext = t[np.newaxis, :]

x_puro = np.exp(-gamma_ext * t_ext / 2.0) * (
    np.cos(w_d_ext * t_ext) + (gamma_ext / (2.0 * w_d_ext)) * np.sin(w_d_ext * t_ext)
)

# GRÁFICO CON RUIDO ESTÁNDAR (sigma = 0.02)

X_obs_02 = x_puro + np.random.normal(0, 0.02, size=(N_senales, N_t))

indices = [
    np.argmin(k + gamma), np.argmax(k - gamma),
    np.argmax(gamma - k), np.argmax(k + gamma)
]
titulos = [
    f"Baja Freq / Lento Decaimiento\n($k={k[indices[0]]:.2f}, \\gamma={gamma[indices[0]]:.2f}$)",
    f"Alta Freq / Lento Decaimiento\n($k={k[indices[1]]:.2f}, \\gamma={gamma[indices[1]]:.2f}$)",
    f"Baja Freq / Rápido Decaimiento\n($k={k[indices[2]]:.2f}, \\gamma={gamma[indices[2]]:.2f}$)",
    f"Alta Freq / Rápido Decaimiento\n($k={k[indices[3]]:.2f}, \\gamma={gamma[indices[3]]:.2f}$)",
]

fig_b, axs = plt.subplots(2, 2, figsize=(10, 6), sharex=True, sharey=True)
for i, idx in enumerate(indices):
    axs.flatten()[i].plot(t, x_puro[idx], "k--", alpha=0.7, label="Señal pura")
    axs.flatten()[i].plot(t, X_obs_02[idx], "r-", alpha=0.8, lw=1, label="Con ruido")
    axs.flatten()[i].set_title(titulos[i], fontsize=10)
    axs.flatten()[i].grid(True, linestyle=":", alpha=0.6)

axs.flatten()[0].legend()
plt.tight_layout()
print("CIERRA LA VENTANA PARA CONTINUAR")
plt.show()


# BARRIDO DE RUIDO Y ENTRENAMIENTO
sigmas = [0, 0.01, 0.02, 0.05, 0.10]

# Listas para guardar los resultados y poder graficarlos
rmse_rf_train, rmse_rf_test = [], []
rmse_mlp_train, rmse_mlp_test = [], []

for s in sigmas:
    print(f"Procesando ruido sigma = {s}")
    
    # 1. Contaminar la señal pura
    ruido = np.random.normal(0, s, size=(N_senales, N_t))
    X_ruidoso = x_puro + ruido
    
    # 2. Separar train/test
    X_train, X_test, y_train, y_test = train_test_split(X_ruidoso, y, test_size=0.20, random_state=42)
    
    # 3. Entrenar Random Forest
    rf = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_pred_train = rf.predict(X_train)
    rf_pred_test = rf.predict(X_test)
    
    # 4. Entrenar Red Neuronal
    mlp = MLPRegressor(hidden_layer_sizes=(64, 32), activation='relu', solver='adam', max_iter=300, random_state=42)
    mlp.fit(X_train, y_train)
    mlp_pred_train = mlp.predict(X_train)
    mlp_pred_test = mlp.predict(X_test)
    
    # 5. Calcular errores (RMSE) para ambos parámetros simultáneamente
    error_rf_train = np.sqrt(mean_squared_error(y_train, rf_pred_train, multioutput='raw_values'))
    error_rf_test = np.sqrt(mean_squared_error(y_test, rf_pred_test, multioutput='raw_values'))
    error_mlp_train = np.sqrt(mean_squared_error(y_train, mlp_pred_train, multioutput='raw_values'))
    error_mlp_test = np.sqrt(mean_squared_error(y_test, mlp_pred_test, multioutput='raw_values'))
    
    rmse_rf_train.append(error_rf_train)
    rmse_rf_test.append(error_rf_test)
    rmse_mlp_train.append(error_mlp_train)
    rmse_mlp_test.append(error_mlp_test)
    
    if s == 0.02:
        print(f"RF Test RMSE : Gamma={error_rf_test[0]:.4f}, K={error_rf_test[1]:.4f}")
        print(f"MLP Test RMSE: Gamma={error_mlp_test[0]:.4f}, K={error_mlp_test[1]:.4f}")

# Convertir listas a arreglos de numpy para graficar 
rmse_rf_train = np.array(rmse_rf_train)
rmse_rf_test = np.array(rmse_rf_test)
rmse_mlp_train = np.array(rmse_mlp_train)
rmse_mlp_test = np.array(rmse_mlp_test)

# RMSE vs SIGMA
fig_d, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Subplot 1: Error en Gamma
ax1.plot(sigmas, rmse_rf_train[:, 0], 'r--', label='RF (Train)')
ax1.plot(sigmas, rmse_rf_test[:, 0], 'r-o', label='RF (Test)')
ax1.plot(sigmas, rmse_mlp_train[:, 0], 'b--', label='MLP (Train)')
ax1.plot(sigmas, rmse_mlp_test[:, 0], 'b-s', label='MLP (Test)')
ax1.set_xlabel('Nivel de Ruido ($\sigma$)')
ax1.set_ylabel('RMSE en $\gamma$')
ax1.set_title('Efecto del Ruido al inferir el Roce ($\gamma$)')
ax1.legend()
ax1.grid(True)

# Subplot 2: Error en K
ax2.plot(sigmas, rmse_rf_train[:, 1], 'r--', label='RF (Train)')
ax2.plot(sigmas, rmse_rf_test[:, 1], 'r-o', label='RF (Test)')
ax2.plot(sigmas, rmse_mlp_train[:, 1], 'b--', label='MLP (Train)')
ax2.plot(sigmas, rmse_mlp_test[:, 1], 'b-s', label='MLP (Test)')
ax2.set_xlabel('Nivel de Ruido ($\sigma$)')
ax2.set_ylabel('RMSE en $k$')
ax2.set_title('Efecto del Ruido al inferir la Frecuencia ($k$)')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.show()
