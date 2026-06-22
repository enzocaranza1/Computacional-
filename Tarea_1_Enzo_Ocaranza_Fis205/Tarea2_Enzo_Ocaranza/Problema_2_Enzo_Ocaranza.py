import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import quad

np.random.seed(42)


m_e_c2 = 0.51099895  # MeV
m_p_c2 = 938.272088  # MeV
I_agua = 75.0e-6  # MeV 
Z_agua, A_agua = 10.0, 18.01528
K = 0.307075  # MeV cm^2 / mol


def stopping_power(E):
    
    if E < 0.05:  # Valor de seguridad para evitar divergencia a baja energía
        return stopping_power(0.05) * np.sqrt(0.05 / E)

    gamma = 1.0 + (E / m_p_c2)
    beta_sq = 1.0 - (1.0 / (gamma**2))
    T_max = (2.0 * m_e_c2 * beta_sq * (gamma**2)) / (
        1.0 + 2.0 * gamma * (m_e_c2 / m_p_c2) + (m_e_c2 / m_p_c2) ** 2
    )

    arg_log = (2.0 * m_e_c2 * beta_sq * T_max) / (I_agua**2)
    L = 0.5 * np.log(arg_log) - beta_sq
    return K * (Z_agua / A_agua) * (1.0 / beta_sq) * L


def calcular_rango_csda(E0):
    res, _ = quad(lambda E: 1.0 / stopping_power(E), 1e-3, E0)
    return res

nist_ref = {50.0: 2.227, 150.0: 15.77, 250.0: 37.90}
for E_test in [50.0, 150.0, 250.0]:
    r_calc = calcular_rango_csda(E_test)
    err = abs(r_calc - nist_ref[E_test]) / nist_ref[E_test] * 100
    print(
        f" E0 = {E_test:3.0f} MeV | Nuestro: {r_calc:.4f} cm | NIST: {nist_ref[E_test]:.4f} cm | Error: {err:.2f}%"
    )



N_protones = 10000
E0_sim = 150.0  # MeV
dx = 0.01  # cm 
z_max = 18.0  # cm 

z_bins = np.arange(0, z_max + dx, dx)
z_centros = z_bins[:-1] + dx / 2.0
num_bins = len(z_centros)

R_csda_150 = calcular_rango_csda(E0_sim)

Dose_det = np.zeros(num_bins)
E_det = E0_sim

for idx in range(num_bins):
    if E_det <= 0:
        break
    dE = stopping_power(E_det) * dx

    if E_det - dE < 0:
        dE = E_det  
        Dose_det[idx] += dE
        break

    Dose_det[idx] += dE
    E_det -= dE

Dose_det = Dose_det / dx  # Normalizamos a [MeV/cm]

Dose_stoch = np.zeros(num_bins)
posiciones_de_frenado = []

cte_bohr = K * m_e_c2 * (Z_agua / A_agua) * 1.0  

for i in range(1, N_protones + 1):
    if i % 2500 == 0:
        print(f"    Procesando protón {i} de {N_protones} ({i/N_protones*100:.0f}%)...")

    E_p = E0_sim
    for idx in range(num_bins):
        if E_p <= 0:
            posiciones_de_frenado.append(z_centros[idx])
            break

        # Cinemática del paso
        gamma_p = 1.0 + (E_p / m_p_c2)
        beta_sq_p = 1.0 - (1.0 / (gamma_p**2))

        # 1. Pérdida media 
        dE_mean = stopping_power(E_p) * dx

        # 2. Desviación estándar de Bohr (sigma_E)
        sigma_E = np.sqrt(cte_bohr * (1.0 / beta_sq_p) * dx)

        # 3. Muestreo estocástico de la energía perdida en este paso
        dE_rand = np.random.normal(dE_mean, sigma_E)
        dE_rand = max(0.0, dE_rand)  # Imponemos que no puede "ganar" energía

        if E_p - dE_rand <= 0:
            dE_rand = E_p
            Dose_stoch[idx] += dE_rand
            posiciones_de_frenado.append(z_centros[idx])
            break

        Dose_stoch[idx] += dE_rand
        E_p -= dE_rand

# Normalizamos la dosis Monte Carlo:
Dose_stoch = Dose_stoch / (N_protones * dx)

# Cálculo estadístico del ensanchamiento del peak (sigma_R)
posiciones_de_frenado = np.array(posiciones_de_frenado)
R_mean_stoch = np.mean(posiciones_de_frenado)
sigma_R = np.std(posiciones_de_frenado)


print(f" Rango CSDA teórico         : {R_csda_150:.4f} cm")
print(f" Peak Determinista máximo   : {z_centros[np.argmax(Dose_det)]:.4f} cm")
print(f" Peak Estocástico medio    : {R_mean_stoch:.4f} cm")
print(f" Ensanchamiento del Pico  : {sigma_R:.4f} cm ")


plt.figure(figsize=(10, 5.5))

# Curva ideal
plt.plot(
    z_centros,
    Dose_det,
    color="black",
    linestyle="--",
    linewidth=1.5,
    label="Haz Determinista (CSDA puro, sin Straggling)",
)

# Curva real
plt.plot(
    z_centros,
    Dose_stoch,
    color="crimson",
    linewidth=2.0,
    alpha=0.85,
    label=f"Haz Monte Carlo ($10^4$ protones, $\\sigma_R = {sigma_R*10:.2f}$ mm)",
)

# Línea vertical del rango teórico
plt.axvline(
    R_csda_150,
    color="blue",
    linestyle=":",
    linewidth=1.2,
    label=f"$R_{{CSDA}} = {R_csda_150:.2f}$ cm",
)

plt.title(
    "Deposición de Dosis en Agua: Caracterización del Pico de Bragg ($E_0 = 150$ MeV)",
    fontsize=11,
    fontweight="bold",
)
plt.xlabel("Profundidad en el fantoma $z$ [cm]", fontsize=10)
plt.ylabel("Dosis / Stopping Power Electrónico [MeV/cm]", fontsize=10)
plt.xlim(0, 17.0)
plt.ylim(0, max(np.max(Dose_det), np.max(Dose_stoch)) * 1.05)
plt.legend(loc="upper left", fontsize=9)
plt.grid(True, linestyle=":", alpha=0.6)

plt.tight_layout()
plt.show()