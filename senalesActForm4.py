import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, cheby1, firwin, lfilter, freqz

# 1. Definición de la señal de entrada
fs = 1000  # Frecuencia de muestreo
t = np.linspace(0, 1, fs, endpoint=False)  # 1 segundo de duración

# Señal compuesta: suma de 50Hz y 200Hz
signal_clean = np.sin(2 * np.pi * 50 * t) + np.sin(2 * np.pi * 200 * t)

# Ruido blanco
noise = np.random.normal(0, 0.5, t.shape)

# Señal con ruido
signal_noisy = signal_clean + noise

# Gráfica de la señal original
plt.figure(figsize=(12, 4))
plt.plot(t, signal_noisy)
plt.title('Señal compuesta con ruido blanco')
plt.xlabel('Tiempo [s]')
plt.ylabel('Amplitud')
plt.grid()
plt.show()

# 2. Diseño de los filtros

# Parámetros comunes
order = 4
cutoff = 100  # Frecuencia de corte en Hz

# --- Butterworth (IIR) ---
b_butter, a_butter = butter(order, cutoff / (fs / 2), btype='low')

# --- Chebyshev tipo I (IIR) ---
b_cheby, a_cheby = cheby1(order, 1, cutoff / (fs / 2), btype='low')

# --- FIR con ventana Hamming ---
numtaps = 101  # Número de coeficientes
b_fir = firwin(numtaps, cutoff / (fs / 2), window='hamming')
a_fir = 1  # FIR: a = 1

# 3. Aplicación de filtros

signal_butter = lfilter(b_butter, a_butter, signal_noisy)
signal_cheby = lfilter(b_cheby, a_cheby, signal_noisy)
signal_fir = lfilter(b_fir, a_fir, signal_noisy)

# 4. Visualización de los resultados

def plot_filtered_signals(original, butter, cheby, fir, t):
    plt.figure(figsize=(15, 10))

    plt.subplot(4, 1, 1)
    plt.plot(t, original)
    plt.title('Señal original con ruido')
    plt.grid()

    plt.subplot(4, 1, 2)
    plt.plot(t, butter)
    plt.title('Filtro Butterworth')
    plt.grid()

    plt.subplot(4, 1, 3)
    plt.plot(t, cheby)
    plt.title('Filtro Chebyshev Tipo I')
    plt.grid()

    plt.subplot(4, 1, 4)
    plt.plot(t, fir)
    plt.title('Filtro FIR con ventana Hamming')
    plt.xlabel('Tiempo [s]')
    plt.grid()

    plt.tight_layout()
    plt.show()

plot_filtered_signals(signal_noisy, signal_butter, signal_cheby, signal_fir, t)
