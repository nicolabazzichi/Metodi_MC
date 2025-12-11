import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# --- I TUOI PARAMETRI ---
L = 2*np.pi
T = 40 * np.pi
nu = 1e-5
nt = 20000
nx = 200
dt = T/nt
print(f"dt: {dt}")
dx = L/nx
x = np.linspace(0, L, nx, endpoint=False)

def func(x, nx):
    u = np.sin(x)
    # Aggiungo rumore (frequenze alte random)
    for k in range(2, int(nx/2)):
        u += np.sin(k * x + np.random.rand() * L) 
    return u / 10
    #return np.sin(x + np.random.normal(0, 0.5, size=len(x)))

f = func(x, nx) 

def calculate_rhs(u_state, nx, dx, nu):
    u_ip = np.roll(u_state, -1) 
    u_im = np.roll(u_state, 1)  
    
    uc = (u_ip - u_im) / (2 * dx)
    uii = (u_ip - 2 * u_state + u_im) / (dx**2.0)
    
    # NOTA: Qui stai usando l'equazione LINEARE (Advection-Diffusion)
    # Se volevi Burgers (non-lineare) dovresti scrivere: u_state * (-uc)
    rhs = (-uc) + nu * uii
    return rhs

# --- IMPOSTAZIONI ANIMAZIONE ---
DURATA_DESIDERATA_SEC = 15
FPS = 60
numero_frame_totali = int(DURATA_DESIDERATA_SEC * FPS)
steps_per_frame = int(nt / numero_frame_totali)
intervallo_ms = 1000 / FPS 

# --- PREPARAZIONE DATI FOURIER (LOG-LOG SETUP) ---
k_values = np.fft.fftfreq(nx, d=dx) * 2 * np.pi
k_values = np.fft.fftshift(k_values) 

# 1. Creiamo un filtro per prendere solo le frequenze positive (k > 0)
idx_pos = k_values > 0 
k_pos = k_values[idx_pos]

# --- PREPARAZIONE GRAFICO ---
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))
plt.subplots_adjust(hspace=0.4) 

# -- Grafico 1: Spazio Fisico --
ax1.set_xlim(0, L)
ax1.set_ylim(-1.5, 1.5)
ax1.grid(alpha=0.3)
ax1.set_title("Spazio Fisico: u(x, t)")
ax1.set_xlabel("Posizione x")
ax1.plot(x, f, c='black', linestyle='--', label='Start', alpha=0.5)
line1, = ax1.plot(x, f, c='green', label='u(x,t)')
time_text = ax1.text(0.02, 0.95, '', transform=ax1.transAxes)
ax1.legend(loc='upper right')

# -- Grafico 2: Spazio di Fourier (LOG-LOG) --
# 2. Impostiamo le scale logaritmiche
ax2.set_xscale('log')
ax2.set_yscale('log')

# Limiti assi basati sui k positivi
ax2.set_xlim(k_pos.min(), k_pos.max())
ax2.set_ylim(1e-6, 1.0) # Imposta un minimo sensato per evitare log(0)
ax2.grid(True, which="both", ls="-", alpha=0.3)
ax2.set_title("Spettro di Fourier (Log-Log)")
ax2.set_xlabel("Numero d'onda k")
ax2.set_ylabel("|FFT|")

# Calcolo FFT iniziale e filtro
fft_init_full = np.abs(np.fft.fftshift(np.fft.fft(f))) / nx
fft_init_pos = fft_init_full[idx_pos] # Prendo solo i positivi

line2, = ax2.plot(k_pos, fft_init_pos, c='red', marker='.', linestyle='-')

# Variabili di stato
u_old = f.copy()
current_step = 0

# --- FUNZIONE UPDATE ---
def update(frame):
    global u_old, current_step
    
    # Avanzamento Fisica
    for _ in range(steps_per_frame):
        if current_step >= nt: break
        
        # RK4 Integration
        u_state = u_old.copy() # Copia di sicurezza
        for l in range(4, 0, -1): # Range corretto include 1
            F = calculate_rhs(u_state, nx, dx, nu)
            u_state = u_old + (1 / l) * dt * F 
        u_old = u_state
        current_step += 1
    
    # 1. Aggiorno grafico Fisico
    line1.set_ydata(u_old)
    time_text.set_text(f"Step: {current_step}/{nt}")
    
    # 2. Aggiorno grafico Fourier (Log-Log)
    fft_full = np.abs(np.fft.fftshift(np.fft.fft(u_old))) / nx
    
    # Filtro solo i dati positivi per il plot logaritmico
    fft_pos_data = fft_full[idx_pos]
    
    # Protezione anti-crash: sostituisco zeri assoluti con un numero piccolissimo
    fft_pos_data = np.maximum(fft_pos_data, 1e-20)
    
    line2.set_ydata(fft_pos_data)
    
    return line1, line2, time_text

# --- AVVIO ANIMAZIONE ---
anim = FuncAnimation(
    fig, update, frames=numero_frame_totali, 
    interval=intervallo_ms, blit=True
)

plt.show()