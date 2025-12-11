import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# --- I TUOI PARAMETRI ---
L = 2 * np.pi
T = 10 * np.pi
nu = 1e-2    # Viscosità sufficiente per smussare lo shock
nt = 5000
nx = 200
dt = T / nt
dx = L / nx
x = np.linspace(0, L, nx, endpoint=False)

# --- FUNZIONE INIZIALE ---
def func(x):
    # Parte con un semplice seno, ma puoi decommentare la parte random se vuoi
    u = np.sin(x)
    return 0.5 * u 

f = func(x) 

# --- CALCOLO RHS (Equazione di Burgers) ---
def calculate_rhs(u_state, nx, dx, nu):
    # u_t = -u * u_x + nu * u_xx
    
    u_ip = np.roll(u_state, -1) # u_{i+1}
    u_im = np.roll(u_state, 1)  # u_{i-1}
    
    # Derivata prima (centrata)
    uc = (u_ip - u_im) / (2 * dx)
    
    # Derivata seconda (centrata)
    uii = (u_ip - 2 * u_state + u_im) / (dx**2.0)
    
    # NOTA: Qui c'è u_state * (-uc), che rende l'equazione NON LINEARE (Burgers)
    rhs = u_state * (-uc) + nu * uii
    return rhs

# --- IMPOSTAZIONI ANIMAZIONE ---
DURATA_DESIDERATA_SEC = 10
FPS = 30
numero_frame_totali = int(DURATA_DESIDERATA_SEC * FPS)
steps_per_frame = int(nt / numero_frame_totali)
intervallo_ms = 1000 / FPS 

# --- PREPARAZIONE DATI FOURIER ---
k_values = np.fft.fftfreq(nx, d=dx) * 2 * np.pi
k_values = np.fft.fftshift(k_values)

# Filtro per LOG-LOG (solo k positivi)
idx_pos = k_values > 0 
k_pos = k_values[idx_pos]

# --- PREPARAZIONE GRAFICO ---
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))
plt.subplots_adjust(hspace=0.4)

# -- Grafico 1: Spazio Fisico --
ax1.set_xlim(0, L)
ax1.set_ylim(-1.0, 1.0) # Adattato all'ampiezza 0.5
ax1.grid(alpha=0.3)
ax1.set_title(f"Burgers Equation (Viscosità={nu})")
ax1.set_xlabel("x")
ax1.plot(x, f, c='black', linestyle='--', label='Start', alpha=0.5)
line1, = ax1.plot(x, f, c='green', label='u(x,t)')
time_text = ax1.text(0.02, 0.95, '', transform=ax1.transAxes)
ax1.legend(loc='upper right')

# -- Grafico 2: Spazio di Fourier (Log-Log) --
ax2.set_xscale('log')
ax2.set_yscale('log')
ax2.set_xlim(k_pos.min(), k_pos.max())
ax2.set_ylim(1e-10, 1.0) 
ax2.grid(True, which="both", ls="-", alpha=0.3)
ax2.set_title("Spettro di Energia (Log-Log)")
ax2.set_xlabel("Numero d'onda k")
ax2.set_ylabel("|FFT|")

# Calcolo FFT iniziale
fft_full = np.abs(np.fft.fftshift(np.fft.fft(f))) / nx
fft_pos = fft_full[idx_pos]
line2, = ax2.plot(k_pos, fft_pos, c='red', marker='.', linestyle='-')

# Aggiungo una linea di riferimento k^-2 (tipica degli shock di Burgers)
ax2.plot(k_pos, 0.1 * k_pos**(-2), 'k--', alpha=0.4, label='Slope $k^{-2}$')
ax2.legend()

# Variabili di stato
u_old = f.copy()
current_step = 0

# --- FUNZIONE UPDATE (CORRETTA) ---
def update(frame):
    global u_old, current_step
    
    # Eseguo N step fisici per ogni frame video
    for _ in range(steps_per_frame):
        if current_step >= nt: break
        
        # --- INIZIO INTEGRATORE RK4 ---
        # Questo è il pezzo che mancava!
        u_state = u_old.copy()
        
        # Ciclo Runge-Kutta Low-Storage (4, 3, 2, 1)
        for l in range(4, 0, -1): 
            F = calculate_rhs(u_state, nx, dx, nu)
            u_state = u_old + (1 / l) * dt * F 
            
        u_old = u_state
        # --- FINE INTEGRATORE ---
        
        current_step += 1
    
    # 1. Aggiorno grafico Fisico
    line1.set_ydata(u_old)
    time_text.set_text(f"Step: {current_step}/{nt}")
    
    # 2. Aggiorno grafico Fourier
    fft_data_full = np.abs(np.fft.fftshift(np.fft.fft(u_old))) / nx
    fft_data_pos = fft_data_full[idx_pos]
    
    # Protezione per il log(0)
    fft_data_pos = np.maximum(fft_data_pos, 1e-20)
    
    line2.set_ydata(fft_data_pos)
    
    return line1, line2, time_text

# --- AVVIO ANIMAZIONE ---
anim = FuncAnimation(
    fig, update, frames=numero_frame_totali, 
    interval=intervallo_ms, blit=True
)

plt.show()