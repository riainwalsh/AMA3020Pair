
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE   = 14
LABEL  = 16
TITLE  = 18
TICKS  = 14
LEGEND = 13

plt.rcParams.update({
    "font.size": BASE,
    "axes.titlesize": TITLE,
    "axes.labelsize": LABEL,
    "xtick.labelsize": TICKS,
    "ytick.labelsize": TICKS,
    "legend.fontsize": LEGEND,
    "legend.title_fontsize": LEGEND,
    "figure.titlesize": TITLE + 2,
})

print("SCRIPT STARTED", flush=True)
print("CWD:", os.getcwd(), flush=True)

MPH_TO_MS = 0.44704
MS_TO_MPH = 2.2369362920544

rho = 1.225
Cd  = 1.1
A   = 0.5
k   = 0.5 * rho * Cd * A

mass = 80.0
g    = 9.81
Crr  = 0.0025
F_roll = Crr * mass * g

U_total = 3.178454  # m/s  (≈ 7.11 mph)
U_mph = U_total * MS_TO_MPH

F_drive = 45.0  


def vrel_mag(v, U, theta_deg):
    th = np.radians(theta_deg)
    return np.sqrt(v**2 + U**2 - 2*v*U*np.cos(th))

def aero_resist_along(v, U, theta_deg):
    
    th = np.radians(theta_deg)
    return k * vrel_mag(v, U, theta_deg) * (v - U*np.cos(th))

def simulate_coastdown(v0_ms, U, theta_deg, t_end=120.0, dt=0.05, v_floor=0.0):
    """
    Coast-down: m dv/dt = -(F_roll + F_aero_along)
    Returns: t, v, x, a
    """
    n = int(t_end/dt) + 1
    t = np.linspace(0, t_end, n)
    v = np.zeros(n)
    x = np.zeros(n)
    a = np.zeros(n)

    v[0] = v0_ms

    for i in range(n-1):
        vi = max(v[i], 1e-8)
        F_aero = aero_resist_along(vi, U, theta_deg)
        F_tot  = F_roll + F_aero
        ai = -(F_tot) / mass

        a[i] = ai
        v_next = max(v_floor, vi + ai*dt)

        x[i+1] = x[i] + vi*dt
        v[i+1] = v_next

        if v_next <= v_floor + 1e-12:
            a[i+1:] = 0.0
            v[i+1:] = v_floor
            x[i+1:] = x[i+1]
            break

    a[-1] = a[-2]
    return t, v, x, a

angles_coast = [0, 45, 90, 135, 180]
v0_mph_cd = 25.0
t_end  = 120.0
dt     = 0.05

plt.figure()
for ang in angles_coast:
    t, v, x, a = simulate_coastdown(v0_mph_cd*MPH_TO_MS, U_total, ang, t_end=t_end, dt=dt, v_floor=0.0)
    plt.plot(v*MS_TO_MPH, a, label=f"{ang}°")
plt.xlabel("Speed (mph)")
plt.ylabel("Acceleration (m/s²)")
plt.title(f"Coast-down: Acceleration vs Speed (U = {U_mph:.2f} mph)")
plt.grid(True)
plt.legend(title="Wind angle")
plt.tight_layout()
plt.savefig("CoastDown_Acceleration_vs_Speed_Angles.pdf", dpi=1200)
plt.close()

plt.figure()
for ang in angles_coast:
    t, v, x, a = simulate_coastdown(v0_mph_cd*MPH_TO_MS, U_total, ang, t_end=t_end, dt=dt, v_floor=0.0)
    plt.plot(t, x, label=f"{ang}°")
plt.xlabel("Time (s)")
plt.ylabel("Distance (m)")
plt.title(f"Coast-down: Distance vs Time (U = {U_mph:.2f} mph, start {v0_mph_cd:.0f} mph)")
plt.grid(True)
plt.legend(title="Wind angle")
plt.tight_layout()
plt.savefig("CoastDown_Distance_vs_Time_Angles.pdf", dpi=1200)
plt.close()

print("Saved coast-down PDFs.", flush=True)


def simulate_constant_force(theta_deg, t_end=200.0, dt=0.05):
    """
    m dv/dt = F_drive - F_roll - F_aero
    """
    t = np.arange(0, t_end+dt, dt)
    v = np.zeros_like(t)

    for i in range(1, len(t)):
        v_prev = v[i-1]
        F_aero = aero_resist_along(v_prev, U_total, theta_deg)
        F_net  = F_drive - F_roll - F_aero
        a      = F_net / mass
        v[i]   = max(0.0, v_prev + a*dt)

    return t, v

angles_force = [0, 30, 60, 90, 120, 150, 180]
plt.figure()
for ang in angles_force:
    t, v = simulate_constant_force(ang)
    plt.plot(t, v*MS_TO_MPH, label=f"{ang}°")
plt.xlabel("Time (s)")
plt.ylabel("Speed (mph)")
plt.title(f"Velocity vs Time (Constant Force {F_drive:.1f} N, U={U_mph:.2f} mph)")
plt.grid(True)
plt.legend(title="Wind angle")
plt.tight_layout()
plt.savefig("Velocity_vs_Time_ConstantForce.pdf", dpi=1200)
plt.close()

print("Saved Velocity_vs_Time_ConstantForce.pdf", flush=True)

m_report   = 92.2
Crr_report = 0.0077

v = np.linspace(0.1, 20.0, 800)  
v_mph = v * MS_TO_MPH

P_roll = (Crr_report*m_report*g)*v
P_aero = k*v**3

v_star = np.sqrt((Crr_report*m_report*g)/k)
v_star_mph = v_star * MS_TO_MPH


P_tot = P_roll + P_aero
share_roll = 100*P_roll/P_tot
share_aero = 100*P_aero/P_tot

plt.figure()
plt.plot(v_mph, share_roll, label="Rolling share (%)")
plt.plot(v_mph, share_aero, label="Aero share (%)")
plt.axhline(50, linewidth=1)
plt.axvline(v_star_mph, linewidth=1)
plt.xlabel("Speed (mph)")
plt.ylabel("Share of total resistive power (%)")
plt.title("Rolling vs Aerodynamic Drag (No Wind) — Share of Total")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("Crossover_Share_Rolling_vs_Aero.pdf", dpi=1200)
plt.close()

ratio = P_aero / P_roll
plt.figure()
plt.plot(v_mph, ratio)
plt.axhline(1, linewidth=1)
plt.axvline(v_star_mph, linewidth=1)
plt.yscale("log")
plt.xlabel("Speed (mph)")
plt.ylabel(r"Ratio $P_{\mathrm{aero}}/P_{\mathrm{roll}}$ (log scale)")
plt.title("Regime Transition (No Wind): Aero vs Rolling Dominance")
plt.grid(True, which="both")
plt.tight_layout()
plt.savefig("Crossover_Ratio_Log.pdf", dpi=1200)
plt.close()

# (c) Crossover_Power_Zoomed.pdf
plt.figure()
plt.plot(v_mph, P_roll, label="Rolling power")
plt.plot(v_mph, P_aero, label="Aero power")
plt.axvline(v_star_mph, linewidth=1)
plt.xlim(0, 20)
plt.ylim(0, 300)
plt.xlabel("Speed (mph)")
plt.ylabel("Power (W)")
plt.title("Rolling vs Aero Power (No Wind)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("Crossover_Power_Zoomed.pdf", dpi=1200)
plt.close()

print("Saved crossover PDFs.", flush=True)


print("\n--- VALUES FOR REPORT ---")
print(f"Wind used (U_total): {U_total:.4f} m/s = {U_mph:.2f} mph")
print(f"Model: rho={rho}, Cd={Cd}, A={A}, k={k:.5f}")
print(f"Rolling: mass={mass}, Crr={Crr}, F_roll={F_roll:.3f} N")
print(f"Constant driving force: F_drive={F_drive:.2f} N")
print(f"REPORT crossover (m={m_report}, Crr={Crr_report}): v* = {v_star_mph:.2f} mph")


terrains = [
    ("Race asphalt",       0.0020),
    ("Smooth asphalt",     0.0025),
    ("Rough asphalt",      0.0060),
    ("Wet/dirty road",     0.0100),
    ("Hard-packed gravel", 0.0120),
    ("Loose gravel",       0.0200),
    ("Soft snow",          0.0500),
]

v_vals = np.linspace(0.5, 15.0, 400)  
v_vals_mph = v_vals * MS_TO_MPH

plt.figure()
for name, Crr_t in terrains:
    Froll_t = Crr_t * mass * g
    P_roll_t = Froll_t * v_vals
    plt.plot(v_vals_mph, P_roll_t, label=f"{name} ({Crr_t:g})")

plt.xlabel("Speed (mph)")
plt.ylabel("Rolling Power $P_{roll}$ (W)")
plt.title("Rolling Power vs Speed for Different Terrain")
plt.grid(True)


leg = plt.legend(
    fontsize=12,          
    frameon=True,
    markerscale=0.75,
    handlelength=0.7,
    handletextpad=0.18,
    borderpad=0.12,
    labelspacing=0.18,
    columnspacing=0.30,
    ncol=1,
    loc="upper left",
    bbox_to_anchor=(0.02, 0.98),
)

frame = leg.get_frame()
frame.set_edgecolor("black")
frame.set_linewidth(0.6)
frame.set_alpha(0.95)

plt.tight_layout()
plt.savefig("Rolling_Power_Different_Terrain.pdf", dpi=1200)
plt.close()


v0_mph_ref = 20.0
v0 = v0_mph_ref * MPH_TO_MS

def aero_power(v, U, theta_deg):
    return aero_resist_along(v, U, theta_deg) * v

def total_power(v, U, theta_deg, Crr_local):
    return aero_power(v, U, theta_deg) + (Crr_local * mass * g) * v

def solve_v_for_power(P_target, U, theta_deg, Crr_local, v_low=0.01, v_high=40.0, tol=1e-6, max_iter=200):
    def f(v):
        return total_power(v, U, theta_deg, Crr_local) - P_target

    a, b = v_low, v_high
    fa, fb = f(a), f(b)

    if fa * fb > 0:
        for _ in range(40):
            b *= 1.5
            fb = f(b)
            if fa * fb <= 0:
                break
        else:
            raise RuntimeError("Could not bracket root for speed.")

    for _ in range(max_iter):
        m = 0.5*(a+b)
        fm = f(m)
        if abs(fm) < tol:
            return m
        if fa * fm <= 0:
            b, fb = m, fm
        else:
            a, fa = m, fm
    return 0.5*(a+b)


Pmax_ref = total_power(v0, 0.0, 0.0, Crr)

theta_head = 180.0
labels = []
v_head_mph = []

for name, Crr_t in terrains:
    v_hw = solve_v_for_power(Pmax_ref, U_total, theta_head, Crr_t)
    labels.append(name)
    v_head_mph.append(v_hw * MS_TO_MPH)

plt.figure()
plt.bar(labels, v_head_mph)
plt.xticks(rotation=30, ha="right", fontsize=TICKS)  
plt.yticks(fontsize=TICKS)
plt.ylabel("Predicted max speed into headwind (mph)")
plt.title(f"Terrain effect on headwind max speed\n(U={U_mph:.1f} mph, Pmax={Pmax_ref:.0f} W)")
plt.grid(True, axis="y")
plt.tight_layout()
plt.savefig("Headwind_MaxSpeed_vs_Terrain.pdf", dpi=1200)
plt.close()

print("Saved terrain PDFs.", flush=True)
print("\nTerrain -> Crr -> headwind max speed (mph)")
for (name, Crr_t), vmph in zip(terrains, v_head_mph):
    print(f"{name:18s}  Crr={Crr_t:>6g}   v_head={vmph:6.2f}")


v_mph_grid = np.linspace(0, 35, 250)
theta_grid = np.linspace(0, 180, 250)
V, TH = np.meshgrid(v_mph_grid * MPH_TO_MS, theta_grid)

Fpar = aero_resist_along(V, U_total, TH)


plt.figure()
im = plt.imshow(
    Fpar,
    origin="lower",
    aspect="auto",
    extent=[v_mph_grid.min(), v_mph_grid.max(), theta_grid.min(), theta_grid.max()]
)
cbar = plt.colorbar(im)
cbar.set_label(r"$F_{parallel}$ (N)", fontsize=LABEL)
cbar.ax.tick_params(labelsize=TICKS)

plt.xlabel("Cyclist speed (mph)")
plt.ylabel("Wind angle (deg)  (0° tailwind, 180° headwind)")
plt.title(f"Aerodynamic resistance along motion $F_\\parallel$ (U = {U_mph:.1f} mph)")
plt.tight_layout()
plt.savefig("Heatmap_Fparallel_vs_Speed_Angle.pdf", dpi=1200)
plt.close()

print("Saved Heatmap_Fparallel_vs_Speed_Angle.pdf", os.path.exists("Heatmap_Fparallel_vs_Speed_Angle.pdf"))


v_mph_grid2 = np.linspace(0, 35, 220)
theta_grid2 = np.linspace(0, 180, 220)
V2, TH2 = np.meshgrid(v_mph_grid2 * MPH_TO_MS, theta_grid2)

Fpar2 = aero_resist_along(V2, U_total, TH2)

plt.figure()
cs = plt.contourf(v_mph_grid2, theta_grid2, Fpar2, levels=25)
cbar = plt.colorbar(cs)
cbar.set_label("Air resistance (N)", fontsize=LABEL)
cbar.ax.tick_params(labelsize=TICKS)

plt.xlabel("Cyclist speed (mph)")
plt.ylabel("Wind angle (deg)")
plt.title(f"Contours of $Force_\\parallel(v,\\theta)$")
plt.tight_layout()
plt.savefig("Contour_Fparallel_vs_Speed_Angle.pdf", dpi=1200)
plt.close()

print("Saved Contour_Fparallel_vs_Speed_Angle.pdf", os.path.exists("Contour_Fparallel_vs_Speed_Angle.pdf"))
