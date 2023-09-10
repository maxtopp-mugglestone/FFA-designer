from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import ffa


def r0_update(val):
    #Function to update r0 input to lattice when slider is used. 
    r0 = val
    Latt.r0 = val
    draw_latt()
    qUpdate()
    ax[0].set_xlim([-0.1, r0 * 1.1])
    ax[0].set_ylim([-0.1, r0 * 1.1])
    fig.canvas.draw_idle()


def tF_update(val):
    #Function to update tF input to lattice when slider is used.
    Latt.tF = np.radians(val)
    draw_latt()
    qUpdate()
    fig.canvas.draw_idle()


def Nc_update(val):
    #Function to update Nc input to lattice when slider is used.
    Latt.Nc = val
    draw_latt()
    qUpdate()
    fig.canvas.draw_idle()


def bd_update(val):
    #Function to update BD input to lattice when slider is used.
    Latt.BD = np.radians(val)
    draw_latt()
    qUpdate()
    fig.canvas.draw_idle()


def bf_update(val):
    #Function to update BF input to lattice when slider is used.
    Latt.BF = np.radians(val)
    draw_latt()
    qUpdate()
    fig.canvas.draw_idle()


def k_update(val):
    #Function to update k input to lattice when slider is used.
    Latt.k = val
    qUpdate()


def qUpdate():
    #function to update tune plot (right hand subplot) and text labels. 
    q = ffa.tune(Latt)
    tunepoint.set_xdata(q[0])
    tunepoint.set_ydata(q[2])

    tr_x.set_text("$Tr[M_x] = {:03f}$".format(np.real(np.trace(Latt.tMatrix[:2, :2]))))
    tr_y.set_text("$Tr[M_y] = {:03f}$".format(np.real(np.trace(Latt.tMatrix[2:, 2:]))))
    q_x.set_text("$q_x = {:03f}$".format(q[0]))
    q_y.set_text("$q_y = {:03f}$".format(q[2]))


def toggle(self):
    #Function to toggle between triplet and FODO lattices. 
    global Latt
    global trip
    if not trip:
        Latt = ffa.TripletLattice(
            [Latt.Nc, Latt.r0, Latt.BF, Latt.BD, Latt.tF],
            Latt.k,
            Latt.ffl,
            Latt.dfl,
            Latt.spi,
        )
        btog.label.set_text("Triplet")
        trip = True
    else:
        Latt = ffa.FodoLattice(
            [Latt.Nc, Latt.r0, Latt.BF, Latt.BD, Latt.tF],
            Latt.k,
            Latt.ffl,
            Latt.dfl,
            Latt.spi,
        )
        trip = False
        btog.label.set_text("FODO")
    draw_latt()
    qUpdate()


def draw_latt():
    #Function to redraw closed orbit plot when lattice is adjusted. 
    tData = ffa.trajData(Latt)
    rData = ffa.rList(Latt)
    trajLine.set_xdata(tData[:, 1])
    trajLine.set_ydata(tData[:, 0])
    rLine.set_xdata(rData[:, 1])
    rLine.set_ydata(rData[:, 0])
    update_text()
    fig.canvas.draw_idle()

def update_text():
    rho_f.set_text(r"$\rho_F = {:03f}$".format(Latt.rhoF))
    rho_d.set_text(r"$\rho_D = {:03f}$".format(Latt.rhoD))
    r_1.set_text(r"$r_1 = {:03f}$".format(Latt.r1))
    r_2.set_text(r"$r_2 = {:03f}$".format(Latt.r2))
    r_3.set_text(r"$r_3 = {:03f}$".format(Latt.r3))
    t_d.set_text(r"$\theta_D = {:03f}^\circ$".format(np.degrees(Latt.tD)))




#Generate figure with two side-by-side subplots.
fig, ax = plt.subplots(1, 2)

#Adjust position/scale of subplots to make room for sliders on left hand side
fig.subplots_adjust(left=0.25, bottom=0.25)

#Set starting parameters of lattice
Nc = 16
r0 = 4
bf = 2.5
bd = 1.25
tf = 22.5
k = 8.0095
trip = False #set default flag for triplet lattice toggle to false

#initialise global lattice object 
Latt = ffa.FodoLattice([Nc, r0, np.radians(bf), np.radians(bd), np.radians(tf)], k, 0, 0, 0)

#plot radii and trajectory of closed orbit
rdata = ffa.rList(Latt)
(rLine,) = ax[0].plot(rdata[:, 1], rdata[:, 0], color="k")
tdata = ffa.trajData(Latt)
(trajLine,) = ax[0].plot(tdata[:, 1], tdata[:, 0], color="r")

#format closed orbit plot
ax[0].set_aspect("equal")
ax[0].set_xlim([-0.1, r0 * 1.1])
ax[0].set_ylim([-0.1, r0 * 1.1])

#draw resonance lines on tune plot
order = 4 #maximum order of resonance to include
x = np.linspace(0, 1)
cycle = ["k"]  # color cycle for resonance lines if used
for n in reversed(np.arange(order + 1)):
    lw = 3 / (n + 1) # change linewidth so higher order resonances have thinner lines
    for m in np.arange(n):
        col = cycle[
            n % len(cycle)
        ]  # currently inactive; cycle through colours for different order of resonance line
        ax[1].vlines(m * 1 / n, 0, 1, color=col, linewidth=2 * lw)
        ax[1].hlines(m * 1 / n, 0, 1, color=col, linewidth=2 * lw)

        ax[1].plot(m * 1 / n + x, n * x, color=col, linewidth=lw, ls="--")
        ax[1].plot(m * 1 / n + x, 1 - n * x, color=col, linewidth=lw, ls="--")

        ax[1].plot(n * x, m * 1 / n + x, color=col, linewidth=lw, ls="--")
        ax[1].plot(1 - n * x, m * 1 / n + x, color=col, linewidth=lw, ls="--")

#format tune plot
ax[1].set_aspect("equal")
ax[1].set_xlim(0, 0.5)
ax[1].set_ylim(0, 0.5)
ax[1].set_xlabel("$q_x$")
ax[1].set_ylabel("$q_y$")

#draw current tune of lattice on tuneplot
q = ffa.tune(Latt)
(tunepoint,) = ax[1].plot(q[0], q[2], c="r", marker="x", markersize = 10)

# Make vertical sliders for geometry input parameters
sspace = 0.04
R_ax = fig.add_axes([sspace, 0.25, 0.0225, 0.63])
R_slider = Slider(
    ax=R_ax, label="r0", valmin=0, valmax=25, valinit=r0, orientation="vertical"
)
R_slider.on_changed(r0_update)

tf_ax = fig.add_axes([sspace * 2, 0.25, 0.0225, 0.63])
tf_slider = Slider(
    ax=tf_ax,
    label="tF",
    valmin=-90,
    valmax=90,
    valinit=tf,
    orientation="vertical",
)
tf_slider.on_changed(tF_update)

Nc_ax = fig.add_axes([sspace * 3, 0.25, 0.0225, 0.63])
allowed_Nc = np.arange(1, 100)
Nc_slider = Slider(
    ax=Nc_ax,
    label="Nc",
    valmin=1,
    valmax=64,
    valstep=allowed_Nc,
    valinit=Nc,
    orientation="vertical",
)
Nc_slider.on_changed(Nc_update)

bf_ax = fig.add_axes([sspace * 4, 0.25, 0.0225, 0.63])
bf_slider = Slider(
    ax=bf_ax, label="BF", valmin=0, valmax=45/2, valinit=bf, orientation="vertical"
)
bf_slider.on_changed(bf_update)

bd_ax = fig.add_axes([sspace * 5, 0.25, 0.0225, 0.63])
bd_slider = Slider(
    ax=bd_ax, label="BD", valmin=0, valmax=45/2, valinit=bd, orientation="vertical"
)
bd_slider.on_changed(bd_update)

#make horizontal slider for k-value input
k_ax = fig.add_axes([0.05, 0.05, 0.85, 0.0225])
k_slider = Slider(
    ax=k_ax, label="k", valmin=0, valmax=20, valinit=k, orientation="horizontal"
)
k_slider.on_changed(k_update)

#add button to toggle between FODO and triplet lattices
axtog = fig.add_axes([0.7, 0.1, 0.1, 0.075])
btog = Button(axtog, "FODO")
btog.on_clicked(toggle)

#write trace of 2d transfer matrices
tr_x = fig.text(
    0.05, 0.15, "$Tr[M_x] = {:03f}$".format(np.real(np.trace(Latt.tMatrix[:2, :2])))
)
tr_y = fig.text(
    0.05, 0.1, "$Tr[M_y] = {:03f}$".format(np.real(np.trace(Latt.tMatrix[2:, 2:])))
)

#write radius of curvature
rho_f = fig.text(
    0.15, 0.15, r"$\rho_F = {:03f}$".format(Latt.rhoF)
)
rho_d = fig.text(
    0.15, 0.1, r"$\rho_D = {:03f}$".format(Latt.rhoD)
)


#write radius of curvature
r_1 = fig.text(
    0.25, 0.15, r"$r_1 = {:03f}$".format(Latt.r1)
)
r_2 = fig.text(
    0.25, 0.1, r"$r_2 = {:03f}$".format(Latt.r2)
)
r_3 = fig.text(
    0.35, 0.15, r"$r_3 = {:03f}$".format(Latt.r3)
)
t_d = fig.text(
    0.35, 0.1, r"$\theta_D = {:03f}^\circ$".format(np.degrees(Latt.tD))
)

#write tune of lattice
q_x = fig.text(0.85, 0.15, "$q_x = {:03f}$".format(q[0]))
q_y = fig.text(0.85, 0.1, "$q_y = {:03f}$".format(q[2]))

#display plot in interactive window
plt.show()
