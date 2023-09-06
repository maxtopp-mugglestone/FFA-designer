from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button


class Mag:
    '''
    Magnet object for an hFFA lattice
    
    Attributes
    ----------
    k : float
        Field index k in terms of hFFA scaling law (r/r0)**k (unitless)
    rho : float
        Radius of curvature of beam within magnet (units metres)
    rin : float
        Radius of closed orbit at entrance of element with respect to machine centre (units metres)
    rout : float
        Radius of closed orbit at exit of element with respect to machine centre (units metres)
    theta : float
        Angle of curvature within magnet (units radians)
    beta : float
        Opening angle of element with respect to machine centre (units radians)
    centre : float
        Azimuthal position of element centre with respect to machine centre (units radians)

    Properties
    ----------
    tM : 
        Returns 4x4 array corresponding to the transfer matrix of the element
    plotTraj : 
        Returns 2d numpy array of coordinates along trajectory of beam through element
    plotRs :
        Returns 2d numpy array with coordinates of start and end point of trajectory through element
    '''
    def __init__(self, k, rho, rin, rout, theta, beta, centre, fl, spi):
        self.k = k
        self.rho = rho
        self.rin = rin
        self.rout = rout
        self.theta = theta
        self.beta = beta
        self.centre = centre
        self.fl = fl
        self.spi = spi

    @property
    def tM(self):
        r = (self.rin + self.rout) / 2
        wx = np.emath.sqrt((1 / self.rho) * (1 / self.rho + self.k / r))
        wz = np.emath.sqrt(self.k / (self.rho * r))
        L = np.abs(self.theta * self.rho)
        tM = np.array(
            [
                [np.cos(wx * L), np.sin(wx * L) / wx, 0, 0],
                [-wx * np.sin(wx * L), np.cos(wx * L), 0, 0],
                [0, 0, np.cosh(wx * L), np.sinh(wz * L) / wz],
                [0, 0, wz * np.sinh(wz * L), np.cosh(wz * L)],
            ]
        )
        return tM

    @property
    def plotTraj(self):
        point1 = [
            self.rout * np.cos(self.beta / 2 + self.centre),
            self.rout * np.sin(self.centre + self.beta / 2),
        ]
        point2 = [
            self.rin * np.cos(self.centre - self.beta / 2),
            self.rin * np.sin(self.centre - self.beta / 2),
        ]
        return generate_arc(point1, point2, self.rho)

    @property
    def plotRs(self):
        en = np.array(
            [
                self.rin * np.cos(self.centre - self.beta / 2),
                self.rin * np.sin(self.centre - self.beta / 2),
            ]
        )
        ex = np.array(
            [
                self.rout * np.cos(self.centre + self.beta / 2),
                self.rout * np.sin(self.centre + self.beta / 2),
            ]
        )
        return np.vstack([en, ex])


class Edge:
    '''
    Edge focussing object for an hFFA lattice
    Adds focussing effect of extra 'wedge' travelled through magnet as function of transverse position
    
    Attributes
    ----------
    alpha : float
        angle of reference trajectory with respect to magnet end plane normal (units radians)
    rho : float
        Radius of curvature of beam within magnet (units metres)
    

    Properties
    ----------
    tM : 
        Returns 4x4 array corresponding to the transfer matrix of the element
    plotTraj : 
        Returns empty array as element has zero length
    plotRs :
        Returns 2d numpy array of zeros (element has zero length)
    '''
    def __init__(self, alpha, rho):
        self.tM = np.array(
            [[1, 0, 0, 0], [np.tan(alpha) / rho, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
        )
        self.rho, self.alpha = rho, alpha

    @property
    def plotTraj(self):
        return np.empty(0), np.empty(0)

    @property
    def plotRs(self):
        return np.zeros([2, 2])


class Fringe:
    '''
    Thin lens fringe field object for an hFFA lattice
    Assumes a linear fringe field falloff over length L
    
    Attributes
    ----------
    alpha : float
        angle of reference trajectory with respect to magnet end plane normal (units radians)
    r : float
        radius of reference trajectory at position of element, measured with respect to machine centre (units metres)
    k : float
        Field index k in terms of hFFA scaling law (r/r0)**k (unitless)
    L : float
        Length of fringe field linear falloff (units metres) from zero to nominal B-field of magnet
    rho : float
        Radius of curvature of beam within magnet (units metres)

    Properties
    ----------
    tM : 
        Returns 4x4 array corresponding to the transfer matrix of the element
    plotTraj : 
        Returns 2d numpy array of coordinates along trajectory of beam through element
    plotRs :
        Returns 2d numpy array with coordinates of start and end point of trajectory through element
    '''
    def __init__(self, alpha, rho, r, L, k):
        self.k = k
        self.fl = L
        self.r = r
        self.rho = rho
        self.alpha = alpha
        self.tM = np.array(
            [
                [1, 0, 0, 0],
                [-self.k * self.fl / (r * self.rho), 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, (self.k * self.fl / (r * self.rho) - np.tan(alpha) / rho), 1],
            ]
        )

    @property
    def plotTraj(self):
        return np.empty(0), np.empty(0)

    @property
    def plotRs(self):
        return np.zeros([2, 2])


class Drift:
    '''
    Drift space element
    
    Attributes
    ----------
    L : float
        Length of drift (units metres)
    b1 : float
        Azimuthal position of element start (units radians), used for plotting
    b2 : float
        Azimuthal position of element start (units radians), used for plotting
    r1 : float
        radius of reference trajectory at start of element, measured with respect to machine centre (units metres), used for plotting
    r2 : float
        radius of reference trajectory at start of element, measured with respect to machine centre (units metres), used for plotting

    Properties
    ----------
    tM : 
        Returns 4x4 array corresponding to the transfer matrix of the element
    plotTraj : 
        Returns 2d numpy array of coordinates along trajectory of beam through element
    plotRs :
        Returns 2d numpy array with coordinates of start and end point of trajectory through element
    '''
    def __init__(self, L, b1, b2, r1, r2):
        self.L = L
        self.b1 = b1
        self.b2 = b2
        self.r1 = r1
        self.r2 = r2

    @property
    def tM(self):
        mat = np.array(
            [[1, self.L, 0, 0], [0, 1, 0, 0], [0, 0, 1, self.L], [0, 0, 0, 1]]
        )
        return mat

    @property
    def plotTraj(self):
        return [self.r1 * np.cos(self.b1), self.r2 * np.cos(self.b2)], [
            self.r1 * np.sin(self.b1),
            self.r2 * np.sin(self.b2),
        ]

    @property
    def plotRs(self):
        return np.array(
            [
                [self.r1 * np.cos(self.b1), self.r1 * np.sin(self.b1)],
                [self.r2 * np.cos(self.b2), self.r2 * np.sin(self.b2)],
            ]
        )


class FodoLattice:
    '''
    hFFA FODO lattice object
        A FODO lattice is defined as a lattice with evenly spaced F (normal bend) and D (reverse bend) elements.
        The lattice is then azimuthally symmetric about the centroids for the F and D magnets. 
    
    Attributes
    ----------
    Nc : int
        Number of cells in full ring. Determines opening angle of cell. 
    r0 : float
        radius of reference trajectory at centre of F-magnet, measured with respect to machine centre (units metres)
    BF : float
        beta_F -- opening angle of F-magnet in half-cell, defined with respect to machine centre (units radians)
    BD : float
        beta_D -- opening angle of D-magnet in half-cell, defined with respect to machine centre (units radians)
    tF : float
        theta_F -- bending angle of F-magnet in half-cell
    k : float
        Field index k in terms of hFFA scaling law (r/r0)**k (unitless)
    ffl : float
        Length of linear fringe field ramp in F-magnet (units metres)
    dfl : float
        Length of linear fringe field ramp in D-magnet (units metres)
    spi : float
        Spiral angle of lattice (units radians)

    Properties
    ----------
    tC : 
        returns float corresponding to cell opening angle
    vars : 
        returns r1, r2, r3, tD, rhoF, rhoD
            r1 : orbit radius at exit of F-magnet
            r2 : orbit radius at entrance of D-magnet
            r3 : orbit radius at centre of D-magnet (boundary of half-cell)
            tD : bending angle in D-magnet
            rhoF : bending radius in F-magnet
            rhoD : bending radius in D-magnet
    elements:
        Returns list of objects corresponding to each element in the cell.
        List is returned beginning in the middle of the F-magnet.
    tMatrix : 
        Returns 4x4 array corresponding to the transfer matrix of the cell
    '''
    def __init__(self, params, k, ffl, dfl, spi) -> None:
        self.Nc, self.r0, self.BF, self.BD, self.tF = params
        self.k = k
        self.ffl = ffl
        self.dfl = dfl
        self.spi = spi
        pass

    @property
    def tC(self):
        return 2 * np.pi / self.Nc

    @property
    def vars(self):
        BF = self.BF
        BD = self.BD
        tF = self.tF
        r0 = self.r0
        Nc = self.Nc
        rhoF = np.tan(BF) / (np.sin(tF) + (1 - np.cos(tF)) * np.tan(BF)) * r0
        tD = self.tF - np.pi / Nc
        r1 = rhoF * np.sin(tF) / np.sin(BF)
        r2 = (
            r1
            * (np.cos(BF) + np.tan(tF) * np.sin(BF))
            / (np.cos(np.pi / Nc - BD) + np.tan(tF) * np.sin(np.pi / Nc - BD))
        )
        rhoD = r2 * np.sin(BD) / np.sin(tD)
        r3 = r2 * np.cos(BD) - rhoD * (1 - np.cos(tD))
        return r1, r2, r3, tD, rhoF, rhoD

    @property
    def elements(self):
        r1, r2, r3, tD, rhoF, rhoD = self.vars
        fMag = Mag(
            self.k, rhoF, self.r0, r1, self.tF, self.BF, self.BF / 2, self.ffl, self.spi
        )
        fMag2 = Mag(
            self.k,
            rhoF,
            r1,
            self.r0,
            self.tF,
            self.BF,
            self.tC - self.BF / 2,
            self.ffl,
            self.spi,
        )
        dMag = Mag(
            self.k, -rhoD, r2, r2, tD * 2, self.BD * 2, self.tC / 2, self.dfl, self.spi
        )
        lDrift = (
            r2 * np.sin(np.pi / self.Nc - self.BF - self.BD) / np.cos(self.tF - self.BF)
        )
        Drift1 = Drift(lDrift, self.BF, self.tC / 2 - self.BD, r1, r2)
        Drift2 = Drift(lDrift, self.tC - self.BF, self.tC / 2 + self.BD, r1, r2)
        return [
            fMag,
            Edge(self.tF - self.BF, rhoF),
            Fringe(self.tF - self.BF, rhoF, r1, self.ffl, self.k),
            Drift1,
            Fringe(tD + self.BD, rhoD, r2, self.dfl, self.k),
            Edge(tD + self.BD, rhoD),
            dMag,
            Edge(tD + self.BD, rhoD),
            Fringe(tD + self.BD, rhoD, r3, self.dfl, self.k),
            Drift2,
            Fringe(self.tF - self.BF, rhoF, r1, self.ffl, self.k),
            Edge(self.tF - self.BF, rhoF),
            fMag2,
        ]

    @property
    def tMatrix(self):
        tM = np.eye(4)
        for element in self.elements:
            tM = np.matmul(element.tM, tM)
        return tM


class TripletLattice:
    '''
    hFFA Triplet lattice object
        For positive theta_F this represents a DFD triplet lattice. 
        A DFD triplet lattice is defined as a lattice with a normal bend element (F-magnet) sandwiched by two reverse bend elements (D-magnet) .
        The lattice is then azimuthally symmetric about the centroids for the F magnet and the centre of the drift. 
        Inputting negative theta_F switches the F and D magnets, so the lattice becomes an FDF triplet. 

        Definitions are explained in terms of a DFD triplet lattice. 
        
    
    Attributes
    ----------
    Nc : int
        Number of cells in full ring. Determines opening angle of cell. 
    r0 : float
        radius of reference trajectory at centre of F-magnet, measured with respect to machine centre (units metres)
    BF : float
        beta_F -- opening angle of F-magnet in half-cell, defined with respect to machine centre (units radians)
    BD : float
        beta_D -- opening angle of D-magnet in half-cell, defined with respect to machine centre (units radians)
    tF : float
        theta_F -- bending angle of F-magnet in half-cell
    k : float
        Field index k in terms of hFFA scaling law (r/r0)**k (unitless)
    ffl : float
        Length of linear fringe field ramp in F-magnet (units metres)
    dfl : float
        Length of linear fringe field ramp in D-magnet (units metres)
    spi : float
        Spiral angle of lattice (units radians)

    Properties
    ----------
    tC : 
        returns float corresponding to cell opening angle
    vars : 
        returns r1, r2, r3, tD, rhoF, rhoD
            r1 : orbit radius at exit of F-magnet/entrance of D-magnet
            r2 : orbit radius at exit of D-magnet
            r3 : orbit radius at boundary of half-cell (centre of drift)
            tD : bending angle in D-magnet
            rhoF : bending radius in F-magnet
            rhoD : bending radius in D-magnet
    elements:
        Returns list of objects corresponding to each element in the cell.
        List is returned beginning in the middle of the F-magnet.
    tMatrix : 
        Returns 4x4 array corresponding to the transfer matrix of the cell
    '''
    def __init__(self, params, k, ffl, dfl, spi) -> None:
        self.Nc, self.r0, self.BF, self.BD, self.tF = params
        self.k = k
        self.ffl = ffl
        self.dfl = dfl
        self.spi = spi
        pass

    @property
    def tC(self):
        return 2 * np.pi / self.Nc

    @property
    def vars(self):
        BF = self.BF
        BD = self.BD
        tF = self.tF
        r0 = self.r0
        Nc = self.Nc
        rhoF = np.tan(BF) / (np.sin(tF) + (1 - np.cos(tF)) * np.tan(BF)) * r0
        tD = tF - np.pi / Nc
        r1 = rhoF * np.sin(tF) / np.sin(BF)
        rhoD = (
            rhoF
            * np.sin(tF)
            / np.sin(BF)
            * (
                np.sin(np.pi / Nc - BF)
                - np.cos(np.pi / Nc - BF) * np.tan(np.pi / Nc - BF - BD)
            )
            / (
                np.sin(tF - np.pi / Nc)
                - (1 - np.cos(tF - np.pi / Nc)) * np.tan(np.pi / Nc - BF - BD)
            )
        )
        r2 = (
            r1 * np.cos(BF)
            - rhoD * np.sin(tD) * np.sin(np.pi / Nc)
            - rhoD * (1 - np.cos(tD)) * np.cos(np.pi / Nc)
        ) / np.cos(BF + BD)
        r3 = r2 * np.cos(np.pi / Nc - BF - BD)
        return r1, r2, r3, tD, rhoF, rhoD

    @property
    def elements(self):
        r1, r2, r3, tD, rhoF, rhoD = self.vars
        fMag = Mag(
            self.k, rhoF, self.r0, r1, self.tF, self.BF, self.BF / 2, self.ffl, self.spi
        )
        dMag = Mag(
            self.k,
            -rhoD,
            r1,
            r2,
            tD,
            self.BD,
            self.BF + self.BD / 2,
            self.dfl,
            self.spi,
        )
        lDrift = r2 * np.sin(np.pi / self.Nc - self.BF - self.BD)
        cdrift = Drift(lDrift, self.BF + self.BD, self.tC / 2, r2, r3)
        fMag2 = Mag(
            self.k,
            rhoF,
            r1,
            self.r0,
            self.tF,
            self.BF,
            self.tC - self.BF / 2,
            self.ffl,
            self.spi,
        )
        dMag2 = Mag(
            self.k,
            -rhoD,
            r2,
            r1,
            tD,
            self.BD,
            self.tC - (self.BF + self.BD / 2),
            self.dfl,
            self.spi,
        )
        cdrift2 = Drift(lDrift, self.tC / 2, self.tC - self.BF - self.BD, r3, r2)
        return [
            fMag,
            Edge(self.tF - self.BF, rhoF),
            Fringe(self.tF - self.BF, rhoF, r1, self.ffl, self.k),
            Fringe(tD + self.BD, rhoD, r3, self.dfl, self.k),
            Edge(tD + self.BD, rhoD),
            dMag,
            Edge(tD + self.BD, rhoD),
            Fringe(tD + self.BD, rhoD, r3, self.dfl, self.k),
            cdrift,
            cdrift2,
            Fringe(tD + self.BD, rhoD, r3, self.dfl, self.k),
            Edge(tD + self.BD, rhoD),
            dMag2,
            Edge(tD + self.BD, rhoD),
            Fringe(tD + self.BD, rhoD, r3, self.dfl, self.k),
            Fringe(self.tF - self.BF, rhoF, r1, self.ffl, self.k),
            Edge(self.tF - self.BF, rhoF),
            fMag2,
        ]

    @property
    def tMatrix(self):
        tM = np.eye(4)
        for element in self.elements:
            tM = np.matmul(element.tM, tM)
        return tM


def compute_point_distance(point1, point2, r):
    # Calculate the distance between the two points

    distance = np.sqrt((point2[0] - point1[0]) ** 2 + (point2[1] - point1[1]) ** 2)

    # Calculate the direction vector between the two points
    direction_x = (point2[0] - point1[0]) / distance
    direction_y = (point2[1] - point1[1]) / distance

    # Calculate the midpoint between the two points
    midpoint_x = (point1[0] + point2[0]) / 2
    midpoint_y = (point1[1] + point2[1]) / 2

    # Calculate the distance from the midpoint to the intersection point
    distance_intersection = np.sqrt(r**2 - (distance / 2) ** 2)

    # Calculate the coordinates of the intersection point
    intersection_x = midpoint_x + np.sign(r) * (distance_intersection * direction_y)
    intersection_y = midpoint_y - np.sign(r) * (distance_intersection * direction_x)

    return intersection_x, intersection_y


def generate_arc(point1, point2, r, num_points=100):
    # Calculate the intersection point with distance r from point1 and point2
    intersection_point = compute_point_distance(point1, point2, r)

    # Calculate the center of the arc
    center_x = intersection_point[0]
    center_y = intersection_point[1]

    # Calculate the start and end angles for the arc
    start_angle = np.arctan2(point1[1] - center_y, point1[0] - center_x)
    end_angle = np.arctan2(point2[1] - center_y, point2[0] - center_x)

    # Generate the angles for the arc
    if abs(end_angle % (2 * np.pi) - start_angle % (2 * np.pi)) < abs(
        end_angle - start_angle
    ):
        theta = np.linspace(
            start_angle % (2 * np.pi), end_angle % (2 * np.pi), num_points
        )
    else:
        theta = np.linspace(start_angle, end_angle, num_points)

    # Generate the x and y coordinates for the arc
    x = center_x + np.abs(r) * np.cos(theta)
    y = center_y + np.abs(r) * np.sin(theta)

    return x, y


def trajData(Latt):
    #Return 2d numpy array of XY data for trajectory through lattice object
    data = np.empty([0, 2])
    for element in Latt.elements[::-1]:
        point = np.array([element.plotTraj[0], element.plotTraj[1]])
        data = np.vstack([data, point.T])
    return data


def rList(Latt):
    #Return list of radii for given lattice (used for plotting)
    list = np.zeros(2)
    for element in Latt.elements[::-1]:
        for r in element.plotRs:
            list = np.vstack([list, r, np.zeros(2)])
    return list


def tune(Latt):
    #Compute 2d tune of lattice from eigenvalues of transfer matrix.
    eig = np.linalg.eig(Latt.tMatrix)[0]
    return np.angle(eig) / (2 * np.pi)


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
    q = tune(Latt)
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
        Latt = TripletLattice(
            [Latt.Nc, Latt.r0, Latt.BF, Latt.BD, Latt.tF],
            Latt.k,
            Latt.ffl,
            Latt.dfl,
            Latt.spi,
        )
        btog.label.set_text("Triplet")
        trip = True
    else:
        Latt = FodoLattice(
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
    tData = trajData(Latt)
    rData = rList(Latt)
    trajLine.set_xdata(tData[:, 1])
    trajLine.set_ydata(tData[:, 0])
    rLine.set_xdata(rData[:, 1])
    rLine.set_ydata(rData[:, 0])
    fig.canvas.draw_idle()


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
Latt = FodoLattice([Nc, r0, np.radians(bf), np.radians(bd), np.radians(tf)], k, 0, 0, 0)

#plot radii and trajectory of closed orbit
rdata = rList(Latt)
(rLine,) = ax[0].plot(rdata[:, 1], rdata[:, 0], color="k")
tdata = trajData(Latt)
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
q = tune(Latt)
(tunepoint,) = ax[1].plot(q[0], q[2], c="r", marker="x", markersize = 10)

# Make vertical sliders for geometry input parameters
sspace = 0.04
R_ax = fig.add_axes([sspace, 0.25, 0.0225, 0.63])
R_slider = Slider(
    ax=R_ax, label="r0", valmin=0, valmax=100, valinit=r0, orientation="vertical"
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
    valmax=100,
    valstep=allowed_Nc,
    valinit=Nc,
    orientation="vertical",
)
Nc_slider.on_changed(Nc_update)

bf_ax = fig.add_axes([sspace * 4, 0.25, 0.0225, 0.63])
bf_slider = Slider(
    ax=bf_ax, label="BF", valmin=0, valmax=90, valinit=bf, orientation="vertical"
)
bf_slider.on_changed(bf_update)

bd_ax = fig.add_axes([sspace * 5, 0.25, 0.0225, 0.63])
bd_slider = Slider(
    ax=bd_ax, label="BD", valmin=0, valmax=90, valinit=bd, orientation="vertical"
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

#write tune of lattice
q_x = fig.text(0.85, 0.15, "$q_x = {:03f}$".format(q[0]))
q_y = fig.text(0.85, 0.1, "$q_y = {:03f}$".format(q[2]))

#display plot in interactive window
plt.show()
