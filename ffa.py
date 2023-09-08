from __future__ import division
import numpy as np


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
        
        dMag1 = Mag(
            self.k, -rhoD, r2, r3, tD * 2, self.BD, self.tC / 2 - self.BD/2, self.dfl, self.spi
        )
        dMag2 = Mag(
            self.k, -rhoD, r3, r2, tD * 2, self.BD, self.tC / 2 + self.BD/2, self.dfl, self.spi
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
            dMag1,
            dMag2,
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
