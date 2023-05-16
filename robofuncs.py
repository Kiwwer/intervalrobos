import numpy as np

def convertToIntervalForm(base, plat, shift, angles):
    for i in range(len(base)):
        base[i] = np.array([Interval.valueToInterval(it) for it in base[i]])
    for i in range(len(plat)):
        plat[i] = np.array([Interval.valueToInterval(it) for it in plat[i]])
    shift = np.array([Interval.valueToInterval(it) for it in shift])
    angles = [Interval.valueToInterval(it) for it in angles]
    return base, plat, shift, angles

def rotationMatrix3D(angles):
    constzero = Interval.valueToInterval(0)
    constone = Interval.valueToInterval(1)
    p1 = np.array([[Interval.cos(angles[0]), Interval.sin(angles[0]), constzero],
                   [-Interval.sin(angles[0]), Interval.cos(angles[0]), constzero],
                   [constzero, constzero, constone]])
    p1i = np.array([[Interval.cos(angles[0]), -Interval.sin(angles[0]), constzero],
                   [Interval.sin(angles[0]), Interval.cos(angles[0]), constzero],
                   [constzero, constzero, constone]])
    p2 = np.array([[constone, constzero, constzero],
                   [constzero, Interval.cos(angles[1]), Interval.sin(angles[1])],
                   [constzero, -Interval.sin(angles[1]), Interval.cos(angles[1])]])
    p2i = np.array([[constone, constzero, constzero],
                   [constzero, Interval.cos(angles[1]), -Interval.sin(angles[1])],
                   [constzero, Interval.sin(angles[1]), Interval.cos(angles[1])]])
    p3 = np.array([[Interval.cos(angles[2]), Interval.sin(angles[2]), constzero],
                   [-Interval.sin(angles[2]), Interval.cos(angles[2]), constzero],
                   [constzero, constzero, constone]])
    p3i = np.array([[Interval.cos(angles[2]), -Interval.sin(angles[2]), constzero],
                   [Interval.sin(angles[2]), Interval.cos(angles[2]), constzero],
                   [constzero, constzero, constone]])    
    return p3 @ p2 @ p1, p1i, p2i, p3i

def rotationMatrix3DINT(angles):
    constzero = Decimal(0)
    constone = Decimal(1)
    p1 = np.array([[Interval.cos(angles[0]), Interval.sin(angles[0]), constzero],
                   [-Interval.sin(angles[0]), Interval.cos(angles[0]), constzero],
                   [constzero, constzero, constone]])
    p1i = np.array([[Interval.cos(angles[0]), -Interval.sin(angles[0]), constzero],
                   [Interval.sin(angles[0]), Interval.cos(angles[0]), constzero],
                   [constzero, constzero, constone]])
    p2 = np.array([[constone, constzero, constzero],
                   [constzero, Interval.cos(angles[1]), Interval.sin(angles[1])],
                   [constzero, -Interval.sin(angles[1]), Interval.cos(angles[1])]])
    p2i = np.array([[constone, constzero, constzero],
                   [constzero, Interval.cos(angles[1]), -Interval.sin(angles[1])],
                   [constzero, Interval.sin(angles[1]), Interval.cos(angles[1])]])
    p3 = np.array([[Interval.cos(angles[2]), Interval.sin(angles[2]), constzero],
                   [-Interval.sin(angles[2]), Interval.cos(angles[2]), constzero],
                   [constzero, constzero, constone]])
    p3i = np.array([[Interval.cos(angles[2]), -Interval.sin(angles[2]), constzero],
                   [Interval.sin(angles[2]), Interval.cos(angles[2]), constzero],
                   [constzero, constzero, constone]])    
    return p3 @ p2 @ p1, p1i, p2i, p3i

def rotationMatrix3DSC(s, c):
    constzero = Interval.valueToInterval(0)
    constone = Interval.valueToInterval(1)
    p1 = np.array([[c[0], s[0], constzero],
                   [-s[0], c[0], constzero],
                   [constzero, constzero, constone]])
    p1i = np.array([[c[0], -s[0], constzero],
                    [s[0], c[0], constzero],
                    [constzero, constzero, constone]])
    p2 = np.array([[constone, constzero, constzero],
                   [constzero, c[1], s[1]],
                   [constzero, -s[1], c[1]]])
    p2i = np.array([[constone, constzero, constzero],
                    [constzero, c[1], -s[1]],
                    [constzero, s[1], c[1]]])
    p3 = np.array([[c[2], s[2], constzero],
                   [-s[2], c[2], constzero],
                   [constzero, constzero, constone]])
    p3i = np.array([[c[2], -s[2], constzero],
                    [s[2], c[2], constzero],
                    [constzero, constzero, constone]])    
    return p3 @ p2 @ p1, p1i, p2i, p3i

def rotationMatrix3DSCINT(s, c):
    constzero = Decimal(0)
    constone = Decimal(1)
    p1 = np.array([[c[0], s[0], constzero],
                   [-s[0], c[0], constzero],
                   [constzero, constzero, constone]])
    p1i = np.array([[c[0], -s[0], constzero],
                    [s[0], c[0], constzero],
                    [constzero, constzero, constone]])
    p2 = np.array([[constone, constzero, constzero],
                   [constzero, c[1], s[1]],
                   [constzero, -s[1], c[1]]])
    p2i = np.array([[constone, constzero, constzero],
                    [constzero, c[1], -s[1]],
                    [constzero, s[1], c[1]]])
    p3 = np.array([[c[2], s[2], constzero],
                   [-s[2], c[2], constzero],
                   [constzero, constzero, constone]])
    p3i = np.array([[c[2], -s[2], constzero],
                    [s[2], c[2], constzero],
                    [constzero, constzero, constone]])    
    return p3 @ p2 @ p1, p1i, p2i, p3i

def InverseSG(base, plat, shift, angles, rotationMatrixFunction):
    R, _, _, _ = rotationMatrixFunction(angles)
    lengths = []
    for i in range(len(base)):
        platR0 = shift.T + R.T @ plat[i].T
        diff = platR0 - base[i]
        lengths.append(diff[0] ** 2 + diff[1] ** 2 + diff[2] ** 2)
    return lengths

def InverseSGSC(base, plat, shift, s, c, rotationMatrixFunction):
    R, _, _, _ = rotationMatrixFunction(s, c)
    lengths = []
    for i in range(len(base)):
        platR0 = shift.T + R.T @ plat[i].T
        diff = platR0 - base[i]
        lengths.append(diff[0] ** 2 + diff[1] ** 2 + diff[2] ** 2)
    for i in range(len(s)):
        lengths.append(s[i] ** 2 + c[i] ** 2)
    return np.array(lengths)

def getParallelotop(base, plat, lengths):
    interval = [None] * (len(angles) + len(plat[0]))
    for i in range(len(plat[0])):
        dlength = Interval.positive(np.linalg.norm(plat[0])).mid()
        interval[i] = base[0][i] + Interval([-lengths[0][1], lengths[0][1]]) + Interval([-dlength, dlength])
    for j in range(1, len(plat)):
        for i in range(len(plat[0])):     
            dlength = Interval.positive(np.linalg.norm(plat[j])).mid()
            interval[i] = interval[i].intersect(base[j][i] + Interval([-lengths[j][1], lengths[j][1]]) + Interval([-dlength, dlength]))
    for i in range(len(angles)):
        interval[i + len(plat[0])] = Interval([-decpi(), decpi()])
    return interval  

def getParallelotopSC(base, plat, lengths):
    interval = [None] * (len(angles) * 2 + len(plat[0]))
    for i in range(len(plat[0])):
        dlength = Interval.positive(np.linalg.norm(plat[0])).mid()
        interval[i] = base[0][i] + Interval([-lengths[0][1], lengths[0][1]]) + Interval([-dlength, dlength])
    for j in range(1, len(plat)):
        for i in range(len(plat[0])):     
            dlength = Interval.positive(np.linalg.norm(plat[j])).mid()
            interval[i] = interval[i].intersect(base[j][i] + Interval([-lengths[j][1], lengths[j][1]]) + Interval([-dlength, dlength]))
    for i in range(len(angles) * 2):
        interval[i + len(plat[0])] = Interval([-1, 1])
    return interval

def dfs0(i, x, base, plat, rotationMatrixFunction):
    angles = [x[-3], x[-2], x[-1]]
    r, _, _, _ = rotationMatrixFunction(angles)
    b = [0, 0, 0]
    b[0] = x[0] + r[0][0] * plat[i][0] + r[0][1] * plat[i][1] + r[0][2] * plat[i][2]
    b[1] = x[1] + r[1][0] * plat[i][0] + r[1][1] * plat[i][1] + r[1][2] * plat[i][2]
    b[2] = x[2] + r[2][0] * plat[i][0] + r[2][1] * plat[i][1] + r[2][2] * plat[i][2]
    y = Interval.sqrt(Interval.positive((base[i][0] - b[0]) ** 2 + (base[i][1] - b[1]) ** 2 + (base[i][2] - b[2]) ** 2))
    return (base[i][0] - b[0]) / y
def dfs1(i, x, base, plat, rotationMatrixFunction):
    angles = [x[-3], x[-2], x[-1]]
    r, _, _, _ = rotationMatrixFunction(angles)
    b = [0, 0, 0]
    b[0] = x[0] + r[0][0] * plat[i][0] + r[0][1] * plat[i][1] + r[0][2] * plat[i][2]
    b[1] = x[1] + r[1][0] * plat[i][0] + r[1][1] * plat[i][1] + r[1][2] * plat[i][2]
    b[2] = x[2] + r[2][0] * plat[i][0] + r[2][1] * plat[i][1] + r[2][2] * plat[i][2]
    y = Interval.sqrt(Interval.positive((base[i][0] - b[0]) ** 2 + (base[i][1] - b[1]) ** 2 + (base[i][2] - b[2]) ** 2))
    return (base[i][1] - b[1]) / y
def dfs2(i, x, base, plat, rotationMatrixFunction):
    angles = [x[-3], x[-2], x[-1]]
    r, _, _, _ = rotationMatrixFunction(angles)
    b = [0, 0, 0]
    b[0] = x[0] + r[0][0] * plat[i][0] + r[0][1] * plat[i][1] + r[0][2] * plat[i][2]
    b[1] = x[1] + r[1][0] * plat[i][0] + r[1][1] * plat[i][1] + r[1][2] * plat[i][2]
    b[2] = x[2] + r[2][0] * plat[i][0] + r[2][1] * plat[i][1] + r[2][2] * plat[i][2]
    y = Interval.sqrt(Interval.positive((base[i][0] - b[0]) ** 2 + (base[i][1] - b[1]) ** 2 + (base[i][2] - b[2]) ** 2))
    return (base[i][2] - b[2]) / y

def dfa0(i, x, base, plat, rotationMatrixFunction):
    angles = [x[-3], x[-2], x[-1]]
    r, _, _, _ = rotationMatrixFunction(angles)
    b = [0, 0, 0]
    b[0] = x[0] + r[0][0] * plat[i][0] + r[0][1] * plat[i][1] + r[0][2] * plat[i][2]
    b[1] = x[1] + r[1][0] * plat[i][0] + r[1][1] * plat[i][1] + r[1][2] * plat[i][2]
    b[2] = x[2] + r[2][0] * plat[i][0] + r[2][1] * plat[i][1] + r[2][2] * plat[i][2]
    y = Interval.sqrt(Interval.positive((base[i][0] - b[0]) ** 2 + (base[i][1] - b[1]) ** 2 + (base[i][2] - b[2]) ** 2))
    dr = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    dr[0][0] = -Interval.sin(angles[0]) * Interval.cos(angles[2]) - Interval.cos(angles[0]) * Interval.cos(angles[1]) * Interval.sin(angles[2])
    dr[0][1] = Interval.sin(angles[0]) * Interval.sin(angles[2]) - Interval.cos(angles[0]) * Interval.cos(angles[1]) * Interval.cos(angles[2])
    dr[0][2] = Interval.cos(angles[0]) * Interval.sin(angles[1])
    dr[1][0] = Interval.cos(angles[0]) * Interval.cos(angles[2]) - Interval.sin(angles[0]) * Interval.cos(angles[1]) * Interval.sin(angles[2])
    dr[1][1] = -Interval.cos(angles[0]) * Interval.sin(angles[2]) + Interval.sin(angles[0]) * Interval.cos(angles[1]) * Interval.cos(angles[2])
    dr[1][2] = Interval.sin(angles[0]) * Interval.sin(angles[1])
    dr[2][0] = 0
    dr[2][1] = 0
    dr[2][2] = 0
    db = [0, 0, 0]
    db[0] = dr[0][0] * plat[i][0] + dr[0][1] * plat[i][1] + dr[0][2] * plat[i][2]
    db[1] = dr[1][0] * plat[i][0] + dr[1][1] * plat[i][1] + dr[1][2] * plat[i][2]
    db[2] = dr[2][0] * plat[i][0] + dr[2][1] * plat[i][1] + dr[2][2] * plat[i][2]
    return (-db[0] * (base[i][0] - b[0]) - db[1] * (base[i][1] - b[1]) - db[2] * (base[i][2] - b[2])) / y
def dfa1(i, x, base, plat, rotationMatrixFunction):
    angles = [x[-3], x[-2], x[-1]]
    r, _, _, _ = rotationMatrixFunction(angles)
    b = [0, 0, 0]
    b[0] = x[0] + r[0][0] * plat[i][0] + r[0][1] * plat[i][1] + r[0][2] * plat[i][2]
    b[1] = x[1] + r[1][0] * plat[i][0] + r[1][1] * plat[i][1] + r[1][2] * plat[i][2]
    b[2] = x[2] + r[2][0] * plat[i][0] + r[2][1] * plat[i][1] + r[2][2] * plat[i][2]
    y = Interval.sqrt(Interval.positive((base[i][0] - b[0]) ** 2 + (base[i][1] - b[1]) ** 2 + (base[i][2] - b[2]) ** 2))
    dr = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    dr[0][0] = Interval.sin(angles[0]) * Interval.sin(angles[1]) * Interval.sin(angles[2])
    dr[0][1] = Interval.sin(angles[0]) * Interval.sin(angles[1]) * Interval.cos(angles[2])
    dr[0][2] = Interval.sin(angles[0]) * Interval.cos(angles[1])
    dr[1][0] = -Interval.cos(angles[0]) * Interval.sin(angles[1]) * Interval.sin(angles[2])
    dr[1][1] = Interval.cos(angles[0]) * Interval.sin(angles[1]) * Interval.cos(angles[2])
    dr[1][2] = -Interval.cos(angles[0]) * Interval.cos(angles[1])
    dr[2][0] = Interval.cos(angles[1]) * Interval.sin(angles[2])
    dr[2][1] = Interval.cos(angles[1]) * Interval.cos(angles[2])
    dr[2][2] = -Interval.sin(angles[1])
    db = [0, 0, 0]
    db[0] = dr[0][0] * plat[i][0] + dr[0][1] * plat[i][1] + dr[0][2] * plat[i][2]
    db[1] = dr[1][0] * plat[i][0] + dr[1][1] * plat[i][1] + dr[1][2] * plat[i][2]
    db[2] = dr[2][0] * plat[i][0] + dr[2][1] * plat[i][1] + dr[2][2] * plat[i][2]
    return (-db[0] * (base[i][0] - b[0]) - db[1] * (base[i][1] - b[1]) - db[2] * (base[i][2] - b[2])) / y
def dfa2(i, x, base, plat, rotationMatrixFunction):
    angles = [x[-3], x[-2], x[-1]]
    r, _, _, _ = rotationMatrixFunction(angles)
    b = [0, 0, 0]
    b[0] = x[0] + r[0][0] * plat[i][0] + r[0][1] * plat[i][1] + r[0][2] * plat[i][2]
    b[1] = x[1] + r[1][0] * plat[i][0] + r[1][1] * plat[i][1] + r[1][2] * plat[i][2]
    b[2] = x[2] + r[2][0] * plat[i][0] + r[2][1] * plat[i][1] + r[2][2] * plat[i][2]
    y = Interval.sqrt(Interval.positive((base[i][0] - b[0]) ** 2 + (base[i][1] - b[1]) ** 2 + (base[i][2] - b[2]) ** 2))
    dr = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    dr[0][0] = -Interval.cos(angles[0]) * Interval.sin(angles[2]) - Interval.sin(angles[0]) * Interval.cos(angles[1]) * Interval.cos(angles[2])
    dr[0][1] = -Interval.cos(angles[0]) * Interval.cos(angles[2]) + Interval.sin(angles[0]) * Interval.cos(angles[1]) * Interval.sin(angles[2])
    dr[0][2] = 0
    dr[1][0] = -Interval.sin(angles[0]) * Interval.sin(angles[2]) + Interval.cos(angles[0]) * Interval.cos(angles[1]) * Interval.cos(angles[2])
    dr[1][1] = -Interval.sin(angles[0]) * Interval.cos(angles[2]) + Interval.cos(angles[0]) * Interval.cos(angles[1]) * Interval.sin(angles[2])
    dr[1][2] = 0
    dr[2][0] = Interval.sin(angles[1]) * Interval.cos(angles[2])
    dr[2][1] = -Interval.sin(angles[1]) * Interval.sin(angles[2])
    dr[2][2] = 0
    db = [0, 0, 0]
    db[0] = dr[0][0] * plat[i][0] + dr[0][1] * plat[i][1] + dr[0][2] * plat[i][2]
    db[1] = dr[1][0] * plat[i][0] + dr[1][1] * plat[i][1] + dr[1][2] * plat[i][2]
    db[2] = dr[2][0] * plat[i][0] + dr[2][1] * plat[i][1] + dr[2][2] * plat[i][2]
    return (-db[0] * (base[i][0] - b[0]) - db[1] * (base[i][1] - b[1]) - db[2] * (base[i][2] - b[2])) / y

def getJackobian(x, base, plat, rotationMatrixFunction):
    J = []
    for i in range(len(base)):
        J.append([dfs0(i, x, base, plat, rotationMatrixFunction),
                  dfs1(i, x, base, plat, rotationMatrixFunction),
                  dfs2(i, x, base, plat, rotationMatrixFunction),
                  dfa0(i, x, base, plat, rotationMatrixFunction),
                  dfa1(i, x, base, plat, rotationMatrixFunction),
                  dfa2(i, x, base, plat, rotationMatrixFunction)])
    return np.array(J)

def getJackobianINT(x, base, plat, rotationMatrixFunction):
    J = []
    for i in range(len(base)):
        J.append([dfs0(i, x, base, plat, rotationMatrixFunction),
                  dfs1(i, x, base, plat, rotationMatrixFunction),
                  dfs2(i, x, base, plat, rotationMatrixFunction),
                  dfa0(i, x, base, plat, rotationMatrixFunction),
                  dfa1(i, x, base, plat, rotationMatrixFunction),
                  dfa2(i, x, base, plat, rotationMatrixFunction)])
    J = np.array(J)
    for i in range(6):
        for j in range(6):
            J[i][j] = J[i][j].mid()
    return J

def dfds0(i, x, base, plat, r):
    s = [x[-6], x[-5], x[-4]]
    c = [x[-3], x[-2], x[-1]]
    b = [0, 0, 0]
    b[0] = x[0] + r[0][0] * plat[i][0] + r[0][1] * plat[i][1] + r[0][2] * plat[i][2]
    return 2 * (base[i][0] - b[0])
def dfds1(i, x, base, plat, r):
    s = [x[-6], x[-5], x[-4]]
    c = [x[-3], x[-2], x[-1]]
    b = [0, 0, 0]
    b[1] = x[1] + r[1][0] * plat[i][0] + r[1][1] * plat[i][1] + r[1][2] * plat[i][2]
    return 2 * (base[i][1] - b[1])
def dfds2(i, x, base, plat, r):
    s = [x[-6], x[-5], x[-4]]
    c = [x[-3], x[-2], x[-1]]
    b = [0, 0, 0]
    b[2] = x[2] + r[2][0] * plat[i][0] + r[2][1] * plat[i][1] + r[2][2] * plat[i][2]
    return 2 * (base[i][2] - b[2])

def dfdsi0(i, x, base, plat, r):
    s = [x[-6], x[-5], x[-4]]
    c = [x[-3], x[-2], x[-1]]
    b = [0, 0, 0]
    b[0] = x[0] + r[0][0] * plat[i][0] + r[0][1] * plat[i][1] + r[0][2] * plat[i][2]
    b[1] = x[1] + r[1][0] * plat[i][0] + r[1][1] * plat[i][1] + r[1][2] * plat[i][2]
    b[2] = x[2] + r[2][0] * plat[i][0] + r[2][1] * plat[i][1] + r[2][2] * plat[i][2]
    dr = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    dr[0][0] = c[1] * s[2]
    dr[0][1] = -c[1] * c[2]
    dr[0][2] = s[1]
    dr[1][0] = c[2]
    dr[1][1] = -s[2]
    dr[1][2] = 0
    dr[2][0] = 0
    dr[2][1] = 0
    dr[2][2] = 0
    db = [0, 0, 0]
    db[0] = dr[0][0] * plat[i][0] + dr[0][1] * plat[i][1] + dr[0][2] * plat[i][2]
    db[1] = dr[1][0] * plat[i][0] + dr[1][1] * plat[i][1] + dr[1][2] * plat[i][2]
    db[2] = dr[2][0] * plat[i][0] + dr[2][1] * plat[i][1] + dr[2][2] * plat[i][2]
    return 2 * (-db[0] * (base[i][0] - b[0]) - db[1] * (base[i][1] - b[1]) - db[2] * (base[i][2] - b[2]))
def dfdsi1(i, x, base, plat, r):
    s = [x[-6], x[-5], x[-4]]
    c = [x[-3], x[-2], x[-1]]
    b = [0, 0, 0]
    b[0] = x[0] + r[0][0] * plat[i][0] + r[0][1] * plat[i][1] + r[0][2] * plat[i][2]
    b[1] = x[1] + r[1][0] * plat[i][0] + r[1][1] * plat[i][1] + r[1][2] * plat[i][2]
    b[2] = x[2] + r[2][0] * plat[i][0] + r[2][1] * plat[i][1] + r[2][2] * plat[i][2]
    dr = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    dr[0][0] = 0
    dr[0][1] = 0
    dr[0][2] = s[0]
    dr[1][0] = 0
    dr[1][1] = 0
    dr[1][2] = -c[0]
    dr[2][0] = s[2]
    dr[2][1] = c[2]
    dr[2][2] = 0
    db = [0, 0, 0]
    db[0] = dr[0][0] * plat[i][0] + dr[0][1] * plat[i][1] + dr[0][2] * plat[i][2]
    db[1] = dr[1][0] * plat[i][0] + dr[1][1] * plat[i][1] + dr[1][2] * plat[i][2]
    db[2] = dr[2][0] * plat[i][0] + dr[2][1] * plat[i][1] + dr[2][2] * plat[i][2]
    return 2 * (-db[0] * (base[i][0] - b[0]) - db[1] * (base[i][1] - b[1]) - db[2] * (base[i][2] - b[2]))
def dfdsi2(i, x, base, plat, r):
    s = [x[-6], x[-5], x[-4]]
    c = [x[-3], x[-2], x[-1]]
    b = [0, 0, 0]
    b[0] = x[0] + r[0][0] * plat[i][0] + r[0][1] * plat[i][1] + r[0][2] * plat[i][2]
    b[1] = x[1] + r[1][0] * plat[i][0] + r[1][1] * plat[i][1] + r[1][2] * plat[i][2]
    b[2] = x[2] + r[2][0] * plat[i][0] + r[2][1] * plat[i][1] + r[2][2] * plat[i][2]
    dr = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    dr[0][0] = -s[0] * c[1]
    dr[0][1] = -c[0]
    dr[0][2] = 0
    dr[1][0] = c[0] * c[1]
    dr[1][1] = -s[0]
    dr[1][2] = 0
    dr[2][0] = s[1]
    dr[2][1] = 0
    dr[2][2] = 0
    db = [0, 0, 0]
    db[0] = dr[0][0] * plat[i][0] + dr[0][1] * plat[i][1] + dr[0][2] * plat[i][2]
    db[1] = dr[1][0] * plat[i][0] + dr[1][1] * plat[i][1] + dr[1][2] * plat[i][2]
    db[2] = dr[2][0] * plat[i][0] + dr[2][1] * plat[i][1] + dr[2][2] * plat[i][2]
    return 2 * (-db[0] * (base[i][0] - b[0]) - db[1] * (base[i][1] - b[1]) - db[2] * (base[i][2] - b[2]))

def dfdco0(i, x, base, plat, r):
    s = [x[-6], x[-5], x[-4]]
    c = [x[-3], x[-2], x[-1]]
    b = [0, 0, 0]
    b[0] = x[0] + r[0][0] * plat[i][0] + r[0][1] * plat[i][1] + r[0][2] * plat[i][2]
    b[1] = x[1] + r[1][0] * plat[i][0] + r[1][1] * plat[i][1] + r[1][2] * plat[i][2]
    b[2] = x[2] + r[2][0] * plat[i][0] + r[2][1] * plat[i][1] + r[2][2] * plat[i][2]
    dr = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    dr[0][0] = c[2]
    dr[0][1] = -s[2]
    dr[0][2] = 0
    dr[1][0] = c[1] * s[2]
    dr[1][1] = -c[1] * c[2]
    dr[1][2] = -s[1]
    dr[2][0] = 0
    dr[2][1] = 0
    dr[2][2] = 0
    db = [0, 0, 0]
    db[0] = dr[0][0] * plat[i][0] + dr[0][1] * plat[i][1] + dr[0][2] * plat[i][2]
    db[1] = dr[1][0] * plat[i][0] + dr[1][1] * plat[i][1] + dr[1][2] * plat[i][2]
    db[2] = dr[2][0] * plat[i][0] + dr[2][1] * plat[i][1] + dr[2][2] * plat[i][2]
    return 2 * (-db[0] * (base[i][0] - b[0]) - db[1] * (base[i][1] - b[1]) - db[2] * (base[i][2] - b[2]))
def dfdco1(i, x, base, plat, r):
    s = [x[-6], x[-5], x[-4]]
    c = [x[-3], x[-2], x[-1]]
    b = [0, 0, 0]
    b[0] = x[0] + r[0][0] * plat[i][0] + r[0][1] * plat[i][1] + r[0][2] * plat[i][2]
    b[1] = x[1] + r[1][0] * plat[i][0] + r[1][1] * plat[i][1] + r[1][2] * plat[i][2]
    b[2] = x[2] + r[2][0] * plat[i][0] + r[2][1] * plat[i][1] + r[2][2] * plat[i][2]
    dr = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    dr[0][0] = -s[0] * s[2]
    dr[0][1] = -s[0] * c[2]
    dr[0][2] = 0
    dr[1][0] = c[0] * s[2]
    dr[1][1] = -c[0] * c[2]
    dr[1][2] = 0
    dr[2][0] = 0
    dr[2][1] = 0
    dr[2][2] = 1
    db = [0, 0, 0]
    db[0] = dr[0][0] * plat[i][0] + dr[0][1] * plat[i][1] + dr[0][2] * plat[i][2]
    db[1] = dr[1][0] * plat[i][0] + dr[1][1] * plat[i][1] + dr[1][2] * plat[i][2]
    db[2] = dr[2][0] * plat[i][0] + dr[2][1] * plat[i][1] + dr[2][2] * plat[i][2]
    return 2 * (-db[0] * (base[i][0] - b[0]) - db[1] * (base[i][1] - b[1]) - db[2] * (base[i][2] - b[2]))
def dfdco2(i, x, base, plat, r):
    s = [x[-6], x[-5], x[-4]]
    c = [x[-3], x[-2], x[-1]]
    b = [0, 0, 0]
    b[0] = x[0] + r[0][0] * plat[i][0] + r[0][1] * plat[i][1] + r[0][2] * plat[i][2]
    b[1] = x[1] + r[1][0] * plat[i][0] + r[1][1] * plat[i][1] + r[1][2] * plat[i][2]
    b[2] = x[2] + r[2][0] * plat[i][0] + r[2][1] * plat[i][1] + r[2][2] * plat[i][2]
    dr = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    dr[0][0] = c[0]
    dr[0][1] = -s[0] * c[1]
    dr[0][2] = 0
    dr[1][0] = s[0]
    dr[1][1] = -c[0] * c[1]
    dr[1][2] = 0
    dr[2][0] = 0
    dr[2][1] = s[1]
    dr[2][2] = 0
    db = [0, 0, 0]
    db[0] = dr[0][0] * plat[i][0] + dr[0][1] * plat[i][1] + dr[0][2] * plat[i][2]
    db[1] = dr[1][0] * plat[i][0] + dr[1][1] * plat[i][1] + dr[1][2] * plat[i][2]
    db[2] = dr[2][0] * plat[i][0] + dr[2][1] * plat[i][1] + dr[2][2] * plat[i][2]
    return 2 * (-db[0] * (base[i][0] - b[0]) - db[1] * (base[i][1] - b[1]) - db[2] * (base[i][2] - b[2]))

def getJackobianSC(x, base, plat, rotationMatrixFunction):
    s = [x[-6], x[-5], x[-4]]
    c = [x[-3], x[-2], x[-1]]
    R, _, _, _ = rotationMatrixFunction(s, c)
    J = []
    c0 = Interval.valueToInterval(0)
    for i in range(6):
        J.append([dfds0(i, x, base, plat, R),
                  dfds1(i, x, base, plat, R),
                  dfds2(i, x, base, plat, R),
                  dfdsi0(i, x, base, plat, R),
                  dfdsi1(i, x, base, plat, R),
                  dfdsi2(i, x, base, plat, R),
                  dfdco0(i, x, base, plat, R),
                  dfdco1(i, x, base, plat, R),
                  dfdco2(i, x, base, plat, R)])
    J.append([c0, c0, c0, 2 * x[-6], c0, c0, 2 * x[-3], c0, c0])
    J.append([c0, c0, c0, c0, 2 * x[-5], c0, c0, 2 * x[-2], c0])
    J.append([c0, c0, c0, c0, c0, 2 * x[-4], c0, c0, 2 * x[-1]])

    return np.array(J)


def getJackobianSCINT(x, base, plat, rotationMatrixFunction):
    s = [x[-6], x[-5], x[-4]]
    c = [x[-3], x[-2], x[-1]]
    R, _, _, _ = rotationMatrixFunction(s, c)
    J = []
    c0 = Decimal(0)
    for i in range(6):
        J.append([dfds0(i, x, base, plat, R),
                  dfds1(i, x, base, plat, R),
                  dfds2(i, x, base, plat, R),
                  dfdsi0(i, x, base, plat, R),
                  dfdsi1(i, x, base, plat, R),
                  dfdsi2(i, x, base, plat, R),
                  dfdco0(i, x, base, plat, R),
                  dfdco1(i, x, base, plat, R),
                  dfdco2(i, x, base, plat, R)])
    J.append([c0, c0, c0, 2 * x[-6], c0, c0, 2 * x[-3], c0, c0])
    J.append([c0, c0, c0, c0, 2 * x[-5], c0, c0, 2 * x[-2], c0])
    J.append([c0, c0, c0, c0, c0, 2 * x[-4], c0, c0, 2 * x[-1]])
    J = np.array(J)
    for i in range(6):
        for j in range(9):
            J[i][j] = J[i][j].mid()
    return J
