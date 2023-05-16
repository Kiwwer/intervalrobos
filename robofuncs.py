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

def funcSC(x, base, plat, rotationMatrixFunction, lengths):
    shifts = np.array([x[0], x[1], x[2]])
    s = [x[-6], x[-5], x[-4]]
    c = [x[-3], x[-2], x[-1]]
    ls = InverseSGSC(base, plat, shifts, s, c, rotationMatrixFunction)
    lss = ls - lengths
    return ls - lengths

def func(x, base, plat, rotationMatrixFunction, lengths):
    shifts = np.array([x[0], x[1], x[2]])
    a = [x[-3], x[-2], x[-1]]
    ls = InverseSG(base, plat, shifts, a, rotationMatrixFunction)
    lss = ls - lengths
    return ls - lengths

def C1it(i, x, base, plat, lengths):
    s = [x[-6], x[-5], x[-4]]
    c = [x[-3], x[-2], x[-1]]
    a00 = s[i] ** 2
    a10 = c[i] ** 2
    a20 = a00 + a10
    y = a20 - lengths[-3 + i]
    if not Interval([0, 0]).isIn(y):
        return 0
    y = y.intersect(Interval([0, 0]))
    a20 = a20.intersect(y + lengths[-3 + i])
    a00 = a00.intersect(a20 - a10)
    a10 = a10.intersect(a20 - a00)
    c1 = c[i].intersect(Interval.positive(a10) ** 0.5)
    c2 = c[i].intersect(-(Interval.positive(a10) ** 0.5))
    if c1 != Interval(['-INF', '-INF']) and c2 != Interval(['-INF', '-INF']):
        c[i] = c2.union(c1)
    elif c1 != Interval(['-INF', '-INF']):
        c[i] = c1
    else:
        c[i] = c2
    s1 = s[i].intersect(Interval.positive(a00) ** 0.5)
    s2 = s[i].intersect(-(Interval.positive(a00) ** 0.5))
    if s1 != Interval(['-INF', '-INF']) and s2 != Interval(['-INF', '-INF']):
        s[i] = s2.union(s1)
    elif s1 != Interval(['-INF', '-INF']):
        s[i] = s1
    else:
        s[i] = s2
    s[i] = s[i].intersect(x[-6 + i])
    c[i] = c[i].intersect(x[-3 + i])
    x_6 = s[0]  
    x_5 = s[1]  
    x_4 = s[2]
    x_3 = c[0]  
    x_2 = c[1]  
    x_1 = c[2]
    if not x_1.isIn(x[-1]):
        return 0
    if not x_2.isIn(x[-2]):
        return 0
    if not x_3.isIn(x[-3]):
        return 0
    if not x_4.isIn(x[-4]):
        return 0
    if not x_5.isIn(x[-5]):
        return 0
    if not x_6.isIn(x[-6]):
        return 0
    x[-6] = x_6 
    x[-5] = x_5
    x[-4] = x_4
    x[-3] = x_3
    x[-2] = x_2
    x[-1] = x_1
    return x


def C1i(i, x, base, plat, lengths):
    s = [x[-6], x[-5], x[-4]]
    c = [x[-3], x[-2], x[-1]]

    a001 = c[0] * c[2]
    a002 = -s[0] * c[1]
    a003 = a002 * s[2]
    a00 = a001 + a003
    a011 = -c[0] * s[2]
    a012 = -s[0] * c[1]
    a013 = a012 * c[2]
    a01 = a011 + a013
    a02 = s[0] * s[1]
    a101 = s[0] * c[2]
    a102 = c[0] * c[1]
    a103 = a102 * s[2]
    a10 = a101 + a103
    a111 = -s[0] * s[2]
    a112 = c[0] * c[1]
    a113 = a112 * c[2]
    a11 = a111 + a113
    a12 = -c[0] * s[1]
    a20 = s[1] * s[2]
    a21 = s[1] * c[2]
    a22 = c[1]

    a300 = a00 * plat[i][0]
    a301 = a01 * plat[i][1]
    a302 = a02 * plat[i][2]
    a303 = a300 + a301
    a304 = a303 + a302
    a30 = a304 + x[0]
    a310 = a10 * plat[i][0]
    a311 = a11 * plat[i][1]
    a312 = a12 * plat[i][2]
    a313 = a310 + a311
    a314 = a313 + a312
    a31 = a314 + x[1]
    a320 = a20 * plat[i][0]
    a321 = a21 * plat[i][1]
    a322 = a22 * plat[i][2]
    a323 = a320 + a321
    a324 = a323 + a322
    a32 = a324 + x[2]

    a401 = base[i][0] - a30
    a40 = a401 ** 2
    a411 = base[i][1] - a31
    a41 = a411 ** 2
    a421 = base[i][2] - a32
    a42 = a421 ** 2
    
    a51 = a40 + a41
    a5 = a51 + a42
    y = a5 - lengths[i]
    if not Interval([0, 0]).isIn(y):
        return 0
    y = y.intersect(Interval([0, 0]))
    a5 = a5.intersect(y + lengths[i])
    a51 = a51.intersect(a5 - a42)
    a42 = a42.intersect(a5 - a51)
    a41 = a41.intersect(a51 - a40)
    a40 = a40.intersect(a51 - a41)
    
    a421 = a421.intersect(Interval.sqrt(Interval.positive(a42)))
    a411 = a411.intersect(Interval.sqrt(Interval.positive(a41)))
    a401 = a401.intersect(Interval.sqrt(Interval.positive(a40)))
    a32 = a32.intersect(Interval.abs(base[i][2] - a421).union(-Interval.abs(base[i][2] - a421)))
    a31 = a31.intersect(Interval.abs(base[i][1] - a411).union(-Interval.abs(base[i][1] - a411)))
    a30 = a30.intersect(Interval.abs(base[i][0] - a401).union(-Interval.abs(base[i][0] - a401)))

    a324 = a324.intersect(a32 - x[2])
    x2 = x[2].intersect(a32 - a324)
    a314 = a314.intersect(a31 - x[1])
    x1 = x[1].intersect(a31 - a314)
    a304 = a304.intersect(a30 - x[0])
    x0 = x[0].intersect(a30 - a304)

    a323 = a323.intersect(a324 - a322)
    a322 = a322.intersect(a324 - a323)
    a321 = a321.intersect(a323 - a320)
    a320 = a320.intersect(a323 - a321)
    a22 = a22.intersect(a322 / plat[i][2])
    a21 = a21.intersect(a321 / plat[i][1])
    a20 = a20.intersect(a320 / plat[i][0])
    a313 = a313.intersect(a314 - a312)
    a312 = a312.intersect(a314 - a313)
    a311 = a311.intersect(a313 - a310)
    a310 = a310.intersect(a313 - a311)
    a12 = a12.intersect(a312 / plat[i][2])
    a11 = a11.intersect(a311 / plat[i][1])
    a10 = a10.intersect(a310 / plat[i][0])
    a303 = a303.intersect(a304 - a302)
    a302 = a302.intersect(a304 - a303)
    a301 = a301.intersect(a303 - a300)
    a300 = a300.intersect(a303 - a301)
    a02 = a02.intersect(a302 / plat[i][2])
    a01 = a01.intersect(a301 / plat[i][1])
    a00 = a00.intersect(a300 / plat[i][0])

    c[1] = c[1].intersect(a22)
    s[1] = s[1].intersect(a21 / c[2])
    c[2] = c[2].intersect(a21 / s[1])
    s[1] = s[1].intersect(a20 / s[2])
    s[2] = s[2].intersect(a20 / s[1])
    s[1] = s[1].intersect(a12 / -c[0])
    c[0] = c[0].intersect(a12 / -s[1])
    a113 = a113.intersect(a11 - a111)
    a111 = a111.intersect(a11 - a113)
    a112 = a112.intersect(a113 / c[2])
    c[2] = c[2].intersect(a113 / a112)
    c[0] = c[0].intersect(a112 / c[1])
    c[1] = c[1].intersect(a112 / c[0])
    s[0] = s[0].intersect(a111 / -s[2])
    s[2] = s[2].intersect(a111 / -s[0])
    a103 = a103.intersect(a10 - a101)
    a101 = a101.intersect(a10 - a103)
    a102 = a102.intersect(a103 / s[2])
    s[2] = s[2].intersect(a103 / a102)
    c[0] = c[0].intersect(a102 / c[1])
    c[1] = c[1].intersect(a102 / c[0])
    s[0] = s[0].intersect(a101 / c[2])
    c[2] = c[2].intersect(a101 / s[0])
    s[0] = s[0].intersect(a02 / s[1])
    s[1] = s[1].intersect(a02 / s[0])
    a013 = a013.intersect(a01 - a011)
    a011 = a011.intersect(a01 - a013)
    a012 = a012.intersect(a013 / c[2])
    c[2] = c[2].intersect(a013 / a012)
    s[0] = s[0].intersect(a012 / -c[1])
    c[1] = c[1].intersect(a012 / -s[0])
    s[2] = s[2].intersect(a011 / -c[0])
    c[0] = c[0].intersect(a011 / -s[2])
    a003 = a003.intersect(a00 - a001)
    a001 = a001.intersect(a00 - a003)
    a002 = a002.intersect(a003 / s[2])
    s[2] = s[2].intersect(a003 / a002)
    s[0] = s[0].intersect(a002 / -c[1])
    c[1] = c[1].intersect(a002 / -s[0])
    c[2] = c[2].intersect(a001 / c[0])
    c[0] = c[0].intersect(a001 / c[2])
    x_6 = s[0]  
    x_5 = s[1]  
    x_4 = s[2]
    x_3 = c[0]  
    x_2 = c[1]  
    x_1 = c[2]
    if not x0.isIn(x[0]):
        return 0
    if not x1.isIn(x[1]):
        return 1
    if not x2.isIn(x[2]):
        return 2
    if not x_1.isIn(x[-1]):
        return 3
    if not x_2.isIn(x[-2]):
        return 4
    if not x_3.isIn(x[-3]):
        return 5
    if not x_4.isIn(x[-4]):
        return 6
    if not x_5.isIn(x[-5]):
        return 7
    if not x_6.isIn(x[-6]):
        return 8
    x[0] = x0
    x[1] = x1
    x[2] = x2
    x[-6] = x_6 
    x[-5] = x_5
    x[-4] = x_4
    x[-3] = x_3
    x[-2] = x_2
    x[-1] = x_1
    return x

def C1iplus(i, x, base, plat, lengths):
    s = [x[-6], x[-5], x[-4]]
    c = [x[-3], x[-2], x[-1]]

    a1 = s[2] * plat[i][0]
    a2 = c[2] * plat[i][1]
    a3 = s[1] * plat[i][2]
    a4 = c[2] * plat[i][0]
    a5 = s[2] * plat[i][1]
    a6 = c[1] * plat[i][2]
    a7 = a1 + a2
    a8 = a4 - a5
    a9 = c[1] * a7
    a10 = c[0] * a8
    a11 = s[0] * a8
    a12 = s[1] * a7
    a13 = a3 - a9
    a14 = a12 + a6
    a15 = a13 * s[0]
    a16 = -a13 * c[0]
    a17 = a15 + a10
    a18 = a16 + a11

    a001 = a17 + x[0]
    a002 = a18 + x[1]
    a003 = a14 + x[2]

    a401 = base[i][0] - a001
    a40 = a401 ** 2
    a411 = base[i][1] - a002
    a41 = a411 ** 2
    a421 = base[i][2] - a003
    a42 = a421 ** 2
    a51 = a40 + a41
    a66 = a51 + a42
    y = a66 - lengths[i]
    if not Interval([0, 0]).isIn(y):
        return 0
    y = y.intersect(Interval([0, 0]))
    a66 = a66.intersect(y + lengths[i])
    a51 = a51.intersect(a66 - a42)
    a42 = a42.intersect(a66 - a51)
    a41 = a41.intersect(a51 - a40)
    a40 = a40.intersect(a51 - a41)
    
    a421 = a421.intersect(Interval.sqrt(Interval.positive(a42)).union(-Interval.sqrt(Interval.positive(a42))))
    a411 = a411.intersect(Interval.sqrt(Interval.positive(a41)).union(-Interval.sqrt(Interval.positive(a41))))
    a401 = a401.intersect(Interval.sqrt(Interval.positive(a40)).union(-Interval.sqrt(Interval.positive(a40))))
    a003 = a003.intersect(base[i][2] - a421)
    a002 = a002.intersect(base[i][1] - a411)
    a001 = a001.intersect(base[i][0] - a401)

    a14 = a14.intersect(a003 - x[2])
    x2 = x[2].intersect(a003 - a14)
    a18 = a18.intersect(a002 - x[1])
    x1 = x[1].intersect(a002 - a18)
    a17 = a17.intersect(a001 - x[0])
    x0 = x[0].intersect(a001 - a17)

    a16 = a16.intersect(a18 - a11)
    a11 = a11.intersect(a18 - a16)
    a15 = a15.intersect(a17 - a10)
    a10 = a10.intersect(a17 - a15)
    a13 = a13.intersect(a16 / -c[0])
    c[0] = c[0].intersect(a16 / -a13)
    a13 = a13.intersect(a15 / -s[0])
    s[0] = s[0].intersect(a15 / -a13)
    a12 = a12.intersect(a14 - a6)
    a6 = a6.intersect(a14 - a12)
    a9 = a9.intersect(a3 - a13)
    a3 = a3.intersect(a9 + a13)
    a7 = a7.intersect(a12 / s[1])
    a7 = a7.intersect(a9 / c[1])
    c[1] = c[1].intersect(a9 / a7)
    s[1] = s[1].intersect(a12 / a7)
    a8 = a8.intersect(a11 / s[0])
    a8 = a8.intersect(a10 / c[0])
    c[0] = c[0].intersect(a10 / a8)
    s[0] = s[0].intersect(a11 / a8)
    a4 = a4.intersect(a5 + a8)
    a5 = a5.intersect(a4 - a8)
    a1 = a1.intersect(a7 - a2)
    a2 = a2.intersect(a7 - a1)
    c[1] = c[1].intersect(a6 / plat[i][2])
    s[2] = s[2].intersect(a5 / plat[i][1])
    c[2] = c[2].intersect(a4 / plat[i][0])
    s[1] = s[1].intersect(a3 / plat[i][2])
    c[2] = c[2].intersect(a2 / plat[i][1])
    s[2] = s[2].intersect(a1 / plat[i][0])

    x_6 = s[0]  
    x_5 = s[1]  
    x_4 = s[2]
    x_3 = c[0]  
    x_2 = c[1]  
    x_1 = c[2]
    if not x0.isIn(x[0]):
        return 0
    if not x1.isIn(x[1]):
        return 1
    if not x2.isIn(x[2]):
        return 2
    if not x_1.isIn(x[-1]):
        return 3
    if not x_2.isIn(x[-2]):
        return 4
    if not x_3.isIn(x[-3]):
        return 5
    if not x_4.isIn(x[-4]):
        return 6
    if not x_5.isIn(x[-5]):
        return 7
    if not x_6.isIn(x[-6]):
        return 8
    x[0] = x0
    x[1] = x1
    x[2] = x2
    x[-6] = x_6 
    x[-5] = x_5
    x[-4] = x_4
    x[-3] = x_3
    x[-2] = x_2
    x[-1] = x_1
    return x

def C1iANG(i, x, base, plat, lengths):
    a = [x[-3], x[-2], x[-1]]
    s = [Interval.sin(a[0]), Interval.sin(a[1]), Interval.sin(a[2])]
    c = [Interval.cos(a[0]), Interval.cos(a[1]), Interval.cos(a[2])]

    a1 = s[2] * plat[i][0]
    a2 = c[2] * plat[i][1]
    a3 = s[1] * plat[i][2]
    a4 = c[2] * plat[i][0]
    a5 = s[2] * plat[i][1]
    a6 = c[1] * plat[i][2]
    a7 = a1 + a2
    a8 = a4 - a5
    a9 = c[1] * a7
    a10 = c[0] * a8
    a11 = s[0] * a8
    a12 = s[1] * a7
    a13 = a3 - a9
    a14 = a12 + a6
    a15 = a13 * s[0]
    a16 = -a13 * c[0]
    a17 = a15 + a10
    a18 = a16 + a11

    a001 = a17 + x[0]
    a002 = a18 + x[1]
    a003 = a14 + x[2]

    a401 = base[i][0] - a001
    a40 = a401 ** 2
    a411 = base[i][1] - a002
    a41 = a411 ** 2
    a421 = base[i][2] - a003
    a42 = a421 ** 2
    
    a51 = a40 + a41
    a66 = a51 + a42
    y = a66 - lengths[i]
    if not Interval([0, 0]).isIn(y):
        return 0
    y = y.intersect(Interval([0, 0]))
    a66 = a66.intersect(y + lengths[i])
    a51 = a51.intersect(a66 - a42)
    a42 = a42.intersect(a66 - a51)
    a41 = a41.intersect(a51 - a40)
    a40 = a40.intersect(a51 - a41)
    
    a421 = a421.intersect(Interval.sqrt(Interval.positive(a42)).union)
    a411 = a411.intersect(Interval.sqrt(Interval.positive(a41)).union)
    a401 = a401.intersect(Interval.sqrt(Interval.positive(a40)).union(-Interval.sqrt(Interval.positive(a40))))
    a003 = a003.intersect(Interval.abs(base[i][2] - a421).union(-Interval.abs(base[i][2] - a421)))
    a002 = a002.intersect(Interval.abs(base[i][1] - a411).union(-Interval.abs(base[i][1] - a411)))
    a001 = a001.intersect(Interval.abs(base[i][0] - a401).union(-Interval.abs(base[i][0] - a401)))

    a14 = a14.intersect(a003 - x[2])
    x2 = x[2].intersect(a003 - a14)
    a18 = a18.intersect(a002 - x[1])
    x1 = x[1].intersect(a002 - a18)
    a17 = a17.intersect(a001 - x[0])
    x0 = x[0].intersect(a001 - a17)

    if not x0.isIn(x[0]):
        return 0
    if not x1.isIn(x[1]):
        return 1
    if not x2.isIn(x[2]):
        return 2
    x[0] = x0
    x[1] = x1
    x[2] = x2
    return x

def CGS(A, p, b):
    Adiag = np.diag(np.diag(A))
    Adiagi = Interval.inverseDiag(Adiag)
    n, n = A.shape
    Aextdiag = np.array(A)
    for i in range(n):
        Aextdiag[i][i] = Interval.valueToInterval(0)
    return Interval.getVectIntersect(p, Adiagi @ (b - Aextdiag @ p))

def CGE(A, p, b):
    for i in range(len(p) - 1):
        if Interval(['-0', '0']).intersect(A[i][i]) != Interval(['-INF', '-INF']):
            for i in range(len(p)):
                p[i] = Interval(['-INF', 'INF'])
            return p
        for j in range(i + 1, len(p)):
            aa = A[j][i] / A[i][i]
            b[j] = b[j] - aa * b[i]
            for k in range(i + 1, len(p)):
                A[j][k] = A[j][k] - aa * A[i][k]
        for i in range(len(p)):
            ss = Interval(["0", "0"])
            for j in range(i + 1, len(p)):
                ss += A[i][j] * p[j]
            p[i] = (b[i] - ss) / A[i][i]
        return p

def CGSP(A, p, b):
    print(A)
    A0 = Interval.getMid(A)
    A0i = inverse_matrix(A0)
    A1 = A0i @ A
    test = Interval([0, 0])
    b1 = A0i @ b
    return CGS(A1, p, b1)

def CK(f, x, funcJackobian, base, plat, rotationMatrixFunction, lengths):
    x0 = Interval.getVectMid(x)
    M = inverse_matrix(funcJackobian(x0, base, plat, rotationMatrixFunction))
    J = funcJackobian(x, base, plat, rotationMatrixFunction)
    n = len(x)
    I = np.eye(n, n)
    for i in range(n):
        for j in range(n):
            I[i][j] = Interval([I[i][j], I[i][j]])
    jf = I - M @ J
    r = x0 - M @ f(x0, base, plat, rotationMatrixFunction) + jf @ (x - x0)
    return Interval.getVectIntersect(x, r)

def CNGS(f, x, funcJackobian, base, plat, rotationMatrixFunction, lengths):
    x0 = Interval.getVectMid(x)
    A = funcJackobian(x, base, plat, rotationMatrixFunction)
    p = x - x0
    p = CGS(A, p, -np.array(f(x0, base, plat, rotationMatrixFunction, lengths)))
    return Interval.getVectIntersect(x, p + x0)

def CNGE(f, x, funcJackobian, base, plat, rotationMatrixFunction, lengths):
    x0 = Interval.getVectMid(x)
    A = funcJackobian(x, base, plat, rotationMatrixFunction)
    p = x - x0
    p = CGE(A, p, -np.array(f(x0, base, plat, rotationMatrixFunction, lengths)))
    return Interval.getVectIntersect(x, p + x0)

def inverse_matrix(matrix_origin):
    n = matrix_origin.shape[0]
    npeye = np.empty((n, n), dtype=Decimal)
    for i in range(n):
        for j in range(n):
            if i != j:
                npeye[i][j] = Decimal('0')
            else:
                npeye[i][j] = Decimal('1')
    m = np.hstack((matrix_origin, npeye))
    for nrow, row in enumerate(m):
        divider = row[nrow]
        row /= divider
        for lower_row in m[nrow+1:]:
            factor = lower_row[nrow]
            lower_row -= factor * row
    for k in range(n - 1, 0, -1):
        for row_ in range(k - 1, -1, -1):
            if m[row_, k]:
                m[row_, :] -= m[k, :] * m[row_, k]
    return m[:,n:].copy()

def vectorWidthSmartANG(x, base, plat, eps, coef):
    imw = 0
    maxwidth = 0
    mwbest = Decimal("-INF")
    xm = []
    for i in range(len(x)):
        xm.append(x[i].mid())
    xm = np.array(xm)
    dfx = getJackobianINT(xm, base, plat, rotationMatrix3DINT)
    for i in range(len(x)):
        if x[i].width() * coef[i] < eps:
            if (x[i].width() * coef[i] > maxwidth):
                maxwidth = x[i].width() 
            continue
        mw = Decimal("-INF")
        for j in range(len(x)):
            nmw = decabs(x[i].width() * coef[i] * dfx[j][i])
            if nmw > mw:
                mw = nmw
        if (mw > mwbest):
            imw = i
        if (x[i].width() * coef[i] > maxwidth):
            maxwidth = x[i].width() * coef[i]
    return imw, maxwidth

def SimpleSIVIAXSC(x, eps, base, plat, lengths, coef, cntr):
    cntr += 1
    if cntr % 100 == 0:
        print(cntr)
        print(x)
    x0 = np.array(x)
    for i in range(9):
        if (i < 6):
            # x = C1i(i, x, base, plat, lengths)
            x = C1iplus(i, x, base, plat, lengths)
        else:
            x = C1it(i - 6, x, base, plat, lengths)
        if isinstance(x, int):
            return [], cntr
    if cntr % 100 == 0:
        print(x)
    while True:
        x1 = np.array(x)
        x = CNGS(funcSC, x, getJackobianSC, base, plat, rotationMatrix3DSC, lengths)
        for i in range(9):
            if (x[i] == Interval(["-INF", "-INF"])):
                return [], cntr
        if (np.all(x == x1)):
            break
    if cntr % 100 == 0:
        print(x)
    while True:
        x1 = np.array(x)
        x = CNGE(funcSC, x, getJackobianSC, base, plat, rotationMatrix3DSC, lengths)
        for i in range(9):
            if (x[i] == Interval(["-INF", "-INF"])):
                return [], cntr
        if (np.all(x == x1)):
            break
    if cntr % 100 == 0:
        print(x)
    while True:
        x1 = np.array(x)
        x = CK(funcSC, x, getJackobianSC, base, plat, rotationMatrix3DSC, lengths)
        for i in range(9):
            if (x[i] == Interval(["-INF", "-INF"])):
                return [], cntr
        if (np.all(x == x1)):
            break
    if cntr % 100 == 0:
        print(x)
    wtf = funcSC(x, base, plat, rotationMatrix3DSC, lengths)
    for i in range(9):
        if not Interval([0, 0]).isIn(wtf[i]):
            return [], cntr
    i, w = vectorWidthSmart(x, base, plat, eps, coef)
    if (w < eps):
        return [x], cntr
    x1 = np.array(x)
    x1[i] = Interval([x[i][0], x[i].mid()])
    x2 = np.array(x)
    x2[i] = Interval([x[i].mid(), x[i][1]])
    v1, cntr = SimpleSIVIAX1i(x1, eps, base, plat, lengths, coef, cntr)
    v2, cntr = SimpleSIVIAX1i(x2, eps, base, plat, lengths, coef, cntr)
    v = []
    for i in range(len(v1)):
        v.append(v1[i])
    for i in range(len(v2)):
        v.append(v2[i])
    return v, cntr

def SimpleSIVIAX(x, eps, base, plat, lengths, coef, cntr):
    if cntr > 100:
        return [], 101
    cntr += 1
    if cntr % 100 == 0:
        print(cntr)
    x0 = np.array(x)
    for i in range(6):
        x = C1iANG(i, x, base, plat, lengths)
        if isinstance(x, int):
            return [], cntr
    while True:
        x1 = np.array(x)
        x = CNGS(func, x, getJackobian, base, plat, rotationMatrix3D, lengths)
        for i in range(6):
            if (x[i] == Interval(["-INF", "-INF"])):
                return [], cntr
        if (np.all(x == x1)):
            break
    while True:
        x1 = np.array(x)
        x = CNGE(func, x, getJackobian, base, plat, rotationMatrix3D, lengths)
        for i in range(6):
            if (x[i] == Interval(["-INF", "-INF"])):
                return [], cntr
        if (np.all(x == x1)):
            break
    wtf = func(x, base, plat, rotationMatrix3D, lengths)
    for i in range(6):
        if not Interval([0, 0]).isIn(wtf[i]):
            return [], cntr
    i, w = vectorWidthSmart(x, base, plat, eps, coef)
    if (w < eps):
        return [x], cntr
    x1 = np.array(x)
    x1[i] = Interval([x[i][0], x[i].mid()])
    x1[i] = Interval([x[i][0], x1[i].mid()])
    x2 = np.array(x)
    x2[i] = Interval([x1[i][1], x[i][1]])
    v1, cntr = SimpleSIVIAX(x1, eps, base, plat, lengths, coef, cntr)
    v2, cntr = SimpleSIVIAX(x2, eps, base, plat, lengths, coef, cntr)
    v = []
    for i in range(len(v1)):
        v.append(v1[i])
    for i in range(len(v2)):
        v.append(v2[i])
    return v, cntr
