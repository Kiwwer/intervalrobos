from IntervalLib.py import *
from robofuncs.py import *

# base = [[0, 0, 0], [1, 0, 0], [1, 1, 0], [1, 2, 0], [0, 2, 0], [0, 1, 0]]
# plat = [[0, 0, 0], [1, 0, 0], [1, 1, 0], [1, 2, 0], [0, 2, 0], [0, 1, 0]]
# shift = [-1, 1, 5]
# angles = [1, 1, 1]
# known other possible approx solutions (scipy.optimize): [-2.791, -0.08059,  4.382, -0.4795,  0.2473, 2.662]
# # # # # # # # # # # #
# base = [[1, 0, 0], [2, 0, 1], [2, 1, 2], [2, 2, 3], [1, 2, 4], [0, 2, 5]]
# plat = [[0, 0, 0], [1, 0, 0], [1, 1, 0], [1, 2, 0], [0, 2, 0], [0, 1, 0]]
# shift = [0, 0, 0]
# angles = [0.4, 0.5, 0.6]
# known other possible approx solutions (scipy.optimize): [0.4120, 0.8024, 0.1015, -0.8078, -0.06118, 1.3478]
# # # # # # # # # # # #
# base = [[0, 0, 0], [0, 0.5, 0], [1, 0, 0], [0, 1.5, 0], [0, 1, 0], [1, 0.5, 0]]
# plat = [[0.5, 0.5, 0.5], [1, 1, 0], [0.5, 0, 0.5], [0, 1, 0], [0.6, -0.5, 1], [2, 0.7, 0.1]]
# shift = [1, 1, 5]
# angles = [-1, 5, 6]


base = [[0, 0, 0], [0, 0.5, 0], [1, 0, 0], [0, 1.5, 0], [0, 1, 0], [1, 0.5, 0]]
plat = [[0.1, 0.2, 0.4], [0.1, 0.7, 0.5], [1.1, 0.2, 0.6], [0.1, 1.7, 0.7], [0.1, 1.2, 0.8], [1.1, 0.7, 0.9]]
shift = [1.1, 1.2, 1.3]
angles = [0, 0, 0]
# known other possible approx solutions (scipy.optimize): [-1.746, -1.031, 1.30, 0.157, 0, -0.157]
base, plat, shift, angles = convertToIntervalForm(base, plat, shift, angles)
s = [Interval.sin(angles[0]), Interval.sin(angles[1]), Interval.sin(angles[2])]
c = [Interval.cos(angles[0]), Interval.cos(angles[1]), Interval.cos(angles[2])]
lengths = np.array(InverseSGSC(base, plat, shift, s, c, rotationMatrix3DSC))
parallelotop = getParallelotop3DSC(base, plat, 3, lengths)
parallelotop[-5][0] = Decimal("+0")
parallelotop[2][0] = Decimal("+0")
coef = [1, 1, 1, 1, 1, 1, 1, 1, 1]
Interval.multiintervalmode = 0
Interval.calcprecision = 10
Interval.precision = 5
accuracy = 1
res = SimpleSIVIAXSC(parallelotop, accuracy, base, plat, lengths, coef, 0)
