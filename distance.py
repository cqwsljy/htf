import numpy.sin as sin
import numpy.cos as cos
import numpy.arccos as arccos
import numpy as np
import math.pi as pi
def dist(lonA,latA,lonB,LatB):
	R = 6371.004
	MlatA = 90 - latA
	MlatB = 90 -LatB
	MlonA = lonA
	MlonB = lonB
	C = sin(MlatA) * sin(MlonB) * cos(MlonA - MlonB) + cos(MlonA) * cos(MlonB)
	dist = R * arccos(C) * pi / 180
	return dist