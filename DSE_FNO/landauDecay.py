import math, cmath
import numpy as np
from mpmath import findroot, erfc

def decayRate(k):
    return np.sqrt(2) * float(np.imag(findroot(
        lambda x: 1 + 1 / k ** 2 + 1j * x * cmath.exp(-x ** 2 / (2 * k ** 2)) * erfc(-1j * x / (math.sqrt(2) * k)) / (
            math.sqrt(2 / math.pi) * k ** 3), 0.01j, solver='muller')))

def period(k):
    return 2 * np.pi / float(np.real(findroot(
        lambda x: 1 + 1 / k ** 2 + 1j * x * cmath.exp(-x ** 2 / (2 * k ** 2)) * erfc(-1j * x / (math.sqrt(2) * k)) / (
            math.sqrt(2 / math.pi) * k ** 3), 0.01j, solver='muller')))
