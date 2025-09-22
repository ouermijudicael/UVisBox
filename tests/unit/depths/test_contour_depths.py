from uvisbox.BandDepths.Stat.contour_banddepth import choose2
import math
import numpy as np

def test_choose2():
    for i in range(2,100):
        assert choose2(i) == math.comb(i, 2), f"choose2({i}) failed"