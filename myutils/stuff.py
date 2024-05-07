import numpy as np

def bitmask_values(N):
    bitmask = []
    pw=0
    test = True
    while test:
        if N % 2 == 1:
            bitmask.append(2**pw)
        if N // 2 == 0:
            test = False
        N //= 2
        pw += 1
    print(bitmask)
    
    
    