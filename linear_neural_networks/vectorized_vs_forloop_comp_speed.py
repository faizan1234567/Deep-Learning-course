"""In this program, for loop implementation and vectorized implementation will be compared for implementing 
deep neural networks. It turns out that vectorized implementation is much faster than for loop implementation, please see the
demonstration below"""
import time
import math
import numpy as np
import torch 

class Timer:
    def __init__(self):
        self.times = []
        self.start()
    
    def start(self):
        self.tik = time.time()
        return self.tik
    
    def stop(self):
        self.times.append(time.time() - self.tik)
        return self.times[-1]

    def sum(self):
        return math.sum(self.times)
    
    def avg(self):
        return self.times/len(self.times)

    
if __name__ == "__main__":
    n = 10000
    a = np.ones((n, 1))
    b = np.ones((n, 1))
    print("for loop implmentation time")
    result = 0
    timer = Timer()
    for i in range(n):
        result += a[i] + b[i]
    for_duration = timer.stop()
    print('{} obtained in {} ms in for loop implemenation'.format(result.squeeze(), for_duration*1000))

    print('Now checking time for vectorized implementation')
    vec_timer = Timer()
    result = np.sum(a + b)
    vec_duration=  vec_timer.stop()
    print('{} obtained in {} ms in vectorized implemenation'.format(result, vec_duration*1000))

    print('done!!')



    

