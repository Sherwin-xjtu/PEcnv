import numpy as np
import pandas as pd


a = 2 / 10


def adajustingBaseDepth(index, raw_depth, n1, n2):
    limit2 = len(raw_depth) - (index + n2 +1)
    adjusted_depth = raw_depth[index]
    if len(raw_depth) < n1 or len(raw_depth) < n2:
        raise ValueError("n1 or n2 given must be less than len of the sequence.")
    if index <= n1:
        adjusted_depth = adajustedEdge(index, raw_depth)
    elif limit2 <= 0:
        front_bases = raw_depth[n1:index+1]
        around_avg = np.mean(front_bases)
        adjusted_depth = (1 - a) * adajustingBaseDepth(index - 1, raw_depth, n1, n2) + a * around_avg
    else:
        around_bases =  raw_depth[index + n1:index+n2+1]
        around_avg = np.mean(around_bases)
        adjusted_depth = (1 - a) * adajustingBaseDepth(index - 1, raw_depth, n1, n2) + a * around_avg
    # return (1 - a) * adajustingBaseDepth(index - 1, raw_depth) + a * raw_depth[index]
    return adjusted_depth


def adajustedEdge(index , raw_depth):
    if index <=1:
        return raw_depth[index]
    return (1-a)*adajustedEdge(index-1,raw_depth) + a*raw_depth[index]
    # if 2 < 3:
    #     raise RuntimeError("Bad line from bedcov:\n%r" % Prices)

if __name__ == '__main__':
    real_data = np.concatenate((np.zeros(800), np.ones(200),
                                np.zeros(800), .8*np.ones(200), np.zeros(800)))
    # np.random.seed(0x5EED)
    noisy_data = real_data + np.random.standard_normal(len(real_data)) * .2
    noisy_data_arr = pd.DataFrame(noisy_data)
    noisy_data_li = noisy_data_arr[0].tolist()
    bases_depth = noisy_data_li

    for i in range(len(bases_depth) -1):
        n1, n2 = 1,1
        print(bases_depth[i], adajustingBaseDepth(i, bases_depth, n1, n2))
        print(bases_depth[i], adajustedEdge(i,bases_depth))

