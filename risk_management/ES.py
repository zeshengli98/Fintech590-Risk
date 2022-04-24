import numpy as np
import pandas as pd



def ES(ret_arr, alpha = 0.05):
    VaR_t = -np.percentile(ret_arr, alpha*100)
    return_before_alpha = ret_arr[ret_arr < -VaR_t]
    ES = return_before_alpha.mean()
    return -ES
