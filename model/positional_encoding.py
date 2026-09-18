import numpy as np
from numpy.typing import NDArray


class Solution:
    def get_positional_encoding(self, seq_len: int, d_model: int) -> NDArray[np.float64]:
        # pe = np.zeros((seq_len, d_model))
        # for p in range(seq_len):
        #     for i in range(d_model):
        #         denom = 10000**(2*(i//2)/d_model)
        #         if (i % 2 == 0):
        #             pe[p,i] = np.sin(p/denom)
        #         else:
        #             pe[p,i] = np.cos(p/denom)
        
        position = np.arange(seq_len)[:, None]
        div_term = np.exp(np.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe = np.zeros((seq_len, d_model))
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)        

        return (np.round(pe, 5))