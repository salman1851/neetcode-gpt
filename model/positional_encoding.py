import numpy as np
from numpy.typing import NDArray


class Solution:
    def get_positional_encoding(self, seq_len: int, d_model: int) -> NDArray[np.float64]:
        pe = np.zeros((seq_len, d_model))
        for p in range(seq_len):
            for i in range(d_model):
                denom = 10000**(2*(i//2)/d_model)
                if (i % 2 == 0):
                    pe[p,i] = np.sin(p/denom)
                else:
                    pe[p,i] = np.cos(p/denom)
        return (np.round(pe, 5))