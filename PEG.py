import numpy as np

from PE import PE


class PEG:
    def __init__(self, M, N_0, num_PE):
        self.M = M
        self.N_0 = N_0
        self.num_PE = num_PE

        self.PEs = [None] * self.num_PE
        for i in range(self.num_PE):
            self.PEs[i] = PE(self.M, self.N_0)

        self.collect = None

    # Send appropriate A and B data to each PE
    def multiply(self, A_pj_data, B_ji):
        a_dims = np.shape(A_pj_data)
        a_data = A_pj_data

        for i in range(a_dims[0]):
            self.PEs[i % self.num_PE].pu_mult(a_data[i, 0], a_data[i, 1], a_data[i, 2], B_ji)

    # Accumulate partial C across all PEs
    def accum(self):
        p_sum = np.zeros((self.M, self.num_PE))
        for i in range(self.num_PE):
            p_sum = p_sum + self.PEs[i].acc

        return p_sum

    # Reset scratch pad of all PUs
    def rst(self):
        for i in range(self.num_PE):
            self.PEs[i].rst_scratch()
