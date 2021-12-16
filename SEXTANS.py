import numpy as np
import math

from PEG import PEG


class SEXTANS:
    def __init__(self, M, P, K_0, N_0, num_PE, D):
        self.M = M
        self.P = P
        self.K_0 = K_0
        self.N_0 = N_0
        self.num_PE = num_PE
        self.D = D

        self.PEGs = [None] * self.P
        for i in range(self.P):
            self.PEGs[i] = PEG(self.M, self.N_0, self.num_PE)

        self.collect = None

    # Complete matrix multiplication for given A and B matrix, alpha, beta, and Cin
    def dot_multiply(self, A, B, alpha, beta, Cin):
        a_dims = np.shape(A)
        b_dims = np.shape(B)
        self.collect = np.zeros((a_dims[0], b_dims[1]))

        if a_dims[0] != self.M:
            print('Row dimension of Sparse matrix is not equal to M value')
            return

        if a_dims[1] != b_dims[0]:
            print('Invalid dimensions for dot multiplication')
            return

        for j in range(math.ceil(a_dims[1]/self.K_0)):                                  # For all K_0 in K columns of A
            A_j = A[:, self.K_0*j:self.K_0*(j+1)]
            A_pjs = np.zeros((self.P, math.ceil(a_dims[0]/self.P), np.shape(A_j)[1]))
            for p in range(a_dims[0]):                                                  # For all PEGs in M create A_pj
                A_pjs[p % self.P, math.floor(p / self.P)] = A_j[p]

            for i in range(math.ceil(b_dims[1]/self.N_0)):                              # For all N_0 in N columns of B create B_ji
                B_ji = B[self.K_0*j:self.K_0*(j+1), self.N_0*i:self.N_0*(i+1)]
                for peg in range(self.P):                                               # For all PEGs send scheduled A_pj and B_ji
                    self.PEGs[peg].multiply(self.schedule(A_pjs[peg], peg), B_ji)

                # Add partial C to collect
                self.collect[:, i*self.N_0:(i+1)*self.N_0] = self.collect[:, i*self.N_0:(i+1)*self.N_0] + self.accum(np.shape(B_ji)[1])
                self.rst()

        return alpha * self.collect + beta * Cin

    # Complete non-zero and out of order scheduling
    def schedule(self, A_pj, p):
        dims = np.shape(A_pj)

        a_data = np.zeros((0, 3))
        for row in range(dims[0]):
            for col in range(dims[1]):
                if A_pj[row, col] != 0:
                    a_val_row_col = np.array([[A_pj[row, col], row*self.P + p, col]])
                    a_data = np.append(a_data, a_val_row_col, axis=0)
        if self.D <= 0:
            return a_data

        row_viable = np.zeros(dims[0])              # Array to track next viable space for each row
        a_sched = np.zeros((dims[0]*dims[1], 3))    # Array to store full schedule
        for d in a_data:
            r = int((d[1]-p)/self.P)
            i = int(row_viable[r])
            entered = False
            while not entered:
                if np.array_equal(a_sched[i], np.zeros(3)):
                    a_sched[i] = d
                    row_viable[r] = i + self.D
                    entered = True
                else:
                    i = i + 1
        return a_sched

    # Accumulate result C across all PEGs
    def accum(self, cols):
        sum = np.zeros((self.M, cols))
        for i in range(self.P):
            for j in range(self.num_PE):
                addend = np.zeros((self.M, cols))
                for k in range(cols):
                    addend[:, k] = self.PEGs[i].PEs[j].PUs[k].scratch
                sum = sum + addend
        return sum

    # Reset scratch pad of all PUs
    def rst(self):
        for i in range(self.P):
            self.PEGs[i].rst()
