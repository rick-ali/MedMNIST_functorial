import torch
import torch.nn as nn

class D4RegularRepresentation(nn.Module):
    def __init__(self, device, dtype=torch.float32):
        super(D4RegularRepresentation, self).__init__()

        # Define the 8x8 matrices for each element in D_4
        self.matrices = {
            # 1
            0: torch.tensor([
                [1, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 1]
            ], device=device, dtype=dtype,requires_grad=False),
            
            # r
            1: torch.tensor([
                [0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 1],
                [0, 0, 0, 0, 1, 0, 0, 0]
            ], device=device, dtype=dtype,requires_grad=False),

            # rr
            2: torch.tensor([
                [0., 0., 1., 0., 0., 0., 0., 0.],
                [0., 0., 0., 1., 0., 0., 0., 0.],
                [1., 0., 0., 0., 0., 0., 0., 0.],
                [0., 1., 0., 0., 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 1., 0.],
                [0., 0., 0., 0., 0., 0., 0., 1.],
                [0., 0., 0., 0., 1., 0., 0., 0.],
                [0., 0., 0., 0., 0., 1., 0., 0.]
            ], device=device, dtype=dtype,requires_grad=False),

            # rrr
            3: torch.tensor([
                [0., 0., 0., 1., 0., 0., 0., 0.],
                [1., 0., 0., 0., 0., 0., 0., 0.],
                [0., 1., 0., 0., 0., 0., 0., 0.],
                [0., 0., 1., 0., 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 1.],
                [0., 0., 0., 0., 1., 0., 0., 0.],
                [0., 0., 0., 0., 0., 1., 0., 0.],
                [0., 0., 0., 0., 0., 0., 1., 0.]
            ], device=device, dtype=dtype,requires_grad=False),

            # srr
            4: torch.tensor([
                [0., 0., 1., 0., 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 1.],
                [1., 0., 0., 0., 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 1., 0.],
                [0., 1., 0., 0., 0., 0., 0., 0.],
                [0., 0., 0., 1., 0., 0., 0., 0.],
                [0., 0., 0., 0., 1., 0., 0., 0.],
                [0., 0., 0., 0., 0., 1., 0., 0.]
            ], device=device, dtype=dtype,requires_grad=False),

            # s
            5: torch.tensor([
                [1, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 1]
            ], device=device, dtype=dtype,requires_grad=False),

            # srrr
            6: torch.tensor([
                [0., 0., 0., 1., 0., 0., 0., 0.],
                [0., 0., 0., 0., 1., 0., 0., 0.],
                [0., 1., 0., 0., 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 1.],
                [0., 0., 1., 0., 0., 0., 0., 0.],
                [1., 0., 0., 0., 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 1., 0., 0.],
                [0., 0., 0., 0., 0., 0., 1., 0.]
            ], device=device, dtype=dtype,requires_grad=False),

            # sr
            7: torch.tensor([
                [0., 1., 0., 0., 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 1., 0.],
                [0., 0., 0., 1., 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 1., 0., 0.],
                [1., 0., 0., 0., 0., 0., 0., 0.],
                [0., 0., 1., 0., 0., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 1.],
                [0., 0., 0., 0., 1., 0., 0., 0.]
            ], device=device, dtype=dtype,requires_grad=False),
        }

    def forward(self, x):
        """
        Given a number x (0 to 7), return the corresponding 8x8 matrix.
        """
        return self.matrices[x]
    
    def mapping(self) -> dict:
        mapping = {0:'1', 1:'r', 2:'rr', 3:'rrr', 4:'srr', 5:'s', 6:'srrr', 7:'sr'}
        return mapping
    

class D8RegularRepresentation(nn.Module):
    def __init__(self, device, dtype=torch.float32):
        super(D8RegularRepresentation, self).__init__()

        self.matrices = self.get_d16_regular_representation(device, dtype)

    # -- Group Multiplication Function for D_{16} --
    # Ordering: 
    #   Indices 0..7       : r^0, r^1, ..., r^7
    #   Indices 8..15      : r^0 s, r^1 s, ..., r^7 s
    def group_mult(self, i, j):
        """
        Multiplies two elements of D_{16} given by their indices.
        Returns the index of the product.
        """
        # Case 1: Both rotations
        if i < 8 and j < 8:
            return (i + j) % 8
        # Case 2: Rotation * Reflection
        elif i < 8 and j >= 8:
            b = j - 8
            return 8 + ((i + b) % 8)
        # Case 3: Reflection * Rotation
        elif i >= 8 and j < 8:
            a = i - 8
            return 8 + ((a - j) % 8)
        # Case 4: Reflection * Reflection
        else:
            a = i - 8
            b = j - 8
            return (a - b) % 8
        
    def d16_inverse(self, i: int) -> int:
        """
        Returns the inverse of a D16 element given its index.
        - For rotations (i in 0..7): inverse is (8-i) mod 8.
        - For reflections (i in 8..15): inverse is itself.
        """
        if i < 8:
            return (-i) % 8  # same as (8 - i) mod 8, since 0's inverse is 0.
        else:
            return i

    # -- Regular Representation --
    # For each group element g (identified by index i), the regular representation
    # is given by a 16x16 permutation matrix P_i acting on the group algebra.
    # The (k, j) entry of P_i is 1 if g * (element with index j) equals 
    # the element with index k, and 0 otherwise.
    def get_d16_regular_representation(self, device, dtype):
        rep = {}
        n = 16  # Order of D16
        for i in range(n):
            M = torch.zeros(n, n, device=device, dtype=dtype, requires_grad=False)
            # For each basis element j, the product g * (element j) is element with index k:
            for j in range(n):
                k = self.group_mult(i, j)
                M[k, j] = 1  # Place a 1 at row k, column j.
            rep[i] = M
        return rep

    def forward(self, x):
        """
        Given a number x (0 to 15), return the corresponding 16x16 matrix.
        """
        return self.matrices[x]
    
    def mapping(self) -> dict:
        #   Indices 0..7       : r^0, r^1, ..., r^7
        #   Indices 8..15      : r^0 s, r^1 s, ..., r^7 s
        mapping = {
            0:'1', 1:'r', 2:'rr', 3:'rrr', 
            4:'rrrr', 5:'rrrrr', 6:'rrrrrr', 7:'rrrrrrr',
            8:'s', 9:'r s', 10:'r^2 s', 11:'r^3 s',
            12:'r^4 s', 13:'r^5 s', 14:'r^6 s', 15:'r^7 s'
        }
        return mapping