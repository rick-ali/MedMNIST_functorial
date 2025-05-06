import torch
import torch.nn as nn

class D4RegularRepresentation(nn.Module):
    def __init__(self, device, dtype=torch.float32):
        super(D4RegularRepresentation, self).__init__()
        # Precompute the 8 permutation matrices
        self.matrices = self.get_d4_regular_representation(device, dtype)

    def group_mult(self, i: int, j: int) -> int:
        """
        Multiply elements of D4 by their indices 0..7:
          0..3 : r^0, r^1, r^2, r^3
          4..7 : r^0 s, r^1 s, r^2 s, r^3 s
        Returns index of product g_i * g_j.
        """
        # rotation * rotation
        if i < 4 and j < 4:
            return (i + j) % 4
        # rotation * reflection
        elif i < 4 and j >= 4:
            b = j - 4
            return 4 + ((i + b) % 4)
        # reflection * rotation
        elif i >= 4 and j < 4:
            a = i - 4
            return 4 + ((a - j) % 4)
        # reflection * reflection
        else:
            a = i - 4
            b = j - 4
            return (a - b) % 4

    def d4_inverse(self, i: int) -> int:
        """
        Inverse in D4:
         - rotations (0..3): inverse is (-i) mod 4
         - reflections (4..7): each is its own inverse
        """
        if i < 4:
            return (-i) % 4
        else:
            return i

    def get_d4_regular_representation(self, device, dtype):
        rep = {}
        n = 8  # order of D4
        for i in range(n):
            M = torch.zeros(n, n, device=device, dtype=dtype, requires_grad=False)
            for j in range(n):
                k = self.group_mult(i, j)
                M[k, j] = 1
            rep[i] = M
        return rep

    def forward(self, x: int) -> torch.Tensor:
        """
        Given an index x (0..7), returns the corresponding 8×8 permutation matrix.
        """
        return self.matrices[x]

    def mapping(self) -> dict:
        """
        Human‐readable names for the 8 group elements:
        0:'1', 1:'r', 2:'r^2', 3:'r^3',
        4:'s', 5:'r s', 6:'r^2 s', 7:'r^3 s'
        """
        return {
            0: '1',
            1: 'r',
            2: 'r^2',
            3: 'r^3',
            4: 's',
            5: 'r s',
            6: 'r^2 s',
            7: 'r^3 s',
        }


class D8RegularRepresentation(nn.Module):
    def __init__(self, device, dtype=torch.float32):
        super(D8RegularRepresentation, self).__init__()

        self.matrices = self.get_d8_regular_representation(device, dtype)

    # -- Group Multiplication Function for D_{8} --
    # Ordering: 
    #   Indices 0..7       : r^0, r^1, ..., r^7
    #   Indices 8..15      : r^0 s, r^1 s, ..., r^7 s
    def group_mult(self, i, j):
        """
        Multiplies two elements of D_{8} given by their indices.
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
        
    def d8_inverse(self, i: int) -> int:
        """
        Returns the inverse of a D8 element given its index.
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
    def get_d8_regular_representation(self, device, dtype):
        rep = {}
        n = 16  # Order of D8
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


class C4xC4RegularRepresentation(nn.Module):
    def __init__(self, device=None, dtype=torch.float32):
        """
        Regular representation of C4 x C4 of order 16.
        Elements are pairs (i,j) with i,j in 0..3, indexed as 4*i + j.
        """
        super().__init__()
        self.device = device if device is not None else torch.device('cpu')
        self.dtype  = dtype
        self.matrices = self._build_representation()

    def _index(self, i: int, j: int) -> int:
        """Map pair (i,j) to index in 0..15."""
        return 4 * (i % 4) + (j % 4)

    def group_mult(self, a: int, b: int) -> int:
        """
        Multiply two elements of C4 x C4.
        a, b in 0..15.  Return index of a*b.
        """
        i1, j1 = divmod(a, 4)
        i2, j2 = divmod(b, 4)
        # component-wise addition mod 4
        return self._index(i1 + i2, j1 + j2)

    def inverse(self, a: int) -> int:
        """
        Inverse in C4 x C4 is negation mod 4 in each component.
        """
        i, j = divmod(a, 4)
        return self._index(-i, -j)

    def _build_representation(self):
        """
        Builds a dict of 16 permutation matrices M[a] of size 16×16,
        where M[a] realizes left‐multiplication by the element a.
        """
        rep = {}
        n = 16
        for a in range(n):
            M = torch.zeros(n, n, device=self.device, dtype=self.dtype, requires_grad=False)
            for b in range(n):
                c = self.group_mult(a, b)
                M[c, b] = 1
            rep[a] = M
        return rep

    def forward(self, idx: int) -> torch.Tensor:
        """
        Given an element index in 0..15, returns the corresponding 16×16 matrix.
        """
        return self.matrices[idx]

    def mapping(self) -> dict:
        """
        Returns a human‐readable map from index to group element.
        We'll write elements as r^i s^j.
        """
        mp = {}
        for i in range(4):
            for j in range(4):
                idx = self._index(i, j)
                mp[idx] = f"r^{i} s^{j}"
        return mp


