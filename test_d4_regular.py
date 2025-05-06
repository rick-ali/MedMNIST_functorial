import pytest
import torch


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



import pytest
import torch

@pytest.fixture(scope="module")
def d4():
    """Fixture to create a D4RegularRepresentation instance."""
    device = torch.device('cpu')
    return D4RegularRepresentation(device)

@pytest.mark.parametrize("i,j,expected", [
    # rotation * rotation
    (1, 1, 2),        # r * r = r^2
    (3, 2, 1),        # r^3 * r^2 = r^5 = r^1
    (2, 3, 1),        # r^2 * r^3 = r^5 = r^1
    (0, 2, 2),        # e * r^2 = r^2
    # rotation * reflection
    (1, 4, 5),        # r * s = r s
    (2, 5, 7),        # r^2 * (r s) = r^3 s
    (3, 7, 6),        # r^3 * (r^3 s) = r^6 s = r^2 s -> index 6
    # reflection * rotation
    (4, 1, 7),        # s * r = r^{-1} s = r^3 s -> index 7
    (5, 2, 7),        # (r s) * r^2 = r^{-1} s = r^3 s -> index 7
    # reflection * reflection
    (4, 4, 0),        # s * s = e
    (5, 7, 2),        # (r s)*(r^3 s) = r^{-2} = r^2
])
def test_group_mult(d4, i, j, expected):
    """Test group multiplication against expected indices."""
    assert d4.group_mult(i, j) == expected

@pytest.mark.parametrize("i,expected", [
    (0, 0),
    (1, 3),
    (2, 2),
    (3, 1),
    (4, 4),
    (5, 5),
    (6, 6),
    (7, 7)
])
def test_inverse(d4, i, expected):
    """Test inverse mapping for all elements."""
    assert d4.d4_inverse(i) == expected


def test_permutation_matrices(d4):
    """Ensure each representation matrix is a valid permutation matrix."""
    for idx, M in d4.matrices.items():
        # entries are 0 or 1
        assert torch.logical_or(M == 0, M == 1).all()
        # each column sums to 1
        col_sums = M.sum(dim=0)
        assert torch.all(col_sums == 1)
        # each row sums to 1
        row_sums = M.sum(dim=1)
        assert torch.all(row_sums == 1)


def test_regular_representation_homomorphism(d4):
    """Test that P_i @ P_j == P_{i * j}."""
    for i in range(8):
        for j in range(8):
            P_i = d4(i)
            P_j = d4(j)
            P_ij = d4(d4.group_mult(i, j))
            # matrix multiplication
            product = P_i.matmul(P_j)
            assert torch.equal(product, P_ij)


def test_mapping_names(d4):
    """Check that mapping() returns correct human-readable labels."""
    mapping = d4.mapping()
    expected = {
        0: '1',
        1: 'r',
        2: 'r^2',
        3: 'r^3',
        4: 's',
        5: 'r s',
        6: 'r^2 s',
        7: 'r^3 s',
    }
    assert mapping == expected

if __name__ == '__main__':
    pytest.main()
