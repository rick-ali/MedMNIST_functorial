import torch
import math
from math import factorial

def wigner_small_d(l: int, beta: torch.Tensor) -> torch.Tensor:
    """
    Compute the Wigner small-d matrix d^l_{m',m}(beta) of shape (2l+1, 2l+1).
    Uses the standard summation formula with correct factorial bounds.
    """
    m_vals = torch.arange(-l, l+1, dtype=torch.int64, device=beta.device)
    d = torch.zeros((2*l+1, 2*l+1), dtype=torch.cfloat, device=beta.device)
    half = beta / 2
    cos_h = torch.cos(half)
    sin_h = torch.sin(half)
    # normalization factor sqrt((l+m')!(l-m')!(l+m)!(l-m)! )
    for i, mp in enumerate(m_vals):
        for j, m in enumerate(m_vals):
            # summation bounds
            k_min = max(0, (m - mp).item())
            k_max = min((l + m).item(), (l - mp).item())
            term_sum = torch.zeros((), dtype=torch.cfloat, device=beta.device)
            norm = math.sqrt(
                factorial(l+mp.item()) * factorial(l-mp.item()) *
                factorial(l+m.item())  * factorial(l-m.item())
            )
            for k in range(k_min, k_max+1):
                # factorial arguments all non-negative by bounds
                den = (
                    factorial(l+m.item()-k) * factorial(k) *
                    factorial(mp.item()-m.item()+k) * factorial(l-mp.item()-k)
                )
                coef = ((-1)**k) * norm / math.sqrt(den)
                exp_cos = 2*l + m.item() - mp.item() - 2*k
                exp_sin = mp.item() - m.item() + 2*k
                term = coef * (cos_h**exp_cos) * (sin_h**exp_sin)
                term_sum = term_sum + term
            d[i, j] = term_sum
    return d


def so3_representation(l: int,
                       alpha: torch.Tensor,
                       beta: torch.Tensor,
                       gamma: torch.Tensor) -> torch.Tensor:
    """
    Return the SO(3) irreducible matrix D^l(alpha,beta,gamma) (unitary) of shape (2l+1,2l+1).
    """
    d = wigner_small_d(l, beta)
    m_vals = torch.arange(-l, l+1, dtype=alpha.dtype, device=alpha.device)
    e_alpha = torch.exp(-1j * m_vals * alpha)
    e_gamma = torch.exp(-1j * m_vals * gamma)
    D = torch.diag(e_alpha) @ d @ torch.diag(e_gamma)
    return D


def o3_representation(l: int,
                      alpha: torch.Tensor,
                      beta: torch.Tensor,
                      gamma: torch.Tensor,
                      parity: int = 1) -> torch.Tensor:
    """
    O(3) irrep: parity = +1 for proper rotations, -1 for improper.
    """
    return parity * so3_representation(l, alpha, beta, gamma)

# -------------------------
# Test suite

def _random_angles():
    return (torch.rand(()) * 2*math.pi,
            torch.rand(()) * math.pi,
            torch.rand(()) * 2*math.pi)

def test_identity(l=3):
    alpha, beta, gamma = torch.tensor(0.), torch.tensor(0.), torch.tensor(0.)
    D = so3_representation(l, alpha, beta, gamma)
    I = torch.eye(2*l+1, dtype=D.dtype, device=D.device)
    assert torch.allclose(D, I, atol=1e-6), "SO3 identity failed"
    O_pos = o3_representation(l, alpha, beta, gamma, parity=1)
    assert torch.allclose(O_pos, I, atol=1e-6), "O3 identity (parity=1) failed"
    O_neg = o3_representation(l, alpha, beta, gamma, parity=-1)
    assert torch.allclose(O_neg, -I, atol=1e-6), "O3 identity (parity=-1) failed"

def test_unitarity(l=2):
    alpha, beta, gamma = _random_angles()
    D = so3_representation(l, torch.tensor(alpha), torch.tensor(beta), torch.tensor(gamma))
    I = torch.eye(2*l+1, dtype=D.dtype, device=D.device)
    assert torch.allclose(D.conj().transpose(-2, -1) @ D, I, atol=1e-5), "SO3 unitarity failed"

def test_parity_squared(l=2):
    alpha, beta, gamma = _random_angles()
    O = o3_representation(l, torch.tensor(alpha), torch.tensor(beta), torch.tensor(gamma), parity=-1)
    I = torch.eye(2*l+1, dtype=O.dtype, device=O.device)
    assert torch.allclose(O @ O, I, atol=1e-6), "O3 parity square failed"

if __name__ == "__main__":
    test_identity()
    test_unitarity()
    test_parity_squared()
    print("All tests passed.")
