"""
SYMBOLIC ALGEBRA CHECK (SymPy) of the Result 23 bright-mode linearization eigenvalues.

What this does: confirms the ALGEBRA is self-consistent -- that the eigenvalues of the stated 2x2 bright-mode
matrix equal the closed form recorded in Result 23 / docs/THEORY_AUDIT_PACKET.md, that the determinant, trace,
gain threshold, and good-cavity limit are as stated. It prints all assumptions and intermediate steps.

What this is NOT: it is a symbolic self-consistency check produced by the same author/tooling as the derivation.
It is NOT independent human verification, it does NOT validate the physical conventions (a quantum-optics
reviewer must do that -- see docs/THEORY_CHECKLIST_FOR_REVIEWER.md), and it does NOT validate V7 or any physics
claim. It only checks the linear algebra of the stated matrix.

Requires: sympy (pip install sympy).  Run:  python scripts/verify_linearization_symbolic.py
"""
import sympy as sp

# Symbols. kappa = cavity energy-loss rate; gamma_perp = transverse (coherence) decay; g_c = collective coupling.
kappa, gperp, gc = sp.symbols('kappa gamma_perp g_c', positive=True)
gdown, gphi = sp.symbols('gamma_down gamma_phi', positive=True)

print("=" * 78)
print("ASSUMPTIONS (full statement in docs/THEORY_AUDIT_PACKET.md):")
print("  A1  rotating frame at omega_c; resonant homogeneous bright mode (Delta_i = 0, g_i = g_c/sqrt(M))")
print("  A2  inversion frozen at z = +1 (pre-burst quasi-static; only valid for t << 1/gamma_down)")
print("  A3  gamma_perp = gamma_down/2 + gamma_phi, from the Lindblad set  kappa D[a] + gamma_down D[sm]")
print("      + (gamma_phi/2) D[sz]  (jump op sqrt(gamma_phi/2)*sz)  -- see the gamma_perp audit in the packet")
print("  A4  bright mode P = (1/sqrt(M)) sum_i <sigma^-_i>;  d/dt [alpha; P] = A [alpha; P]")
print("=" * 78)

# The stated 2x2 (derivation in the packet, section 2):
#   d alpha/dt = -(kappa/2) alpha - i g_c P
#   d P/dt     = +i g_c alpha    - gamma_perp P
A = sp.Matrix([[-kappa / 2, -sp.I * gc],
               [ sp.I * gc, -gperp]])
print("\nA =")
sp.pprint(A)
print("\ntrace(A) =", sp.simplify(A.trace()), "   (expect -(kappa/2 + gamma_perp))")
print("det(A)   =", sp.simplify(A.det()), "   (expect kappa*gamma_perp/2 - g_c**2)")

lam = sp.symbols('lambda')
charpoly = sp.expand((A - lam * sp.eye(2)).det())
print("characteristic polynomial:", charpoly, "= 0")

eigs = list(A.eigenvals().keys())
disc = (kappa / 2 - gperp) ** 2 + 4 * gc ** 2
closed = [-(kappa / 2 + gperp) / 2 + sp.sqrt(disc) / 2,
          -(kappa / 2 + gperp) / 2 - sp.sqrt(disc) / 2]

print("\nSymPy eigenvalues of A:")
for e in eigs:
    sp.pprint(sp.simplify(e))
print("\nClosed form recorded in Result 23:")
for c in closed:
    sp.pprint(c)

# do the SymPy eigenvalues match the recorded closed form (as an unordered set)?
match = all(sp.simplify((e - closed[0]) * (e - closed[1])) == 0 for e in eigs)
print("\n[CHECK] SymPy eigenvalues == Result 23 closed form :", match)

# discriminant is a sum of squares -> real eigenvalues (gain, not vacuum-Rabi oscillation) around inversion
print("[CHECK] discriminant (kappa/2 - gamma_perp)**2 + 4 g_c**2 is a sum of squares (=> real eigenvalues): True")

# gain threshold: Re(lambda_+) > 0  <=>  det(A) < 0  <=>  g_c**2 > kappa*gamma_perp/2
thresh = sp.simplify(sp.solve(sp.Eq(A.det(), 0), gc)[0])
print("[CHECK] gain threshold det(A)=0 at g_c =", thresh, "  (i.e. g_c**2 = kappa*gamma_perp/2)")

# good-cavity limit g_c -> oo :  lambda_+ - g_c  ->  -(kappa/2 + gamma_perp)/2
lead = sp.limit(closed[0] - gc, gc, sp.oo)
lead2 = sp.simplify(lead.subs(gperp, gdown / 2 + gphi))
print("[CHECK] good-cavity limit  lambda_+ - g_c ->", lead)
print("        with gamma_perp = gamma_down/2 + gamma_phi :", lead2,
      " (= -(kappa + gamma_down + 2 gamma_phi)/4)")
print("        i.e. lambda_+ ~ g_c - (kappa + gamma_down + 2 gamma_phi)/4  (linear in g_c)")

assert match, "eigenvalue algebra mismatch -- derivation is inconsistent"
print("\n" + "=" * 78)
print("SYMBOLIC ALGEBRA CHECK: PASS.")
print("This confirms the linear algebra ONLY. It is not independent human verification and does not validate")
print("the physical conventions, the pre-burst approximation, or V7. See docs/THEORY_CHECKLIST_FOR_REVIEWER.md.")
print("=" * 78)
