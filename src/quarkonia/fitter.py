from .gem_solver import QuarkoniumSystem, solve_gem
from .observables import get_mass


def residuals(params, m_1, m_2, pdg_data, r):
    alpha_s, b, c = params
    sys = QuarkoniumSystem(m_1, m_2, alpha_s, b, c)

    evals_s, u_s, _, _ = solve_gem(sys, r, l=0)
    evals_p, u_p, _, _ = solve_gem(sys, r, l=1)

    calc = {
        "(1^1S)": get_mass(evals_s, u_s, r, sys, 0, spin=0, l=0),
        "(1^3S)": get_mass(evals_s, u_s, r, sys, 0, spin=1, l=0),
        "(1^1P)": get_mass(evals_p, u_p, r, sys, 0, spin=0, l=1, j=1),
        "(1^3P_0)": get_mass(evals_p, u_p, r, sys, 0, spin=1, l=1, j=0),
        "(1^3P_1)": get_mass(evals_p, u_p, r, sys, 0, spin=1, l=1, j=1),
        "(1^3P_2)": get_mass(evals_p, u_p, r, sys, 0, spin=1, l=1, j=2),
    }

    return [
        (calc[state] - exp_m) * 1000.0
        for state, exp_m in pdg_data.items()
        if exp_m is not None and state in calc
    ]
