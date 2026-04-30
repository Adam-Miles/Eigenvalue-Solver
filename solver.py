import math
import numpy as np
import sympy as sp


# ============================================================
# Select the example here: 1, 2, or 3
# ============================================================
EXAMPLE = 2


# ============================================================
# Problem data
# ============================================================
x = sp.symbols('x', real=True)

if EXAMPLE == 1:
    # y'' + lambda y = 0, y(0)=0, y'(1)=0
    # Equivalent Sturm-Liouville form: (p y')' + lambda w y = 0
    a = 0.0
    b = 1.0
    p = sp.Integer(1)
    w = sp.Integer(1)

elif EXAMPLE == 2:
    # (x^3 y')' + lambda x y = 0, y(1)=0, y'(2)=0
    a = 1.0
    b = 2.0
    p = x**3
    w = x

elif EXAMPLE == 3:
    # (e^x y')' + lambda e^x y = 0, y(0)=0, y'(1)=0
    a = 0.0
    b = 1.0
    p = sp.exp(x)
    w = sp.exp(x)

else:
    raise ValueError("EXAMPLE must be 1, 2, or 3.")


# ============================================================
# Numerical parameters
# ============================================================
N_STEPS = 20000           # number of grid points for RK4
ANGLE_TOL = 1e-12         # tolerance for g(lambda)
BISECTION_MAX_ITERS = 200
SCAN_POINTS = 400         # used to find a sign change bracket


# ============================================================
# Lambdified coefficient functions
# ============================================================
p_eval = sp.lambdify(x, p, "math")
w_eval = sp.lambdify(x, w, "math")


# ============================================================
#  lower and upper bounds on lam from the manuscript
# valid when p(x),w(x) > 0 on [a,b]
# ============================================================
def compute_bounds():
    I = float(sp.integrate(w, (x, a, b)).evalf())

    f = sp.simplify(p * w)
    df = sp.diff(f, x)

    critical_points = sp.solve(df, x)
    candidates = []

    for cp in critical_points:
        cp_num = sp.N(cp)
        if cp_num.is_real:
            cp_float = float(cp_num)
            if a <= cp_float <= b:
                candidates.append(float(sp.N(f.subs(x, cp_num))))

    candidates.append(float(sp.N(f.subs(x, a))))
    candidates.append(float(sp.N(f.subs(x, b))))

    m = min(candidates)
    M = max(candidates)

    b1 = m * math.pi**2 / (4.0 * I**2)
    b2 = M * math.pi**2 / (4.0 * I**2)

    return b1, b2


# ============================================================
# returns the function F satisfying
#   theta' = lambda w(x) sin^2(theta) + cos^2(theta)/p(x) = F(theta, lam)
# ============================================================
def get_F(lam):
    def F(x_val, theta):
        p_val = p_eval(x_val)
        w_val = w_eval(x_val)
        return lam * w_val * math.sin(theta)**2 + (math.cos(theta)**2) / p_val
    return F

# ============================================================
# Numerically solves theta(b, lam)  using the Runge-Kutta 4 method where 
# theta' =  F(theta, lam) and theta(a, lam) = 0
# ============================================================
def runge_kutta4_theta(lam, n_steps=N_STEPS):
    h = (b - a) / (n_steps - 1)
    theta = 0.0
    x_n = a

    F = get_F(lam)

    for _ in range(n_steps - 1):
        k1 = F(x_n, theta)
        k2 = F(x_n + 0.5 * h, theta + 0.5 * h * k1)
        k3 = F(x_n + 0.5 * h, theta + 0.5 * h * k2)
        k4 = F(x_n + h, theta + h * k3)

        theta += (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        x_n += h

    return theta


def g(lam):
    return runge_kutta4_theta(lam) - (math.pi / 2.0)


# ============================================================
# Checks that the a priori bounds are valid and extends the bounds if necessary
# ============================================================
def find_bracket():
    # Start with a priori bounds if available.
    b_1, b_2 = compute_bounds()

    # Make sure left < right
    if b_1 > b_2:
        b_1, b_2 = b_2, b_1

    # Scan for a sign change on [b_1, b_2] to make sure that g has a zero
    grid = np.linspace(b_1, b_2, SCAN_POINTS)
    vals = [g(float(t)) for t in grid]

    for i in range(len(grid) - 1):
        if vals[i] == 0.0:
            return float(grid[i]), float(grid[i])
        if vals[i] * vals[i + 1] < 0.0:
            return float(grid[i]), float(grid[i + 1])

    # If no sign change was found, enlarge the interval a few times.
    for _ in range(10):
        width = b_2 - b_1
        b_1 = max(0.0, b_1 - width)
        b_2 = b_2 + width
        grid = np.linspace(b_1, b_2, SCAN_POINTS)
        vals = [g(float(t)) for t in grid]

        for i in range(len(grid) - 1):
            if vals[i] == 0.0:
                return float(grid[i]), float(grid[i])
            if vals[i] * vals[i + 1] < 0.0:
                return float(grid[i]), float(grid[i + 1])

    raise RuntimeError("Could not find a sign-change bracket for the first eigenvalue.")


# ============================================================
# Bisection method to estimate lambda_1
# ============================================================
def bisect_root(b_1, b_2, tol=ANGLE_TOL, max_iters=BISECTION_MAX_ITERS):
    g_left = g(b_1)
    g_right = g(b_2)

    if abs(g_left) < tol:
        return b_1
    if abs(g_right) < tol:
        return b_2

    if g_left * g_right > 0.0:
        raise ValueError("Bisection requires a sign-change bracket.")

    for _ in range(max_iters):
        mid = 0.5 * (b_1 + b_2)
        g_mid = g(mid)

        if abs(g_mid) < tol:
            return mid

        if g_left * g_mid < 0.0:
            b_2 = mid
            g_right = g_mid
        else:
            b_1 = mid
            g_left = g_mid

    return 0.5 * (b_1 + b_2)


# ============================================================
# Main routine
# ============================================================
def compute_first_eigenvalue():
    b_1, b_2 = find_bracket()
    lam = bisect_root(b_1, b_2)

    print(f"Example {EXAMPLE}")
    print(f"Bracket: [{b_1:.16f}, {b_2:.16f}]")
    print(f"Approximate first eigenvalue: {lam:.16f}")
    print(f"Residual g(lambda): {g(lam):.3e}")

    if EXAMPLE == 1:
        exact = math.pi**2 / 4.0
        print(f"Exact value:               {exact:.16f}")
        print(f"Absolute error:            {abs(lam - exact):.3e}")

    elif EXAMPLE == 2:
        # Reference value quoted in the report
        reference = 2.8025510350225100
        print(f"Reference value:           {reference:.16f}")
        print(f"Absolute error:            {abs(lam - reference):.3e}")

    elif EXAMPLE == 3:
        # Reference value quoted in the report
        reference = 1.6085328764616391
        print(f"Reference value:           {reference:.16f}")
        print(f"Absolute error:            {abs(lam - reference):.3e}")


if __name__ == "__main__":
    compute_first_eigenvalue()

