import numpy as np

from .afis_utils import Triangular, Trapezoidal, InferiorBorder, SuperiorBorder, Gaussian

# ============================================================================
# Type Detection and Helper Functions
# ============================================================================

def _is_gaussian(mf):
    """Check if a membership function is Gaussian."""
    return isinstance(mf, Gaussian)


def _is_list_format(mf):
    """Check if membership function is in [a,b,c,d] list format."""
    return isinstance(mf, (list, tuple)) and len(mf) == 4


def _needs_numerical(A, B):
    """
    Check if numerical calculation is needed.
    
    Use numerical for any class-based membership function (Gaussian, Triangular, etc.).
    Only use analytical when BOTH inputs are in [a,b,c,d] list format.
    """
    return not (_is_list_format(A) and _is_list_format(B))


def _evaluate_membership(mf, x):
    """
    Evaluate membership function at point(s) x.
    Supports: Gaussian, Triangular, Trapezoidal, InferiorBorder, SuperiorBorder, 
              or [a, b, c, d] list format (trapezoidal).
    
    Parameters:
        mf: membership function (object or list)
        x: float or np.array of points to evaluate
    
    Returns:
        float or np.array of membership values
    """
    # Handle class-based membership functions
    if hasattr(mf, 'pertinence'):
        if isinstance(x, np.ndarray):
            return np.array([mf.pertinence(xi) for xi in x])
        else:
            return mf.pertinence(x)
    
    # Handle [a, b, c, d] list format (trapezoidal)
    if isinstance(mf, (list, tuple)) and len(mf) == 4:
        a, b, c, d = mf
        if isinstance(x, np.ndarray):
            result = np.zeros_like(x, dtype=float)
            # Left slope: (a, b]
            mask_left = (x > a) & (x < b)
            if b != a:
                result[mask_left] = (x[mask_left] - a) / (b - a)
            # Core: [b, c]
            mask_core = (x >= b) & (x <= c)
            result[mask_core] = 1.0
            # Right slope: [c, d)
            mask_right = (x > c) & (x < d)
            if d != c:
                result[mask_right] = (d - x[mask_right]) / (d - c)
            return result
        else:
            if x <= a or x >= d:
                return 0.0
            elif a < x < b:
                return (x - a) / (b - a) if b != a else 1.0
            elif b <= x <= c:
                return 1.0
            elif c < x < d:
                return (d - x) / (d - c) if d != c else 1.0
    
    raise ValueError(f"Unsupported membership function type: {type(mf)}")


def _get_core_bounds(mf):
    """
    Get the core region [underline_a^1, overline_a^1] where μ(x) = 1.
    
    Returns:
        tuple (lower_bound, upper_bound) of the core region
    """
    if isinstance(mf, Gaussian):
        # Gaussian core is a single point
        return (mf.center, mf.center)
    
    elif isinstance(mf, Triangular):
        return (mf.top, mf.top)
    
    elif isinstance(mf, Trapezoidal):
        return (mf.top1, mf.top2)
    
    elif isinstance(mf, InferiorBorder):
        # Core extends from -infinity to top, but we return top as both bounds
        # In practice, use U[0] as lower bound when needed
        return (float('-inf'), mf.top)
    
    elif isinstance(mf, SuperiorBorder):
        # Core extends from top to +infinity
        return (mf.top, float('inf'))
    
    elif isinstance(mf, (list, tuple)) and len(mf) == 4:
        # [a, b, c, d] format: core is [b, c]
        return (mf[1], mf[2])
    
    raise ValueError(f"Unsupported membership function type: {type(mf)}")


def _get_support_bounds(mf, U):
    """
    Get the practical support bounds for a membership function within universe U.
    For Gaussians, uses 4*sigma range (covers ~99.99% of area).
    
    Parameters:
        mf: membership function
        U: tuple (U_min, U_max) - universe of discourse
    
    Returns:
        tuple (lower_bound, upper_bound) of practical support
    """
    if isinstance(mf, Gaussian):
        lower = max(U[0], mf.center - 4 * mf.sigma)
        upper = min(U[1], mf.center + 4 * mf.sigma)
        return (lower, upper)
    
    elif isinstance(mf, Triangular):
        return (max(U[0], mf.ini), min(U[1], mf.end))
    
    elif isinstance(mf, Trapezoidal):
        return (max(U[0], mf.ini), min(U[1], mf.end))
    
    elif isinstance(mf, InferiorBorder):
        return (U[0], min(U[1], mf.end))
    
    elif isinstance(mf, SuperiorBorder):
        return (max(U[0], mf.ini), U[1])
    
    elif isinstance(mf, (list, tuple)) and len(mf) == 4:
        return (max(U[0], mf[0]), min(U[1], mf[3]))
    
    raise ValueError(f"Unsupported membership function type: {type(mf)}")


# ============================================================================
# Numerical Calculations (for Gaussian and generic inputs)
# ============================================================================

def nu_numerical(mf, U, disc=1000):
    """
    Calculate normalized area ν(A) of a fuzzy set numerically.
    
    ν(A) = (1 / |U|) * ∫_U μ_A(u) du
    
    Parameters:
        mf: membership function
        U: tuple (U_min, U_max) - universe of discourse
        disc: discretization points for numerical integration
    
    Returns:
        float: normalized area
    """
    u = np.linspace(U[0], U[1], disc)
    mu_values = _evaluate_membership(mf, u)
    
    # Trapezoidal integration
    area = np.trapezoid(mu_values, u)
    
    # Normalize by universe length
    return area / (U[1] - U[0])


def nu_A_vee_B_numerical(A, B, U, disc=1000):
    """
    Calculate normalized area ν(A ∨ B) numerically using the supremum formula.
    
    The supremum (join) is defined as:
        (A ∨ B)(u) = 1           if u ∈ [min(underline_a^1, underline_b^1), max(overline_a^1, overline_b^1)]
        (A ∨ B)(u) = max(A(u), B(u))  otherwise
    
    Parameters:
        A, B: membership functions
        U: tuple (U_min, U_max) - universe of discourse
        disc: discretization points
    
    Returns:
        float: normalized area ν(A ∨ B)
    """
    # Get core bounds for both sets
    core_A = _get_core_bounds(A)
    core_B = _get_core_bounds(B)
    
    # Combined core region: [min of lower bounds, max of upper bounds]
    combined_core_lower = min(core_A[0], core_B[0])
    combined_core_upper = max(core_A[1], core_B[1])
    
    # Handle infinite bounds (from InferiorBorder/SuperiorBorder)
    if combined_core_lower == float('-inf'):
        combined_core_lower = U[0]
    if combined_core_upper == float('inf'):
        combined_core_upper = U[1]
    
    # Discretize universe
    u = np.linspace(U[0], U[1], disc)
    
    # Evaluate both membership functions
    mu_A = _evaluate_membership(A, u)
    mu_B = _evaluate_membership(B, u)
    
    # Calculate supremum: max in general, 1 in combined core
    supremum = np.maximum(mu_A, mu_B)
    
    # Apply core region (where supremum is 1)
    core_mask = (u >= combined_core_lower) & (u <= combined_core_upper)
    supremum[core_mask] = 1.0
    
    # Trapezoidal integration
    area = np.trapezoid(supremum, u)
    
    # Normalize by universe length
    return area / (U[1] - U[0])


# ============================================================================
# Analytical Calculations (original implementation for trapezoidal inputs)
# ============================================================================

def line_equation(x1, y1, x2, y2):
    slope = (y2 - y1) / (x2 - x1)
    intercept = y1 - slope * x1
    return slope, intercept


def def_integral_line(m, intercept, int_limits):
    return 0.5 * m * (int_limits[1]**2 - int_limits[0]**2) + intercept * (int_limits[1] - int_limits[0])


def find_intersection(m1, int_1, m2, int_2):
    x = (int_2 - int_1) / (m1 - m2)
    y = m1 * x + int_1
    return x, y


def nu_A_vee_B(A, B, U):
    A_crisp_left  = 1 if A[0] == A[1] else 0
    A_crisp_right = 1 if A[2] == A[3] else 0
    B_crisp_left  = 1 if B[0] == B[1] else 0
    B_crisp_right = 1 if B[2] == B[3] else 0

    # Left section
    if A_crisp_left == 0 and B_crisp_left == 0:
        m_A1, int_A1 = line_equation(A[0], 0, A[1], 1)
        m_B1, int_B1 = line_equation(B[0], 0, B[1], 1)
        if (B[0] <= A[0]) and (B[1] <= A[1]):
            Area_left = def_integral_line(m_B1, int_B1, [B[0], B[1]])
        elif (A[0] <= B[0]) and (A[1] <= B[1]):
            Area_left = def_integral_line(m_A1, int_A1, [A[0], A[1]])
        elif (A[0] < B[0]) and (B[1] < A[1]):
            u_c, _ = find_intersection(m_A1, int_A1, m_B1, int_B1)
            Area_left = (def_integral_line(m_A1, int_A1, [A[0], u_c]) +
                         def_integral_line(m_B1, int_B1, [u_c, B[1]]))
        elif (B[0] < A[0]) and (A[1] < B[1]):
            u_c, _ = find_intersection(m_A1, int_A1, m_B1, int_B1)
            Area_left = (def_integral_line(m_B1, int_B1, [B[0], u_c]) +
                         def_integral_line(m_A1, int_A1, [u_c, A[1]]))

    if A_crisp_left == 0 and B_crisp_left == 1:
        m_A1, int_A1 = line_equation(A[0], 0, A[1], 1)
        if (B[0] <= A[0]) and (B[1] <= A[1]):
            Area_left = 0
        elif (A[0] <= B[0]) and (A[1] <= B[1]):
            Area_left = def_integral_line(m_A1, int_A1, [A[0], A[1]])
        elif (A[0] < B[0]) and (B[1] < A[1]):
            Area_left = def_integral_line(m_A1, int_A1, [A[0], B[0]])

    if A_crisp_left == 1 and B_crisp_left == 0:
        m_B1, int_B1 = line_equation(B[0], 0, B[1], 1)
        if (A[0] <= B[0]) and (A[1] <= B[1]):
            Area_left = 0
        elif (B[0] <= A[0]) and (B[1] <= A[1]):
            Area_left = def_integral_line(m_B1, int_B1, [B[0], B[1]])
        elif (B[0] < A[0]) and (A[1] < B[1]):
            Area_left = def_integral_line(m_B1, int_B1, [B[0], A[0]])

    if A_crisp_left == 1 and B_crisp_left == 1:
        Area_left = 0

    # Middle section
    Area_middle = max(A[2], B[2]) - min(A[1], B[1])

    # Right section
    if A_crisp_right == 0 and B_crisp_right == 0:
        m_A2, int_A2 = line_equation(A[2], 1, A[3], 0)
        m_B2, int_B2 = line_equation(B[2], 1, B[3], 0)
        if (B[2] <= A[2]) and (B[3] <= A[3]):
            Area_right = def_integral_line(m_A2, int_A2, [A[2], A[3]])
        elif (A[2] <= B[2]) and (A[3] <= B[3]):
            Area_right = def_integral_line(m_B2, int_B2, [B[2], B[3]])
        elif (A[2] < B[2]) and (B[3] < A[3]):
            u_c, _ = find_intersection(m_A2, int_A2, m_B2, int_B2)
            Area_right = (def_integral_line(m_B2, int_B2, [B[2], u_c]) +
                          def_integral_line(m_A2, int_A2, [u_c, A[3]]))
        elif (B[2] < A[2]) and (A[3] < B[3]):
            u_c, _ = find_intersection(m_A2, int_A2, m_B2, int_B2)
            Area_right = (def_integral_line(m_A2, int_A2, [A[2], u_c]) +
                          def_integral_line(m_B2, int_B2, [u_c, B[3]]))

    if A_crisp_right == 0 and B_crisp_right == 1:
        m_A2, int_A2 = line_equation(A[2], 1, A[3], 0)
        if (A[2] <= B[2]) and (A[3] <= B[3]):
            Area_right = 0
        elif (A[2] <= B[2]) and (B[3] <= A[3]):
            Area_right = def_integral_line(m_A2, int_A2, [B[2], A[3]])
        elif (B[2] <= A[2]) and (B[3] <= A[3]):
            Area_right = def_integral_line(m_A2, int_A2, [A[2], A[3]])

    if A_crisp_right == 1 and B_crisp_right == 0:
        m_B2, int_B2 = line_equation(B[2], 1, B[3], 0)
        if (A[2] <= B[2]) and (A[3] <= B[3]):
            Area_right = def_integral_line(m_B2, int_B2, [B[2], B[3]])
        elif (B[2] <= A[2]) and (B[3] <= A[3]):
            Area_right = 0
        elif (B[2] < A[2]) and (A[3] < B[3]):
            Area_right = def_integral_line(m_B2, int_B2, [A[3], B[3]])

    if A_crisp_right == 1 and B_crisp_right == 1:
        Area_right = 0

    Norm_Area = (1 / (U[1] - U[0])) * (Area_left + Area_middle + Area_right)

    return Norm_Area


# ============================================================================
# Auto-dispatcher: Chooses analytical or numerical based on input types
# ============================================================================

def nu_A_vee_B_auto(A, B, U, disc=1000):
    """
    Smart dispatcher for ν(A ∨ B) calculation.
    
    Automatically chooses between:
    - Analytical calculation (nu_A_vee_B) for trapezoidal [a,b,c,d] inputs
    - Numerical calculation (nu_A_vee_B_numerical) if any input is Gaussian
    
    Parameters:
        A, B: membership functions (Gaussian objects or [a,b,c,d] lists)
        U: tuple (U_min, U_max) - universe of discourse
        disc: discretization points for numerical calculation (default 1000)
    
    Returns:
        float: normalized area ν(A ∨ B)
    """
    if _needs_numerical(A, B):
        return nu_A_vee_B_numerical(A, B, U, disc)
    else:
        # Original analytical calculation for [a,b,c,d] format
        return nu_A_vee_B(A, B, U)

