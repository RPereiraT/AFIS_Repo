import numpy as np
from . import A_vee_B
from .afis_utils import Gaussian, InferiorBorder, SuperiorBorder, Triangular, Trapezoidal

# ============================================================================
# Utility Functions
# ============================================================================

def _is_gaussian(x):
    return isinstance(x, Gaussian)


def _has_gaussian_input(input_list):
    return any(_is_gaussian(inp) for inp in input_list)


def _get_input_core(inp):
    """Extract core bounds [lower, upper] from an input."""
    if _is_gaussian(inp):
        return [inp.center, inp.center]
    return [inp[1], inp[2]]


def _mf_to_abcd(s):
    """Convert a membership function object to [a, b, c, d] list format."""
    if isinstance(s, InferiorBorder):
        return [s.top, s.top, s.top, s.end]
    elif isinstance(s, SuperiorBorder):
        return [s.ini, s.top, s.top, s.top]
    elif isinstance(s, Triangular):
        return [s.ini, s.top, s.top, s.end]
    elif isinstance(s, Trapezoidal):
        return [s.ini, s.top1, s.top2, s.end]
    raise ValueError(f"Unsupported membership function type: {type(s)}")


def format_FN_N_Dim(fuzzy_number_n_dim, ftype='crisp_input'):
    """Convert fuzzy number(s) to [a,b,c,d] list format used by A_vee_B.

    ftype options:
      'crisp_input'    — n-D crisp input list: [a, b] → [[a,a,a,a], [b,b,b,b]]
      'rule_antecedent'— n-D FuzzySet antecedents → [[a,b,c,d], ...]
      'rule_consequent'— single FuzzySet consequent → [a,b,c,d]

    Gaussian membership functions are passed through unchanged.
    """
    if ftype == 'crisp_input':
        return [x if _is_gaussian(x) else [x] * 4 for x in fuzzy_number_n_dim]

    elif ftype == 'rule_antecedent':
        return [
            ant.set if _is_gaussian(ant.set) else _mf_to_abcd(ant.set)
            for ant in fuzzy_number_n_dim
        ]

    elif ftype == 'rule_consequent':
        return _mf_to_abcd(fuzzy_number_n_dim.set)

    return []


def fuzzy_imp(a, b, imp_type='luka', param=1):
    """Implements fuzzy implication I(a, b). a and b can be arrays."""
    a, b = np.asarray(a), np.asarray(b)

    if imp_type == 'luka':
        return np.minimum(1, 1 - a + b)
    elif imp_type == 'godel':
        return np.where(a <= b, 1, b)
    elif imp_type == 'goguen':
        return np.where(a <= b, 1, b / a)
    elif imp_type == 'hamacher':
        return np.where(
            a <= b, 1,
            b * (param + a - param * a) / (b * (param + a - param * a) + a - b)
        )
    elif imp_type == 'dombi':
        eps = 1e-10
        a_s, b_s = np.clip(a, eps, 1 - eps), np.clip(b, eps, 1 - eps)
        diff = np.maximum(((1 - b_s) / b_s) ** param - ((1 - a_s) / a_s) ** param, 0)
        return np.where(a <= b, 1, 1 / (1 + diff ** (1 / param)))

    raise ValueError(f"Unknown implication type: {imp_type}")


def tnorms(scalar, arr, tnorm_type='product'):
    """Implements a t-norm. scalar is a single number, arr is an array."""
    if isinstance(tnorm_type, str):
        if tnorm_type == 'product':
            return scalar * arr
        elif tnorm_type == 'min':
            return np.minimum(scalar, arr)
        elif tnorm_type == 'luka':
            return np.maximum(0, scalar + arr - 1)
        return scalar * arr  # fallback
    else:
        param = tnorm_type[1]
        if tnorm_type[0] == 'hamacher':
            num = scalar * arr
            den = param + (1 - param) * (scalar + arr - scalar * arr)
            return num / np.where(den == 0, 1e-10, den)
        return scalar * arr  # fallback


def aggregation(x, agg_method='avg'):
    """Aggregation function."""
    if isinstance(agg_method, str):
        if agg_method == 'avg':
            return np.mean(x)
        elif agg_method == 'product':
            return np.prod(x)
        elif agg_method == 'min':
            return np.min(x)
        elif agg_method == 'max':
            return np.max(x)
        return np.mean(x)  # fallback
    elif agg_method[0] == 'weighted_avg':
        weights = np.asarray(agg_method[1])
        if weights.sum() == 0:
            return np.mean(x)
        return np.dot(np.asarray(x), weights) / weights.sum()
    return np.mean(x)  # fallback


def membership(FN, U, disc=100):
    """Calculate membership of a fuzzy number [a,b,c,d] over universe U."""
    X = np.linspace(U[0], U[1], disc)
    FN_U = np.zeros(disc)
    core_mask = (X >= FN[1]) & (X <= FN[2])
    FN_U[core_mask] = 1.0
    if FN[0] < FN[1]:
        left_mask = (X >= FN[0]) & (X < FN[1])
        FN_U[left_mask] = (X[left_mask] - FN[0]) / (FN[1] - FN[0])
    if FN[2] < FN[3]:
        right_mask = (X > FN[2]) & (X <= FN[3])
        FN_U[right_mask] = (X[right_mask] - FN[3]) / (FN[2] - FN[3])
    return FN_U


def _parse_set_name(repr_string):
    """Extract fuzzy set name(s) from a FuzzySet string representation."""
    result = []
    for i in range(2, len(repr_string)):
        if repr_string[i:i+2] == ' =':
            result.append(repr_string[i-2:i])
    return result


def cuenta_antecedentes(rule_base):
    ant_names_per_rule = [
        _parse_set_name(str(rule.antecedents)) for rule in rule_base.ruleBase
    ]
    n_dims = len(ant_names_per_rule[0])
    antecedents_by_dim = [
        sorted({row[j] for row in ant_names_per_rule})
        for j in range(n_dims)
    ]
    ant_positions = [
        [antecedents_by_dim[j].index(row[j]) for j in range(n_dims)]
        for row in ant_names_per_rule
    ]
    return antecedents_by_dim, ant_names_per_rule, ant_positions


def _intervals_intersect(intervals_1, intervals_2):
    """Check pairwise intersection between corresponding interval pairs."""
    return [
        not (i1[0] > i2[1] or i2[0] > i1[1])
        for i1, i2 in zip(intervals_1, intervals_2)
    ]


# ============================================================================
# Core Functions
# ============================================================================

def sort_antecedents_spatially(antecedents_list, dimension_index):
    """
    Sort antecedents spatially by core positions for a given dimension.
    Returns: (spatially_sorted_antecedents, cores, alphabetical_to_spatial_map, spatial_to_alphabetical_map)
    """
    seen = set()
    unique_ants = []
    for ant_list in antecedents_list:
        ant = ant_list[dimension_index]
        if id(ant) not in seen:
            unique_ants.append(ant)
            seen.add(id(ant))

    str_order = sorted(range(len(unique_ants)), key=lambda k: str(unique_ants[k]))
    alphabetical = [unique_ants[i] for i in str_order]

    formatted = format_FN_N_Dim(alphabetical, 'rule_antecedent')
    cores_alpha = [ant[1:3] for ant in formatted]

    # Sort spatially by core center (Equation 24: a_1^1 ≤ ā_1^1 < a_2^1 ≤ ā_2^1)
    core_centers = [(c[0] + c[1]) / 2.0 for c in cores_alpha]
    spatial_order = sorted(range(len(core_centers)), key=lambda k: (core_centers[k], cores_alpha[k][0]))
    spatially_sorted = [alphabetical[i] for i in spatial_order]
    cores_spatial = [cores_alpha[i] for i in spatial_order]

    # Validate Equation (24): ā_j^1 < a_{j+1}^1
    for k in range(len(cores_spatial) - 1):
        if cores_spatial[k][1] >= cores_spatial[k + 1][0]:
            raise ValueError(
                f"Equation (24) violation in dimension {dimension_index}: "
                f"Core {k} right ({cores_spatial[k][1]}) >= Core {k+1} left ({cores_spatial[k + 1][0]})"
            )

    alpha_to_spatial = {alpha_pos: spatial_order.index(alpha_pos) for alpha_pos in range(len(alphabetical))}
    spatial_to_alpha = {sp: spatial_order[sp] for sp in range(len(spatial_order))}

    return spatially_sorted, cores_spatial, alpha_to_spatial, spatial_to_alpha


def D_LR(input_cores_matrix_format, antec_cores_by_dim_spatial):
    """
    Calculate D_LR with denominators based on spatial ordering.

    Equations (25)-(29): dl, dr, delta_DL, delta_DR, Den_L, Den_R
    """
    D_L, D_R, delta_DL, delta_DR, Den_L, Den_R = [], [], [], [], [], []

    for j, (C_input, Cj) in enumerate(zip(input_cores_matrix_format, antec_cores_by_dim_spatial)):
        C = [C_input] * len(Cj)
        not_intersecting = ~np.array(_intervals_intersect(C, Cj))

        Cj_left  = np.array([c[0] for c in Cj])
        Cj_right = np.array([c[1] for c in Cj])
        C_left   = np.array([c[0] for c in C])
        C_right  = np.array([c[1] for c in C])

        dl = (Cj_left - C_right) * not_intersecting   # Equation (25)
        dr = (C_left - Cj_right) * not_intersecting   # Equation (26)

        delta_DL.append(np.where(dl >= 0, 1, 0))      # Equation (27)
        delta_DR.append(np.where(dr >= 0, 1, 0))
        D_L.append(dl)
        D_R.append(dr)

        # Equation (28): Den_L = a_j^1 - ā_{j-1}^1
        den_L = np.zeros(len(Cj_left))
        if len(Cj_left) > 1:
            den_L[1:] = Cj_left[1:] - Cj_right[:-1]
            if np.any(den_L[1:] <= 0):
                raise ValueError(f"Den_L non-positive in dimension {j}: {den_L[1:]}, Cores: {Cj}")

        # Equation (29): Den_R = a_{j+1}^1 - ā_j^1
        den_R = np.zeros(len(Cj_left))
        if len(Cj_left) > 1:
            den_R[:-1] = Cj_left[1:] - Cj_right[:-1]
            if np.any(den_R[:-1] <= 0):
                raise ValueError(f"Den_R non-positive in dimension {j}: {den_R[:-1]}, Cores: {Cj}")

        Den_L.append(den_L)
        Den_R.append(den_R)

    return D_L, D_R, delta_DL, delta_DR, Den_L, Den_R


def A_FIS(input, rule_base, agg_method, disc, t_norm_type, imp_params=['luka', 1], A_FIS_type='conjunctive', **kwargs):
    """
    A-FIS: Fuzzy Inference System based on A-subsethood measure.

    For each rule:
    1. Calculate S_A (A-subsethood activation)
    2. Scale consequent membership by S_A using t-norm
    3. Take supremum of all scaled consequents
    """
    num_dims = len(rule_base.inputRanges)
    num_rules = len(rule_base.ruleBase)
    rule_antecedents = [rule.antecedents for rule in rule_base.ruleBase]

    imp_type = imp_params[0]
    imp_param = imp_params[1] if len(imp_params) > 1 else 1
    has_gaussian = _has_gaussian_input(input)

    # Build consistent antecedent indexing per dimension
    antec_fmt_by_dim = []
    cores_by_dim_spatial = []
    spatial_pos_per_rule = []
    alpha_pos_per_rule = []

    for j in range(num_dims):
        unique = list({ant[j] for ant in rule_antecedents})
        str_order = sorted(range(len(unique)), key=lambda k: str(unique[k]))
        alphabetical = [unique[i] for i in str_order]

        antec_fmt_by_dim.append(format_FN_N_Dim(alphabetical, 'rule_antecedent'))

        _, cores_spatial, alpha_to_spatial, _ = sort_antecedents_spatially(rule_antecedents, j)
        cores_by_dim_spatial.append(cores_spatial)

        spatial_dim, alpha_dim = [], []
        for rule_ants in rule_antecedents:
            ant = rule_ants[j]
            try:
                alpha_pos = alphabetical.index(ant)
            except ValueError:
                alpha_pos = next(i for i, a in enumerate(alphabetical) if id(a) == id(ant))
            spatial_dim.append(alpha_to_spatial[alpha_pos])
            alpha_dim.append(alpha_pos)
        spatial_pos_per_rule.append(spatial_dim)
        alpha_pos_per_rule.append(alpha_dim)

    # SVI for each antecedent in each dimension
    svi_by_dim = [
        [
            fuzzy_imp(
                A_vee_B.nu_A_vee_B_auto(input[j], ant, rule_base.inputRanges[j]),
                A_vee_B.nu_A_vee_B_auto(ant, ant, rule_base.inputRanges[j]),
                imp_type, imp_param
            )
            for ant in antec_fmt_by_dim[j]
        ]
        for j in range(num_dims)
    ]

    # Check if input exactly equals any rule's antecedent in ALL dimensions
    if has_gaussian:
        rules_with_exact_match = [False] * num_rules
    else:
        rules_with_exact_match = [
            all(
                input[j] == antec_fmt_by_dim[j][alpha_pos_per_rule[j][rule_idx]]
                for j in range(num_dims)
            )
            for rule_idx in range(num_rules)
        ]

    if any(rules_with_exact_match):
        rule_activations = [1.0 if m else 0.0 for m in rules_with_exact_match]
    else:
        input_cores = [_get_input_core(input[j]) for j in range(num_dims)]
        DL, DR, delta_DL, delta_DR, Den_L, Den_R = D_LR(input_cores, cores_by_dim_spatial)

        rule_activations = []
        for rule_idx in range(num_rules):
            sa_per_dim = []
            for j in range(num_dims):
                alpha_pos = alpha_pos_per_rule[j][rule_idx]
                sp = spatial_pos_per_rule[j][rule_idx]
                n_ants = len(cores_by_dim_spatial[j])

                svi = svi_by_dim[j][alpha_pos]
                dl, dr = DL[j][sp], DR[j][sp]
                d_dl, d_dr = delta_DL[j][sp], delta_DR[j][sp]

                sl = svi * d_dl if sp == 0 else svi * max(1 - dl / Den_L[j][sp], 0) * d_dl
                sr = svi * d_dr if sp == n_ants - 1 else svi * max(1 - dr / Den_R[j][sp], 0) * d_dr
                sa_per_dim.append(max(sl, sr))

            rule_activations.append(aggregation(sa_per_dim, agg_method))

    rule_activations = np.array(rule_activations)

    # Scale each consequent by its activation and take pointwise supremum
    U = np.linspace(rule_base.outputRange[0], rule_base.outputRange[1], disc)
    scaled_outputs = [
        tnorms(
            rule_activations[i],
            membership(format_FN_N_Dim(rule_base.ruleBase[i].consequent, 'rule_consequent'),
                       rule_base.outputRange, disc),
            t_norm_type
        )
        for i in range(num_rules)
    ]
    output = np.maximum.reduce(scaled_outputs)

    # Build activation dict keyed by consequent name (highest activation per consequent)
    cons_names = [_parse_set_name(str(rule.consequent))[0] for rule in rule_base.ruleBase]
    max_values = {}
    for rule_idx, (sa, name) in enumerate(zip(rule_activations, cons_names)):
        current_max, _ = max_values.get(name, (float('-inf'), None))
        if sa > current_max:
            max_values[name] = (sa, rule_idx)
    max_values = dict(sorted(max_values.items(), key=lambda item: item[1], reverse=True))

    return output, U, max_values
