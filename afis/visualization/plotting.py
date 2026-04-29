"""
A-FIS Visualization and Diagnostic Tools

This module provides plotting and diagnostic functions for A-FIS:
- Rule base visualization (1D, 2D, 3D)
- Supremum visualization
- Diagnostic tables (SVI, S_A)
- Activation curve analysis
"""

import random
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

try:
    from tqdm.auto import tqdm
except ImportError:
    from tqdm import tqdm

# Import from core module
from ..core.A_FIS import (
    A_FIS, format_FN_N_Dim, cuenta_antecedentes, 
    sort_antecedents_spatially, D_LR, fuzzy_imp
)
from ..core.afis_utils import (
    FuzzyRuleBase, FuzzyRule, FuzzySet, 
    Trapezoidal, Triangular, Gaussian, centroid
)
from ..core.A_vee_B import (
    _evaluate_membership, _get_core_bounds, 
    nu_numerical, nu_A_vee_B_auto
)

# ============================================================================
# Experiment Setup
# ============================================================================

def create_rule_base(antecedents, consequents, input_ranges, output_range, rules, strength=1.0):
    """
    Create a rule base from user-defined antecedents and consequents.
    
    Parameters:
    - antecedents: List of lists, one per dimension. Each inner list contains FuzzySet objects.
                   Example: [[A1_fs, A2_fs, A3_fs], [B1_fs, B2_fs, B3_fs]]
    - consequents: List of FuzzySet objects. Example: [Y1_fs, Y2_fs, Y3_fs]
    - input_ranges: List of (min, max) tuples, one per dimension
    - output_range: (min, max) tuple for output
    - rules: List of (antecedent_indices, consequent_index) tuples.
             Example 1D: [([0], 1), ([1], 0), ([2], 2)]
             Example 3D: [([0,0,0], 0), ([0,0,1], 0), ...]
    - strength: Rule strength (default 1.0)
    
    Returns:
    - FuzzyRuleBase object
    """
    num_dims = len(antecedents)
    
    # Validate inputs
    if len(input_ranges) != num_dims:
        raise ValueError(f"Number of input_ranges ({len(input_ranges)}) must match number of dimensions ({num_dims})")
    
    for dim_idx, dim_antecedents in enumerate(antecedents):
        if not isinstance(dim_antecedents, list) or len(dim_antecedents) == 0:
            raise ValueError(f"Dimension {dim_idx} must have a non-empty list of antecedents")
    
    rule_base = FuzzyRuleBase()
    rule_base.setInputRanges(input_ranges)
    rule_base.setOutputRange(output_range)
    
    # Create rules
    for antecedent_indices, consequent_idx in rules:
        if len(antecedent_indices) != num_dims:
            raise ValueError(f"Rule antecedent_indices {antecedent_indices} must have {num_dims} elements")
        
        # Get antecedents for this rule
        rule_antecedents = []
        for dim_idx, ant_idx in enumerate(antecedent_indices):
            if ant_idx >= len(antecedents[dim_idx]):
                raise ValueError(f"Antecedent index {ant_idx} out of range for dimension {dim_idx} (max: {len(antecedents[dim_idx])-1})")
            rule_antecedents.append(antecedents[dim_idx][ant_idx])
        
        # Get consequent
        if consequent_idx >= len(consequents):
            raise ValueError(f"Consequent index {consequent_idx} out of range (max: {len(consequents)-1})")
        rule_consequent = consequents[consequent_idx]
        
        # Create and append rule
        rule = FuzzyRule(antecedents=rule_antecedents, consequent=rule_consequent, strength=strength)
        rule_base.appendRule(rule)
    
    return rule_base


def run_afis(input_formatted, rule_base, agg_method='avg', disc=100, t_norm_type='product', imp_params=['luka', 1]):
    """
    Run A_FIS and return output, domain, and defuzzified value.
    
    Parameters:
    - input_formatted: Formatted input (from format_FN_N_Dim)
    - rule_base: FuzzyRuleBase object
    - agg_method: Aggregation method ('avg', 'product', 'min', 'max')
    - disc: Discretization
    - t_norm_type: T-norm type ('product', 'min', 'luka')
    - imp_params: Implication parameters
    
    Returns:
    - output: Membership function array
    - U: Output domain values
    - y_crisp: Defuzzified output
    - max_values: Dictionary of activated rules
    """
    # Store imp_params on rule_base so diagnostic functions can use it
    rule_base._last_imp_params = imp_params
    rule_base._last_agg_method = agg_method
    
    output, U, max_values = A_FIS(
        input=input_formatted,
        rule_base=rule_base,
        agg_method=agg_method,
        disc=disc,
        t_norm_type=t_norm_type,
        imp_params=imp_params,
        A_FIS_type='conjunctive'
    )
    y_crisp = centroid(U, output)
    return output, U, y_crisp, max_values

# ============================================================================
# Plotting Functions
# ============================================================================

def plot_results(rule_base, input_data, output, U, y_crisp, title_suffix=""):
    """
    Plot antecedents with inputs and consequents with output.
    Works for any number of dimensions.
    
    Parameters:
    - rule_base: FuzzyRuleBase object
    - input_data: For 1D: single Trapezoidal object
                  For ND: list of Trapezoidal objects (one per dimension)
    - output: Membership function array (fuzzy output)
    - U: Output domain values
    - y_crisp: Defuzzified output value
    - title_suffix: Optional suffix for plot titles
    """
    num_dims = len(rule_base.inputRanges)
    
    # Auto-detect 1D vs ND: if input_data is a single Trapezoidal, wrap it in a list
    if isinstance(input_data, Trapezoidal):
        input_list = [input_data]
        is_1d = True
    else:
        input_list = input_data
        is_1d = (num_dims == 1)
    
    if len(input_list) != num_dims:
        raise ValueError(f"Number of inputs ({len(input_list)}) must match number of dimensions ({num_dims})")
    
    # Plot antecedents with inputs (one plot per dimension)
    for dim_idx in range(num_dims):
        fig = go.Figure()
        
        # Get unique antecedents for this dimension
        antecedents = []
        for rule in rule_base.ruleBase:
            ant = rule.antecedents[dim_idx]
            if ant not in antecedents:
                antecedents.append(ant)
        
        x_domain = np.linspace(rule_base.inputRanges[dim_idx][0], 
                              rule_base.inputRanges[dim_idx][1], 200)
        for ant in antecedents:
            y_membership = [ant.set.pertinence(xi) for xi in x_domain]
            fig.add_trace(go.Scatter(x=x_domain, y=y_membership, 
                                    mode='lines', name=f'{ant.name}', 
                                    line=dict(width=2),
                                    opacity=0.5))
        
        # Add input for this dimension
        input_mf = input_list[dim_idx]
        
        # Check input type and handle appropriately
        is_gaussian = isinstance(input_mf, Gaussian)
        
        if is_gaussian:
            # Gaussian input: always draw as membership function
            y_input = [input_mf.pertinence(xi) for xi in x_domain]
            fig.add_trace(go.Scatter(x=x_domain, y=y_input, 
                                    mode='lines', name=f'Input (Gaussian)', 
                                    line=dict(color='red', width=3)))
        elif hasattr(input_mf, 'ini'):
            # Trapezoidal-style input: check if crisp
            is_crisp = (input_mf.ini == input_mf.top1 == 
                        input_mf.top2 == input_mf.end)
            
            if is_crisp:
                # Draw crisp input as a vertical line
                crisp_value = input_mf.ini
                fig.add_trace(go.Scatter(x=[crisp_value, crisp_value], y=[0, 1], 
                                        mode='lines', name=f'Input (crisp={crisp_value:.2f})', 
                                        line=dict(color='red', width=3)))
            else:
                # Draw fuzzy input as membership function
                y_input = [input_mf.pertinence(xi) for xi in x_domain]
                fig.add_trace(go.Scatter(x=x_domain, y=y_input, 
                                        mode='lines', name='Input', 
                                        line=dict(color='red', width=3)))
        else:
            # Generic membership function with pertinence method
            y_input = [input_mf.pertinence(xi) for xi in x_domain]
            fig.add_trace(go.Scatter(x=x_domain, y=y_input, 
                                    mode='lines', name='Input', 
                                    line=dict(color='red', width=3)))
        
        # Title depends on dimensionality
        if is_1d:
            title = f"Antecedents and Input{title_suffix}"
            xaxis_title = "Input Domain"
        else:
            title = f"Dimension {dim_idx+1} - Antecedents and Input{title_suffix}"
            xaxis_title = f"Input {dim_idx+1} Domain"
        
        fig.update_layout(title=title, xaxis_title=xaxis_title, 
                         yaxis_title="Membership", yaxis_range=[0, 1.1], 
                         height=400, showlegend=True)
        fig.show()
    
    # Plot consequents with output (same for both 1D and ND)
    fig2 = go.Figure()
    consequents = []
    for rule in rule_base.ruleBase:
        cons = rule.consequent
        if cons not in consequents:
            consequents.append(cons)
    
    x_domain_out = np.linspace(rule_base.outputRange[0], 
                              rule_base.outputRange[1], 200)
    for cons in consequents:
        y_membership = [cons.set.pertinence(xi) for xi in x_domain_out]
        fig2.add_trace(go.Scatter(x=x_domain_out, y=y_membership, 
                                 mode='lines', name=f'{cons.name}', 
                                 line=dict(width=2),
                                 opacity=0.5))
    
    fig2.add_trace(go.Scatter(x=U, y=output, mode='lines', 
                             name='Output', line=dict(color='red', width=3)))
    fig2.add_trace(go.Scatter(x=[y_crisp, y_crisp], y=[0, 1], 
                             mode='lines', 
                             name=f'Defuzzified = {y_crisp:.2f}', 
                             line=dict(color='green', width=2, dash='dot')))
    fig2.update_layout(title=f"Consequents and Output{title_suffix}", 
                      xaxis_title="Output Domain", 
                      yaxis_title="Membership", 
                      yaxis_range=[0, 1.1], height=400, showlegend=True)
    fig2.show()


def plot_antecedents_stacked(rule_base, input_data, title="Antecedents by Dimension"):
    """
    Plot all dimensions stacked vertically in a single figure.
    Each row shows antecedents + input for one dimension.
    """
    num_dims = len(rule_base.inputRanges)
    
    # Handle single Trapezoidal input
    if isinstance(input_data, Trapezoidal):
        input_list = [input_data]
    else:
        input_list = input_data
    
    # Create subplots - one row per dimension
    fig = make_subplots(
        rows=num_dims, cols=1,
        subplot_titles=[f"Dimension {i+1}" for i in range(num_dims)],
        vertical_spacing=0.08
    )
    
    # Track which legend entries we've added (to avoid duplicates)
    legend_added = set()
    
    for dim_idx in range(num_dims):
        row = dim_idx + 1
        
        # Get unique antecedents for this dimension
        antecedents = []
        for rule in rule_base.ruleBase:
            ant = rule.antecedents[dim_idx]
            if ant not in antecedents:
                antecedents.append(ant)
        
        # Domain for this dimension
        x_domain = np.linspace(rule_base.inputRanges[dim_idx][0], 
                              rule_base.inputRanges[dim_idx][1], 200)
        
        # Plot each antecedent
        for ant in antecedents:
            y_membership = [ant.set.pertinence(xi) for xi in x_domain]
            show_legend = ant.name not in legend_added
            fig.add_trace(
                go.Scatter(x=x_domain, y=y_membership, 
                          mode='lines', name=ant.name,
                          line=dict(width=2), opacity=0.6,
                          showlegend=show_legend,
                          legendgroup=ant.name),
                row=row, col=1
            )
            legend_added.add(ant.name)
        
        # Plot input for this dimension
        input_trap = input_list[dim_idx]
        is_crisp = (input_trap.ini == input_trap.top1 == 
                    input_trap.top2 == input_trap.end)
        
        input_name = f"Input_{dim_idx+1}"
        show_legend = input_name not in legend_added
        
        if is_crisp:
            crisp_val = input_trap.ini
            fig.add_trace(
                go.Scatter(x=[crisp_val, crisp_val], y=[0, 1],
                          mode='lines', name=input_name,
                          line=dict(color='red', width=3),
                          showlegend=show_legend,
                          legendgroup="inputs"),
                row=row, col=1
            )
        else:
            y_input = [input_trap.pertinence(xi) for xi in x_domain]
            fig.add_trace(
                go.Scatter(x=x_domain, y=y_input,
                          mode='lines', name=input_name,
                          line=dict(color='red', width=3),
                          showlegend=show_legend,
                          legendgroup="inputs"),
                row=row, col=1
            )
        legend_added.add(input_name)
        
        # Update y-axis range for this subplot
        fig.update_yaxes(range=[0, 1.1], row=row, col=1)
    
    # Update layout
    fig.update_layout(
        title=title,
        height=250 * num_dims,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
    )
    
    fig.show()
    return fig


# ============================================================================
# Supremum Visualization (General - works with any membership function type)
# ============================================================================

def plot_supremum(input_mf, antecedent_mf, U, disc=500, imp_type='luka', imp_param=1):
    """
    Visualize the supremum of two fuzzy sets and compute SVI.
    
    General function that works with any membership function type:
    Triangular, Trapezoidal, Gaussian, InferiorBorder, SuperiorBorder, or [a,b,c,d] list.
    
    Parameters:
    - input_mf: Input membership function (e.g., Gaussian, Triangular)
    - antecedent_mf: Antecedent membership function
    - U: Universe of discourse as tuple (U_min, U_max)
    - disc: Discretization points (default 500)
    - imp_type: Implication type for SVI calculation (default 'luka')
    - imp_param: Implication parameter (default 1)
    
    Returns:
    - fig: Plotly figure object
    - svi: Computed SVI value
    """
    u = np.linspace(U[0], U[1], disc)
    
    # Evaluate membership functions
    mu_X = _evaluate_membership(input_mf, u)
    mu_A = _evaluate_membership(antecedent_mf, u)
    
    # Get core bounds
    core_X = _get_core_bounds(input_mf)
    core_A = _get_core_bounds(antecedent_mf)
    
    # Handle infinite bounds
    core_X_lower = max(U[0], core_X[0]) if core_X[0] != float('-inf') else U[0]
    core_X_upper = min(U[1], core_X[1]) if core_X[1] != float('inf') else U[1]
    core_A_lower = max(U[0], core_A[0]) if core_A[0] != float('-inf') else U[0]
    core_A_upper = min(U[1], core_A[1]) if core_A[1] != float('inf') else U[1]
    
    combined_core = (min(core_X_lower, core_A_lower), max(core_X_upper, core_A_upper))
    
    # Calculate supremum
    mu_supremum = np.maximum(mu_X, mu_A)
    core_mask = (u >= combined_core[0]) & (u <= combined_core[1])
    mu_supremum[core_mask] = 1.0
    
    # Calculate ν-measures
    nu_X = nu_numerical(input_mf, U, disc)
    nu_A = nu_numerical(antecedent_mf, U, disc)
    nu_XvA = nu_A_vee_B_auto(input_mf, antecedent_mf, U, disc)
    
    # SVI calculation
    svi = fuzzy_imp(nu_XvA, nu_A, imp_type, imp_param)
    
    # Create figure
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=(
            f"Input X vs Antecedent A",
            f"Supremum (X ∨ A) — ν(X∨A)={nu_XvA:.4f}, ν(A)={nu_A:.4f}"
        ),
        vertical_spacing=0.15
    )
    
    # --- Plot 1: Individual membership functions ---
    fig.add_trace(
        go.Scatter(x=u, y=mu_X, name="X (Input)", 
                   line=dict(color="#2ecc71", width=2)),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=u, y=mu_A, name="A (Antecedent)", 
                   line=dict(color="#3498db", width=2)),
        row=1, col=1
    )
    
    # Mark cores (only if not infinite/very wide)
    if core_X_lower == core_X_upper:
        fig.add_trace(
            go.Scatter(x=[core_X_lower], y=[1], name="Core X", 
                       mode="markers", marker=dict(size=10, color="#2ecc71", symbol="diamond")),
            row=1, col=1
        )
    if core_A_lower == core_A_upper:
        fig.add_trace(
            go.Scatter(x=[core_A_lower], y=[1], name="Core A", 
                       mode="markers", marker=dict(size=10, color="#3498db", symbol="diamond")),
            row=1, col=1
        )
    
    # --- Plot 2: Supremum with filled area ---
    fig.add_trace(
        go.Scatter(x=u, y=mu_supremum, name="X ∨ A (Supremum)",
                   fill="tozeroy", fillcolor="rgba(155, 89, 182, 0.3)",
                   line=dict(color="#9b59b6", width=2)),
        row=2, col=1
    )
    
    # Add reference lines for X and A
    fig.add_trace(
        go.Scatter(x=u, y=mu_X, name="X", 
                   line=dict(color="#2ecc71", width=1, dash="dash"), showlegend=False),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(x=u, y=mu_A, name="A", 
                   line=dict(color="#3498db", width=1, dash="dash"), showlegend=False),
        row=2, col=1
    )
    
    # Highlight combined core region (if it's a visible interval)
    if combined_core[1] - combined_core[0] > (U[1] - U[0]) * 0.01:
        fig.add_vrect(
            x0=combined_core[0], x1=combined_core[1],
            fillcolor="rgba(241, 196, 15, 0.2)", line_width=0,
            annotation_text="Core", annotation_position="top",
            row=2, col=1
        )
    
    # Layout
    fig.update_layout(
        height=700,
        title_text=f"SVI = I({nu_XvA:.4f}, {nu_A:.4f}) = {svi:.4f}",
        title_x=0.5,
        template="plotly_white"
    )
    
    fig.update_xaxes(title_text="Universe U", row=2, col=1)
    fig.update_yaxes(title_text="μ(u)", range=[0, 1.1])
    
    return fig, svi


def plot_antecedents_3d(rule_base, input_data, title="3D View: Rules and Input"):
    """
    3D visualization showing each rule's antecedent region and the input point.
    Only works for 3-dimensional rule bases.
    """
    num_dims = len(rule_base.inputRanges)
    if num_dims != 3:
        raise ValueError(f"3D plot requires exactly 3 dimensions, got {num_dims}")
    
    input_list = [input_data] if isinstance(input_data, Trapezoidal) else input_data
    
    fig = go.Figure()
    
    colors = ['blue', 'green', 'orange', 'purple', 'cyan', 'magenta', 'yellow', 'brown']
    
    for rule_idx, rule in enumerate(rule_base.ruleBase):
        x_core = [rule.antecedents[0].set.top1, rule.antecedents[0].set.top2]
        y_core = [rule.antecedents[1].set.top1, rule.antecedents[1].set.top2]
        z_core = [rule.antecedents[2].set.top1, rule.antecedents[2].set.top2]
        
        x = [x_core[0], x_core[1], x_core[1], x_core[0], x_core[0], x_core[1], x_core[1], x_core[0]]
        y = [y_core[0], y_core[0], y_core[1], y_core[1], y_core[0], y_core[0], y_core[1], y_core[1]]
        z = [z_core[0], z_core[0], z_core[0], z_core[0], z_core[1], z_core[1], z_core[1], z_core[1]]
        
        color = colors[rule_idx % len(colors)]
        ant_names = f"{rule.antecedents[0].name}, {rule.antecedents[1].name}, {rule.antecedents[2].name}"
        
        fig.add_trace(go.Mesh3d(
            x=x, y=y, z=z,
            i=[0, 0, 0, 0, 4, 4, 2, 2, 0, 0, 1, 1],
            j=[1, 2, 4, 5, 5, 6, 3, 6, 1, 4, 2, 5],
            k=[2, 3, 5, 6, 6, 7, 6, 7, 4, 7, 6, 6],
            color=color,
            opacity=0.3,
            name=f"Rule {rule_idx}: {ant_names} → {rule.consequent.name}",
            showlegend=True
        ))
    
    inp_x = (input_list[0].top1 + input_list[0].top2) / 2
    inp_y = (input_list[1].top1 + input_list[1].top2) / 2
    inp_z = (input_list[2].top1 + input_list[2].top2) / 2
    
    fig.add_trace(go.Scatter3d(
        x=[inp_x], y=[inp_y], z=[inp_z],
        mode='markers',
        marker=dict(size=10, color='red', symbol='diamond'),
        name=f"Input ({inp_x:.1f}, {inp_y:.1f}, {inp_z:.1f})"
    ))
    
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="Dimension 1",
            yaxis_title="Dimension 2",
            zaxis_title="Dimension 3",
            xaxis=dict(range=rule_base.inputRanges[0]),
            yaxis=dict(range=rule_base.inputRanges[1]),
            zaxis=dict(range=rule_base.inputRanges[2])
        ),
        height=700,
        showlegend=True
    )
    
    return fig


# ============================================================================
# Diagnostic Functions
# ============================================================================

def show_svi_table(rule_base, input_formatted, imp_params=None):
    """
    Display SVI and final A-subsethood measure (S_A) for each rule.
    Works for both 1D and ND cases.
    """
    if imp_params is None:
        imp_params = getattr(rule_base, '_last_imp_params', ['luka', 1])
    
    Antecedentes_list = [rule.antecedents for rule in rule_base.ruleBase]
    ANT_names, _, ANT_positions_alphabetical = cuenta_antecedentes(rule_base)
    num_dims = len(rule_base.inputRanges)
    
    antec_by_dim_alphabetical = []
    cores_by_dim_spatial = []
    alphabetical_to_spatial_maps = []
    spatial_positions_per_rule = []
    
    for j in range(num_dims):
        aux = [ant[j] for ant in Antecedentes_list]
        aux = list(set(aux))
        aux_str = [str(ant) for ant in aux]
        aux_str_order = sorted(range(len(aux_str)), key=lambda k: aux_str[k])
        aux_alphabetical = [aux[i] for i in aux_str_order]
        
        aux_alphabetical_formatted = format_FN_N_Dim(aux_alphabetical, 'rule_antecedent')
        antec_by_dim_alphabetical.append(aux_alphabetical_formatted)
        
        aux_spatial, cores_spatial, alpha_to_spatial, spatial_to_alpha = sort_antecedents_spatially(Antecedentes_list, j)
        cores_by_dim_spatial.append(cores_spatial)
        alphabetical_to_spatial_maps.append(alpha_to_spatial)
        
        spatial_positions_for_dim = []
        for rule_idx, rule_ants in enumerate(Antecedentes_list):
            ant = rule_ants[j]
            try:
                alpha_pos = aux_alphabetical.index(ant)
            except ValueError:
                alpha_pos = next((i for i, a in enumerate(aux_alphabetical) if id(a) == id(ant)), None)
            spatial_pos = alpha_to_spatial[alpha_pos]
            spatial_positions_for_dim.append(spatial_pos)
        spatial_positions_per_rule.append(spatial_positions_for_dim)
    
    # Import for numerical calculations
    from ..core import A_vee_B as A_vee_B_module
    
    imp_type = imp_params[0]
    imp_param = imp_params[1] if len(imp_params) > 1 else 1
    
    SVI_list = []
    for dim_idx in range(num_dims):
        domain = rule_base.inputRanges[dim_idx]
        inp = input_formatted[dim_idx]
        svi_dim = []
        for ant in antec_by_dim_alphabetical[dim_idx]:
            nu_X_vee_A = A_vee_B_module.nu_A_vee_B_auto(inp, ant, domain)
            nu_A = A_vee_B_module.nu_A_vee_B_auto(ant, ant, domain)
            svi = fuzzy_imp(nu_X_vee_A, nu_A, imp_type, imp_param)
            svi_dim.append(svi)
        SVI_list.append(svi_dim)
    
    # Prepare input cores for D_LR
    input_cores_matrix_format = []
    for j in range(num_dims):
        inp = input_formatted[j]
        if isinstance(inp, Gaussian):
            input_cores_matrix_format.append([inp.center, inp.center])
        else:
            input_cores_matrix_format.append([inp[1], inp[2]])
    
    DL, DR, delta_DL, delta_DR, Den_L, Den_R = D_LR(input_cores_matrix_format, cores_by_dim_spatial)
    
    S_A_values = []
    S_A_per_dim_all_rules = []
    
    for rule_idx, pos_alphabetical in enumerate(ANT_positions_alphabetical):
        if num_dims == 1:
            j = 0
            alpha_pos = pos_alphabetical[0]
            spatial_pos = spatial_positions_per_rule[0][rule_idx]
            
            svi = SVI_list[0][alpha_pos]
            dl = DL[0][spatial_pos]
            dr = DR[0][spatial_pos]
            delta_dl = delta_DL[0][spatial_pos]
            delta_dr = delta_DR[0][spatial_pos]
            
            num_spatial_antecedents = len(cores_by_dim_spatial[0])
            is_spatially_first = (spatial_pos == 0)
            is_spatially_last = (spatial_pos == num_spatial_antecedents - 1)
            
            if is_spatially_first:
                sl = svi * delta_dl
            else:
                den_L_value = Den_L[0][spatial_pos]
                sl = svi * max((1 - dl / den_L_value), 0) * delta_dl
            
            if is_spatially_last:
                sr = svi * delta_dr
            else:
                den_R_value = Den_R[0][spatial_pos]
                sr = svi * max((1 - dr / den_R_value), 0) * delta_dr
            
            sA = max(sr, sl)
            S_A_values.append(sA)
            S_A_per_dim_all_rules.append([sA])
        else:
            S_A_per_dim = []
            
            for dim_idx, alpha_pos in enumerate(pos_alphabetical):
                spatial_pos = spatial_positions_per_rule[dim_idx][rule_idx]
                
                svi = SVI_list[dim_idx][alpha_pos]
                dl = DL[dim_idx][spatial_pos]
                dr = DR[dim_idx][spatial_pos]
                delta_dl = delta_DL[dim_idx][spatial_pos]
                delta_dr = delta_DR[dim_idx][spatial_pos]
                
                num_spatial_antecedents = len(cores_by_dim_spatial[dim_idx])
                is_spatially_first = (spatial_pos == 0)
                is_spatially_last = (spatial_pos == num_spatial_antecedents - 1)
                
                if is_spatially_first:
                    sl = svi * delta_dl
                else:
                    den_L_value = Den_L[dim_idx][spatial_pos]
                    sl = svi * max((1 - dl / den_L_value), 0) * delta_dl
                
                if is_spatially_last:
                    sr = svi * delta_dr
                else:
                    den_R_value = Den_R[dim_idx][spatial_pos]
                    sr = svi * max((1 - dr / den_R_value), 0) * delta_dr
                
                sA_dim = max(sr, sl)
                S_A_per_dim.append(sA_dim)
            
            S_A_per_dim_all_rules.append(S_A_per_dim)
            sA_agg = np.mean(S_A_per_dim)
            S_A_values.append(sA_agg)
    
    # Display tables
    print("\nRule Base:")
    print("=" * 90)
    if num_dims == 1:
        print(f"{'Rule':<10} {'Antecedent':<20} {'Consequent':<20}")
        print("-" * 90)
        for rule_idx in range(len(rule_base.ruleBase)):
            ant = Antecedentes_list[rule_idx][0]
            ant_name = ant.name if hasattr(ant, 'name') else str(ant)
            cons = rule_base.ruleBase[rule_idx].consequent
            cons_name = cons.name if hasattr(cons, 'name') else str(cons)
            print(f"{rule_idx:<10} {ant_name:<20} {cons_name:<20}")
    else:
        print(f"{'Rule':<10} {'Antecedents':<50} {'Consequent':<20}")
        print("-" * 90)
        for rule_idx in range(len(rule_base.ruleBase)):
            ant_names = ", ".join([ant.name if hasattr(ant, 'name') else str(ant) for ant in Antecedentes_list[rule_idx]])
            cons = rule_base.ruleBase[rule_idx].consequent
            cons_name = cons.name if hasattr(cons, 'name') else str(cons)
            print(f"{rule_idx:<10} {ant_names:<50} {cons_name:<20}")
    print("=" * 90)
    
    print("\nA-subsethood Measures:")
    if num_dims == 1:
        print("=" * 70)
        print(f"{'Rule':<10} {'SVI':<15} {'A-subsethood (S_A)':<20}")
        print("-" * 70)
        for rule_idx in range(len(rule_base.ruleBase)):
            alpha_pos = ANT_positions_alphabetical[rule_idx][0]
            svi = SVI_list[0][alpha_pos]
            sA = S_A_values[rule_idx]
            print(f"{rule_idx:<10} {svi:<15.6f} {sA:<20.6f}")
        print("=" * 70)
    else:
        dim_headers = [f"S_A(dim{d+1})" for d in range(num_dims)]
        dim_header_str = "  ".join([f"{h:<12}" for h in dim_headers])
        
        table_width = 10 + num_dims * 14 + 15
        print("=" * table_width)
        print(f"{'Rule':<10} {dim_header_str}  {'S_A (agg)':<15}")
        print("-" * table_width)
        
        for rule_idx in range(len(rule_base.ruleBase)):
            sA = S_A_values[rule_idx]
            sA_per_dim = S_A_per_dim_all_rules[rule_idx]
            sA_dim_str = "  ".join([f"{s:<12.6f}" for s in sA_per_dim])
            print(f"{rule_idx:<10} {sA_dim_str}  {sA:<15.6f}")
        
        print("=" * table_width)


def show_detailed_diagnostic(rule_base, input_formatted, imp_params=None, agg_method=None):
    """Show A_FIS diagnostic — delegates to show_svi_table for per-rule details."""
    if imp_params is None:
        imp_params = getattr(rule_base, '_last_imp_params', ['luka', 1])
    if agg_method is None:
        agg_method = getattr(rule_base, '_last_agg_method', 'avg')
    print(f"Implication: {imp_params[0]} (param={imp_params[1] if len(imp_params) > 1 else 1})")
    print(f"Aggregation: {agg_method}")
    show_svi_table(rule_base, input_formatted, imp_params)


def show_detailed_diagnostic_nd(rule_base, input_formatted, imp_params=None, agg_method=None):
    """Alias for show_detailed_diagnostic (backwards compatibility)."""
    return show_detailed_diagnostic(rule_base, input_formatted, imp_params, agg_method)


# ============================================================================
# Experiment Functions
# ============================================================================

def test_swapped_rule_base(rule_base, input_formatted, shuffle_type='reversed', random_seed=None):
    """Test with swapped rule order and compare outputs."""
    rule_base_swapped = FuzzyRuleBase()
    rule_base_swapped.setInputRanges(rule_base.inputRanges)
    rule_base_swapped.setOutputRange(rule_base.outputRange)

    if shuffle_type == 'reversed':
        for i in range(rule_base.size()-1, -1, -1):
            rule_base_swapped.appendRule(rule_base.ruleBase[i])
    elif shuffle_type == 'random':
        if random_seed is not None:
            random.seed(random_seed)
        indices = list(range(rule_base.size()))
        random.shuffle(indices)
        for i in indices:
            rule_base_swapped.appendRule(rule_base.ruleBase[i])
    else:
        raise ValueError(f"shuffle_type must be 'reversed' or 'random', got '{shuffle_type}'")
    
    _, _, y_crisp_swapped, _ = run_afis(input_formatted, rule_base_swapped)
    return y_crisp_swapped


def test_multiple_random_shuffles(rule_base, input_formatted, num_tests=10, random_seed=None):
    """Test order independence with multiple random rule shuffles."""
    _, _, y_crisp_original, _ = run_afis(input_formatted, rule_base)
    
    shuffled_outputs = []
    if random_seed is not None:
        random.seed(random_seed)
    
    for test_num in range(num_tests):
        rule_base_shuffled = FuzzyRuleBase()
        rule_base_shuffled.setInputRanges(rule_base.inputRanges)
        rule_base_shuffled.setOutputRange(rule_base.outputRange)
        
        indices = list(range(rule_base.size()))
        random.shuffle(indices)
        for i in indices:
            rule_base_shuffled.appendRule(rule_base.ruleBase[i])
        
        _, _, y_crisp_shuffled, _ = run_afis(input_formatted, rule_base_shuffled)
        shuffled_outputs.append(y_crisp_shuffled)
    
    diffs = [abs(y_crisp_original - y_shuffled) for y_shuffled in shuffled_outputs]
    max_diff = max(diffs)
    mean_diff = sum(diffs) / len(diffs)
    
    return {
        'original': y_crisp_original,
        'shuffled_outputs': shuffled_outputs,
        'diffs': diffs,
        'max_diff': max_diff,
        'mean_diff': mean_diff,
        'num_tests': num_tests
    }


# ============================================================================
# Activation Curves Functions
# ============================================================================

def compute_activation_curves(rule_base, dim, base_width=0, step=1, imp_params=None):
    """
    Compute activation curves (S_A vs input position) for a single dimension.
    """
    from ..core import A_vee_B as A_vee_B_module
    
    half_base = base_width / 2
    
    if imp_params is None:
        imp_params = getattr(rule_base, '_last_imp_params', ['luka', 1])
    
    imp_type = imp_params[0]
    imp_param = imp_params[1] if len(imp_params) > 1 else 1
    
    print(f"Using implication: {imp_type} with parameter {imp_param}")
    
    universe_min, universe_max = rule_base.inputRanges[dim]
    first_Ci = universe_min + half_base
    last_Ci = universe_max - half_base
    
    C_i_values = np.arange(first_Ci, last_Ci + step, step)
    
    print(f"Dimension {dim}: Sweeping from C_i={first_Ci} to C_i={last_Ci}, step={step}")
    
    Antecedentes_list = [rule.antecedents for rule in rule_base.ruleBase]
    
    aux = [ant[dim] for ant in Antecedentes_list]
    aux_unique = list(set(aux))
    aux_str = [str(ant) for ant in aux_unique]
    aux_str_order = sorted(range(len(aux_str)), key=lambda k: aux_str[k])
    aux_alphabetical = [aux_unique[i] for i in aux_str_order]
    aux_alphabetical_formatted = format_FN_N_Dim(aux_alphabetical, 'rule_antecedent')
    
    _, cores_spatial, alpha_to_spatial, spatial_to_alpha = sort_antecedents_spatially(Antecedentes_list, dim)
    
    antecedent_order = []
    for spatial_pos in range(len(aux_alphabetical)):
        alpha_idx = spatial_to_alpha[spatial_pos]
        ant = aux_alphabetical[alpha_idx]
        antecedent_order.append(ant.name if hasattr(ant, 'name') else f"Ant_{spatial_pos}")
    
    S_A_curves = {name: [] for name in antecedent_order}
    
    for C_i in tqdm(C_i_values, desc=f"Dim {dim} activation curves"):
        ini = C_i - half_base
        end = C_i + half_base
        input_sweep = Trapezoidal(ini=ini, top1=C_i, top2=C_i, end=end)
        
        input_fs = [FuzzySet("Input", input_sweep)]
        input_fmt = format_FN_N_Dim(input_fs, ftype='rule_antecedent')
        
        domain = rule_base.inputRanges[dim]
        
        SVI_list = []
        for ant in aux_alphabetical_formatted:
            nu_X_vee_A = A_vee_B_module.nu_A_vee_B(input_fmt[0], ant, domain)
            nu_A = A_vee_B_module.nu_A_vee_B(ant, ant, domain)
            svi = fuzzy_imp(nu_X_vee_A, nu_A, imp_type, imp_param)
            SVI_list.append(svi)
        
        input_cores = [[input_fmt[0][1], input_fmt[0][2]]]
        DL, DR, delta_DL, delta_DR, Den_L, Den_R = D_LR(input_cores, [cores_spatial])
        
        num_ants = len(cores_spatial)
        
        for alpha_idx, ant in enumerate(aux_alphabetical):
            ant_name = ant.name if hasattr(ant, 'name') else f"Ant_{alpha_idx}"
            spatial_pos = alpha_to_spatial[alpha_idx]
            
            svi = SVI_list[alpha_idx]
            dl = DL[0][spatial_pos]
            dr = DR[0][spatial_pos]
            delta_dl = delta_DL[0][spatial_pos]
            delta_dr = delta_DR[0][spatial_pos]
            
            is_first = (spatial_pos == 0)
            is_last = (spatial_pos == num_ants - 1)
            
            if is_first:
                sl = svi * delta_dl
            else:
                den_L = Den_L[0][spatial_pos]
                sl = svi * max((1 - dl / den_L), 0) * delta_dl
            
            if is_last:
                sr = svi * delta_dr
            else:
                den_R = Den_R[0][spatial_pos]
                sr = svi * max((1 - dr / den_R), 0) * delta_dr
            
            sA = max(sr, sl)
            S_A_curves[ant_name].append(sA)
    
    print(f"  Complete! {len(antecedent_order)} activation curves computed.")
    
    return C_i_values, S_A_curves, antecedent_order


def plot_activation_curves(rule_base, dim, C_i_values, S_A_curves, antecedent_order):
    """Plot antecedents with activation curves overlaid for a single dimension."""
    fig = go.Figure()
    
    universe_min, universe_max = rule_base.inputRanges[dim]
    x_domain = np.linspace(universe_min, universe_max, 200)
    
    antecedents = []
    ant_names_seen = set()
    for rule in rule_base.ruleBase:
        ant = rule.antecedents[dim]
        ant_name = ant.name if hasattr(ant, 'name') else str(ant)
        if ant_name not in ant_names_seen:
            antecedents.append(ant)
            ant_names_seen.add(ant_name)
    
    colors = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A', '#19D3F3', '#FF6692', '#B6E880']
    
    color_map = {}
    for i, ant in enumerate(antecedents):
        ant_name = ant.name if hasattr(ant, 'name') else str(ant)
        color_map[ant_name] = colors[i % len(colors)]
    
    for ant in antecedents:
        ant_name = ant.name if hasattr(ant, 'name') else str(ant)
        y_membership = [ant.set.pertinence(xi) for xi in x_domain]
        fig.add_trace(go.Scatter(
            x=x_domain, 
            y=y_membership, 
            mode='lines', 
            name=f'{ant_name}',
            line=dict(width=2, color=color_map[ant_name]),
            opacity=0.5
        ))
    
    for ant_name in antecedent_order:
        fig.add_trace(go.Scatter(
            x=C_i_values,
            y=S_A_curves[ant_name],
            mode='lines',
            name=f'S_A({ant_name})',
            line=dict(width=3, color=color_map[ant_name]),
            opacity=1.0
        ))
    
    fig.update_layout(
        title=f"Dimension {dim}: Antecedents and Activation Curves (S_A vs C_i)",
        xaxis_title=f"Universe / Input Center Position (C_i)",
        yaxis_title="Membership / Activation (S_A)",
        yaxis_range=[0, 1.1],
        height=500,
        showlegend=True,
        legend=dict(x=1.02, y=1, xanchor='left')
    )
    
    return fig

