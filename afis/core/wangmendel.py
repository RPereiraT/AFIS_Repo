"""
Wang-Mendel Fuzzy Rule Learning Algorithm

This module provides functions for automatic generation of fuzzy IF-THEN rules
from input-output data pairs using the Wang-Mendel method.

Uses afis.core.afis_utils classes.
"""

import numpy as np
import string
import copy
from collections import defaultdict

import plotly.graph_objects as go

# Import from core.utils (self-contained)
from .afis_utils import (
    FuzzySet, FuzzyRule, FuzzyRuleBase,
    Triangular, InferiorBorder, SuperiorBorder
)


# =============================================================================
# Fuzzy Partitioning
# =============================================================================

def get_fuzzy_regions(arr, n_regions, name_prefix='A', margin=0.05):
    """
    Create triangular fuzzy partitions for a variable.
    
    Parameters
    ----------
    arr : array-like
        Data values to determine the range.
    n_regions : int
        Number of fuzzy regions (must be >= 2).
    name_prefix : str
        Prefix for fuzzy set names (e.g., 'A' -> A1, A2, A3, ...).
    margin : float
        Percentage margin to extend beyond min/max (default 5%).
        Set to 0 for exact min/max boundaries.
        
    Returns
    -------
    regions : list of FuzzySet
        List of fuzzy sets covering the variable's range.
    """
    if n_regions < 2:
        raise ValueError("The number of fuzzy regions must be at least 2!")
    
    # Calculate range with margin
    data_min, data_max = min(arr), max(arr)
    ini = data_min - margin * abs(data_min) if margin > 0 else data_min
    end = data_max + margin * abs(data_max) if margin > 0 else data_max
    
    step = (end - ini) / (n_regions - 1)
    
    regions = []
    
    # First region: InferiorBorder (plateau on left)
    fset1 = FuzzySet(f'{name_prefix}1', InferiorBorder(ini, ini + step))
    regions.append(fset1)
    
    # Middle regions: Triangular
    for i in range(2, n_regions):
        top = ini + (i - 1) * step
        fset = FuzzySet(f'{name_prefix}{i}', Triangular(top - step, top, top + step))
        regions.append(fset)
    
    # Last region: SuperiorBorder (plateau on right)
    fsetN = FuzzySet(f'{name_prefix}{n_regions}', SuperiorBorder(end - step, end))
    regions.append(fsetN)
    
    return regions


# =============================================================================
# Rule Learning
# =============================================================================

def learn_fuzzy_rules(X, y,
                      n_regions_inputs=None,
                      n_regions_output=5,
                      name_prefix_inputs=None,
                      name_prefix_output="Out",
                      margin=0.05):
    """
    Generate fuzzy IF-THEN rules from data using the Wang-Mendel method.
    
    For each data point (x, y):
    1. Find the fuzzy region with maximum membership for each input dimension
    2. Find the fuzzy region with maximum membership for the output
    3. Create a rule with strength = product of all membership degrees
    
    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Input data.
    y : array-like, shape (n_samples,)
        Output/target data.
    n_regions_inputs : list of int, optional
        Number of fuzzy regions per input dimension. Default: 5 for all.
    n_regions_output : int
        Number of fuzzy regions for the output. Default: 5.
    name_prefix_inputs : list of str, optional
        Name prefixes for input dimensions. Default: ['A', 'B', 'C', ...].
    name_prefix_output : str
        Name prefix for output. Default: 'Out'.
    margin : float
        Margin for fuzzy region boundaries. Default: 0.05 (5%).
        
    Returns
    -------
    rule_base : FuzzyRuleBase
        The generated fuzzy rule base (one rule per data point).
    """
    X = np.array(X) if not isinstance(X, np.ndarray) else X
    y = np.array(y) if not isinstance(y, np.ndarray) else y
    
    n_samples, n_dims = X.shape
    
    # Default: 5 regions per input dimension
    if n_regions_inputs is None:
        n_regions_inputs = [5] * n_dims
    
    # Default: alphabetical prefixes (A, B, C, ...)
    if name_prefix_inputs is None:
        name_prefix_inputs = [chr(ord('A') + i) for i in range(n_dims)]
    
    # Create fuzzy partitions for each input dimension
    input_regions = []
    for j in range(n_dims):
        regions = get_fuzzy_regions(
            X[:, j], n_regions_inputs[j], 
            name_prefix_inputs[j], margin
        )
        input_regions.append(regions)
    
    # Create fuzzy partitions for the output
    output_regions = get_fuzzy_regions(y, n_regions_output, name_prefix_output, margin)
    
    # Generate rules
    rule_base = FuzzyRuleBase()
    
    for i in range(n_samples):
        antecedents = []
        strength = 1.0
        
        # Find best antecedent for each input dimension
        for j in range(n_dims):
            max_pertinence = 0
            best_antecedent = None
            
            for region in input_regions[j]:
                pertinence = region.set.pertinence(X[i, j])
                if pertinence > max_pertinence:
                    best_antecedent = region
                    max_pertinence = pertinence
            
            antecedents.append(best_antecedent)
            strength *= max_pertinence
        
        # Find best consequent
        max_pertinence = 0
        best_consequent = None
        
        for region in output_regions:
            pertinence = region.set.pertinence(y[i])
            if pertinence > max_pertinence:
                best_consequent = region
                max_pertinence = pertinence
        
        strength *= max_pertinence
        
        # Create and append rule
        rule = FuzzyRule(antecedents, best_consequent, strength)
        rule_base.appendRule(rule)
    
    # Set ranges
    input_ranges = [(X[:, j].min(), X[:, j].max()) for j in range(n_dims)]
    output_range = (y.min(), y.max())
    
    rule_base.setInputRanges(input_ranges)
    rule_base.setOutputRange(output_range)
    
    return rule_base


# =============================================================================
# Rule Base Cleaning & Filtering
# =============================================================================

def clean_rule_base(rule_base):
    """
    Remove duplicate rules (same antecedents), keeping only the strongest.
    
    When multiple rules have identical antecedents but different consequents,
    this keeps only the rule with the highest strength.
    
    Parameters
    ----------
    rule_base : FuzzyRuleBase
        The rule base to clean.
        
    Returns
    -------
    cleaned_rule_base : FuzzyRuleBase
        Rule base with no duplicate antecedents.
    """
    # Group rules by antecedents
    antecedent_groups = defaultdict(list)
    
    for rule in rule_base.ruleBase:
        # Use tuple of antecedent ids as key (handles object identity)
        key = tuple(id(ant) for ant in rule.antecedents)
        antecedent_groups[key].append(rule)
    
    # Keep strongest rule from each group
    strongest_rules = []
    for rules in antecedent_groups.values():
        strongest = max(rules, key=lambda r: r.strength)
        strongest_rules.append(strongest)
    
    # Create new rule base
    cleaned = FuzzyRuleBase()
    cleaned.ruleBase = strongest_rules
    cleaned.setInputRanges(rule_base.inputRanges)
    cleaned.setOutputRange(rule_base.outputRange)
    
    return cleaned


def filter_rules_by_consequents(rule_base, n_by_rule=5):
    """
    Filter rules to ensure balanced coverage across all consequents.
    
    For each unique consequent, keeps only the top n_by_rule rules
    (by strength). This prevents the rule base from being dominated
    by rules for a single output region.
    
    Parameters
    ----------
    rule_base : FuzzyRuleBase
        The rule base to filter.
    n_by_rule : int
        Maximum number of rules to keep per consequent.
        
    Returns
    -------
    filtered_rule_base : FuzzyRuleBase
        Rule base with balanced consequent coverage.
    """
    # Group rules by consequent name
    consequent_groups = defaultdict(list)
    for rule in rule_base.ruleBase:
        consequent_groups[rule.consequent.name].append(rule)
    
    # Select top n_by_rule from each group
    selected_rules = []
    for name, rules in consequent_groups.items():
        # Sort by strength (descending)
        sorted_rules = sorted(rules, key=lambda r: r.strength, reverse=True)
        # Take top n_by_rule (or all if fewer)
        selected_rules.extend(sorted_rules[:n_by_rule])
    
    # Create new rule base
    filtered = FuzzyRuleBase()
    filtered.ruleBase = selected_rules
    filtered.setInputRanges(rule_base.inputRanges)
    filtered.setOutputRange(rule_base.outputRange)
    
    return filtered


def filter_rules_by_strength(rule_base, n_rules):
    """
    Keep only the top n rules by strength.
    
    Parameters
    ----------
    rule_base : FuzzyRuleBase
        The rule base to filter.
    n_rules : int
        Number of rules to keep.
        
    Returns
    -------
    filtered_rule_base : FuzzyRuleBase
        Rule base with top n_rules.
    min_strength : float
        The minimum strength threshold used.
    """
    sorted_rules = sorted(rule_base.ruleBase, key=lambda r: r.strength, reverse=True)
    
    n_rules = min(n_rules, len(sorted_rules))
    min_strength = sorted_rules[n_rules - 1].strength if n_rules > 0 else 0
    
    filtered = FuzzyRuleBase()
    filtered.ruleBase = sorted_rules[:n_rules]
    filtered.setInputRanges(rule_base.inputRanges)
    filtered.setOutputRange(rule_base.outputRange)
    
    return filtered, min_strength


# =============================================================================
# High-Level Rule Base Generation
# =============================================================================

def _count_by_consequent(rule_base):
    """Count rules per consequent, return dict sorted by consequent name."""
    from collections import Counter
    counts = Counter(rule.consequent.name for rule in rule_base.ruleBase)
    return dict(sorted(counts.items()))


def filter_rules_top_n(rule_base, n_rules):
    """
    Filter rules keeping only the top N by strength (traditional Wang-Mendel).
    
    Parameters
    ----------
    rule_base : FuzzyRuleBase
        The rule base to filter.
    n_rules : int
        Maximum number of rules to keep.
        
    Returns
    -------
    filtered_rule_base : FuzzyRuleBase
        Rule base with at most n_rules, sorted by strength (descending).
    """
    # Sort rules by strength (descending) and keep top n_rules
    sorted_rules = sorted(rule_base.ruleBase, key=lambda r: r.strength, reverse=True)
    filtered_rules = sorted_rules[:n_rules]
    
    # Create new rule base
    from .afis_utils import FuzzyRuleBase
    new_rb = FuzzyRuleBase()
    new_rb.ruleBase = filtered_rules
    new_rb.inputRanges = rule_base.inputRanges
    new_rb.outputRange = rule_base.outputRange
    return new_rb


def generate_rule_base(X_train, y_train, n_fuzzy_part_dim, n_rules, 
                       filter_method='balanced', verbose=False, margin=0.05):
    """
    Generate a complete fuzzy rule base using the Wang-Mendel algorithm.
    
    This is a high-level function that combines:
    1. Rule learning from data
    2. Duplicate removal (keep strongest)
    3. Filtering to target number of rules
    
    Parameters
    ----------
    X_train : np.ndarray
        Training input data, shape (n_samples, n_features).
    y_train : np.ndarray
        Training target values, shape (n_samples,).
    n_fuzzy_part_dim : int
        Number of fuzzy partitions per dimension.
    n_rules : int
        Target maximum number of rules.
    filter_method : str, optional
        'balanced' - Keep N rules per consequent for balanced coverage (default)
        'top_n' - Keep top N rules by strength (traditional Wang-Mendel)
    verbose : bool, optional
        If True, print rule statistics after each step. Default False.
    margin : float, optional
        Percentage margin to extend fuzzy regions beyond data min/max.
        Default 0.05 (5%). Set to 0 for exact data boundaries.
        
    Returns
    -------
    rule_base : FuzzyRuleBase
        The generated and filtered fuzzy rule base.
    """
    n_dim = X_train.shape[1]
    n_by_rule = max(1, n_rules // n_dim)
    
    # Generate dimension names (A, B, C, ... for inputs, next letter for output)
    input_names = list(string.ascii_uppercase)[:n_dim]
    output_name = string.ascii_uppercase[n_dim]
    
    # Step 1: Generate rules from data
    rule_base = learn_fuzzy_rules(
        X_train, y_train,
        n_regions_inputs=[n_fuzzy_part_dim] * n_dim,
        n_regions_output=n_fuzzy_part_dim,
        name_prefix_inputs=input_names,
        name_prefix_output=output_name,
        margin=margin
    )
    
    # Step 2: Remove duplicate antecedents, keep strongest
    rule_base_cleaned = clean_rule_base(rule_base)
    
    if verbose:
        counts = _count_by_consequent(rule_base_cleaned)
        print(f"  After cleaning (unique antecedents): {len(rule_base_cleaned.ruleBase)} rules")
        print(f"    By consequent: {counts}")
    
    # Step 3: Filter rules
    if filter_method == 'top_n':
        # Traditional Wang-Mendel: keep top N by strength
        rule_base_filtered = filter_rules_top_n(rule_base_cleaned, n_rules)
        filter_desc = f"top {n_rules} by strength"
    else:
        # Balanced: keep N per consequent
        rule_base_filtered = filter_rules_by_consequents(
            rule_base_cleaned, n_by_rule=n_by_rule
        )
        # Iteratively reduce if still over target
        while len(rule_base_filtered.ruleBase) > n_rules and n_by_rule > 1:
            n_by_rule -= 1
            rule_base_filtered = filter_rules_by_consequents(
                rule_base_cleaned, n_by_rule=n_by_rule
            )
        filter_desc = f"max {n_by_rule}/consequent"
    
    if verbose:
        counts = _count_by_consequent(rule_base_filtered)
        print(f"  After filtering ({filter_desc}): {len(rule_base_filtered.ruleBase)} rules")
        print(f"    By consequent: {counts}")
    
    return rule_base_filtered


# =============================================================================
# Visualization
# =============================================================================

def plot_consequents_distribution(rule_base):
    """
    Plot histogram of consequent occurrences in the rule base.
    
    Parameters
    ----------
    rule_base : FuzzyRuleBase
        The rule base to visualize.
        
    Returns
    -------
    fig : plotly.graph_objects.Figure
        The histogram figure.
    """
    consequent_names = sorted([rule.consequent.name for rule in rule_base.ruleBase])
    
    # Count occurrences
    counts = defaultdict(int)
    for name in consequent_names:
        counts[name] += 1
    
    names = sorted(counts.keys())
    values = [counts[n] for n in names]
    
    fig = go.Figure(go.Bar(x=names, y=values))
    fig.update_layout(
        title='Consequent Distribution in Rule Base',
        xaxis_title='Consequent',
        yaxis_title='Count',
        height=400
    )
    
    return fig


def print_rule_base_summary(rule_base):
    """
    Print a summary of the rule base.
    
    Parameters
    ----------
    rule_base : FuzzyRuleBase
        The rule base to summarize.
    """
    print(f"Rule Base Summary")
    print(f"=" * 50)
    print(f"Number of rules: {len(rule_base.ruleBase)}")
    print(f"Input dimensions: {len(rule_base.inputRanges)}")
    print(f"Input ranges: {rule_base.inputRanges}")
    print(f"Output range: {rule_base.outputRange}")
    
    # Count unique consequents
    consequents = set(rule.consequent.name for rule in rule_base.ruleBase)
    print(f"Unique consequents: {len(consequents)}")
    
    # Strength statistics
    strengths = [rule.strength for rule in rule_base.ruleBase]
    print(f"Strength: min={min(strengths):.4f}, max={max(strengths):.4f}, "
          f"mean={np.mean(strengths):.4f}")

