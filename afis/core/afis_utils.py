"""
A-FIS Utility Classes and Functions

Contains:
- Membership function classes: Triangular, Trapezoidal, InferiorBorder, SuperiorBorder, Gaussian
- Fuzzy rule structures: FuzzySet, FuzzyRule, FuzzyRuleBase
- Defuzzification: centroid

Original code by Renato Lopes Moura, with minor modifications:
https://github.com/renatolm/wang-mendel
"""

import numpy as np

# ============================================================================
# Membership Function Classes
# ============================================================================

class Triangular:
    def __init__(self, ini, top, end):
        self.ini = ini
        self.top = top
        self.end = end
        
    def __repr__(self):
        return "triang(x, {}, {}, {})".format(self.ini, self.top, self.end)
    
    def __str__(self):
        return "triang(x, {}, {}, {})".format(self.ini, self.top, self.end)
        
    def pertinence(self, x):
        if (x > self.ini) and (x < self.top):
            return (x / (self.top - self.ini)) - (self.ini / (self.top - self.ini))
        elif (x > self.top) and (x < self.end):
            return (-x / (self.end - self.top)) + (self.end / (self.end - self.top))
        elif x == self.top:
            return 1
        else:
            return 0


class Trapezoidal:
    def __init__(self, ini, top1, top2, end):
        self.ini = ini
        self.top1 = top1
        self.top2 = top2
        self.end = end
        
    def __repr__(self):
        return "trap(x, {}, {}, {}, {})".format(self.ini, self.top1, self.top2, self.end)
    
    def __str__(self):
        return "trap(x, {}, {}, {}, {})".format(self.ini, self.top1, self.top2, self.end)
        
    def pertinence(self, x):
        if (x > self.ini) and (x < self.top1):
            return (x / (self.top1 - self.ini)) - (self.ini / (self.top1 - self.ini))
        elif (x > self.top2) and (x < self.end):
            return (-x / (self.end - self.top2)) + (self.end / (self.end - self.top2))
        elif (x >= self.top1) and (x <= self.top2):
            return 1
        else:
            return 0


class InferiorBorder:
    def __init__(self, top, end):
        self.top = top
        self.end = end
        
    def __repr__(self):
        return "inf_border(x, {}, {})".format(self.top, self.end)
    
    def __str__(self):
        return "inf_border(x, {}, {})".format(self.top, self.end)
        
    def pertinence(self, x):
        if x <= self.top:
            return 1
        elif (x > self.top) and (x < self.end):
            return (-x / (self.end - self.top)) + (self.end / (self.end - self.top))
        else:
            return 0


class SuperiorBorder:
    def __init__(self, ini, top):
        self.ini = ini
        self.top = top
        
    def __repr__(self):
        return "sup_border(x, {}, {})".format(self.ini, self.top)
    
    def __str__(self):
        return "sup_border(x, {}, {})".format(self.ini, self.top)
    
    def pertinence(self, x):
        if x <= self.ini:
            return 0
        elif (x > self.ini) and (x < self.top):
            return (x / (self.top - self.ini)) - (self.ini / (self.top - self.ini))
        else:
            return 1


class Gaussian:
    """
    Gaussian membership function: μ(x) = exp(-(x - center)² / (2 * sigma²))
    
    The core (where μ=1) is a single point at 'center'.
    The support extends theoretically to infinity, but for practical purposes
    we consider 4*sigma on each side (covers ~99.99% of the area).
    """
    def __init__(self, center, sigma):
        self.center = center
        self.sigma = sigma
        self.top = center
        
    def __repr__(self):
        return "gaussian(x, center={}, σ={})".format(self.center, self.sigma)
    
    def __str__(self):
        return "gaussian(x, center={}, σ={})".format(self.center, self.sigma)
    
    def pertinence(self, x):
        """Evaluate Gaussian membership at point x."""
        return np.exp(-((x - self.center) ** 2) / (2 * self.sigma ** 2))


# ============================================================================
# Fuzzy Rule Structures
# ============================================================================

class FuzzyRuleBase:

    def __init__(self):
        self.ruleBase = []
        self.inputRanges = []
        self.outputRange = None

    def appendRule(self, rule):
        self.ruleBase.append(rule)

    def size(self):
        return len(self.ruleBase)
    
    def setInputRanges(self, inputRanges):
        self.inputRanges = inputRanges
        
    def setOutputRange(self, outputRange):
        self.outputRange = outputRange

    def printRule(self, index):
        print("antecedents:")
        for antecedent in self.ruleBase[index].antecedents:
            print(antecedent.name)
        print("consequent:", self.ruleBase[index].consequent.name)
        print("strength:", self.ruleBase[index].strength)


class FuzzyRule:

    def __init__(self, antecedents, consequent, strength):		
        self.antecedents = antecedents
        self.consequent = consequent
        self.strength = strength


class FuzzySet:
    
    def __init__(self, name, fset):
        self.name = name
        self.set = fset
        
    def __repr__(self):
        return "{} = {}".format(self.name, str(self.set))
    
    def __str__(self):
        return "{} = {}".format(self.name, str(self.set))


# ============================================================================
# Defuzzification
# ============================================================================

def centroid(x, f_x):
    """Centroid defuzzification: returns crisp value or 0 if total membership is zero."""
    x, f_x = np.asarray(x), np.asarray(f_x)
    if len(x) != len(f_x):
        raise ValueError("Argument arrays must have the same size")
    den = f_x.sum()
    return float(np.dot(x, f_x) / den) if den > 0 else 0

