import itertools

def input_space(domains):
    """akes a list of domains of attributes and returns the 
    collection of all the objects in the input space."""
    return itertools.product(*domains)

def all_possible_functions(X):
    """takes the entire input space (for some problem) and
      returns a set of functions that is F."""
    X = tuple(X)
    result = set()
    mutations = itertools.product((False, True), repeat=len(X))
    for p in mutations:
        def f(x,p=p):
            return {X[i]:p[i] for i in range(len(X))}[x]
        result.add(f)
    return result

def version_space(H, D):
    """Takes a set of hypotheses H, and a training data set D, and
    returns the version space"""
    V = set()
    for h in H:
        if all (h(x) == y for x, y in D):
            V.add(h)
    return V

def less_general_or_equal(ha, hb, X):
    """takes two hypotheses ha and hb, and an input space X and 
    returns True if and only if ha is less general or equal to hb."""
    support_a = {x for x in X if ha(x)}
    support_b = {x for x in X if hb(x)}
    return support_a <= support_b


def decode(code):
    """Takes a 4-tuple of integers and returns the corresponding hypothesis.
    The first and last two integers of the tuple correspond to opposing corners of a rectangle."""
    values = list(code)
    x1, y1, x2, y2 = code
    lower_x, upper_x = min(x1, x2), max(x1, x2)
    lower_y, upper_y = min(y1, y2), max(y1, y2)
    def h(x):
        return lower_x <= x[0] <= upper_x and lower_y <= x[1] <= upper_y
    return h

def match(code, x):
    """Takes a code and returns True if the corresponding hypothesis returns
    True (positive) for the given input."""
    return decode(code)(x)
    
def lge(code_a, code_b):
    """Takes two codes and returns True if code_a is less general or equal
    to code_b."""
    x1_a, y1_a, x2_a, y2_a = code_a
    x1_b, y1_b, x2_b, y2_b = code_b
    return (x1_a <= x1_b and y1_a <= y1_b and x2_a >= x2_b and y2_a >= y2_b)

def initial_S(domains):
    """Takes a list of domains and returns a set where each element is a
    code for the initial members of S."""
    return [tuple([min(domain) for domain in domains] + [max(domain) for domain in domains])]

    
def initial_G(domains):
    """Takes a list of domains and returns a set where each element is a
    code for the initial members of G."""
    return [tuple(['?' for _ in domains] * 4)]

def minimal_generalisations(code, x):
    """Takes a code (corresponding to a hypothesis) and returns the set of all
    codes that are the minimal generalisations of the given code with respect
    to the given input x."""
    x1, y1, x2, y2 = code
    x_x, y_x = x
    return [(min(x1, x_x), min(y1, y_x), max(x2, x_x), max(y2, y_x))]


def minimal_specialisations(cc, domains, x):
    """Takes a code (corresponding to a hypothesis) and returns the set of all
    codes that are the minimal specialisations of the given code with respect
    to the given input x."""
    x1, y1, x2, y2 = cc
    x_x, y_x = x
    min_x, max_x = domains[0]
    min_y, max_y = domains[1]
    return [(x1 if x1 != x_x else min_x, y1, x2, y2), (x1, y1 if y1 != y_x else min_y, x2, y2),
            (x1, y1, x2 if x2 != x_x else max_x, y2), (x1, y1, x2, y2 if y2 != y_x else max_y)]


def cea_trace(domains, D):
    S_trace, G_trace = [], []
    S = initial_S(domains)
    G = initial_G(domains)
    S_trace.append(S.copy())
    G_trace.append(G.copy())

    for x, y in D:
        if y:  # if positive
            G = [g for g in G if match(g, x)]
            S = [S for S in S if match(S, x)] + [s for s in minimal_generalisations(S, x) if any(lge(g, S) for g in G)]
        else:  # if negative
            S = [S for S in S if not match(S, x)]
            G = [g for g in G if not match(g, x)] + [g for g in minimal_specialisations(G, domains, x) if any(lge(g, S) for S in S)]

        S_trace.append(S.copy())
        G_trace.append(G.copy())

    return S_trace, G_trace


import math

def probability_lower_bound(test_outcomes, deviation):
    n = len(test_outcomes)
    error_rate = sum(1 for outcome in test_outcomes if not outcome) / n
    exponent = -2 * n * deviation ** 2
    return math.exp(exponent)

print(probability_lower_bound([True, False] * 500, 0.05)) # 0.986524106001829
