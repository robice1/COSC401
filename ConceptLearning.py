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
        if all (h(x) == y for x, in D):
            V.add(h)
    return V

def less_general_or_equal(ha, hb, X):
    """takes two hypotheses ha and hb, and an input space X and 
    returns True if and only if ha is less general or equal to hb."""
    support_a = {x for x in X if ha(x)}
    support_b = {x for x in X if hb(x)}
    return support_a <= support_b

X = list(range(1000))

def h2(x): return x % 2 == 0
def h3(x): return x % 3 == 0
def h6(x): return x % 6 == 0

H = [h2, h3, h6]

for ha in H:
    for hb in H:
        print(ha.__name__, "<=", hb.__name__, "?", less_general_or_equal(ha, hb, X))