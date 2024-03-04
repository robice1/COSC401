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
    x1, y1, x2, y2 = code
    lower_x, upper_x = min(x1, x2), max(x1, x2)
    lower_y, upper_y = min(y1, y2), max(y1, y2)
    def h(x):
        return lower_x <= x[0] <= upper_x and lower_y <= x[1] <= upper_y
    return h

import itertools

h1 = decode((1, 4, 7, 9))
h2 = decode((7, 9, 1, 4))
h3 = decode((1, 9, 7, 4))
h4 = decode((7, 4, 1, 9))


for x in itertools.product(range(-2, 11), repeat=2):
    if len({h(x) for h in [h1, h2, h3, h4]}) != 1:
        print("Inconsistent prediction for", x)
        break
else:
    print("OK")