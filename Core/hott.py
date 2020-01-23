import ot


def hott(p, q, C, threshold=None):
    """ Hierarchical optimal topic transport."""
    k = len(p)
    if threshold is None:
        threshold = 1. / (k + 1)
    id_p = p > threshold
    id_q = q > threshold
    C_reduced = C[id_p][:, id_q].copy(order='C')
    return ot.emd2(p[id_p]/(p[id_p].sum()), q[id_q]/(q[id_q].sum()), C_reduced)


def hoftt(p, q, C):
    """ Hierarchical optimal full topic transport."""
    return ot.emd2(p, q, C)
