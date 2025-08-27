import numpy as np
from mbi import FactoredInference_NO_NOISE, Dataset, Domain
from scipy import sparse
from disjoint_set import DisjointSet
import networkx as nx
import itertools
from cdp2adp import cdp_rho
from scipy.special import logsumexp
import argparse

"""
This is a generalization of the winning mechanism from the 
2018 NIST Differential Privacy Synthetic Data Competition.

Unlike the original implementation, this one can work for any discrete dataset,
and does not rely on public provisional data for measurement selection.  
"""

def MST(data, outcome, admissible):
    # rho = cdp_rho(epsilon, delta)
    # sigma = np.sqrt(3/(2*rho))
    print(f"the target: {outcome}, the admssibles: {admissible}")
    cliques = [(col,) for col in data.domain]
    log1 = measure(data, cliques)
    data, log1, undo_compress_fn = compress_domain(data, log1)
    cliques = select(data, log1, outcome, admissible)
    # print("2 way marginals selected for greedy: ")
    # print(cliques)
    log2 = measure(data, cliques)
    engine = FactoredInference_NO_NOISE(data.domain, iters=5000)
    est = engine.estimate(log1+log2)
    synth = est.synthetic_data()
    return undo_compress_fn(synth)

def measure(data, cliques, weights=None):
    if weights is None:
        weights = np.ones(len(cliques))
    weights = np.array(weights) / np.linalg.norm(weights)
    measurements = []
    for proj, wgt in zip(cliques, weights):
        x = data.project(proj).datavector()
        y = x
        # y = x + np.random.normal(loc=0, scale=sigma/wgt, size=x.size)
        Q = sparse.eye(x.size)
        measurements.append( (Q, y, proj) )
    return measurements

def compress_domain(data, measurements):
    supports = {}
    new_measurements = []
    for Q, y, proj in measurements:
        col = proj[0]
        # sup = y >= 3*sigma
        sup = np.ones_like(y, dtype=bool)
        supports[col] = sup
        # if supports[col].sum() == y.size:
        new_measurements.append( (Q, y, proj) )
        # else: # need to re-express measurement over the new domain
        #     y2 = np.append(y[sup], y[~sup].sum())
        #     I2 = np.ones(y2.size)
        #     I2[-1] = 1.0 / np.sqrt(y.size - y2.size + 1.0)
        #     y2[-1] /= np.sqrt(y.size - y2.size + 1.0)
        #     I2 = sparse.diags(I2)
        #     new_measurements.append( (I2, y2, sigma, proj) )
    undo_compress_fn = lambda data: reverse_data(data, supports)
    return transform_data(data, supports), new_measurements, undo_compress_fn

def exponential_mechanism(q, eps, sensitivity, prng=np.random, monotonic=False):
    coef = 1.0 if monotonic else 0.5
    scores = coef*eps/sensitivity*q
    probas = np.exp(scores - logsumexp(scores))
    return prng.choice(q.size, p=probas)

def select(data, measurement_log,outcome,admissible, cliques=[] ):
    engine = FactoredInference_NO_NOISE(data.domain, iters=1000)
    est = engine.estimate(measurement_log)

    weights = {}
    candidates = list(itertools.combinations(data.domain.attrs, 2))
    print(candidates)
    print("*"*10)

    to_remove = []
    for a, b in candidates:
        print(a, b)
        if (a in outcome and b not in admissible) or (b in outcome and a not in admissible):
            to_remove.append((a, b))

    # Remove the collected pairs after iteration
    for pair in to_remove:
        candidates.remove(pair)

    print("*"*10)
    print(candidates)
    for a, b in candidates:
        xhat = est.project([a, b]).datavector()
        x = data.project([a, b]).datavector()
        weights[a,b] = np.linalg.norm(x - xhat, 1)

    T = nx.Graph()
    T.add_nodes_from(data.domain.attrs)
    ds = DisjointSet()

    for e in cliques:
        T.add_edge(*e)
        ds.union(*e)

    r = len(list(nx.connected_components(T)))
    # epsilon = np.sqrt(8*rho/(r-1))
    for i in range(r-1):
        candidates = [e for e in candidates if not ds.connected(*e)] 
        wgts = np.array([weights[e] for e in candidates])
        # print(wgts)
        idx = np.argmax(wgts) # Mirror view :))))
        # print(idx)
        # idx = wgts[0]
        e = candidates[idx]
        T.add_edge(*e)
        ds.union(*e)

    #print(list(T.edges))

    return list(T.edges)

def transform_data(data, supports):
    df = data.df.copy()
    newdom = {}
    for col in data.domain:
        support = supports[col]
        size = support.sum()
        newdom[col] = int(size)
        if size < support.size:
            newdom[col] += 1
        mapping = {}
        idx = 0
        for i in range(support.size):
            mapping[i] = size
            if support[i]:
                mapping[i] = idx
                idx += 1
        assert idx == size
        df[col] = df[col].map(mapping)
    newdom = Domain.fromdict(newdom)
    return Dataset(df, newdom)

def reverse_data(data, supports):
    df = data.df.copy()
    newdom = {}
    for col in data.domain:
        support = supports[col]
        mx = support.sum()
        newdom[col] = int(support.size)
        idx, extra = np.where(support)[0], np.where(~support)[0]
        mask = df[col] == mx
        if extra.size == 0:
            pass
        else:
            df.loc[mask, col] = np.random.choice(extra, mask.sum())
        df.loc[~mask, col] = idx[df.loc[~mask, col]]
    newdom = Domain.fromdict(newdom)
    return Dataset(df, newdom)

def default_params():
    """
    Return default parameters to run this program

    :returns: a dictionary of default parameter settings for each command line argument
    """
    params = {}
    params['dataset'] = '../data/adult.csv'
    params['domain'] = '../data/adult-domain.json'
    params['epsilon'] = 1.0
    params['delta'] = 1e-9
    params['degree'] = 2
    params['num_marginals'] = None
    params['max_cells'] = 10000

    return params


if __name__ == '__main__':

    description = ''
    formatter = argparse.ArgumentDefaultsHelpFormatter
    parser = argparse.ArgumentParser(description=description, formatter_class=formatter)
    parser.add_argument('--dataset', help='dataset to use')
    parser.add_argument('--domain', help='domain to use')
    parser.add_argument('--epsilon', type=float, help='privacy parameter')
    parser.add_argument('--delta', type=float, help='privacy parameter')

    parser.add_argument('--degree', type=int, help='degree of marginals in workload')
    parser.add_argument('--num_marginals', type=int, help='number of marginals in workload')
    parser.add_argument('--max_cells', type=int, help='maximum number of cells for marginals in workload')

    parser.add_argument('--save', type=str, help='path to save synthetic data')

    parser.set_defaults(**default_params())
    args = parser.parse_args()

    data = Dataset.load(args.dataset, args.domain)

    workload = list(itertools.combinations(data.domain, args.degree))
    workload = [cl for cl in workload if data.domain.size(cl) <= args.max_cells]
    if args.num_marginals is not None:
        workload = [workload[i] for i in prng.choice(len(workload), args.num_marginals, replace=False)]

    synth = MST(data, args.epsilon, args.delta)
  
    if args.save is not None:
        synth.df.to_csv(args.save, index=False)
 
    errors = []
    for proj in workload:
        X = data.project(proj).datavector()
        Y = synth.project(proj).datavector()
        e = 0.5*np.linalg.norm(X/X.sum() - Y/Y.sum(), 1)
        errors.append(e)
    print('Average Error: ', np.mean(errors)) 
