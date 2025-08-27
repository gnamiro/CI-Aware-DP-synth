import numpy as np
from mbi import Domain, Dataset, CliqueVector
from mbi.junction_tree import JunctionTree
from functools import reduce
import pickle
import networkx as nx
import itertools
import pandas as pd

class GraphicalModel:
    def __init__(self, domain, cliques, total = 1.0, elimination_order=None):
        """ Constructor for a GraphicalModel

        :param domain: a Domain object
        :param total: the normalization constant for the distribution
        :param cliques: a list of cliques (not necessarilly maximal cliques)
            - each clique is a subset of attributes, represented as a tuple or list
        :param elim_order: an elimination order for the JunctionTree algorithm
            - Elimination order will impact the efficiency by not correctness.  
              By default, a greedy elimination order is used
        """
        self.domain = domain
        self.total = total
        tree = JunctionTree(domain, cliques, elimination_order)
        self.junction_tree = tree

        self.cliques = tree.maximal_cliques() # maximal cliques
        self.message_order = tree.mp_order()
        self.sep_axes = tree.separator_axes()
        self.neighbors = tree.neighbors()
        self.elimination_order = tree.elimination_order

        self.size = sum(domain.size(cl) for cl in self.cliques)
        if self.size*8 > 4*10**9:
            import warnings
            message = 'Size of parameter vector is %.2f GB. ' % (self.size*8 / 10**9) 
            message += 'Consider removing some measurements or finding a better elimination order'
            warnings.warn(message)
        # print("222222222222222222222222222222")
        # print("GM:40: Information of Junction Tree")
        # print("cliques")
        # print(self.cliques)
        # print("------------------")
        # print("message Order")
        # print(self.message_order)
        # print("------------------")
        # print("sep axes")
        # print(self.sep_axes)
        # print("------------------")
        # print("neighbors")
        # print(self.neighbors)
        # print("------------------")
        # print("elimination order")
        # print(self.elimination_order)


    @staticmethod
    def save(model, path):
        pickle.dump(model, open(path, 'wb'))

    @staticmethod
    def load(path):
        return pickle.load(open(path, 'rb'))

    def project(self, attrs, potentials=None):
        """ Project the distribution onto a subset of attributes.
            I.e., compute the marginal of the distribution

        :param attrs: a subset of attributes in the domain, represented as a list or tuple
        :return: a Factor object representing the marginal distribution
        """
        # use precalculated marginals if possible
        if type(attrs) is list:
            attrs = tuple(attrs)
        if hasattr(self, 'marginals'):
            for cl in self.cliques:
                if set(attrs) <= set(cl):
                    return self.marginals[cl].project(attrs)

        elim = self.domain.invert(attrs)
        # print(self.cliques)
        # print([attrs])
        elim_order = greedy_order(self.domain, self.cliques + [attrs], elim)
        # print(elim_order)
        pots = list(self.potentials.values()) if potentials == None else potentials.values()
        ans = variable_elimination_logspace(pots, elim_order, self.total)
        return ans.project(attrs)

    def krondot(self, matrices):
        """ Compute the answer to the set of queries Q1 x Q2 X ... x Qd, where 
            Qi is a query matrix on the ith attribute and "x" is the Kronecker product
        This may be more efficient than computing a supporting marginal then multiplying that by Q.
        In particular, if each Qi has only a few rows.
        
        :param matrices: a list of matrices for each attribute in the domain
        :return: the vector of query answers
        """
        assert all(M.shape[1] == n for M, n in zip(matrices, self.domain.shape)), \
            'matrices must conform to the shape of the domain'
        logZ = self.belief_propagation(self.potentials, logZ=True)
        factors = [self.potentials[cl].exp() for cl in self.cliques]
        Factor = type(factors[0]) # infer the type of the factors
        elim = self.domain.attrs
        for attr, Q in zip(elim, matrices):
            d = Domain(['%s-answer'%attr, attr], Q.shape)
            factors.append(Factor(d, Q))
        result = variable_elimination(factors, elim)
        result = result.transpose(['%s-answer'%a for a in elim])
        return result.datavector(flatten=False) * self.total / np.exp(logZ)

    # def calculate_many_marginals(self, projections):
    #     """ Calculates marginals for all the projections in the list using
    #         Algorithm for answering many out-of-clique queries (section 10.3 in Koller and Friedman)
    
    #     This method may be faster than calling project many times
        
    #     :param projections: a list of projections, where 
    #         each projection is a subset of attributes (represented as a list or tuple)
    #     :return: a list of marginals, where each marginal is represented as a Factor
    #     """

    #     self.marginals = self.belief_propagation(self.potentials)
    #     sep = self.sep_axes
    #     neighbors = self.neighbors
    #     # first calculate P(Cj | Ci) for all neighbors Ci, Cj
    #     conditional = {}
    #     for Ci in neighbors:
    #         for Cj in neighbors[Ci]:
    #             Sij = sep[(Cj, Ci)]
    #             Z = self.marginals[Cj]
    #             conditional[(Cj,Ci)] = Z / Z.project(Sij)

    #     # now iterate through pairs of cliques in order of distance
    #     pred, dist = nx.floyd_warshall_predecessor_and_distance(self.junction_tree.tree,weight=False)
    #     results = {}
    #     for Ci,Cj in sorted(itertools.combinations(self.cliques,2),key=lambda X:dist[X[0]][X[1]]):
    #         Cl = pred[Ci][Cj]
    #         Y = conditional[(Cj,Cl)]
    #         if Cl == Ci:
    #             X = self.marginals[Ci]
    #             results[(Ci, Cj)] = results[(Cj, Ci)] = X*Y
    #         else:
    #             X = results[(Ci, Cl)]
    #             S = set(Cl) - set(Ci) - set(Cj)
    #             results[(Ci, Cj)] = results[(Cj, Ci)] = (X*Y).sum(S)
            
    #     results = { self.domain.canonical(key[0]+key[1]) : results[key] for key in results }
        
    #     answers = { }
    #     for proj in projections:
    #         # print(proj, set(proj))
    #         for attr in results:
    #             # print(attr)
    #             if set(proj) <= set(attr):
    #                 answers[tuple(proj)] = results[attr].project(proj)
    #                 break
    #         if tuple(proj) not in answers:
    #             # Just use variable elimination
    #             answers[tuple(proj)] = self.project(proj)

    #     return answers

    def calculate_many_marginals(self, projections, marginals):
        """ Calculates marginals for all the projections in the list using
            Algorithm for answering many out-of-clique queries (section 10.3 in Koller and Friedman)

        This method may be faster than calling project many times
        
        :param projections: a list of projections, where 
            each projection is a subset of attributes (represented as a list or tuple)
        :param marginals: precomputed marginals as input, represented as a dictionary of cliques and their factors
        :return: a list of marginals, where each marginal is represented as a Factor
        """

        # Use the passed marginals instead of self.belief_propagation
        self.marginals = marginals
        sep = self.sep_axes
        neighbors = self.neighbors

        # First calculate P(Cj | Ci) for all neighbors Ci, Cj
        conditional = {}
        for Ci in neighbors:
            for Cj in neighbors[Ci]:
                Sij = sep[(Cj, Ci)]
                Z = self.marginals[Cj]
                conditional[(Cj, Ci)] = Z / Z.project(Sij)

        # Now iterate through pairs of cliques in order of distance
        pred, dist = nx.floyd_warshall_predecessor_and_distance(self.junction_tree.tree, weight=False)
        results = {}
        # print(self.cliques)
        for Ci, Cj in sorted(itertools.combinations(self.cliques, 2), key=lambda X: dist[X[0]][X[1]]):
            Cl = pred[Ci][Cj]
            Y = conditional[(Cj, Cl)]
            if Cl == Ci:
                X = self.marginals[Ci]
                results[(Ci, Cj)] = results[(Cj, Ci)] = X * Y
            else:
                X = results[(Ci, Cl)]
                S = set(Cl) - set(Ci) - set(Cj)
                results[(Ci, Cj)] = results[(Cj, Ci)] = (X * Y).sum(S)

        results = {self.domain.canonical(key[0] + key[1]): results[key] for key in results}

        answers = {}
        # print("grpahicalMODEL205:results here!!!!!")
        # print(results)
        # print(projections)
        for proj in projections:
            proj_tuple = tuple(proj)
            # print(proj, set(proj))
            for attr in results:
                # print(attr)
                if set(proj_tuple) <= set(attr):
                    answers[proj_tuple] = results[attr].project(proj)
                    break
            if proj_tuple not in answers:
                # Just use variable elimination
                answers[proj_tuple] = self.project(proj)

        return answers


    def datavector(self, flatten=True):
        """ Materialize the explicit representation of the distribution as a data vector. """
        logp = sum(self.potentials[cl] for cl in self.cliques)
        ans = np.exp(logp - logp.logsumexp())
        wgt = ans.domain.size() / self.domain.size()
        return ans.expand(self.domain).datavector(flatten) * wgt * self.total

    def belief_propagation(self, potentials, logZ=False):
        """ Compute the marginals of the graphical model with given parameters
        
        Note this is an efficient, numerically stable implementation of belief propagation
    
        :param potentials: the (log-space) parameters of the graphical model
        :param logZ: flag to return logZ instead of marginals
        :return marginals: the marginals of the graphical model
        """
        
        beliefs = { cl : potentials[cl].copy() for cl in potentials }
        messages = {}
        
        for i,j in self.message_order:
            sep = beliefs[i].domain.invert(self.sep_axes[(i,j)])
            if (j,i) in messages:
                tau = beliefs[i] - messages[(j,i)]
            else:
                tau = beliefs[i]
            messages[(i,j)] = tau.logsumexp(sep)
            beliefs[j] += messages[(i,j)]

        cl = self.cliques[0]      
        if logZ: return beliefs[cl].logsumexp()

        logZ = beliefs[cl].logsumexp()
        # print("Printing the value for LOGZ")
        # print(logZ)
        # print("beleifs before applying logZ")
        # print(beliefs[cl])
        
        ## Changes
        for cl in self.cliques:
            beliefs[cl] += np.log(self.total) - logZ
            # beliefs[cl] -= logZ
            beliefs[cl] = beliefs[cl].exp(out=beliefs[cl]) 

        # print("beleifs after applying logZ")
        # print(beliefs[cl])   

        return CliqueVector(beliefs)

    def mle(self, marginals):
        """ Compute the model parameters from the given marginals

        :param marginals: target marginals of the distribution
        :param: the potentials of the graphical model with the given marginals
        """
        potentials = {}
        variables = set()
        # print("203: graphical Model: ")
        # print(self.cliques)
        for cl in self.cliques:
            new = tuple(variables & set(cl))
            #factor = marginals[cl] / marginals[cl].project(new)
            variables.update(cl)
            # print("&&&&&&&&&&&&&&&")
            # print("clique")
            # print(cl)
            # print("-------------")
            # print("variables: ")
            # print(variables)
            potentials[cl] = marginals[cl].log() - marginals[cl].project(new).log()
        return CliqueVector(potentials)

    def fit(self, data):
        from mbi import Factor
        # from mbi.torch_factor import Factor
        assert data.domain.contains(self.domain), 'model domain not compatible with data domain'
        marginals = {}
        for cl in self.cliques:
            x = data.project(cl).datavector()
            dom = self.domain.project(cl)
            marginals[cl] = Factor(dom, x)
        self.potentials = self.mle(marginals)

    def _regularizer_loss_CMI_sklearn(self, marginals, constraint):
        """
        constraint = X, Y, Z
        Simple implementation of KL divergence for calculating the conditional dependency between two attributes.
        """
        assert len(constraint) == 3, "your constraint is not conditional"

        projections = [
            [constraint[-1]],  # Z
            [constraint[0], constraint[-1]],  # XZ
            [constraint[1], constraint[-1]],  # YZ
            constraint  # XYZ
        ]


        projected_marginals = self.calculate_many_marginals(projections, marginals)
        regularizer_loss = 0
        
        # Extract the counts for each relevant subset
        _count_Z = projected_marginals[tuple(constraint[-1])]
        _count_XZ = projected_marginals[tuple([constraint[0], constraint[-1]])]
        _count_YZ = projected_marginals[tuple([constraint[1], constraint[-1]])]
        _count_XYZ = projected_marginals[tuple(constraint)]

        # Convert counts to probabilities as numpy arrays
        prob_Z = np.array(_count_Z.dataprobs(False).T, dtype=np.float32)
        prob_XZ = np.array(_count_XZ.dataprobs(False).T, dtype=np.float32)
        prob_YZ = np.array(_count_YZ.dataprobs(False).T, dtype=np.float32)
        prob_XYZ = np.array(_count_XYZ.dataprobs(False).T, dtype=np.float32)

        # Initialize CMI
        cmi = 0.0

        # Loop over all possible values of X, Y, and Z
        for x in range(prob_XYZ.shape[0]):
            for y in range(prob_XYZ.shape[1]):
                for z in range(prob_XYZ.shape[2]):
                    p_xyz = prob_XYZ[x, y, z]
                    p_xz = prob_XZ[x, z]
                    p_yz = prob_YZ[y, z]
                    p_z = prob_Z[z]

                    # Calculate conditional mutual information contribution
                    if p_xyz > 0:
                        cmi += p_xyz * np.log(p_xyz * p_z / (p_xz * p_yz + 1e-12) + 1e-12)

        regularizer_loss += max(cmi, 0)  # Clamp to non-negative values

        return float(regularizer_loss)

    def synthetic_data_full_dist(self, rows=None, method='round'):
        """ Generate synthetic tabular data from the full joint distribution using belief propagation. """
        
        # Step 1: Compute the full joint distribution across all variables
        # print("GraphicalModel: CMI score after regularization: ")
        # print(self._regularizer_loss_CMI_sklearn(self.marginals, ['X', 'Y', 'Z']))
        
        # Get total rows for synthetic data
        total = int(self.total) if rows is None else rows
        cols = self.domain.attrs
        

        # Get the full joint distribution counts using belief propagation
        joint_probabilities = self.compute_joint_distribution().datavector(flatten=True)
        # print("After transpose")
        # print(joint_probabilities)
        # print("CMI for joint")
        # print(CMI_over_joint_prob(joint_probabilities))
    
        # Create a list of all possible combinations of values for each variable
        variable_values = [range(size) for size in self.domain.shape]
        
        # Generate combinations in XYZ order
        all_combinations = np.array(np.meshgrid(*variable_values, indexing='ij')).reshape(len(cols), -1).T

        # print(all_combinations)
        
        # Ensure probabilities match the shape of all combinations
        if len(joint_probabilities) != len(all_combinations):
            raise ValueError("Mismatch between joint distribution and combinations of values.")
        
        # Step 2: Generate synthetic samples from the joint distribution
        # Sample indices based on joint probabilities
        sample_indices = np.random.choice(len(joint_probabilities), size=total, replace=True, p=joint_probabilities)
        
        # Select corresponding rows from all_combinations
        synthetic_data = all_combinations[sample_indices]
        
        # Convert synthetic data to a DataFrame
        df = pd.DataFrame(synthetic_data, columns=cols)
        # print(df.corr())
        # print(calculate_cmi(df.copy(), 'X', 'Y', 'Z'))

        # Calculate the synthetic data marginals and CMI to compare with the original distribution
        # synthetic_marginals = df.groupby(['X', 'Y', 'Z']).size().unstack(fill_value=0) / total
        # synthetic_CMI = CMI_over_joint_prob(synthetic_marginals.values.flatten())
        # print("Empirical CMI from synthetic data:", synthetic_CMI)
        return Dataset(df, self.domain)

    def synthetic_data(self, rows=None, method='round'):
        """ Generate synthetic tabular data from the distribution.  
            Valid options for method are 'round' and 'sample'."""
        # print("354GraphicalMOdel: CMI score: ")
        # print(self._regularizer_loss_CMI_sklearn(self.potentials, ['X', 'Y', 'Z']))
        total = int(self.total) if rows is None else rows
        cols = self.domain.attrs
        data = np.zeros((total, len(cols)), dtype=int)
        df = pd.DataFrame(data, columns = cols)
        cliques = [set(cl) for cl in self.cliques]

        def synthetic_col(counts, total):
            if method == 'sample':
                probas = counts / counts.sum()
                return np.random.choice(counts.size, total, True, probas)
            counts *= total / counts.sum()
            frac, integ = np.modf(counts)
            integ = integ.astype(int)
            extra = total - integ.sum()
            if extra > 0:
                idx = np.random.choice(counts.size, extra, False, frac / frac.sum())
                integ[idx] += 1
            vals = np.repeat(np.arange(counts.size), integ)
            np.random.shuffle(vals)
            return vals

        order = self.elimination_order[::-1]
        # print(order)
        col = order[0]
        marg = self.project([col]).datavector(flatten=False)
        # print("lets print the first col values")
        # print(marg)
        df.loc[:,col] = synthetic_col(marg, total)
        used = { col }

        for col in order[1:]:
            relevant = [cl for cl in cliques if col in cl]
            relevant = used.intersection(set.union(*relevant))
            proj = tuple(relevant)
            used.add(col)
            marg = self.project(proj + (col,)).datavector(flatten=False)
            # print(f"after visiting the values from {relevant}, we have {proj} as intersection with seen attrs")
            # print("Now let see the marginal values::")
            # print(marg)

            def foo(group):
                idx = group.name
                vals = synthetic_col(marg[idx], group.shape[0])
                group[col] = vals
                return group

            if len(proj) >= 1:
                df = df.groupby(list(proj), group_keys=False).apply(foo)
            else:
                df[col] = synthetic_col(marg, df.shape[0])
            
            

        return Dataset(df, self.domain)
    
    
    def compute_joint_distribution(self):
        """
        Compute the full joint distribution from the clique potentials using variable elimination.
        
        :return: Factor representing the full joint distribution over all variables
        """
        # Use all the variables in the domain as the elimination set
        # elim = self.domain.attrs
        # print("attributes of the domain!")
        # print(self.domain.attrs)
        joint_distribution = self.project(self.domain.attrs)
        # print(joint_distribution)
        # Normalize to ensure the joint distribution sums to 1
        # print("TOtal sum")
        # print(joint_distribution.sum())
        joint_distribution /= self.total
        
        return joint_distribution

def variable_elimination_logspace(potentials, elim, total):
    """ run variable elimination on a list of **logspace** factors """
    k = len(potentials)
    psi = dict(zip(range(k), potentials))
    for z in elim:
        psi2 = [psi.pop(i) for i in list(psi.keys()) if z in psi[i].domain]
        phi = reduce(lambda x,y: x+y, psi2, 0)
        tau = phi.logsumexp([z])
        psi[k] = tau
        k += 1
    ans = reduce(lambda x,y: x+y, psi.values(), 0)
    return (ans - ans.logsumexp() + np.log(total)).exp()

def variable_elimination(factors, elim):
    """ run variable elimination on a list of (non-logspace) factors """
    k = len(factors)
    psi = dict(zip(range(k), factors))
    for z in elim:
        psi2 = [psi.pop(i) for i in list(psi.keys()) if z in psi[i].domain]
        phi = reduce(lambda x,y: x*y, psi2, 1)
        tau = phi.sum([z])
        psi[k] = tau
        k += 1
    return reduce(lambda x,y: x*y, psi.values(), 1)

def greedy_order(domain, cliques, elim):
    order = []
    unmarked = set(elim)
    cliques = set(cliques)
    total_cost = 0
    # print("Let's find out about the greedy order algorithm")
    # print(domain)
    # print(cliques)
    # print(elim)
    # print(unmarked)
    # print(cliques)
    for k in range(len(elim)):
        # print("costs")
        
        cost = { }
        for a in unmarked:
            # all cliques that have a
            neighbors = list(filter(lambda cl: a in cl, cliques))
            # variables in this "super-clique"
            variables = tuple(set.union(set(), *map(set, neighbors)))
            # domain for the resulting factor
            newdom = domain.project(variables)
            # cost of removing a
            cost[a] = newdom.size()

        # find the best variable to eliminate
        # print(cost)
        a = min(cost, key=lambda a: cost[a])
        # print("find the minimum variable")
        # print(a)
        # do some cleanup
        order.append(a)
        unmarked.remove(a)
        neighbors = list(filter(lambda cl: a in cl, cliques))
        variables = tuple(set.union(set(), *map(set, neighbors)) - { a })
        cliques -= set(neighbors)
        cliques.add(variables)
        total_cost += cost[a]
    # print(f"the final order is: {order}")
    return order

def CMI_over_joint_prob(joint_probabilities):
    # Step 1: Reshape the joint probabilities to XYZ order
    P_XYZ = joint_probabilities.reshape(2, 2, 3)

    # Step 2: Calculate marginals by summing over the appropriate dimensions
    # P(Z): Sum over X and Y
    P_Z = np.sum(P_XYZ, axis=(0, 1))

    # P(X, Z): Sum over Y
    P_XZ = np.sum(P_XYZ, axis=1)

    # P(Y, Z): Sum over X
    P_YZ = np.sum(P_XYZ, axis=0)

    # Initialize CMI
    cmi = 0.0

    # Loop over all possible values of X, Y, and Z
    for x in range(P_XYZ.shape[0]):
        for y in range(P_XYZ.shape[1]):
            for z in range(P_XYZ.shape[2]):
                p_xyz = P_XYZ[x, y, z]
                p_xz = P_XZ[x, z]
                p_yz = P_YZ[y, z]
                p_z = P_Z[z]

                # Calculate conditional mutual information contribution
                if p_xyz > 0:
                    cmi += p_xyz * np.log2(p_xyz * p_z / (p_xz * p_yz + 1e-12) + 1e-12)

  # Clamp to non-negative values

    return float(max(cmi, 0))


