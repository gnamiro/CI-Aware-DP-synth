import numpy as np
from scipy.special import logsumexp

class Factor:
    def __init__(self, domain, values):
        """ Initialize a factor over the given domain

        :param domain: the domain of the factor
        :param values: the ndarray of factor values (for each element of the domain)

        Note: values may be a flattened 1d array or a ndarray with same shape as domain
        """
        assert domain.size() == values.size, 'domain size does not match values size'
        assert values.ndim == 1 or values.shape == domain.shape, 'invalid shape for values array'
        self.domain = domain
        self.values = values.reshape(domain.shape)

    @staticmethod
    def zeros(domain):
        return Factor(domain, np.zeros(domain.shape))
    
    @staticmethod
    def ones(domain):
        return Factor(domain, np.ones(domain.shape))

    @staticmethod
    def random(domain):
        return Factor(domain, np.random.rand(*domain.shape))

    @staticmethod
    def uniform(domain):
        return Factor.ones(domain) / domain.size()

    @staticmethod
    def active(domain, structural_zeros):
        """ create a factor that is 0 everywhere except in positions present in 
            'structural_zeros', where it is -infinity

        :param: domain: the domain of this factor
        :param: structural_zeros: a list of values that are not possible
        """
        idx = tuple(np.array(structural_zeros).T)
        vals = np.zeros(domain.shape)
        vals[idx] = -np.inf
        return Factor(domain, vals)

    def expand(self, domain):
        assert domain.contains(self.domain), 'expanded domain must contain current domain'
        dims = len(domain) - len(self.domain)
        values = self.values.reshape(self.domain.shape + tuple([1]*dims))
        ax = domain.axes(self.domain.attrs)
        values = np.moveaxis(values, range(len(ax)), ax)
        values = np.broadcast_to(values, domain.shape)
        return Factor(domain, values)

    def transpose(self, attrs):
        assert set(attrs) == set(self.domain.attrs), 'attrs must be same as domain attributes'
        newdom = self.domain.project(attrs)
        ax = newdom.axes(self.domain.attrs)
        values = np.moveaxis(self.values, range(len(ax)), ax)
        return Factor(newdom, values)

    def project(self, attrs, agg = 'sum'):
        """ 
        project the factor onto a list of attributes (in order)
        using either sum or logsumexp to aggregate along other attributes
        """
        assert agg in ['sum','logsumexp'], 'agg must be sum or logsumexp'
        marginalized = self.domain.marginalize(attrs)
        if agg == 'sum':
            ans = self.sum(marginalized.attrs)
        elif agg == 'logsumexp':
            ans = self.logsumexp(marginalized.attrs)
        return ans.transpose(attrs)

    def sum(self, attrs = None):
        if attrs is None:
            return np.sum(self.values)
        axes = self.domain.axes(attrs)
        values = np.sum(self.values, axis=axes) 
        newdom = self.domain.marginalize(attrs)
        return Factor(newdom, values)

    def logsumexp(self, attrs = None):
        if attrs is None:
            return logsumexp(self.values)
        axes = self.domain.axes(attrs)
        values = logsumexp(self.values, axis=axes) 
        newdom = self.domain.marginalize(attrs)
        return Factor(newdom, values)

    def logaddexp(self, other):
        newdom = self.domain.merge(other.domain)
        factor1 = self.expand(newdom)
        factor2 = self.expand(newdom)
        return Factor(newdom, np.logaddexp(factor1.values, factor2.values))

    def max(self, attrs = None):
        if attrs is None:
            return self.values.max()
        axes = self.domain.axes(attrs)
        values = np.max(self.values, axis=axes)
        newdom = self.domain.marginalize(attrs)
        return Factor(newdom, values)

    def condition(self, evidence):
        """ evidence is a dictionary where 
                keys are attributes, and 
                values are elements of the domain for that attribute """
        slices = [evidence[a] if a in evidence else slice(None) for a in self.domain]
        newdom = self.domain.marginalize(evidence.keys())
        values = self.values[tuple(slices)]
        return Factor(newdom, values)

    def copy(self, out=None):
        if out is None:
            return Factor(self.domain, self.values.copy())
        np.copyto(out.values, self.values)
        return out

    def elem_mul(self, other):
        if np.isscalar(other):
            new_values = np.nan_to_num(other * self.values)
            return Factor(self.domain, new_values)
        
        # Identify the intersection of attributes to align both factors for element-wise multiplication
        common_attrs = set(self.domain.attrs) & set(other.domain.attrs)
        
        # Project both factors onto the common attributes before multiplying
        factor1_proj = self.project(list(common_attrs))
        factor2_proj = other.project(list(common_attrs))
        
        # Ensure both factors have the same shape by aligning their values
        if factor1_proj.domain != factor2_proj.domain:
            raise ValueError("Domains of factors do not match for multiplication.")
        
        # Perform element-wise multiplication
        new_values = factor1_proj.values * factor2_proj.values
        
        # Return a new factor with the common domain and the multiplied values
        return Factor(factor1_proj.domain, new_values)


    def __mul__(self, other):
        if np.isscalar(other):
            new_values = np.nan_to_num(other*self.values)
            return Factor(self.domain, new_values)
        #print(self.values.max(), other.values.max(), self.domain, other.domain)
        # print("domains")
        # print(self.domain)
        # print(other.domain)
        newdom = self.domain.merge(other.domain)
        # print(newdom)
        factor1 = self.expand(newdom)
        factor2 = other.expand(newdom)
        return Factor(newdom, factor1.values * factor2.values)       

    def __add__(self, other):
        # import torch
        # if isinstance(other, (int, float, torch.Tensor)) and other.ndim == 0:
        #     return Factor(self.domain, other + self.values)
        if np.isscalar(other):
            return Factor(self.domain, other + self.values)
        newdom = self.domain.merge(other.domain)
        factor1 = self.expand(newdom)
        factor2 = other.expand(newdom)
        # print("--"*10)
        # print(type(factor1.values), type(factor2.values))
        # print(factor1.values.shape, factor2.values.shape)
        return Factor(newdom, factor1.values + factor2.values)

    def __iadd__(self, other):
        if np.isscalar(other):
            self.values += other
            return self
        factor2 = other.expand(self.domain)
        self.values += factor2.values
        return self 

    def __imul__(self, other):
        if np.isscalar(other):
            self.values *= other
            return self
        factor2 = other.expand(self.domain)
        self.values *= factor2.values
        return self 

    def __radd__(self, other):
        return self.__add__(other)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __sub__(self, other):
        if np.isscalar(other):
            return Factor(self.domain, self.values - other)
        other = Factor(other.domain, np.where(other.values==-np.inf, 0, -other.values))
        return self + other

    def __truediv__(self, other):
        #assert np.isscalar(other), 'divisor must be a scalar'
        if np.isscalar(other):
            new_values = self.values / other
            new_values = np.nan_to_num(new_values)
            return Factor(self.domain, new_values)
        tmp = other.expand(self.domain)
        vals = np.divide(self.values, tmp.values, where=tmp.values>0)
        vals[tmp.values<=0] = 0.0
        return Factor(self.domain, vals) 

    def exp(self, out=None):
        if out is None:
            return Factor(self.domain, np.exp(self.values))
        np.exp(self.values, out=out.values)
        return out

    def log(self, out=None):
        if out is None:
            return Factor(self.domain, np.log(self.values + 1e-100))
        np.log(self.values, out=out.values)
        return out

    def datavector(self, flatten=True):
        """ Materialize the data vector """
        if flatten:
            return self.values.flatten()
        return self.values

    def dataprobs(self, flatten=True):
        total_count = self.values.sum()
        if flatten:
            return self.values.flatten()/total_count
        
        return self.values/total_count

    # def __repr__(self):
    #     pass
    # def __str__(self):
    #     pass

    def __repr__(self):
        """
        Custom repr method to display the class name and dictionary content.
        """
        return f"{self.__class__.__name__}({super().__repr__()})"

    def __str__(self):
        """
        Custom string representation for printing.
        """
        result = "Factor with the following domain and values:\n"
        result += f"domain: {self.domain}, value: {self.values}\n"
        return result