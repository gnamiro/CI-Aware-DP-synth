import numpy as np

class CliqueVector(dict):
    """ This is a convenience class for simplifying arithmetic over the 
        concatenated vector of marginals and potentials.

        These vectors are represented as a dictionary mapping cliques (subsets of attributes)
        to marginals/potentials (Factor objects)
    """
    def __init__(self, dictionary):
        self.dictionary = dictionary
        dict.__init__(self, dictionary)

    def __str__(self):
        """
        Custom string representation for printing.
        """
        result = "CliqueVector with the following cliques and factors:\n"
        for cl, factor in self.items():
            result += f"Clique: {cl}, Factor: {factor}\n"
        return result
        
    @staticmethod
    def zeros(domain, cliques):
        from mbi import Factor
        # from mbi.torch_factor import Factor
        return CliqueVector({ cl : Factor.zeros(domain.project(cl)) for cl in cliques })

    @staticmethod
    def ones(domain, cliques):
        from mbi import Factor
        # from mbi.torch_factor import Factor
        return CliqueVector({ cl : Factor.ones(domain.project(cl)) for cl in cliques })

    @staticmethod
    def uniform(domain, cliques):
        from mbi import Factor
        # from mbi.torch_factor import Factor
        return CliqueVector({ cl : Factor.uniform(domain.project(cl)) for cl in cliques })

    @staticmethod
    def random(domain, cliques, prng=np.random):
        from mbi import Factor
        # from mbi.torch_factor import Factor
        return CliqueVector({ cl : Factor.random(domain.project(cl), prng) for cl in cliques })

    @staticmethod
    def normal(domain, cliques, prng=np.random):
        from mbi import Factor
        # from mbi.torch_factor import Factor
        return CliqueVector({ cl : Factor.normal(domain.project(cl), prng) for cl in cliques })

    @staticmethod
    def from_data(data, cliques):
        from mbi import Factor
        # from mbi.torch_factor import Factor
        ans = {}
        for cl in cliques:
            mu = data.project(cl)
            ans[cl] = Factor(mu.domain, mu.datavector())
        return CliqueVector(ans)

    def combine(self, other):
        # combines this CliqueVector with other, even if they do not share the same set of factors
        # used for warm-starting optimization
        # Important note: if other contains factors not defined within this CliqueVector, they 
        # are ignored and *not* combined into this CliqueVector
        for cl in other:
            for cl2 in self:
                if set(cl) <= set(cl2):
                    self[cl2] += other[cl]
                    break

    def __truediv__(self, other):
        if np.isscalar(other):
            # Element-wise division by a scalar
            result = {cl: self[cl] / other for cl in self}
        elif isinstance(other, CliqueVector):
            # Element-wise division by another CliqueVector
            if set(self.keys()) != set(other.keys()):
                raise ValueError("Both CliqueVectors must have the same cliques for element-wise division.")
            result = {cl: self[cl] / other[cl] for cl in self}
        else:
            raise TypeError("Unsupported operand type(s) for /: 'CliqueVector' and '{}'".format(type(other).__name__))
        return CliqueVector(result)

    def __mul__(self, const):
        ans = { cl : const*self[cl] for cl in self }
        return CliqueVector(ans)
    
    def __rmul__(self, const):
        return self.__mul__(const)
    
    def __add__(self, other):
        if np.isscalar(other):
            ans = { cl : self[cl] + other for cl in self }
        else:
            ans = { cl : self[cl] + other[cl] for cl in self }
        return CliqueVector(ans)
    
    def __sub__(self, other):
        return self + -1*other

    # Alternatively, you can define this as __eq__ to use `==` directly
    def __eq__(self, other):
        return self.is_equal(other)

    def __ne__(self, other):
        return not self.__eq__(other)

    def __repr__(self):
        """
        Custom repr method to display the class name and dictionary content.
        """
        return f"{self.__class__.__name__}({super().__repr__()})"

    def __str__(self):
        """
        Custom string representation for printing.
        """
        result = "CliqueVector with the following cliques and factors:\n"
        for cl, factor in self.items():
            result += f"Clique: {cl}, Factor: {factor}\n"
        return result


    def exp(self):
        ans = { cl : self[cl].exp() for cl in self }
        return CliqueVector(ans)
    
    def log(self):
        ans = { cl : self[cl].log() for cl in self }
        return CliqueVector(ans)

    def dot(self, other):
        return sum( (self[cl]*other[cl]).sum() for cl in self )

    def size(self):
        return sum(self[cl].domain.size() for cl in self)

    def elementwise_mul(self, other):
        """
        Perform element-wise multiplication of two CliqueVectors.
        
        :param other: another CliqueVector with the same keys.
        :return: a new CliqueVector with the result of the element-wise multiplication.
        """
        if set(self.keys()) != set(other.keys()):
            raise ValueError("Both CliqueVectors must have the same cliques.")
        
        result = {cl: self[cl] * other[cl] for cl in self}
        return CliqueVector(result)


    def is_equal(self, other, tol=1e-6):
        """
        Check equality between this CliqueVector instance and another.
        :param other: The other CliqueVector instance to compare with.
        :param tol: Tolerance for numerical comparisons (default is 1e-6).
        :return: True if equal, False otherwise.
        """
        # Check if the other object is also a CliqueVector
        if not isinstance(other, CliqueVector):
            return False

        # Check if the cliques (keys) are the same
        if set(self.keys()) != set(other.keys()):
            return False

        # Iterate through each clique and compare the corresponding factors
        for cl in self:
            factor1 = self[cl]
            factor2 = other[cl]

            # Check if domains are the same
            if factor1.domain != factor2.domain:
                return False

            # Check if the values (data arrays) are close within the tolerance
            if not np.allclose(factor1.values, factor2.values, atol=tol):
                return False

        return True

