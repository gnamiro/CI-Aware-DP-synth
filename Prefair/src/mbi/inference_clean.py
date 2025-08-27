import numpy as np
from mbi import Domain, GraphicalModel, callbacks, CliqueVector
from scipy.sparse.linalg import LinearOperator, eigsh, lsmr, aslinearoperator
from scipy import optimize, sparse
from functools import partial
from collections import defaultdict
from sklearn.metrics import mutual_info_score
import torch

# np.random.seed(42)
debugging = True
class FactoredInference_torch:
    def __init__(self, domain, constraint, backend = 'numpy', structural_zeros = {}, metric='L2', log=False, iters=1000, warm_start=False, elim_order=None):
        """
        Class for learning a GraphicalModel from  measurements on a data distribution
        
        :param domain: The domain information (A Domain object)
        :param backend: numpy or torch backend
        :param structural_zeros: An encoding of the known (structural) zeros in the distribution.
            Specified as a dictionary where 
                - each key is a subset of attributes of size r
                - each value is a list of r-tuples corresponding to impossible attribute settings
        :param metric: The optimization metric.  May be L1, L2 or a custom callable function
            - custom callable function must consume the marginals and produce the loss and gradient
            - see FactoredInference._marginal_loss for more information
        :param log: flag to log iterations of optimization
        :param iters: number of iterations to optimize for
        :param warm_start: initialize new model or reuse last model when calling infer multiple times
        :param elim_order: an elimination order for the JunctionTree algorithm
            - Elimination order will impact the efficiency by not correctness.  
              By default, a greedy elimination order is used
        """
        self.domain = domain
        self.backend = backend
        self.metric = metric
        self.log = log
        self.iters = iters
        self.warm_start = warm_start
        self.history = []
        self.elim_order = elim_order
        self.constraint = constraint
        if backend == 'torch':
            from mbi.torch_factor import Factor
            self.Factor = Factor
        else:
            from mbi import Factor
            self.Factor= Factor

        self.structural_zeros = CliqueVector({})
        for cl in structural_zeros:
            dom = self.domain.project(cl)
            fact = structural_zeros[cl]
            self.structural_zeros[cl] = self.Factor.active(dom,fact)

    def estimate(self, measurements, total = None, engine='PMD', reg_method='CMI', tv_method=None, callback=None, options = {}):
        """ 
        Estimate a GraphicalModel from the given measurements

        :param measurements: a list of (Q, y, proj) tuples, where
            Q is the measurement matrix (a numpy array or scipy sparse matrix or LinearOperator)
            y is the answers to the measurement queries
            proj defines the marginal used for this measurement set (a subset of attributes)
        :param total: The total number of records (if known)
        :param engine: the optimization algorithm to use, options include:
            MD - Mirror Descent with armijo line search
            RDA - Regularized Dual Averaging
            IG - Interior Gradient
            PMD - Mirror Descent with Regularizer
        :param callback: a function to be called after each iteration of optimization
        :param options: solver specific options passed as a dictionary
            { param_name : param_value }
        
        :return model: A GraphicalModel that best matches the measurements taken
        """
        measurements = self.fix_measurements(measurements)
        options['callback'] = callback # NVM!!!
        self.reg_method = reg_method
        self.tv_method = tv_method
        if callback is None and self.log:
            options['callback'] = callbacks.Logger(self)
        if engine == 'MD':
            self.mirror_descent(measurements, total, **options) # CHANGE
        elif engine == 'MCD':
            self.mirror_coordinate_descent(measurements, total, **options)
        elif engine == 'PMD':
            self.proximal_mirror_descent(measurements, total, **options)
        elif engine == 'RDA':
            self.dual_averaging(measurements, total, **options)
        elif engine == 'IG':
            self.interior_gradient(measurements, total, **options)
        elif engine == 'Indep':
            self.zero_shot(measurements, total, **options)
        
        if self.constraint is not None:
            print("Okay Last Last Last check!!!")
            print(self.calculate_regularizer_loss(self.model.potentials, self.constraint, method=self.reg_method, tv_method=self.tv_method))
            # print(self.calculate_conditional_mutual_information(self.model.marginals, self.constraint))
        return self.model


    def fix_measurements(self, measurements):
        assert type(measurements) is list, 'measurements must be a list, given ' + measurements
        # assert all(len(m)==3 for m in measurements), \
            # 'each measurement must be a 3-tuple (Q, y,proj)'
        ans = []
        for Q, y, proj in measurements:
            assert Q is None or Q.shape[0] == y.size, 'shapes of Q and y are not compatible'
            if type(proj) is list:
                proj = tuple(proj)
            if type(proj) is not tuple:
                proj = (proj,)
            if Q is None:
                Q = sparse.eye(self.domain.size(proj))
            assert all(a in self.domain for a in proj), str(proj) + ' not contained in domain'
            assert Q.shape[1] == self.domain.size(proj), 'shapes of Q and proj are not compatible'
            ans.append( (Q, y, proj) )
        return ans

    def interior_gradient(self, measurements, total, lipschitz = None,c = 1,sigma = 1,callback=None):
        """ Use the interior gradient algorithm to estimate the GraphicalModel
            See https://epubs.siam.org/doi/pdf/10.1137/S1052623403427823 for more information

        :param measurements: a list of (Q, y, proj) tuples, where
            Q is the measurement matrix (a numpy array or scipy sparse matrix or LinearOperator)
            y is the answers to the measurement queries
            proj defines the marginal used for this measurement set (a subset of attributes)
        :param total: The total number of records (if known)
        :param lipschitz: the Lipchitz constant of grad L(mu)
            - automatically calculated for metric=L2
            - doesn't exist for metric=L1
            - must be supplied for custom callable metrics
        :param c, sigma: parameters of the algorithm
        :param callback: a function to be called after each iteration of optimization
        """
        assert self.metric != 'L1', 'dual_averaging cannot be used with metric=L1'
        assert not callable(self.metric) or lipschitz is not None,'lipschitz constant must be supplied'
        self._setup(measurements, total)
        # what are c and sigma?  For now using 1
        model = self.model
        domain, cliques, total = model.domain, model.cliques, model.total
        L = self._lipschitz(measurements) if lipschitz is None else lipschitz
        if self.log:
            print('Lipchitz constant:', L)
    
        theta = model.potentials
        x = y = z = model.belief_propagation(theta)
        c0 = c
        l = sigma/L
        for k in range(1, self.iters+1):
            a = (np.sqrt((c*l)**2 + 4*c*l) - l*c) / 2
            y = (1 - a)*x + a*z
            c *= (1-a)
            _, g = self._marginal_loss(y) 
            theta = theta - a/c/total * g
            z = model.belief_propagation(theta)
            x = (1-a)*x + a*z
            if callback is not None:
                callback(x)

        model.marginals = x
        model.potentials = model.mle(x) 

    def dual_averaging(self, measurements, total = None, lipschitz = None, callback=None):
        """ Use the regularized dual averaging algorithm to estimate the GraphicalModel
            See https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/xiao10JMLR.pdf

        :param measurements: a list of (Q, y, proj) tuples, where
            Q is the measurement matrix (a numpy array or scipy sparse matrix or LinearOperator)
            y is the answers to the measurement queries
            proj defines the marginal used for this measurement set (a subset of attributes)
        :param total: The total number of records (if known)
        :param lipschitz: the Lipchitz constant of grad L(mu)
            - automatically calculated for metric=L2
            - doesn't exist for metric=L1
            - must be supplied for custom callable metrics
        :param callback: a function to be called after each iteration of optimization
        """
        assert self.metric != 'L1', 'dual_averaging cannot be used with metric=L1'
        assert not callable(self.metric) or lipschitz is not None,'lipschitz constant must be supplied'
        self._setup(measurements, total)
        model = self.model
        domain, cliques, total = model.domain, model.cliques, model.total
        L = self._lipschitz(measurements) if lipschitz is None else lipschitz
        print('Lipchitz constant:', L)
        if L == 0: return
 
        theta = model.potentials
        gbar = CliqueVector({ cl : self.Factor.zeros(domain.project(cl)) for cl in cliques })
        w = v = model.belief_propagation(theta)
        beta = 0

        for t in range(1, self.iters+1):
            c = 2.0 / (t + 1)
            u = (1-c)*w + c*v
            _, g = self._marginal_loss(u) # not interested in loss of this query point
            gbar = (1-c)*gbar + c*g
            theta = -t*(t+1)/(4*L+beta)/self.model.total * gbar 
            v = model.belief_propagation(theta)
            w = (1-c)*w + c*v
           
            if callback is not None:
                callback(w)

        model.marginals = w
        model.potentials = model.mle(w) 

    def mirror_descent(self, measurements, total = None, stepsize = None, callback=None):
        """ Use the mirror descent algorithm to estimate the GraphicalModel
            See https://web.iem.technion.ac.il/images/user-files/becka/papers/3.pdf
        
        :param measurements: a list of (Q, y, proj) tuples, where
            Q is the measurement matrix (a numpy array or scipy sparse matrix or LinearOperator)
            y is the answers to the measurement queries
            proj defines the marginal used for this measurement set (a subset of attributes)
        :param stepsize: The step size function for the optimization (None or scalar or function)
            if None, will perform line search at each iteration (requires smooth objective)
            if scalar, will use constant step size
            if function, will be called with the iteration number
        :param total: The total number of records (if known)
        :param callback: a function to be called after each iteration of optimization
        """
        assert not (self.metric == 'L1' and stepsize is None), \
                'loss function not smooth, cannot use line search (specify stepsize)'
        
        self._setup(measurements, total)
        model = self.model
        cliques, theta = model.cliques, model.potentials
        lambda_reg = 0
        lambda_main = 1

        use_regularizer = self.constraint is not None

        mu = model.belief_propagation(theta)
        ans = self._marginal_loss(mu)
        if use_regularizer:
            reg_loss = self.calculate_conditional_mutual_information(theta, self.constraint)
        else:
            reg_loss = 0.0

        if lambda_main* ans[0] + lambda_reg*reg_loss == 0:
            return ans[0]

        nols = stepsize is not None
        if np.isscalar(stepsize):
            alpha = float(stepsize)
            stepsize = lambda t: alpha
        if stepsize is None:
            alpha = 1.0 / self.model.total**2
            stepsize = lambda t: 2.0*alpha

        # alpha = initial_alpha
        delta = 1e-3 # Small step for finite difference
    
        
        for t in range(1, 3*self.iters + 1):
            
            if t == 5001:
                lambda_main = 0
                lambda_reg = 10 # 10

            if callback is not None:
                callback(mu)

            omega, nu = theta, mu
            curr_loss = lambda_main*ans[0] + lambda_reg*reg_loss
            
            if use_regularizer:
                # marginal_grad = self._numerical_marginal_gradient_in_theta(theta)
                reg_grad = self._numerical_reg_gradient_spsa(theta, delta)
                norm_marginal = np.sqrt(ans[1].dot(ans[1]))
                reg_grad_ = reg_grad
                norm_regularizer = np.sqrt(reg_grad.dot(reg_grad))

                dL = lambda_main * ans[1] + lambda_reg * reg_grad
            else:
                dL = ans[1]

            if use_regularizer and debugging:
                print("Next Iteration...")
                print("Iteration: ", t)
                print("learning rate: ", alpha)
                print('Gradient Norm', np.sqrt(dL.dot(dL)))
                print('Gradient Norm Loss', np.sqrt(ans[1].dot(ans[1])))
                print('Gradient Norm Reg', np.sqrt(reg_grad.dot(reg_grad)))
                print("Current Loss", curr_loss)
                print("Current Marginal loss: ", ans[0])
                print(f"Current Regularizer loss = {lambda_reg} * CMI: ", lambda_reg*reg_loss)
                print("Current CMI: ", reg_loss)
                print("theta values: ")
                print(theta)
                print("model potentials")
                print(self.model.potentials)
                
                print("grad reg:")
                print(reg_grad)
                print("Current values of marginals")
                print(nu)

            alpha = stepsize(t)

            for i in range(25):
                if t > 5000:
                    alpha = 4*1.0 / self.model.total 
                theta = omega - alpha*dL
                mu = model.belief_propagation(theta)
                ans = self._marginal_loss(mu)
                if use_regularizer:
                    reg_loss = self.calculate_conditional_mutual_information(theta, self.constraint)
                else:
                    reg_loss = 0.0


                new_loss = lambda_main * ans[0] + lambda_reg * reg_loss
                
                if nols or curr_loss - new_loss >= 0.5*alpha*(dL.dot(nu-mu)):
                    # print("Stop step when Armijo condition satisfied:", i)
                    # loss_reduction = True
                    break
                
                alpha *= 0.5
                # else: 
                    
                # else:
                #     alpha = 1.0 / self.model.total**2  # Reduce alpha by 0.8 instead of halving
                
            # if loss_reduction:
            #     stepsize = lambda t: alpha

        model.potentials = theta
        model.marginals = mu
        if debugging:
            print("--"*16)
            print("potentials:")
            print(model.potentials)
            print("Marginals:")
            print(model.marginals)
            if use_regularizer:
                print("Final regularizer loss:")
                print(self.calculate_conditional_mutual_information(theta, self.constraint))

        return ans[0]

    def mirror_coordinate_descent(self, measurements, total=None, stepsize=None, callback=None):
        """
        Alternating optimization: 
        - First update using the loss function gradient while keeping the regularizer fixed.
        - Then update using the regularizer gradient while keeping the loss function fixed.
        """
        assert not (self.metric == 'L1' and stepsize is None), \
            'loss function not smooth, cannot use line search (specify stepsize)'

        self._setup(measurements, total)
        model = self.model
        cliques, theta = model.cliques, model.potentials
        lambda_reg = 1  # Regularization coefficient
    
        use_regularizer = self.constraint is not None

        mu = model.belief_propagation(theta)
        ans = self._marginal_loss(mu)  # Loss function f(theta)
        if use_regularizer:
            reg_loss = self._regularizer_loss_CMI_sklearn(mu, self.constraint)
        else:
            reg_loss = 0.0

        if ans[0] + lambda_reg * reg_loss == 0:
            return ans[0] + lambda_reg * reg_loss

        nols = stepsize is not None
        if np.isscalar(stepsize):
            alpha = float(stepsize)
            stepsize = lambda t: alpha
        if stepsize is None:
            alpha = 1.0 / self.model.total**2
            stepsize = lambda t: 2.0 * alpha

        
        print("Initial learning rate: ", alpha)
        delta = 1e-3 # Small step for finite difference
        # reg_grad = None
        for t in range(1, 2*self.iters + 1):
            print("Next Iteration...")

            if callback is not None:
                callback(mu)

            omega, nu = theta, mu  # Current theta and marginals
            curr_loss = ans[0] + lambda_reg * reg_loss  # Total current loss

            if t % 2 == 0:  # Odd iterations: update using the loss function f
                print(f"Iteration {t}: Update based on loss function (f)")
                dL = ans[1]  # Gradient of the loss function f
            else:  # Even iterations: update using the regularizer R
                print(f"Iteration {t}: Update based on regularizer (R)")
                if use_regularizer:
                    reg_grad = self._numerical_reg_gradient_spsa(theta, delta)
                    dL = ans[1] + lambda_reg * reg_grad
                else:
                    dL = ans[1]
                    print(dL)

            # print(f"Gradient Norm = {np.sqrt(dL.dot(dL))}")
            # print(f"Current Loss = {curr_loss}")
            # print(f"Current Loss Function f(theta): {ans[0]}")
            # print(f"Current Regularizer Loss R(theta): {reg_loss}")

            print("Iteration: ", t)
            print("learning rate: ", alpha)
            print('Gradient Norm', np.sqrt(dL.dot(dL)))
            print('Gradient Norm Loss', np.sqrt(ans[1].dot(ans[1])))
            if use_regularizer:
                print('Gradient Norm Reg', np.sqrt(reg_grad.dot(reg_grad)))
            print("Current Loss", curr_loss)
            print("Current Loss function: ", ans[0])
            print(f"Current Regularizer loss = {lambda_reg} * CMI: ", lambda_reg*reg_loss)
            print("Current CMI: ", reg_loss)
            print("theta values: ")
            print(theta)
            if use_regularizer:
                print("grad reg:")
                print(reg_grad)
            print("Current values of marginals")
            print(nu)

            # Update the parameters theta using the current gradient dL
            alpha = stepsize(t)
            if t == 5000:
                df = self.model.synthetic_data()
            for i in range(25):
                theta = omega - alpha * dL  # Gradient step
                mu = model.belief_propagation(theta)
                ans = self._marginal_loss(mu)  # Recompute loss after step
                if use_regularizer:
                    reg_loss = self._regularizer_loss_CMI_sklearn(mu, self.constraint)
                else:
                    reg_loss = 0.0
                # Armijo condition to check if step size is sufficient
                if nols or curr_loss - (ans[0] + lambda_reg * reg_loss) >= 0.5 * alpha * dL.dot(nu - mu):
                    print(f"Stop step when Armijo condition satisfied: {i}")
                    break
                # if t < 5000:
                alpha *= 0.5  # Reduce the step size

        # Update the model parameters after optimization
        model.potentials = theta
        model.marginals = mu
        print("--" * 16)
        print("Final potentials (theta):")
        print(model.potentials)
        print("Final Marginals (mu):")
        print(model.marginals)

        return ans[0]

    def proximal_mirror_descent(self, measurements, total = None, stepsize = None, callback=None):
        """ Use the mirror descent algorithm to estimate the GraphicalModel
            See https://web.iem.technion.ac.il/images/user-files/becka/papers/3.pdf
        
        :param measurements: a list of (Q, y, proj) tuples, where
            Q is the measurement matrix (a numpy array or scipy sparse matrix or LinearOperator)
            y is the answers to the measurement queries
            proj defines the marginal used for this measurement set (a subset of attributes)
        :param stepsize: The step size function for the optimization (None or scalar or function)
            if None, will perform line search at each iteration (requires smooth objective)
            if scalar, will use constant step size
            if function, will be called with the iteration number
        :param total: The total number of records (if known)
        :param callback: a function to be called after each iteration of optimization
        """
        assert not (self.metric == 'L1' and stepsize is None), \
                'loss function not smooth, cannot use line search (specify stepsize)'
        
        self._setup(measurements, total)
        model = self.model
        cliques, theta = model.cliques, model.potentials
        lambda_reg = 0
        lambda_main = 1

        use_regularizer = self.constraint is not None

        mu = model.belief_propagation(theta)
        ans = self._marginal_loss(mu)
        if use_regularizer:
            reg_loss = self.calculate_regularizer_loss(theta, self.constraint, method=self.reg_method, tv_method=self.tv_method)
        else:
            reg_loss = 0.0

        if lambda_main* ans[0] + lambda_reg*reg_loss == 0:
            return ans[0]

        nols = stepsize is not None
        if np.isscalar(stepsize):
            alpha = float(stepsize)
            stepsize = lambda t: alpha
        if stepsize is None:
            alpha = 1.0 / self.model.total
            stepsize = lambda t: 2.0*alpha

        # alpha = initial_alpha
        delta = 1e-3 # Small step for finite difference
    
        
        for t in range(1, 2*self.iters + 1):
            alpha = 1/100
            
            if callback is not None:
                callback(mu)

            # lambda_reg = min(30, t/300) # CMI
            lambda_reg = min(20, t/500)
            
            marg_grad_norm = np.sqrt(ans[1].dot(ans[1]))
            marg_grad = ans[1]
            marg_grad_rescaled = marg_grad / marg_grad_norm  # Ensure normalized

            omega, nu = theta, mu
            curr_loss = lambda_main*ans[0] + lambda_reg*reg_loss
            
            if use_regularizer:

                reg_grad = self._numerical_reg_gradient_spsa(theta)
                reg_grad_norm = np.sqrt(reg_grad.dot(reg_grad))
                reg_grad_rescaled = reg_grad / reg_grad_norm

                dL = lambda_main * marg_grad_rescaled + lambda_reg * reg_grad
            else:
                dL = marg_grad

            if use_regularizer and debugging:
                print("Next Iteration...")
                print("Iteration: ", t)
                print("learning rate: ", alpha)
                print('Gradient Norm', np.sqrt(dL.dot(dL)))
                print('Gradient Norm Loss', np.sqrt(ans[1].dot(ans[1])))
                print("Current Marginal Loss: ", ans[0])
                if use_regularizer:
                    print('Gradient Norm Reg', np.sqrt(reg_grad.dot(reg_grad)))
                    print(f"Current Regularizer loss = {lambda_reg} * CMI: ", lambda_reg*reg_loss)
                    # print("grad reg:")
                    # print(reg_grad)
                print(marg_grad_rescaled)
                print(reg_grad)
                print("Current CMI: ", reg_loss)
                print("Current Loss", curr_loss)

            for i in range(1):
                theta = omega - alpha*dL
                mu = model.belief_propagation(theta)
                ans = self._marginal_loss(mu)
                if use_regularizer:
                    reg_loss = self.calculate_regularizer_loss(theta, self.constraint, method=self.reg_method, tv_method=self.tv_method)
                else:
                    reg_loss = 0.0


                new_loss = lambda_main * ans[0] + lambda_reg * reg_loss
                
                if nols or curr_loss - new_loss >= 0.5*alpha*(dL.dot(nu-mu)):
                    break
                
                # alpha *= 0.5
                
        model.potentials = theta
        model.marginals = mu
        if debugging:
            print("--"*16)
            print("potentials:")
            print(model.potentials)
            print("Marginals:")
            print(model.marginals)
            if use_regularizer:
                print("Final regularizer loss:")
                print(self.calculate_regularizer_loss(theta, self.constraint, method=self.reg_method, tv_method=self.tv_method))

        return ans[0]
    
    def zero_shot(self, measurements, total, stepsize = None, callback=None):
        self._setup(measurements, total)
        model = self.model
        cliques, theta = model.cliques, model.potentials
        mu = model.belief_propagation(theta)
        ans = self._marginal_loss(mu)
        model.potentials = theta
        model.marginals = mu
        use_regularizer = True
        if debugging:
            print("--"*16)
            print("potentials:")
            print(model.potentials)
            print("Marginals:")
            print(model.marginals)
            if use_regularizer:
                print("Final regularizer loss:")
                print(self.calculate_conditional_mutual_information(theta, self.constraint))

        return ans[0]


    def _marginal_loss(self, marginals, metric=None):
        """ Compute the loss and gradient for a given dictionary of marginals

        :param marginals: A dictionary with keys as projections and values as Factors
        :return loss: the loss value
        :return grad: A dictionary with gradient for each marginal 
        """
        if metric is None:
            metric = self.metric

        if callable(metric):
            return metric(marginals)

        loss = 0.0
        gradient = { }

        for cl in marginals:
            mu = marginals[cl]
            gradient[cl] = self.Factor.zeros(mu.domain)
            for Q, y, proj in self.groups[cl]:
                c = 1
                mu2 = mu.project(proj)
                x = mu2.datavector()
                diff = c*(Q @ x - y)

                if metric == 'L1':
                    loss += abs(diff).sum()
                    sign = diff.sign() if hasattr(diff, 'sign') else np.sign(diff)
                    grad = c*(Q.T @ sign)
                else:
                    loss_component = 0.5*(diff @ diff)
                    loss += loss_component

                    grad = c*(Q.T @ diff)
                gradient[cl] += self.Factor(mu2.domain, grad)
        return float(loss), CliqueVector(gradient)


    def _numerical_marginal_gradient_in_theta(self, theta, epsilon=1e-3, perturbation_scale=0.01):
        import copy
        """
        Compute the gradient of the regularizer numerically using SPSA for each clique.
        :param theta: The parameter of the graphical model (CliqueVector instance).
        :param epsilon: Small perturbation value for finite difference calculation.
        :param perturbation_scale: Scale for random perturbations in SPSA.
        :return: A gradient CliqueVector with the same structure as theta.
        """
        
        grad_dict = {}
        for cl in theta:
            # Generate random perturbation vector
            delta = (np.random.choice([1, -1], size=theta[cl].values.shape) * perturbation_scale)
            
            # Perturb theta in both positive and negative directions along delta
            theta_pos = copy.deepcopy(theta)
            theta_neg = copy.deepcopy(theta)
            theta_pos[cl].values += delta
            theta_neg[cl].values -= delta

            # Calculate regularizer losses for perturbed thetas
            mu_pos = self.model.belief_propagation(theta_pos)
            marginal_loss_pos, _ = self._marginal_loss(mu_pos)
            
            mu_neg = self.model.belief_propagation(theta_neg)
            marginal_loss_neg, _ = self._marginal_loss(mu_neg)

            # SPSA gradient approximation: (reg_loss_pos - reg_loss_neg) / (2 * delta)
            grad_factor = theta[cl].copy()
            print(f"for the {cl}")
            print(marginal_loss_pos - marginal_loss_neg)
            grad_factor.values = (marginal_loss_pos - marginal_loss_neg) / (2 * delta)
            
            # Store the computed gradient for the current clique
            grad_dict[cl] = grad_factor

        # Return the gradient as a CliqueVector
        return CliqueVector(grad_dict)


    def _numerical_reg_gradient_spsa(self, theta, epsilon=1e-3, perturbation_scale=1e-2):
        import copy
        """
        Compute the gradient of the regularizer numerically using SPSA for each clique.
        :param theta: The parameter of the graphical model (CliqueVector instance).
        :param epsilon: Small perturbation value for finite difference calculation.
        :param perturbation_scale: Scale for random perturbations in SPSA.
        :return: A gradient CliqueVector with the same structure as theta.
        """
        grad_dict = {}
        print("---test---"*10)
        for cl in theta:
            # Generate random perturbation vector
            delta = (np.random.choice([1, -1], size=theta[cl].values.shape) * perturbation_scale)
            # Perturb theta in both positive and negative directions along delta
            theta_pos = copy.deepcopy(theta)
            theta_neg = copy.deepcopy(theta)
            theta_pos[cl].values += delta
            theta_neg[cl].values -= delta

            # Calculate regularizer losses for perturbed thetas
            # mu_pos = self.model.belief_propagation(theta_pos)
            reg_loss_pos = self.calculate_regularizer_loss(theta_pos, self.constraint, method=self.reg_method, tv_method=self.tv_method)
            
            # mu_neg = self.model.belief_propagation(theta_neg)
            reg_loss_neg = self.calculate_regularizer_loss(theta_neg, self.constraint, method=self.reg_method, tv_method=self.tv_method)

            # SPSA gradient approximation: (reg_loss_pos - reg_loss_neg) / (2 * delta)
            grad_factor = theta[cl].copy()
            grad_factor.values = (reg_loss_pos - reg_loss_neg) / (2 * delta)
            
            # Store the computed gradient for the current clique
            grad_dict[cl] = grad_factor

        # Return the gradient as a CliqueVector
        return CliqueVector(grad_dict)


    def _regularizer_loss_L1(self, marginals, constraint):
        """
        constraint = X, Y, Z
        Measure the dependency between two attributes using the L1 norm of the difference 
        between the actual counts and the expected counts (assuming independence).
        """
        assert len(constraint) == 3, "your constraint is not conditional"
        regularizer_loss = 0
        
        for cl in marginals:
            mu = marginals[cl]
            _count_Z = mu.project(constraint[-1])
            _count_XZ = mu.project([constraint[0], constraint[-1]])
            _count_YZ = mu.project([constraint[1], constraint[-1]])
            _count_XYZ = mu.project(constraint)
            # print("lets print the marginals for constraints")
            # print(_count_Z)
            # print(_count_XZ)
            # print(_count_YZ)
            # print(_count_XYZ)

            # Get the actual counts as numpy arrays
            count_Z = np.array(_count_Z.datavector(), dtype=np.float32)
            count_XZ = np.array(_count_XZ.datavector(), dtype=np.float32).reshape(10, 2)  # Reshape to (10, 2)
            count_YZ = np.array(_count_YZ.datavector(), dtype=np.float32).reshape(10, 2)  # Reshape to (10, 2)
            count_XYZ = np.array(_count_XYZ.datavector(), dtype=np.float32)

            # Calculate P(X|Z) and P(Y|Z) as probabilities
            p_x_given_z = count_XZ / count_Z[:, np.newaxis]  # Shape should now be (10, 2)
            p_y_given_z = count_YZ / count_Z[:, np.newaxis]  # Shape should now be (10, 2)

            # Calculate P(XY|Z) = P(X|Z) * P(Y|Z) for each Z

            p_xy_given_z_list = []
            for i in range(p_x_given_z.shape[0]):  # Loop over each Z
                p_y_given_z_i = p_y_given_z[i].reshape(-1, 1)  # Shape (2, 1) - values for Y given Z=i
                p_x_given_z_i = p_x_given_z[i].reshape(1, -1)  # Shape (1, 2) - values for X given Z=i
                p_xy_given_z_i = p_y_given_z_i * p_x_given_z_i  # Shape (2, 2)

                # Flatten and store P(XY|Z)
                p_xy_given_z_list.append(p_xy_given_z_i.flatten())

            # Concatenate all P(XY|Z) values across Z
            p_xy_given_z = np.concatenate(p_xy_given_z_list)  # Shape (40,)
            # print(p_xy_given_z.shape, p_x_given_z.shape, p_y_given_z.shape, count_XYZ.shape, count_Z.shape)
            # Repeat count_Z appropriately to match the shape of p_xy_given_z
            count_Z_repeated = np.repeat(count_Z, p_x_given_z.shape[1] * p_y_given_z.shape[1])

            # Calculate expected counts assuming independence
            expected_count_xyz = p_xy_given_z * count_Z_repeated

            # Calculate the L1 norm of the difference between actual and expected counts
            diff = count_XYZ - expected_count_xyz
            l1_norm = np.linalg.norm(diff, 1)
            regularizer_loss += l1_norm

        return float(regularizer_loss)

    def _regularizer_loss_CMI_sklearn(self, marginals, constraint):
        """
        constraint = X, Y, Z
        Simple implementation of KL divergence for calculating the conditional dependency between two attributes.
        """
        assert len(constraint) == 3, "your constraint is not conditional"

        flatten_constrained = [attr for proj in constraint for attr in proj]  
        
        projections = [
            constraint[-1],                          # Z
            constraint[0] + constraint[-1],          # X + Z
            constraint[1] + constraint[-1],          # Y + Z
            flatten_constrained                        # X + Y + Z
        ]
        


        projected_marginals = self.model.calculate_many_marginals(projections, marginals)
        regularizer_loss = 0
        
        # Extract counts for each relevant subset
        _count_Z = projected_marginals[tuple(constraint[-1])]
        _count_XZ = projected_marginals[tuple(constraint[0] + constraint[-1])]
        _count_YZ = projected_marginals[tuple(constraint[1] + constraint[-1])]
        _count_XYZ = projected_marginals[tuple(flatten_constrained)]  # Flatten constraint for XYZ

        # Convert counts to probabilities as numpy arrays
        prob_Z = np.array(_count_Z.dataprobs(False), dtype=np.float32)
        prob_XZ = np.array(_count_XZ.dataprobs(False), dtype=np.float32)
        prob_YZ = np.array(_count_YZ.dataprobs(False), dtype=np.float32)
        prob_XYZ = np.array(_count_XYZ.dataprobs(False), dtype=np.float32)

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
                        cmi += p_xyz * np.log2(p_xyz * p_z / (p_xz * p_yz + 1e-12) + 1e-12)

        regularizer_loss += max(cmi, 0)  # Clamp to non-negative values

        return float(regularizer_loss)

    def calculate_regularizer_loss(self, potentials, constraint, method='CMI', tv_method='L1'):
        """
        Choose to calculate either Conditional Mutual Information (CMI) or Total Variation Distance (TVD).
        
        Parameters:
        - potentials: The model potentials (parameters).
        - constraint: A list of three elements, where each element is a list of attributes 
                    (e.g., [X_list, Y_list, Z_list]).
        - method: 'CMI' for calculating Conditional Mutual Information, or 'TVD' for Total Variation Distance.
        - tv_method: The method to calculate TV distance ('L1' or 'L2') if method='TVD'.

        Returns:
        - float: The regularizer loss based on the chosen method.
        """
        if method == 'CMI':
            return self.calculate_conditional_mutual_information(potentials, constraint)
        elif method == 'TVD':
            # Calculate TVD using the specified method (L1 or L2)
            _count_Z = self.model.project(tuple(constraint[-1]), potentials)
            _count_XZ = self.model.project(tuple(constraint[0] + constraint[-1]), potentials)
            _count_YZ = self.model.project(tuple(constraint[1] + constraint[-1]), potentials)
            _count_XYZ = self.model.project(tuple([attr for proj in constraint for attr in proj]), potentials)

            prob_Z = np.array(_count_Z.dataprobs(False), dtype=np.float32)
            prob_XZ = np.array(_count_XZ.dataprobs(False), dtype=np.float32)
            prob_YZ = np.array(_count_YZ.dataprobs(False), dtype=np.float32)
            prob_XYZ = np.array(_count_XYZ.dataprobs(False), dtype=np.float32)

            prob_Z_b, prob_XZ_b, prob_YZ_b = align_shapes(prob_Z, prob_XZ, prob_YZ, prob_XYZ)
            
            return self.calculate_total_variation_distance(prob_XYZ, prob_XZ_b, prob_YZ_b, prob_Z_b, method=tv_method)
        else:
            raise ValueError("Method must be either 'CMI' or 'TVD'.")

    def calculate_conditional_mutual_information(self, potentials, constraint):
        """
        Calculate conditional mutual information for a generalized constraint.
        
        Parameters:
        - marginals: The marginals calculated for the model.
        - constraint: A list of three elements, where each element is a list of attributes 
                    (e.g., [X_list, Y_list, Z_list]).
        
        Returns:
        - float: Conditional mutual information based on the specified constraints.
        """
        assert len(constraint) == 3, "constraint must have three elements (e.g., [X, Y, Z])"
        
        flatten_constrained = [attr for proj in constraint for attr in proj]  
        # Define projections based on the constraint list
        projections = [
            constraint[-1],                          # Z
            constraint[0] + constraint[-1],          # X + Z
            constraint[1] + constraint[-1],          # Y + Z
            flatten_constrained                        # X + Y + Z
        ]

        regularizer_loss = 0

        _count_Z = self.model.project(tuple(constraint[-1]), potentials)
        _count_XZ = self.model.project(tuple(constraint[0] + constraint[-1]), potentials)
        _count_YZ = self.model.project(tuple(constraint[1] + constraint[-1]), potentials)
        _count_XYZ = self.model.project(tuple(flatten_constrained), potentials)
        
        # Convert counts to probabilities as numpy arrays
        prob_Z = np.array(_count_Z.dataprobs(False), dtype=np.float32)
        prob_XZ = np.array(_count_XZ.dataprobs(False), dtype=np.float32)
        prob_YZ = np.array(_count_YZ.dataprobs(False), dtype=np.float32)
        prob_XYZ = np.array(_count_XYZ.dataprobs(False), dtype=np.float32)

        prob_Z_b, prob_XZ_b, prob_YZ_b = align_shapes(prob_Z, prob_XZ, prob_YZ, prob_XYZ)
        cmi_values = prob_XYZ * np.log2(
            prob_XYZ * prob_Z_b /
            (prob_XZ_b * prob_YZ_b + 1e-12) + 1e-12
        )
        cmi = np.sum(cmi_values)
        regularizer_loss += max(cmi, 0)

        return float(regularizer_loss)

    def calculate_total_variation_distance(self, prob_XYZ, prob_XZ_b, prob_YZ_b, prob_Z_b, method='L1'):
        """
        Calculate the Total Variation (TV) distance between two distributions.

        Parameters:
        - prob_XYZ: The probability distribution for P_XYZ.
        - prob_XZ_b: The probability distribution for P_XZ.
        - prob_YZ_b: The probability distribution for P(Y|Z).
        - method: The method to calculate TV distance ('L1' or 'L2').

        Returns:
        - float: The Total Variation distance between the distributions.
        """
        if method == 'L1':
            # L1 norm: Manhattan distance
            tv_distance = np.sum(np.abs(prob_XYZ - prob_XZ_b * prob_YZ_b / (prob_Z_b+1e-12)))
        elif method == 'L2':
            # L2 norm: Euclidean distance
            tv_distance = np.sqrt(np.sum((prob_XYZ - prob_XZ_b * prob_YZ_b / (prob_Z_b+1e-12)) ** 2))
        else:
            raise ValueError("Method must be either 'L1' or 'L2'.")

        return tv_distance



    def _regularizer_loss_CMI(self, marginals, constraint):
        """
        constraint = X, Y, Z
        Simple implementation of KL divergence for calculating the conditional dependency between two attributes.
        """
        assert len(constraint) == 3, "your constraint is not conditional"
        regularizer_loss = 0
        
        for cl in marginals:
            mu = marginals[cl]
            _count_Z = mu.project(constraint[-1])
            _count_XZ = mu.project([constraint[0], constraint[-1]])
            _count_YZ = mu.project([constraint[1], constraint[-1]])
            _count_XYZ = mu.project(constraint)

            prob_Z = _count_Z.dataprobs(False).T
            prob_ZX = _count_XZ.dataprobs(False).T
            prob_ZY = _count_YZ.dataprobs(False).T
            prob_ZYX = _count_XYZ.dataprobs(False).T

            # Convert all probabilities to numpy arrays
            prob_Z = np.array(prob_Z, dtype=np.float32)
            prob_ZX = np.array(prob_ZX, dtype=np.float32)
            prob_ZY = np.array(prob_ZY, dtype=np.float32)
            prob_ZYX = np.array(prob_ZYX, dtype=np.float32)

            # Calculate P(X|Z) and P(Y|Z)
            p_x_given_z = prob_ZX / prob_Z[:, np.newaxis]
            p_y_given_z = prob_ZY / prob_Z[:, np.newaxis]

            # Reshape prob_ZYX for division
            prob_zyx_array = prob_ZYX.reshape(10, -1)
            p_yx_given_z = prob_zyx_array / prob_Z[:, np.newaxis]

            # Calculate P(X|Z) * P(Y|Z) for each Z
            p_y_x_given_z_list = []
            for i in range(p_yx_given_z.shape[0]):  # Loop over each Z
                p_y_given_z_i = p_y_given_z[i].reshape(-1, 1)  # Shape (2, 1) - 2 values for Y given Z=i
                p_x_given_z_i = p_x_given_z[i].reshape(1, -1)  # Shape (2, 1) - 2 values for X given Z=i
                p_y_x_given_z_i = p_y_given_z_i * p_x_given_z_i  # Shape (2, 2)
                p_y_x_given_z_list.append(p_y_x_given_z_i.flatten())

            p_y_x_given_z = np.concatenate(p_y_x_given_z_list)  # Shape (40,)

            # Add a small epsilon for numerical stability
            epsilon = 1e-8
            p_yx_given_z = p_yx_given_z.flatten() + epsilon
            p_y_x_given_z = p_y_x_given_z + epsilon

            # Calculate mutual information
            mi = np.sum(prob_zyx_array.flatten() * np.log2(p_yx_given_z / p_y_x_given_z))
            mi = max(mi, 0)  # Clamp to non-negative values
            regularizer_loss += mi

        return float(regularizer_loss)

    def _setup(self, measurements, total):
        """ Perform necessary setup for running estimation algorithms
       
        1. If total is None, find the minimum variance unbiased estimate for total and use that
        2. Construct the GraphicalModel 
            * If there are structural_zeros in the distribution, initialize factors appropriately
        3. Pre-process measurements into groups so that _marginal_loss may be evaluated efficiently
        """
        if total is None:
            # find the minimum variance estimate of the total given the measurements
            variances = np.array([])
            estimates = np.array([])
            for Q, y, proj in measurements:
                o = np.ones(Q.shape[1])
                v = lsmr(Q.T, o, atol=0, btol=0)[0]
                if np.allclose(Q.T.dot(v), o):
                    variances = np.append(variances, np.dot(v, v))
                    estimates = np.append(estimates, np.dot(v, y))
            if estimates.size == 0:
                total = 1
            else:
                variance = 1.0 / np.sum(1.0 / variances)
                estimate = variance * np.sum(estimates / variances)
                total = max(1, estimate)
        print(f"The total number is:: ", total)
        #if not self.warm_start or not hasattr(self, 'model'):
        # initialize the model and parameters
        cliques = [m[-1] for m in measurements] 
        if self.structural_zeros is not None:
            cliques += list(self.structural_zeros.keys())

        model = GraphicalModel(self.domain,cliques,total,elimination_order=self.elim_order)

        model.potentials = CliqueVector.zeros(self.domain, model.cliques)
        model.potentials.combine(self.structural_zeros)
        if self.warm_start and hasattr(self, 'model'):
            model.potentials.combine(self.model.potentials)
        self.model = model  
 
        # group the measurements into model cliques 
        cliques = self.model.cliques
        #self.groups = { cl : [] for cl in cliques }
        self.groups = defaultdict(lambda: [])
        for Q,y,proj in measurements:
            if self.backend == 'torch':
                # import torch
                device = self.Factor.device
                y = torch.tensor(y, dtype=torch.float32, device=device)
                if isinstance(Q, np.ndarray):
                    Q = torch.tensor(Q, dtype=torch.float32, device=device)
                elif sparse.issparse(Q):
                    Q = Q.tocoo()
                    idx = torch.LongTensor([Q.row, Q.col])
                    vals = torch.FloatTensor(Q.data)
                    Q = torch.sparse.FloatTensor(idx, vals).to(device)

                # else Q is a Linear Operator, must be compatible with torch 
            m = (Q, y, proj)
            for cl in sorted(cliques, key=model.domain.size):
                if set(proj) <= set(cl):
                    self.groups[cl].append(m)
                    break

    def _lipschitz(self, measurements):
        """ compute lipschitz constant for L2 loss 

            Note: must be called after _setup
        """
        eigs = { cl : 0.0 for cl in self.model.cliques }
        for Q, _, proj in measurements:
            for cl in self.model.cliques:
                if set(proj) <= set(cl):
                    n = self.domain.size(cl)
                    p = self.domain.size(proj)
                    Q = aslinearoperator(Q)
                    Q.dtype = np.dtype(Q.dtype)
                    eig = eigsh(Q.H * Q, 1)[0][0]
                    eigs[cl] += eig * n / p
                    break
        return max(eigs.values())

    def infer(self, measurements, total=None, engine='MD', callback=None, options={}):
        import warnings
        message = "Function infer is deprecated.  Please use estimate instead."
        warnings.warn(message, DeprecationWarning)
        return self.estimate(measurements, total, engine, callback, options)



def align_shapes(prob_Z, prob_XZ, prob_YZ, prob_XYZ):
    """
    Align the shapes of prob_Z, prob_XZ, and prob_YZ to match prob_XYZ for broadcasting.

    Parameters:
    - prob_Z: P(Z), smallest array (e.g., (7, 9, 6, 5)).
    - prob_XZ: P(X, Z), next largest array (e.g., (3, 7, 9, 6, 5)).
    - prob_YZ: P(Y, Z), next largest array (e.g., (2, 3, 7, 9, 6, 5)).
    - prob_XYZ: P(X, Y, Z), full joint probability array.

    Returns:
    - Aligned versions of prob_Z, prob_XZ, and prob_YZ for broadcasting with prob_XYZ.
    """
    shape_Z = prob_Z.shape
    shape_XZ = prob_XZ.shape
    shape_YZ = prob_YZ.shape
    shape_XYZ = prob_XYZ.shape

    # Determine how many dimensions each component (X, Y, Z) has in prob_XYZ
    num_X = len(shape_XYZ) - len(shape_YZ)  # Number of X dimensions
    num_Y = len(shape_YZ) - len(shape_Z)                                   # Number of Y dimensions
    num_Z = len(shape_Z)                                                   # Number of Z dimensions

    # Adjust the shapes dynamically
    aligned_prob_Z = prob_Z.reshape((1,) * num_X + (1,) * num_Y + shape_Z)  # Pad Z for X and Y dimensions
    aligned_prob_XZ = prob_XZ.reshape(shape_XZ[:num_X] + (1,) * num_Y + shape_XZ[num_X:])  # Pad XZ for Y dimensions
    aligned_prob_YZ = prob_YZ.reshape((1,) * num_X + shape_YZ[:num_Y] + shape_YZ[num_Y:])  # Pad YZ for X dimensions

    return aligned_prob_Z, aligned_prob_XZ, aligned_prob_YZ
