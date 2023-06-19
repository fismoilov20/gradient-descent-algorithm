import torch

class Optimizable:
    '''
    This interface is used for objects that have parameters that need 
    to be optimized. It is similar to torch.nn.Module, but with additional 
    features to support hyperoptimization. The Optimizable interface provides 
    more control over parameter detachments compared to the Parameter 
    interface used by torch.nn.Module. At the lowest level, an Optimizable 
    object operates as follows:
        o = MyOptimizable(...)
        o.initialize()
        loop {
            o.begin()
            o.zero_grad()
            loss = --compute loss function from parameters--
            loss.backward()
            o.step()
        }
    Optimizables recursively handle updates to their optimizers.
    '''
    def __init__(self, parameters, optimizer):
        self.parameters = parameters # a dict mapping names to tensors
        self.optimizer = optimizer   # which must itself be Optimizable!
        self.all_params_with_gradients = []

    def initialize(self):
        # Initialize the parameters of the object, for example using a Kaiming initializer. 
        # This step ensures that the parameters are initialized with suitable values before optimization begins.
        pass
    
    def begin(self):
    # Enable gradient tracking on the current parameters. This allows 
    # the gradients of the parameters to be computed during the optimization process.
        for param in self.all_params_with_gradients:
            param.grad = None
        self.all_params_with_gradients.clear()
        for name, param in self.parameters.items():
            param.requires_grad_() # keep gradient information...
            param.retain_grad()    # even if not a leaf...
            self.all_params_with_gradients.append(param)
        self.optimizer.begin()

    def zero_grad(self):
        # Set all gradients of the parameters to zero. This ensures that the gradients from previous iterations or computations are cleared before computing the gradients for the current optimization step.
        for param in self.all_params_with_gradients:
            param.grad = torch.zeros_like(param)
        self.optimizer.zero_grad()

    '''Note: at this point, you would typically call the .backward() method on 
    the loss function. This computes the gradients of the loss with respect to 
    the parameters, allowing for backpropagation and subsequent gradient updates 
    during the optimization process.'''

    def step(self):
        # Update parameters
        pass

class NoOpOptimizer(Optimizable):
    # NoOpOptimizer is a top-level optimizer in the stack that has no effect on the optimizers below it.
    
    def __init__(self):
        pass

    def initialize(self):
        pass

    def begin(self):
        pass

    def zero_grad(self):
        pass

    def step(self, params):
        pass

    def __str__(self):
        return ''

class SGD(Optimizable):
    # A hyperoptimizable SGD.
    def __init__(self, alpha=0.01, mu=0.0, optimizer=NoOpOptimizer()):
        self.mu = mu
        self.state = {}
        parameters = {
            'alpha': torch.tensor(alpha),
            'mu': torch.tensor(mu)
        }
        super().__init__(parameters, optimizer)

    def step(self, params):
        self.optimizer.step(self.parameters)
        for name, param in params.items():
            g = param.grad.detach()
            p = param.detach()
            if self.mu != 0.0:
                if name not in self.state:
                    buf = self.state[name] = g
                else:
                    buf = self.state[name].detach()
                    buf = buf * self.parameters['mu'] + g
                g = self.state[name] = buf
            params[name] = p - g * self.parameters['alpha']
        
    def __str__(self):
        return 'sgd / '+ str(self.optimizer)



class Adam(Optimizable):
    # A hyperoptimizable Adam optimizer.
    def clamp(x):
        return (x.tanh() + 1.) / 2.

    def unclamp(y):
        z = y * 2. - 1.
        return ((1. + z) / (1. - z)).log() / 2.

    def __init__(self, alpha=0.001, beta1=0.9, beta2=0.999, log_eps=-8., optimizer=NoOpOptimizer()):
        self.eps = 10. ** log_eps
        parameters = {
            'alpha': torch.tensor(alpha),
            'beta1': Adam.unclamp(torch.tensor(beta1)),
            'beta2': Adam.unclamp(torch.tensor(beta2)),
        }
        super().__init__(parameters, optimizer)
        self.num_stepments = 0
        self.cache = {}

    def step(self, params):
        self.num_stepments += 1
        self.optimizer.step(self.parameters)
        t = self.num_stepments
        beta1 = Adam.clamp(self.parameters['beta1'])
        beta2 = Adam.clamp(self.parameters['beta2'])
        for name, param in params.items():
            if name not in self.cache:
                self.cache[name] = {
                    'm': torch.zeros_like(param),
                    'v': torch.zeros_like(param) +\
                            self.eps
# It is important to NOTE that a small "fudge factor" is added in this step to account
# for the fact that the sqrt function is not differentiable exactly at zero.
                }
            g = param.grad.detach()
            self.cache[name]['m'] = m =\
                beta1 * self.cache[name]['m'].detach() + (1. - beta1) * g
            self.cache[name]['v'] = v =\
                beta2 * self.cache[name]['v'].detach() + (1. - beta2) * g * g
            self.all_params_with_gradients.append(m)
            self.all_params_with_gradients.append(v)

            m_hat = m / (1. - beta1 ** float(t))
            v_hat = v / (1. - beta2 ** float(t))

            dparam = m_hat / (v_hat ** 0.5 + self.eps)
            params[name] = param.detach() - self.parameters['alpha'] * dparam

    def __str__(self):
        return 'adam / ' + str(self.optimizer)

class ModuleWrapper(Optimizable):
    # This class aims to convert a torch.nn.Module into an Optimizable, taking care of 
    # the necessary internal mechanisms required for updating parameters correctly.
    def __init__(self, module, optimizer=NoOpOptimizer()):
        self.module = module
        parameters = {k:v for k, v in module.named_parameters(recurse=True)}
        super().__init__(parameters, optimizer)
    
    def initialize(self):
        self.optimizer.initialize()
    
    def zero_grad(self):
        # Set all gradients to zero. 
        self.module.zero_grad()
        for param in self.all_params_with_gradients:
            param.grad = torch.zeros_like(param)
        self.optimizer.zero_grad()
    
    def forward(self, *xyz):
        return self.module(*xyz)
    
    def train(self):
        self.module.train()
    
    def eval(self):
        self.module.eval()
    
    def step(self):
        self.optimizer.step(self.parameters)
        def set_param(m, k, v):
            kk = k
            while '.' in k:
                sm = k[:k.index('.')]
                k = k[k.index('.') + 1:]
                m = m._modules[sm]

            m._parameters[k] = None
            m._parameters[k] = self.parameters[kk]

        for k, v in self.module.named_parameters(recurse=True):
            set_param(self.module, k, v)