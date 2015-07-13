from copy import deepcopy
from frozendict import frozendict
from ProbabPy import OnePDF, OnePMF


class HiddenMarkovModel:
    def __init__(self, state_var, observation_var,
                 state_prior_pdf, transition_pdf_template, observation_pdf_template):
        self.state_var = state_var
        self.observation_var = observation_var
        self.state_prior_pdf = state_prior_pdf
        self.transition_pdf_template = transition_pdf_template
        self.observation_pdf_template = observation_pdf_template

    def transition_pdf(self, t):
        return self.transition_pdf_template.shift_time_subscripts(t)

    def observation_pdf(self, t):
        return self.observation_pdf_template.shift_time_subscripts(t)

    def forward_pdf(self, t___list, observations___dict={}):
        if isinstance(t___list, int):
            t = t___list
            if t == 0:
                f = self.state_prior_pdf * self.observation_pdf(t)
                if t in observations___dict:
                    f = f.at({(self.observation_var, t): observations___dict[t]})
                return f
            else:
                f = (self.forward_pdf(t - 1, observations___dict) *
                     self.transition_pdf(t))\
                    .marg((self.state_var, t - 1))\
                    * self.observation_pdf(t)
                if t in observations___dict:
                    f = f.at({(self.observation_var, t): observations___dict[t]})
                return f
        elif isinstance(t___list, (list, range, tuple)):
            d = {}
            t = 0
            f = [self.state_prior_pdf * self.observation_pdf(t)]
            if t in observations___dict:
                f[t] = f[t].at({(self.observation_var, t): observations___dict[t]})
            if t in t___list:
                d[t] = f[t]
            for t in range(1, max(t___list) + 1):
                f += [(f[t - 1] * self.transition_pdf(t))
                      .marg((self.state_var, t - 1))
                      * self.observation_pdf(t)]
                if t in observations___dict:
                    f[t] = f[t].at({(self.observation_var, t): observations___dict[t]})
                if t in t___list:
                    d[t] = f[t]
            return d

    def backward_factor(self, t___list, observations___dict={}, max_t=0):
        T = max(max(observations___dict.keys()), max_t)
        if isinstance(t___list, int):
            t = t___list
            if t == T:
                state_var_symbol = {(self.state_var, t): self.observation_pdf(t).Vars[(self.state_var, t)]}
                if self.observation_pdf_template.is_discrete_finite():
                    var_values___frozen_dicts = self.observation_pdf(t).Params['NegLogP'].keys()
                    state_var_domain =\
                        set(frozendict({(self.state_var, t): var_values___frozen_dict[(self.state_var, t)]})
                            for var_values___frozen_dict in var_values___frozen_dicts)
                    return OnePMF(var_names_and_syms=state_var_symbol, var_names_and_values=state_var_domain,
                                  cond={(self.state_var, t): None})
                else:
                    return OnePDF(cond={(self.observation_var, t): None})
            else:
                b = self.transition_pdf(t + 1) * self.observation_pdf(t + 1)
                if (t + 1) in observations___dict:
                    b = b.at({(self.observation_var, t + 1): observations___dict[t + 1]})
                b = (b * self.backward_factor(t + 1, observations___dict))\
                    .marg((self.state_var, t + 1))
                return b
        elif isinstance(t___list, (list, range, tuple)):
            d = {}
            t = T
            state_var_symbol = {(self.state_var, t): self.observation_pdf(t).Vars[(self.state_var, t)]}
            if self.observation_pdf_template.is_discrete_finite():
                var_values___frozen_dicts = self.observation_pdf(t).Param['NegLogP'].keys()
                state_var_domain =\
                    set(frozendict({(self.state_var, t): var_values___frozen_dict[(self.state_var, t)]})
                        for var_values___frozen_dict in var_values___frozen_dicts)
                b = {t: OnePMF(var_names_and_syms=state_var_symbol, var_names_and_values=state_var_domain,
                               cond={(self.state_var, t): None})}
            else:
                b = {t: OnePDF(cond={(self.observation_var, t): None})}
            if t in t___list:
                d[t] = b[t]
            for t in reversed(range(min(t___list), T)):
                b[t] = self.transition_pdf(t + 1) * self.observation_pdf(t + 1)
                if (t + 1) in observations___dict:
                    b[t] = b[t].at({(self.observation_var, t + 1): observations___dict[t + 1]})
                b[t] = (b[t] * b[t + 1])\
                    .marg((self.state_var, t + 1))
                if t in t___list:
                    d[t] = b[t]
            return d

    def infer_state(self, t___list, observations___dict={}):
        conditions___dict = {}
        for t, value in observations___dict.items():
            conditions___dict[(self.observation_var, t)] = value
        if isinstance(t___list, int):
            t = t___list
            return (self.forward_pdf(t, observations___dict) *
                    self.backward_factor(t, observations___dict))\
                .cond(conditions___dict)\
                .norm()
        elif isinstance(t___list, (list, range, tuple)):
            d = {}
            forward = self.forward_pdf(t___list, observations___dict)
            backward = self.backward_factor(t___list, observations___dict)
            for t in t___list:
                d[t] = (forward[t] * backward[t])\
                    .cond(conditions___dict)\
                    .norm()
            return d

    def max_a_posteriori_joint_distributions(self, observations___list, leave_last_state_unoptimized=False):
        observations___list = deepcopy(observations___list)
        T = len(observations___list) - 1
        if T:
            last_observation = observations___list.pop(T)
            f = self.max_a_posteriori_joint_distributions(observations___list, True) *\
                (self.transition_pdf(T) *
                 self.observation_pdf(T).at({(self.observation_var, T): last_observation})).max()
        else:
            f = self.state_prior_pdf * self.observation_pdf(0).at({(self.observation_var, 0): observations___list[0]})
        if leave_last_state_unoptimized:
            return f.max(leave_unoptimized={(self.state_var, T)})
        else:
            return f.max()

    def max_a_posteriori_state_sequence(self, observations___list):
        m = self.max_a_posteriori_joint_distributions(observations___list)
        if m.is_discrete_finite():
            d = set(m.Param['NegLogP']).pop()
        else:
            d = m.scope
        return [d[(self.state_var, t)] for t in range(len(observations___list))]
