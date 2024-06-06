import numpy as np
import cvxpy as cp

class ClfCbfController:
    def __init__(self, p1=1e0, p2=1e2, clf_rate=1.0, cbf_rate=0.5):
        # -------- optimizer parameters ---------
        # Control Lipschitz penalty
        self.p1 = p1
        # CLF relaxation penalty (for CLF-CBF-QP)
        self.p2 = p2

        # CLF decay rate
        self.rateV = clf_rate
        # CBF decay rate
        self.rateh = cbf_rate

        # -------- solver status ---------
        self.solve_fail = True
        
        # -------- previous control ----------
        self.prev_u = None

        self.ref_u = np.ones(4) * 0.1

    def generate_controller(self, rbt_config, goal_config, cbf_h_val, cbf_h_grad, cbf_t_grad):
        """
        This function generates control input (left edge length) for a N-link 2D continuum arm using CLF-CBF-QP method
        """
        num_links = len(rbt_config)
        
        if self.prev_u is None:
            self.prev_u = np.zeros(num_links)

        # -------- Clf ---------
        V = 0.5 * np.sum((rbt_config - goal_config) ** 2)


        dV_dtheta = rbt_config - goal_config

        # -------- control input ---------
        u = cp.Variable(num_links)
        delta = cp.Variable()

        print('barrier value:', cbf_h_val)
        # print('barrier grad:', cbf_h_grad)
        # print('time derivative:', cbf_t_grad)
        
        # formulate clf-cbf-qp constraints
        constraints = [
            #dV_dtheta.T @ u + self.rateV * V <= delta,
            cbf_h_grad @ u + cbf_t_grad + self.rateh * cbf_h_val  >= 0,
            delta >= 0.0,
            cp.abs(u) <= 0.5
        ]

        # -------- Solver --------
        # formulate objective function
        # obj = cp.Minimize(self.p1 * cp.norm(self.prev_u - u) ** 2 + self.p2 * cp.square(delta))

        obj = cp.Minimize(self.p1 * cp.norm(u) ** 2)

        prob = cp.Problem(obj, constraints)

        # solving problem
        prob.solve(solver='SCS', verbose=False)

        # Check if the problem was solved successfully
        if prob.status == "infeasible":
            self.solve_fail = True
            print("-------------------------- SOLVER NOT OPTIMAL -------------------------")
            print("[In solver] solver status: ", prob.status)
            print("[In solver] h = ", cbf_h_val)
            print("[In solver] dot_h = ", cbf_h_grad)
            self.prev_u = np.zeros(num_links)
            return np.zeros(num_links)
        else:
            self.solve_fail = False
            self.prev_u = u.value


        return u.value