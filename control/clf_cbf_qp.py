import numpy as np
import cvxpy as cp

class ClfCbfController:
    def __init__(self, p1=1e0, p2=1e2, clf_rate=1.0, cbf_rate=3.0):
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

    def generate_controller(self, rbt_config, cbf_h_val, cbf_h_grad, cbf_t_grad, sdf_val, sdf_grad):
        """
        This function generates control input (left edge length) for a N-link 2D continuum arm using CLF-CBF-QP method
        """
        num_links = len(rbt_config)

        num_obstacles = len(cbf_h_val)
        
        if self.prev_u is None:
            self.prev_u = np.zeros(num_links)


        # -------- CLF ---------
        V = 0.5 * sdf_val ** 2
        dV_dtheta = sdf_val * sdf_grad

        # print('lyapunov value:', V)

        # -------- control input ---------
        u = cp.Variable(num_links)
        delta = cp.Variable()

        # assume osafety margin of 0.1
        # print('barrier value:', cbf_h_val - 0.1)


        # print('barrier grad:', cbf_h_grad)
        # print('time derivative:', cbf_t_grad)
        
        # formulate clf-cbf-qp constraints
        constraints = [
            dV_dtheta @ u + self.rateV * V <= delta,
            #cbf_h_grad @ u + cbf_t_grad + self.rateh * (cbf_h_val - 0.1)  >= 0,
            delta >= 0.0,
            cp.abs(u) <= 0.2
        ]

        for i in range(num_obstacles):
            # print('ith obstacle:', cbf_h_vals[i])
            # print('ith velocity:', cbf_t_grads[i])
            constraints.append(cbf_h_grad[i].T @ u + cbf_t_grad[i] + self.rateh * (cbf_h_val[i] - 0.1) >= 0)

        # -------- Solver --------
        # formulate objective function
        obj = cp.Minimize(self.p1 * cp.norm(self.prev_u - u) ** 2 + self.p2 * cp.square(delta))

        # this is pure CBF-QP
        # obj = cp.Minimize(self.p1 * cp.norm(u) ** 2)

        prob = cp.Problem(obj, constraints)

        # solving problem
        prob.solve(solver='ECOS', verbose=False)

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