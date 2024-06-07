import numpy as np
import cvxpy as cp

class ClfQpController:
    def __init__(self, p1=1e0, clf_rate=1.0):
        # -------- optimizer parameters ---------
        # Control effort penalty
        self.p1 = p1
        # CLF decay rate
        self.rateV = clf_rate
        # -------- previous control ----------
        self.prev_u = None

    def generate_controller(self, rbt_config, sdf_val, sdf_grad):
        num_links = len(rbt_config)
        if self.prev_u is None:
            self.prev_u = np.zeros(num_links)

        # -------- CLF ---------
        V = 0.5 * sdf_val ** 2
        dV_dtheta = sdf_val * sdf_grad

        # -------- control input ---------
        u = cp.Variable(num_links)

        # formulate clf-qp constraints
        constraints = [
            dV_dtheta @ u + self.rateV * V <= 0,
            cp.abs(u) <= 0.5
        ]

        # -------- Solver --------
        # formulate objective function
        obj = cp.Minimize(self.p1 * cp.norm(u) ** 2)
        prob = cp.Problem(obj, constraints)

        # solving problem
        prob.solve(solver='SCS', verbose=False)

        # Check if the problem was solved successfully
        if prob.status == "infeasible":
            print("-------------------------- SOLVER NOT OPTIMAL -------------------------")
            print("[In solver] solver status: ", prob.status)
            self.prev_u = np.zeros(num_links)
            return np.zeros(num_links)
        else:
            self.prev_u = u.value
            return u.value