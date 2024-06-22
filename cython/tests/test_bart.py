import numpy as np
import pymc as pm

import pymc_bart as pmb


def main():

    np.random.seed(0)
    n = 100
    X = np.random.uniform(0, 10, n)
    Y = np.sin(X) + np.random.normal(0, 0.5, n)

    with pm.Model() as model:
        mu = pmb.BART("mu", X.reshape(-1, 1), Y, m=20)
        y = pm.Normal("y", mu, sigma=1., observed=Y)
        step = pmb.PGBART([mu], num_particles=5)

    # print(step.a_tree.tree_structure)
    step.astep(1)

if __name__ == "__main__":
    main()