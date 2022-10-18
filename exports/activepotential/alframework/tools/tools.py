import numpy as np
import os

def compute_empirical_formula(S):
    uniques = np.unique(S,return_counts=True)
    arg_sort = np.argsort(uniques[0])
    return "_".join([i+str(j).zfill(2) for i,j in zip(uniques[0][arg_sort],uniques[1][arg_sort])])

def random_rotation_matrix(deflection=1.0, randnums=None):
    if randnums is None:
        randnums = np.random.uniform(size=(3,))

    theta, phi, z = randnums

    theta = theta * 2.0 * deflection * np.pi  # Rotation about the pole (Z).
    phi = phi * 2.0 * np.pi  # For direction of pole deflection.
    z = z * 2.0 * deflection  # For magnitude of pole deflection.

    r = np.sqrt(z)
    Vx, Vy, Vz = V = (
        np.sin(phi) * r,
        np.cos(phi) * r,
        np.sqrt(2.0 - z)
    )

    st = np.sin(theta)
    ct = np.cos(theta)

    R = np.array(((ct, st, 0), (-st, ct, 0), (0, 0, 1)))

    M = (np.outer(V, V) - np.eye(3)).dot(R)
    return M

def sae_linear_fitting(datadir, sae_out, elements, Ekey='energy', energy_unit=1.0):
    from sklearn import linear_model
    import pyanitools as pyt

    smap = dict()
    for i, Z in enumerate(elements):
        smap.update({Z: i})

    Na = len(smap)
    files = os.listdir(datadir)

    X = []
    y = []
    for f in files:
        print(f)
        adl = pyt.anidataloader(datadir + f)
        for data in adl:
            # print(data['path'])
            S = data['species']

            if data[Ekey].size > 0:
                E = energy_unit * np.array(data[Ekey], order='C', dtype=np.float64)
                S = S[0:data['coordinates'].shape[1]]
                unique, counts = np.unique(S, return_counts=True)
                x = np.zeros(Na, dtype=np.float64)
                for u, c in zip(unique, counts):
                    x[smap[u]] = c

                for e in E:
                    X.append(np.array(x))
                    y.append(np.array(e))

    X = np.array(X)
    y = np.array(y).reshape(-1, 1)

    lin = linear_model.LinearRegression(fit_intercept=False)
    lin.fit(X, y)

    coef = lin.coef_
    print(coef)

    sae = open(sae_out, 'w')
    for i, c in enumerate(coef[0]):
        sae.write(next(key for key, value in smap.items() if value == i) + ',' + str(i) + '=' + str(c) + '\n')
    sae.close()
