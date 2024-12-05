import numpy as np
import numpy.linalg as npl
import csv
import matplotlib.pyplot as plt


def generate_centers(
        num_centers, domain=(0, 1), random_seed=42
        ):
    np.random.seed(random_seed)
    centers = np.sort(np.random.uniform(domain[0], domain[1], num_centers))
    return centers


def Exact_Function(x):
    return np.exp(np.sin(np.pi*x))


def MultiquadricRBF(distance_matrix, shape_param):
    return np.sqrt(1+(distance_matrix*shape_param)**2)


def LinearlyVaryingSP(N, min_e, max_e):
    indices = np.arange(0, N)
    shape_params = min_e + ((max_e - min_e)/N-1)*indices
    return shape_params


def ExponentiallyVaryingEpsilon(N, min_e, max_e):
    indices = np.arange(1, N+1)
    shape_params = np.sqrt(
            min_e**2*(max_e**2/min_e**2)**((indices - 1)/(N-1))
            )
    return shape_params


def RandomShapeParam(e_min, e_max, N, random_seed=42):
    np.random.seed(random_seed)
    shape_params = e_min + (e_max - e_min) * np.random.rand(N)
    return shape_params


def DistanceMatrix(points1, points2):
    # broadcasting
    diff = points1[:, np.newaxis] - points2
    matrix = abs(diff)
    return matrix


def SysMatrix(centers, RBF, shape_param):
    d_matrix = DistanceMatrix(points1=centers, points2=centers)
    s_matrix = RBF(d_matrix, shape_param)
    return s_matrix


def VaryShapeSysMatrix(centers, RBF, shape_param):
    d_matrix = DistanceMatrix(points1=centers, points2=centers)

    matrix = np.zeros_like(d_matrix)
    for i in range(len(shape_param)):
        matrix[:, i] = RBF(d_matrix[:, i], shape_param[i])

    return matrix


def EvalMatrix(centers, eval_points, RBF, shape_param):
    d_matrix = DistanceMatrix(points1=eval_points, points2=centers)
    return RBF(d_matrix, shape_param)


def VaryShapeEvalMatrix(centers, eval_points, RBF, shape_param):
    d_matrix = DistanceMatrix(points1=eval_points, points2=centers)
    matrix = np.zeros_like(d_matrix)
    for i in range(len(shape_param)):
        matrix[:, i] = RBF(d_matrix[:, i], shape_param[i])

    return RBF(d_matrix, shape_param)


def RecordPointWiseError(interp1, interp2, interp3, interp4, eval_points, known_func):
    with open("PointWiseErrors.csv", mode='w', newline="") as file:
        writer = csv.writer(file)
        writer.writerow([
            "x",
            "Constant Epsilon",
            "Linearly Varying Epsilon",
            "Exponentially Varying Epsilon",
            "Random Varying Epsilon"
            ])
        error_data = np.zeros((5, len(interp1)))

        for i, x in enumerate(eval_points):
            error_data[0, i] = x
            error_data[1, i] = np.abs(interp1[i] - known_func[i])
            error_data[2, i] = np.abs(interp2[i] - known_func[i])
            error_data[3, i] = np.abs(interp3[i] - known_func[i])
            error_data[4, i] = np.abs(interp4[i] - known_func[i])

            writer.writerow([
                f"{error_data[0, i]:1.5f}",
                f"{error_data[1, i]:1.5e}",
                f"{error_data[2, i]:1.5e}",
                f"{error_data[3, i]:1.5e}",
                f"{error_data[4, i]:1.5e}"
                ])

        return None


def Interpolate(
        centers, eval_points, RBF, shape_param, known_func, varying_shape: bool
        ):
    if varying_shape is True:
        sys_matrix = VaryShapeSysMatrix(
                centers=centers, RBF=RBF, shape_param=shape_param
                )
        exp_coeffs = npl.solve(sys_matrix, known_func)

        eval_matrix = VaryShapeEvalMatrix(
                centers=centers,
                eval_points=eval_points,
                RBF=RBF,
                shape_param=shape_param
                )

        return eval_matrix@exp_coeffs

    else:
        sys_matrix = SysMatrix(
                centers=centers, RBF=RBF, shape_param=shape_param
                )
        exp_coeffs = npl.solve(sys_matrix, known_func)

        eval_matrix = EvalMatrix(
                centers=centers,
                eval_points=eval_points,
                RBF=RBF,
                shape_param=shape_param
                )

        return eval_matrix@exp_coeffs


def StorePointWiseError(interp1, interp2, interp3, interp4, known_func):

    error_data = np.zeros((4, len(interp1)))
    for i in range(len(interp1)):
        error_data[0, i] = np.abs(interp1[i] - known_func[i])
        error_data[1, i] = np.abs(interp2[i] - known_func[i])
        error_data[2, i] = np.abs(interp3[i] - known_func[i])
        error_data[3, i] = np.abs(interp4[i] - known_func[i])

    return error_data


def main() -> None:

    CENTERS = generate_centers(200)
    EVAL_POINTS = np.linspace(0, 1, 250, endpoint=True)
    N = len(CENTERS)

    func_at_centers = Exact_Function(CENTERS)
    func_at_evals = Exact_Function(EVAL_POINTS)
    # our different shape parameter strategies
    lv_shape = LinearlyVaryingSP(N=N, min_e=1.8, max_e=7.6)
    ev_shape = ExponentiallyVaryingEpsilon(N=N, min_e=2.2, max_e=7.2)
    CON_SHAPE = 2.4
    rand_shape = RandomShapeParam(N=N, e_min=2.1, e_max=7.6)

    con_interp = Interpolate(
            centers=CENTERS,
            eval_points=EVAL_POINTS,
            RBF=MultiquadricRBF,
            shape_param=CON_SHAPE,
            known_func=func_at_centers,
            varying_shape=False
            )

    lv_interp = Interpolate(
            centers=CENTERS,
            eval_points=EVAL_POINTS,
            RBF=MultiquadricRBF,
            shape_param=lv_shape,
            known_func=func_at_centers,
            varying_shape=True
            )

    ev_interp = Interpolate(
            centers=CENTERS,
            eval_points=EVAL_POINTS,
            RBF=MultiquadricRBF,
            shape_param=ev_shape,
            known_func=func_at_centers,
            varying_shape=True
            )

    rand_interp = Interpolate(
            centers=CENTERS,
            eval_points=EVAL_POINTS,
            RBF=MultiquadricRBF,
            shape_param=rand_shape,
            known_func=func_at_centers,
            varying_shape=True
            )

    error_data = StorePointWiseError(
            interp1=con_interp,
            interp2=lv_interp,
            interp3=ev_interp,
            interp4=rand_interp,
            known_func=func_at_evals
            )

    RecordPointWiseError(
            con_interp,
            lv_interp,
            ev_interp,
            rand_interp,
            EVAL_POINTS,
            func_at_evals
            )

    kappa_cons = npl.cond(SysMatrix(CENTERS, MultiquadricRBF, CON_SHAPE))
    kappa_lv = npl.cond(VaryShapeSysMatrix(CENTERS, MultiquadricRBF, lv_shape))
    kappa_ec = npl.cond(VaryShapeSysMatrix(CENTERS, MultiquadricRBF, ev_shape))
    kappa_rand = npl.cond(
            VaryShapeSysMatrix(CENTERS, MultiquadricRBF, rand_shape)
            )

    max_error_cons = npl.norm(con_interp-func_at_evals, np.inf)
    max_error_lv = npl.norm(lv_interp-func_at_evals, np.inf)
    max_error_ev = npl.norm(ev_interp-func_at_evals, np.inf)
    max_error_rand = npl.norm(rand_interp-func_at_evals, np.inf)

    print(f"Constant Shape = {CON_SHAPE}: Kappa = {kappa_cons:1.5e}" +
          f"| Max Error = {max_error_cons:1.5e}")
    print(f"Linearly Varying Shape: Kappa = {kappa_lv:1.5e}" +
          f"| Max Error = {max_error_lv:1.5e}")
    print(f"Exponentially Varying Shape: Kappa = {kappa_ec:1.5e}" +
          f"| Max Error = {max_error_ev:1.5e}")
    print(f"Random Shape: Kappa = {kappa_rand:1.5e}" +
          f"| Max Error = {max_error_rand:1.5e}")

    # plots of pointwise errors
    plt.semilogy(
            EVAL_POINTS,
            error_data[0, :],
            'r',
            label=f"$\\epsilon =${CON_SHAPE}"
            )
    plt.semilogy(
            EVAL_POINTS,
            error_data[1, :],
            'g',
            label='Linearly Varying $\\epsilon$'
            )
    plt.semilogy(
            EVAL_POINTS,
            error_data[2, :],
            'b',
            label='Exponentially Varying $\\epsilon$'
            )
    plt.semilogy(
            EVAL_POINTS,
            error_data[3, :],
            'm',
            label='Random $\\epsilon$'
            )
    plt.title('Point Wise Error For Each Shape Parameter Strategy')
    plt.grid(True)
    plt.xlabel("x")
    plt.ylabel("Error")
    plt.legend()
    plt.show()

    return None


main()
