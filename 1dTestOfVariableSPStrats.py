# Sorry about all the indentation, my code editor's
# LSP freaks out if the lines go past 80 characters.
#
#
# TO DO:
# There is definitely opportunity for some object
# oriented programming here which could simplify the
# code. Could also be over engineering.
import numpy as np
import numpy.linalg as npl
import csv
import matplotlib.pyplot as plt


def generate_centers(
        num_centers, domain=(0, 1), random_seed=42
        ):
    # after a bunch of trial and error with choosing centers,
    # the internet offered this approached and it gave me the
    # best results in terms of condition numbers and error.
    # for all shape param strategies
    np.random.seed(random_seed)
    centers = np.sort(np.random.uniform(domain[0], domain[1], num_centers))
    return centers


def Exact_Function(x):
    # this was chosen from Derek Sturgill's Thesis
    # TO DO: try more functions
    return np.exp(np.sin(np.pi*x))


def MultiquadricRBF(distance_matrix, shape_param):
    # would be interesting to see if each strategy has
    # different levels of success depending on the kernel
    return np.sqrt(1+(distance_matrix*shape_param)**2)


# From Sturgill and Kansa/Sarra's paper
def LinearlyVaryingSP(N, min_e, max_e):
    indices = np.arange(0, N)
    shape_params = min_e + ((max_e - min_e)/N-1)*indices
    return shape_params


# Dr. Sarra's ! RANDOM SHAPE PARAMETER
def RSP(e_min, e_max, N, random_seed=42):
    np.random.seed(random_seed)
    shape_params = e_min + (e_max - e_min) * np.random.rand(N)
    return shape_params


# From Kansa's ESP
def ESP(N, min_e, max_e):
    indices = np.arange(1, N+1)
    shape_params = np.sqrt(
            min_e**2*(max_e**2/min_e**2)**((indices - 1)/(N-1))
            )
    return shape_params


# New Shape params paper Sinusoidal Shape Parameter
def SSP(e_min, e_max, N):
    indices = np.arange(1, N+1)
    shape_params = (
            e_min + (e_max - e_min)*np.sin(
                (indices-1)*(np.pi / (2*(N-1)))
                )
            )
    return shape_params


# linearly decreasing as opposed to increasing
def DLSP(e_max, e_min, N):
    indices = np.arange(1, N+1)
    shape_params = (
            e_max + ((e_min - e_max)/(N-1))*indices
            )
    return shape_params


# trigonometric shape param
def TSP(e_min, e_max, N):
    indices = np.arange(1, N+1)
    shape_params = (
            e_min + (e_max - e_min)*np.sin(indices)
            )
    return shape_params


# This one seems very clever. HYBRID SHAPE PARAMETER
def HSP(e_min, e_max, N):
    ssp = SSP(e_min, e_max, N)
    dlsp = DLSP(e_max, e_min, N)
    esp = ESP(N, e_min, e_max)

    shape_params = np.zeros(N)

    for j in range(1, N + 1):
        if j % 3 == 1:  # j = 3k + 1
            shape_params[j - 1] = ssp[j - 1]
        elif j % 3 == 2:  # j = 3k + 2
            shape_params[j - 1] = dlsp[j - 1]
        else:  # j = 3k + 3
            shape_params[j - 1] = esp[j - 1]

    return shape_params


# Binary Shape Parameter
def BSP(e_min, e_max, N):
    shape_params = np.zeros(N)

    for j in range(1, N+1):
        if j % 2 == 0:
            shape_params[j-1] = e_max
        else:  # j % 2 == 1:
            shape_params[j-1] = e_min
    return shape_params


# RBF stuff
def DistanceMatrix(points1, points2):
    # broadcasting !
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
    # since the original function for getting eval matrix
    # would not suffice for varying shape params
    d_matrix = DistanceMatrix(points1=eval_points, points2=centers)
    matrix = np.zeros_like(d_matrix)
    for i in range(len(shape_param)):
        matrix[:, i] = RBF(d_matrix[:, i], shape_param[i])

    return RBF(d_matrix, shape_param)


def RecordPointWiseError(
        interp1, interp2, interp3,
        interp4, interp5, interp6,
        interp7, interp8, interp9,
        eval_points, known_func
        ):
    # this stores the data in a csv. Honestly, I chose
    # csv file some what arbitrarly. Might be better to just do .txt
    with open("PointWiseErrors.csv", mode='w', newline="") as file:
        writer = csv.writer(file)
        writer.writerow([
            "x",
            "Constant Epsilon",
            "LSP",
            "ESP",
            "RSP",
            "TSP",
            "SSP",
            "DLSP",
            "HSP",
            "BSP"
            ])
        error_data = np.zeros((10, len(interp1)))

        for i, x in enumerate(eval_points):
            # while I could have just done nested loops, nested loops are
            # slow and just trying to keep the script efficient.
            error_data[0, i] = x
            error_data[1, i] = np.abs(interp1[i] - known_func[i])
            error_data[2, i] = np.abs(interp2[i] - known_func[i])
            error_data[3, i] = np.abs(interp3[i] - known_func[i])
            error_data[4, i] = np.abs(interp4[i] - known_func[i])
            error_data[5, i] = np.abs(interp5[i] - known_func[i])
            error_data[6, i] = np.abs(interp6[i] - known_func[i])
            error_data[7, i] = np.abs(interp7[i] - known_func[i])
            error_data[8, i] = np.abs(interp8[i] - known_func[i])
            error_data[9, i] = np.abs(interp9[i] - known_func[i])

            writer.writerow([
                f"{error_data[0, i]:1.5f}",
                f"{error_data[1, i]:1.5e}",
                f"{error_data[2, i]:1.5e}",
                f"{error_data[3, i]:1.5e}",
                f"{error_data[4, i]:1.5e}",
                f"{error_data[5, i]:1.5e}",
                f"{error_data[6, i]:1.5e}",
                f"{error_data[7, i]:1.5e}",
                f"{error_data[8, i]:1.5e}",
                f"{error_data[9, i]:1.5e}",
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


def StorePointWiseError(
        interp1, interp2, interp3,
        interp4, interp5, interp6,
        interp7, interp8, interp9,
        known_func
        ):
    # this just stores all the data in an array making it easier
    # to handle by putting it all in one place.
    error_data = np.zeros((9, len(interp1)))
    for i in range(len(interp1)):
        error_data[0, i] = np.abs(interp1[i] - known_func[i])
        error_data[1, i] = np.abs(interp2[i] - known_func[i])
        error_data[2, i] = np.abs(interp3[i] - known_func[i])
        error_data[3, i] = np.abs(interp4[i] - known_func[i])
        error_data[4, i] = np.abs(interp5[i] - known_func[i])
        error_data[5, i] = np.abs(interp6[i] - known_func[i])
        error_data[6, i] = np.abs(interp7[i] - known_func[i])
        error_data[7, i] = np.abs(interp8[i] - known_func[i])
        error_data[8, i] = np.abs(interp9[i] - known_func[i])
    return error_data


def main() -> None:

    CENTERS = generate_centers(200)
    EVAL_POINTS = np.linspace(0, 1, 250, endpoint=True)
    N = len(CENTERS)

    func_at_centers = Exact_Function(CENTERS)
    func_at_evals = Exact_Function(EVAL_POINTS)

    # our different shape parameter strategies
    #
    # The e_min's and e_max's were determined such that
    # the condition number of the system matrices were
    # as similar as possible
    #
    # I feel like there is probably a better way to
    # do this.
    CON_SHAPE = 2.4
    lv_shape = LinearlyVaryingSP(N=N, min_e=1.8, max_e=7.6)
    ev_shape = ESP(N=N, min_e=2.2, max_e=7.2)
    rand_shape = RSP(N=N, e_min=2.1, e_max=8.5)
    sin_shape = SSP(N=N, e_min=3.1, e_max=6.5)
    tsp_shape = TSP(N=N, e_min=2.2, e_max=7.2)
    dlsp_shape = DLSP(N=N, e_min=2.2, e_max=7.2)
    hsp_shape = HSP(N=N, e_min=2.2, e_max=7.2)
    bsp_shape = BSP(N=N, e_min=2.2, e_max=7.2)

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
    tsp_interp = Interpolate(
            centers=CENTERS,
            eval_points=EVAL_POINTS,
            RBF=MultiquadricRBF,
            shape_param=tsp_shape,
            known_func=func_at_centers,
            varying_shape=True,
            )
    ssp_interp = Interpolate(
            centers=CENTERS,
            eval_points=EVAL_POINTS,
            RBF=MultiquadricRBF,
            shape_param=sin_shape,
            known_func=func_at_centers,
            varying_shape=True,
            )
    dlsp_interp = Interpolate(
            centers=CENTERS,
            eval_points=EVAL_POINTS,
            RBF=MultiquadricRBF,
            shape_param=dlsp_shape,
            known_func=func_at_centers,
            varying_shape=True,
            )
    hsp_interp = Interpolate(
            centers=CENTERS,
            eval_points=EVAL_POINTS,
            RBF=MultiquadricRBF,
            shape_param=hsp_shape,
            known_func=func_at_centers,
            varying_shape=True,
            )
    bsp_interp = Interpolate(
            centers=CENTERS,
            eval_points=EVAL_POINTS,
            RBF=MultiquadricRBF,
            shape_param=bsp_shape,
            known_func=func_at_centers,
            varying_shape=True,
            )

    error_data = StorePointWiseError(
            interp1=con_interp,
            interp2=lv_interp,
            interp3=ev_interp,
            interp4=rand_interp,
            interp5=tsp_interp,
            interp6=ssp_interp,
            interp7=dlsp_interp,
            interp8=hsp_interp,
            interp9=bsp_interp,
            known_func=func_at_evals
            )

    RecordPointWiseError(
            con_interp,
            lv_interp,
            ev_interp,
            rand_interp,
            tsp_interp,
            ssp_interp,
            dlsp_interp,
            hsp_interp,
            bsp_interp,
            EVAL_POINTS,
            func_at_evals
            )

    kappa_cons = npl.cond(SysMatrix(CENTERS, MultiquadricRBF, CON_SHAPE))
    kappa_lv = npl.cond(VaryShapeSysMatrix(CENTERS, MultiquadricRBF, lv_shape))
    kappa_ec = npl.cond(VaryShapeSysMatrix(CENTERS, MultiquadricRBF, ev_shape))
    kappa_rand = npl.cond(
            VaryShapeSysMatrix(CENTERS, MultiquadricRBF, rand_shape)
            )
    kappa_sin = npl.cond(
            VaryShapeSysMatrix(CENTERS, MultiquadricRBF, sin_shape)
            )
    kappa_tsp = npl.cond(
            VaryShapeSysMatrix(CENTERS, MultiquadricRBF, tsp_shape)
            )
    kappa_dlsp = npl.cond(
            VaryShapeSysMatrix(CENTERS, MultiquadricRBF, dlsp_shape)
            )
    kappa_hsp = npl.cond(
            VaryShapeSysMatrix(CENTERS, MultiquadricRBF, hsp_shape)
            )
    kappa_bsp = npl.cond(
            VaryShapeSysMatrix(CENTERS, MultiquadricRBF, bsp_shape)
            )

    max_error_cons = npl.norm(con_interp-func_at_evals, np.inf)
    max_error_lv = npl.norm(lv_interp-func_at_evals, np.inf)
    max_error_ev = npl.norm(ev_interp-func_at_evals, np.inf)
    max_error_rand = npl.norm(rand_interp-func_at_evals, np.inf)
    max_error_ssp = npl.norm(ssp_interp-func_at_evals, np.inf)
    max_error_tsp = npl.norm(tsp_interp-func_at_evals, np.inf)
    max_error_dlsp = npl.norm(dlsp_interp-func_at_evals, np.inf)
    max_error_hsp = npl.norm(hsp_interp-func_at_evals, np.inf)
    max_error_bsp = npl.norm(bsp_interp-func_at_evals, np.inf)

    print(f"Constant Shape = {CON_SHAPE}: Kappa = {kappa_cons:1.5e}" +
          f"| Max Error = {max_error_cons:1.5e}")
    print(f"Linearly Varying Shape: Kappa = {kappa_lv:1.5e}" +
          f"| Max Error = {max_error_lv:1.5e}")
    print(f"Exponentially Varying Shape: Kappa = {kappa_ec:1.5e}" +
          f"| Max Error = {max_error_ev:1.5e}")
    print(f"Random Shape: Kappa = {kappa_rand:1.5e}" +
          f"| Max Error = {max_error_rand:1.5e}")
    print(f"Sin Shape: Kappa = {kappa_sin:1.5e}" +
          f"| Max Error = {max_error_ssp:1.5e}")
    print(f"TSP: Kappa = {kappa_tsp:1.5e}" +
          f"| Max Error = {max_error_tsp:1.5e}")
    print(f"DLSP Shape: Kappa = {kappa_dlsp:1.5e}" +
          f"| Max Error = {max_error_dlsp:1.5e}")
    print(f"HSP Shape: Kappa = {kappa_hsp:1.5e}" +
          f"| Max Error = {max_error_hsp:1.5e}")
    print(f"BSP Shape: Kappa = {kappa_bsp:1.5e}" +
          f"| Max Error = {max_error_bsp:1.5e}")

    # plots of pointwise errors
    # TO DO: put the plotting into a separate file
    # and read the csv data into it to plot.
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
    plt.semilogy(
            EVAL_POINTS,
            error_data[4, :],
            'y',
            label='sin $\\epsilon$'
            )

    plt.title('Point Wise Error For Each Shape Parameter Strategy')
    plt.grid(True)
    plt.xlabel("x")
    plt.ylabel("Error")
    plt.legend()
    plt.show()

    return None


main()
