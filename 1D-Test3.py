# Test3: 1 / (x**2 + 1)
#
# TO DO:
# There is definitely opportunity for some object
# oriented programming here which could simplify the
# code. Could also be over engineering.
import numpy as np
import numpy.linalg as npl
import csv
from rbfCenters import R1Points
from scipy.optimize import dual_annealing


def Exact_Function(x):
    return 1 / (x**2 + 1)


def MultiquadricRBF(distance_matrix, shape_param):
    return np.sqrt(1 + (distance_matrix * shape_param) ** 2)


def objective(params, centers, RBF, strategy_func, N, target_cond):
    e_min, e_max = params
    if e_min >= e_max:
        return 1e20

    shape_params = strategy_func(N, e_min, e_max)
    sys_matrix = VaryShapeSysMatrix(centers, RBF, shape_params)
    cond_num = npl.cond(sys_matrix)
    return abs(cond_num - target_cond)


# this function optimizes the e_min, e_max for each variable
# sp strategy. Very Very effective, but also computationally intesive
# on the cpu and easily the greatest contributer to the runtime complexity.
def tune_shape_params(target_cond, centers, RBF, strategy_func, N):
    bounds = [(0.5, 10.0), (1.5, 20.0)]
    result = dual_annealing(
        objective,
        bounds=bounds,
        args=(centers, RBF, strategy_func, N, target_cond),
        maxiter=250,
        initial_temp=5230.0,
        visit=2.62,
        accept=-5.0,
    )

    best_e_min, best_e_max = result.x
    return best_e_min, best_e_max


# From Sturgill's and Kansa/Sarra's paper
def LSP(N, min_e, max_e):
    indices = np.arange(0, N)
    shape_params = min_e + ((max_e - min_e)/(N-1))*indices
    return shape_params


# Dr. Sarra's ! RANDOM SHAPE PARAMETER
def RSP(N, e_min, e_max):
    np.random.seed(42)
    shape_params = e_min + (e_max - e_min) * np.random.rand(N)
    return shape_params


# From Kansa's ESP
def ESP(N, min_e, max_e):
    indices = np.arange(1, N+1)
    shape_params = np.sqrt(
            min_e**2*(max_e**2/min_e**2)**((indices - 1)/(N-1))
            )
    return shape_params


# Sinusoidal Shape Parameter
# NOTE: not the same as TSP
def SSP(N, e_min, e_max):
    N = int(N)
    indices = np.arange(1, N+1)
    shape_params = (
            e_min + (e_max - e_min)*np.sin(
                (indices-1)*(np.pi / (2*(N-1)))
                )
            )
    return shape_params


# LINEARLY DECREASING SHAPE PARAM
def DLSP(N, e_min, e_max):
    N = int(N)
    indices = np.arange(1, N+1)
    shape_params = (
            e_max + ((e_min - e_max)/(N-1))*indices
            )
    return shape_params


# trigonometric shape param.
# this one shows a lot of success and look very similar to
# Dr. Sarra's RSP.
def TSP(N, e_min, e_max):
    N = int(N)
    indices = np.arange(1, N+1)
    shape_params = (
            e_min + (e_max - e_min)*np.sin(indices)
            )
    return shape_params


# This one seems very clever. HYBRID SHAPE PARAMETER
# Starting to think its a bit over engineered.
def HSP(N, e_min, e_max):
    N = int(N)
    ssp = SSP(N, e_min, e_max)
    dlsp = DLSP(N, e_max, e_min)
    esp = ESP(N, e_min, e_max)

    shape_params = np.zeros(N)

    for j in range(1, N + 1):
        if j % 3 == 1:  # j = 3k + 1
            shape_params[j-1] = ssp[j-1]
        elif j % 3 == 2:  # j = 3k + 2
            shape_params[j-1] = dlsp[j-1]
        else:  # j = 3k + 3
            shape_params[j-1] = esp[j-1]

    return shape_params


# Binary Shape Parameter
def BSP(N, e_min, e_max):
    N = int(N)
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
    matrix = np.abs(diff)
    return matrix


def SysMatrix(centers, RBF, shape_param):
    d_matrix = DistanceMatrix(centers, centers)
    return RBF(d_matrix, shape_param)


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

    return matrix


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

        return eval_matrix@exp_coeffs, sys_matrix

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

        return eval_matrix@exp_coeffs, sys_matrix


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


def RecordData(
        interps, func_at_evals, sys_matrices, error_data, e_mins, e_maxs
        ):
    kappa = np.zeros(len(sys_matrices))
    norms = np.zeros(len(interps))
    avg_err = np.zeros((len(interps)))
    for i in range(len(sys_matrices)):
        kappa[i] = npl.cond(sys_matrices[i])
        norms[i] = npl.norm(interps[i] - func_at_evals, np.inf)
        avg_err[i] = np.average(error_data[i, :])

    with open("Test3/Test3-NormsConds.csv", mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Shape Parameter', "max error", "condition number",
                         "average error", "e_min", "e_max"])
        writer.writerow([
            'CSP', f"{norms[0]:1.5e}", f"{kappa[0]:1.5e}",
            f"{avg_err[0]:1.5e}", f"{e_mins[0]:1.5f}", f"{e_maxs[0]:1.5f}"
            ])
        writer.writerow([
            'LSP', f"{norms[1]:1.5e}", f"{kappa[1]:1.5e}",
            f"{avg_err[1]:1.5e}", f"{e_mins[1]:1.5f}", f"{e_maxs[1]:1.5f}"
            ])
        writer.writerow([
            'ESP', f"{norms[2]:1.5e}", f"{kappa[2]:1.5e}",
            f"{avg_err[2]:1.5e}", f"{e_mins[2]:1.5f}", f"{e_maxs[2]:1.5f}"
            ])
        writer.writerow([
            'RSP', f"{norms[3]:1.5e}", f"{kappa[3]:1.5e}",
            f"{avg_err[3]:1.5e}", f"{e_mins[3]:1.5f}", f"{e_maxs[3]:1.5f}"
            ])
        writer.writerow([
            'TSP', f"{norms[4]:1.5e}", f"{kappa[4]:1.5e}",
            f"{avg_err[4]:1.5e}", f"{e_mins[4]:1.5f}", f"{e_maxs[4]:1.5f}"
            ])
        writer.writerow([
            'SSP', f"{norms[5]:1.5e}", f"{kappa[5]:1.5e}",
            f"{avg_err[5]:1.5e}", f"{e_mins[5]:1.5f}", f"{e_maxs[5]:1.5f}"
            ])
        writer.writerow([
            'DLSP', f"{norms[6]:1.5e}", f"{kappa[6]:1.5e}",
            f"{avg_err[6]:1.5e}", f"{e_mins[6]:1.5f}", f"{e_maxs[6]:1.5f}"
            ])
        writer.writerow([
            'HSP', f"{norms[7]:1.5e}", f"{kappa[7]:1.5e}",
            f"{avg_err[7]:1.5e}", f"{e_mins[7]:1.5f}", f"{e_maxs[7]:1.5f}"
            ])
        writer.writerow([
            'BSP', f"{norms[8]:1.5e}", f"{kappa[8]:1.5e}",
            f"{avg_err[8]:1.5e}", f"{e_mins[8]:1.5f}", f"{e_maxs[8]:1.5f}"
            ])

    return None


def RecordPointWiseError(
        interp1, interp2, interp3,
        interp4, interp5, interp6,
        interp7, interp8, interp9,
        eval_points, known_func
        ):
    # this stores the data in a csv. Honestly, I chose
    # csv file some what arbitrarly. Might be better to just do .txt
    with open("Test3/PointWiseErrors-Test3.csv", mode='w', newline="") as file:
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
            # and honestly, this helps me keep everything in order
            # by being able to see it.
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


def main() -> None:

    # Dr.Sarra's RBF TOOL BOX
    CENTERS = R1Points(
            N=200, A=-0.005, B=1.005, a0=0
            )
    EVAL_POINTS = np.linspace(0, 1, 250)
    N = int(len(CENTERS))
    CON_SHAPE = 3

    func_at_centers = Exact_Function(CENTERS)
    func_at_evals = Exact_Function(EVAL_POINTS)

    target_cond = npl.cond(SysMatrix(CENTERS, MultiquadricRBF, CON_SHAPE))
    lsp_min_e, lsp_max_e = tune_shape_params(
            target_cond=target_cond,
            centers=CENTERS,
            RBF=MultiquadricRBF,
            strategy_func=LSP,
            N=N,
            )
    esp_min_e, esp_max_e = tune_shape_params(
            target_cond=target_cond,
            centers=CENTERS,
            RBF=MultiquadricRBF,
            strategy_func=ESP,
            N=N,
            )
    rsp_min_e, rsp_max_e = tune_shape_params(
            target_cond=target_cond,
            centers=CENTERS,
            RBF=MultiquadricRBF,
            strategy_func=RSP,
            N=N,
            )
    tsp_min_e, tsp_max_e = tune_shape_params(
            target_cond=target_cond,
            centers=CENTERS,
            RBF=MultiquadricRBF,
            strategy_func=TSP,
            N=N,
            )
    ssp_min_e, ssp_max_e = tune_shape_params(
            target_cond=target_cond,
            centers=CENTERS,
            RBF=MultiquadricRBF,
            strategy_func=SSP,
            N=N,
            )
    dlsp_min_e, dlsp_max_e = tune_shape_params(
            target_cond=target_cond,
            centers=CENTERS,
            RBF=MultiquadricRBF,
            strategy_func=DLSP,
            N=N,
            )
    hsp_min_e, hsp_max_e = tune_shape_params(
            target_cond=target_cond,
            centers=CENTERS,
            RBF=MultiquadricRBF,
            strategy_func=HSP,
            N=N,
            )
    bsp_min_e, bsp_max_e = tune_shape_params(
            target_cond=target_cond,
            centers=CENTERS,
            RBF=MultiquadricRBF,
            strategy_func=BSP,
            N=N,
            )
    e_mins = [
            CON_SHAPE,
            lsp_min_e,
            esp_min_e,
            rsp_min_e,
            tsp_min_e,
            ssp_min_e,
            dlsp_min_e,
            hsp_min_e,
            bsp_min_e
            ]

    e_maxs = [
            3.0,
            lsp_max_e,
            esp_max_e,
            rsp_max_e,
            tsp_max_e,
            ssp_max_e,
            dlsp_max_e,
            hsp_max_e,
            bsp_max_e
            ]

    lv_shape = LSP(N=N, min_e=lsp_min_e, max_e=lsp_max_e)
    ev_shape = ESP(N=N, min_e=esp_min_e, max_e=esp_max_e)
    rand_shape = RSP(N=N, e_min=rsp_min_e, e_max=rsp_max_e)
    sin_shape = SSP(N=N, e_min=ssp_min_e, e_max=ssp_max_e)
    tsp_shape = TSP(N=N, e_min=tsp_min_e, e_max=tsp_max_e)
    dlsp_shape = DLSP(N=N, e_min=dlsp_min_e, e_max=dlsp_max_e)
    hsp_shape = HSP(N=N, e_min=hsp_min_e, e_max=hsp_max_e)
    bsp_shape = BSP(N=N, e_min=bsp_min_e, e_max=bsp_max_e)

    csp_interp, csp_sys = Interpolate(
            centers=CENTERS,
            eval_points=EVAL_POINTS,
            RBF=MultiquadricRBF,
            shape_param=CON_SHAPE,
            known_func=func_at_centers,
            varying_shape=False
            )
    lsp_interp, lsp_sys = Interpolate(
            centers=CENTERS,
            eval_points=EVAL_POINTS,
            RBF=MultiquadricRBF,
            shape_param=lv_shape,
            known_func=func_at_centers,
            varying_shape=True
            )
    esp_interp, esp_sys = Interpolate(
            centers=CENTERS,
            eval_points=EVAL_POINTS,
            RBF=MultiquadricRBF,
            shape_param=ev_shape,
            known_func=func_at_centers,
            varying_shape=True
            )
    rsp_interp, rsp_sys = Interpolate(
            centers=CENTERS,
            eval_points=EVAL_POINTS,
            RBF=MultiquadricRBF,
            shape_param=rand_shape,
            known_func=func_at_centers,
            varying_shape=True
            )
    tsp_interp, tsp_sys = Interpolate(
            centers=CENTERS,
            eval_points=EVAL_POINTS,
            RBF=MultiquadricRBF,
            shape_param=tsp_shape,
            known_func=func_at_centers,
            varying_shape=True,
            )
    ssp_interp, ssp_sys = Interpolate(
            centers=CENTERS,
            eval_points=EVAL_POINTS,
            RBF=MultiquadricRBF,
            shape_param=sin_shape,
            known_func=func_at_centers,
            varying_shape=True,
            )
    dlsp_interp, dlsp_sys = Interpolate(
            centers=CENTERS,
            eval_points=EVAL_POINTS,
            RBF=MultiquadricRBF,
            shape_param=dlsp_shape,
            known_func=func_at_centers,
            varying_shape=True,
            )
    hsp_interp, hsp_sys = Interpolate(
            centers=CENTERS,
            eval_points=EVAL_POINTS,
            RBF=MultiquadricRBF,
            shape_param=hsp_shape,
            known_func=func_at_centers,
            varying_shape=True,
            )
    bsp_interp, bsp_sys = Interpolate(
            centers=CENTERS,
            eval_points=EVAL_POINTS,
            RBF=MultiquadricRBF,
            shape_param=bsp_shape,
            known_func=func_at_centers,
            varying_shape=True,
            )

    interps = [
            csp_interp,
            lsp_interp,
            esp_interp,
            rsp_interp,
            tsp_interp,
            ssp_interp,
            dlsp_interp,
            hsp_interp,
            bsp_interp,
            ]

    sys_matrices = [
            csp_sys,
            lsp_sys,
            esp_sys,
            rsp_sys,
            tsp_sys,
            ssp_sys,
            dlsp_sys,
            hsp_sys,
            bsp_sys
            ]

    error_data = StorePointWiseError(
            interp1=csp_interp,
            interp2=lsp_interp,
            interp3=esp_interp,
            interp4=rsp_interp,
            interp5=tsp_interp,
            interp6=ssp_interp,
            interp7=dlsp_interp,
            interp8=hsp_interp,
            interp9=bsp_interp,
            known_func=func_at_evals,
            )

    RecordPointWiseError(
            csp_interp,
            lsp_interp,
            esp_interp,
            rsp_interp,
            tsp_interp,
            ssp_interp,
            dlsp_interp,
            hsp_interp,
            bsp_interp,
            EVAL_POINTS,
            func_at_evals
            )

    RecordData(
            interps, func_at_evals, sys_matrices, error_data, e_mins, e_maxs
            )
    return None


main()
