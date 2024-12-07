# f(x) = x^3 + 3x^2 +12x+6
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
from rbfCenters import R1Points


def generate_centers(
        num_centers, domain=(0, 1), random_seed=42
        ):
    # after a bunch of trial and error with choosing centers,
    # the internet offered this approached and it gave me the
    # best results in terms of condition numbers and error.
    # for all center choosing strategies that I've tried
    # TO DO: switch to the center strategy that reduces the
    # error near bounds
    # or.... for the sake of science it might be better to
    # just do evenly distributed centers.
    np.random.seed(random_seed)
    centers = np.sort(np.random.uniform(domain[0], domain[1], num_centers))
    return centers


def Exact_Function(x):
    return x**3 + 3*(x**2) + 12*x + 6


def MultiquadricRBF(distance_matrix, shape_param):
    # would be interesting to see if each strategy has
    # different levels of success depending on the kernel
    return np.sqrt(1+(distance_matrix*shape_param)**2)


# From Sturgill's and Kansa/Sarra's paper
def LSP(N, min_e, max_e):
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
# works suprisingly well considering how bad the
# LSP is.
def DLSP(e_max, e_min, N):
    indices = np.arange(1, N+1)
    shape_params = (
            e_max + ((e_min - e_max)/(N-1))*indices
            )
    return shape_params


# trigonometric shape param.
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


def RecordData(interps, func_at_evals, sys_matrices, error_data):
    kappa = np.zeros(len(sys_matrices))
    norms = np.zeros(len(interps))
    avg_err = np.zeros((len(interps)))
    for i in range(len(sys_matrices)):
        kappa[i] = npl.cond(sys_matrices[i])
        norms[i] = npl.norm(interps[i] - func_at_evals, np.inf)
        avg_err[i] = np.average(error_data[i, :])

    with open("Test2/Test2-NormsConds.csv", mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Shape Parameter', "max error", "condition number",
                         "average error"])
        writer.writerow([
            'CSP', f"{norms[0]:1.5e}", f"{kappa[0]:1.5e}", f"{avg_err[0]:1.5e}"
            ])
        writer.writerow([
            'LSP', f"{norms[1]:1.5e}", f"{kappa[1]:1.5e}", f"{avg_err[1]:1.5e}"
            ])
        writer.writerow([
            'ESP', f"{norms[2]:1.5e}", f"{kappa[2]:1.5e}", f"{avg_err[2]:1.5e}"
            ])
        writer.writerow([
            'RSP', f"{norms[3]:1.5e}", f"{kappa[3]:1.5e}", f"{avg_err[3]:1.5e}"
            ])
        writer.writerow([
            'TSP', f"{norms[4]:1.5e}", f"{kappa[4]:1.5e}", f"{avg_err[4]:1.5e}"
            ])
        writer.writerow([
            'SSP', f"{norms[5]:1.5e}", f"{kappa[5]:1.5e}", f"{avg_err[5]:1.5e}"
            ])
        writer.writerow([
            'DLSP', f"{norms[6]:1.5e}", f"{kappa[6]:1.5e}",
            f"{avg_err[6]:1.5e}"
            ])
        writer.writerow([
            'HSP', f"{norms[7]:1.5e}", f"{kappa[7]:1.5e}", f"{avg_err[7]:1.5e}"
            ])
        writer.writerow([
            'BSP', f"{norms[8]:1.5e}", f"{kappa[8]:1.5e}", f"{avg_err[8]:1.5e}"
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
    with open("Test2/PointWiseErrors-Test2.csv", mode='w', newline="") as file:
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

    # Dr. Sarra's RBF TOOL BOX
    CENTERS = R1Points(
            N=200, A=-0.005, B=1.005, a0=0
            )
    EVAL_POINTS = np.linspace(0, 1, 250, endpoint=True)
    N = len(CENTERS)

    func_at_centers = Exact_Function(CENTERS)
    func_at_evals = Exact_Function(EVAL_POINTS)

    # our different shape parameter strategies
    #
    # The e_min's and e_max's were determined such that
    # the condition number of the system matrices were
    # as similar as possible
    # TO DO: get the condition numbers closer
    CON_SHAPE = 2.4
    lv_shape = LSP(N=N, min_e=3.7, max_e=8.31)
    ev_shape = ESP(N=N, min_e=2.2, max_e=7.2)
    rand_shape = RSP(N=N, e_min=2.75, e_max=8.82)
    sin_shape = SSP(N=N, e_min=3.1, e_max=6.5)
    tsp_shape = TSP(N=N, e_min=2.1, e_max=5.6955)
    dlsp_shape = DLSP(N=N, e_min=2.2, e_max=6.2)
    hsp_shape = HSP(N=N, e_min=2.58, e_max=6.2)
    bsp_shape = BSP(N=N, e_min=2.3, e_max=6.2)

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

    RecordData(interps, func_at_evals, sys_matrices, error_data)
    return None


main()
