import numpy as np
import numpy.linalg as npl
import csv
import matplotlib.pyplot as plt


class RBFs:
    def __init__(self):
        return self

    def Multiquadric(distance_matrix, shape_param):
        return np.sqrt(1+(distance_matrix*shape_param)**2)

    def InverseMQ(distance_matrix, shape_param):
        return 1 / (np.sqrt(1+(distance_matrix*shape_param)**2))

    def Guass(distance_matrix, shape_param):
        return np.exp(-(shape_param*distance_matrix)**2)

    def InverseQ(distance_matrix, shape_param):
        return 1 / (1 + (shape_param * distance_matrix)**2)


def Franke(x, y):
    return 0.75*np.exp(-0.25*(9*x-2)**2-0.25*(9*y-2)**2) +\
            0.75*np.exp(-((9*x+1)**2)/49.0-((9*y+1)**2)/10.0) +\
            0.5*np.exp(-0.25*(9*x-7)**2-0.25*(9*y-3)**2) -\
            0.2*np.exp(-(9*x-4)**2-(9*y-7)**2)


def FetchPoints(file):
    points = np.loadtxt(file)
    return points


def DistanceMatrix(x_vals1, x_vals2, y_vals1, y_vals2):
    # broadcasting
    x_diff = x_vals2[:, np.newaxis] - x_vals1
    y_diff = y_vals2[:, np.newaxis] - y_vals1
    matrix = np.sqrt(x_diff**2 + y_diff**2)
    return matrix


def SystemMatrix(centers, RBF, shape_param):
    x_centers, y_centers = centers[:, 0], centers[:, 1]
    distance_matrix = DistanceMatrix(
            x_centers,
            x_centers,
            y_centers,
            y_centers,
            )
    matrix = RBF(distance_matrix=distance_matrix, shape_param=shape_param)
    return matrix


def ExpansionCoefficients(system_matrix, function):
    coefficients = npl.solve(system_matrix, function)
    return coefficients


def EvalutationMatrix(centers, eval_points, RBF, shape_param):
    x_centers, y_centers = centers[:, 0], centers[:, 1]
    x_evals, y_evals = eval_points[:, 0], eval_points[:, 1]

    d_matrix = DistanceMatrix(x_centers, x_evals, y_centers, y_evals)
    return RBF(d_matrix, shape_param)


def Interpolation(eval_matrix, expansion_coefficients):
    return eval_matrix@expansion_coefficients


# computing the error between the interpolation and known function
def Norm(interpolated_func, known_function):
    return npl.norm(interpolated_func - known_function, np.inf)


def Kappa(system_matrix):
    return npl.cond(system_matrix)


def main() -> None:
    CENTERS = 'xc.txt'
    EVAL_POINTS = 'x.txt'

    centers = FetchPoints(CENTERS)
    eval_points = FetchPoints(EVAL_POINTS)
    # A range of shape parameters
    shape_param = np.arange(0.2, 12.1, 0.1)
    # this is just an initialization a 3xN array that stores condition number,
    # error, and shape parameter for each shape parameter as an index.
    interpolation_data = np.zeros((3, len(shape_param)))
    known_function = Franke(centers[:, 0], centers[:, 1])

    # this writes the data to a csv file instead of printing
    # to the console
    with open("data.csv", mode='w', newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Epsilon", "Kappa", "e"])

        # Each iteration is a interpolation of the Franke function with
        # index of the interations being each shape paramter.
        for i, epsilon in enumerate(shape_param):
            sys_matrix = SystemMatrix(
                    centers=centers,
                    RBF=RBFs.Multiquadric,
                    shape_param=epsilon,
                    )
            expansion_coefficients = ExpansionCoefficients(
                    system_matrix=sys_matrix,
                    function=known_function,
                    )

            eval_matrix = EvalutationMatrix(
                    centers=centers,
                    eval_points=eval_points,
                    RBF=RBFs.Multiquadric,
                    shape_param=epsilon,
                    )

            interp = Interpolation(
                    eval_matrix=eval_matrix,
                    expansion_coefficients=expansion_coefficients,
                    )

            error = Norm(
                    interp,
                    Franke(eval_points[:,  0], eval_points[:, 1])
                    )

            condition_num = Kappa(
                    system_matrix=sys_matrix
                    )

            interpolation_data[0, i] = epsilon
            interpolation_data[1, i] = error
            interpolation_data[2, i] = condition_num

            writer.writerow([
                    f"{interpolation_data[0, i]:2.2f}",
                    f"{interpolation_data[2, i]:1.4e}",
                    f"{interpolation_data[1, i]:1.4e}"
                    ])

    # plots
    fig = plt.figure(1)
    ax1 = fig.add_subplot(121)
    ax1.semilogy(
            interpolation_data[0, :],
            interpolation_data[1, :],
            'g'
            )
    ax1.set_title("Shape Parameter vs. Error")
    ax1.set_xlabel("$\\epsilon$")
    ax1.set_ylabel("$e$")
    plt.grid(
            True,
            which='both',
            axis='both'
            )

    ax2 = fig.add_subplot(122)
    ax2.semilogy(
            interpolation_data[0, :],
            interpolation_data[2, :],
            )
    ax2.set_title("Shape Parameter vs. Condition Number")
    ax2.set_xlabel("$\\epsilon$")
    ax2.set_ylabel("$\\kappa$")

    plt.tight_layout()
    plt.grid(
            True,
            which='both',
            axis='both'
            )
    plt.show()

    return None


main()
