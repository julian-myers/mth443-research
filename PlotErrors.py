import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


FILE1 = 'Test1/PointWiseErrors-Test1.csv'
FILE2 = 'Test2/PointWiseErrors-Test2.csv'
FILE3 = 'Test3/PointWiseErrors-Test3.csv'


def ReadCSV(file: str):
    df = pd.read_csv(file, skiprows=1)
    df.to_numpy()
    df = np.array(df)
    return df


def Plot(data, test_num: int) -> None:
    fig, axs = plt.subplots(4, 2)

    # ---------------first-row----------------
    # first column
    axs[0, 0].semilogy(data[:, 0], data[:, 1], label="$\\epsilon=2.4$")
    axs[0, 0].semilogy(
            data[:, 0],
            data[:, 2],
            label='LSP: $\\epsilon_{min}=3.7, \\epsilon_{max}=8.31$',
            )
    axs[0, 0].set_title('LSP')
    axs[0, 0].legend()
    axs[0, 0].grid(True)
    # second column
    axs[0, 1].semilogy(data[:, 0], data[:, 1], label="$\\epsilon=2.4$")
    axs[0, 1].semilogy(
            data[:, 0],
            data[:, 3],
            label='ESP: $\\epsilon_{min}=2.2, \\epsilon_{max}=7.2$',
            )
    axs[0, 1].set_title('ESP')
    axs[0, 1].legend()
    axs[0, 1].grid(True)

    # ---------------second-row---------------
    # first col
    axs[1, 0].semilogy(data[:, 0], data[:, 1], label="$\\epsilon=2.4$")
    axs[1, 0].semilogy(
            data[:, 0],
            data[:, 4],
            label='RSP: $\\epsilon_{min}=2.75, \\epsilon_{max}=8.82$',
            )
    axs[1, 0].set_title('RSP')
    axs[1, 0].legend()
    axs[1, 0].grid(True)
    # second col
    axs[1, 1].semilogy(data[:, 0], data[:, 1], label="$\\epsilon=2.4$")
    axs[1, 1].semilogy(
            data[:, 0],
            data[:, 5],
            label='TSP: $\\epsilon_{min}=2.1, \\epsilon_{max}=5.6955$',
            )
    axs[1, 1].set_title('TSP')
    axs[1, 1].legend()
    axs[1, 1].grid(True)

    # ---------------third-row----------------
    # first col
    axs[2, 0].semilogy(data[:, 0], data[:, 1], label="$\\epsilon=2.4$")
    axs[2, 0].semilogy(
            data[:, 0],
            data[:, 6],
            label='SSP: $\\epsilon_{min}=3.1, \\epsilon_{max}=6.5$',
            )
    axs[2, 0].set_title('SSP')
    axs[2, 0].legend()
    axs[2, 0].grid(True)
    # second col
    axs[2, 1].semilogy(data[:, 0], data[:, 1], label="$\\epsilon=2.4$")
    axs[2, 1].semilogy(
            data[:, 0],
            data[:, 7],
            label='DLSP: $\\epsilon_{min}=2.2, \\epsilon_{max}=6.2$',
            )
    axs[2, 1].set_title('DLSP')
    axs[2, 1].legend()
    axs[2, 1].grid(True)

    # ---------------fourth-row---------------
    # first col
    axs[3, 0].semilogy(data[:, 0], data[:, 1], label="$\\epsilon=2.4$")
    axs[3, 0].semilogy(
            data[:, 0],
            data[:, 8],
            label='HSP: $\\epsilon_{min}=2.58, \\epsilon_{max}=6.2$',
            )
    axs[3, 0].set_title('HSP')
    axs[3, 0].legend()
    axs[3, 0].grid(True)
    # second col
    axs[3, 1].semilogy(data[:, 0], data[:, 1], label="$\\epsilon=2.4$")
    axs[3, 1].semilogy(
            data[:, 0],
            data[:, 9],
            label='BSP: $\\epsilon_{min}=2.3, \\epsilon_{max}=6.2$',
            )
    axs[3, 1].set_title('BSP')
    axs[3, 1].legend()
    axs[3, 1].grid(True)

    if test_num == 1:
        fig.suptitle('$f(x) = e^{\\sin(\\pi x)}$')
    elif test_num == 2:
        fig.suptitle('$f(x) = x^3 + 3x^2 + 12x + 6$')
    elif test_num == 3:
        fig.suptitle('$f(x) = \\frac{1}{1 + x^2}$')

    return None


def main() -> None:

    test1_data = ReadCSV(FILE1)
    test2_data = ReadCSV(FILE2)
    test3_data = ReadCSV(FILE3)

    Plot(test1_data, test_num=1)
    Plot(test2_data, test_num=2)
    Plot(test3_data, test_num=3)

    plt.show()
    return None


main()
