import csv
import numpy as np
import pandas as pd


FILE = "PointWiseErrors.csv"


def OpenFile(file_name):
    data = pd.read_csv(file_name)
    return data


def CalcAvg(data):
    return np.average(data)


def RecordAvgs(data1, data2, data3, data4):
    with open("AvgErrors.csv", mode='w') as file:
        writer = csv.writer(file)
        writer.writerow([
            "Constant",
            "Linearly Varying",
            "Exponentially Varying"
            "Random"
            ])

        writer.writerow([
            f"{data1:1.5e}",
            f"{data2:1.5e}",
            f"{data3:1.5e}",
            f"{data4:1.5e}",
            ])


def main():
    data = OpenFile(FILE)
    df = pd.DataFrame(data)
    df = df.to_numpy()

    con_errors = df[:, 1]
    lin_errors = df[:, 2]
    exp_errors = df[:, 3]
    rand_errors = df[:, 4]

    con_avg = CalcAvg(con_errors)
    lin_avg = CalcAvg(lin_errors)
    exp_avg = CalcAvg(exp_errors)
    rand_avg = CalcAvg(rand_errors)

    RecordAvgs(con_avg, lin_avg, exp_avg, rand_avg)

    print(f"constant: {con_avg:1.5e}" +
          f"linear: {lin_avg:1.5e}" +
          f"exponential: {exp_avg:1.5e}" +
          f"random: {rand_avg:1.5e}"
          )
    return None


main()
