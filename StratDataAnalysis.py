import csv
import numpy as np
import pandas as pd


FILE1 = "PointWiseErrors-Test3.csv"
FILE2 = "ConditionNumbers3.csv"
FILE3 = "MaxErrors3.csv"


def OpenFile(file_name):
    data = pd.read_csv(file_name)
    df = pd.DataFrame(data)
    df = df.to_numpy()
    return df


def RecordStats(data1, data2, data3):
    num_cols = len(data1[1, :])
    with open("Test3-data.csv", mode='w') as file:
        writer = csv.writer(file)
        writer.writerow([
            "CSP",
            "LSP",
            "ESP",
            "RSP",
            "TSP",
            "SSP",
            "DLSP",
            "HSP",
            "BSP",
            ])

        avgs = []
        for i in range(1, num_cols):
            avgs.append(np.average(data1[:, i]))

        writer.writerow([
            f"{avgs[0]:1.5e}",
            f"{avgs[1]:1.5e}",
            f"{avgs[2]:1.5e}",
            f"{avgs[3]:1.5e}",
            f"{avgs[4]:1.5e}",
            f"{avgs[5]:1.5e}",
            f"{avgs[6]:1.5e}",
            f"{avgs[7]:1.5e}",
            f"{avgs[8]:1.5e}",
            ])
        writer.writerow(data2)
        writer.writerow(data3)

    return None


def main() -> None:
    pointwise_err = OpenFile(FILE1)
    cond_nums = OpenFile(FILE2)
    max_errs = OpenFile(FILE3)

    cond_nums = cond_nums.flatten().tolist()
    max_errs = max_errs.flatten().tolist()

    RecordStats(pointwise_err, cond_nums, max_errs)
    return None


main()
