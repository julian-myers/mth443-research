import csv
import numpy as np
import pandas as pd


FILE11 = "PointWiseErrors-Test1.csv"
FILE21 = "ConditionNumbers1.csv"
FILE31 = "MaxErrors1.csv"

FILE12 = "PointWiseErrors-Test2.csv"
FILE22 = "ConditionNumbers2.csv"
FILE32 = "MaxErrors2.csv"

FILE13 = "PointWiseErrors-Test3.csv"
FILE23 = "ConditionNumbers3.csv"
FILE33 = "MaxErrors3.csv"


def OpenFile(file_name):
    data = pd.read_csv(file_name)
    df = pd.DataFrame(data)
    df = df.to_numpy()
    return df


def RecordStats(data1, data2, data3, testnum):
    num_cols = len(data1[1, :])
    with open(f"Test{testnum}-data.csv", mode='w') as file:
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
    pointwise_err1 = OpenFile(FILE11)
    cond_nums1 = OpenFile(FILE21)
    max_errs1 = OpenFile(FILE31)

    cond_nums1 = cond_nums1.flatten().tolist()
    max_errs1 = max_errs1.flatten().tolist()

    RecordStats(pointwise_err1, cond_nums1, max_errs1, testnum='1')
    pointwise_err2 = OpenFile(FILE12)
    cond_nums2 = OpenFile(FILE22)
    max_errs2 = OpenFile(FILE32)

    cond_nums2 = cond_nums2.flatten().tolist()
    max_errs2 = max_errs2.flatten().tolist()

    RecordStats(pointwise_err2, cond_nums2, max_errs2, testnum='2')
    pointwise_err3 = OpenFile(FILE13)
    cond_nums3 = OpenFile(FILE23)
    max_errs3 = OpenFile(FILE33)

    cond_nums3 = cond_nums3.flatten().tolist()
    max_errs3 = max_errs3.flatten().tolist()

    RecordStats(pointwise_err3, cond_nums3, max_errs3, testnum='3')
    return None


main()
