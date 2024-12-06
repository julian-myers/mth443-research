import csv
import numpy as np
import pandas as pd


FILE = "PointWiseErrors.csv"


def OpenFile(file_name):
    data = pd.read_csv(file_name)
    df = pd.DataFrame(data)
    df = df.to_numpy()
    return df


def RecordStats(data):
    num_cols = len(data[1, :])
    with open("AvgErrors.csv", mode='w') as file:
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
            avgs.append(np.average(data[:, i]))

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
    return None


def main() -> None:
    data = OpenFile(FILE)
    RecordStats(data)
    return None


main()
