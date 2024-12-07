import pandas as pd


FILE1 = 'Test1-data.csv'
FILE2 = 'Test2-data.csv'
FILE3 = 'Test3-data.csv'


def main() -> None:

    df1 = pd.read_csv(FILE1)
    df2 = pd.read_csv(FILE2)
    df3 = pd.read_csv(FILE3)

    df1 = df1.transpose()
    df2 = df2.transpose()
    df3 = df3.transpose()

    df1.to_csv('Test1-data.csv', index=False)
    df2.to_csv('Test2-data.csv', index=False)
    df3.to_csv('Test3-data.csv', index=False)

    return None


main()
