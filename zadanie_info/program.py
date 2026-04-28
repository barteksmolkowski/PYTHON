import os

import pandas as pd
from tabulate import tabulate

script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

df = pd.read_csv("it_database.txt", sep=";")

df["BirthDate"] = pd.to_datetime(df["BirthDate"]).dt.strftime("%Y-%m-%d")

TASK1 = df[
    (df["ClassID"].isin([1, 3, 5])) & (df["Physics"] >= 3) & (df["Chemistry"] >= 3)
]
print(
    "TASK 1 - Classes 1, 3, 5 and Physics/Chemistry grades >= 3:\n"
    f"{tabulate(TASK1.values.tolist(), headers=TASK1.columns.tolist(), tablefmt='psql')}\n"
)

TASK2 = df[df["Birthplace"].isin(["Warsaw", "Wroclaw"])][
    ["FirstName", "LastName", "Polish", "Physics"]
]
print(
    "TASK 2 - Polish and Physics grades (Birthplace: Warsaw or Wroclaw):\n"
    f"{tabulate(TASK2.values.tolist(), headers=TASK2.columns.tolist(), tablefmt='psql')}\n"
)

TASK3 = df[
    (df["Gender"] == "F") & (pd.to_datetime(df["BirthDate"]).dt.year.isin([1965, 1969]))
]
print(
    "TASK 3 - Females born in 1965 and 1969:\n"
    f"{tabulate(TASK3.values.tolist(), headers=TASK3.columns.tolist(), tablefmt='psql')}\n"
)

TASK4 = df.groupby("ClassID")["Polish"].mean().reset_index()
TASK4.columns = ["Class_ID", "Average_Polish_Grade"]
print(
    "TASK 4 - Average Polish grade per class:\n"
    f"{tabulate(TASK4.values.tolist(), headers=TASK4.columns.tolist(), tablefmt='psql')}\n"
)

TASK5 = df["ClassID"].value_counts().reset_index()
TASK5.columns = ["Class_ID", "Student_Count"]
TASK5 = TASK5.sort_values(by="Class_ID")
print(
    "TASK 5 - Number of students per class:\n"
    f"{tabulate(TASK5.values.tolist(), headers=TASK5.columns.tolist(), tablefmt='psql')}"
)
