import pandas as pd
import numpy as np
import os

base_dir = os.path.dirname(os.path.dirname(__file__))
file_path = os.path.join(base_dir, "data", "student-mat.csv")

cols = [
    "studytime",   # study_hours
    "absences",    # attendance
    "G1", "G2",    # previous_marks
    "internet",
    "Medu", "Fedu",
    "G3",
]
df=pd.read_csv(file_path)
df = df[cols]

# rename
df.rename(columns={"studytime": "study_hours"}, inplace=True)

# attendance (important logic)
df['attendance'] = 100 - df['absences']

# previous marks
df['previous_marks'] = (df['G1'] + df['G2']) / 2

# parental education
df['parental_education'] = (df['Medu'] + df['Fedu']) / 2
df['sleep_hours'] = np.random.randint(5, 9, size=len(df))
df['assignments_completed'] = np.random.randint(1, 10, size=len(df))
df['result'] = df['G3'].apply(lambda x: 1 if x >= 10 else 0)
df.drop(["absences", "G1", "G2", "Medu", "Fedu"], axis=1, inplace=True)
print(df)