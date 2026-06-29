import pandas as pd

df=pd.read_csv("../data/survey_results_public.csv")


# print the all uniqe skills user select in which 
# skills_column = df["LanguageHaveWorkedWith"]
# skills_column = skills_column.dropna()
# all_skills = skills_column.str.split(";")
# flat_skills = [skill for sublist in all_skills for skill in sublist]
# unique_skills = set(flat_skills)
# print("Total unique skills:", len(unique_skills))
# print(unique_skills)

# print the all uniqe db 
# db_skills = df["DatabaseHaveWorkedWith"]
# db_column = db_skills.dropna()
# all_db = db_column.str.split(";")
# flat_skills = [skill for sublist in all_db for skill in sublist]
# unique_db = set(flat_skills)
# print("Total unique db  skills:", len(unique_db))
# print(unique_db)


# print the all uniqe PlatformHaveWorkedWith 
# db_skills = df["PlatformHaveWorkedWith"]
# db_column = db_skills.dropna()
# all_db = db_column.str.split(";")
# flat_skills = [skill for sublist in all_db for skill in sublist]
# unique_db = set(flat_skills)
# print("Total unique db  skills:", len(unique_db))
# print(unique_db)


# print all the unieq ToolsTechHaveWorkedWith
# db_skills = df["ToolsTechHaveWorkedWith"]
# db_column = db_skills.dropna()
# all_db = db_column.str.split(";")
# flat_skills = [skill for sublist in all_db for skill in sublist]
# unique_db = set(flat_skills)
# print("Total unique db  skills:", len(unique_db))
# print(unique_db)


# print all the unieq ToolsTechHaveWorkedWith
# db_skills = df["EdLevel"]
# db_column = db_skills.dropna()
# all_db = db_column.str.split(";")
# flat_skills = [skill for sublist in all_db for skill in sublist]
# unique_db = set(flat_skills)
# print("Total unique db  skills:", len(unique_db))
# print(unique_db)

# db_skills = df["DevType"]
# db_column = db_skills.dropna()
# all_db = db_column.str.split(";")
# flat_skills = [skill for sublist in all_db for skill in sublist]
# unique_db = set(flat_skills)
# print("Total unique db  skills:", len(unique_db))
# print(unique_db)

# db_skills = df["YearsCodePro"]
# db_column = db_skills.dropna()
# all_db = db_column.str.split(";")
# flat_skills = [skill for sublist in all_db for skill in sublist]
# unique_db = set(flat_skills)
# print("Total unique db  skills:", len(unique_db))
# print(unique_db)


# db_skills = df["OrgSize"]
# db_column = db_skills.dropna()
# all_db = db_column.str.split(";")
# flat_skills = [skill for sublist in all_db for skill in sublist]
# unique_db = set(flat_skills)
# print("Total unique db  skills:", len(unique_db))
# print(unique_db)


db_skills = df["Country"]
db_column = db_skills.dropna()
all_db = db_column.str.split(";")
flat_skills = [skill for sublist in all_db for skill in sublist]
unique_db = set(flat_skills)
print("Total unique db  skills:", len(unique_db))
print(unique_db)




# print(df.columns)
# print(df.head())