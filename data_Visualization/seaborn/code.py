import seaborn as sns
import matplotlib.pyplot as plt


# print("hello jii am learn seaborn library")
sns.set_theme()
# names=sns.get_dataset_names()
# load all names for avilable data 
# print("name is ",names)

# load only need data based on the name 
tips_data=sns.load_dataset("tips")
# print(tips_data.head())

# sns.relplot(
#     data=tips_data,
#     x="total_bill",
#     y="tip",
#     hue="smoker",
#     style="smoker",
#     size="size"
    
# )
# plt.show()


sns.scatterplot(
    data=tips_data,
    x="total_bill",
    y="tip",
    hue="time",
)
plt.show()