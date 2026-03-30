import numpy as np
import matplotlib.pyplot as plt

# np.random.seed(0)
# score=np.random.normal(70,10,100)

# bins=[30,50,70,90,100]
# plt.hist(score,color="skyblue",edgecolor="black",bins=20)
# plt.show()

legit_transactions = [
    2.99, 5.49, 8.99, 12.50, 14.99, 19.99, 23.45, 29.99, 34.99, 39.50,
    45.00, 49.99, 55.25, 60.00, 75.99, 89.99, 120.50, 150.00, 199.99,
    249.99, 300.75, 450.00, 600.00, 850.00, 1200.00
]
fraud_transactions = [
    50, 100, 150, 200, 300, 500, 500, 750, 1000, 1000,
    1200, 1500, 1500, 2000, 2500, 3000, 3000
]
plt.hist(fraud_transactions,color="green",edgecolor="black",alpha=0.5 ,label="fraud",bins=20)
plt.hist(legit_transactions,color="skyblue" ,edgecolor="black",alpha=0.5 ,label="legial",bins=20)
plt.legend()
plt.axvline(
    x=1500,
    label="mid value",
    color="red",
    linewidth="2",
    linestyle="--"
)
plt.show()