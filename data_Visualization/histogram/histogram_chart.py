import numpy as np
import matplotlib.pyplot as plt

# np.random.seed(0)
# score=np.random.normal(70,10,100)

# bins=[30,50,70,90,100]
# plt.hist(score,color="skyblue",edgecolor="black",bins=20)
# plt.show()

# legit_transactions = [
#     2.99, 5.49, 8.99, 12.50, 14.99, 19.99, 23.45, 29.99, 34.99, 39.50,
#     45.00, 49.99, 55.25, 60.00, 75.99, 89.99, 120.50, 150.00, 199.99,
#     249.99, 300.75, 450.00, 600.00, 850.00, 1200.00
# ]
# fraud_transactions = [
#     50, 100, 150, 200, 300, 500, 500, 750, 1000, 1000,
#     1200, 1500, 1500, 2000, 2500, 3000, 3000
# ]
# plt.hist(fraud_transactions,color="green",edgecolor="black",alpha=0.5 ,label="fraud",bins=20)
# plt.hist(legit_transactions,color="skyblue" ,edgecolor="black",alpha=0.5 ,label="legial",bins=20)
# plt.legend()
# plt.axvline(
#     x=1500,
#     label="mid value",
#     color="red",
#     linewidth="2",
#     linestyle="--"
# )
# plt.show()

# create a mult pul digram based on data set 


# x=[1,2,3,4,5,6,7,8,9]
# y1=[np.sqrt(i) for  i in x]  # square root of values
# y2=[i*1 for i in x]          # dubble of values 
# y3=[i**2 for i in x]         # square of value 
# y4=[i**3 for i in x]         # cube of value 

# fig,ax=plt.subplots(2,2)
# ax[0][0].plot(x,y1)
# ax[0][0].set_title("plot 1 square root")
# ax[0][0].set_xlabel("x-label values")
# ax[0][0].set_ylabel("y-label values")

# ax[0][1].plot(x,y2)
# ax[0][1].set_title("plot 2 dubble")
# ax[1][0].plot(x,y3)
# ax[1][0].set_title("plot 3 square")
# ax[1][1].plot(x,y4)
# ax[1][1].set_title("plot 4 cube values")

# fig.tight_layout()
# fig.suptitle("Multipul plots")
# plt.show()



# test for  the city data represented 
# days = ["Mon", "Tue", "Wed", "Thu", "Fri"]

# cities = ["New York", "London", "Delhi", "Tokyo"]

# temperatures = [
#     [22, 23, 21, 24, 25],  # New York
#     [18, 19, 17, 20, 21],  # London
#     [30, 32, 31, 33, 34],  # Delhi
#     [25, 26, 24, 27, 28]   # Tokyo
# ]

# fig,ax=plt.subplots(2,2)

# without loop 
# ax[0][0].plot(days,temperatures[0])
# ax[0][0].set_title(cities[0])
# ax[0][1].plot(days,temperatures[1])
# ax[0][1].set_title(cities[1])
# ax[1][0].plot(days,temperatures[2])
# ax[1][0].set_title(cities[2])
# ax[1][1].plot(days,temperatures[3])
# ax[1][1].set_title(cities[3])

# with loop optimize sol
# city_no=0
# for i in range(2):
#     for j in range(2):
#         ax[i][j].plot(days,temperatures[city_no] ,marker="o")
#         ax[i][j].set_title(cities[city_no])
#         ax[i][j].grid(True)
#         city_no+=1

# fig.suptitle("temperatures in  cities over the week")
# fig.tight_layout()
# plt.show()

