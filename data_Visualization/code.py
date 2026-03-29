import matplotlib.pyplot  as plt

# this is the line chart 
# first data plot
# x=[1,2,3,4,5]
# y=[6,7,8,9,10]
# plt.plot(x,y)
# plt.show()

# real data plot 
# oscar_movies = [
#     "Money Heist",
#     "Game of Thrones",
#     "Lucifer",
#     "Fast and Furious",
#     "Harry Potter"
# ]

# years = [2000, 2004, 2008, 2009, 2012]

# revenue = [100, 205,300,400,500]

# plt.plot(years,revenue)
# plt.title("Movies Revens With Year")
# plt.xlabel("Years")
# plt.ylabel("Revenus")
# plt.show()

# this is the bar chat 
# oscar_movies = [
#     "Money Heist",
#     "Game of Thrones",
#     "Lucifer",
#     "Fast and Furious",
#     "Harry Potter"
# ]
# years = [2008, 2009, 2010, 2012, 2015]
# revenue = [1006,170,427,233,788]

# plt.bar(years,revenue)
# plt.title("movies bar chart revenus")
# plt.xlabel("years")
# plt.ylabel("revenu")
# for i in range(len(years)):
#     plt.text(years[i],revenue[i],revenue[i],ha="center")

# plt.show()


# this are the scatter char

people = [
    "Person A", "Person B", "Person C", "Person D", "Person E",
    "Person F", "Person G", "Person H", "Person I", "Person J"
]

age = [22, 25, 30, 35, 40, 45, 50, 55, 60, 65]

blood_pressure = [110, 115, 120, 122, 125, 130, 135, 123, 145, 150]
 
plt.scatter(age,blood_pressure)
plt.title("blood VS bp")
plt.xlabel("age")
plt.ylabel("blood _pres.")
plt.show()