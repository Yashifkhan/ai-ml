import pandas as pd 
from sklearn.neighbors import KNeighborsRegressor


df=pd.read_csv("restaurant_food_price.csv")

# print(df)
unique_foods = df["food_type"].unique()
# print("unique_foods",unique_foods)
count=1
for i in unique_foods:
    if count ==0:
        print("Available food types:")
    print(count, i )
    count+=1
user_input=int(input("Select the food : "))

user_lat = 28.7045
user_lon = 77.1028
food=""
if user_input == 1:
   print("User Selected Pizza")
   food="Pizza"
elif user_input == 2:
    print("User Selected Burger")
    food="Burgar"
elif user_input == 3 :
    print("User Selected Noodles")
    food="Noodles"
elif user_input == 4:
    print("User Selected Coffee")
    food="Coffee"
else:
    print("Invalid input try again")
    # return
    
print(food)
# print(df.head())
filtered_df = df[df["food_type"] == food]

X=filtered_df[["latitude","longitude"]]
y=filtered_df["price"]

model=KNeighborsRegressor(n_neighbors=3,weights='distance')
model.fit(X,y)
prediction = model.predict([[user_lat, user_lon]])

print(filtered_df)
print(f"Predicted {food} Price:", prediction[0])
    
