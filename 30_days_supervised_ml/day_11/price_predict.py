import pandas as pd 

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

user_location={29.099191369077577, 75.9610772580298}
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
    
