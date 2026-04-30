# mse function error ko niklatra hai or 
# gradient descent us error ko less krta hai inside the loop 
# and hmm error ko kame krte hai using the loop 
# loop kitni bar chalna chiye ,
# data kitna kitna dena hia model ko 

#  we have 3 type of gradient descent 
# Batch GD ek bar me sara dxata de do or re try kro 
# Stochastic GD (SGD) one by one data do re try kro 
# Mini-batch GD data ko batch me devide kro or fir data do 

# main ly use in production 
# Mini-batch GD



# code implementation  for understanding 
# import numpy as np
# from sklearn.linear_model import SGDRegressor

# # what is reshap
# # conver the row into column  

# X = np.array([1,2,3,4,5]).reshape(-1,1)
# y = np.array([2,4,6,8,10])


# # model declaer part 
# model=SGDRegressor(
#     max_iter=1,
#     learning_rate="constant",
#     eta0=0.01,
#     warm_start=True
# )

# # model paramerts 
# epochs=5
# batch_size=2
# n=len(X)


# # accutaly work mini batch method of  Gd
# for epochs in range(epochs):
    
#     for i in range(0,n,batch_size):
        
#         # create batch 
#         X_batch=X[i:i+batch_size]
#         y_batch=y[i:i+batch_size]
        
#         model.partial_fit(X_batch,y_batch)
        
#         # print("X is :",X_batch)
#         # print("y is :",y_batch)
#         print(f"Batch {X_batch.flatten()} -> m={model.coef_[0]:.3f}, b={model.intercept_[0]:.3f}")

#         # print("model result is : ",model.coef_)  

# print("\nFinal Model:")
# print("m =", model.coef_[0])
# print("b =", model.intercept_[0])

# # -----------------------------
# # Prediction
# # -----------------------------
# print("\nPrediction for x=10:", model.predict([[10]])[0])


# code for direct use in production 

import numpy as np
from sklearn.linear_model import SGDRegressor

X = np.array([1,2,3,4,5]).reshape(-1,1)
y = np.array([2,4,6,8,10])

# X = X.reshape(-1,1)

model = SGDRegressor(max_iter=1000, learning_rate='constant', eta0=0.01)

model.fit(X, y)

print(model.coef_)      # m
print(model.intercept_) # b