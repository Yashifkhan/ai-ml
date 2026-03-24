
import numpy as np

# create array 
arr1=np.array([1,2,3,4,5,6])           #one dim.array
arr2=np.array([[1,2,3],[1,2,3]])        #two dim.array
arr3=np.array([
    [[1,2], [3,4], [5,6]],
    [[7,8], [9,10], [11,12]],
    [[13,14], [15,16], [17,18]]
])       #three dim.array

# print("arr",arr1)
# print("arr2",arr2)
# print("arr3",arr3)

# calculate the mean 
mean=np.mean(arr1)

# cal culate the standed devision 
std_div=np.std(arr1)
normal_arr=(arr1-mean)/std_div
# print(mean)
print("normalize arra",normal_arr)