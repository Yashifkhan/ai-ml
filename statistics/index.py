
# statics have two part 
# 1. Descriptive Statistics 
# 2.Inferential Statistics


# 1 Descriptive Statistics 
# 1.1 mean ,avarge 
# x = Observations
# n = number of terms
import numpy as np 
# arr=[3,5,7,9]

# manully 
def cal_mean(arr):
    count=0
    len_items=0
    for i in arr:
        count+=i
        len_items+=1
    return count /len_items
# print(cal_mean(arr))    

# with nop 
# mean=np.mean(arr)
# print(mean)


# ************ Measures of Central Tendency ************

# 1.2 mode 
# The most frequently occurring value in the dataset.
# arr=[1,3,1,5,2,3,4,1,2,1]
# import scipy.stats as stats

# manully 
# def count_freq(arr):
#     new_dict={}
#     for i in arr:
#         new_dict[i]=new_dict.get(i,0)+1
#         most_freq=max(new_dict.values())
#         most_freq_key=max(new_dict,key=new_dict.get)
#     return most_freq_key
# print(count_freq(arr))

# with statistics library 
# mode=stats.mode(arr)
# print(mode[0])


# 1.3 median 
# The median is the middle value in a sorted dataset 

# manully 
# arr=[1,2,3,4,5]
# def count_median(arr):
#     arr.sort()
#     n=len(arr)
#     if n %2 ==1:
#         return arr[n // 2]
#     else:
#         return (arr[n //2-1] + arr[n//2])/2
    
# print(count_median(arr))

# with numpy 
# import numpy as np 
# arr=[1,2,3,4,5,6]
# median=np.median(arr)
# print(median)


#1.2.1 Measure of Variability

# 1.2 learn Measure of Variability
# 1.2.1 range 
# largest value in data set - smallst value in data set 
# arr=[1,2,3,4,5]
# max_val=max(arr)
# min_val=min(arr)
# range_is=max_val-min_val
# print(range_is)


# 1.2.2 variance 
# max value - in all value in data set ans sqrt /n 
# arr=[2, 4, 6, 8]
# def mean(arr):
#     count=0
#     for i in arr:
#         count+=i
#     return count/(len(arr))

# mean_val=mean(arr)
# new_arr=[]
# for i in arr:
#     new_arr.append((i-mean_val))
    
# new_list2=[]
# for i in new_arr:
#     new_list2.append(abs(i)**2)

# variance=sum(new_list2)/len(arr)
# print(variance)



# libaray return sample varince 
# import statistics

# # sample data
# arr = [2,4,6,8]
# # variance
# print("Var = ", (statistics.variance(arr)))


# 1.2.3 Standard deviation  sqrt of varince 
# import statistics
# arr = [1, 2, 3, 4, 5]
# print("Std = ", (statistics.stdev(arr)))

# σ=Variance
# Step 3: Take square root

# sqrt(5) ==2.23

# print("start lerninf from one vedio ")

# print("leatning the random var and probaility distrutbition")

# learn the varines and stand deviantion 
# and learn data skewness and how to fit it 




# basics complete learn of statics ,now try on data and prectice it 