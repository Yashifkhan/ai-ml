# print("start larning oops")
# class students:
#     def __init__(self,name,age,cgpa):
#         self.name =name
#         self.age = age
#         self.cgpa = cgpa
    
#     def get_cgpa(self):
#         return self.cgpa
    
# std1=students("yashif",22,9.5)
# std2=students("hello",20,8.5)
# print("cgpa of yashif : ",std1.get_cgpa())
# print("cgpa of hello : ",std2.get_cgpa())




# class Bank:
#     def __init__(self,name,balance,pin):
#         self.name = name             #public atribute
#         self._balance = balance      #protected atribute
#         self.__pin = pin            #private atribute ,we get the value with acc1.Bank__pin
    
#     def get_pin(self):
#         return self.__pin
    
#     def set_pin(self,new_pin):
#         self.__pin = new_pin
    
# account1 = Bank("yashif",1000,1234)
# print("account holder name : ",account1.name)          #accessing public atribute
# print("account balance : ",account1._balance)     #accessing protected atribute
# # print("account pin : ",account1.__pin)          #accessing private atribute (will raise an error) this value get by a specil function geter and seter
# # print("account pin : ",account1.get_pin())      #accessing private atribute using getter method
# account1.set_pin(5678)                          #modifying private atribute using setter method
# print("account pin after modification : ",account1.get_pin())      #accessing private a





# inheritence this are inhert the property of parent or base clas
class Teacher:
    def __init__(self,salary):
        self.salary = salary
        
class Student:
    def __init__(self, cgpa):
        self.cgpa = cgpa
    
class Ta(Teacher,Student):
    def __init__(self,salary,cgpa):
        super().__init__(salary)   #call the constructor of teacher class
        super().__init__(cgpa)     #call the constructor of student class
    

print("hii")
ta1=Ta(50000,9.5)
print("ta salary : ",ta1.salary)
# print("ta cgpa : ",ta1.cgpa)