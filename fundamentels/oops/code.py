# print("start larning oops")
class students:
    def __init__(self,name,age,cgpa):
        self.name =name
        self.age = age
        self.cgpa = cgpa
    
    def get_cgpa(self):
        return self.cgpa
    
std1=students("yashif",22,9.5)
std2=students("hello",20,8.5)
print("cgpa of yashif : ",std1.get_cgpa())
print("cgpa of hello : ",std2.get_cgpa())