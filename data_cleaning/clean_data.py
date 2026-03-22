import json 


# load the data 
def load_data(file_name):
    with open(file_name, 'r') as f:
        data = json.load(f)
    return data
data =load_data("product_data.json")


# cleaning the data 
def clean_data(data):
    text_to_num={"one":1,"two":2,"three":3,"four":4,"five":5}
    cleaned_data=[]
    unique_value=set()
    for user in data:
         
        # clean the data 
        raw_rating=user['rating'].strip().lower()
        if (raw_rating in text_to_num):
           user['rating']= text_to_num[raw_rating]    
        user[raw_rating]=raw_rating
        
        # remove data which is not present 
        raw_age=user.get("age")
        if(raw_age == None):
            raw_age=None
        user["age"]=raw_age
        
        # remove the duplicat value
        if(user["name"] in unique_value):
            continue
        unique_value.add(user["name"])
        cleaned_data.append(user)
            
    return cleaned_data
data = (clean_data(data))

# get the meaningfull insight 
def get_insight(data):
    total_rating=0
    for user in data:
        total_rating +=float(user["rating"])
    print("avg of rating is : " , total_rating/len(data))
    
    # get the percentage of user feed back 
    feed_back=0
    for user in data:
        if (float(user["rating"]) < 3):
            feed_back +=1
    
    print("perseon given rating in precent : ",feed_back/len(data)*100)
        
# data=get_insight(data)

def get_recomandation(data):
    recommendation = []
    for user in data:
        current_reco={}
        current_reco["name"]=user["name"]
        
        if(float(user["rating"]) >=4):
            current_reco["brand"]="apple"
        else :
            current_reco["brand"] ="ssmsung"
            
        recommendation.append(current_reco)
    
    return recommendation

print(get_recomandation(data))


    
    
    

