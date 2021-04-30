from main import model
import pickle
import numpy as np
 
#---save the model to disk---
filename = 'telstrafinalized_model.sav'
 
#---write to the file using write and binary mode---
pickle.dump(model, open(filename, 'wb'))

loaded_model = pickle.load(open(filename, 'rb')) 


location= 438
severity_type = 1
resource_type = 2
log_feature = 98
volume = 3
event_type = 22 

prediction = loaded_model.predict([[ location, severity_type, resource_type, log_feature, volume, event_type]])

print(prediction)
if (prediction[0]==0):
    print("No fault")
    
elif (prediction==[1]):
    
    print("Minor fault")

else: 
    print("Major fault")
