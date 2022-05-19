import os
from datetime import datetime
if __name__ == "__main__":
    
    path = "./Results/" + datetime.now().strftime("%y%m%d-%H%M%S-") + "blah"
    try:
        os.makedirs(path)
    except:
        pass
    
    