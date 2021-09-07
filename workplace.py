def askForLocation():
    location = input("Where are you, home, jgh or cedar?")
    return location

def _workplace(location):
        switch = {
                "home": "/home/hosseinj/data/Case1_elekta_sim",
                "jgh": "/home/majd/data/TG186 Vallidation/Elekta/",  
                "cedar": 2 
                }
        return switch.get(location, "invalid input")
