def askForLocation():
    location = input("Where are you, home or jgh?")
    return location

def _workplace(location):
        switch = {
                "home": 0,
                "jgh": 1  
                }
        return switch.get(location, "invalid input")
