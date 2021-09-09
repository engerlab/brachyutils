"""
Date 
        2021/9/9

Purpose
        the functions here allow the user to setup their homedirectory for several computers.

Author
        Hossein Jafarzadeh
        Enger Lab
        McGill University

Inputs
        inputs are obtained from the user on the terminal (interactive input) 

    Dependencies
        none

Outputs
        a string specifying the directory of the mother_dir used in validate_sims
"""
def askForLocation():
    location = input("Where are you, home, jgh or cedar? \n")
    return location

def _workplace(location):
        switch = {
                "home": "/home/hosseinj/data/Case1_elekta_sim",
                "jgh": "/home/majd/data/TG186 Vallidation/Elekta/",  
                "cedar": 2 
                }
        return switch.get(location, "invalid input")
