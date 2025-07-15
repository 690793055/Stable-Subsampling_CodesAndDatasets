import numpy as np 
import pyunidoe as pydoe

stat=pydoe.gen_ud(n=100, s=13, q=100, init="rand", crit="MD2", maxiter=100, vis=True)
print("The initial design: ")
print(stat["initial_design"])

print("The final design: ")
print(stat["final_design"])