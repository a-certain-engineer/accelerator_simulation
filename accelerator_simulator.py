from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt


"""  
i want to programme a simulator for bunches of partices inside particle accelerators bunches along a certain trajectory
we will have a generation of a particle bunch and some optical elements defined as some fields acting on the particles
by making the particle go over eache of the optical element we will see how the particle interact and at the end how the 
distribution has changed

we will use SI unit

"""
#number of particles and useful constants
N=1000
c=299792458

#first and foremost we can define the datatype we are going to use, we define a class particles with the property of the particle we need

#define the particles kind(electron,positron, proton , muon...)

class particle_kind():
    charge: float
    mass: float
    name: str
    
electron_kind = particle_kind()
electron_kind.charge = -1.60217e-19
electron_kind.mass = 9.10938e-31
electron_kind.name = "electron"

proton_kind=particle_kind()
proton_kind.charge=1.60217e-19
proton_kind.mass=1.67262129e-27
proton_kind.name="proton"

#i know of course that neutrons cannot be accelerated and are mostly unaffected by magnetic field but if eventually i want to 
# implement particle interaction it might be  interesting to have interaction with neutron as well

neutron_kind=particle_kind()
neutron_kind.charge=0
neutron_kind.mass=1.67492749804e-27

muon_kind=particle_kind()
muon_kind.charge=-1.60217e-19
muon_kind.mass=1.88353e-28
muon_kind.name="muon"


class particle():
    kind: particle_kind  
    position: np.ndarray  
    velocity: np.ndarray  
    
    def __init__(self, kind=None, position=None, velocity=None):
        self.kind = kind
        self.position = np.array(position) if position is not None else np.zeros(3)
        self.velocity = np.array(velocity) if velocity is not None else np.zeros(3)

#now we can creact the array of struct that will the ensamble of our particles in the bunch
#initialization of the particles we will use 


ensemble=[]

for i in range(N):
    vx=np.random.normal(0,0.001)
    vy=np.random.normal(0,0.001)
    vz=np.random.normal(c-1000,0.1)
    
    velocity=np.array([vx,vy,vz])

    new_particle= particle(kind=electron_kind, 
                           position=np.random.normal(loc=0, scale=0.01,size= 3),
                           velocity=velocity
                           
                           )
    ensemble.append(new_particle)
    
#let's print the main characteristic at time 0 of ouor particle ensamble
    
print("=" * 60)
print("PARTICLE ENSEMBLE VERIFICATION")
print("=" * 60)

# Basic statistics
print(f"\nENSEMBLE OVERVIEW:")
print(f"Total particles: {len(ensemble)}")
print(f"Particle kind: {ensemble[0].kind.name if ensemble else 'N/A'}")



print(f"\nPOSITION STATISTICS:")
positions = np.array([p.position for p in ensemble])
print(f"X: mean = {np.mean(positions[:, 0]):.6f}, std = {np.std(positions[:, 0]):.6f}")
print(f"Y: mean = {np.mean(positions[:, 1]):.6f}, std = {np.std(positions[:, 1]):.6f}")
print(f"Z: mean = {np.mean(positions[:, 2]):.6f}, std = {np.std(positions[:, 2]):.6f}")


print(f"\nVELOCITY STATISTICS:")
velocities = np.array([p.velocity for p in ensemble])
print(f"VX: mean = {np.mean(velocities[:, 0]):.6f}, std = {np.std(velocities[:, 0]):.6f}")
print(f"VY: mean = {np.mean(velocities[:, 1]):.6f}, std = {np.std(velocities[:, 1]):.6f}")
print(f"VZ: mean = {np.mean(velocities[:, 2]):.6f}, std = {np.std(velocities[:, 2]):.6f}")

#now let's write a function to evaluate the emittance of the beam for a given axis
# we use the definition of the emittance RMS

def emittance(ensemble,asse):
    velocities=np.array([p.velocity for p in ensemble])
    positions=np.array([p.position for p in ensemble])
    avg_vel2=np.mean(velocities[:,asse]**2)
    avg_pos2=np.mean(positions[:,asse]**2)
    avg_product=np.mean(velocities[:,asse]*positions[:,asse]**2)
    emittance=np.pi*np.sqrt(avg_pos2*avg_vel2-avg_product**2)
    return emittance

#let's create a function to visualize a number of particles in a screen as in a phase space

def visualize(ensemble,asse):
   
    plt.xlabel("x")
    plt.ylabel("x'")
    plt.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    plt.axvline(x=0, color='k', linestyle='-', linewidth=0.5)
    plt.xlim(-0.04, 0.04)
    plt.ylim(-0.004, 0.004)
    
    
    for i in range(N):
        if (ensemble[i].velocity[asse] < 0.002 and  ensemble[i].velocity[asse] > -0.002
        and  ensemble[i].position[asse] <0.02 and  ensemble[i].position[asse] > -0.02 ):
            color='blue'
        else : 
            color ='red'
        plt.scatter(ensemble[i].position[asse],ensemble[i].velocity[asse],color=color)
        
    plt.show()
    
    
    
    

#let's define the magnetic fields with which the electron will interact we can imagine the fields as perfectly
# in the x,y plane since the particle is travelling in z direction(much higher velocities in that direction)

#define the space in which we will evaluate our different fields, imagine the dimensions of the vacuum chamber

x=np.linspace(-0.05, 0.05,10)
y=np.linspace(-0.05, 0.05,10)

#drift element in which the particle just drift away 
def drift (length_element, ensemble,N):
    velocities=np.array([p.velocity for p in ensemble])
    t=length_element/np.mean(velocities[:,2])    
    for i in range(N):
        
        ensemble[i].position[0]=ensemble[i].position[0]+ensemble[i].velocity[0]*t
        ensemble[i].position[1]=ensemble[i].position[1]+ensemble[i].velocity[1]*t
        ensemble[i].position[2]=ensemble[i].position[2]+length_element
   
    return ensemble
    
    

#magnetic dipole 
"""
parametric to visualize how a dipole field should look like 
(x,y)=np.meshgrid(x,y)
u=0
v=-1
plt.quiver(x,y,u,v)
plt.show()
"""
def dipole(length_element, Bx, By, ensemble,N):
    velocities=np.array([p.velocity for p in ensemble])
    t=length_element/np.mean(velocities[:,2])  
    time_steps=1000
    dt=t/time_steps
    for i in range(N):
        for j in range(time_steps):
            B=np.array([Bx,By,0])
            vxb=np.cross(ensemble[i].velocity,B)
            F= ensemble[i].kind.charge*vxb
            ensemble[i].velocity+= F*dt/ensemble[i].kind.mass
            ensemble[i].position[0]=ensemble[i].position[0]+ensemble[i].velocity[0]*dt
            ensemble[i].position[1]=ensemble[i].position[1]+ensemble[i].velocity[1]*dt
            ensemble[i].position[2]=ensemble[i].position[2]+ensemble[i].velocity[2]*dt
   
    return ensemble
    
    
    
#magnetic quadrupole
"""
parametric to visualize how a quadrupole field should look like 

(x,y)=np.meshgrid(x,y)
u=y
v=x
plt.quiver(x,y,u,v)
plt.show()

"""

def quadrupole(length_element,B_gradient,ensemble,N):
    velocities=np.array([p.velocity for p in ensemble])
    t=length_element/np.mean(velocities[:,2])  
    time_steps=1000
    dt=t/time_steps
    for i in range(N):
        for j in range (time_steps):
            Bx=B_gradient*ensemble[i].position[1]
            By=B_gradient*ensemble[i].position[0]
            B=np.array([Bx,By,0])
            vxb=np.cross(ensemble[i].velocity,B)
            F= ensemble[i].kind.charge*vxb
            ensemble[i].velocity+= F*dt/ensemble[i].kind.mass
            ensemble[i].position[0]=ensemble[i].position[0]+ensemble[i].velocity[0]*dt
            ensemble[i].position[1]=ensemble[i].position[1]+ensemble[i].velocity[1]*dt
            ensemble[i].position[2]=ensemble[i].position[2]+ensemble[i].velocity[2]*dt
   
    return ensemble
            
            
            
    
    
    
#magnetic octupole ancora da fare

#now we can define a function to evaluate the change of the particle characteristic entering an accelerating cavity
#with an extremely simplified version of a cavity where we have a constant field even during in time
#of course this ix extremely idealized but it's just supposed to get a sense of the change in velocity as it approaches c

def cavity(ensemble, length_element,E):
    t=10e-9
    time_steps=10
    dt=t/time_steps
    for i in range (N):
        for j in range(time_steps):
            gamma=1/(np.sqrt(1-(ensemble[i].velocity[2]/c)**2))
            ensemble[i].velocity[2]+=ensemble[i].kind.charge/ensemble[i].kind.mass/gamma*E*dt
            
        ensemble[i].position[2]+=length_element
    
    return ensemble


visualize(ensemble,0)

ensemble=drift(0.1,ensemble,N)

visualize(ensemble,0)

  
print("=" * 60)
print("PARTICLE ENSEMBLE FINAL")
print("=" * 60)

# Basic statistics
print(f"\nENSEMBLE FINAL OVERVIEW:")

print(f"Particle kind: {ensemble[0].kind.name if ensemble else 'N/A'}")



print(f"\nPOSITION STATISTICS:")
positions = np.array([p.position for p in ensemble])
print(f"X: new mean = {np.mean(positions[:, 0]):.6f}, std = {np.std(positions[:, 0]):.6f}")
print(f"Y: new mean = {np.mean(positions[:, 1]):.6f}, std = {np.std(positions[:, 1]):.6f}")
print(f"Z: new mean = {np.mean(positions[:, 2]):.6f}, std = {np.std(positions[:, 2]):.6f}")


print(f"\nVELOCITY STATISTICS:")
velocities = np.array([p.velocity for p in ensemble])
print(f"VX: new mean = {np.mean(velocities[:, 0]):.6f}, std = {np.std(velocities[:, 0]):.6f}")
print(f"VY: new mean = {np.mean(velocities[:, 1]):.6f}, std = {np.std(velocities[:, 1]):.6f}")
print(f"VZ: new mean = {np.mean(velocities[:, 2]):.6f}, std = {np.std(velocities[:, 2]):.6f}")

