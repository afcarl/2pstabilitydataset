# Run script with python run.py sim_id, where sim_id is an integer to use for the sim_id (and the RNG seed). Runs a base system and a shadow system
import numpy as np
import rebound
from random import random, uniform, seed
import time
import sys
import math

def collision(reb_sim, col):
    reb_sim.contents._status = 5 # causes simulation to stop running and have flag for whether sim stopped due to collision
    return 0

def run_random(sim_id, integrator="whfast", dt=None, maxorbs=1.e9, betamin=1., betamax=30., shadow=False, runstr=None):
    if dt is None:
        dt = 2.*math.sqrt(3)/100. # dt approx 3.5 % of innermost orbital period

    datapath = '../data/random/'

    a1 = 1. # All distances in units of the innermost semimajor axis (always at 1)
    Mstar = 1. # All masses in units of stellar mass

    logMmin = np.log10(1.e-7) # 1/3 Mars around Sun
    logMmax = np.log10(1.e-4) # 2 Nep around Sun
    logincmin = np.log10(1.e-3)
    logincmax = np.log10(1.e-1) # max mutual inclination of 11.4 degrees. Absolute of 5.7 deg

    seed(sim_id)

    M1 = 10.**uniform(logMmin, logMmax)
    M2 = 10.**uniform(logMmin, logMmax)
    M3 = 10.**uniform(logMmin, logMmax)

    hill12 = a1*((M1+M2)/3.)**(1./3.)
    beta1 = uniform(betamin, betamax)
    a2 = a1 + beta1*hill12

    hill23 = a2*((M2+M3)/3.)**(1./3.)
    beta2 = uniform(betamin, betamax)
    a3 = a2 + beta2*hill23

    minhill = min(hill12, hill23)

    ecrit1 = (a2-a1)/a1
    ecrit21 = (a2-a1)/a2
    ecrit23 = (a3-a2)/a2
    ecrit3 = (a3-a2)/a3
    
    logemax1 = np.log10(ecrit1)
    logemax2 = np.log10(min(ecrit21, ecrit23))
    logemax3 = np.log10(ecrit3)

    logemin1 = np.log10(M2/ecrit1**2)
    logemin2 = np.log10(max(M1/ecrit21**2, M3/ecrit23**2))
    logemin3 = np.log10(M2/ecrit3**2)

    #print("M1 = {0}, M2 = {1}, M3 = {2}".format(M1, M2, M3))
    #print("beta1 = {0}, beta2 = {1}".format(beta1, beta2))
    #print("a1 = {0}, a2 = {1}, a3 = {2}".format(a1, a2, a3))
    #print("emax1 = {0}, emax2 = {1}, emax3 = {2}".format(10**logemax1, 10**logemax2, 10**logemax3))
    #print("emin1 = {0}, emin2 = {1}, emin3 = {2}".format(10**logemin1, 10**logemin2, 10**logemin3))
    #print("minhill = {0}".format(minhill))

    emax = 0.999
    e1 = min(10.**uniform(logemin1, logemax1), emax) # make sure ecc < 1
    e2 = min(10.**uniform(logemin2, logemax2), emax)
    e3 = min(10.**uniform(logemin3, logemax3), emax)
    
    i1 = 10.**uniform(logincmin, logincmax)
    i2 = 10.**uniform(logincmin, logincmax)
    i3 = 10.**uniform(logincmin, logincmax)

    sim = rebound.Simulation()
    sim.integrator=integrator
    sim.ri_whfast.safe_mode = 0
    sim.G = 4*np.pi**2

    sim.add(m=1.)
    sim.add(m=M1, a=a1, e=e1, pomega=random()*2.*np.pi, inc=i1, Omega=random()*2.*np.pi, f=random()*2.*np.pi, r=minhill)
    sim.add(m=M2, a=a2, e=e2, pomega=random()*2.*np.pi, inc=i2, Omega=random()*2.*np.pi, f=random()*2.*np.pi, r=minhill)
    sim.add(m=M3, a=a3, e=e3, pomega=random()*2.*np.pi, inc=i3, Omega=random()*2.*np.pi, f=random()*2.*np.pi, r=minhill)
    sim.move_to_com()
    ps = sim.particles

    if shadow:
        kicksize=1.e-11
        ps[2].x += kicksize

    if integrator=="whfast":
        sim.dt = dt*sim.particles[1].P
    sim.collision = "direct"
    sim.collision_resolve = collision
        
    #runstr = "{0:0=7d}.bin".format(sim_id)
    if shadow:
        shadowstr = 'shadow'
    else:
        shadowstr = ''

    if runstr:
        sim.save(datapath+'initial_conditions/'+shadowstr+'runs/ic'+runstr)
        sim.initSimulationArchive(datapath+'simulation_archives/'+shadowstr+'runs/sa'+runstr, interval=maxorbs/1000.)

    E0 = sim.calculate_energy()
    t0 = time.time()
    sim.integrate(maxorbs) # will stop if collision occurs
    Ef = sim.calculate_energy()    
    Eerr = abs((Ef-E0)/E0)

    if runstr:
        sim.save(datapath+'final_conditions/'+shadowstr+'runs/fc'+runstr)

    return (sim.t, Eerr, time.time()-t0)
