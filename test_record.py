import numpy as np
import matplotlib.pyplot as plt
from py_wake.examples.data.dtu10mw._dtu10mw import DTU10MW
from py_wake.utils.plotting import setup_plot
import dynamiks.utils
print(dir(dynamiks.utils))


# setup PyWake AEP and gradient function
from py_wake.wind_farm_models.engineering_models import PropagateDownwind
from py_wake.site._site import UniformSite
from py_wake.deficit_models.gaussian import NiayifarGaussianDeficit
from py_wake.deflection_models.gcl_hill_vortex import GCLHillDeflection
from py_wake.turbulence_models.crespo import CrespoHernandez
from py_wake.utils.gradients import autograd
from py_wake.rotor_avg_models.rotor_avg_model import CGIRotorAvg
import os
import numpy as np
import geojson
import geopandas as gpd

U = 10   # wind speed in m/s
TI = 0.1 # turbulence intensity (10%)


wt = DTU10MW()
wfm = PropagateDownwind(UniformSite(ws=U, ti=TI), wt, NiayifarGaussianDeficit(),
                        deflectionModel=GCLHillDeflection(),
                        turbulenceModel=CrespoHernandez(),
                        rotorAvgModel=CGIRotorAvg(21))

wt_x = np.array([316030.33, 316050.11, 316096.62, 317848.16, 319758.56, 317919.63, 
                 317889.61, 319720.91, 321553.09, 321652.26, 321627.03, 323416.21, 323440.36,])
wt_y = np.array([4549465.08, 4551316.05, 4553165.91, 4553188.73, 4553154.72, 4551229.41,
                 4549485.14, 4549453.15, 4549461.59, 4551337.22, 4553042.26, 4551307.78, 4549535.70])


wd_lst = np.arange(165, 265,2)
#yaw = np.ones((3, len(wd_lst))) # one deg misalignment as initial guess to get out of local minimum at 0deg

yaw = np.ones((13, len(wd_lst)))  # Now 12 turbines

def aep(yaw):
    return wfm.aep(wt_x, wt_y, yaw=yaw.reshape((13, len(wd_lst))), tilt=0, wd=wd_lst)

def daep(yaw):
    return [autograd(aep)(yaw)]

def plot(yaw,wd):
    wfm(wt_x, wt_y, yaw=yaw, tilt=0, wd=wd).flow_map().plot_wake_map()
    

import topfarm
import subprocess
import sys
import wetb


    
from topfarm._topfarm import TopFarmProblem
from topfarm.cost_models.cost_model_wrappers import AEPCostModelComponent
cost_comp = AEPCostModelComponent(input_keys=['yaw'], n_wt=len(yaw.flatten()),
                                  cost_function=aep, cost_gradient_function=daep)
tf = TopFarmProblem(design_vars={'yaw': (yaw.flatten(), -40, 40)}, cost_comp=cost_comp, n_wt=len(yaw.flatten()))

tf.optimize()
yaw_tabular = tf.state['yaw'].reshape((13, len(wd_lst)))

yaw_tabular = np.round(yaw_tabular).astype(int)


print(yaw_tabular)

print(str(yaw_tabular.tolist()).replace(" ",""))


yaw_tabular = np.array([
    np.round(10 * np.sin(np.linspace(0, 2*np.pi, len(wd_lst))))  # pattern example
    for _ in range(13)
], dtype=int)

for i, y_ in enumerate(yaw_tabular):
    plt.plot(wd_lst, y_, label=f'WT {i}')
setup_plot(xlabel='Wind direction [deg]', ylabel='Yaw misalignment [deg]')

for wd in [195,225]:
    plt.figure(figsize=(8,2))
    plot(yaw_tabular[:,wd_lst==wd][:,0],wd)
    plt.title(f'{wd} deg')
    
    
def simple_wind_farm_controller(flowSimulation):
    wd = flowSimulation.wind_direction
    wd_index = np.argmin(np.abs(wd_lst - wd))
    yaw = yaw_tabular[:,wd_index]
    flowSimulation.windTurbines.yaw = yaw
    
def wind_direction_changer(flowSimulation):
    flowSimulation.wind_direction = 260+flowSimulation.time/100
    
from dynamiks.utils.test_utils import DefaultDWMFlowSimulation, DemoSite
from dynamiks.dwm.particle_motion_models import HillVortexParticleMotion
from dynamiks.wind_turbines.pywake_windturbines import PyWakeWindTurbines
from dynamiks.views import XYView, EastNorthView, MultiView

wts = PyWakeWindTurbines(x=wt_x, y=wt_y, windTurbine=DTU10MW())
fs = DefaultDWMFlowSimulation(windTurbines=wts, particleMotionModel=HillVortexParticleMotion(),
                          d_particle=.1, n_particles=100, ti=TI, ws=U,
                          step_handlers=[wind_direction_changer, simple_wind_farm_controller])

"""
fs.visualize(2000, dt=10, interval=.1, view=EastNorthView(
    x=np.linspace(wt_x.min() - 500, wt_x.max() + 500, 500), y=np.linspace(wt_y.min() - 500, wt_y.max() + 500, 500),
    visualizers=[lambda fs: plt.title(f'Time: {fs.time}s, wind direction: {fs.wind_direction}deg')]), id='WindFarmControlSimple')
"""
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FFMpegWriter

fig, ax = plt.subplots(figsize=(10, 4))

# Prepare a view (same as used in fs.visualize)
view = EastNorthView(
    x=np.linspace(wt_x.min() - 500, wt_x.max() + 500, 500),
    y=np.linspace(wt_y.min() - 500, wt_y.max() + 500, 500),
    visualizers=[
        lambda fs: plt.title(f"Time: {fs.time:.1f}s, wind direction: {fs.wind_direction:.1f}°")
    ]
)

def update_plot(i):
    fs.step()  # advance one timestep
    ax.clear()  # still needed to reset the figure
    view(fs)  # let EastNorthView handle plotting internally

ani = animation.FuncAnimation(fig, update_plot, frames=2000, interval=100)

writer = FFMpegWriter(fps=10, metadata=dict(artist='DYNAMIKS User'), bitrate=1800)
ani.save("Revolution_SouthFork_Wind_simulation.mp4", writer=writer, dpi=200)

print('done')
