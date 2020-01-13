import pychrono as chrono
try:
   from pychrono import irrlicht as chronoirr
except:
   print('Could not import ChronoIrrlicht')
import numpy as np
from gym import Env, spaces
from collections import OrderedDict
import os


class ChronoBaseEnv(Env):
   def __init__(self):
      chronopath = os.path.join(os.path.dirname(__file__), 'data')
      chrono.SetChronoDataPath(chronopath)
      self.render_setup = False
      return 

   def step(self, ac):
       raise NotImplementedError
       
   def reset(self):
       raise NotImplementedError

   def get_ob(self):
          raise NotImplementedError

                 
   def is_done(self):
          raise NotImplementedError
   
   def ScreenCapture(self, interval):
          try: 
              self.myapplication.SetVideoframeSave(True)
              self.myapplication.SetVideoframeSaveInterval(interval)
              
          except:
                 print('No ChIrrApp found. Cannot save video frames.')
                 
                 
   def Render(self):
         raise NotImplementedError

    
    
   def convert_observation_to_space(self, observation):
        if isinstance(observation, dict):
            space = spaces.Dict(OrderedDict([
                (key, self.convert_observation_to_space(value))
                for key, value in observation.items()
            ]))
        elif isinstance(observation, np.ndarray):
            low = np.full(observation.shape, -float('inf'))
            high = np.full(observation.shape, float('inf'))
            space = spaces.Box(low, high, dtype=observation.dtype)
        else:
            raise NotImplementedError(type(observation), observation)
    
        return space

 
   def _set_observation_space(self, observation):
        self.observation_space = self.convert_observation_to_space(observation)
        return self.observation_space
    
   def __del__(self):
        if self.render_setup:
            self.myapplication.GetDevice().closeDevice()
            print('Destructor called, Device deleted.')
        else:
            print('Destructor called, No device to delete.')
        
   def __setstate__(self, state):
        self.__init__()
        return {self}

   def __getstate__(self):

        return {}
        
