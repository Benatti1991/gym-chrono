import pychrono as chrono
import pychrono.vehicle as veh
try:
   import pychrono.sensor as sens
except:
   print('Could not import Chrono Sensor')
#import pychrono.irrlicht as chronoirr
import numpy as np
import math
import os
from gym_chrono.envs.ChronoBase import  ChronoBaseEnv

# openai-gym imports
import gym
from gym import spaces
from gym_chrono.envs.utils.utilities import SetChronoDataDirectories, CalcInitialPose, areColliding

# from control_utilities.chrono_utilities import setDataDirectory
# ----------------------------------------------------------------------------------------------------
# Set data directory
#
# This is useful so data directory paths don't need to be changed everytime
# you pull from or push to github. To make this useful, make sure you perform
# step 2, as defined for your operating system.
#
# For Linux or Mac users:
#   Replace bashrc with the shell your using. Could be .zshrc.
#   1. echo 'export CHRONO_DATA_DIR=<chrono's data directory>' >> ~/.bashrc
#       Ex. echo 'export CHRONO_DATA_DIR=/home/user/chrono/data/' >> ~/.zshrc
#   2. source ~/.zshrc
#
# For Windows users:
#   Link as reference: https://helpdeskgeek.com/how-to/create-custom-environment-variables-in-windows/
#   1. Open the System Properties dialog, click on Advanced and then Environment Variables
#   2. Under User variables, click New... and create a variable as described below
#       Variable name: CHRONO_DATA_DIR
#       Variable value: <chrono's data directory>
#           Ex. Variable value: C:\Users\user\chrono\data\
# ----------------------------------------------------------------------------------------------------
"""
try:
    chrono.SetChronoDataPath(os.environ['CHRONO_DATA_DIR'])
    veh.SetDataPath(os.path.join(os.environ['CHRONO_DATA_DIR'], 'vehicle', ''))
except:
    raise Exception('Cannot find CHRONO_DATA_DIR environmental variable. Explanation located in chrono_sim.py file')
"""
def checkFile(file):
    if not os.path.exists(file):
        raise Exception('Cannot find {}. Explanation located in chrono_sim.py file'.format(file))


"""
def GetInitPose(p1, p2, z=0.55, reversed=0):
    initLoc = chrono.ChVectorD(p1[0], p1[1], z)

    vec = chrono.ChVectorD(p2[0], p2[1], z) - chrono.ChVectorD(p1[0], p1[1], z)
    theta = math.atan2((vec%chrono.ChVectorD(1,0,0)).Length(),vec^chrono.ChVectorD(1,0,0))
    if reversed:
        theta *= -1
    initRot = chrono.ChQuaternionD()
    initRot.Q_from_AngZ(theta)

    return initLoc, initRot
"""
class camera_obstacle_avoidance(ChronoBaseEnv):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}
    def __init__(self):
        ChronoBaseEnv.__init__(self)
        SetChronoDataDirectories()
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        self.camera_width  = 80
        self.camera_height = 45
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(low=0, high=255, shape=(self.camera_height, self.camera_width, 3), dtype=np.uint8)

        self.info =  {"timeout": 10000.0}
        self.timestep = 3e-3
        # ---------------------------------------------------------------------
        #
        #  Create the simulation system and add items
        #
        self.Xtarg = 100.0
        self.Ytarg = 0.0
        self.timeend = 20
        self.control_frequency = 10

        self.initLoc = chrono.ChVectorD(0, 0, 1.0)
        self.initRot = chrono.ChQuaternionD(1, 0, 0, 0)
        self.terrain_model = veh.RigidTerrain.PatchType_BOX
        self.terrainHeight = 0  # terrain height (FLAT terrain only)
        self.terrainLength = 250.0  # size in X direction
        self.terrainWidth = 15.0  # size in Y direction
        self.render_setup = False
        self.play_mode = False
        self.manager = 0

    def reset(self):
        del self.manager
        material = chrono.ChMaterialSurfaceNSC()
        self.vehicle = veh.HMMWV_Reduced()
        self.vehicle.SetContactMethod(chrono.ChContactMethod_NSC)
        self.vehicle.SetChassisCollisionType(veh.ChassisCollisionType_PRIMITIVES)

        self.vehicle.SetChassisFixed(False)
        self.vehicle.SetInitPosition(chrono.ChCoordsysD(self.initLoc, self.initRot))
        self.vehicle.SetPowertrainType(veh.PowertrainModelType_SHAFTS)
        self.vehicle.SetDriveType(veh.DrivelineType_AWD)
        #self.vehicle.SetSteeringType(veh.SteeringType_PITMAN_ARM)
        self.vehicle.SetTireType(veh.TireModelType_TMEASY)
        self.vehicle.SetTireStepSize(self.timestep)
        self.vehicle.Initialize()

        #self.vehicle.SetStepsize(self.timestep)
        if self.play_mode == True :
            self.vehicle.SetChassisVisualizationType(veh.VisualizationType_MESH)
            self.vehicle.SetWheelVisualizationType(veh.VisualizationType_MESH)
            self.vehicle.SetTireVisualizationType(veh.VisualizationType_MESH)
        else:
            self.vehicle.SetChassisVisualizationType(veh.VisualizationType_PRIMITIVES)
            self.vehicle.SetWheelVisualizationType(veh.VisualizationType_PRIMITIVES)
        self.vehicle.SetSuspensionVisualizationType(veh.VisualizationType_PRIMITIVES)
        self.vehicle.SetSteeringVisualizationType(veh.VisualizationType_PRIMITIVES)
        self.chassis_body = self.vehicle.GetChassisBody()
        self.chassis_body.GetCollisionModel().ClearModel()
        size = chrono.ChVectorD(3,2,0.2)
        self.chassis_body.GetCollisionModel().AddBox(material, 0.5 * size.x, 0.5 * size.y, 0.5 * size.z)
        self.chassis_body.GetCollisionModel().BuildModel()

        # Driver
        self.driver = veh.ChDriver(self.vehicle.GetVehicle())

        # Rigid terrain
        self.system = self.vehicle.GetSystem()
        self.terrain = veh.RigidTerrain(self.system)
        patch = self.terrain.AddPatch(material, chrono.ChVectorD(0, 0, self.terrainHeight - .5), chrono.VECT_Z,
                                 self.terrainLength, self.terrainWidth, 10)
        material.SetFriction(0.9)
        material.SetRestitution(0.01)
        #patch.SetContactMaterialProperties(2e7, 0.3)
        patch.SetTexture(veh.GetDataFile("terrain/textures/tile4.jpg"), 200, 200)
        patch.SetColor(chrono.ChColor(0.8, 0.8, 0.5))
        self.terrain.Initialize()

        ground_body = patch.GetGroundBody()
        ground_asset = ground_body.GetAssets()[0]
        visual_asset = chrono.CastToChVisualization(ground_asset)
        vis_mat = chrono.ChVisualMaterial()
        vis_mat.SetKdTexture(chrono.GetChronoDataFile("concrete.jpg"))
        visual_asset.material_list.append(vis_mat)

        # create obstacles
        self.boxes = []
        for i in range(3):
            box = chrono.ChBodyEasyBox(2, 2, 10, 1000, True, True)
            box.SetPos(chrono.ChVectorD(25 + 25*i, (np.random.rand(1)[0]-0.5)*10, .05))
            box.SetBodyFixed(True)
            box_asset = box.GetAssets()[0]
            visual_asset = chrono.CastToChVisualization(box_asset)

            vis_mat = chrono.ChVisualMaterial()
            vis_mat.SetAmbientColor(chrono.ChVectorF(0, 0, 0))
            vis_mat.SetDiffuseColor(chrono.ChVectorF(.2, .2, .9))
            vis_mat.SetSpecularColor(chrono.ChVectorF(.9, .9, .9))

            visual_asset.material_list.append(vis_mat)
            visual_asset.SetStatic(True)
            self.boxes.append(box)
            self.system.Add(box)

        # Set the time response for steering and throttle inputs.
        # NOTE: this is not exact, since we do not render quite at the specified FPS.
        steering_time = 1.0
        # time to go from 0 to +1 (or from 0 to -1)
        throttle_time = .5
        # time to go from 0 to +1
        braking_time = 0.3
        # time to go from 0 to +1
        self.SteeringDelta = (self.timestep / steering_time)
        self.ThrottleDelta = (self.timestep / throttle_time)
        self.BrakingDelta  =(self.timestep / braking_time)

        self.manager = sens.ChSensorManager(self.system)
        self.manager.scene.AddPointLight(chrono.ChVectorF(100, 100, 100), chrono.ChVectorF(1, 1, 1), 4000.0)
        self.manager.scene.AddPointLight(chrono.ChVectorF(-100, -100, 100), chrono.ChVectorF(1, 1, 1), 4000.0)
        # ------------------------------------------------
        # Create a self.camera and add it to the sensor manager
        # ------------------------------------------------
        self.camera = sens.ChCameraSensor(
            self.chassis_body,  # body camera is attached to
            50,  # scanning rate in Hz
            chrono.ChFrameD(chrono.ChVectorD(1.5, 0, .875)),
            # offset pose
            self.camera_width,  # number of horizontal samples
            self.camera_height,  # number of vertical channels
            chrono.CH_C_PI / 3,  # horizontal field of view
            #(self.camera_height / self.camera_width) * chrono.CH_C_PI / 3.  # vertical field of view
        )
        self.camera.SetName("Camera Sensor")
        self.camera.PushFilter(sens.ChFilterRGBA8Access())
        self.manager.AddSensor(self.camera)
        # -----------------------------------------------------------------
        # Create a filter graph for post-processing the data from the lidar
        # -----------------------------------------------------------------

        self.d_old = np.linalg.norm(self.Xtarg + self.Ytarg)
        self.step_number = 0
        self.c_f = 0
        self.isdone = False
        self.render_setup = False
        if self.play_mode:
            self.render()

        return self.get_ob()

    def step(self, ac):
        self.ac = ac.reshape((-1,))
        # Collect output data from modules (for inter-module communication)

        for i in range(round(1/(self.control_frequency*self.timestep))):
            self.driver_inputs = self.driver.GetInputs()
            # Update modules (process inputs from other modules)
            time = self.system.GetChTime()
            self.driver.Synchronize(time)
            self.vehicle.Synchronize(time, self.driver_inputs, self.terrain)
            self.terrain.Synchronize(time)

            trot = np.clip( (self.ac[0,]+1)/2 , self.driver.GetThrottle()-self.ThrottleDelta, self.driver.GetThrottle()+self.ThrottleDelta)
            self.driver.SetThrottle(trot)
            steer = np.clip(self.ac[1,], self.driver.GetSteering() - self.SteeringDelta,  self.driver.GetSteering() + self.SteeringDelta)
            self.driver.SetSteering(steer)
            self.driver.SetBraking(0)
            #self.driver.SetTargetBraking(self.ac[2,])

            # Advance simulation for one timestep for all modules
            self.driver.Advance(self.timestep)
            self.vehicle.Advance(self.timestep)
            self.terrain.Advance(self.timestep)
            self.system.DoStepDynamics(self.timestep)
            self.manager.Update()

            for box in self.boxes:
                self.c_f += box.GetContactForce().Length()

        self.rew = self.calc_rew()
        self.obs = self.get_ob()
        self.is_done()
        return self.obs, self.rew, self.isdone, self.info


    def get_ob(self):

        camera_data_RGBA8 = self.camera.GetMostRecentRGBA8Buffer()
        if camera_data_RGBA8.HasData():
            rgb = camera_data_RGBA8.GetRGBA8Data()[:,:,0:3]
        else:
            rgb = np.zeros((self.camera_height,self.camera_width,3))
            #print('NO DATA \n')

        return rgb

    def calc_rew(self):
        dist_coeff = 0.1
        #time_cost = -2
        progress = self.calc_progress()
        rew = dist_coeff*progress #+ time_cost*self.system.GetChTime()
        return rew

    def is_done(self):

        collision = not(self.c_f == 0)
        if self.system.GetChTime() > self.timeend:
            print('Time out')
            self.isdone = True
        if collision:
            print('Collision')
            self.isdone = True
        elif self.chassis_body.GetPos().z < -1 or abs(self.chassis_body.GetPos().y) + 1.2 > self.terrainWidth/2 :
            #self.rew += -500
            print('Fell of terrain')
            self.isdone = True

        elif self.chassis_body.GetPos().x > self.Xtarg :
            self.rew += 3000
            print('Success')
            self.isdone = True


    def render(self, mode='human'):
        if not (self.play_mode==True):
            raise Exception('Please set play_mode=True to render')

        if not self.render_setup:
            vis_camera = sens.ChCameraSensor(
                self.chassis_body,  # body camera is attached to
                30,  # scanning rate in Hz
                chrono.ChFrameD(chrono.ChVectorD(-8, 0, 3), chrono.Q_from_AngAxis(chrono.CH_C_PI / 20, chrono.ChVectorD(0, 1, 0))),
                # offset pose
                1280,  # number of horizontal samples
                720,  # number of vertical channels
                chrono.CH_C_PI / 3,  # horizontal field of view
                #(720/1280) * chrono.CH_C_PI / 3.  # vertical field of view
            )
            vis_camera.SetName("Vis Camera Sensor")
            vis_camera.PushFilter(sens.ChFilterVisualize(1280, 720))
            self.manager.AddSensor(vis_camera)
            self.render_setup = True

        if (mode == 'rgb_array'):
            return self.get_ob()
        """
        self.myapplication.BeginScene(True, True, chronoirr.SColor(255, 140, 161, 192))
        self.myapplication.DrawAll()
        self.myapplication.EndScene()
        self.step_number += 1
        self.myapplication.Synchronize("", self.driver_inputs)
        """
    def calc_progress(self):
        d = np.linalg.norm([self.Ytarg - self.chassis_body.GetPos().y, self.Xtarg - self.chassis_body.GetPos().x])
        progress = -(d - self.d_old) / self.timestep
        self.d_old = d
        return progress

    def close(self):
        del self


    def __del__(self):
        del self.manager

