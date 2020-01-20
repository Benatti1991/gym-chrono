# =============================================================================
# PROJECT CHRONO - http:#projectchrono.org
#
# Copyright (c) 2014 projectchrono.org
# All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be found
# in the LICENSE file at the top level of the distribution and at
# http:#projectchrono.org/license-chrono.txt.
#
# =============================================================================
# Authors: Simone Benatti
# =============================================================================
#
# RoboSimian on rigid terrain
#
# =============================================================================
import os
import math
import numpy as np
import pychrono as chrono
import pychrono.robosimian as robosimian
import pychrono.vehicle as vehicle
from gym_chrono.envs.ChronoBase import  ChronoBaseEnv
from gym import spaces
#import pychrono.mkl as mkl
try:
   from pychrono import irrlicht as chronoirr
except:
   print('Could not import ChronoIrrlicht')



# =============================================================================

class Raycaster:

    def __init__(self, sys, origin, dims, spacing):
        self.m_sys = sys
        self.m_origin = origin
        self.m_dims = dims
        self.m_spacing = spacing
        self.m_points = []
        
        self.m_body = sys.NewBody()
        self.m_body.SetBodyFixed(True)
        self.m_body.SetCollide(False)
        self.m_sys.AddBody(self.m_body)
        

        self.m_glyphs = chrono.ChGlyphs()
        self.m_glyphs.SetGlyphsSize(0.004)
        self.m_glyphs.SetZbufferHide(True)
        self.m_glyphs.SetDrawMode(chrono.ChGlyphs.GLYPH_POINT)
        self.m_body.AddAsset(self.m_glyphs)


    def Update(self):
        m_points = []
        direc = self.m_origin.GetA().Get_A_Zaxis()
        nx = round(self.m_dims[0]/self.m_spacing)
        ny = round(self.m_dims[1]/self.m_spacing)
        for ix in range(nx):
            for iy in range(ny):
                x_local = -0.5 * self.m_dims[0] + ix * self.m_spacing
                y_local = -0.5 * self.m_dims[1] + iy * self.m_spacing
                from_vec = self.m_origin.TransformPointLocalToParent(chrono.ChVectorD(x_local, y_local, 0.0))
                to = from_vec + direc * 100
                result = chrono.ChRayhitResult()
                self.m_sys.GetCollisionSystem().RayHit(from_vec, to, result)
                if (result.hit):
                    m_points.append(result.abs_hitPoint)
            
        
        
        self.m_glyphs.Reserve(0)
        for point_id in range(m_points.size()) :
            self.m_glyphs.SetGlyphPoint(point_id, m_points[point_id], chrono.ChColor(1, 1, 0))
        


# =============================================================================
class RobosimianSCM_Env(ChronoBaseEnv):
    def __init__(self, render = False):
        self.render_active = render
        ChronoBaseEnv.__init__(self)
        self.info =  {"timeout": 100000.0}
        self.Xtarg = 100
        self.Ytarg = 0
        self.d_old = np.linalg.norm( [self.Ytarg , self.Xtarg ] )
        low = np.full(76, -1000)
        high = np.full(76, 1000)
        self.observation_space = spaces.Box(low, high, dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(28,), dtype=np.float32)
        chrono.SetChronoDataPath('/home/simonebenatti/codes/radu-chrono/chrono/data/')
        self.time_step = 1e-3
        
        # Drop the robot on rigid terrain
        self.drop = True
        
        # Robot locomotion mode
        self.mode = robosimian.LocomotionMode_WALK
        
        # Contact method (system type)
        self.contact_method = chrono.ChMaterialSurfaceSMC()
        
        # Phase durations
        self.duration_pose = .5          # Interval to assume initial pose
        self.duration_settle_robot = 0.25  # Interval to allow robot settling on terrain
        self.duration_sim = 10            # Duration of actual locomotion simulation
        
        # Output frequencies
        self.output_fps = 100
        self.render_fps = 60
        
        # Output directories
        self.out_dir = "./ROBOSIMIAN_RIGID"
        self.pov_dir = self.out_dir + "/POVRAY"
        self.img_dir = self.out_dir + "/IMG"
        
        # POV-Ray and/or IMG output
        self.data_output = True
        self.povray_output = False
        self.image_output = False
        
        self.LimbList = [robosimian.FL, robosimian.FR, robosimian.RL, robosimian.RR]
        # BlockedWheel : wheel ang clamped to 0
        self.max_ang =     [0.8, 2.4, 1.8, 2.0, 1.8, 2.0, 2.9, 0]
        self.max_omegas =  [0.53, 0.58, 0.80, 0.69, 0.43, 0.69, 0.66, 1.18]
        self.max_torques = [80, 152, 240, 116, 110, 107, 8.2, 6.7]
        
        self.render_setup = False
        
                # =============================================================================
        
        # ------------
        # Timed events
        # ------------
        
        self.time_create_terrain = self.duration_pose                       # create terrain after robot assumes initial pose
        self.time_start = self.time_create_terrain +self. duration_settle_robot  # start actual simulation after robot settling
        self.time_end = self.time_start + self.duration_sim                      # end simulation after specified duration
        
        # -------------
        # Create system
        # -------------
        
        if  isinstance(self.contact_method, chrono.ChMaterialSurfaceNSC):
                self.my_sys = chrono.ChSystemNSC()
                chrono.ChCollisionModel.SetDefaultSuggestedEnvelope(0.001)
                chrono.ChCollisionModel.SetDefaultSuggestedMargin(0.001)
        
        if  isinstance(self.contact_method, chrono.ChMaterialSurfaceSMC):
        		self.my_sys = chrono.ChSystemSMC()
        
        
        
        self.my_sys.SetSolverMaxIterations(200)
        #my_sys.SetMaxiter(200)
        #my_sys.SetMaxItersSolverStab(1000)
        self.my_sys.SetSolverType(chrono.ChSolver.Type_BARZILAIBORWEIN)
        
        self.my_sys.Set_G_acc(chrono.ChVectorD(0, 0, -9.8))
        
        self.first_time = True
        
    def CreateTerrain(self, length, width, height, offset) :
        # Deformable terrain properties (LETE sand)
        ##Kphi = 5301e3    # Bekker Kphi
        ##Kc = 102e3       # Bekker Kc
        ##n = 0.793        # Bekker n exponent
        ##coh = 1.3e3      # Mohr cohesive limit (Pa)
        ##phi = 31.1       # Mohr friction limit (degrees)
        ##K = 1.2e-2       # Janosi shear coefficient (m)
        ##E_elastic = 2e8  # Elastic stiffness (Pa/m), before plastic yeld
        ##damping = 3e4    # Damping coefficient (Pa*s/m)

        # Deformable terrain properties (CDT FGS dry - 6/29/2018)
        Kphi = 6259.1e3  # Bekker Kphi
        Kc = 5085.6e3  # Bekker Kc
        n = 1.42  # Bekker n exponent
        coh = 1.58e3  # Mohr cohesive limit (Pa)
        phi = 34.1  # Mohr friction limit (degrees)
        K = 22.17e-3  # Janosi shear coefficient (m)
        E_elastic = 2e8  # Elastic stiffness (Pa/m), before plastic yeld
        damping = 3e4  # Damping coefficient (Pa*s/m)

        # Initial number of divisions per unit (m)
        factor = 8

        # Mesh divisions
        ndivX = math.ceil(length * factor)
        ndivY = math.ceil(width * factor)

        terrain = vehicle.SCMDeformableTerrain(self.robot.GetSystem())
        terrain.SetPlane(
            chrono.ChCoordsysD(chrono.ChVectorD(length / 2 - offset, 0, 0), chrono.Q_from_AngX(chrono.CH_C_PI_2)))
        terrain.SetSoilParametersSCM(Kphi, Kc, n, coh, phi, K, E_elastic, damping)
        terrain.SetPlotType(vehicle.SCMDeformableTerrain.PLOT_SINKAGE, 0, 0.15)
        terrain.Initialize(height, length, width, ndivX, ndivY)
        terrain.SetAutomaticRefinement(True)
        terrain.SetAutomaticRefinementResolution(1.0 / 64)

        # Enable moving patch feature
        terrain.EnableMovingPatch(self.robot.GetChassisBody(), chrono.ChVectorD(0, 0, 0), 3.0, 2.0)

        return terrain
    
    
    def SetContactProperties(self):
        friction = 0.8
        Y = 1e7
        cr = 0.0
    
        CM = self.robot.GetSystem().GetContactMethod()
        
        #if  isinstance(CM, chrono.ChMaterialCompositeNSC):
        if  CM == 0:
            self.robot.GetSledBody().GetMaterialSurfaceNSC().SetFriction(friction)
            self.robot.GetWheelBody(robosimian.FR).GetMaterialSurfaceNSC().SetFriction(friction)
            self.robot.GetWheelBody(robosimian.FL).GetMaterialSurfaceNSC().SetFriction(friction)
            self.robot.GetWheelBody(robosimian.RR).GetMaterialSurfaceNSC().SetFriction(friction)
            self.robot.GetWheelBody(robosimian.RL).GetMaterialSurfaceNSC().SetFriction(friction)
    
            
        elif  CM == 1:
        #elif  isinstance(CM, chrono.ChMaterialCompositeSMC):
            self.robot.GetSledBody().GetMaterialSurfaceSMC().SetFriction(friction)
            self.robot.GetWheelBody(robosimian.FR).GetMaterialSurfaceSMC().SetFriction(friction)
            self.robot.GetWheelBody(robosimian.FL).GetMaterialSurfaceSMC().SetFriction(friction)
            self.robot.GetWheelBody(robosimian.RR).GetMaterialSurfaceSMC().SetFriction(friction)
            self.robot.GetWheelBody(robosimian.RL).GetMaterialSurfaceSMC().SetFriction(friction)
    
            self.robot.GetSledBody().GetMaterialSurfaceSMC().SetYoungModulus(Y)
            self.robot.GetWheelBody(robosimian.FR).GetMaterialSurfaceSMC().SetYoungModulus(Y)
            self.robot.GetWheelBody(robosimian.FL).GetMaterialSurfaceSMC().SetYoungModulus(Y)
            self.robot.GetWheelBody(robosimian.RR).GetMaterialSurfaceSMC().SetYoungModulus(Y)
            self.robot.GetWheelBody(robosimian.RL).GetMaterialSurfaceSMC().SetYoungModulus(Y)
    
            self.robot.GetSledBody().GetMaterialSurfaceSMC().SetRestitution(cr)
            self.robot.GetWheelBody(robosimian.FR).GetMaterialSurfaceSMC().SetRestitution(cr)
            self.robot.GetWheelBody(robosimian.FL).GetMaterialSurfaceSMC().SetRestitution(cr)
            self.robot.GetWheelBody(robosimian.RR).GetMaterialSurfaceSMC().SetRestitution(cr)
            self.robot.GetWheelBody(robosimian.RL).GetMaterialSurfaceSMC().SetRestitution(cr)
    
        else :    
            raise Exception('Unvalid contact method')
        

        
    
    def get_joint_state(self):
        angles = []
        omegas = []
        #torques = []
        for limb in self.LimbList:
            limbangles = self.robot.GetMotorAngles(limb)
            angles.append(limbangles)
            limbomegas = self.robot.GetMotorOmegas(limb)
            omegas.append(limbomegas)
            #limbtorques = self.robot.GetMotorTorques(limb)
            #torques.append(limbtorques)
        states = [angles, omegas]#, torques]
        return states

    def get_joint_torques(self):

        torques = []
        for limb in self.LimbList:

            limbtorques = self.robot.GetMotorTorques(limb)
            torques.append(limbtorques)

        return torques
    
    def Actuate(self, arr):
        assert arr.size == 32
        arr_rs = arr.reshape([4,8])
        actuation = tuple(map(tuple, arr_rs))
        return actuation
    def reset(self):
        print('RESETTING')
        self.my_sys.Clear()
        self.my_sys.SetChTime(0.0)
        
        if not self.first_time:
            del self.robot
            del self.driver
            del self.ground
            if self.render_active:
                self.myapplication.GetDevice().closeDevice()
                del self.myapplication
                self.render_setup = False
            

        # -----------------------
        # Create RoboSimian robot
        # -----------------------
        
        self.robot = robosimian.RoboSimian(self.my_sys, True, True)
        
        # Set output directory
        
        #self.robot.SetOutputDirectory(self.out_dir)
        
        # Set actuation mode for wheel motors
        
        ##robot.SetMotorActuationMode(robosimian::ActuationMode::ANGLE)
        
        # Control collisions (default: True for sled and wheels only)
        
        ##robot.SetCollide(robosimian::CollisionFlags::NONE)
        ##robot.SetCollide(robosimian::CollisionFlags::ALL)
        ##robot.SetCollide(robosimian::CollisionFlags::LIMBS)
        ##robot.SetCollide(robosimian::CollisionFlags::CHASSIS | robosimian::CollisionFlags::WHEELS)
        
        # Set visualization modes (default: all COLLISION)

        self.robot.SetVisualizationTypeChassis(robosimian.VisualizationType_VisualizationType_MESH)
        self.robot.SetVisualizationTypeSled(robosimian.VisualizationType_VisualizationType_MESH)
        self.robot.SetVisualizationTypeLimbs(robosimian.VisualizationType_VisualizationType_MESH)
        
        # Initialize Robosimian robot
        
        ##robot.Initialize(ChCoordsys<>(chrono.ChVectorD(0, 0, 0), QUNIT))
        self.robot.Initialize(chrono.ChCoordsysD(chrono.ChVectorD(0, 0, 0), chrono.Q_from_AngX(chrono.CH_C_PI)))
        
        # -----------------------------------
        # Create a driver and attach to robot
        # -----------------------------------
        
        if self.mode == robosimian.LocomotionMode_WALK:
        		self.driver = robosimian.Driver(
        			"",                                                           # start input file
        			chrono.GetChronoDataFile("robosimian/actuation/walking_cycle.txt"),  # cycle input file
        			"",                                                           # stop input file
        			True)
        elif self.mode == robosimian.LocomotionMode_SCULL:
        		self.driver = robosimian.Driver(
        			chrono.GetChronoDataFile("robosimian/actuation/sculling_start.txt"),   # start input file
        			chrono.GetChronoDataFile("robosimian/actuation/sculling_cycle2.txt"),  # cycle input file
        			chrono.GetChronoDataFile("robosimian/actuation/sculling_stop.txt"),    # stop input file
        			True)
        
        elif self.mode == robosimian.LocomotionMode_INCHWORM:
        		self.driver = robosimian.Driver(
        			chrono.GetChronoDataFile("robosimian/actuation/inchworming_start.txt"),  # start input file
        			chrono.GetChronoDataFile("robosimian/actuation/inchworming_cycle.txt"),  # cycle input file
        			chrono.GetChronoDataFile("robosimian/actuation/inchworming_stop.txt"),   # stop input file
        			True)
        
        elif self.mode == robosimian.LocomotionMode_DRIVE:
        		self.driver = robosimian.Driver(
        			chrono.GetChronoDataFile("robosimian/actuation/driving_start.txt"),  # start input file
        			chrono.GetChronoDataFile("robosimian/actuation/driving_cycle.txt"),  # cycle input file
        			chrono.GetChronoDataFile("robosimian/actuation/driving_stop.txt"),   # stop input file
        			True)
        else:
            raise('Unvalid contact method')
        
        self.cbk = robosimian.RobotDriverCallback(self.robot)
        self.driver.RegisterPhaseChangeCallback(self.cbk)
        
        self.driver.SetTimeOffsets(self.duration_pose, self.duration_settle_robot)
        self.robot.SetDriver(self.driver)
        self.driver.SetDrivingMode(True)
        
        # -------------------------------
        # Cast rays into collision models
        # -------------------------------
        
        ##Raycaster self.caster(my_sys, ChFrame<>(chrono.ChVectorD(2, 0, -1), Q_from_AngY(-CH_C_PI_2)), ChVector2<>(2.5, 2.5),
        #/ 0.02)
        self.caster = Raycaster(self.my_sys, chrono.ChFrameD(chrono.ChVectorD(0, -2, -1), chrono.Q_from_AngX(-chrono.CH_C_PI_2)), [2.5, 2.5], 0.02)

        
        # -----------------------------
        # Initialize output directories
        # -----------------------------
        try:
            os.mkdir(self.out_dir)
        except:
            print('could not create out_dir')    
        
        if (self.povray_output) :
            os.mkdir(self.pov_dir)
            
        if (self.image_output) :
            os.mkdir(self.img_dir)
        
        # ---------------------------------
        # Run simulation for specified time
        # ---------------------------------
        
        #self.output_steps = math.ceil((1.0 / self.output_fps) / self.time_step)
        #self.render_steps = math.ceil((1.0 / self.render_fps) / self.time_step)
        #sim_frame = 0
        #self.output_frame = 0
        #render_frame = 0
        
        terrain_created = False
        """
        angle_save = np.zeros([8,1])
        omega_save = np.zeros([8,1])
        torque_save = np.zeros([8,1])
        saved = False
        """
        while ( self.my_sys.GetChTime() < self.duration_pose + self.duration_settle_robot) :
        	##self.caster.Update()
            if (self.drop and not terrain_created and self.my_sys.GetChTime() > self.time_create_terrain) :
        		# Set terrain height
                z = self.robot.GetWheelPos(robosimian.FR).z - 0.15
        
        		# Rigid terrain parameters
                length = 8
                width = 2
        
                # Create terrain
                #hdim = chrono.ChVectorD(length / 2, width / 2, 0.1)
                #loc = chrono.ChVectorD(length / 4, 0, z - 0.1)
                self.ground = self.CreateTerrain(length, width, z, length / 4)
                self.SetContactProperties()
                self.robot.GetChassisBody().SetBodyFixed(False)
                terrain_created = True
                if self.render_active:
                    self.myapplication.AssetBindAll()
                    self.myapplication.AssetUpdateAll()
                print('Terrain Created')
        	
        

            """
            if (data_output and sim_frame % output_steps == 0) :
                robot.Output()
                if  my_sys.GetChTime() > time_create_terrain + 10: 
                    #angle_save = np.concatenate([np.amax(angle_save,1).reshape((8, 1)), np.amin(angle_save,1).reshape((8, 1))],1)
                    angle_save.dump(out_dir+'/angles.dat')
                    #omega_save = np.concatenate([np.amax(omega_save,1).reshape((8, 1)), np.amin(omega_save,1).reshape((8, 1))],1)
                    omega_save.dump(out_dir+'/omegas.dat')
                    #torque_save = np.concatenate([np.amax(torque_save,1).reshape((8, 1)), np.amin(torque_save,1).reshape((8, 1))],1)
                    torque_save.dump(out_dir+'/torques.dat')
                    saved = True
            
        # Output POV-Ray date and/or snapshot images
            if(sim_frame % self.render_steps == 0) :
                if (povray_output) :
                    filename = self.pov_dir + '/data_' + str(render_frame + 1) +'04d.dat' 
                    chrono.WriteShapesPovray(my_sys, filename)
        		
                if (self.image_output) :
                    filename = self.img_dir + '/img_' + str(render_frame + 1) +'04d.jpg' 
                    image = self.myapplication.GetVideoDriver().createScreenShot()
                    if (image) :
                        self.myapplication.GetVideoDriver().writeImageToFile(image, filename)
                        image.drop()
        			
        		
        
                render_frame += 1
        	
        
        	##time = my_sys.GetChTime()
        	##A = CH_C_PI / 6
        	##freq = 2
        	##val = 0.5 * A * (1 - std::cos(CH_C_2PI * freq * time))
        	##robot.Activate(robosimian::FR, "joint2", time, val)
        	##robot.Activate(robosimian::RL, "joint5", time, val)
            
            if my_sys.GetChTime() > time_create_terrain +1:
                state = get_joint_state(robot)
                angles = np.reshape(state[0], (4, 8)).T
                angle_save = np.concatenate([angle_save, angles],1)
                omegas = np.reshape(state[1], (4, 8)).T
                omega_save = np.concatenate([omega_save, omegas],1)
                torques = np.reshape(state[2], (4, 8)).T
                torque_save = np.concatenate([torque_save, torques],1)
            """
            #x0 = np.asarray(self.driver.GetActuation()).flatten()
            #delta_x = (np.random.rand(32)-0.5)*2*7e-3
            #actuation = self.Actuate(x0+delta_x)
            #self.driver.SetActuation(actuation)
            if self.render_active:
                self.render()
            self.robot.DoStepDynamics(self.time_step)
            """
            if saved:
                self.myapplication.GetDevice().closeDevice()
                del self.myapplication
                break
        	##if (my_sys.GetNcontacts() > 0) {
        	##    robot.ReportContacts()
        	##}
            """
            #sim_frame += 1
        self.numsteps = 0
        self.isdone = False
        self.first_time =False
        return self.get_ob()
        

    def step(self, action):
        self.numsteps += 1
        # Getting last joints rotations
        x0 = np.asarray(self.driver.GetActuation()).flatten()
        # Force wheel speed to 0
        # BlockedWheels: set wheels rot to 0
        for i, el in enumerate(action):
            if i % 8 == 7:
                action = np.insert(action, i, 0)
        # BlockedWheels: set last wheel rot to 0
        action = np.concatenate([action, [0]])
        self.actions = action
        # actions are speeds, next angles = speed + timestep
        self.delta_angles = action * self.time_step
        
        for (i, ang) in enumerate(self.delta_angles):
             self.delta_angles[i] = ang * self.max_omegas[i%8]
        # Un-clamped new angles
        uc_ang = x0 + self.delta_angles
        # Clamping angle values in limits
        for (i, ang) in enumerate(uc_ang):
            uc_ang[i] = np.clip(ang, -self.max_ang[i%8], self.max_ang[i%8])
        # Resahaping actions 32 -> [4,8]
        actuation = self.Actuate(uc_ang)
        # Pass new angle setpoints
        self.driver.SetActuation(actuation)
        
        self.robot.DoStepDynamics(self.time_step)
        # to get height
        #print(self.robot.GetChassisBody().GetPos().z)
        
        
        obs= self.get_ob()
        rew = self.calc_rew()    
        
        self.is_done()
        return obs, rew, self.isdone, self.info     


    def render(self, mode='human'):
        if not self.render_setup:
                assert self.render_active
                self.myapplication = chronoirr.ChIrrApp(self.my_sys, "RoboSimian - Rigid terrain",
    									chronoirr.dimension2du(800, 600))
                self.myapplication.AddTypicalLogo(chrono.GetChronoDataPath() + 'logo_pychrono_alpha.png')
                self.myapplication.AddTypicalSky()
                self.myapplication.AddTypicalCamera(chronoirr.vector3df(1, -2.75, 0.2), chronoirr.vector3df(1, 0, 0))
                self.myapplication.AddTypicalLights(chronoirr.vector3df(100, 100, 100), chronoirr.vector3df(100, -100, 80))
                self.myapplication.AddLightWithShadow(chronoirr.vector3df(10, -6, 3), chronoirr.vector3df(0, 0, 0), 3, -10, 10,
                							   40, 512)
                
                
                self.myapplication.AssetBindAll()
                self.myapplication.AssetUpdateAll()
                self.render_setup = True
        self.myapplication.GetDevice().run()
        self.myapplication.BeginScene(True, True, chronoirr.SColor(255, 140, 161, 192))
        self.myapplication.DrawAll()
        self.myapplication.EndScene()
        if mode=='rgb_array':
            return np.zeros((32,32,3))

    def get_ob(self):
        self.chassis_pos = [self.robot.GetChassisBody().GetPos().x, self.robot.GetChassisBody().GetPos().y, self.robot.GetChassisBody().GetPos().z]
        vel = self.robot.GetChassisBody().GetRot().RotateBack(self.robot.GetChassisBody().GetPos_dt())
        self.chassis_v   = [vel.x,vel.y, vel.z]
        rot = self.robot.GetChassisBody().GetRot().Q_to_Euler123()
        self.chassis_rot = [rot.x, rot.y, rot.z]
        omega = self.robot.GetChassisBody().GetWvel_loc()
        self.chassis_omega = [ omega.x, omega.y, omega.z ]
        joints = self.get_joint_state()
        arr1 = np.asarray([self.chassis_pos +  self.chassis_rot + self.chassis_v + self.chassis_omega]).flatten()
        arr2 = np.asarray([joints]).flatten()
        return np.concatenate([arr1,arr2])
        
    def is_done(self):
        if ( self.robot.GetChassisBody().GetPos().z < -0.5 or self.numsteps *self.time_step>15):
                 self.isdone = True
                 
    def calc_rew(self):
              
              electricity_cost     = -.5    # cost for using motors -- this parameter should be carefully tuned against reward for making progress, other values less improtant
              #stall_torque_cost    = -0.1    # cost for running electric current through a motor even at zero rotational speed, small

              #joints_at_limit_cost = -0.2    # discourage stuck joints
              
              power_cost  = electricity_cost  * np.linalg.norm( self.actions) #float(np.abs(self.ac*self.q_dot_mot).mean())  # let's assume we have DC motor with controller, and reverse current braking. BTW this is the formula of motor power
              #Reduced stall cost to avoid joints at limit
              torque_cost = 0
              torques = np.asarray(self.get_joint_torques()).flatten()
              for (i, torque) in enumerate(torques):
                  if (abs(torque) > self.max_torques[i%8]):
                      torque_cost += -1
              #self.alive_bonus =  +1 if self.body_abdomen.GetContactForce().Length() == 0 else -1
              progress = self.calc_progress()
              rew = 10*progress + 0.1*(power_cost) + torque_cost
              return rew
    def calc_progress(self):
              d = np.linalg.norm( [self.Ytarg - self.robot.GetChassisBody().GetPos().y, self.Xtarg - self.robot.GetChassisBody().GetPos().x] )
              progress = -(d - self.d_old )/self.time_step
              self.d_old = d
              return progress     
