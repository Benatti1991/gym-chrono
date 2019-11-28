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
class RoboSimianRigidEnv(ChronoBaseEnv):
    def __init__(self):
        ChronoBaseEnv.__init__(self)
        low = np.full(69, -1000)
        high = np.full(69, 1000)
        self.observation_space = spaces.Box(low, high, dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(28,), dtype=np.float32)
        chrono.SetChronoDataPath('C:/codes/ChronoRobosimian/data/')
        self.time_step = 1e-3
        
        # Drop the robot on rigid terrain
        self.drop = True
        
        # Robot locomotion mode
        self.mode = robosimian.LocomotionMode_WALK
        
        # Contact method (system type)
        self.contact_method = chrono.ChMaterialSurfaceSMC()
        
        # Phase durations
        self.duration_pose = 1.0          # Interval to assume initial pose
        self.duration_settle_robot = 0.5  # Interval to allow robot settling on terrain
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
        self.max_ang =     [0.8, 2.4, 1.8, 2.0, 1.8, 2.0, 2.9, None]
        self.max_omegas =  [0.53, 0.58, 0.80, 0.69, 0.43, 0.69, 0.66, 1.18]
        self.max_torques = [80, 152, 240, 116, 110, 107, 8.2, 6.7]
        
        self.render_setup = False
        
    def CreateTerrain(self, length, width, height, offset) :
        ground = self.my_sys.NewBody()
        ground.SetBodyFixed(True)
        ground.SetCollide(True)
    
        ground.GetCollisionModel().ClearModel()
        ground.GetCollisionModel().AddBox(length / 2, width / 2, 0.1, chrono.ChVectorD(offset, 0, height - 0.1))
        ground.GetCollisionModel().BuildModel()
    
        box = chrono.ChBoxShape()
        box.GetBoxGeometry().Size = chrono.ChVectorD(length / 2, width / 2, 0.1)
        box.GetBoxGeometry().Pos = chrono.ChVectorD(offset, 0, height - 0.1)
        ground.AddAsset(box)
    
        texture = chrono.ChTexture()
        texture.SetTextureFilename(chrono.GetChronoDataFile("pinkwhite.png"))
        texture.SetTextureScale(10 * length, 10 * width)
        ground.AddAsset(texture)
    
        ground.AddAsset(chrono.ChColorAsset(0.8, 0.8, 0.8))
    
        self.my_sys.AddBody(ground)
    
        return ground
    
    
    def SetContactProperties(robot):
        friction = 0.8
        Y = 1e7
        cr = 0.0
    
        CM = robot.GetSystem().GetContactMethod()
        
        #if  isinstance(CM, chrono.ChMaterialCompositeNSC):
        if  CM == 0:
            robot.GetSledBody().GetMaterialSurfaceNSC().SetFriction(friction)
            robot.GetWheelBody(robosimian.FR).GetMaterialSurfaceNSC().SetFriction(friction)
            robot.GetWheelBody(robosimian.FL).GetMaterialSurfaceNSC().SetFriction(friction)
            robot.GetWheelBody(robosimian.RR).GetMaterialSurfaceNSC().SetFriction(friction)
            robot.GetWheelBody(robosimian.RL).GetMaterialSurfaceNSC().SetFriction(friction)
    
            
        elif  CM == 1:
        #elif  isinstance(CM, chrono.ChMaterialCompositeSMC):
            robot.GetSledBody().GetMaterialSurfaceSMC().SetFriction(friction)
            robot.GetWheelBody(robosimian.FR).GetMaterialSurfaceSMC().SetFriction(friction)
            robot.GetWheelBody(robosimian.FL).GetMaterialSurfaceSMC().SetFriction(friction)
            robot.GetWheelBody(robosimian.RR).GetMaterialSurfaceSMC().SetFriction(friction)
            robot.GetWheelBody(robosimian.RL).GetMaterialSurfaceSMC().SetFriction(friction)
    
            robot.GetSledBody().GetMaterialSurfaceSMC().SetYoungModulus(Y)
            robot.GetWheelBody(robosimian.FR).GetMaterialSurfaceSMC().SetYoungModulus(Y)
            robot.GetWheelBody(robosimian.FL).GetMaterialSurfaceSMC().SetYoungModulus(Y)
            robot.GetWheelBody(robosimian.RR).GetMaterialSurfaceSMC().SetYoungModulus(Y)
            robot.GetWheelBody(robosimian.RL).GetMaterialSurfaceSMC().SetYoungModulus(Y)
    
            robot.GetSledBody().GetMaterialSurfaceSMC().SetRestitution(cr)
            robot.GetWheelBody(robosimian.FR).GetMaterialSurfaceSMC().SetRestitution(cr)
            robot.GetWheelBody(robosimian.FL).GetMaterialSurfaceSMC().SetRestitution(cr)
            robot.GetWheelBody(robosimian.RR).GetMaterialSurfaceSMC().SetRestitution(cr)
            robot.GetWheelBody(robosimian.RL).GetMaterialSurfaceSMC().SetRestitution(cr)
    
        else :    
            raise Exception('Unvalid contact method')
        
    
    def SetGroundContactProperties(ground, sys) :
        friction = 0.8
        Y = 1e7
        cr = 0.0
        CM = sys.GetContactMethod()
     
        if  CM == 0:
        #if  isinstance(CM, chrono.ChMaterialCompositeNSC):
            ground.GetMaterialSurfaceNSC().SetFriction(friction)
            
        elif  CM == 1:
        #elif  isinstance(CM, chrono.ChMaterialCompositeNSC):
            ground.GetMaterialSurfaceSMC().SetFriction(friction)
            ground.GetMaterialSurfaceSMC().SetYoungModulus(Y)
            ground.GetMaterialSurfaceSMC().SetRestitution(cr)
            
        else:
            
            raise Exception('Unvalid contact method')
        
    
    def get_state(self):
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
    
    def Actuate(arr):
        assert arr.size == 32
        arr_rs = arr.reshape([4,8])
        actuation = tuple(map(tuple, arr_rs))
        return actuation
    def reset(self):
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
        
        
        
        self.my_sys.SetMaxItersSolverSpeed(200)
        #my_sys.SetMaxiter(200)
        #my_sys.SetMaxItersSolverStab(1000)
        self.my_sys.SetSolverType(chrono.ChSolver.Type_BARZILAIBORWEIN)
        
        self.my_sys.Set_G_acc(chrono.ChVectorD(0, 0, -9.8))
        ##my_sys.Set_G_acc(ChVector<double>(0, 0, 0))
        
        # -----------------------
        # Create RoboSimian robot
        # -----------------------
        
        self.robot = robosimian.RoboSimian(self.my_sys, True, True)
        
        # Set output directory
        
        self.robot.SetOutputDirectory(self.out_dir)
        
        # Set actuation mode for wheel motors
        
        ##robot.SetMotorActuationMode(robosimian::ActuationMode::ANGLE)
        
        # Control collisions (default: True for sled and wheels only)
        
        ##robot.SetCollide(robosimian::CollisionFlags::NONE)
        ##robot.SetCollide(robosimian::CollisionFlags::ALL)
        ##robot.SetCollide(robosimian::CollisionFlags::LIMBS)
        ##robot.SetCollide(robosimian::CollisionFlags::CHASSIS | robosimian::CollisionFlags::WHEELS)
        
        # Set visualization modes (default: all COLLISION)
        
        ##robot.SetVisualizationTypeChassis(robosimian::VisualizationType::MESH)
        ##robot.SetVisualizationTypeLimb(robosimian::FL, robosimian::VisualizationType::COLLISION)
        ##robot.SetVisualizationTypeLimb(robosimian::FR, robosimian::VisualizationType::COLLISION)
        ##robot.SetVisualizationTypeLimb(robosimian::RL, robosimian::VisualizationType::COLLISION)
        ##robot.SetVisualizationTypeLimb(robosimian::RR, robosimian::VisualizationType::COLLISION)
        ##robot.SetVisualizationTypeLimbs(robosimian::VisualizationType::NONE)
        ##robot.SetVisualizationTypeChassis(robosimian::VisualizationType::MESH)
        ##robot.SetVisualizationTypeSled(robosimian::VisualizationType::MESH)
        ##robot.SetVisualizationTypeLimbs(robosimian::VisualizationType::MESH)
        
        # Initialize Robosimian robot
        
        ##robot.Initialize(ChCoordsys<>(chrono.ChVectorD(0, 0, 0), QUNIT))
        self.robot.Initialize(chrono.ChCoordsysD(chrono.ChVectorD(0, 0, 0), chrono.Q_from_AngX(chrono.CH_C_PI)))
        
        # -----------------------------------
        # Create a driver and attach to robot
        # -----------------------------------
        
        if self.mode == robosimian.LocomotionMode_WALK:
        		driver = robosimian.Driver(
        			"",                                                           # start input file
        			chrono.GetChronoDataFile("robosimian/actuation/walking_cycle.txt"),  # cycle input file
        			"",                                                           # stop input file
        			True)
        elif self.mode == robosimian.LocomotionMode_SCULL:
        		driver = robosimian.Driver(
        			chrono.GetChronoDataFile("robosimian/actuation/sculling_start.txt"),   # start input file
        			chrono.GetChronoDataFile("robosimian/actuation/sculling_cycle2.txt"),  # cycle input file
        			chrono.GetChronoDataFile("robosimian/actuation/sculling_stop.txt"),    # stop input file
        			True)
        
        elif self.mode == robosimian.LocomotionMode_INCHWORM:
        		driver = robosimian.Driver(
        			chrono.GetChronoDataFile("robosimian/actuation/inchworming_start.txt"),  # start input file
        			chrono.GetChronoDataFile("robosimian/actuation/inchworming_cycle.txt"),  # cycle input file
        			chrono.GetChronoDataFile("robosimian/actuation/inchworming_stop.txt"),   # stop input file
        			True)
        
        elif self.mode == robosimian.LocomotionMode_DRIVE:
        		driver = robosimian.Driver(
        			chrono.GetChronoDataFile("robosimian/actuation/driving_start.txt"),  # start input file
        			chrono.GetChronoDataFile("robosimian/actuation/driving_cycle.txt"),  # cycle input file
        			chrono.GetChronoDataFile("robosimian/actuation/driving_stop.txt"),   # stop input file
        			True)
        else:
            raise('Unvalid contact method')
        
        cbk = robosimian.RobotDriverCallback(self.robot)
        driver.RegisterPhaseChangeCallback(cbk)
        
        driver.SetTimeOffsets(self.duration_pose, self.duration_settle_robot)
        self.robot.SetDriver(driver)
        driver.SetDrivingMode(True)
        
        # -------------------------------
        # Cast rays into collision models
        # -------------------------------
        
        ##Raycaster self.caster(my_sys, ChFrame<>(chrono.ChVectorD(2, 0, -1), Q_from_AngY(-CH_C_PI_2)), ChVector2<>(2.5, 2.5),
        #/ 0.02)
        self.caster = Raycaster(self.my_sys, chrono.ChFrameD(chrono.ChVectorD(0, -2, -1), chrono.Q_from_AngX(-chrono.CH_C_PI_2)), [2.5, 2.5], 0.02)
        
        # -------------------------------
        # Create the visualization window
        # -------------------------------
        

        
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
        
        self.output_steps = math.ceil((1.0 / self.output_fps) / self.time_step)
        self.render_steps = math.ceil((1.0 / self.render_fps) / self.time_step)
        sim_frame = 0
        #self.output_frame = 0
        #render_frame = 0
        
        terrain_created = False
        """
        angle_save = np.zeros([8,1])
        omega_save = np.zeros([8,1])
        torque_save = np.zeros([8,1])
        saved = False
        """
        
        while ( my_sys.GetChTime() < self.duration_pose + self.duration_settle_robot) :
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
                ground = CreateTerrain(self.robot, length, width, z, length / 4)
                self.SetContactProperties(self.robot)
                self.SetGroundContactProperties(ground, self.my_sys)
                if (self.render_setup):
                    self.application.AssetBind(ground)
                    self.application.AssetUpdate(ground)
                robot.GetChassisBody().SetBodyFixed(False)
                terrain_created = True
        	
        

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
                    image = self.application.GetVideoDriver().createScreenShot()
                    if (image) :
                        self.application.GetVideoDriver().writeImageToFile(image, filename)
                        image.drop()
        			
        		
        
                render_frame += 1
        	
        
        	##time = my_sys.GetChTime()
        	##A = CH_C_PI / 6
        	##freq = 2
        	##val = 0.5 * A * (1 - std::cos(CH_C_2PI * freq * time))
        	##robot.Activate(robosimian::FR, "joint2", time, val)
        	##robot.Activate(robosimian::RL, "joint5", time, val)
            
            if my_sys.GetChTime() > time_create_terrain +1:
                state = get_state(robot)
                angles = np.reshape(state[0], (4, 8)).T
                angle_save = np.concatenate([angle_save, angles],1)
                omegas = np.reshape(state[1], (4, 8)).T
                omega_save = np.concatenate([omega_save, omegas],1)
                torques = np.reshape(state[2], (4, 8)).T
                torque_save = np.concatenate([torque_save, torques],1)
            """
            x0 = np.asarray(driver.GetActuation()).flatten()
            delta_x = (np.random.rand(32)-0.5)*2*7e-3
            actuation = self.Actuate(x0+delta_x)
            driver.SetActuation(actuation)
            self.robot.DoStepDynamics(self.time_step)
            """
            if saved:
                self.application.GetDevice().closeDevice()
                del self.application
                break
        	##if (my_sys.GetNcontacts() > 0) {
        	##    robot.ReportContacts()
        	##}
            """
            sim_frame += 1
        


    def render(self):
        if not self.render_setup:
                self.application = chronoirr.ChIrrApp(self.my_sys, "RoboSimian - Rigid terrain",
    									chronoirr.dimension2du(800, 600))
                self.application.AddTypicalLogo(chrono.GetChronoDataPath() + 'logo_pychrono_alpha.png')
                self.application.AddTypicalSky()
                self.application.AddTypicalCamera(chronoirr.vector3df(1, -2.75, 0.2), chronoirr.vector3df(1, 0, 0))
                self.application.AddTypicalLights(chronoirr.vector3df(100, 100, 100), chronoirr.vector3df(100, -100, 80))
                self.application.AddLightWithShadow(chronoirr.vector3df(10, -6, 3), chronoirr.vector3df(0, 0, 0), 3, -10, 10,
                							   40, 512)
                
                self.application.AssetBindAll()
                self.application.AssetUpdateAll()
                self.render_setup = True
        self.myapplication.GetDevice().run()
        self.application.BeginScene(True, True, chronoirr.SColor(255, 140, 161, 192))
        self.application.DrawAll()
        self.myapplication.EndScene()

