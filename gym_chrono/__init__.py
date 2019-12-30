import logging
from gym.envs.registration import register

register(
    id='chrono_pendulum-v0',
    entry_point='gym_chrono.envs:ChronoPendulum') # NAme of the CLASS after the colon

register(
    id='chrono_ant-v0',
    entry_point='gym_chrono.envs:ChronoAnt'
)

register(
    id='chrono_hexapod-v0',
    entry_point='gym_chrono.envs:ChronoHexapod'
    #timestep_limit=1000,
    #reward_threshold=1.0,
    #nondeterministic = True,
)

register(
    id='ChronoRacer3Reach-v0',
    entry_point='gym_chrono.envs:ChronoComauR3'
    #timestep_limit=1000,
    #reward_threshold=10.0,
    #nondeterministic = True,
)

register(
    id='RobosimianRigidEnv-v0',
    entry_point='gym_chrono.envs:RobosimianRigidEnv'
    #timestep_limit=1000,
    #reward_threshold=10.0,
    #nondeterministic = True,
)