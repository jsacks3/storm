try:
    from  isaacgym import gymapi
    from isaacgym import gymutil
except Exception:
    print("ERROR: gym not loaded, this is okay when generating doc")

from .sim_robot import RobotSim

class CartpoleSim(RobotSim):
    def _set_dof_properties(self, env_handle, robot_handle, robot_dof_props):
        robot_dof_props['driveMode'].fill(gymapi.DOF_MODE_EFFORT)
        robot_dof_props['stiffness'].fill(0.0)
        robot_dof_props['damping'].fill(0.0)
        self.gym.set_actor_dof_properties(env_handle, robot_handle, robot_dof_props)