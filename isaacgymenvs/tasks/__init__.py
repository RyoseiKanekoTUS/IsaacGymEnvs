from .franka_cabinet import FrankaCabinet
from .door_hook import DoorHook
from .ur3_door_hook import UR3_DoorHook




# Mappings from strings to environments
isaacgym_task_map = {
    "FrankaCabinet": FrankaCabinet,
    "DoorHook": DoorHook,
    "DoorHookSAC": DoorHook,
    "UR3_DoorHook": UR3_DoorHook
    
}
