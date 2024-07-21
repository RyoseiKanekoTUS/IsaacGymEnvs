from .franka_cabinet import FrankaCabinet
from .door_test import DoorTest
from .door_test_2 import DoorTest_2
from .door_hook import DoorHook
from .ur3_door_hook import UR3_DoorHook




# Mappings from strings to environments
isaacgym_task_map = {
    "FrankaCabinet": FrankaCabinet,
    "DoorTest": DoorTest,
    "DoorTest_2": DoorTest_2,
    "DoorHook": DoorHook,
    "DoorHookSAC": DoorHook,
    "UR3_DoorHook": UR3_DoorHook
    
}
