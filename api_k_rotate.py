
import random
import google.generativeai as genai

API_KEYS = ["AIzaSyD9QwMHGOU3daSB-Gldijp70ojpnbcjzgo"]



def create_rotator(keys, rotation_interval=3):
    return {
        "keys": keys,
        "rotation_interval": rotation_interval,
        "current_index": 0,
        "request_count": 0
    }

def rotate(rotator):
    rotator["request_count"] += 1
    if rotator["request_count"] % rotator["rotation_interval"] == 0:
        rotator["current_index"] = (rotator["current_index"] + 1) % len(rotator["keys"])
    return rotator["keys"][rotator["current_index"]]

# Create a global rotator
api_rotator = create_rotator(API_KEYS)

def get_next_api_key():
    return rotate(api_rotator)