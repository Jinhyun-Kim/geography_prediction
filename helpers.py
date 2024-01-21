import socket

def get_ip_address():
    """Get the current IP address of the machine."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # Doesn't even have to be reachable
        s.connect(('10.255.255.255', 1))
        IP = s.getsockname()[0]
    except Exception:
        IP = '127.0.0.1'
    finally:
        s.close()
    return IP

import os
def has_write_permission(directory_path):
    """Check if the current user has write permission for the specified directory."""
    return os.access(directory_path, os.W_OK)