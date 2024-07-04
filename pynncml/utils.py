import torch


def get_working_device() -> torch.device:
    """
    Get the working device for the current session.
    """
    working_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Current Working Device is set to:" + str(working_device))
    return working_device
