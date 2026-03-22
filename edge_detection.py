import numpy as np

def identify_centroid(camera_matrix):
    """Calculates the center of mass of all oil pixels (1) in the local view."""
    rows, cols = np.where(camera_matrix > 0.5)
    if len(rows) > 0:
        return np.mean(cols), np.mean(rows)
    return None

def check_geometric_lock(camera_matrix):
    """
    Implements the 'perfect geometric edge' rule:
    Center pixel is oil (1) AND has at least one water neighbor (0).
    """
    size = camera_matrix.shape[0]
    half = size // 2
    
    current_cell = camera_matrix[half, half]
    if current_cell < 0.5:
        return False
        
    # Check 3x3 vicinity for any water
    vicinity = camera_matrix[max(0, half-1):min(size, half+2), 
                             max(0, half-1):min(size, half+2)]
    return np.any(vicinity < 0.5)
