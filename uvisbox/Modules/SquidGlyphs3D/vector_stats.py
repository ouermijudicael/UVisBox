 
import numpy as np
from sklearn.decomposition import PCA
from uvisbox.Core.BandDepths.vector_depths import *
# from uvisbox.Core.BandDepths.vector_depths import calculate_spread_3D



def getDirectionalVariations(vectors, depths, depth_threshold, min_vectors, median_vectors, max_vectors):
    """
    Compute the directional variation of the vectors.
    
    Parameters:
    ----------
    positions : numpy.ndarray
        Array of shape (num_points, 3) where the last dimension contains the x, y, and z coordinates of the positions.
    vectors : numpy.ndarray
        Array of shape (num_points, num_ensemble_members, 3) where the last dimension contains the vector components.
    median_vectors : numpy.ndarray
        Array of shape (num_points, 3) where the last dimension contains the median vector components.
    max_vectors : numpy.ndarray
        Array of shape (num_points, 3) where the last dimension contains the max vector components.
    min_vectors : numpy.ndarray
        Array of shape (num_points, 3) where the last dimension contains the min vector components.
    domain : numpy.ndarray
        Array of shape (3, 2) where the first dimension contains the x, y, and z domain limits.
    
    Returns:
    -------
    directional_variation : numpy.ndarray
        Array of shape (num_points, 4, 2) where the 4 represents the
        (pca variance, pca first component, pca second component, pca mean) and the 2 represents
        the x and y components of the pca components.
    """

    num_points = vectors.shape[0]
    num_ensemble_members = vectors.shape[1]
    local_median_vector = np.zeros(3)
    local_XX = np.zeros([num_ensemble_members, 2])
    directional_variation = np.zeros([num_points, 3, 2])

    for i_p in range(num_points):
        r = median_vectors[i_p][0]
        theta = median_vectors[i_p][1]
        phi = median_vectors[i_p][2]
        local_median_vector[0] = r*np.sin(theta)*np.cos(phi)
        local_median_vector[1] = r*np.sin(theta)*np.sin(phi)
        local_median_vector[2] = r*np.cos(theta)
        phi = -phi
        theta = -theta

        Ry = np.array([[np.cos(theta), 0, np.sin(theta)], [0, 1, 0], [-np.sin(theta), 0, np.cos(theta)]])
        Rz = np.array([[np.cos(phi), -np.sin(phi), 0], [np.sin(phi), np.cos(phi), 0], [0, 0, 1]])
        ii_m = 0
        for i_m in range(num_ensemble_members):
            if(depths[i_p][i_m] >= depth_threshold):
                tmp = vectors[i_p][i_m]
                vec = np.zeros(3)
                vec[0] = tmp[0]*np.sin(tmp[1])*np.cos(tmp[2])
                vec[1] = tmp[0]*np.sin(tmp[1])*np.sin(tmp[2])
                vec[2] = tmp[0]*np.cos(tmp[1])
                vec = np.dot(Ry, np.dot(Rz, vec))
                vec =  vec * (max_vectors[i_p][0]/(np.absolute(vec[2]) + 1.0e-10))
                local_XX[ii_m][0] = vec[0]
                local_XX[ii_m][1] = vec[1]
                ii_m = ii_m + 1
        local_X = local_XX[0:ii_m]
        n_local_points, n_local_features = local_X.shape
        pca = PCA(n_components=2)

        if n_local_points > 2 and n_local_features > 2: # ensures that we have enough points and features for PCA
            pca.fit(local_X)
            pca_components = pca.components_
            pca_mean = pca.mean_
            pca_variance = pca.explained_variance_
            v0_scale = pca_variance[0]
            v1_scale = pca_variance[1]
        else:
            pca_components = np.zeros((2, 2))
            v0_scale = 1.0
            v1_scale = 1.0

        if(np.absolute(v0_scale) < 1.0e-20): # ensure non-zero scale
            v0_scale = 1.0
            v1_scale = 1.0

        directional_variation[i_p][0][0] = v0_scale
        directional_variation[i_p][0][1] = np.maximum(v1_scale, 0.1*v0_scale)
        directional_variation[i_p][1][0] = pca_components[0][0]
        directional_variation[i_p][1][1] = pca_components[0][1]
        directional_variation[i_p][2][0] = pca_components[1][0]
        directional_variation[i_p][2][1] = pca_components[1][1]


    return directional_variation

