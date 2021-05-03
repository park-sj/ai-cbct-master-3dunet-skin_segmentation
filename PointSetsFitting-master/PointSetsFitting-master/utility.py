import numpy as np
import point_sets_fitting
import os
import glob

Result_path = './Result/'

def calculate_fitting_error(set_a: np.ndarray, set_b: np.ndarray) -> float:
    set_a = point_sets_fitting.to_homogeneous_repr(set_a)
    set_b = point_sets_fitting.to_homogeneous_repr(set_b)

    diff_set_b = set_a - set_b

    accum_norm = 0
    for dif_vec in diff_set_b.transpose():
        accum_norm += np.linalg.norm(dif_vec)

    return accum_norm / diff_set_b.shape[1]


def return_obj_path(dir_path: list) -> np.ndarray:

    dir_names = os.listdir(dir_path)

    obj_path_name = []

    for name in dir_names:
        src = os.path.join(dir_path, name)
        obj_path_name = obj_path_name + [file for file in os.listdir(src) if file.endswith(".obj")] + glob.glob(src+"\*.obj")

    obj_path_name = np.array(obj_path_name).reshape(-1, 2)

    return obj_path_name

def create_obj_file(vertices_set: np.ndarray, faces_set: np.ndarray, file_name: np.str) -> None:

    vertices_set = np.round(vertices_set, 3)

    obj_file = open(Result_path + file_name + '_transformed.obj', 'w')

    for column_index in range(vertices_set.shape[1]):
        obj_file.write("v ")
        for row_index in range(vertices_set.shape[0]):
            obj_file.write("{0} ".format(vertices_set[row_index][column_index]))
        obj_file.write("\n")

    obj_file.write("\n"*2)

    for column_index in range(faces_set.shape[1]):
        obj_file.write("f ")
        for row_index in range(faces_set.shape[0]):
            obj_file.write("{0} ".format(faces_set[row_index][column_index]+1))
        obj_file.write("\n")

    obj_file.close()