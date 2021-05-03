import pywavefront
import numpy
import point_sets_fitting
import utility

if __name__ == '__main__':

    obj_path_name = utility.return_obj_path("C:/Users/user/Desktop/Mesh/InofitLab")

    target = pywavefront.Wavefront(obj_path_name[0][1], collect_faces=True)
    target_vertices = numpy.array(target.vertices, dtype='f').transpose()

    for template_path in obj_path_name:
        template = pywavefront.Wavefront(template_path[1], collect_faces=True)
        template_vertices = numpy.array(template.vertices, dtype='f').transpose()
        faces = numpy.array(template.mesh_list[0].faces).transpose()

        rigid_transformation, error = point_sets_fitting.point_sets_fitting(template_vertices, target_vertices)

        trainsformation_result = rigid_transformation[0:3, 0:3] @ template_vertices + rigid_transformation[0:3, 3].reshape(-1, 1)

        print("Original fitting Error of {0} = {1}".format(template_path[0], utility.calculate_fitting_error(template_vertices, target_vertices)))
        print("Transformed fitting Error {0}= {1}\n".format(template_path[0], utility.calculate_fitting_error(trainsformation_result, target_vertices)))

        utility.create_obj_file(trainsformation_result, faces, template_path[0])
