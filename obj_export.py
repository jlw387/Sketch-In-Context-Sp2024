
# Assumes a list of 3-Tuples for both verts and faces
def obj_exporter(filepath, verts, faces, normals=[]):
    with open(filepath, 'w') as file:
        file.write("# Vertices\n")
        for vertex in verts:
            file.write("v %f %f %f\n" % (vertex[0],vertex[1],vertex[2]))

        file.write("# Normals\n")
        for normal in normals:
            file.write("vn %f %f %f\n" % (normal[0],normal[1],normal[2]))

        file.write("\n# Faces\n")
        for face in faces:
            file.write("f %d %d %d\n" % (face[0]+1,face[1]+1,face[2]+1))


def test_export():
    test_verts = [(0.0,0.0,0.0), (1.0,0.0,0.0), (0.0,1.0,0.0), (0.0,0.0,1.0)]
    test_faces = [(1,3,2), (1,2,4), (1,4,3), (2,3,4)]

    obj_exporter("testFile.obj", test_verts, test_faces)