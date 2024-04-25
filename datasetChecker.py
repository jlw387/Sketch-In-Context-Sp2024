import os

def search_for_file(parent_folder, file_name):
    folders_without_file = []
    for folder_name in os.listdir(parent_folder):
        folder_path = os.path.join(parent_folder, folder_name)
        if os.path.isdir(folder_path):
            if file_name not in os.listdir(folder_path):
                folders_without_file.append(folder_name)
    return folders_without_file

if __name__ == "__main__":
    parent_folder = "./SDFDatasets/Gen_1712888438"
    file_name = "surface_points.csv"
    folders_without_file = search_for_file(parent_folder, file_name)
    print("Folders without '{}' file:".format(file_name))
    for folder in folders_without_file:
        print(folder)