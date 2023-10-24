import os
import shutil

# Folder path where the images are located
folder_path = 'cifar10/train/truck'

# List all files in the folder
files = os.listdir(folder_path)

counter = 1001

# Iterate through the files in the folder
for filename in files:
    # Define the new name you want to assign
    new_name = f'truck_{counter}'  # You can adjust the renaming logic

    # Create the full path for the old and new file names
    old_file_path = os.path.join(folder_path, filename)
    new_file_path = os.path.join(folder_path, new_name)

    # Rename the file without moving it
    try:
        shutil.move(old_file_path, new_file_path)
        counter += 1;
        print(f'Renamed: {filename} to {new_name}')
    except Exception as e:
        print(f'Error renaming {filename}: {str(e)}')

# The images remain in the same folder with the new names
