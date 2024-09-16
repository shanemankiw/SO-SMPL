import os

folder_path = (
    "outputs/human2human/dataset_man_Make_his_hair_ginger@20230726-081810/save"
)

# Get a list of all the PNG files in the folder
png_files = [file for file in os.listdir(folder_path) if file.endswith(".png")]

# Iterate over the PNG files and rename them with leading zeros
for i, png_file in enumerate(png_files):
    # Extract the number from the filename
    number = int(png_file.split("-")[0][2:])
    # Generate the new filename with leading zeros
    new_filename = f"it{number:05d}-{0}.png"
    # Rename the file
    os.rename(
        os.path.join(folder_path, png_file), os.path.join(folder_path, new_filename)
    )
