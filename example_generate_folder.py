from Instance_Generator import Folder_Generator

# Speify path to save instances
output_folder = "example_folder/"

# Specify the parameters for each instance
N_list = [100] *50
K_list = [5] *50
L_list = [5] *50
beta_list = [2.0] *50

# Generate the folder
generator = Folder_Generator(output_folder, N_list, K_list, L_list, beta_list)
generator.generate_folder()


