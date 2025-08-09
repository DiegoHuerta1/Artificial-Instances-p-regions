from .Single_Instance import Single_Instance_Generator
import os

class Folder_Generator:
    """ 
    Class to generate a folder full of graph instances for the p-regions problem.
    """

    def __init__(self, output_folder: str,
                mult_num_points: list[int], mult_num_regions: list[int], 
                mult_num_features: list[int], mult_beta_parameter: list[float]):
        """
        Class constructor

        Args:
            output_folder (str): Prefix path for all instances
            mult_num_points (list[int]): Number of points for each instance
            num_regions (list[int]): Number of regions for each instance
            num_features (list[int]): Number of features for each instance
            beta_parameter (list[float]): Beta parameter for each instance
        """

        # Verify that all parameters consider the same number of instances
        self.num_instances = len(mult_num_points)
        assert len(mult_num_regions) == self.num_instances
        assert len(mult_num_features) == self.num_instances
        assert len(mult_beta_parameter) == self.num_instances

        # Set parameters
        os.makedirs(output_folder, exist_ok=True)
        self.output_folder = output_folder
        self.N_list: list[int] = mult_num_points
        self.K_list: list[int] = mult_num_regions
        self.L_list: list[int] = mult_num_features
        self.beta_list: list[float] = mult_beta_parameter


    def generate_folder(self):
        """ 
        Main class to generate all instances in the folder
        """
        # Generate all instances
        for instance_idx in range(self.num_instances):
            generator = Single_Instance_Generator(num_points = self.N_list[instance_idx],
                                                  num_regions = self.K_list[instance_idx],
                                                  num_features = self.L_list[instance_idx], 
                                                  beta_parameter = self.beta_list[instance_idx],
                                                  seed = instance_idx)
            generator.generate_instance()
            generator.save_instance(self.output_folder + f"{instance_idx}.pkl")



