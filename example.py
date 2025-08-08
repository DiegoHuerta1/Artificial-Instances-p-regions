from  Instance_Generator import Single_Instance_Generator

# Generate the instance
generator = Single_Instance_Generator(num_points = 20, num_regions = 3, num_features = 3,
                                    beta_parameter = 0, seed = 10)
generator.generate_instance()

# Visualize partition
generator.draw_partition(node_size = 200, with_labels = True)
print("\nPartition: ")
for k, P_k in generator.graph["P"].items():
    print(f"{k}: {P_k}")

# Visualize production vectors
print("\nNode Attributes:")



