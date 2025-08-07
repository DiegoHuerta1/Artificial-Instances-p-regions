from  Instance_Generator import Single_Instance_Generator


generator = Single_Instance_Generator(num_points=10, num_regions=3, num_features=3, beta_parameter=0, seed=1)
generator.generate_graph()

print(generator.graph["pos"])
print(generator.graph["pos"][2])





