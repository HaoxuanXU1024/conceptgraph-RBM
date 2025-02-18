def scale_transformation_elements(input_file, output_file):
    import numpy as np
    
    # Read the file and process each line
    with open(input_file, 'r') as file:
        lines = file.readlines()
    
    new_lines = []
    for line in lines:
        # Convert line into a list of floats
        values = np.array(line.split(), dtype=float)
        
        # Scale the specified elements by 100
        values[[3]] /= 1
        
        # Convert back to string and format
        new_line = ' '.join(f"{v:.15e}" for v in values)
        new_lines.append(new_line + '\n')
    
    # Write the new data to output file
    with open(output_file, 'w') as file:
        file.writelines(new_lines)

# You can call this function with the path to your 'traj.txt' and the desired output file name:
# scale_transformation_elements('traj.txt', 'traj_new.txt')
scale_transformation_elements('/data1/haoxuan/concept-graphs/conceptgraph/Datasets/Replica/test1/traj_base.txt', '/data1/haoxuan/concept-graphs/conceptgraph/Datasets/Replica/test1/traj.txt')
