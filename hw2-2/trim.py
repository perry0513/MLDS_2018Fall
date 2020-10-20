new_output = open('./MLDS_hw2_2_data/no_space_input.txt', 'w')
output = open('./MLDS_hw2_2_data/test_input.txt', 'r').read().splitlines()
for line in output:
	line = ''.join(line.split())
	new_output.write(line+'\n')
	
	
