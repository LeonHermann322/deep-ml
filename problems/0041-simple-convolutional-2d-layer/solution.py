import numpy as np

def simple_conv2d(input_matrix: np.ndarray, kernel: np.ndarray, padding: int, stride: int):
	input_height, input_width = input_matrix.shape
	kernel_height, kernel_width = kernel.shape
	
	padded_input = np.pad(input_matrix, pad_width=padding, mode="constant", constant_values=0)

	output_height = (padded_input.shape[0] - kernel_height) // stride + 1
	output_width = (padded_input.shape[1] - kernel_width) // stride + 1

	output = np.zeros([output_height, output_width])

	for i in range(output_height):
		for j in range(output_width):
			top_left = [i * stride, j * stride]
			window = padded_input[top_left[0]:top_left[0]+kernel_height,top_left[1]:top_left[1]+kernel_width]
			result = np.sum(window * kernel)
			output[i,j] = result
    
	return output
