

def multiply_by_a(vector):
    A = [[1, 1, 1, 1], [1, -1, 1, -1], [1, 1, -1, -1], [1, -1, -1, 1]]

    output_vector = []
    for i in range(4):
        for j in range(4):
            if j == 0:
                if A[i][j] == 1:
                    output_vector.append(vector[j])
                else:
                    output_vector.append(-vector[j])
            else:
                if A[i][j] == 1:
                    output_vector[i] = output_vector[i] + vector[j]
                else:
                    output_vector[i] = output_vector[i] - vector[j]
    return output_vector


def quaternion_mult(tf_op, inputs, kernels):
    """[summary]

    Args:
        tf_op ([type]): function taking as parameter (input, kernel)
        input ([type]): [description]
        kernel ([type]): [description]
    """
    if len(inputs) == 4:
        kernel_sum = multiply_by_a(kernels)
        input_sum = multiply_by_a(inputs)

        output_sum = []
        for i in range(4):
            output_sum.append(tf_op(input_sum[i], kernel_sum[i]))
        output_sum = multiply_by_a(output_sum)

        # other convolution
        output_rest = [
            tf_op(inputs[0], kernels[0]),
            tf_op(inputs[2], kernels[3]),
            tf_op(inputs[1], kernels[2]),
            tf_op(inputs[3], kernels[1]),
        ]

        outputs = [output_sum[i]/4 - 2*output_rest[i] for i in range(4)]
        outputs[0] = -outputs[0]
    else:
        outputs = [tf_op(inputs[0], kernels[i]) for i in range(4)]
    return outputs
