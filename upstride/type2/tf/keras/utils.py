

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
