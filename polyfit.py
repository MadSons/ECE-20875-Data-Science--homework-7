import numpy as np
import matplotlib.pyplot as plt


# Return fitted model parameters to the dataset at datapath for each choice in degrees.
# Input: datapath as a string specifying a .txt file, degrees as a list of positive integers.
# Output: paramFits, a list with the same length as degrees, where paramFits[i] is the list of
# coefficients when fitting a polynomial of d = degrees[i].
def main(datapath, degrees):
    paramFits = []
    file = open(datapath, 'r')
    data = file.readlines()
    x = []
    y = []
    for line in data:
        [i, j] = line.split()
        x.append(float(i))
        y.append(float(j))

    # iterate through each n in degrees, calling the feature_matrix and least_squares functions to solve
    # for the model parameters in each case. Append the result to paramFits each time.

    for n in degrees:
        X = feature_matrix(x, n)
        B = least_squares(X, y)
        paramFits.append(B)

    return paramFits


# Return the feature matrix for fitting a polynomial of degree d based on the explanatory variable
# samples in x.
# Input: x as a list of the independent variable samples, and d as an integer.
# Output: X, a list of features for each sample, where X[i][j] corresponds to the jth coefficient
# for the ith sample. Viewed as a matrix, X should have dimension #samples by d+1.
def feature_matrix(x, d):

    # fill in
    # There are several ways to write this function. The most efficient would be a nested list comprehension
    # which for each sample in x calculates x^d, x^(d-1), ..., x^0.
    # mat_x = []
    # for row in range(len(x)):
    #     mat_x.append([])
    #     for col in range(0, d+1):
    #         mat_x[row].append(x[row]**(d-col))
    return [[x[row]**(d-col) for col in range(d+1)] for row in range(len(x))]


# Return the least squares solution based on the feature matrix X and corresponding target variable samples in y.
# Input: X as a list of features for each sample, and y as a list of target variable samples.
# Output: B, a list of the fitted model parameters based on the least squares solution.
def least_squares(X, y):
    X = np.array(X)
    y = np.array(y)

    # fill in
    # Use the matrix algebra functions in numpy to solve the least squares equations. This can be done in just one line.
    B = np.linalg.inv(X.T @ X) @ (X.T @ y)
    return B


if __name__ == "__main__":
    datapath = "poly.txt"

    # Read the input file, "poly.txt", assuming it has two columns, where each row is of the form [x y]
    file = open(datapath, 'r')
    data = file.readlines()
    x = []
    y = []
    for line in data:
        [i, j] = line.split()
        x.append(float(i))
        y.append(float(j))

    # Problem 1. 
    # Complete 'main, 'feature_matrix', and 'lease_squares' functions. 

    # Problem 2. 
    # Write out the resulting estimated functions for each d.
    degrees = [1, 2, 3, 4, 5]
    paramFits = main(datapath, degrees)

    for idx in range(len(degrees)):
        print("y_hat(x_"+str(degrees[idx])+")")
        print(paramFits[idx])
        print("****************")

    # Problem 3. 
    # Visualize the dataset and these fitted models on a single graph
    # Use the 'scatter' and 'plot' functions in the `matplotlib.pyplot` module. 

    # if x = 3, what is the predicted value of y?
    val = 3
    y_at_3 = paramFits[2][0] * val ** 3 + paramFits[2][1] * val ** 2 + paramFits[2][2] * val + paramFits[2][3]
    print(f'At x = 3, the predicted value of y is {y_at_3}')
    # Draw a scatter plot
    plt.scatter(x, y, color='black', label='data')
    x.sort()
    x = np.array(x)
    y = np.array(y)

    for i in range(len(degrees)):
        new_y = 0
        for j in range(len(paramFits[i])):
            new_y += paramFits[i][j] * x**(i+1-j)
        text = 'd = ' + str(i+1)
        plt.plot(x, new_y, label=text)

    plt.xlabel('x', fontsize=16)
    plt.ylabel('y', fontsize=16)
    plt.legend(fontsize=10, loc='upper left')

    plt.show()


    
