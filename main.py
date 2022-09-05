"""
In this program vector is to be understood as a 1D list and a matrix is a 2D
list of lists.
"""

import os  # This is used to list files in a directory
import struct  # This is used to unpack bytes into useable values
from random import shuffle, uniform  # Used in creating random lists and values.
import json  # Used to read saved network

# Following two lines are used when images are plotted and gif file is created.
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

# To create a NumPy implementation and compare to a pure python implementation
import numpy as np


def add(U, V):
    """
    Vector addition.
    ================
    Does normal vector addition from linear algebra.

    Parameters
    ==========
    U : A list
    V : A list

    Returns
    =======
    Addition of vectors (lists) U and V
    """
    assert len(U) == len(V), "Dimensions do not match!"

    return [u + v for u, v in zip(U, V)]


def sub(U, V):
    """
    Vector subtraction
    ==================
    Does normal vector subtraction.

    Parameters
    ==========
    U : A list (Vector)
    V : A list (Vector). length of U and V are equal

    Returns
    =======
    """
    assert len(U) == len(V), "Dimensions do not match!"

    return [u - v for u, v in zip(U, V)]


def scalar_multiplication(scalar, V):
    """
    Vector scalar multiplication
    ============================
    Multiplies each element in vector by a scalar.

    Parameters
    ==========
    scalar : An number to multiply to the vector V
    V : A vector (list)

    Returns
    =======
    A vector (list) : scalar * V
    """

    assert isinstance(scalar, int) or isinstance(scalar, float), \
        "Scalar is neither float nor integer!"

    assert isinstance(V, list), "Vector V is not a list!"

    def rec(inst):
        if isinstance(inst, int) or isinstance(inst, float):
            return inst * scalar
        else:
            return [rec(i) for i in inst]

    return rec(V)


def transpose(M):
    """
    Transposing of matrix
    =====================
    Takes in a matrix from linear algebra, and findes the transpose.

    Parameters
    ==========
    M : A matrix (List of lists)

    Returns
    =======
    The matrix M transposed
    """
    assert isinstance(M, list), "M must be a list!"
    assert all(list(map(lambda x: isinstance(x, list), M))), \
        "M must contain only lists!"

    return list(zip(*M))


def multiply(V, M):
    """
    Matrix multiplication
    =====================
    Takes two matrixs and does normal matrix multiplication from linear algebra.

    Parameters
    ==========
    V : A vector (list)
    M : A matrix (list of lists)

    Returns
    =======
    V multiplied on M (list)
    """
    cols = transpose(M)

    assert isinstance(V, list), "V must be a list!"
    assert len(cols[0]) == len(V), "Dimensions do not match"

    new_V = []
    for col in cols:
        new_V.append(sum([c * v for c, v in zip(col, V)]))

    return new_V


def shape(M):
    """
    Shape of matrix
    ===============
    Findes the dimentions of a matrix.

    Parameters
    ==========
    M : A matrix (list of lists)

    Returns
    =======
    A string containing the dimensions of the matrix.
    """
    return f"M_{len(M)},{len(M[0])}"


def element_wise(V1, V2):
    """
    Element multiplication
    ======================
    Multiplies elements with same index, for the two vectors.

    Parameters
    ==========
    V1 : A vector (list)
    V2 : A vector (list)

    Returns
    =======
    A list from the elementwise multiplication of V1 and V2
    """

    assert isinstance(V1, list) and isinstance(V2, list), \
        "V1 and V2 must be lists!"
    assert len(V1) == len(V2), \
        "Dimensions of V1 must be equal to V2!"

    return [v1 * v2 for v1, v2 in zip(V1, V2)]


def read_labels(file_name):
    """
    Reading of label file
    =====================
    First the function checks if the right file has been loaded, by finding the
    magic number. Then read each label in the file and returns them in a list.

    Parameters
    ==========
    file_name : Name of the file to read the labels from

    Returns
    =======
    A list containing all the labels in the file 'file_name'
    """
    assert isinstance(file_name, str), "file_name must be a str!"

    labels = []

    with open(file_name, 'rb') as file:
        magic, size = struct.unpack('>II', file.read(8))

        assert magic == 2049, "Wrong file type given. Magic number != 2049"

        labels = [_ for _ in file.read()]

    return labels


def read_images(file_name):
    """
    Reading of images
    =================
    First the function checks if the right file has been loaded, by finding the
    magic number. Then collects each image in a list of lists and returns all
    images in a list.

    Parameters
    ==========
    file_name : Name of the file to read the images from

    Returns
    =======
    A list of images (list of list of lists)
    """
    assert isinstance(file_name, str), "file_name must be a str!"

    images_read = []

    with open(file_name, 'rb') as file:
        magic, size, num_rows, num_cols = struct.unpack('>IIII', file.read(16))

        assert magic == 2051, "Wrong file type given. Magic number != 2049"

        for image_num in range(size):

            image = []

            for _row in range(num_rows):
                row = [_ for _ in file.read(num_cols)]

                image.append(row)

            images_read.append(image)

        return images_read


def plot_images(images, labels, prediction=None):
    """
    Visualisation of images
    =======================
    Outputs each image with the correct label, if no predicitons are givin, else
    the image's title will say if the prediction is correct and what has been
    predicted.

    Parameters
    ==========
    images : The images to plot (list of list of lists)
    labels : The respective label to each image in the parameter 'images'
    prediction : List of predictions corresponding to the images. Plotted along
                 images
    """
    for i, (image, label) in enumerate(zip(images, labels)):
        plt.imshow(image)
        if prediction is None:
            plt.title(label)
        else:
            pred_i = prediction[i]
            tstr = f"Correct? {label == pred_i}. Act {label},\ Pred {pred_i}"
            plt.title(tstr)

        plt.show()


def linear_load(file_name):
    """
    Linear load
    ================
    This code will load the given file using json.

    Parameters
    ==========
    file_name : Name of the file to load network from

    Returns
    =======
    Network loaded from the file 'file_name'
    """
    assert isinstance(file_name, str), "file_name must be a str!"
    with open(file_name) as infile:
        return json.load(infile)


def linear_save(file_name, network):
    """
    Linear save
    ================
    This code will save a given network.

    Parameters
    ==========
    network : A network containing both the weights and biases
    file_name : The file to which the network is saved
    """
    with open(file_name, 'w') as outfile:
        json.dump(str(network), outfile)


def image_to_vector(image):
    """
    Image convertion
    ================
    Takes each element in a image and scales it from [0,255] to [0,1], and
    returns it a vector.

    Paramerers
    ==========
    image : An image to convert to a list

    Returns
    =======
    The image converted to a list
    """
    return scalar_multiplication(1 / 255, [i for row in image for i in row])


def mean_square_error(U, V):
    """
    Mean square error
    =================
    Finds the mean sqaure error for the elements of two list, given that they
    are of equal length.

    Parameters
    ==========
    U : list
    V : list

    Returns
    =======
    Mean square error between U and V
    """
    assert len(U) == len(V)

    return sum([(u - v) ** 2 for u, v in zip(U, V)]) / len(U)


def argmax(V):
    """
    Element of highest value
    ========================
    Finds the max value of a list and returns the index.

    Parameters
    ==========
    V : a list fromwhich the max argument is found

    Returns
    =======
    Index of the value in the list V, which has the highest value.
    """
    assert isinstance(V, list), "V must be a list!"

    return V.index(max(V))


def categorical(label, classes=10):
    """
    Creates a vector with a '1' at the index of the label
    =====================================================
    Creates a list of '0' the length of classes, and change the element which
    has the same inx as the given label to '1'.

    Parameters
    ==========
    label : The label element in the returned list which is non-zero
    classes : Number of elements in the returned list

    Returns
    =======
    A list with 'classes' elements with a 1 in 'label's place and zero
    everywhere else.
    """
    assert 0 <= label <= 9, "Label must be between 0 and 9!"

    L = [0] * classes
    L[label] = 1
    return L


def predict(network, image):
    """
    Predictions
    ===========
    Uses the linear algebra expression 'xA + b' to get a vector of values, where
    the index with the highest value is the prediction. Here A and b are from
    the network and x is the image.

    Parameters
    ==========
    network : Contains both the weights (list of lists) and bias (list)
    image : The image to predict the digit representation of using the network

    Return
    ======
    Prediction of the digit representation of the image given
    """
    return add(multiply(image, network[0]), network[1])


def evaluate(network, images, labels):
    """
    Evalution of the prediction
    ===========================
    Finds the prediction by taking the max index from the predict function, the
    accuracy by making a list of booleans and determining the ratio of right
    predictions to all predictions. The cost is the mean square error of the
    predictions to categorical vector of the label.

    Parameters
    ==========
    network : Contains both the weights (list of lists) and bias (list)
    images : A batch containing a list of images to evaluate
    labels : Digit label corresponding to the images

    Returns
    =======
    Tuple containing the prediction, cost and accuracy of the model evaluation.
    """
    accuracy = []
    cost = []
    predictions = []
    for image, label in zip(images, labels):
        resp = predict(network, image_to_vector(image))
        pred_digit = argmax(resp)
        predictions.append(pred_digit)
        accuracy.append(pred_digit == label)
        cost.append(mean_square_error(resp, categorical(label)))

    cost = sum(cost) / len(cost)
    accuracy = sum(accuracy) / len(accuracy)

    return predictions, cost, accuracy


def list_to_matrix(V):
    """
    Converts image vector to image matrix
    =====================================
    Takes the image previously converted to a vector and returns it as a matrix.

    Parameters
    ==========
    V : A list to convert to a matrix

    Returns
    =======
    The list V converted to a matrix
    """
    img = []
    for i in range(28):
        row = []
        for j in range(28):
            row.append(V[i * 28 + j])
        img.append(row)

    return img


def visualize_network(A):
    """
    Gives a visual representation of the network
    ============================================
    Takes the matrix A from the trained network and takes each of the ten
    columbs, convertes them to images and plots them with imshow.

    Parameters
    ==========
    A : Network to visualize each column of (Column is the digit representation
        in the network)
    """
    for count, A_row in enumerate(transpose(A)):
        img = list_to_matrix(A_row)
        plt.imshow(img)
        plt.title(count)
        plt.show()


def create_batches(values, batch_size):
    """
    Creating batches
    ====================
    This function creates batches with a given set of values.
    Batch_size determines how many lists, and the values will get spread out
    in these lists.
    This will be used in the learn function.

    Parameters
    ==========
    values : The elements in a list to create batches of
    batch_size : Number of elements in a batch

    Returns
    =======
    values converted into batches
    """
    partitions = []
    col = []
    shuffle(values)
    for i in range(len(values)):
        col.append(values[i])

        if (i + 1) % batch_size == 0:
            partitions.append(col)
            col = []

    if len(col) != 0:
        partitions.append(col)

    return partitions


def update(network, images, labels):
    """
    Updating the network
    ====================
    This function updates the network by computing the gradual change in the
    matrix A and vector b. It does this via the derivitiv funtion of the mean
    sqare error, for A and b respectivly. This is done for each image and label
    gradually getting the network better at guessing the numbers.

    Paramters
    =========
    network : Contains both the weights (list of lists) and bias (list)
    images : The images to go through to update the network
    labels : The digit representation of an image corresponding to 'images'

    Returns
    =======
    A network containing both the weights and bias.
    """
    sigma = 0.1

    A, b = network

    n = len(images)

    # Adjust A and b
    sum_b = [0] * len(b)
    sum_A = [[0] * len(A[0])] * len(A)
    for image, label in zip(images, labels):
        x = image_to_vector(image)
        a = predict((A, b), x)
        y = categorical(label)

        # Adjust b
        sum_b = add(sum_b, scalar_multiplication(2 / 10, sub(a, y)))

        # Adjust A
        for i in range(len(A)):
            A_i = scalar_multiplication(sigma / n,
                                        scalar_multiplication(x[i] * 2 / 10,
                                                              sub(a, y)))
            sum_A[i] = add(A_i, sum_A[i])

    # Putting these values into the actual matrix and vector
    for i in range(len(A)):
        A[i] = sub(A[i], sum_A[i])

    b = sub(b, scalar_multiplication(sigma / n, sum_b))

    return A, b


def random_weights():
    """
    Random weights
    ===================
    This produces random weights later for our learn function. The values in the
    matrix go between 0 and 1 / 784.

    Returns
    =======
    Random weights (aka. the 'A' matrix)
    """
    A = []
    for i in range(28 * 28):
        row = []
        for j in range(10):
            row.append(uniform(0, 1 / 784))
        A.append(row)

    b = []
    for i in range(10):
        b.append(uniform(0, 1))

    return A, b


def print_eval(evaluation, title=None):
    """
    Printing of results
    ===================
    Prints the results after evaluating the function, and putting it in a
    readable format.

    Paramters
    =========
    evaluation : The output from an 'evalution' function
    title : The title that should be printed and formatted before the evaluation
    """
    print()
    if title != None:
        print(title)
        print("=" * len(title))
    print("Results:")
    print(f"\t * {'Pred':<10}:", evaluation[0])
    print(f"\t * {'Cost':<10}:", evaluation[1])
    print(f"\t * {'Accuracy':<10}:", evaluation[2])
    print()


def learn(images, labels, epochs, batch_size, create_gif_anim=False):
    """
    Train a NN to recognize handwritten digits
    ==========================================
    This function will make random weights in the beginning. Thereafter,
    it will create batches, and each of these batches will be affected
    by the update function. The update function make the guessing of the number
    better, and by doing it for several epochs, we get better and better
    accuracy. The cost and the working of this function is hidden in
    print_eval.

    Parameters
    ==========
    images : The images to train the network from
    labels : The digits corresponding to the images
    epochs : Number of iterations through the batches
    batch_size : How many elements in each batch
    create_gif_anim : Whether or not a GIF-representation should be created of
                      the evolution of the weights.

    Returns
    =======
    A network containing both the weights and biases.
    """
    # Random network
    network = random_weights()

    title = "Test before training (Random weights and biases):"
    print_eval(evaluate(network, images[:40], labels[:40]), title)

    batches = create_batches(list(zip(images, labels)), batch_size)
    cost = []  # Used later to visualize change in cost
    accuracy = []  # Used later to visualize change in accuracy
    A_t = []  # This here is later used when animating the weights

    print(f"Beginning training (Batch size: {batch_size}):")
    for epoch in range(epochs):  # predictions, cost, accuracy
        for batch in batches:
            batch_images, batch_labels = list(zip(*batch))
            network = update(network, batch_images, batch_labels)
            A_t.append(list(network[0]))

            # Following lines of code allow for visualization of accuracy/cost
            # _, c, a = evaluate(network, images[:100], labels[:100])
            # accuracy.append(a)
            # cost.append(c)
        print(f" ... Epoch {epoch + 1}/{epochs} completed.")

    title = "Test after NN training:"
    print_eval(evaluate(network, images[:40], labels[:40]), title)

    """
    Following lines are used to visualize the accuracy and cost of the NN model,
    after every batch in its training. May be uncommented if desired
    """

    # plt.plot(list(range(1,len(accuracy)+1)), accuracy)
    # plt.xlabel("Network updates")
    # plt.ylabel("Accuracy")
    # plt.title("NN Prediction accuracy of digits over network updates (Using pure Python)")
    # plt.savefig(r"pp_accuracy.pdf")
    # plt.show()
    # plt.clf()
    # plt.plot(list(range(1,len(cost)+1)), cost)
    # plt.xlabel("Network updates")
    # plt.ylabel("Cost")
    # plt.title("NN cost over network updates (Using pure Python)")
    # plt.savefig(r"pp_cost.pdf")
    # plt.show()

    if create_gif_anim:
        print("Creating GIF representation of changes in weight...")
        create_gif_digits(A_t)

    return network


def test_network(network):
    """
    Test a network on test images
    =============================
    Tests the network and shows how accurate it is, by getting the the images
    and labels it hasn't been trained on.

    Parameters
    ==========
    network : Containing both the weights and biases
    """
    test_image_file = "Digits/t10k-images.idx3-ubyte"
    test_label_file = "Digits/t10k-labels.idx1-ubyte"

    test_images = read_images(test_image_file)
    test_labels = read_labels(test_label_file)

    title = "Test on test images with a given network"
    print_eval(evaluate(network, test_images, test_labels), title)


def get_column(i, n, As):
    """
    Get n'th column from i'th image

    Parameters
    ==========
    i : Index of image to get column from
    n : Which column in the image to return
    As : List of weights

    Returns
    =======
    The n'th column from the i'th image in the list of weights 'As'.
    """

    cols = transpose(As[i])
    return cols[n]


def create_gif_digits(A_t):
    """
    Creates a gif from the columns in the weights matrix

    Parameters
    ==========
    A_t : List of weights
    """

    for digit in range(10):
        fig = plt.figure()
        im = plt.imshow(list_to_matrix(get_column(0, digit, A_t)))

        plt.title(digit)

        def animate(i):
            im.set_array(list_to_matrix(get_column(i, digit, A_t)))
            return [im]

        anim = FuncAnimation(fig, animate, frames=len(A_t) - 1, interval=500)

        anim.save(f"digit-{digit}.gif", writer=PillowWriter(fps=20))


def train_and_save(imgs, labels, num_imgs=500, epochs=2, batches=100):
    """
    Train a NN to recognize handwritten digits, then save model
    ===========================================================
    The NN model is trained with 500 images only (As more than this is quite
    time consuming), with 2 epochs and batch size of 100 images.

    Takes optional arguments num_imgs (Number of images), epochs (Number of
    epochs), batches (How many batches the set should be split into)

    Paramters
    =========
    imgs : List of images to train the network
    labels : Digits corresponding to the given images
    num_imgs : How many images from the training set should be trained on
    epochs : Number of epochs - Number of iterations through the batches
    batches : How many images in a batch
    """

    disc_network = learn(imgs[:num_imgs], labels[:num_imgs], epochs, batches)
    linear_save("mine.weights", disc_network)


def plot_handwritten():
    """
    Plots handwritten digits from the directory "Handwritten"
    """

    for file_name in os.listdir("Handwritten"):
        with open(rf"Handwritten\{file_name}") as file:
            print("Reading file")
            img_a = eval(file.read())
            plt.imshow(img_a)
            plt.title("Handwriting")
            plt.show()


def read_handwriting(network):
    """
    Reads handwritten digits from the directory "Handwritten"and predicts the
    digit written using a NN model.

    Parameters
    ==========
    network : Contains both the weights and bias.
    """

    for file_name in os.listdir("Handwritten"):
        with open(rf"Handwritten\{file_name}") as file:
            print("Reading file")
            img_a = eval(file.read())
            img = image_to_vector(img_a)
            print(len(img_a), len(img_a[0]))
            predicted = argmax(predict(network, img))

            plt.imshow(img_a)
            plt.title(f"Prediction: {predicted}")
            plt.show()


"""
Then, a NN model using NumPy is implemented in seperate functions from
those above.
"""


def np_load_images(file_name):
    """
    Load an image file using NumPy
    ==============================
    The images are read using the function taken from,
    https://numpy.org/doc/stable/reference/generated/numpy.fromfile.html

    To read the values, we need to use an unsigned 8-bit integer, which is here,
    https://numpy.org/doc/stable/reference/arrays.scalars.html#numpy.uint,
    described as an uint8. The datatype uint8 is what we will be using here.

    Parameters
    ==========
    file_name : Name of the file from which the images are loaded.

    Returns
    =======
    A list containing all the images in the file 'file_name'
    """

    # Skip first 4 32-bit integers aka 4*4 bytes. These are the magic bytes,
    # image numbers, row and column sizes.
    with open(file_name, 'rb') as file:
        magic, size, num_rows, num_cols = struct.unpack('>IIII', file.read(16))

    assert magic == 2051, "Incorrect magic number!"

    return np.fromfile(file_name, np.uint8)[4 * 4:].reshape(size, num_rows, num_cols)


def np_predict(network, imgs):
    """
    Given a network and a list of images, this function predicts what digit each
    image represents.

    Parameters
    ==========
    network : Contains both the weights and bias.
    imgs : The images to predict the digit of, using the given network.
           (NumPy array)

    Returns
    =======
    Numpy array of predictions as numpy arrays
    """
    A, b = network
    imgs_vectors = imgs.reshape(len(imgs), 28 * 28) * 1 / 255
    return imgs_vectors @ A + b


def np_mean_square_error(V1, V2):
    """
    Mean square error using NumPy

    Parameters
    ==========
    V1 : NumPy array
    V2 : NumPy array

    Returns
    =======
    Mean square error between the arrays V1 and V2
    """
    return np.sum((V1 - V2) ** 2) / 10


def np_evaluate(network, imgs, labels):
    """
    Evaluates NN on images and labels
    =================================
    The resp variable in this function is a list of lists. Therefore, to find
    the element in the list with the highest value, the numpy function argmax
    is used and is given the argument axis=1, which will return a list
    containing the index of the max value in each row.
    https://numpy.org/doc/stable/reference/generated/numpy.argmax.html

    Parameters
    ==========
    network : Contains both the weights and bias.
    imgs : NumPy array of images to evaluate
    labels : Digits corresponding to the given images

    Returns
    =======
    Prediction, cost and accuracy.
    """
    resp = np_predict(network, imgs)
    preds = resp.argmax(1)
    cost = np_mean_square_error(resp, [categorical(label) for label in labels]) \
           / len(resp)

    # sum(labels == preds), will be a sum of True and False statements, where
    # True and False statements are equivalent to 1 or 0.
    accuracy = np.sum(labels == preds) / len(preds)

    return preds, cost, accuracy


def np_update(network, images, labels):
    """
    Update a network given a batch of images and labels

    Parameters
    ==========
    network : Contains both the weights and bias
    images : NumPy array of images to use to update the network
    labels : Digits corresponding to the given images

    Returns
    =======
    A network containing both the weights and bias.
    """
    A, b = network
    sigma = 0.1
    n = len(images)

    x = images.reshape(len(images), 28 * 28) / 255
    a = np_predict(network, images)
    y = [categorical(label) for label in labels]

    b_update = sigma * (1 / n) * (a - y).sum(axis=0) / 10
    A_update = np.zeros(np.shape(A))

    for i in range(len(a)):
        A_update += sigma * (1 / n) * 2 / 10 * np.array([x[i]]).T * (a[i] - y[i])

    A -= A_update
    b -= b_update

    return A, b


def np_create_batches(imgs, labels, batch_size):
    """
    Create partitions of imgs and labels
    ====================================
    A slightly different function to create batches is created using NumPy,
    this is done to allow for images and labels being seperate arguments. Also,
    the return values are numpy arrays and not lists.

    Parameters
    ==========
    imgs : Images to create batches of
    labels : Labels to create batches of in relation to the given images
    batch_size : Number of elements in a batch

    Returns
    =======
    imgs and labels split up into batches
    """
    img_partitions = []
    label_partitions = []
    img_col = []
    label_col = []
    nums = np.arange(0, 10000)

    # https://numpy.org/doc/stable/reference/random/generated/numpy.random.shuffle.html
    np.random.shuffle(nums)

    for i in nums:
        img_col.append(imgs[i])
        label_col.append(labels[i])

        if (i + 1) % batch_size == 0:
            img_partitions.append(np.array(img_col))
            label_partitions.append(np.array(label_col))
            img_col = []
            label_col = []

    if len(img_col) != 0:
        img_partitions.append(np.array(img_col))
        label_partitions.append(np.array(label_col))

    return img_partitions, label_partitions


def np_learn(images, labels, epochs, batch_size):
    """
    Linear load
    ================
    This code will load the given file using json.

    Parameters
    ==========
    images : Images to develop a network from
    labels : Digits corresponding to images
    epochs : Number of epochs (Number of iterations through the batches)
    batch_size : Number of elements in a batch

    Returns
    =======
    A network containing both the weights and bias.
    """
    img_batches, label_batches = np_create_batches(images, labels, batch_size)

    A = np.random.random((28 * 28, 10)) * 1 / (28 * 28)
    b = np.random.random((1, 10))[0]

    print_eval(np_evaluate((A, b), images, labels), "Before (Training set)")

    accuracy = []  # This list is used to plot accuracy of model over NN update.
    cost = []  # This is used to plot cost of model over NN update.

    print(f"Beginning training (Batch size: {batch_size}):")
    for epoch in range(epochs):
        print(f" ... Epoch {epoch + 1}/{epochs}")
        for batch_imgs, batch_labels in zip(img_batches, label_batches):
            A, b = np_update((A, b), batch_imgs, batch_labels)

            """
            Following 3 commented lines of code, were used to
            create a list containing the cost and accuracy
            of the model, and may be uncommented if these
            are desired.
            """

            # _, c, a = np_evaluate((A, b), images, labels)
            # accuracy.append(a)
            # cost.append(c)

    print_eval(np_evaluate((A, b), images, labels), "After (Training set)")

    """
    These last lines of code within the function are used to plot the cost and
    accuracy over network updates, and may be disregarded.
    """

    # plt.plot(list(range(1,len(accuracy)+1)), accuracy)
    # plt.xlabel("Network updates")
    # plt.ylabel("Accuracy")
    # plt.title("NN Prediction accuracy of digits over network updates (Using NumPy)")
    # plt.savefig(r"C:\Users\rosen\Desktop\accuracy.pdf")
    # plt.show()
    # plt.plot(list(range(1,len(accuracy)+1)), cost)
    # plt.xlabel("Network updates")
    # plt.ylabel("Cost")
    # plt.title("NN cost over network updates (Using NumPy)")
    # plt.savefig(r"C:\Users\rosen\Desktop\cost.pdf")
    # plt.show()

    return A, b


# This is the training files
t_image_file = "Digits/train-images.idx3-ubyte"
t_label_file = "Digits/train-labels.idx1-ubyte"

# This is the reading of the training files
L_labels = read_labels(t_label_file)
L_images = read_images(t_image_file)

# Test the mnist_linear.weights network on the subset of the training set
mnist_network = linear_load("Misc/mnist_linear.weights")
title = "Mnist network test"
print_eval(evaluate(mnist_network, L_images[:100], L_labels[:100]), title)

# Train a network using pure python (On a subset of the train images & labels)
our_network = learn(L_images[:1000], L_labels[:1000], 2, 100, True)
title = "Evaluate our network: "
print_eval(evaluate(our_network, L_images[:100], L_labels[:100]), title)

# These lines are used to test the numpy implementation
np_images = np_load_images("Digits/train-images.idx3-ubyte")
np_labels = read_labels("Digits/train-labels.idx1-ubyte")
A, b = np_learn(np_images, np_labels, 5, 100)
np_images_test = np_load_images("Digits/t10k-images.idx3-ubyte")
np_labels_test = read_labels("Digits/t10k-labels.idx1-ubyte")
title = "Test of obtained network on test images"
print_eval(np_evaluate((A, b), np_images_test, np_labels_test), title)
