import tensorflow as tf
import numpy as np
import cv2
import os


# Modifies the image by changing the color of a number of pixels, by a random factor
# img - the image, squashed
# args = [prob, factor] where:
# prob = [p, 1-p] - a set of complementary probabilities. The 2nd one is the probability that a pixel of the image will
#   be selected to change color
# factor = a number between 0 and 1 that determines the maximum amount a color can change, as a fraction of the current
#   color
# Side note - this function has suffered a lot of changes, the current form is not necessarily the most inspired
def mutate(img, args):
    prob = args[0]
    factor = args[1]

    #Create a random mask for the image (1 = the pixel will change color)
    mask = np.random.choice([0, 1], size=(1, 784), p=prob)

    #Do a positive update
    random = np.multiply(np.random.random((1, 784)), factor)
    random = np.multiply(random, mask)
    ones = np.ones((1, 784))
    diff = np.subtract(ones, img)
    delta = np.multiply(diff, random)
    toRet = img + delta

    #Do a negative update
    random = np.multiply(np.random.random((1, 784)), min(2 * factor, 1))
    random = np.multiply(random, mask)
    toRet = toRet - np.multiply(toRet, random)

    return [toRet, args]


# Performs an image convolution with a gaussian kernel
# NOT USED
def convolve(img, w = 28, h = 28, kw = 3, kh = 3):
    gauss = np.divide(np.asarray([
        [1, 4, 6, 4, 1],
        [4, 16, 24, 16, 4],
        [6, 24, 36, 24, 6],
        [4, 16, 24, 16, 4],
        [1, 4, 6, 4, 1],
    ]), 256.)

    img2d = np.reshape(img, (h, w))
    #kernel = np.divide(np.multiply(np.random.random(size=(kh, kw)), 1), float(kw * kh))
    output=np.zeros(shape=(h,w))
    cv2.filter2D(src=img2d, dst=output, ddepth=-1, kernel=gauss)
    return np.reshape(output, (1, 784))

# Randomly resets to 0 40% of the image's pixels
# NOT USED
def randomReset(img):
  mask = np.random.choice([0, 1], size=(1, 784), p=[0.40, 0.60])
  return np.multiply(img, mask)

# Restores a squashed, inverted and normalized image. Usually used before showing the image.
# Note: squashed means covnerted to 1D (a 28X28 2D image will be converted to a 784 1D image,
#   inverted means 1 is BLACK and 0 is WHITE, normalized means that all values are between [0,1]
def get_proper_img(img, w, h):
    image = img.reshape((w, h))
    temp = np.ones((w, h))
    image = np.multiply((temp - image), 256)
    return image

# Concatenates the image img to the image src, adding extra space to src at the bottom and placing text in that area
# Used to build the row of images at the end of the search
def concatenate(src, img, axis=1, extraH = None, text=""):
    if extraH is not None:
        extra = np.ones(shape=(extraH, img.shape[1]))
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(extra, text, (3, 13), font, 0.25, (255, 255, 255))
        img = np.concatenate((img, extra), axis=0)
    if src is None:
        return img
    separator = np.zeros(shape = (src.shape[0], 1))
    src = np.concatenate((src, separator), axis=axis)
    src = np.concatenate((src, img), axis=axis)
    return src

# Saves a squashed, inverted and normalized image to disk
def writeImg(img, w, h, name):
    image = get_proper_img(img, w, h)
    cv2.imwrite(name, image)

#The function that performs the search. It handles both First Choice and Simmulated Annealing. It is also able to
# perform a "best out of N" hill climbing, though this is not used
#
# alter_image_fn - the name of the function that will be used to modify the image; "mutate" is the function currently used
# aif_args - a set of parameters to be passed to the above function as an array; see the "mutate" definition
# sess, x, y the neural network as read from disk (the model and placeholders for the input/output)
# current - the initial image
# output_folder - the path where resulting images will be stored
# output_every_n_steps - will output images every n successful steps
# no_beams - to perform "best out of n" hill climbing; not sufficiently tested, especially with simmulated annealing
# annealing_schedule_fn - a function taking as input the current iteration and returning the T value for the simmulated
#   annealing formula; its presence or absence also determines if simmulated annealing is used or not.
def generic_search(alter_image_fn, aif_args, sess, x, y, current, output_folder, output_every_n_steps = 10, no_beams = 1,
                   annealing_schedule_fn = None):
    lastScore = -1000000
    #[mutation, new_args] = alter_image_fn(current, aif_args)
    iter = 0
    steps = 1
    failed = 0
    os.makedirs(output_folder, exist_ok=True)

    # This will store the modified images at each step (more than one if no_beams is more than one)
    beams = [[current, aif_args]]

    # Save the initial image
    writeImg(current, 28, 28, output_folder + "\\output-" + str(iter) + "-" + str(steps) + str(lastScore) + ".png")

    # This is the report image, saved at the end
    checkpoints = concatenate(None, get_proper_img(current, 28, 28), extraH=16, text="Initial")

    # Perform 100,000 iterations. This should really be a function parameter
    while iter < 100000:
        # Calculate the scores for the current set of mutations
        scores = np.ndarray(shape = [len(beams)], dtype=np.float32)
        for b in range(0, len(beams)):
            scores[b] = sess.run(y, feed_dict={x: beams[b][0]})[0][0]
        # This stores the index of the highest score
        maxIdx = np.unravel_index(scores.argmax(), scores.shape)[0]

        # Determine if the current step should be accepted
        accept = False

        # If we are using simmulated annealing
        if annealing_schedule_fn is not None:
            # If the score is higher than previous, accept it
            if scores[maxIdx] > lastScore:
                accept = True
            # else use the formula
            else:
                delta = scores[maxIdx] - lastScore
                accept_threshold = np.exp(delta / annealing_schedule_fn(iter))
                if accept_threshold == 0:
                    accept = False
                else:
                    choice = np.random.uniform(0, 1)
                    accept = choice < accept_threshold
        # If we are using first-choice hill climbing
        else:
            accept = scores[maxIdx] > lastScore

        # If the current best score should be accepted
        if accept:
            #Make the best score image the current image
            current = beams[maxIdx][0]
            aif_args = beams[maxIdx][1]
            lastScore = scores[maxIdx]
            print(str(iter) + "," + str(lastScore) + ",")
            steps += 1

            #If necessary, save the image
            if steps % output_every_n_steps == 0:
                writeImg(current, 28, 28, output_folder + "\\output-" + str(iter) + "-" + str(steps) + "-" + str(lastScore) + ".png")
            failed = 0

            #If at these steps, update the report image. The steps list should also be a function paramerter...
            if steps in [50, 100, 200, 400, 600, 800, 1000, 1200, 1500, 2000, 3000, 5000]:
                checkpoints = concatenate(checkpoints, get_proper_img(current, 28, 28), extraH=16, text=str(steps))
        else:
            failed += 1

        #Modify the current image no_beams times
        beams = []
        for b in range(0,no_beams):
            beams.append(alter_image_fn(np.copy(current), aif_args))
        iter += 1

    # Write the last image and the report image
    writeImg(current, 28, 28, output_folder + "\\output-" + str(iter) + "-" + str(steps) + str(lastScore) + ".png")
    checkpoints = concatenate(checkpoints, get_proper_img(current, 28, 28), extraH=16, text="Final")
    cv2.imwrite(output_folder + "\\overview.png", checkpoints)


#Load a neural network from the given path
def getNN(path):
    sess = tf.Session()
    saver = tf.train.import_meta_graph(path + '.meta')
    saver.restore(sess, path)
    graph = sess.graph
    x = graph.get_tensor_by_name("x:0")
    y = graph.get_tensor_by_name("output_pre:0")
    return [sess, x, y]


#The annealing function
def annealing1(iter):
    return np.power(0.99, iter)


# Used to test the network output on the training samples
# Useful in analysis, otherwise not used
def test_folder(folder, sess, x, y):
    values = []
    for f in os.listdir(folder):
        img = cv2.imread(folder + "\\" + f)
        img = cv2.split(img)[0]
        img = np.reshape(img, (1, 784))
        img = 1 - img
        p = sess.run(y, feed_dict={x: img})
        values.append(p[0][0])
    values_np = np.asarray(values)
    return values_np

if __name__ == "__main__":

    # Load the neural network to be used. See the project folder for other networks or the neural_networks.py file for
    # training others
    [sess, x, y] = getNN("C:\\NNPaint\\2layers-5")

    # Perform 6 searches
    for k in range (1, 2):
        # Create the initial image
        positions = np.random.uniform(low=0, high=783, size=200).astype(np.int32)
        # positions = np.multiply(np.ones(shape=200), 28 * 14 + 14).astype(np.int32)
        values = np.random.uniform(size=200)
        current = np.zeros(shape=(1, 784))
        for i in range(0, positions.shape[0]):
            current[0, positions[i]] = values[i]

        # Perform the serch
        generic_search(mutate, [[0.995, 0.005], 1], sess, x, y, current, "../img/2layers5-fchc-" + str(k), output_every_n_steps=50,
                       no_beams=1, annealing_schedule_fn=None)


