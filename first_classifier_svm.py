# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in

import sys
import csv
import os
from os import listdir
from os.path import isfile, join
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from scipy import ndimage
import cv2
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn import svm
from sklearn import linear_model
from sklearn.externals import joblib  # to save the SVM model so don't need to re-train every time


def main():
    print "helloworld"

    # Input data files are available in the "../input/" directory.
    # For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

    from subprocess import check_output
    #print(check_output(["ls", "../input"]).decode("utf8"))

    # Any results you write to the current directory are saved as output.

    #http://docs.opencv.org/trunk/d1/d89/tutorial_py_orb.html
    #img_rows, img_cols= 350, 425
    gray = cv2.imread('train/LAG/img_00091.jpg', 0)
    #plt.imshow(im_array, cmap='gray')
    #img = cv2.imread('image.jpg')
    #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    detector = cv2.SIFT()
    kp1, des1 = detector.detectAndCompute(gray, None)
    img=cv2.drawKeypoints(gray,kp1,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.imwrite('sift_keypoints.jpg',img)

    #img = im_array #cv2.imread('simple.jpg',0)
    # Initiate ORB detector

    # find the keypoints with ORB
    #kp = orb.detect(img,None)
    # compute the descriptors with ORB
    #kp, des = orb.compute(img, kp)
    # draw only keypoints location,not size and orientation
    #img2 = cv2.drawKeypoints(img, kp, None, color=(0,255,0), flags=0)
    #plt.imshow(img2)


    # great, first step:  grab all the filenames and directories, so we can classify them
    mydirvalues = [d for d in os.listdir(os.path.dirname(os.path.abspath(__file__)))]
    print(mydirvalues)
    onlyfiles = [f for f in listdir("train/") if isfile(join("train/", f))]
    print(onlyfiles)

    dir_names = [d for d in listdir("train/") if not isfile(join("train/", d))]
    print(dir_names)

    file_paths = {}
    class_num = 0
    for d in dir_names:
        fnames = [f for f in listdir("train/"+d+"/") if isfile(join("train/"+d+"/", f))]
        print(fnames)
        file_paths[(d, class_num, "train/"+d+"/")] = fnames
        class_num += 1


    # General steps:
    # Extract feature from each file as HOG or similar... or SIFT... or Similar...
    # map each to feature space... and train some kind of classifier on that, SVM?
    # do the same for each feature in test set...
    training_data = np.array([])
    training_labels = np.array([])
    #training_data = []
    #training_labels = []
    for key in file_paths:
        category = key[1]
        directory_path = key[2]
        file_list = file_paths[key]

        # Stop early, while testing, so it doesn't take FOR-EV-ER (FOR-EV-ER)
        i = 0

        # read in the file and get its SIFT features
        for fname in file_list:
            fpath = directory_path + fname
            print(fpath)
            # extract features!
            gray = cv2.imread(fpath,0)
            gray = cv2.resize(gray, (1200, 750))  # resize so we're always comparing same-sized images
            # may need to re-size so that all images have the same total NxN after running through SIFT
            detector = cv2.SIFT()
            kp1, des1 = detector.detectAndCompute(gray, None)
            # X=Training vector
            # Y=Category to predict

            # >> Should randomize vectors before slicing
            # could also duplicate a few features if needed to hit a hgher value
            if len(kp1) < 1000:
                continue

            des1 = des1.astype(np.float64)
            np.random.shuffle(des1)
            des1 = des1[0:1000,:] # trim vector so all are same size
            vector_data = des1.reshape(1,128000)
            list_data = vector_data.tolist()
            #training_data.append(list_data)

            if len(training_data) == 0:
                training_data = np.append(training_data, vector_data)
                training_data = training_data.reshape(1,128000)#reshape(1000,128)
            else:
                training_data   = np.concatenate((training_data, vector_data), axis=0)
            training_labels = np.append(training_labels,category)

            #training_data.append(vector_data)
            #training_labels.append(category)

            # early stop
            i += 1
            if i > 2:
                break

    # Alright! Now we've got features extracted and labels
    clf = svm.SVC(kernel='linear', C = 1.0)
    #X = np.array(training_data)
    #y = np.array(training_labels).astype(float)
    X = training_data
    y = training_labels
    y = y.reshape(y.shape[0],)
    clf.fit(X,y)

    # logistic regression
    logreg = linear_model.LogisticRegression(C=1e5)
    logreg.fit(X, y)
    # Now, extract one of the images and predict it
    gray = cv2.imread('test_stg1/img_00071.jpg', 0)  # Correct is LAG --> Class 3
    kp1, des1 = detector.detectAndCompute(gray, None)

    des1 = des1[0:1000, :]   # trim vector so all are same size
    vector_data = des1.reshape(1, 128000)
    print("Linear SVM Prediction:")
    print(clf.predict(vector_data))
    print("Logistic Regression Prediction:")
    print(logreg.predict(vector_data))
    print("Logistic Regression Probability:")   # << What they are looking for...
    print(logreg.predict_proba(vector_data))

    # save SVM model
    joblib.dump(clf, 'filename.pkl')
    # to load SVM model, use:  clf = joblib.load('filename.pkl')

    # try a few more, see what kind of accuracy we get
    prediction_output_list = []  # list of lists, containing logistic regression for each file
    fnames = [f for f in listdir("test_stg1/") if isfile(join("test_stg1/", f))]
    print "Testing File Names:"
    print(fnames)

    # early stoppage...
    # only do 10
    #i = 0
    for f in fnames:
        file_name = "test_stg1/" + f
        print("---Evaluating File at: " + file_name)
        gray = cv2.imread(file_name, 0)  # Correct is LAG --> Class 3
        gray = cv2.resize(gray, (1200, 750))  # resize so we're always comparing same-sized images
        kp1, des1 = detector.detectAndCompute(gray, None)

        # ensure we have at least 1000 keypoints to analyze
        if len(kp1) < 1000:
            current_len = len(kp1)
            vectors_needed = 1000 - current_len
            repeated_vectors = des1[0:vectors_needed, :]
            # concatenate repeats onto des1
            while len(des1) < 1000:
                des1 = np.concatenate((des1, repeated_vectors), axis=0)
            # duplicate data just so we can run the model.
            des1[current_len:1000, :] = des1[0:vectors_needed, :]

        np.random.shuffle(des1)  # shuffle the vector so we get a representative sample
        des1 = des1[0:1000, :]   # trim vector so all are same size
        vector_data = des1.reshape(1, 128000)
        print("Linear SVM Prediction:")
        print(clf.predict(vector_data))
        print("Logistic Regression Prediction:")
        print(logreg.predict(vector_data))
        print("Logistic Regression Probability:")   # << What they are looking for...
        logistic_prediction = logreg.predict_proba(vector_data)
        print(logistic_prediction)

        # format list for csv output
        csv_output_list = []
        csv_output_list.append(f)
        for elem in logistic_prediction:
            for value in elem:
                csv_output_list.append(value)

        # append filename to make sure we have right format to write to csv
        print("CSV Output List Formatted:")
        print(csv_output_list)


        # and append this file to the output_list (of lists)
        prediction_output_list.append(csv_output_list)


        # Uncomment to stop early
        #if i > 10:
        #    break
        #i += 1




    # Write to csv
    #f = open('sift_and_svm_submission.csv', 'wt')  # sys.argv[1]
    try:
        with open("sift_and_svm_submission.csv", "wb") as f:
            writer = csv.writer(f)
            headers = ['image', 'ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT']
            writer.writerow(headers)
            writer.writerows(prediction_output_list)


    finally:
        f.close()


def write_to_csv():
    return

def listDirectories():
    return None


if __name__ == "__main__":
    main()
