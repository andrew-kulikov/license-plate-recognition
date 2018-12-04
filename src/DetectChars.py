# DetectChars.py
import math
import os
import random

import PossibleChar
import Preprocess
import Recognitor
import cv2
import numpy as np

kNearest = cv2.ml.KNearest_create()

MIN_PIXEL_WIDTH = 2
MIN_PIXEL_HEIGHT = 8

MIN_ASPECT_RATIO = 0.25
MAX_ASPECT_RATIO = 1.0

MIN_PIXEL_AREA = 80

MIN_DIAG_SIZE_MULTIPLE_AWAY = 0.3
MAX_DIAG_SIZE_MULTIPLE_AWAY = 5.0

MAX_CHANGE_IN_AREA = 0.5

MAX_CHANGE_IN_WIDTH = 0.8
MAX_CHANGE_IN_HEIGHT = 0.2

MAX_ANGLE_BETWEEN_CHARS = 12.0

MIN_NUMBER_OF_MATCHING_CHARS = 3

RESIZED_CHAR_IMAGE_WIDTH = 20
RESIZED_CHAR_IMAGE_HEIGHT = 30

MIN_CONTOUR_AREA = 100


def loadKNNDataAndTrainKNN():
    try:
        npa_classifications = np.loadtxt("classifications.txt", np.float32)
    except:
        print("error, unable to open classifications.txt, exiting program\n")
        os.system("pause")
        return False

    try:
        npa_flattened_images = np.loadtxt("flattened_images.txt", np.float32)
    except:
        print("error, unable to open flattened_images.txt, exiting program\n")
        os.system("pause")
        return False

    # reshape numpy array to 1d, necessary to pass to call to train
    npa_classifications = npa_classifications.reshape((npa_classifications.size, 1))

    kNearest.setDefaultK(1)

    kNearest.train(npa_flattened_images, cv2.ml.ROW_SAMPLE, npa_classifications)

    return True


def detectCharsInPlates(possible_plates):
    plate_count = 0
    contours = []

    if len(possible_plates) == 0:
        return possible_plates

    for possiblePlate in possible_plates:

        possiblePlate.imgGrayscale, possiblePlate.imgThresh = Preprocess.preprocess(possiblePlate.imgPlate)

        if Recognitor.showSteps:
            cv2.imshow("5a", possiblePlate.imgPlate)
            cv2.imshow("5b", possiblePlate.imgGrayscale)
            cv2.imshow("5c", possiblePlate.imgThresh)

        # increase size of plate image for easier viewing and char detection
        possiblePlate.imgThresh = cv2.resize(possiblePlate.imgThresh, (0, 0), fx=1.6, fy=1.6)

        # threshold again to eliminate any gray areas
        thresholdValue, possiblePlate.imgThresh = cv2.threshold(possiblePlate.imgThresh, 0.0, 255.0,
                                                                cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        if Recognitor.showSteps:
            cv2.imshow("5d", possiblePlate.imgThresh)

        # find all possible chars in the plate,
        # this function first finds all contours, then only includes contours that could be chars
        # (without comparison to other chars yet)
        list_of_possible_chars_in_plate = findPossibleCharsInPlate(possiblePlate.imgGrayscale, possiblePlate.imgThresh)

        if Recognitor.showSteps:
            height, width, num_channels = possiblePlate.imgPlate.shape
            img_contours = np.zeros((height, width, 3), np.uint8)
            del contours[:]

            for possibleChar in list_of_possible_chars_in_plate:
                contours.append(possibleChar.contour)

            cv2.drawContours(img_contours, contours, -1, Recognitor.SCALAR_WHITE)

            cv2.imshow("6", img_contours)

        # given a list of all possible chars, find groups of matching chars within the plate
        matching_chars = findListOfListsOfMatchingChars(list_of_possible_chars_in_plate)

        if Recognitor.showSteps:
            img_contours = np.zeros((height, width, 3), np.uint8)
            del contours[:]

            for listOfMatchingChars in matching_chars:
                random_blue = random.randint(0, 255)
                random_green = random.randint(0, 255)
                random_red = random.randint(0, 255)

                for matchingChar in listOfMatchingChars:
                    contours.append(matchingChar.contour)

                cv2.drawContours(img_contours, contours, -1, (random_blue, random_green, random_red))

            cv2.imshow("7", img_contours)

        if len(matching_chars) == 0:

            if Recognitor.showSteps:
                print("chars found in plate number " + str(
                    plate_count) + " = (none), click on any image and press a key to continue . . .")
                plate_count = plate_count + 1
                cv2.destroyWindow("8")
                cv2.destroyWindow("9")
                cv2.destroyWindow("10")
                cv2.waitKey(0)

            possiblePlate.strChars = ""
            continue

        for i in range(0, len(matching_chars)):
            matching_chars[i].sort(
                key=lambda c: c.intCenterX)
            matching_chars[i] = removeInnerOverlappingChars(matching_chars[i])

        if Recognitor.showSteps:
            img_contours = np.zeros((height, width, 3), np.uint8)

            for listOfMatchingChars in matching_chars:
                random_blue = random.randint(0, 255)
                random_green = random.randint(0, 255)
                random_red = random.randint(0, 255)

                del contours[:]

                for matchingChar in listOfMatchingChars:
                    contours.append(matchingChar.contour)

                cv2.drawContours(img_contours, contours, -1, (random_blue, random_green, random_red))

            cv2.imshow("8", img_contours)

        # within each possible plate, suppose the longest list of potential matching chars is the actual list of chars
        len_of_longest_list_of_chars = 0
        id_of_longest_list_of_chars = 0

        # loop through all the vectors of matching chars, get the index of the one with the most chars
        for i in range(0, len(matching_chars)):
            if len(matching_chars[i]) > len_of_longest_list_of_chars:
                len_of_longest_list_of_chars = len(matching_chars[i])
                id_of_longest_list_of_chars = i

        # suppose that the longest list of matching chars within the plate is the actual list of chars
        longest_list_of_matching_chars_in_plate = matching_chars[id_of_longest_list_of_chars]

        if Recognitor.showSteps:
            img_contours = np.zeros((height, width, 3), np.uint8)
            del contours[:]

            for matchingChar in longest_list_of_matching_chars_in_plate:
                contours.append(matchingChar.contour)

            cv2.drawContours(img_contours, contours, -1, Recognitor.SCALAR_WHITE)

            cv2.imshow("9", img_contours)

        possiblePlate.strChars = recognizeCharsInPlate(possiblePlate.imgThresh, longest_list_of_matching_chars_in_plate)

        if Recognitor.showSteps:
            print("chars found in plate number " + str(
                plate_count) + " = " + possiblePlate.strChars + ", click on any image and press a key to continue ...")
            plate_count = plate_count + 1
            cv2.waitKey(0)

    if Recognitor.showSteps:
        print("\nchar detection complete, click on any image and press a key to continue . . .\n")
        cv2.waitKey(0)

    return possible_plates


def findPossibleCharsInPlate(imgGrayscale, img_thresh):
    list_of_possible_chars = []
    img_thresh_copy = img_thresh.copy()

    # find all contours in plate
    img_contours, contours, npa_hierarchy = cv2.findContours(img_thresh_copy, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:  # for each contour
        possible_char = PossibleChar.PossibleChar(contour)

        if checkIfPossibleChar(
                possible_char):  # if contour is a possible char, note this does not compare to other chars (yet) . . .
            list_of_possible_chars.append(possible_char)  # add to list of possible chars

    return list_of_possible_chars


def checkIfPossibleChar(possible_char):
    # this function is a 'first pass' that does a rough check on a contour to see if it could be a char,
    # note that we are not (yet) comparing the char to other chars to look for a group
    return (possible_char.intBoundingRectArea > MIN_PIXEL_AREA and
            possible_char.intBoundingRectWidth > MIN_PIXEL_WIDTH and possible_char.intBoundingRectHeight > MIN_PIXEL_HEIGHT and
            MIN_ASPECT_RATIO < possible_char.fltAspectRatio < MAX_ASPECT_RATIO)


def findListOfListsOfMatchingChars(list_of_possible_chars):
    # with this function, we start off with all the possible chars in one big list
    # the purpose of this function is to re-arrange the one big list of chars into a list of lists of matching chars,
    # note that chars that are not found to be in a group of matches do not need to be considered further
    list_of_lists_of_matching_chars = []

    for possibleChar in list_of_possible_chars:  # for each possible char in the one big list of chars
        list_of_matching_chars = findListOfMatchingChars(possibleChar, list_of_possible_chars)

        list_of_matching_chars.append(possibleChar)

        if len(list_of_matching_chars) < MIN_NUMBER_OF_MATCHING_CHARS:
            continue

        # if we get here, the current list passed test as a "group" or "cluster" of matching chars
        list_of_lists_of_matching_chars.append(list_of_matching_chars)  # so add to our list of lists of matching chars

        # remove the current list of matching chars from the big list so we don't use those same chars twice,
        # make sure to make a new big list for this since we don't want to change the original big list
        possible_chars_distinct = list(set(list_of_possible_chars) - set(list_of_matching_chars))

        matching_char_lists = findListOfListsOfMatchingChars(possible_chars_distinct)

        for recursiveListOfMatchingChars in matching_char_lists:
            list_of_lists_of_matching_chars.append(recursiveListOfMatchingChars)

        break

    return list_of_lists_of_matching_chars


def findListOfMatchingChars(possibleChar, listOfChars):
    # the purpose of this function is, given a possible char and a big list of possible chars,
    # find all chars in the big list that are a match for the single possible char,
    # and return those matching chars as a list
    listOfMatchingChars = []  # this will be the return value

    for possibleMatchingChar in listOfChars:  # for each char in big list
        if possibleMatchingChar == possibleChar:
            continue

        distance_between_chars = distanceBetweenChars(possibleChar, possibleMatchingChar)

        angle_between_chars = angleBetweenChars(possibleChar, possibleMatchingChar)

        change_in_area = float(
            abs(possibleMatchingChar.intBoundingRectArea - possibleChar.intBoundingRectArea)) / float(
            possibleChar.intBoundingRectArea)

        change_in_width = float(
            abs(possibleMatchingChar.intBoundingRectWidth - possibleChar.intBoundingRectWidth)) / float(
            possibleChar.intBoundingRectWidth)
        change_in_height = float(
            abs(possibleMatchingChar.intBoundingRectHeight - possibleChar.intBoundingRectHeight)) / float(
            possibleChar.intBoundingRectHeight)

        # check if chars match
        if (distance_between_chars < (possibleChar.fltDiagonalSize * MAX_DIAG_SIZE_MULTIPLE_AWAY) and
                angle_between_chars < MAX_ANGLE_BETWEEN_CHARS and
                change_in_area < MAX_CHANGE_IN_AREA and
                change_in_width < MAX_CHANGE_IN_WIDTH and
                change_in_height < MAX_CHANGE_IN_HEIGHT):
            listOfMatchingChars.append(
                possibleMatchingChar)

    return listOfMatchingChars


# use Pythagorean theorem to calculate distance between two chars
def distanceBetweenChars(c1, c2):
    x = abs(c1.intCenterX - c2.intCenterX)
    y = abs(c1.intCenterY - c2.intCenterY)

    return math.sqrt((x ** 2) + (y ** 2))


# use basic trigonometry (SOH CAH TOA) to calculate angle between chars
def angleBetweenChars(c1, c2):
    dx = float(abs(c1.intCenterX - c2.intCenterX))
    dy = float(abs(c1.intCenterY - c2.intCenterY))

    if dx != 0.0:
        angle_rad = math.atan(dy / dx)
    else:
        angle_rad = 1.5708

    angle_deg = angle_rad * (180.0 / math.pi)

    return angle_deg


# if we have two chars overlapping or to close to each other to possibly be separate chars, remove the inner (smaller),
# this is to prevent including the same char twice if two contours are found for the same char,
# for example for the letter 'O' both the inner ring and the outer ring may be found as contours,
# but we should only include the char once
def removeInnerOverlappingChars(listOfMatchingChars):
    listOfMatchingCharsWithInnerCharRemoved = list(listOfMatchingChars)  # this will be the return value

    for currentChar in listOfMatchingChars:
        for otherChar in listOfMatchingChars:
            if currentChar != otherChar:
                if distanceBetweenChars(currentChar, otherChar) < (
                        currentChar.fltDiagonalSize * MIN_DIAG_SIZE_MULTIPLE_AWAY):
                    if currentChar.intBoundingRectArea < otherChar.intBoundingRectArea:
                        if currentChar in listOfMatchingCharsWithInnerCharRemoved:
                            listOfMatchingCharsWithInnerCharRemoved.remove(currentChar)
                    else:
                        if otherChar in listOfMatchingCharsWithInnerCharRemoved:
                            listOfMatchingCharsWithInnerCharRemoved.remove(otherChar)

    return listOfMatchingCharsWithInnerCharRemoved


# this is where we apply the actual char recognition
def recognizeCharsInPlate(img_thresh, list_of_matching_chars):
    plate_text = ""

    height, width = img_thresh.shape

    img_thresh_color = np.zeros((height, width, 3), np.uint8)

    list_of_matching_chars.sort(key=lambda c: c.intCenterX)  # sort chars from left to right

    cv2.cvtColor(img_thresh, cv2.COLOR_GRAY2BGR,
                 img_thresh_color)  # make color version of threshold image so we can draw contours in color on it

    for currentChar in list_of_matching_chars:  # for each char in plate
        pt1 = (currentChar.intBoundingRectX, currentChar.intBoundingRectY)
        pt2 = ((currentChar.intBoundingRectX + currentChar.intBoundingRectWidth),
               (currentChar.intBoundingRectY + currentChar.intBoundingRectHeight))

        cv2.rectangle(img_thresh_color, pt1, pt2, Recognitor.SCALAR_GREEN, 2)  # draw green box around the char

        # crop char out of threshold image
        imgROI = img_thresh[
                 currentChar.intBoundingRectY: currentChar.intBoundingRectY + currentChar.intBoundingRectHeight,
                 currentChar.intBoundingRectX: currentChar.intBoundingRectX + currentChar.intBoundingRectWidth]

        imgROIResized = cv2.resize(imgROI, (
        RESIZED_CHAR_IMAGE_WIDTH, RESIZED_CHAR_IMAGE_HEIGHT))  # resize image, this is necessary for char recognition

        npaROIResized = imgROIResized.reshape(
            (1, RESIZED_CHAR_IMAGE_WIDTH * RESIZED_CHAR_IMAGE_HEIGHT))  # flatten image into 1d numpy array

        npaROIResized = np.float32(npaROIResized)  # convert from 1d numpy array of ints to 1d numpy array of floats

        retval, npaResults, neigh_resp, dists = kNearest.findNearest(npaROIResized,
                                                                     k=1)  # finally we can call findNearest !!!

        strCurrentChar = str(chr(int(npaResults[0][0])))  # get character from results

        plate_text = plate_text + strCurrentChar  # append current char to full string

    if Recognitor.showSteps:
        cv2.imshow("10", img_thresh_color)

    return plate_text
