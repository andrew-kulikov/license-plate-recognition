# DetectPlates.py

import math
import random

import DetectChars
import PossibleChar
import PossiblePlate
import Preprocess
import Recognitor
import cv2
import numpy as np

PLATE_WIDTH_PADDING_FACTOR = 1.3
PLATE_HEIGHT_PADDING_FACTOR = 1.5


def detectPlatesInScene(img_original_scene):
    listOfPossiblePlates = []

    height, width, num_channels = img_original_scene.shape

    img_contours = np.zeros((height, width, 3), np.uint8)

    cv2.destroyAllWindows()

    if Recognitor.showSteps:
        cv2.imshow("0", img_original_scene)

    img_grayscale_scene, img_thresh_scene = Preprocess.preprocess(img_original_scene)

    if Recognitor.showSteps:
        cv2.imshow("1a", img_grayscale_scene)
        cv2.imshow("1b", img_thresh_scene)

        # find all possible chars in the scene,
        # this function first finds all contours, then only includes contours that could be chars
        # (without comparison to other chars yet)
    listOfPossibleCharsInScene = findPossibleCharsInScene(img_thresh_scene)

    if Recognitor.showSteps:
        print("step 2 - len(listOfPossibleCharsInScene) = " + str(
            len(listOfPossibleCharsInScene)))  # 131 with MCLRNF1 image

        img_contours = np.zeros((height, width, 3), np.uint8)

        contours = []

        for possibleChar in listOfPossibleCharsInScene:
            contours.append(possibleChar.contour)

        cv2.drawContours(img_contours, contours, -1, Recognitor.SCALAR_WHITE)
        cv2.imshow("2b", img_contours)

    # given a list of all possible chars, find groups of matching chars
    # in the next steps each group of matching chars will attempt to be recognized as a plate
    listOfListsOfMatchingCharsInScene = DetectChars.findListOfListsOfMatchingChars(listOfPossibleCharsInScene)

    if Recognitor.showSteps:
        print("step 3 - listOfListsOfMatchingCharsInScene.Count = " + str(
            len(listOfListsOfMatchingCharsInScene)))  # 13 with MCLRNF1 image

        img_contours = np.zeros((height, width, 3), np.uint8)

        for listOfMatchingChars in listOfListsOfMatchingCharsInScene:
            intRandomBlue = random.randint(0, 255)
            intRandomGreen = random.randint(0, 255)
            intRandomRed = random.randint(0, 255)

            contours = []

            for matchingChar in listOfMatchingChars:
                contours.append(matchingChar.contour)

            cv2.drawContours(img_contours, contours, -1, (intRandomBlue, intRandomGreen, intRandomRed))

        cv2.imshow("3", img_contours)

    for listOfMatchingChars in listOfListsOfMatchingCharsInScene:  # for each group of matching chars
        possiblePlate = extractPlate(img_original_scene, listOfMatchingChars)  # attempt to extract plate

        if possiblePlate.imgPlate is not None:
            listOfPossiblePlates.append(possiblePlate)

    print("\n" + str(len(listOfPossiblePlates)) + " possible plates found")

    if Recognitor.showSteps:
        print("\n")
        cv2.imshow("4a", img_contours)

        for i in range(0, len(listOfPossiblePlates)):
            p2fRectPoints = cv2.boxPoints(listOfPossiblePlates[i].rrLocationOfPlateInScene)

            cv2.line(img_contours, tuple(p2fRectPoints[0]), tuple(p2fRectPoints[1]), Recognitor.SCALAR_RED, 2)
            cv2.line(img_contours, tuple(p2fRectPoints[1]), tuple(p2fRectPoints[2]), Recognitor.SCALAR_RED, 2)
            cv2.line(img_contours, tuple(p2fRectPoints[2]), tuple(p2fRectPoints[3]), Recognitor.SCALAR_RED, 2)
            cv2.line(img_contours, tuple(p2fRectPoints[3]), tuple(p2fRectPoints[0]), Recognitor.SCALAR_RED, 2)

            cv2.imshow("4a", img_contours)

            print("possible plate " + str(i) + ", click on any image and press a key to continue . . .")

            cv2.imshow("4b", listOfPossiblePlates[i].imgPlate)
            cv2.waitKey(0)
        # end for

        print("\nplate detection complete, click on any image and press a key to begin char recognition . . .\n")
        cv2.waitKey(0)

    return listOfPossiblePlates


def findPossibleCharsInScene(img_thresh):
    possible_chars = []

    possible_chars_count = 0

    img_thresh_copy = img_thresh.copy()

    img_contours, contours, npa_hierarchy = cv2.findContours(img_thresh_copy, cv2.RETR_LIST,
                                                             cv2.CHAIN_APPROX_SIMPLE)

    height, width = img_thresh.shape
    img_contours = np.zeros((height, width, 3), np.uint8)

    for i in range(0, len(contours)):

        if Recognitor.showSteps:
            cv2.drawContours(img_contours, contours, i, Recognitor.SCALAR_WHITE)

        possibleChar = PossibleChar.PossibleChar(contours[i])

        if DetectChars.checkIfPossibleChar(possibleChar):
            possible_chars_count = possible_chars_count + 1  # increment count of possible chars
            possible_chars.append(possibleChar)  # and add to list of possible chars

    if Recognitor.showSteps:
        print("\nstep 2 - len(contours) = " + str(len(contours)))
        print("step 2 - intCountOfPossibleChars = " + str(possible_chars_count))
        cv2.imshow("2a", img_contours)

    return possible_chars


def extractPlate(img_original, list_of_matching_chars):
    possible_plate = PossiblePlate.PossiblePlate()  # this will be the return value

    list_of_matching_chars.sort(key=lambda c: c.intCenterX)

    # calculate the center point of the plate
    fltPlateCenterX = (list_of_matching_chars[0].intCenterX + list_of_matching_chars[
        len(list_of_matching_chars) - 1].intCenterX) / 2.0
    fltPlateCenterY = (list_of_matching_chars[0].intCenterY + list_of_matching_chars[
        len(list_of_matching_chars) - 1].intCenterY) / 2.0

    ptPlateCenter = fltPlateCenterX, fltPlateCenterY

    # calculate plate width and height
    intPlateWidth = int((list_of_matching_chars[len(list_of_matching_chars) - 1].intBoundingRectX + list_of_matching_chars[
        len(list_of_matching_chars) - 1].intBoundingRectWidth - list_of_matching_chars[
                             0].intBoundingRectX) * PLATE_WIDTH_PADDING_FACTOR)

    intTotalOfCharHeights = 0

    for matchingChar in list_of_matching_chars:
        intTotalOfCharHeights = intTotalOfCharHeights + matchingChar.intBoundingRectHeight
    # end for

    fltAverageCharHeight = intTotalOfCharHeights / len(list_of_matching_chars)

    intPlateHeight = int(fltAverageCharHeight * PLATE_HEIGHT_PADDING_FACTOR)

    # calculate correction angle of plate region
    fltOpposite = list_of_matching_chars[len(list_of_matching_chars) - 1].intCenterY - list_of_matching_chars[0].intCenterY
    fltHypotenuse = DetectChars.distanceBetweenChars(list_of_matching_chars[0],
                                                     list_of_matching_chars[len(list_of_matching_chars) - 1])
    fltCorrectionAngleInRad = math.asin(fltOpposite / fltHypotenuse)
    fltCorrectionAngleInDeg = fltCorrectionAngleInRad * (180.0 / math.pi)

    # pack plate region center point, width and height, and correction angle into rotated rect member variable of plate
    possible_plate.rrLocationOfPlateInScene = (tuple(ptPlateCenter),
                                               (intPlateWidth, intPlateHeight),
                                               fltCorrectionAngleInDeg)

    # final steps are to perform the actual rotation

    # get the rotation matrix for our calculated correction angle
    rotationMatrix = cv2.getRotationMatrix2D(tuple(ptPlateCenter), fltCorrectionAngleInDeg, 1.0)

    height, width, numChannels = img_original.shape  # unpack original image width and height

    imgRotated = cv2.warpAffine(img_original, rotationMatrix, (width, height))  # rotate the entire image

    imgCropped = cv2.getRectSubPix(imgRotated, (intPlateWidth, intPlateHeight), tuple(ptPlateCenter))

    possible_plate.imgPlate = imgCropped

    return possible_plate
