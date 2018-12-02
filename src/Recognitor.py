import DetectChars
import DetectPlates
import os

import cv2

# module level variables ##########################################################################
SCALAR_BLACK = (0.0, 0.0, 0.0)
SCALAR_WHITE = (255.0, 255.0, 255.0)
SCALAR_YELLOW = (0.0, 255.0, 255.0)
SCALAR_GREEN = (0.0, 255.0, 0.0)
SCALAR_RED = (0.0, 0.0, 255.0)

showSteps = False


def recognize(path):
    train_success = DetectChars.loadKNNDataAndTrainKNN()  # attempt KNN training

    if not train_success:  # if KNN training was not successful
        print("\nerror: KNN training was not successful\n")
        return

    original_scene = cv2.imread(path)

    if original_scene is None:
        print("\nerror: image not read from file \n\n")  # print error message to std out
        os.system("pause")  # pause so user can see error message
        return

    possible_plates = DetectPlates.detectPlatesInScene(original_scene)  # detect plates

    possible_plates = DetectChars.detectCharsInPlates(possible_plates)  # detect chars in plates

    #cv2.imshow("imgOriginalScene", original_scene)

    if len(possible_plates) == 0:
        print("\nno license plates were detected\n")
    else:
        # if we get in here list of possible plates has at leat one plate

        # sort the list of possible plates in DESCENDING order (most number of chars to least number of chars)
        possible_plates.sort(key=lambda possible_plate: len(possible_plate.strChars), reverse=True)

        # suppose the plate with the most recognized chars (the first plate in sorted by string length
        # descending order) is the actual plate
        plate = possible_plates[0]

        #cv2.imshow("imgPlate", plate.imgPlate)  # show crop of plate and threshold of plate
        #cv2.imshow("imgThresh", plate.imgThresh)

        if len(plate.strChars) == 0:  # if no chars were found in the plate
            print("\nno characters were detected\n\n")  # show message
            return

        highlight_plate(original_scene, plate)  # draw red rectangle around plate

        print("\nlicense plate read from image = " + plate.strChars + "\n")  # write license plate text to std out
        print("----------------------------------------")

        write_chars(original_scene, plate)  # write license plate text on the image

        #cv2.imshow("imgOriginalScene", original_scene)  # re-show scene image

        cv2.imwrite("imgOriginalScene.png", original_scene)  # write image out to file
        return original_scene

    cv2.waitKey(0)
    return


def highlight_plate(scene, plate):
    rect_points = cv2.boxPoints(plate.rrLocationOfPlateInScene)  # get 4 vertices of rotated rect

    cv2.line(scene, tuple(rect_points[0]), tuple(rect_points[1]), SCALAR_RED, 2)  # draw 4 red lines
    cv2.line(scene, tuple(rect_points[1]), tuple(rect_points[2]), SCALAR_RED, 2)
    cv2.line(scene, tuple(rect_points[2]), tuple(rect_points[3]), SCALAR_RED, 2)
    cv2.line(scene, tuple(rect_points[3]), tuple(rect_points[0]), SCALAR_RED, 2)


def write_chars(scene, plate):
    scene_height, scene_width, scene_num_channels = scene.shape
    plate_height, plate_width, plate_num_channels = plate.imgPlate.shape

    font_face = cv2.FONT_HERSHEY_SIMPLEX  # choose a plain jane font
    font_scale = float(plate_height) / 30.0  # base font scale on height of plate area
    font_thickness = int(round(font_scale * 1.5))  # base font thickness on font scale

    text_size, baseline = cv2.getTextSize(plate.strChars, font_face, font_scale, font_thickness)

    # unpack roatated rect into center point, width and height, and angle
    ((plate_center_x, plate_center_y), (_, _),
     fltCorrectionAngleInDeg) = plate.rrLocationOfPlateInScene

    plate_center_x = int(plate_center_x)  # make sure center is an integer
    plate_center_y = int(plate_center_y)

    center_x = int(plate_center_x)  # the horizontal location of the text area is the same as the plate

    if plate_center_y < (scene_height * 0.75):  # if the license plate is in the upper 3/4 of the image
        center_y = int(round(plate_center_y)) + int(
            round(plate_height * 1.6))  # write the chars in below the plate
    else:  # else if the license plate is in the lower 1/4 of the image
        center_y = int(round(plate_center_y)) - int(
            round(plate_height * 1.6))  # write the chars in above the plate

    text_size_width, text_size_height = text_size  # unpack text size width and height

    lower_left_text_origin_x = int(
        center_x - (text_size_width / 2))  # calculate the lower left origin of the text area
    lower_left_text_origin_y = int(
        center_y + (text_size_height / 2))  # based on the text area center, width, and height

    # write the text on the image
    cv2.putText(scene, plate.strChars, (lower_left_text_origin_x, lower_left_text_origin_y), font_face,
                font_scale, SCALAR_YELLOW, font_thickness)