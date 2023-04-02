import pathlib
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image
from math import pi
import time
import tensorflow as tf
from object_detection.utils import label_map_util

import anki_vector

# Limit GPU memory usage, 
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

# Load the mouse detection model
PATH_TO_LABELS = './MyMouseModel/annotations/label_map.pbtxt'
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS,
                                                                    use_display_name=True)
PATH_TO_SAVED_MODEL = './MyMouseModel/model'
# Load saved model and build the detection function (it take close to 50 sec to load)
print('Loading model and initialisation ...', end='')
detect_fn = tf.saved_model.load(PATH_TO_SAVED_MODEL)


def is_a_mouse(image):
    """ Function to determine if there is a mouse in the supplied image
    Input value : Image as a np.array
    return value: Dict {'probability':float, 'bonding_box':[ymin,xmin,ymax,xmax]} """

    # Prepare image as done during model development and training
    initial_size = image.shape[0:2]
    # Rescale the image for the model
    image = tf.image.resize_with_pad(image, 640, 640)
    image = tf.cast(image, dtype=tf.uint8)

    # The model expects a batch of images, so add an axis with `tf.newaxis`.
    input_tensor = image[tf.newaxis, ...]
    # Run the detection model on the image
    detections = detect_fn(input_tensor)
    # All outputs are batches tensors.
    # Convert to numpy arrays, and take index [0] to remove the batch dimension.
    # We're only interested in the first num_detections.
    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                  for key, value in detections.items()}
    prob = detections['detection_scores'][0]
    # Bonding box coordinate relative to the inital image size
    ymin = detections['detection_boxes'][0][0] * 640 - (640 - initial_size[0]) / 2
    xmin = detections['detection_boxes'][0][1] * 640 - (640 - initial_size[1]) / 2
    ymax = detections['detection_boxes'][0][2] * 640 - (640 - initial_size[0]) / 2
    xmax = detections['detection_boxes'][0][3] * 640 - (640 - initial_size[1]) / 2

    return {'probability': prob, 'bonding_box': [ymin, xmin, ymax, xmax]}


# Run a first "dummy" detection to past over the delay experienced on the first run
image_dum = np.array(Image.open('./MouseImage/0S_001_0.jpg'))
result_dum = is_a_mouse(image_dum)
print('  Done! ')  # To indicate end of model loading and initialisation


def calcul_path_to_mouse(bonding_box):
    """ Function to calculate the angle (in degree) and distance (in mm) of the mouse
    based on the bonding box coordinate
    Input values: bonding_box = [ymin,xmin,ymax,xmax]
    Output values: Dict {'angle': float, 'distance': float}"""
    #
    # Calibration specific to my Vector
    # this is the calculated actual head angle when running robot.behavior.set_head_angle(degrees(-10))
    theta_head = 11.7
    head_high = 32.  # in mm
    #
    latitude_angle = 90 - theta_head - 50 * (bonding_box[2] - 180) / 360  # Use bonding-box lower part
    longitude_angle = 90 * (320 - (
            bonding_box[1] + (bonding_box[3] - bonding_box[1]) / 2)) / 640  # Use bonding-box horizontal center
    #
    distance = head_high * np.tan(pi * latitude_angle / 180) / np.absolute(np.cos(pi * longitude_angle / 180))

    return {'longitude_angle': longitude_angle, 'latitude_angle': latitude_angle, 'distance': distance}


# Dictionnaire des différentes phrases pour
my_text = {'greeting': 'Hello Paul',
           'ready': 'I am ready to search for mouse',
           'found': 'Yes I think there is a mouse over there',
           'estimate_1a': 'It is about ',
           'estimate_1b': ' centimeter in front of me',
           'get_closer': 'I will try to get closer',
           'catch_it': 'I will try to catch it',
           'got_it': 'Yes I got it',
           'greatest': 'I am the greatest',
           'miss_it': 'Sorry I missed it',
           'next_time': 'Maybe next time',
           'not_found': 'I did not see any mouse',
           'completed': 'The search is completed'}


# Fonction pour l'expression des phrases
def say_text(robot, text):
    """ Function to have Vector saying the text
    input : robot instance, text"""
    robot.behavior.say_text(text)
    return


# Fonction pour lire les sensors d'intérêt pour déterminer si la souris a été capturée
def read_sensor(robot):
    accel = robot.accel
    gyro = robot.gyro
    robot_pose = robot.pose
    pitch_rad = robot.pose_pitch_rad
    return {'accel': accel, 'gyro': gyro, 'robot_pose': robot_pose, 'pitch_rad': pitch_rad}


# Fonction pour la dance suite à la capture de la souris
def succes_dance(robot):
    anime_list = ['anim_dancebeat_scoot_left_01', 'anim_dancebeat_scoot_right_01',
                  'anim_dancebeat_headliftbody_left_large_01', 'anim_dancebeat_headliftbody_right_large_01']
    for i in range(3):
        for anime in anime_list:
            robot.anim.play_animation(anime)
    robot.anim.play_animation_trigger('GreetAfterLongTime')
    robot.anim.play_animation_trigger('DanceBeatGetIn')
    robot.anim.play_animation_trigger('DanceBeatListening')
    return


def main():
    """
    Main program for mouse search by the Vector robot 
    After greeting and job assessment acknowledgement, Vector start to
    turn in place and look if he sees a mouse.
    If yes, he will move toward it. If not, after a complete 360 degrees rotation,
    he will stop searching.
    """

    # import MouseDetectionSupport as mds
    import time
    import anki_vector
    from anki_vector.util import distance_mm, speed_mmps, degrees

    import warnings
    warnings.filterwarnings('ignore')

    # args = anki_vector.util.parse_command_args()
    # with anki_vector.Robot(serial=args.serial, show_viewer=True) as robot:

    # Create a Robot object
    robot = anki_vector.Robot(show_viewer=True)
    # Connect to the Robot
    robot.connect()

    # Greeting, get ready and job assessment acknowledgement
    say_text(robot, my_text['greeting'])
    robot.behavior.set_head_angle(degrees(-10))  # Move head and lift as a search posture
    robot.behavior.set_lift_height(1, duration=2.0)  # 1 = MAX_LIFT_HEIGHT_MM
    say_text(robot, my_text['ready'])
    #
    angle_offset = -3  # Correction for Vector camera angle offset
    angle = 0
    step = 30  # Rotation step in degree
    mouse_detected = False

    while True:
        image = robot.camera.capture_single_image()
        image = np.array(image.raw_image)

        result = is_a_mouse(image)

        if result['probability'] >= 0.7:
            # Compilation des résultats pour analyse ultérieure
            compil_resultat = {'resultat1': result, 'image1': image}
            mouse_detected = True
            break
        robot.behavior.turn_in_place(degrees(step), speed=degrees(20))
        angle += step
        if angle > 360:
            break

    if mouse_detected:
        # Try to move to catch the mouse
        path_to_mouse = calcul_path_to_mouse(result['bonding_box'])
        compil_resultat['estimer1'] = path_to_mouse

        # Rotate the robot to the first estimate of direction
        robot.behavior.turn_in_place(degrees(path_to_mouse['longitude_angle'] + angle_offset), speed=degrees(40))

        say_text(robot, my_text['found'])
        time.sleep(1)
        say_text(robot, my_text['estimate_1a'] + str(round(path_to_mouse['distance'] / 10)) + my_text['estimate_1b'])
        time.sleep(1)
        say_text(robot, my_text['get_closer'])

        # Move toward mouse by 3/4 of the first distance estimate, but no closer then 30mm 
        move_distance = path_to_mouse['distance']
        if move_distance > 120:
            move_distance = 0.75 * move_distance
        else:
            move_distance = move_distance - 30
        robot.behavior.drive_straight(distance_mm(move_distance), speed_mmps(50))

        # Perform a second estimate of mouse location
        robot.behavior.set_head_angle(degrees(-10))  # Ensure head is down
        image = robot.camera.capture_single_image()
        image = np.array(image.raw_image)

        result = is_a_mouse(image)
        compil_resultat['resultat2'] = result
        compil_resultat['image2'] = image

        path_to_mouse = calcul_path_to_mouse(result['bonding_box'])
        compil_resultat['estimer2'] = path_to_mouse

        # Rotation to the new estimated direction
        robot.behavior.turn_in_place(degrees(path_to_mouse['longitude_angle'] + angle_offset), speed=degrees(40))
        #
        # Check is Vector proximity sensor detect the mouse
        proximity_data = robot.proximity.last_sensor_reading
        if proximity_data is not None:
            compil_resultat['proxi_dist'] = proximity_data.distance

        # Try to catch the mouse
        say_text(robot, my_text['catch_it'])
        move_distance = path_to_mouse['distance']
        robot.behavior.drive_straight(distance_mm(move_distance), speed_mmps(30))

        # Perform sensors reading before and after Vector' attempt to catch the mouse
        sensor_reading = {}
        sensor_reading['before'] = read_sensor(robot)
        robot.behavior.set_lift_height(0, duration=1.0)  # Move lift to catch the mouse
        sensor_reading['after'] = read_sensor(robot)

        if sensor_reading['after']['pitch_rad'] > 0.1:  # Such Vector pitch indicate the mouse is stock under its lift
            say_text(robot, my_text['got_it'])
            # Free the mouse move back and celebrate !
            robot.behavior.set_lift_height(1, duration=2.0)
            say_text(robot, my_text['greatest'])
            robot.behavior.drive_straight(distance_mm(-30), speed_mmps(50))
            succes_dance(robot)
        else:
            say_text(robot, my_text['miss_it'])
            say_text(robot, my_text['next_time'])
    else:
        say_text(robot, my_text['not_found'])
        time.sleep(1)
        say_text(robot, my_text['completed'])

        # Disconnect from Vector
    robot.disconnect()


if __name__ == "__main__":
    main()
