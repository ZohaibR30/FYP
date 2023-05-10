import os
import cv2
import numpy as np
from collections import deque

from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import plot_model

modelPath = os.path.join(os.getcwd(), "LRCN\model.h5")

# Specify the height and width to which each video frame will be resized in our dataset.
IMAGE_HEIGHT , IMAGE_WIDTH = 64, 64

# Specify the number of frames of a video that will be fed to the model as one sequence.
SEQUENCE_LENGTH = 1

CLASSES_LIST = ["no-theft", "theft"]

def create_LRCN_model():
    '''
    This function will construct the required LRCN model.
    Returns:
        model: It is the required constructed LRCN model.
    '''

    # We will use a Sequential model for model construction.
    model = Sequential()
    
    # Define the Model Architecture.
    ########################################################################################################################
    
    model.add(TimeDistributed(Conv2D(16, (3, 3), padding='same',activation = 'relu'),
                              input_shape = (SEQUENCE_LENGTH, IMAGE_HEIGHT, IMAGE_WIDTH, 3)))
    
    model.add(TimeDistributed(MaxPooling2D((4, 4)))) 
    model.add(TimeDistributed(Dropout(0.25)))
    
    model.add(TimeDistributed(Conv2D(32, (3, 3), padding='same',activation = 'relu')))
    model.add(TimeDistributed(MaxPooling2D((4, 4))))
    model.add(TimeDistributed(Dropout(0.25)))
    
    model.add(TimeDistributed(Conv2D(64, (3, 3), padding='same',activation = 'relu')))
    model.add(TimeDistributed(MaxPooling2D((2, 2))))
    model.add(TimeDistributed(Dropout(0.25)))
    
    model.add(TimeDistributed(Conv2D(64, (3, 3), padding='same',activation = 'relu')))
    model.add(TimeDistributed(MaxPooling2D((2, 2))))
    #model.add(TimeDistributed(Dropout(0.25)))
                                      
    model.add(TimeDistributed(Flatten()))
                                      
    model.add(LSTM(32))
                                      
    model.add(Dense(len(CLASSES_LIST), activation = 'softmax'))

    ########################################################################################################################

    # Return the constructed LRCN model.
    return model

def gen_frames(camera):  
    # Construct the required LRCN model.
    LRCN_model = create_LRCN_model()
    
    LRCN_model.load_weights(modelPath)
    
    # Display the success message.
    print("Model Created Successfully!")

    try:
        while True:
            predicted_class_name = ''
            frames_queue = deque(maxlen = SEQUENCE_LENGTH)

            success, frame = camera.read()  # read the camera frame

            if not success:
                break
            else:
                resized_frame = cv2.resize(frame, (IMAGE_HEIGHT, IMAGE_WIDTH))
            
                # Normalize the resized frame by dividing it with 255 so that each pixel value then lies between 0 and 1.
                normalized_frame = resized_frame / 255

                # Appending the pre-processed frame into the frames list.
                frames_queue.append(normalized_frame)

                # Pass the normalized frames to the model and get the predicted probabilities.
                predicted_labels_probabilities = LRCN_model.predict(np.expand_dims(frames_queue, axis = 0))[0]

                # Get the index of class with highest probability.
                predicted_label = np.argmax(predicted_labels_probabilities)

                # Get the class name using the retrieved index.
                predicted_class_name = CLASSES_LIST[predicted_label]

                # Write predicted class name on top of the frame.
                cv2.putText(frame, predicted_class_name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                ret, buffer = cv2.imencode('.jpg', frame)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result
    
    finally:
        camera.release()
