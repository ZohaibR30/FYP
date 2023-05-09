import cv2
import numpy as np
from collections import deque
from tensorflow import keras

IMG_SIZE = 224
BATCH_SIZE = 64
MAX_SEQ_LENGTH = 10
NUM_FEATURES = 2048

def build_feature_extractor():
    feature_extractor = keras.applications.InceptionV3(
        weights="imagenet",
        include_top=False,
        pooling="avg",
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
    )
    preprocess_input = keras.applications.inception_v3.preprocess_input

    inputs = keras.Input((IMG_SIZE, IMG_SIZE, 3))
    preprocessed = preprocess_input(inputs)

    outputs = feature_extractor(preprocessed)
    return keras.Model(inputs, outputs, name="feature_extractor")

def get_sequence_model():    
    # Refer to the following tutorial to understand the significance of using `mask`:
    # https://keras.io/api/layers/recurrent_layers/gru/
    
    class_vocab = ['no-theft', 'theft']
    frame_features_input = keras.Input((MAX_SEQ_LENGTH, NUM_FEATURES))
    mask_input = keras.Input((MAX_SEQ_LENGTH,), dtype="bool")

    x = keras.layers.GRU(16, return_sequences=True)(frame_features_input, mask=mask_input)
    x = keras.layers.GRU(8)(x)
    x = keras.layers.Dropout(0.4)(x)
    x = keras.layers.Dense(8, activation="relu")(x)
    output = keras.layers.Dense(len(class_vocab), activation="softmax")(x)

    rnn_model = keras.Model([frame_features_input, mask_input], output)
    return rnn_model

model = get_sequence_model()
feature_extractor = build_feature_extractor()
model.load_weights(r'D:\FYP\GRU\GRUmodel.h5')

def prepare_single_video(frames):
    frames = frames[None, ...]
    frame_mask = np.zeros(shape=(1, MAX_SEQ_LENGTH,), dtype="bool")
    frame_features = np.zeros(shape=(1, MAX_SEQ_LENGTH, NUM_FEATURES), dtype="float32")

    for i, batch in enumerate(frames):
        video_length = batch.shape[0]
        length = min(MAX_SEQ_LENGTH, video_length)
        for j in range(length):
            frame_features[i, j, :] = feature_extractor.predict(batch[None, j, :])
        frame_mask[i, :length] = 1  # 1 = not masked, 0 = masked

    return frame_features, frame_mask

def crop_center_square(frame):
    y, x = frame.shape[0:2]
    min_dim = min(y, x)
    start_x = (x // 2) - (min_dim // 2)
    start_y = (y // 2) - (min_dim // 2)
    return frame[start_y : start_y + min_dim, start_x : start_x + min_dim]


def load_video(camera, max_frames=0, resize=(IMG_SIZE, IMG_SIZE)):
    class_vocab = ['no-theft', 'theft']
    frames = deque(maxlen = MAX_SEQ_LENGTH)

    try:
        while True:
            ret, frame = camera.read()
            frame_return = frame

            if not ret:
                break

            frame = crop_center_square(frame)
            frame = cv2.resize(frame, resize)
            frame = frame[:, :, [2, 1, 0]]
            frame = frame / 255
            frames.append(frame)

            frame_features, frame_mask = prepare_single_video(np.array(frames))
            probabilities = model.predict([frame_features, frame_mask])[0]

            for i in np.argsort(probabilities)[::-1]:
                # print(f"  {i, class_vocab[i]}: {probabilities[i] * 100:5.2f}%")
                
                predicted_label = np.argmax(i)
                predicted_class_name = class_vocab[predicted_label]   

                cv2.putText(frame_return, predicted_class_name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)                    
                ret, buffer = cv2.imencode('.jpg', frame_return)
                frame = buffer.tobytes()

                yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result

    finally:
        camera.release()