import tensorflow as tf
import numpy as np
import cv2

# Load the pre-trained Show and Tell model
model = tf.keras.applications.InceptionV3(include_top=False, weights='imagenet')
encoder = tf.keras.models.Model(inputs=model.input, outputs=model.output)
decoder = tf.keras.models.load_model('show_and_tell_decoder.h5')

# Load the movie
video_file = 'my_movie.mp4'
video_cap = cv2.VideoCapture(video_file)

# Set up variables for video processing
frame_count = int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
frame_width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Process each frame of the movie
for i in range(frame_count):
    # Read the next frame
    ret, frame = video_cap.read()
    if not ret:
        break
    
    # Resize and preprocess the frame
    frame = cv2.resize(frame, (299, 299))
    frame = np.expand_dims(frame, axis=0)
    frame = tf.keras.applications.inception_v3.preprocess_input(frame)
    
    # Use the encoder to generate image features
    features = encoder.predict(frame)
    
    # Use the decoder to generate a caption
    caption = decoder.predict(features)[0]
    caption = ' '.join([tokenizer.index_word[i] for i in np.argmax(caption, axis=1)])
    
    # Print the caption
    print(caption)
    
# Release resources
video_cap.release()
