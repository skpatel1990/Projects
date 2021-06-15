import streamlit as st
import os
import pandas as pd
import tensorflow as tf
from keras_preprocessing.image import ImageDataGenerator
import keras
from PIL import Image, ImageOps
path_to_pic = 'data/temp_file_webapp'

all_breeds = pd.DataFrame({'breed': ['affenpinscher', 'afghan_hound', 'african_hunting_dog',
 'airedale','american_staffordshire_terrier','appenzeller','australian_terrier','basenji','basset',
 'beagle','bedlington_terrier','bernese_mountain_dog','black-and-tan_coonhound','blenheim_spaniel','bloodhound',
 'bluetick','border_collie','border_terrier','borzoi','boston_bull','bouvier_des_flandres','boxer','brabancon_griffon',
 'briard', 'brittany_spaniel', 'bull_mastiff', 'cairn', 'cardigan', 'chesapeake_bay_retriever', 'chihuahua', 'chow',
 'clumber', 'cocker_spaniel', 'collie', 'curly-coated_retriever', 'dandie_dinmont', 'dhole', 'dingo', 'doberman', 'english_foxhound',
 'english_setter','english_springer','entlebucher','eskimo_dog','flat-coated_retriever','french_bulldog','german_shepherd','german_short-haired_pointer','giant_schnauzer',
 'golden_retriever','gordon_setter','great_dane','great_pyrenees','greater_swiss_mountain_dog','groenendael','ibizan_hound','irish_setter','irish_terrier',
 'irish_water_spaniel','irish_wolfhound','italian_greyhound','japanese_spaniel','keeshond','kelpie','kerry_blue_terrier','komondor','kuvasz','labrador_retriever',
 'lakeland_terrier','leonberg','lhasa','malamute','malinois','maltese_dog','mexican_hairless','miniature_pinscher','miniature_poodle','miniature_schnauzer','newfoundland',
 'norfolk_terrier','norwegian_elkhound','norwich_terrier','old_english_sheepdog','otterhound','papillon','pekinese','pembroke','pomeranian','pug',
 'redbone','rhodesian_ridgeback','rottweiler','saint_bernard','saluki','samoyed','schipperke','scotch_terrier','scottish_deerhound','sealyham_terrier','shetland_sheepdog',
 'shih-tzu','siberian_husky','silky_terrier','soft-coated_wheaten_terrier','staffordshire_bullterrier','standard_poodle','standard_schnauzer','sussex_spaniel',
 'tibetan_mastiff','tibetan_terrier','toy_poodle','toy_terrier','vizsla','walker_hound','weimaraner','welsh_springer_spaniel','west_highland_white_terrier','whippet','wire-haired_fox_terrier',
 'yorkshire_terrier']})

@st.cache(allow_output_mutation=True)  # 👈 Added this
def get_model():
    model = tf.keras.models.load_model("data/updated_model", compile = False)
    session = tf.compat.v1.keras.backend.get_session()
    return model

reconstructed_model = get_model()
####
def predict_dog(image_name):
    img_size = (331, 331)
    test_df = pd.DataFrame({'id': [image_name], 'breed': ['unknown']})

    test_datagen=ImageDataGenerator(rescale=1./255.)
    test_generator = test_datagen.flow_from_dataframe(
        dataframe = test_df,
        directory = path_to_pic,
        x_col="id",
        y_col = None,
        seed = 14,
        shuffle = False,
        class_mode = None,
        target_size = img_size,
        color_mode="rgb"
    )

    shape = (331, 331, 3)
    pred = reconstructed_model.predict(test_generator)
    predicted_df = pd.DataFrame(pred, columns = all_breeds.breed)
    final_preds = predicted_df.idxmax(axis = 1)
    to_display = (predicted_df.reset_index().set_index(['index'])
       .rename_axis(['breed'], axis=1)
       .stack()
       .unstack('index')
       .reset_index()
       .filter(['breed', 0])
       .sort_values(by = 0, ascending = False)
       .rename(columns = {0 : 'probability'})
       .head())
    
    return (to_display.breed.values[[0]][0], to_display)
####

st.write("""
         # Dog Breed Classifier
         """
         )
st.write("This is a simple image classification web app to predict dog breed")

file = st.file_uploader("Please upload an image file", type=["jpg", "png"])


if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    if(image.format == "JPEG"):
        image_name = 'temp_file.jpg'
    elif(image.format == "PNG"):
        image_name = 'temp_file.png'
    else:
        st.text("Please Upload the correct format of file")
    image.save(os.path.join(path_to_pic, image_name))
    output = predict_dog(image_name)
    st.text("That looks like a " + str(output[0]) + "!")
    st.image(image, width=None)
    st.text("Top 5 Predicted Dog Breeds")
    st.write(output[1].style.hide_index())
    os.remove(os.path.join(path_to_pic, image_name))