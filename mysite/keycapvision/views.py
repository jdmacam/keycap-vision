from django.shortcuts import render, redirect
from django.http import HttpResponse
from django.conf import settings # allows access to variables in settings.py
from django.core.files.storage import default_storage # for storing the uploaded images to server

import numpy as np
import tensorflow as tf
from tensorflow import keras
import random
import string

# Create your views here.

def landing(request):
    return render(request, 'landing.html')

def upload(request):
    context = {}
    if request.method == 'POST':
        # get image file from the request
        f=request.FILES['file'] 
        # generate name of the temporary image file with unique name for this request
        letters = string.ascii_lowercase
        file_name = "processing/" + ''.join(random.choice(letters) for i in range(5)) + ".jpg" 
        # save to default directory (mysite/media)
        file_name_2 = default_storage.save(file_name, f) 
        file_url = 'mysite' + default_storage.url(file_name_2)

        #img dimensions for the model input
        img_height = 200
        img_width = 300
        img = tf.keras.utils.load_img(file_url, target_size=(img_height, img_width)) # load image
        img_array = tf.keras.utils.img_to_array(img) # convert to array to send thru model
        img_array = tf.expand_dims(img_array, 0) # Create a batch

        predictions = settings.IC_MODEL.predict(img_array)
        score = tf.nn.softmax(predictions[0])
        
        context['prediction'] = settings.IC_CLASSES[np.argmax(score)]
        context['confidence'] = str(round(100 * np.max(score),3))
        all_scores = {}
        for idx,class_name in enumerate(settings.IC_CLASSES):
            all_scores[class_name] = round(100*score.numpy()[idx],3)
        context['all_scores'] = all_scores
        context['prediction_message'] = "Prediction results: " + context['prediction'] + ' with a ' + context['confidence'] + '% confidence'
        context['all_scores_message'] = "Here are the results of the models predictions:"
        default_storage.delete(file_name)
        return render(request,'results.html', context=context)
    else:
        return render(request,'upload.html')

def results(request):
    return render(request, 'results.html')

def upload_reload(request):
    return redirect('/upload')

def keeb_list(request):
    context = {}
    context['classes'] = settings.IC_CLASSES
    return render(request, 'keeb_list.html', context=context)

def jesse(request):
    return render(request, 'jesse.html')
