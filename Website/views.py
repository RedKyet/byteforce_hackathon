from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
import json
import os

def home(request):
    return render(request, 'main.html')


def upload(request):
    
    # print(request.FILES['document'])
    
    uploaded_file = request.FILES['document']
    fs = FileSystemStorage()
    
    fs.delete('cache.jpg')
    name = fs.save('cache.jpg', uploaded_file)
    
    # print(fs.url(name))

    return render(request, 'main.html')
