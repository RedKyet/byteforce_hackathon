from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
import json
import os

def main(request):
    
    return render(request, 'main.html')

def home(request):
    
    return render(request, 'home.html')

def upload(request):
    
    uploaded_file = request.FILES['document']
    fs = FileSystemStorage()
    
    fs.delete('cache.jpg')
    name = fs.save('cache.jpg', uploaded_file)

    return render(request, 'main.html')
