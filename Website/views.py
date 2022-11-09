from django.shortcuts import render
#from Website.static import placeholder # temporary

def home(request):
    return render(request, 'main.html')
from django.core.files.storage import FileSystemStorage

def upload(request):
    import json
    print(request.FILES['document'])
    uploaded_file = request.FILES['document']
    fs = FileSystemStorage()
    name = fs.save(uploaded_file.name, uploaded_file)
    print(fs.url(name))

    #uploaded_file = request.FILES
    #fs = FileSystemStorage()
    #name = fs.save(uploaded_file.name, uploaded_file)
    #context['url'] = fs.url(name)
    return render(request, 'main.html')
