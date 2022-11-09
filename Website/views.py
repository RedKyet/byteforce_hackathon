from django.shortcuts import render
#from Website.static import placeholder # temporary

def home(request):
    return render(request, 'main.html')


def file(request):
    handle_uploaded_file(request.FILES)

def handle_uploaded_file(f):
    with open('name.txt', 'wb+') as destination:
        for chunk in f.chunks():
            destination.write(chunk)
from django.core.files.storage import FileSystemStorage
def upload(request):
    import json
    print(json.dumps(request.GET['document']))
    print(json.dumps(request.FILES))
    print(json.dumps(request.body))

    #uploaded_file = request.FILES
    #fs = FileSystemStorage()
    #name = fs.save(uploaded_file.name, uploaded_file)
    #context['url'] = fs.url(name)
    return render(request, 'main.html')
