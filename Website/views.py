from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
import json
import os



def home(request):
    
    return render(request, 'home.html')

def about(request):
    
    return render(request, 'about.html')

def main(request):
    import uuid
    import secrets
    if not 'id' in request.session:
        id = secrets.token_urlsafe(8)
        request.session['id']=id
    
    # Printing random id using uuid1()
    
    print (request.session['id'])
    import uuid
  
    # Printing random id using uuid1()
    print ("The random id using uuid1() is : ",end="")
    print (uuid.uuid1())
    return render(request, 'main.html')

def upload(request):
    print('dadada otelu e viata mea')
    uploaded_file = request.FILES['document']
    fs = FileSystemStorage()
    path = 'users/'+str(request.session['id'])+'/cache.jpg'
    fs.delete(path)
    name = fs.save(path, uploaded_file)

    return render(request, 'main.html')

def contact(request):
    
    return render(request, 'contact.html')