from django.shortcuts import render
from Website.static import placeholder # temporary

def home(request):
    
   
    return render(request, {placeholder})
