from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from django.utils import timezone
from django.core.paginator import Paginator
from django.db.models import Avg
from PIL import Image as PILImage
import os
from .model_loader import get_model
from .models import Image


def index(request):
    """Main upload page."""
    return render(request, 'classifier/index.html')


def api_docs(request):
    """API documentation page."""
    return render(request, 'classifier/api_docs.html')


def classify(request):
    """Handle image upload and classification."""
    if request.method != 'POST':
        return JsonResponse({'error': 'POST method required'}, status=405)
    
    if 'image' not in request.FILES:
        return JsonResponse({'error': 'No image provided'}, status=400)
    
    image_file = request.FILES['image']
    
    # Validate file extension
    allowed_ext = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']
    file_ext = os.path.splitext(image_file.name)[1].lower()
    
    if file_ext not in allowed_ext:
        return JsonResponse({
            'error': f'Invalid file type. Allowed: {", ".join(allowed_ext)}'
        }, status=400)
    
    try:
        # Save uploaded file temporarily
        file_path = default_storage.save(
            f'uploads/{image_file.name}', 
            ContentFile(image_file.read())
        )
        full_path = os.path.join(default_storage.location, file_path)
        
        # Get image dimensions
        with PILImage.open(full_path) as img:
            width, height = img.size
        
        # Get file size
        file_size = os.path.getsize(full_path)
        
        # Get model and predict
        model = get_model()
        top_k = int(request.POST.get('top_k', 5))
        predictions = model.predict(full_path, top_k=top_k)
        
        # Format response
        results = [
            {
                'class': class_name,
                'confidence': float(prob),
                'confidence_percent': f"{prob * 100:.2f}%"
            }
            for class_name, prob in predictions
        ]
        
        # Check if top prediction is "Miscellaneous Trash"
        if results and results[0]['class'] == 'Miscellaneous Trash':
            # Get client IP
            x_forwarded_for = request.META.get('HTTP_X_FORWARDED_FOR')
            if x_forwarded_for:
                ip_address = x_forwarded_for.split(',')[0]
            else:
                ip_address = request.META.get('REMOTE_ADDR')
            
            # Get user agent
            user_agent = request.META.get('HTTP_USER_AGENT', '')
            
            # Move file to permanent location (already saved in media/uploads/)
            # File is already at: full_path
            # The file_path variable contains the relative path from MEDIA_ROOT
            
            # Create database entry with just the file path
            Image.objects.create(
                image=file_path,  # Store relative path only
                filename=image_file.name,
                file_size=file_size,
                width=width,
                height=height,
                top_prediction=results[0]['class'],
                confidence=results[0]['confidence'],
                all_predictions=results,
                uploaded_at=timezone.now(),
                classified_at=timezone.now(),
                ip_address=ip_address,
                user_agent=user_agent
            )
            
            # Don't delete the file - keep it in storage
            
            return JsonResponse({
                'success': True,
                'predictions': results,
                'top_prediction': results[0],
                'saved_to_database': True,
                'message': 'Image classified as Miscellaneous Trash and saved to database'
            })
        
        # Clean up uploaded file (not Miscellaneous Trash)
        default_storage.delete(file_path)
        
        return JsonResponse({
            'success': True,
            'predictions': results,
            'top_prediction': results[0] if results else None,
            'saved_to_database': False
        })
        
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)


@csrf_exempt
def api_predict(request):
    """API endpoint for image classification."""
    return classify(request)


def dashboard(request):
    """Dashboard to view all Miscellaneous Trash images."""
    # Get all images ordered by most recent first
    images_list = Image.objects.all().order_by('-uploaded_at')
    
    # Pagination - 20 images per page
    paginator = Paginator(images_list, 20)
    page_number = request.GET.get('page', 1)
    images = paginator.get_page(page_number)
    
    # Calculate statistics
    total_images = Image.objects.count()
    avg_confidence = Image.objects.aggregate(avg=Avg('confidence'))['avg']
    
    context = {
        'images': images,
        'total_images': total_images,
        'avg_confidence': avg_confidence,
        'page_obj': images,
    }
    
    return render(request, 'classifier/dashboard.html', context)
