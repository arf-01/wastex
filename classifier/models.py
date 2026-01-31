from django.db import models
from django.utils import timezone


class Image(models.Model):
    """Store uploaded images and their classification results."""
    
    # Image file
    image = models.ImageField(upload_to='uploads/%Y/%m/%d/')
    
    # Original filename
    filename = models.CharField(max_length=255)
    
    # File size in bytes
    file_size = models.IntegerField(null=True, blank=True)
    
    # Image dimensions
    width = models.IntegerField(null=True, blank=True)
    height = models.IntegerField(null=True, blank=True)
    
    # Classification results
    top_prediction = models.CharField(max_length=100, null=True, blank=True)
    confidence = models.FloatField(null=True, blank=True)
    all_predictions = models.JSONField(null=True, blank=True)
    
    # Timestamps
    uploaded_at = models.DateTimeField(default=timezone.now)
    classified_at = models.DateTimeField(null=True, blank=True)
    
    # Metadata
    ip_address = models.GenericIPAddressField(null=True, blank=True)
    user_agent = models.TextField(null=True, blank=True)
    
    class Meta:
        db_table = 'images'
        ordering = ['-uploaded_at']
        verbose_name = 'Image'
        verbose_name_plural = 'Images'
    
    def __str__(self):
        return f"{self.filename} - {self.top_prediction or 'Not classified'}"
    
    def save(self, *args, **kwargs):
        # Set filename if not already set
        if not self.filename and self.image:
            self.filename = self.image.name
        super().save(*args, **kwargs)
