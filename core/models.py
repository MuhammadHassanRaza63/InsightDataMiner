from django.db import models
from django.contrib.auth.models import User

class UserModuleUsage(models.Model):
    MODULE_CHOICES = [
        ('module1', 'Data Preprocessing'),
        ('module2', 'Visual Analytics'),
        ('module3', 'AutoML'),
    ]

    user = models.ForeignKey(User, on_delete=models.CASCADE)
    module = models.CharField(max_length=10, choices=MODULE_CHOICES)
    used_at = models.DateTimeField(auto_now_add=True)
    dataset_name = models.CharField(max_length=255, blank=True, null=True)
    session_data = models.JSONField(blank=True, null=True)  # Store module-specific data

    class Meta:
        unique_together = ['user', 'module']
        ordering = ['-used_at']

    def __str__(self):
        return f"{self.user.username} - {self.module}"

class UserReport(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    generated_at = models.DateTimeField(auto_now_add=True)
    report_data = models.JSONField()  # Store complete report data
    is_downloaded = models.BooleanField(default=False)

    class Meta:
        ordering = ['-generated_at']

    def __str__(self):
        return f"Report for {self.user.username} - {self.generated_at}"
