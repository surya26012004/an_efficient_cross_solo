from django.urls import path
from django.conf import settings
from django.conf.urls.static import static
from django.contrib import admin
from Remote_User import views as remoteuser
from Service_Provider import views as serviceprovider

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', remoteuser.login, name="login"),
    path('Register1/', remoteuser.Register1, name="Register1"),
    path('Predict_false_data_injection_attack_detection/', remoteuser.Predict_false_data_injection_attack_detection, name="Predict_false_data_injection_attack_detection"),
    path('ViewYourProfile/', remoteuser.ViewYourProfile, name="ViewYourProfile"),
    path('serviceproviderlogin/', serviceprovider.serviceproviderlogin, name="serviceproviderlogin"),
    path('View_Remote_Users/', serviceprovider.View_Remote_Users, name="View_Remote_Users"),
    path('charts/<str:chart_type>/', serviceprovider.charts, name="charts"),
    path('charts1/<str:chart_type>/', serviceprovider.charts1, name="charts1"),
    path('likeschart/<str:like_chart>/', serviceprovider.likeschart, name="likeschart"),
    path('View_false_data_injection_attack_detection_Ratio/', serviceprovider.View_false_data_injection_attack_detection_Ratio, name="View_false_data_injection_attack_detection_Ratio"),
    path('train_model/', serviceprovider.train_model, name="train_model"),
    path('View_false_data_injection_attack_detection_Type/', serviceprovider.View_false_data_injection_attack_detection_Type, name="View_false_data_injection_attack_detection_Type"),
    path('Download_Trained_DataSets/', serviceprovider.Download_Trained_DataSets, name="Download_Trained_DataSets"),
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
