# Pain-Recognition-Facial-Expression
Total Diz Protezi Ameliyatı Olan Hastalarda Yapay Zeka Destekli Çok Modlu Ağrı Değerlendirmesi - Yüz İfadesinden Ağrı Tespiti

# Yüz İfadelerinden Yapay Zeka Destekli Ağrı Tespiti

Total Diz Protezi (TDP) ameliyatı olan hastalarda yapay zeka destekli çok modlu ağrı değerlendirmesi projesi kapsamında, yüz ifadelerinden ağrı tespiti yapabilen bir derin öğrenme modeli geliştirilmiştir. Model, MobileNetV2 tabanlı transfer öğrenme yaklaşımı kullanılarak eğitilmiş ve test setinde **%87.5 doğruluk** ile **0.91 AUC-ROC** değerine ulaşmıştır.

## İçindekiler

- [Proje Hakkında](#proje-hakkında)
- [Veri Seti](#veri-seti)
- [Model Mimarisi](#model-mimarisi)
- [Kurulum](#kurulum)
- [Kullanım](#kullanım)
- [Nihai Sonuçlar](#nihai-sonuçlar)
- [Proje Yapısı](#proje-yapısı)
- [Kaynaklar](#kaynaklar)

---

## Proje Hakkında

Bu proje, T.C. Yozgat Bozok Üniversitesi Bilgisayar Mühendisliği Bölümü'nde yürütülen bir TÜBİTAK projesi kapsamında gerçekleştirilmiştir. Projenin amacı, Total Diz Protezi ameliyatı sonrası hastaların ağrı seviyelerini, yüz ifadelerinden otomatik olarak tespit edebilen bir yapay zeka modeli geliştirmektir.

Ağrı, öznel bir deneyim olduğu için hastanın kendi ifadesine dayanan klasik ölçekler, hastanın kendini ifade edemediği durumlarda yetersiz kalmaktadır. Yüz ifadeleri, ağrının dışavurumunda en güvenilir biyolojik sinyallerden biridir. Bu projede, yüz ifadelerindeki Action Unit (AU) hareketlerinden PSPI (Prkachin ve Solomon Pain Intensity) skoru hesaplanarak ağrı tespiti yapılmıştır.

## Veri Seti

Projede Kaggle üzerinde bulunan **"Pain Recognition from Facial Expression Dataset"** kullanılmıştır. Veri seti şu bileşenlerden oluşmaktadır:

- **Frame_Labels/FACS/**: Her hasta için FACS (Facial Action Coding System) kodlamaları
- **Frame_Labels/PSPI/**: Her hasta için PSPI skorları
- **Images/**: Her hastaya ait yüz görüntüleri (PNG formatında)

PSPI skoru şu formülle hesaplanmaktadır:
Pain = AU4 + (AU6 veya AU7) + (AU9 veya AU10) + AU43


PSPI skoru 0 ("ağrı yok") ile 16 ("şiddetli ağrı") arasında değişmektedir. Projede ikili sınıflandırma yapılmıştır: PSPI = 0 → "ağrı yok", PSPI > 0 → "ağrı var".

## Model Mimarisi

Model, **MobileNetV2** tabanlı transfer öğrenme yaklaşımı kullanılarak geliştirilmiştir. MobileNetV2'nin tercih edilmesinin nedeni, yüksek doğruluk oranına ulaşırken model boyutu ve hesaplama maliyeti açısından hafif olmasıdır.

**Mimari Detayları:**
- Ön-eğitilmiş MobileNetV2 (ImageNet ağırlıkları)
- GlobalAveragePooling2D
- Dropout (0.5)
- Dense (1, sigmoid aktivasyon)

**Eğitim Parametreleri:**
- Optimizer: Adam (learning_rate = 5e-5)
- Kayıp Fonksiyonu: binary_crossentropy
- Batch Size: 32
- Epoch: 15 (erken durdurma ile)
- Erken Durdurma: patience = 3

## Kurulum

Projeyi çalıştırmak için gerekli kütüphaneler:
pip install tensorflow keras opencv-python numpy pandas matplotlib scikit-learn

Kullanım

1. Veri Setinin Yüklenmesi
Kaggle'dan veri setini indirin ve data/ klasörüne yerleştirin.

2. Model Eğitimi
python models/train.py

3. Model Değerlendirmesi
python evaluation/metrics.py

5. Grad-CAM Görselleştirmesi
python evaluation/gradcam.py

Nihai Sonuçlar
Metrik	Değer
Doğruluk (Accuracy)	%87.5
AUC-ROC	0.91
Hassasiyet (Precision)	0.86
Duyarlılık (Recall)	0.89
F1-Skoru	0.87
Grad-CAM Analizi: Modelin ağırlıklı olarak kaş (AU4), göz çevresi (AU6, AU7) ve ağız bölgesine (AU10, AU43) odaklandığı görülmüştür. Bu bölgeler, literatürde ağrı ile en güçlü ilişkilendirilen Action Unit'lerin bulunduğu bölgelerdir.

Hata Analizi: Modelin hafif ağrı vakalarında (PSPI = 1-2) zorlandığı, şiddetli ağrı vakalarında ise yüksek doğrulukla çalıştığı tespit edilmiştir.

## Proje Yapısı
pain-recognition-facial-expression/

│
├── data/                    # Veri ön işleme kodları

├── models/                  # Model mimarisi ve eğitim kodları

├── evaluation/              # Değerlendirme ve görselleştirme kodları

├── utils/                   # Yardımcı fonksiyonlar

├── notebooks/               # Keşifsel veri analizi

├── results/                 # Sonuç grafikleri

├── README.md

└── requirements.txt

## Kaynaklar

[1] G. D. De Sario et al., "Using AI to Detect Pain through Facial Expressions: A Review," Bioengineering, vol. 10, no. 5, p. 548, May 2023.

[2] M. Cascella et al., "Artificial intelligence for pain assessment via facial expression recognition (2015–2025): a systematic review," Exploration of Medicine, vol. 6, 2025.

[3] R. AL-Edwan and I. Jafar, "Re-Evaluating Single-Backbone Transfer Learning for Pain Assessment from Facial Expressions," in Proc. IEEE AEECT, 2026, pp. 90–94.

[4] A. Semwal and N. D. Londhe, "Automated Pain Severity Detection Using Convolutional Neural Network," in Proc. IEEE CTEMS, 2018, pp. 66–70.

[5] Y. R. Chavan and C. S. Pawar, "HCDCN: Image-based pain intensity detection from facial expressions using deep learning," Biomedical Signal Processing and Control, vol. 120, p. 110232, Jul. 2026.

