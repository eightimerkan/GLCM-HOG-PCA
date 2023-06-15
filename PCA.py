from sklearn.decomposition import PCA
import numpy as np

# Veri kümesini yükle
X = np.loadtxt('ornek_veri_seti.csv', delimiter=',') # ornek_veri_seti.csv yerine kullanmak istediğiniz veri seti dosyasının adını ve yolunu belirtebilirsiniz.

# PCA modelini oluştur
pca = PCA(n_components=2) # PCA'nın kaç bileşen kullanarak yapılacağını belirtebilirsiniz.

# Veriye PCA modelini uygula
X_pca = pca.fit_transform(X)

# PCA sonuçlarını yazdır
print("Orjinal Veri Şekli: ", X.shape)
print("PCA Uygulanan Veri Şekli: ", X_pca.shape)
print("PCA Bileşenleri: ", pca.components_)
print("Varyans Oranları: ", pca.explained_variance_ratio_)