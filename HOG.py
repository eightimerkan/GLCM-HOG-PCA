from skimage.feature import hog
from skimage import exposure
import matplotlib.pyplot as plt
import cv2

# Görüntüyü oku
image = cv2.imread('ornek_goruntu.jpg', cv2.IMREAD_GRAYSCALE)

# HOG parametrelerini ayarla
orientations = 9
pixels_per_cell = (8, 8)
cells_per_block = (2, 2)

# HOG hesapla
fd, hog_image = hog(image, orientations=orientations, pixels_per_cell=pixels_per_cell,
                    cells_per_block=cells_per_block, visualize=True)

# HOG görüntüsünü normalize et
hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))

# Orjinal ve HOG görüntülerini yan yana göster
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5), sharex=True, sharey=True)

ax1.imshow(image, cmap='gray')
ax1.set_title('Orjinal Görüntü')

ax2.imshow(hog_image_rescaled, cmap='gray')
ax2.set_title('HOG Görüntüsü')

plt.show()