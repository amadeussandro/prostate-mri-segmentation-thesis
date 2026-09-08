import nibabel as nib
import matplotlib.pyplot as plt

file_path = r"..\dataset\raw-prostate158_train\train\020\t2_tumor_reader1.nii.gz"

img = nib.load(file_path)

data = img.get_fdata()

print("=== INFORMASI MRI ===")
print("Shape:", data.shape)
print("Data type:", data.dtype)
print("Min:", data.min())
print("Max:", data.max())
print("Affine:")
print(img.affine)

slice_index = data.shape[2] // 2

plt.imshow(data[:, :, slice_index], cmap="gray")
plt.title(f"Case 100 - ADC MRI - Slice {slice_index}")
plt.axis("off")
plt.show()