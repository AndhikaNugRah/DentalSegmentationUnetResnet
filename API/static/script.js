document.addEventListener('DOMContentLoaded', () => {
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const uploadForm = document.getElementById('upload-form');
    const previewImage = document.getElementById('preview-image');
    const outputImage = document.getElementById('output-image');

    const updatePreview = (file) => {
        const reader = new FileReader();
        reader.onload = (e) => {
            previewImage.src = e.target.result;
            previewImage.style.display = 'block';
        };
        reader.readAsDataURL(file);
    };

    dropZone.addEventListener('dragover', (event) => {
        event.preventDefault();
        dropZone.classList.add('bg-blue-50');
    });

    dropZone.addEventListener('dragleave', (event) => {
        dropZone.classList.remove('bg-blue-50');
    });

    dropZone.addEventListener('drop', (event) => {
        event.preventDefault();
        dropZone.classList.remove('bg-blue-50');
        const files = event.dataTransfer.files;
        if (files.length > 0) {
            fileInput.files = files;
            updatePreview(files[0]);
        }
    });

    fileInput.addEventListener('change', (event) => {
        const files = event.target.files;
        if (files.length > 0) {
            updatePreview(files[0]);
        }
    });

    uploadForm.addEventListener('submit', async (event) => {
        event.preventDefault();
        const formData = new FormData();
        formData.append('file', fileInput.files[0]);

        const response = await fetch('/predict/', {
            method: 'POST',
            body: formData
        });

        if (response.ok) {
            const imageBlob = await response.blob();
            const imageUrl = URL.createObjectURL(imageBlob);
            outputImage.src = imageUrl;
            outputImage.classList.remove('hidden');
        } else {
            alert('Failed to upload image.');
        }
    });
});
