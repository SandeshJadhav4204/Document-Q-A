document.addEventListener('DOMContentLoaded', function() {
    const uploadForm = document.getElementById('uploadForm');
    const uploadStatus = document.getElementById('uploadStatus');
    const questionForm = document.getElementById('questionForm');
    const questionStatus = document.getElementById('questionStatus');
    const newUploadForm = document.getElementById('newUploadForm');
    const newUploadStatus = document.getElementById('newUploadStatus');
    const uploadAnother = document.getElementById('uploadAnother');

    uploadForm.addEventListener('submit', function(event) {
        uploadStatus.innerText = 'Uploading document... Please wait.';
        uploadStatus.style.color = '#007bff';
    });

    questionForm.addEventListener('submit', function(event) {
        questionStatus.innerText = 'Fetching response... Please wait.';
        questionStatus.style.color = '#007bff';
    });

    newUploadForm.addEventListener('submit', function(event) {
        newUploadStatus.innerText = 'Uploading new document... Please wait.';
        newUploadStatus.style.color = '#007bff';
    });

    // Hide the "Upload another file" option initially
    if (uploadAnother) {
        uploadAnother.style.display = 'none';
    }

    // Show the "Upload another file" option after the first file is processed
    if ("{{ current_file }}") {
        uploadAnother.style.display = 'block';
    }
});
