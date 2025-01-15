document.getElementById('donationForm').addEventListener('submit', function(event) {
    event.preventDefault(); // Prevent form submission from reloading the page
  
    // Hide the form and display the thank you message
    document.getElementById('donationForm').style.display = 'none';
    document.getElementById('thankYouMessage').classList.remove('hidden');
  });
  