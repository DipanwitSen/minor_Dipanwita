// Blood Donation Interaction Script
const selectedBloodElement = document.getElementById("selected-blood");
const donationForm = document.getElementById("donation-form");

// Blood type selection function
function selectBloodType(type) {
    selectedBloodElement.textContent = type;
    alert(`You selected ${type} as the donor blood type.`);
}

// Form submission handler
donationForm.addEventListener("submit", function (event) {
    event.preventDefault();

    // Collect form data
    const amount = document.getElementById("amount").value;
    const name = document.getElementById("name").value;
    const contact = document.getElementById("contact").value;
    const location = document.getElementById("location").value;
    const date = document.getElementById("date").value;

    // Display form submission data
    alert(`Thank you, ${name}, for donating ${amount} ml of blood!`);
    console.log({
        bloodType: selectedBloodElement.textContent,
        amount,
        name,
        contact,
        location,
        date,
    });

    // Reset form
    donationForm.reset();
    selectedBloodElement.textContent = "None";
});
