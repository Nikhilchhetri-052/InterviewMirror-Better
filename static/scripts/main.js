document.addEventListener('DOMContentLoaded', () => {
    console.log("InterviewMirror project loaded. Enjoy the modern look!");

    // Handle all role button clicks
    const roleButtons = document.querySelectorAll('.role-btn');

    roleButtons.forEach(button => {
        button.addEventListener('click', () => {
            const role = button.getAttribute('data-role');
            console.log('Role button clicked, role:', role);

            // Decide URL based on login status
            const isLoggedIn = document.body.dataset.authRequired === "true"; 
            if (isLoggedIn) {
                window.location.href = `/interviewafterlogin?role=${role}`;
            } else {
                window.location.href = `/interview?role=${role}`;
            }
        });
    });
});
