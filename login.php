<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Login and Signup</title>
    <link rel="stylesheet" href="login.css">
    <style> h2{color:rgb(64, 64, 64)}</style>
</head>
<body>
    <div class="container">
        <!-- Logo Section -->
        <img src="login-logo.jpg" alt="Logo" width="400" height="200">
 <!-- Check for error message -->
 <?php
 session_start();
 if (isset($_SESSION['error'])) {
     echo '<p class="error">' . $_SESSION['error'] . '</p>';
     unset($_SESSION['error']); // Clear error after displaying
 }

 if (isset($_SESSION['success'])) {
     echo '<p class="success">' . $_SESSION['success'] . '</p>';
     unset($_SESSION['success']); // Clear success message after displaying
 }
 ?>
        <!-- Login Form -->
        <div class="login-form">
            <h2>Login</h2>
            <form action="login-process.php" method="post">
                <input type="text" name="username" placeholder="Username/Email" required>
                <input type="password" name="password" placeholder="Password" required>
                <button type="submit">Login</button>
                <a href="#" class="forgot-password">Forgot Password?</a>
            </form>
            <p>Not a member yet? <a href="#" id="signup-link">Sign up</a></p>
        </div>

        <!-- Signup Form -->
        <div class="signup-form" style="display: none;">
            <h2>Sign Up</h2>
            <form action="signup.php" method="post">
                <input type="email" name="email" placeholder="Email" required>
                <input type="password" name="password" placeholder="Password" required>
                <input type="password" name="confirm_password" placeholder="Confirm Password" required>
                <button type="submit">Sign Up</button>
            </form>
            <p>Already have an account? <a href="#" id="login-link">Login</a></p>
        </div>
    </div>

    <script>
        // Toggle between login and signup forms
        const signupLink = document.getElementById('signup-link');
        const loginLink = document.getElementById('login-link');
        const loginForm = document.querySelector('.login-form');
        const signupForm = document.querySelector('.signup-form');

        signupLink.addEventListener('click', () => {
            loginForm.style.display = 'none';
            signupForm.style.display = 'block';
        });

        loginLink.addEventListener('click', () => {
            signupForm.style.display = 'none';
            loginForm.style.display = 'block';
        });
    </script>
</body>
</html>
