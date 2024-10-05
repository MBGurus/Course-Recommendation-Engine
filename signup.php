<?php
// Start the session
session_start();

// Database connection
$host = 'localhost';
$dbname = 'career_guidance';
$username = 'root'; // Change to your database username
$password = ''; // Change to your database password

try {
    $pdo = new PDO("mysql:host=$host;dbname=$dbname", $username, $password);
    $pdo->setAttribute(PDO::ATTR_ERRMODE, PDO::ERRMODE_EXCEPTION);
} catch (PDOException $e) {
    die("Connection failed: " . $e->getMessage());
}

if ($_SERVER['REQUEST_METHOD'] == 'POST') {
    $email = $_POST['email'];
    $password = $_POST['password'];
    $confirm_password = $_POST['confirm_password'];

    // Check if passwords match
    if ($password != $confirm_password) {
        $_SESSION['error'] = 'Passwords do not match.';
        header('Location: login.html');
        exit();
    }

    // Check if the email already exists in the database
    $stmt = $pdo->prepare('SELECT COUNT(*) FROM users WHERE email = :email');
    $stmt->bindParam(':email', $email);
    $stmt->execute();
    $count = $stmt->fetchColumn();

    if ($count > 0) {
        // Email already exists, set session error message and redirect back to the signup page
        $_SESSION['error'] = 'User exists.';
        header('Location: login.php');
        exit();
    }

    // Hash the password
    $hashed_password = password_hash($password, PASSWORD_DEFAULT);

    // Insert user into the database
    $stmt = $pdo->prepare('INSERT INTO users (email, password) VALUES (:email, :password)');
    $stmt->bindParam(':email', $email);
    $stmt->bindParam(':password', $hashed_password);

    if ($stmt->execute()) {
        // Redirect to login page after successful signup
        $_SESSION['success'] = 'Signup successful! Please log in.';
        header('Location: index.php');
        exit();
    } else {
        $_SESSION['error'] = 'Error during signup.';
        header('Location: login.php');
        exit();
    }
}
?>


