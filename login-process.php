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
    $username_or_email = $_POST['username'];
    $password = $_POST['password'];

    // Check if the email exists in the database
    $stmt = $pdo->prepare('SELECT * FROM users WHERE email = :email');
    $stmt->bindParam(':email', $username_or_email);
    $stmt->execute();
    $user = $stmt->fetch(PDO::FETCH_ASSOC);

    if ($user && password_verify($password, $user['password'])) {
        // Password is correct, set session variables and redirect to index page
        $_SESSION['user_id'] = $user['id'];
        $_SESSION['email'] = $user['email'];

        header('Location: index.php');
        exit();
    } else {
        echo "Invalid email or password.";
    }
}
?>

