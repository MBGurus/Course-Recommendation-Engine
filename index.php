<?php
session_start();

// Check if the user is logged in
if (!isset($_SESSION['user_id'])) {
    header('Location: login.php');
    exit();
}
?>

<!DOCTYPE html>
<html lang="en">

<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Course Recommendation Engine</title>
    <link rel="stylesheet" href="home.css">
    <style>
        .welcome-message {
            text-align: center;
            padding:3px;
            background-color: #f4f4f4;
            border-radius: 8px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            margin: 10px auto;
            margin-right: 5px;
            width: 80%;
            max-width: 600px;
        }
        .welcome-message h1 {
            color: #333;
            font-size: 24px;
            margin-bottom: 10px;
        }
        .welcome-message p {
            color: #666;
            font-size: 16px;
            margin-bottom: 5px;
        }
        .logout-btn {
            display: inline-block;
            padding: 10px 20px;
            font-size: 16px;
            color: #fff;
            background-color: red;
            border: none;
            border-radius: 5px;
            text-decoration: none;
            text-align: center;
        }
        .logout-btn:hover {
            background-color: #0056b3;
        }
    </style>
</head>

<body>

    <header>
        <div class="logo">
            <img src="logo.png" alt="Course Recommendation Engine Logo" />
        </div>
        <nav>
            <ul>
            <a href="logout.php" class="logout-btn">Logout</a>
                <li><a href="home.html">Home</a></li>
                <li><a href="aboutus.html">What we do</a></li>
                <li><a href="dashboard.html">Dashboard</a></li>
                <li><a href="features.html">Features</a></li>
                <li><a href="contact.html">Contact</a></li>
            </ul>
            <div class="welcome-message">
                <h1>Welcome, <?php echo htmlspecialchars($_SESSION['email']); ?>!</h1>
             
             
            </div>
        </nav>
    </header>

    <section class="hero">
        <div class="overlay">
            <div class="hero-text">
                <h1>Course Recommendation Engine</h1>
                <p>Helping students make the best subject and course choices</p>
                <!-- Check if the user is logged in and display the appropriate button -->
                <?php if (isset($_SESSION['user_id'])): ?>
                    <a href="chatbot/templates/user.html" class="btn">Get Started</a>
                <?php else: ?>
                    <a href="login.php" class="btn">Login/Sign In</a>
                <?php endif; ?>
            </div>
        </div>
    </section>

</body>

</html>

