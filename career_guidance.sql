-- phpMyAdmin SQL Dump
-- version 5.2.1
-- https://www.phpmyadmin.net/
--
-- Host: 127.0.0.1
-- Generation Time: Sep 15, 2024 at 04:59 PM
-- Server version: 10.4.28-MariaDB
-- PHP Version: 8.2.4

SET SQL_MODE = "NO_AUTO_VALUE_ON_ZERO";
START TRANSACTION;
SET time_zone = "+00:00";


/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!40101 SET NAMES utf8mb4 */;

--
-- Database: `career_guidance`
--

-- --------------------------------------------------------

--
-- Table structure for table `academic_goals_aspirations`
--

CREATE TABLE `academic_goals_aspirations` (
  `id` int(11) NOT NULL,
  `user_id` int(11) DEFAULT NULL,
  `subjects_for_grade_10` text DEFAULT NULL,
  `goal_career_path` tinyint(1) DEFAULT NULL,
  `goal_exploration` tinyint(1) DEFAULT NULL,
  `goal_interest` tinyint(1) DEFAULT NULL,
  `career_interest` text DEFAULT NULL,
  `university_program` text DEFAULT NULL,
  `subjects_excellence` text DEFAULT NULL,
  `goal_education` tinyint(1) DEFAULT NULL,
  `goal_skills` tinyint(1) DEFAULT NULL,
  `goal_business` tinyint(1) DEFAULT NULL,
  `goal_explore` tinyint(1) DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Dumping data for table `academic_goals_aspirations`
--

INSERT INTO `academic_goals_aspirations` (`id`, `user_id`, `subjects_for_grade_10`, `goal_career_path`, `goal_exploration`, `goal_interest`, `career_interest`, `university_program`, `subjects_excellence`, `goal_education`, `goal_skills`, `goal_business`, `goal_explore`) VALUES
(1, 1, 'math,physics', 1, 0, 0, NULL, NULL, NULL, 1, 0, 0, 0),
(2, 2, 'math,physics', 1, 0, 0, '', '', '', 1, 0, 0, 0),
(3, 3, 'math,physics', 1, 0, 0, '', '', '', 1, 0, 0, 0);

-- --------------------------------------------------------

--
-- Table structure for table `future_plans_vision`
--

CREATE TABLE `future_plans_vision` (
  `id` int(11) NOT NULL,
  `user_id` int(11) DEFAULT NULL,
  `career_goals` text DEFAULT NULL,
  `future_impact` text DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Dumping data for table `future_plans_vision`
--

INSERT INTO `future_plans_vision` (`id`, `user_id`, `career_goals`, `future_impact`) VALUES
(1, 1, 'be a worker', 'change lives'),
(2, 2, 'be a worker', 'change lives'),
(3, 3, 'be a worker', 'change lives');

-- --------------------------------------------------------

--
-- Table structure for table `interests_hobbies`
--

CREATE TABLE `interests_hobbies` (
  `id` int(11) NOT NULL,
  `user_id` int(11) DEFAULT NULL,
  `subjects_activities` text DEFAULT NULL,
  `hobbies_interests` text DEFAULT NULL,
  `interest_area` varchar(255) DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Dumping data for table `interests_hobbies`
--

INSERT INTO `interests_hobbies` (`id`, `user_id`, `subjects_activities`, `hobbies_interests`, `interest_area`) VALUES
(1, 1, 'math', 'soccer', 'Mathematics'),
(2, 2, 'math', 'soccer', 'Mathematics'),
(3, 3, 'math', 'soccer', 'Mathematics');

-- --------------------------------------------------------

--
-- Table structure for table `personality_learning_style`
--

CREATE TABLE `personality_learning_style` (
  `id` int(11) NOT NULL,
  `user_id` int(11) DEFAULT NULL,
  `learning_style` set('visual','auditory','kinesthetic','reading_writing') DEFAULT NULL,
  `work_preference` enum('independent','group') DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Dumping data for table `personality_learning_style`
--

INSERT INTO `personality_learning_style` (`id`, `user_id`, `learning_style`, `work_preference`) VALUES
(1, 1, 'visual', 'independent'),
(2, 2, 'visual', 'independent'),
(3, 3, 'visual', 'independent');

-- --------------------------------------------------------

--
-- Table structure for table `subject_preferences_strengths`
--

CREATE TABLE `subject_preferences_strengths` (
  `id` int(11) NOT NULL,
  `user_id` int(11) DEFAULT NULL,
  `current_subjects` text DEFAULT NULL,
  `challenging_subjects` text DEFAULT NULL,
  `confident_subjects` text DEFAULT NULL,
  `ideal_course` text DEFAULT NULL,
  `subject_preference` enum('theoretical','practical') DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Dumping data for table `subject_preferences_strengths`
--

INSERT INTO `subject_preferences_strengths` (`id`, `user_id`, `current_subjects`, `challenging_subjects`, `confident_subjects`, `ideal_course`, `subject_preference`) VALUES
(1, 1, 'math', 'sience', 'math', 'software development', 'practical'),
(2, 2, 'math', 'sience', 'math', 'software development', 'practical'),
(3, 3, 'math', 'sience', 'math', 'software development', 'practical');

-- --------------------------------------------------------

--
-- Table structure for table `university_career_planning`
--

CREATE TABLE `university_career_planning` (
  `id` int(11) NOT NULL,
  `user_id` int(11) DEFAULT NULL,
  `researched_universities` enum('yes','no') DEFAULT NULL,
  `top_universities` text DEFAULT NULL,
  `criteria_reputation` tinyint(1) DEFAULT NULL,
  `criteria_course_content` tinyint(1) DEFAULT NULL,
  `criteria_location` tinyint(1) DEFAULT NULL,
  `criteria_financial_aid` tinyint(1) DEFAULT NULL,
  `criteria_campus_life` tinyint(1) DEFAULT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Dumping data for table `university_career_planning`
--

INSERT INTO `university_career_planning` (`id`, `user_id`, `researched_universities`, `top_universities`, `criteria_reputation`, `criteria_course_content`, `criteria_location`, `criteria_financial_aid`, `criteria_campus_life`) VALUES
(1, 1, 'no', NULL, 1, 0, 0, 0, 0),
(2, 2, 'no', '', 1, 0, 0, 0, 0),
(3, 3, 'no', '', 1, 0, 0, 0, 0);

-- --------------------------------------------------------

--
-- Table structure for table `users`
--

CREATE TABLE `users` (
  `id` int(11) NOT NULL,
  `email` varchar(255) NOT NULL,
  `password` varchar(255) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Dumping data for table `users`
--

INSERT INTO `users` (`id`, `email`, `password`) VALUES
(1, 'caswellnziyane@gmail.com', '$2y$10$6OxrRKGNHeJXE7MCHXQXJeu2L1qjDWHFJ3SOWJwWkKi5VfRtV.pSi');

-- --------------------------------------------------------

--
-- Table structure for table `user_grades`
--

CREATE TABLE `user_grades` (
  `id` int(11) NOT NULL,
  `grade_level` enum('9','12') NOT NULL,
  `submitted_at` timestamp NOT NULL DEFAULT current_timestamp()
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Dumping data for table `user_grades`
--

INSERT INTO `user_grades` (`id`, `grade_level`, `submitted_at`) VALUES
(1, '9', '2024-09-15 14:51:48'),
(2, '9', '2024-09-15 14:57:46'),
(3, '9', '2024-09-15 14:58:51');

--
-- Indexes for dumped tables
--

--
-- Indexes for table `academic_goals_aspirations`
--
ALTER TABLE `academic_goals_aspirations`
  ADD PRIMARY KEY (`id`),
  ADD KEY `user_id` (`user_id`);

--
-- Indexes for table `future_plans_vision`
--
ALTER TABLE `future_plans_vision`
  ADD PRIMARY KEY (`id`),
  ADD KEY `user_id` (`user_id`);

--
-- Indexes for table `interests_hobbies`
--
ALTER TABLE `interests_hobbies`
  ADD PRIMARY KEY (`id`),
  ADD KEY `user_id` (`user_id`);

--
-- Indexes for table `personality_learning_style`
--
ALTER TABLE `personality_learning_style`
  ADD PRIMARY KEY (`id`),
  ADD KEY `user_id` (`user_id`);

--
-- Indexes for table `subject_preferences_strengths`
--
ALTER TABLE `subject_preferences_strengths`
  ADD PRIMARY KEY (`id`),
  ADD KEY `user_id` (`user_id`);

--
-- Indexes for table `university_career_planning`
--
ALTER TABLE `university_career_planning`
  ADD PRIMARY KEY (`id`),
  ADD KEY `user_id` (`user_id`);

--
-- Indexes for table `users`
--
ALTER TABLE `users`
  ADD PRIMARY KEY (`id`),
  ADD UNIQUE KEY `email` (`email`);

--
-- Indexes for table `user_grades`
--
ALTER TABLE `user_grades`
  ADD PRIMARY KEY (`id`);

--
-- AUTO_INCREMENT for dumped tables
--

--
-- AUTO_INCREMENT for table `academic_goals_aspirations`
--
ALTER TABLE `academic_goals_aspirations`
  MODIFY `id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=4;

--
-- AUTO_INCREMENT for table `future_plans_vision`
--
ALTER TABLE `future_plans_vision`
  MODIFY `id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=4;

--
-- AUTO_INCREMENT for table `interests_hobbies`
--
ALTER TABLE `interests_hobbies`
  MODIFY `id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=4;

--
-- AUTO_INCREMENT for table `personality_learning_style`
--
ALTER TABLE `personality_learning_style`
  MODIFY `id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=4;

--
-- AUTO_INCREMENT for table `subject_preferences_strengths`
--
ALTER TABLE `subject_preferences_strengths`
  MODIFY `id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=4;

--
-- AUTO_INCREMENT for table `university_career_planning`
--
ALTER TABLE `university_career_planning`
  MODIFY `id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=4;

--
-- AUTO_INCREMENT for table `users`
--
ALTER TABLE `users`
  MODIFY `id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=3;

--
-- AUTO_INCREMENT for table `user_grades`
--
ALTER TABLE `user_grades`
  MODIFY `id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=4;

--
-- Constraints for dumped tables
--

--
-- Constraints for table `academic_goals_aspirations`
--
ALTER TABLE `academic_goals_aspirations`
  ADD CONSTRAINT `academic_goals_aspirations_ibfk_1` FOREIGN KEY (`user_id`) REFERENCES `user_grades` (`id`) ON DELETE CASCADE;

--
-- Constraints for table `future_plans_vision`
--
ALTER TABLE `future_plans_vision`
  ADD CONSTRAINT `future_plans_vision_ibfk_1` FOREIGN KEY (`user_id`) REFERENCES `user_grades` (`id`) ON DELETE CASCADE;

--
-- Constraints for table `interests_hobbies`
--
ALTER TABLE `interests_hobbies`
  ADD CONSTRAINT `interests_hobbies_ibfk_1` FOREIGN KEY (`user_id`) REFERENCES `user_grades` (`id`) ON DELETE CASCADE;

--
-- Constraints for table `personality_learning_style`
--
ALTER TABLE `personality_learning_style`
  ADD CONSTRAINT `personality_learning_style_ibfk_1` FOREIGN KEY (`user_id`) REFERENCES `user_grades` (`id`) ON DELETE CASCADE;

--
-- Constraints for table `subject_preferences_strengths`
--
ALTER TABLE `subject_preferences_strengths`
  ADD CONSTRAINT `subject_preferences_strengths_ibfk_1` FOREIGN KEY (`user_id`) REFERENCES `user_grades` (`id`) ON DELETE CASCADE;

--
-- Constraints for table `university_career_planning`
--
ALTER TABLE `university_career_planning`
  ADD CONSTRAINT `university_career_planning_ibfk_1` FOREIGN KEY (`user_id`) REFERENCES `user_grades` (`id`) ON DELETE CASCADE;
COMMIT;

/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
