-- Run this in phpMyAdmin (XAMPP) or the MySQL CLI before starting the app.

CREATE DATABASE IF NOT EXISTS sentiment_db;
USE sentiment_db;

CREATE TABLE IF NOT EXISTS predictions (
    id INT AUTO_INCREMENT PRIMARY KEY,
    review_text TEXT NOT NULL,
    sentiment VARCHAR(20) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
