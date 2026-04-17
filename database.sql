-- ============================================================
--  Smart Traffic & RTO Automation - MySQL Database Schema
-- ============================================================

CREATE DATABASE IF NOT EXISTS rto_automation;
USE rto_automation;

-- ──────────────────────────────────────────────────────────────
--  1. CAMERAS - CCTV camera locations
-- ──────────────────────────────────────────────────────────────
CREATE TABLE cameras (
    camera_id       INT AUTO_INCREMENT PRIMARY KEY,
    camera_name     VARCHAR(50)  NOT NULL,
    location        VARCHAR(150) NOT NULL,
    zone            VARCHAR(100) NOT NULL,
    latitude        DECIMAL(10,7),
    longitude       DECIMAL(10,7),
    status          ENUM('active','inactive','maintenance') DEFAULT 'active',
    installed_on    DATE,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ──────────────────────────────────────────────────────────────
--  2. ZONES - Traffic monitoring zones / areas
-- ──────────────────────────────────────────────────────────────
CREATE TABLE zones (
    zone_id         INT AUTO_INCREMENT PRIMARY KEY,
    zone_name       VARCHAR(100) NOT NULL UNIQUE,
    city            VARCHAR(100) NOT NULL,
    state           VARCHAR(100) DEFAULT 'Uttarakhand',
    zone_type       ENUM('highway','urban','residential','commercial') DEFAULT 'urban',
    speed_limit     INT DEFAULT 60,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ──────────────────────────────────────────────────────────────
--  3. VEHICLE_TYPES - Classification categories
-- ──────────────────────────────────────────────────────────────
CREATE TABLE vehicle_types (
    type_id         INT AUTO_INCREMENT PRIMARY KEY,
    type_name       VARCHAR(30) NOT NULL UNIQUE,
    description     VARCHAR(150)
);

INSERT INTO vehicle_types (type_name, description) VALUES
    ('bike',  'Two-wheelers including motorcycles and scooters'),
    ('car',   'Four-wheelers including sedans, SUVs, hatchbacks'),
    ('bus',   'Public and private buses'),
    ('truck', 'Trucks, lorries, and heavy commercial vehicles');

-- ──────────────────────────────────────────────────────────────
--  4. DETECTIONS - Every vehicle detected by AI
-- ──────────────────────────────────────────────────────────────
CREATE TABLE detections (
    detection_id    BIGINT AUTO_INCREMENT PRIMARY KEY,
    camera_id       INT NOT NULL,
    type_id         INT NOT NULL,
    confidence      DECIMAL(5,2) NOT NULL,
    detected_at     DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    frame_number    INT,
    bbox_x          INT,
    bbox_y          INT,
    bbox_width      INT,
    bbox_height     INT,

    FOREIGN KEY (camera_id) REFERENCES cameras(camera_id),
    FOREIGN KEY (type_id)   REFERENCES vehicle_types(type_id),

    INDEX idx_detected_at (detected_at),
    INDEX idx_camera_time (camera_id, detected_at),
    INDEX idx_type_time   (type_id, detected_at)
);

-- ──────────────────────────────────────────────────────────────
--  5. TRAFFIC_STATS - Aggregated hourly traffic summaries
-- ──────────────────────────────────────────────────────────────
CREATE TABLE traffic_stats (
    stat_id         BIGINT AUTO_INCREMENT PRIMARY KEY,
    camera_id       INT NOT NULL,
    zone_id         INT NOT NULL,
    stat_date       DATE NOT NULL,
    stat_hour       TINYINT NOT NULL,
    bike_count      INT DEFAULT 0,
    car_count       INT DEFAULT 0,
    bus_count       INT DEFAULT 0,
    truck_count     INT DEFAULT 0,
    total_count     INT DEFAULT 0,
    avg_confidence  DECIMAL(5,2),
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (camera_id) REFERENCES cameras(camera_id),
    FOREIGN KEY (zone_id)   REFERENCES zones(zone_id),

    UNIQUE KEY uq_camera_date_hour (camera_id, stat_date, stat_hour),
    INDEX idx_zone_date (zone_id, stat_date)
);

-- ──────────────────────────────────────────────────────────────
--  6. CONGESTION_LOG - Zone congestion snapshots
-- ──────────────────────────────────────────────────────────────
CREATE TABLE congestion_log (
    log_id          BIGINT AUTO_INCREMENT PRIMARY KEY,
    zone_id         INT NOT NULL,
    congestion_pct  DECIMAL(5,2) NOT NULL,
    severity        ENUM('low','medium','high') NOT NULL,
    vehicle_count   INT DEFAULT 0,
    recorded_at     DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (zone_id) REFERENCES zones(zone_id),

    INDEX idx_zone_time (zone_id, recorded_at)
);

-- ──────────────────────────────────────────────────────────────
--  7. DAILY_REPORTS - Daily traffic summary per zone
-- ──────────────────────────────────────────────────────────────
CREATE TABLE daily_reports (
    report_id       BIGINT AUTO_INCREMENT PRIMARY KEY,
    zone_id         INT NOT NULL,
    report_date     DATE NOT NULL,
    total_vehicles  INT DEFAULT 0,
    peak_hour       TINYINT,
    peak_count      INT DEFAULT 0,
    bike_total      INT DEFAULT 0,
    car_total       INT DEFAULT 0,
    bus_total       INT DEFAULT 0,
    truck_total     INT DEFAULT 0,
    avg_congestion  DECIMAL(5,2),
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (zone_id) REFERENCES zones(zone_id),

    UNIQUE KEY uq_zone_date (zone_id, report_date)
);

-- ──────────────────────────────────────────────────────────────
--  8. USERS - RTO officers / admin accounts
-- ──────────────────────────────────────────────────────────────
CREATE TABLE users (
    user_id         INT AUTO_INCREMENT PRIMARY KEY,
    full_name       VARCHAR(100) NOT NULL,
    email           VARCHAR(150) NOT NULL UNIQUE,
    password_hash   VARCHAR(255) NOT NULL,
    role            ENUM('admin','officer','viewer') DEFAULT 'viewer',
    phone           VARCHAR(15),
    is_active       BOOLEAN DEFAULT TRUE,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_login      DATETIME
);

-- ──────────────────────────────────────────────────────────────
--  9. ALERTS - System alerts & notifications
-- ──────────────────────────────────────────────────────────────
CREATE TABLE alerts (
    alert_id        BIGINT AUTO_INCREMENT PRIMARY KEY,
    zone_id         INT,
    camera_id       INT,
    alert_type      ENUM('high_congestion','camera_offline','unusual_traffic','system_error') NOT NULL,
    message         VARCHAR(255) NOT NULL,
    is_read         BOOLEAN DEFAULT FALSE,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (zone_id)   REFERENCES zones(zone_id),
    FOREIGN KEY (camera_id) REFERENCES cameras(camera_id),

    INDEX idx_alert_time (created_at),
    INDEX idx_unread (is_read, created_at)
);

-- ──────────────────────────────────────────────────────────────
--  10. SYSTEM_LOG - AI model & system performance tracking
-- ──────────────────────────────────────────────────────────────
CREATE TABLE system_log (
    log_id          BIGINT AUTO_INCREMENT PRIMARY KEY,
    model_name      VARCHAR(50) DEFAULT 'YOLOv8',
    fps             DECIMAL(5,1),
    accuracy        DECIMAL(5,2),
    frames_processed BIGINT DEFAULT 0,
    status          ENUM('operational','degraded','offline') DEFAULT 'operational',
    logged_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);


-- ============================================================
--  SAMPLE DATA
-- ============================================================

-- Zones
INSERT INTO zones (zone_name, city, zone_type, speed_limit) VALUES
    ('MG Road',      'Dehradun', 'commercial',  40),
    ('NH-48',        'Dehradun', 'highway',     80),
    ('Ring Road',    'Dehradun', 'urban',       60),
    ('City Center',  'Dehradun', 'commercial',  30),
    ('Highway-24',   'Dehradun', 'highway',     80),
    ('Airport Rd',   'Dehradun', 'urban',       60);

-- Cameras
INSERT INTO cameras (camera_name, location, zone, latitude, longitude) VALUES
    ('CAM-01', 'MG Road Junction',     'MG Road',     30.3165000, 78.0322000),
    ('CAM-02', 'NH-48 Toll Plaza',     'NH-48',       30.3400000, 78.0500000),
    ('CAM-03', 'Ring Road Flyover',    'Ring Road',   30.3250000, 78.0450000),
    ('CAM-04', 'Clock Tower Signal',   'City Center', 30.3190000, 78.0380000);

-- Admin user (password: admin123 - hashed with bcrypt placeholder)
INSERT INTO users (full_name, email, password_hash, role, phone) VALUES
    ('Admin RTO', 'admin@rto.gov.in', '$2b$10$placeholder_hash_replace_in_production', 'admin', '9876543210');


-- ============================================================
--  USEFUL VIEWS
-- ============================================================

-- View: Live vehicle counts per zone
CREATE VIEW v_live_zone_counts AS
SELECT
    z.zone_name,
    vt.type_name,
    COUNT(d.detection_id) AS vehicle_count,
    AVG(d.confidence)     AS avg_confidence
FROM detections d
JOIN cameras c      ON d.camera_id = c.camera_id
JOIN zones z        ON c.zone = z.zone_name
JOIN vehicle_types vt ON d.type_id = vt.type_id
WHERE d.detected_at >= NOW() - INTERVAL 1 HOUR
GROUP BY z.zone_name, vt.type_name;

-- View: Hourly summary for today
CREATE VIEW v_today_hourly AS
SELECT
    ts.stat_hour,
    SUM(ts.bike_count)  AS bikes,
    SUM(ts.car_count)   AS cars,
    SUM(ts.bus_count)    AS buses,
    SUM(ts.truck_count)  AS trucks,
    SUM(ts.total_count)  AS total
FROM traffic_stats ts
WHERE ts.stat_date = CURDATE()
GROUP BY ts.stat_hour
ORDER BY ts.stat_hour;

-- View: Current congestion per zone
CREATE VIEW v_zone_congestion AS
SELECT
    z.zone_name,
    cl.congestion_pct,
    cl.severity,
    cl.vehicle_count,
    cl.recorded_at
FROM congestion_log cl
JOIN zones z ON cl.zone_id = z.zone_id
WHERE cl.recorded_at = (
    SELECT MAX(recorded_at) FROM congestion_log cl2 WHERE cl2.zone_id = cl.zone_id
);
