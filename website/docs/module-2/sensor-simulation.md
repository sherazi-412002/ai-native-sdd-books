---
sidebar_position: 3
---

# Sensor Simulation: Advanced Perceptual Systems for Humanoid Robot Digital Twins

## Introduction to Sensor Simulation for Digital Twins

Sensor simulation in digital twin environments for humanoid robots is critical for creating realistic perception systems that enable effective testing and development of algorithms before deployment on physical hardware. The simulation must accurately replicate the characteristics, noise patterns, and limitations of real-world sensors to ensure transferability of learned behaviors and algorithms.

Modern digital twin systems require sophisticated sensor simulation that includes:
- **Realistic Noise Modeling**: Accurate representation of sensor noise, biases, and drift
- **Environmental Interactions**: Proper simulation of how sensors interact with different materials and lighting conditions
- **Temporal Dynamics**: Accurate timing and synchronization between different sensors
- **Physics-Based Rendering**: Proper simulation of sensor-specific physics phenomena

## Theoretical Foundation of Sensor Simulation

### Sensor Modeling Principles

The mathematical foundation for sensor simulation is based on the concept of forward rendering, where real-world phenomena are projected onto sensor-specific representations:

#### LiDAR Sensor Model

LiDAR sensors operate on the principle of time-of-flight measurement, where laser pulses are emitted and the time taken for the reflection to return is measured. The sensor model can be expressed as:

```
d = (c * Δt) / 2
```

Where `d` is the distance, `c` is the speed of light, and `Δt` is the time difference between emission and reception.

The simulated LiDAR measurement includes several noise components:

```cpp
// Advanced LiDAR sensor simulation model
#include <vector>
#include <random>
#include <cmath>
#include <memory>

class LiDARSensorModel {
public:
    struct RayData {
        float distance;
        float intensity;
        uint8_t return_type;  // Single, first, last, etc.
        int8_t noise_level;
        float angle_horizontal;
        float angle_vertical;
    };

    struct SensorConfig {
        float range_min = 0.1f;        // Minimum detectable range (m)
        float range_max = 100.0f;      // Maximum detectable range (m)
        float fov_horizontal = 360.0f; // Horizontal field of view (degrees)
        float fov_vertical = 30.0f;    // Vertical field of view (degrees)
        int beams_horizontal = 1080;   // Number of horizontal beams
        int beams_vertical = 64;       // Number of vertical beams
        float angular_resolution_horizontal;
        float angular_resolution_vertical;
        float noise_std_dev = 0.02f;   // Standard deviation of noise (m)
        float intensity_base = 100.0f; // Base intensity value
        float drop_rate = 0.01f;       // Probability of dropouts
    };

private:
    SensorConfig config_;
    std::mt19937 rng_;
    std::normal_distribution<float> noise_dist_;
    std::uniform_real_distribution<float> dropout_dist_;
    std::vector<std::vector<RayData>> scan_data_;

public:
    LiDARSensorModel(const SensorConfig& config)
        : config_(config),
          rng_(std::random_device{}()),
          noise_dist_(0.0f, config.noise_std_dev),
          dropout_dist_(0.0f, 1.0f) {

        config_.angular_resolution_horizontal =
            config_.fov_horizontal / config_.beams_horizontal;
        config_.angular_resolution_vertical =
            config_.fov_vertical / config_.beams_vertical;

        scan_data_.resize(config_.beams_vertical);
        for (auto& row : scan_data_) {
            row.resize(config_.beams_horizontal);
        }
    }

    void simulateScan(const std::vector<std::vector<float>>& environment_depth_map,
                     const std::vector<std::vector<float>>& reflectivity_map,
                     std::vector<std::vector<RayData>>& output_scan) {

        output_scan.resize(config_.beams_vertical);
        for (auto& row : output_scan) {
            row.resize(config_.beams_horizontal);
        }

        for (int v_idx = 0; v_idx < config_.beams_vertical; ++v_idx) {
            for (int h_idx = 0; h_idx < config_.beams_horizontal; ++h_idx) {
                float base_distance = getEnvironmentDistance(environment_depth_map, h_idx, v_idx);

                if (base_distance > 0 && base_distance < config_.range_max) {
                    // Apply noise model
                    float noise = noise_dist_(rng_);
                    float noisy_distance = base_distance + noise;

                    // Apply dropout probability
                    if (dropout_dist_(rng_) < config_.drop_rate) {
                        noisy_distance = 0.0f; // Invalid measurement
                    }

                    // Apply intensity based on reflectivity and distance
                    float reflectivity = getReflectivity(reflectivity_map, h_idx, v_idx);
                    float intensity = calculateIntensity(reflectivity, noisy_distance);

                    output_scan[v_idx][h_idx] = {
                        noisy_distance,
                        intensity,
                        0, // return type
                        static_cast<int8_t>(noise * 100), // noise level indicator
                        h_idx * config_.angular_resolution_horizontal,
                        v_idx * config_.angular_resolution_vertical
                    };
                } else {
                    output_scan[v_idx][h_idx] = {0.0f, 0.0f, 0, 0, 0.0f, 0.0f};
                }
            }
        }
    }

private:
    float getEnvironmentDistance(const std::vector<std::vector<float>>& depth_map,
                               int h_idx, int v_idx) {
        // Interpolate from environment depth map based on beam angles
        // This is a simplified representation - real implementation would ray-cast
        // against 3D environment models
        if (h_idx < depth_map.size() && v_idx < depth_map[0].size()) {
            return depth_map[h_idx][v_idx];
        }
        return config_.range_max + 1.0f; // Out of range
    }

    float getReflectivity(const std::vector<std::vector<float>>& reflectivity_map,
                         int h_idx, int v_idx) {
        if (h_idx < reflectivity_map.size() && v_idx < reflectivity_map[0].size()) {
            return reflectivity_map[h_idx][v_idx];
        }
        return 0.5f; // Default reflectivity
    }

    float calculateIntensity(float reflectivity, float distance) {
        // Intensity model: decreases with distance squared
        if (distance > 0) {
            float base_intensity = config_.intensity_base * reflectivity / (distance * distance);
            // Apply some additional noise to intensity
            float intensity_noise = (dropout_dist_(rng_) - 0.5f) * 10.0f;
            return std::max(0.0f, base_intensity + intensity_noise);
        }
        return 0.0f;
    }
};
```

#### IMU Sensor Model

Inertial Measurement Units (IMUs) measure linear acceleration and angular velocity. The sensor model includes several error sources:

1. **Bias**: Systematic offset that changes over time
2. **Scale Factor Error**: Deviation from ideal scaling
3. **Cross-Axis Sensitivity**: Crosstalk between different axes
4. **Noise**: Random fluctuations following various statistical models

```cpp
// Advanced IMU sensor simulation
#include <vector>
#include <random>
#include <chrono>

class IMUSensorModel {
public:
    struct IMUReading {
        std::array<float, 3> linear_acceleration;    // m/s²
        std::array<float, 3> angular_velocity;       // rad/s
        std::array<float, 3> magnetic_field;         // µT
        std::chrono::time_point<std::chrono::system_clock> timestamp;
    };

    struct SensorSpecs {
        // Noise parameters (Allan variance coefficients)
        float accel_noise_density = 0.002f;      // m/s/sqrt(Hz)
        float accel_bias_random_walk = 0.0004f;  // m/s²/sqrt(Hz)
        float gyro_noise_density = 0.00024f;     // rad/s/sqrt(Hz)
        float gyro_bias_random_walk = 2.6e-6f;   // rad/s²/sqrt(Hz)

        // Bias parameters
        std::array<float, 3> initial_accel_bias = {0.0f, 0.0f, 0.0f};
        std::array<float, 3> initial_gyro_bias = {0.0f, 0.0f, 0.0f};

        // Scale factor errors
        std::array<float, 3> accel_scale_error = {0.0f, 0.0f, 0.0f};
        std::array<float, 3> gyro_scale_error = {0.0f, 0.0f, 0.0f};

        // Cross-axis sensitivity
        std::array<std::array<float, 3>, 3> accel_cross_axis =
            {{{1.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f}, {0.0f, 0.0f, 1.0f}}};
        std::array<std::array<float, 3>, 3> gyro_cross_axis =
            {{{1.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f}, {0.0f, 0.0f, 1.0f}}};
    };

private:
    SensorSpecs specs_;
    std::mt19937 rng_;
    std::normal_distribution<float> accel_noise_dist_;
    std::normal_distribution<float> gyro_noise_dist_;

    // Current bias states (random walk)
    std::array<float, 3> accel_bias_;
    std::array<float, 3> gyro_bias_;

    float dt_;  // Time step for bias random walk

public:
    IMUSensorModel(const SensorSpecs& specs, float update_rate = 100.0f)
        : specs_(specs),
          rng_(std::random_device{}()),
          accel_noise_dist_(0.0f, specs_.accel_noise_density),
          gyro_noise_dist_(0.0f, specs_.gyro_noise_density),
          dt_(1.0f / update_rate) {

        accel_bias_ = specs_.initial_accel_bias;
        gyro_bias_ = specs_.initial_gyro_bias;
    }

    IMUReading simulateReading(const std::array<float, 3>& true_accel,
                              const std::array<float, 3>& true_gyro,
                              const std::array<float, 3>& true_mag) {

        IMUReading reading;
        reading.timestamp = std::chrono::system_clock::now();

        // Apply noise and bias to accelerometer
        for (int i = 0; i < 3; ++i) {
            // Add noise
            float noise = accel_noise_dist_(rng_) / std::sqrt(dt_);

            // Update bias using random walk model
            float bias_walk = std::normal_distribution<float>(
                0.0f, specs_.accel_bias_random_walk * std::sqrt(dt_)
            )(rng_);
            accel_bias_[i] += bias_walk;

            // Apply all error sources
            float noisy_accel = true_accel[i] + accel_bias_[i] + noise;

            // Apply scale factor error
            noisy_accel *= (1.0f + specs_.accel_scale_error[i]);

            // Apply cross-axis sensitivity
            float corrected_accel = 0.0f;
            for (int j = 0; j < 3; ++j) {
                corrected_accel += specs_.accel_cross_axis[i][j] * noisy_accel;
            }

            reading.linear_acceleration[i] = corrected_accel;
        }

        // Apply noise and bias to gyroscope
        for (int i = 0; i < 3; ++i) {
            // Add noise
            float noise = gyro_noise_dist_(rng_) / std::sqrt(dt_);

            // Update bias using random walk model
            float bias_walk = std::normal_distribution<float>(
                0.0f, specs_.gyro_bias_random_walk * std::sqrt(dt_)
            )(rng_);
            gyro_bias_[i] += bias_walk;

            // Apply all error sources
            float noisy_gyro = true_gyro[i] + gyro_bias_[i] + noise;

            // Apply scale factor error
            noisy_gyro *= (1.0f + specs_.gyro_scale_error[i]);

            // Apply cross-axis sensitivity
            float corrected_gyro = 0.0f;
            for (int j = 0; j < 3; ++j) {
                corrected_gyro += specs_.gyro_cross_axis[i][j] * noisy_gyro;
            }

            reading.angular_velocity[i] = corrected_gyro;
        }

        // Apply simple noise model to magnetometer
        for (int i = 0; i < 3; ++i) {
            float noise = std::normal_distribution<float>(0.0f, 0.1f)(rng_);
            reading.magnetic_field[i] = true_mag[i] + noise;
        }

        return reading;
    }

    // Reset bias states (for calibration purposes)
    void resetBias() {
        accel_bias_ = specs_.initial_accel_bias;
        gyro_bias_ = specs_.initial_gyro_bias;
    }

    // Get current bias states
    const std::array<float, 3>& getAccelBias() const { return accel_bias_; }
    const std::array<float, 3>& getGyroBias() const { return gyro_bias_; }
};
```

### Camera Sensor Model

Camera sensors require sophisticated modeling of optical phenomena including distortion, noise, and dynamic range limitations:

```cpp
// Advanced camera sensor simulation
#include <opencv2/opencv.hpp>
#include <vector>
#include <random>

class CameraSensorModel {
public:
    struct CameraConfig {
        int width = 640;
        int height = 480;
        float fx = 320.0f;           // Focal length x
        float fy = 320.0f;           // Focal length y
        float cx = 320.0f;           // Principal point x
        float cy = 240.0f;           // Principal point y
        std::array<float, 4> distortion_coeffs = {0.0f, 0.0f, 0.0f, 0.0f}; // k1, k2, p1, p2
        float fov_horizontal = 60.0f; // Field of view (degrees)
        float fov_vertical = 45.0f;

        // Noise parameters
        float gaussian_noise_std = 10.0f;    // Standard deviation of Gaussian noise
        float poisson_noise_factor = 0.01f;  // Factor for Poisson noise
        float dark_current = 0.1f;           // Dark current (electrons/second)
        float read_noise = 2.0f;             // Read noise (electrons)
        float quantization_noise = 0.5f;     // Quantization noise
    };

    struct ExposureParams {
        float exposure_time = 0.033f; // 33ms for 30fps
        float gain = 1.0f;
        float gamma = 1.0f;
        float iso = 100.0f;
    };

private:
    CameraConfig config_;
    ExposureParams exposure_params_;
    std::mt19937 rng_;
    std::normal_distribution<float> gaussian_noise_dist_;
    std::poisson_distribution<int> poisson_noise_dist_;

public:
    CameraSensorModel(const CameraConfig& config, const ExposureParams& exposure = {})
        : config_(config), exposure_params_(exposure),
          rng_(std::random_device{}()),
          gaussian_noise_dist_(0.0f, config_.gaussian_noise_std) {
    }

    cv::Mat simulateImage(const cv::Mat& scene_radiance) {
        cv::Mat image = scene_radiance.clone();

        // Apply optical effects and noise
        applyLensEffects(image);
        applySensorNoise(image);
        applyDigitalProcessing(image);

        // Ensure proper data range
        cv::normalize(image, image, 0, 255, cv::NORM_MINMAX);
        image.convertTo(image, CV_8UC3);

        return image;
    }

private:
    void applyLensEffects(cv::Mat& image) {
        // Apply radial and tangential distortion
        cv::Mat camera_matrix = (cv::Mat_<float>(3, 3) <<
            config_.fx, 0, config_.cx,
            0, config_.fy, config_.cy,
            0, 0, 1);

        cv::Mat dist_coeffs = (cv::Mat_<float>(1, 4) <<
            config_.distortion_coeffs[0], config_.distortion_coeffs[1],
            config_.distortion_coeffs[2], config_.distortion_coeffs[3]);

        cv::Mat undistorted;
        cv::undistort(image, undistorted, camera_matrix, dist_coeffs);
        image = undistorted;
    }

    void applySensorNoise(cv::Mat& image) {
        cv::Mat noise = cv::Mat::zeros(image.size(), image.type());

        // Add Gaussian noise
        for (int i = 0; i < image.rows; ++i) {
            for (int j = 0; j < image.cols; ++j) {
                for (int c = 0; c < image.channels(); ++c) {
                    float pixel_noise = gaussian_noise_dist_(rng_);
                    noise.at<cv::Vec3b>(i, j)[c] = static_cast<uint8_t>(
                        std::max(0.0f, std::min(255.0f, pixel_noise))
                    );
                }
            }
        }

        // Add Poisson noise (photon noise)
        cv::Mat poisson_image = image.clone();
        for (int i = 0; i < image.rows; ++i) {
            for (int j = 0; j < image.cols; ++j) {
                for (int c = 0; c < image.channels(); ++c) {
                    float intensity = image.at<cv::Vec3b>(i, j)[c] / 255.0f;
                    float photon_count = intensity * 1000.0f; // Scale to reasonable photon count
                    int noisy_count = std::poisson_distribution<int>(photon_count)(rng_);
                    poisson_image.at<cv::Vec3b>(i, j)[c] =
                        static_cast<uint8_t>((noisy_count / 1000.0f) * 255.0f);
                }
            }
        }

        // Combine with original image
        cv::addWeighted(image, 0.5, poisson_image, 0.5, 0.0, image);
    }

    void applyDigitalProcessing(cv::Mat& image) {
        // Apply gamma correction
        cv::Mat gamma_corrected;
        applyGammaCorrection(image, gamma_corrected, exposure_params_.gamma);

        // Apply gain
        gamma_corrected.convertTo(image, -1, exposure_params_.gain, 0);

        // Add quantization noise
        for (int i = 0; i < image.rows; ++i) {
            for (int j = 0; j < image.cols; ++j) {
                for (int c = 0; c < image.channels(); ++c) {
                    int value = image.at<cv::Vec3b>(i, j)[c];
                    value += static_cast<int>(gaussian_noise_dist_(rng_));
                    image.at<cv::Vec3b>(i, j)[c] =
                        static_cast<uint8_t>(std::max(0, std::min(255, value)));
                }
            }
        }
    }

    void applyGammaCorrection(const cv::Mat& src, cv::Mat& dst, float gamma) {
        cv::Mat lookUpTable(1, 256, CV_8U);
        uchar* ptr = lookUpTable.ptr();

        for (int i = 0; i < 256; ++i) {
            ptr[i] = cv::saturate_cast<uchar>(std::pow(i / 255.0, 1.0 / gamma) * 255.0);
        }

        cv::LUT(src, lookUpTable, dst);
    }
};
```

## Advanced LiDAR Simulation Techniques

### Multi-Echo LiDAR Simulation

Modern LiDAR sensors can detect multiple returns from a single pulse, which is essential for simulating complex environments with transparent or semi-transparent objects:

```cpp
// Multi-echo LiDAR simulation
#include <vector>
#include <map>
#include <algorithm>

struct MultiEchoMeasurement {
    std::vector<float> distances;    // Multiple distance measurements
    std::vector<float> intensities;  // Corresponding intensities
    std::vector<uint8_t> types;      // Return types (first, last, intermediate)
};

class MultiEchoLiDAR {
private:
    LiDARSensorModel::SensorConfig config_;
    float multi_echo_threshold_ = 0.1f; // Minimum distance between echoes

public:
    MultiEchoLiDAR(const LiDARSensorModel::SensorConfig& config)
        : config_(config) {}

    MultiEchoMeasurement simulateMultiEcho(const std::vector<float>& depth_profile) {
        MultiEchoMeasurement result;

        // Sort depth values and identify potential multiple returns
        std::vector<std::pair<float, float>> sorted_depths; // (distance, reflectivity)
        for (size_t i = 0; i < depth_profile.size(); ++i) {
            if (depth_profile[i] > config_.range_min &&
                depth_profile[i] < config_.range_max) {
                sorted_depths.push_back({depth_profile[i], 0.5f}); // Assume default reflectivity
            }
        }

        std::sort(sorted_depths.begin(), sorted_depths.end());

        // Group nearby returns as potential multi-echo returns
        if (!sorted_depths.empty()) {
            float current_dist = sorted_depths[0].first;
            result.distances.push_back(current_dist);
            result.intensities.push_back(calculateIntensity(sorted_depths[0].second, current_dist));
            result.types.push_back(0); // First return

            for (size_t i = 1; i < sorted_depths.size(); ++i) {
                float dist = sorted_depths[i].first;
                if (dist - current_dist > multi_echo_threshold_) {
                    // Significant gap - likely a separate return
                    result.distances.push_back(dist);
                    result.intensities.push_back(calculateIntensity(sorted_depths[i].second, dist));
                    result.types.push_back(1); // Subsequent return
                    current_dist = dist;
                }
            }

            if (!result.distances.empty()) {
                result.types.back() = 2; // Last return
            }
        }

        return result;
    }

private:
    float calculateIntensity(float reflectivity, float distance) {
        if (distance > 0) {
            return reflectivity / (distance * distance);
        }
        return 0.0f;
    }
};
```

## Integration with ROS 2 and Isaac Sim

### ROS 2 Sensor Message Simulation

Proper integration with ROS 2 requires simulating sensor messages with the correct format and timing:

```cpp
// ROS 2 sensor message simulation
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/laser_scan.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <cv_bridge/cv_bridge.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>

class ROSSensorSimulator : public rclcpp::Node {
private:
    rclcpp::Publisher<sensor_msgs::msg::LaserScan>::SharedPtr lidar_pub_;
    rclcpp::Publisher<sensor_msgs::msg::Imu>::SharedPtr imu_pub_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr camera_pub_;
    rclcpp::Publisher<sensor_msgs::msg::CameraInfo>::SharedPtr camera_info_pub_;

    LiDARSensorModel lidar_model_;
    IMUSensorModel imu_model_;
    CameraSensorModel camera_model_;

    rclcpp::TimerBase::SharedPtr timer_;
    size_t scan_count_;

public:
    ROSSensorSimulator() : Node("sensor_simulator"), scan_count_(0) {
        // Initialize publishers
        lidar_pub_ = this->create_publisher<sensor_msgs::msg::LaserScan>("scan", 10);
        imu_pub_ = this->create_publisher<sensor_msgs::msg::Imu>("imu/data", 10);
        camera_pub_ = this->create_publisher<sensor_msgs::msg::Image>("camera/image_raw", 10);
        camera_info_pub_ = this->create_publisher<sensor_msgs::msg::CameraInfo>("camera/camera_info", 10);

        // Initialize sensor models with realistic parameters
        LiDARSensorModel::SensorConfig lidar_config;
        lidar_config.range_min = 0.1f;
        lidar_config.range_max = 100.0f;
        lidar_config.beams_horizontal = 1080;
        lidar_config.fov_horizontal = 360.0f;
        lidar_config.noise_std_dev = 0.02f;
        lidar_model_ = LiDARSensorModel(lidar_config);

        IMUSensorModel::SensorSpecs imu_specs;
        imu_specs.accel_noise_density = 0.002f;
        imu_specs.gyro_noise_density = 0.00024f;
        imu_model_ = IMUSensorModel(imu_specs, 100.0f); // 100Hz update rate

        CameraSensorModel::CameraConfig camera_config;
        camera_config.width = 640;
        camera_config.height = 480;
        camera_config.fx = 320.0f;
        camera_config.fy = 320.0f;
        camera_config.cx = 320.0f;
        camera_config.cy = 240.0f;
        camera_model_ = CameraSensorModel(camera_config);

        // Create timer for sensor simulation
        timer_ = this->create_wall_timer(
            std::chrono::milliseconds(10), // 100Hz for LiDAR/IMU
            std::bind(&ROSSensorSimulator::timer_callback, this)
        );
    }

private:
    void timer_callback() {
        // Simulate and publish LiDAR data
        publish_lidar_data();

        // Simulate and publish IMU data
        publish_imu_data();

        // Publish camera data at lower frequency (e.g., 30Hz)
        if (scan_count_ % 3 == 0) { // Every 3rd call = ~33Hz
            publish_camera_data();
        }

        scan_count_++;
    }

    void publish_lidar_data() {
        auto msg = sensor_msgs::msg::LaserScan();
        msg.header.stamp = this->get_clock()->now();
        msg.header.frame_id = "laser_frame";

        msg.angle_min = -M_PI;
        msg.angle_max = M_PI;
        msg.angle_increment = 2.0 * M_PI / 1080.0;
        msg.time_increment = 0.0; // Not used for simulation
        msg.scan_time = 0.1; // 10Hz
        msg.range_min = 0.1;
        msg.range_max = 100.0;

        // Simulate some range data (in a real system, this would come from scene geometry)
        msg.ranges.resize(1080);
        for (size_t i = 0; i < msg.ranges.size(); ++i) {
            // Create a simple pattern for simulation
            float angle = msg.angle_min + i * msg.angle_increment;
            float distance = 10.0f + 5.0f * std::sin(4 * angle); // 5m to 15m range

            // Add some noise
            std::random_device rd;
            std::mt19937 gen(rd());
            std::normal_distribution<float> noise(0.0, 0.02);
            distance += noise(gen);

            msg.ranges[i] = std::max(0.1f, std::min(100.0f, distance));
        }

        msg.intensities.resize(1080);
        for (size_t i = 0; i < msg.intensities.size(); ++i) {
            msg.intensities[i] = 100.0; // Constant intensity for simplicity
        }

        lidar_pub_->publish(msg);
    }

    void publish_imu_data() {
        auto msg = sensor_msgs::msg::Imu();
        msg.header.stamp = this->get_clock()->now();
        msg.header.frame_id = "imu_frame";

        // Simulate IMU data with some motion
        std::array<float, 3> true_accel = {0.0f, 0.0f, 9.81f}; // Gravity
        std::array<float, 3> true_gyro = {0.01f, 0.0f, 0.0f}; // Small rotation
        std::array<float, 3> true_mag = {0.2f, 0.0f, 0.4f}; // Magnetic field

        auto simulated_reading = imu_model_.simulateReading(true_accel, true_gyro, true_mag);

        // Fill ROS message
        msg.linear_acceleration.x = simulated_reading.linear_acceleration[0];
        msg.linear_acceleration.y = simulated_reading.linear_acceleration[1];
        msg.linear_acceleration.z = simulated_reading.linear_acceleration[2];

        msg.angular_velocity.x = simulated_reading.angular_velocity[0];
        msg.angular_velocity.y = simulated_reading.angular_velocity[1];
        msg.angular_velocity.z = simulated_reading.angular_velocity[2];

        // For quaternion, we'll create a simple orientation
        tf2::Quaternion q;
        q.setRPY(0.01, 0.0, 0.0); // Small roll due to acceleration
        msg.orientation.x = q.x();
        msg.orientation.y = q.y();
        msg.orientation.z = q.z();
        msg.orientation.w = q.w();

        // Set covariance (information matrix) - indicate uncertainty
        for (int i = 0; i < 9; ++i) {
            msg.linear_acceleration_covariance[i] = 0.0;
            msg.angular_velocity_covariance[i] = 0.0;
            msg.orientation_covariance[i] = 0.0;
        }
        // Set diagonal elements to indicate sensor noise
        msg.linear_acceleration_covariance[0] = 0.01; // 100ug^2
        msg.linear_acceleration_covariance[4] = 0.01;
        msg.linear_acceleration_covariance[8] = 0.01;
        msg.angular_velocity_covariance[0] = 0.001; // (0.1 deg/s)^2
        msg.angular_velocity_covariance[4] = 0.001;
        msg.angular_velocity_covariance[8] = 0.001;

        imu_pub_->publish(msg);
    }

    void publish_camera_data() {
        // Create a synthetic image for simulation
        cv::Mat synthetic_image = cv::Mat(480, 640, CV_8UC3);

        // Create a simple test pattern
        for (int i = 0; i < synthetic_image.rows; ++i) {
            for (int j = 0; j < synthetic_image.cols; ++j) {
                synthetic_image.at<cv::Vec3b>(i, j) = cv::Vec3b(
                    static_cast<uint8_t>((i * 255) / synthetic_image.rows),
                    static_cast<uint8_t>((j * 255) / synthetic_image.cols),
                    128
                );
            }
        }

        // Simulate with our camera model
        cv::Mat simulated_image = camera_model_.simulateImage(synthetic_image);

        // Convert to ROS message
        auto img_msg = cv_bridge::CvImage(
            std_msgs::msg::Header(), "bgr8", simulated_image
        ).toImageMsg();

        img_msg->header.stamp = this->get_clock()->now();
        img_msg->header.frame_id = "camera_frame";

        camera_pub_->publish(*img_msg);

        // Publish camera info
        auto info_msg = sensor_msgs::msg::CameraInfo();
        info_msg.header = img_msg->header;
        info_msg.width = 640;
        info_msg.height = 480;
        info_msg.distortion_model = "plumb_bob";
        info_msg.d = {0.0, 0.0, 0.0, 0.0, 0.0}; // No distortion
        info_msg.k = {320.0, 0.0, 320.0, 0.0, 320.0, 240.0, 0.0, 0.0, 1.0};
        info_msg.r = {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
        info_msg.p = {320.0, 0.0, 320.0, 0.0, 0.0, 320.0, 240.0, 0.0, 0.0, 0.0, 1.0, 0.0};

        camera_info_pub_->publish(info_msg);
    }
};

// Example main function to run the ROS 2 sensor simulator
int main(int argc, char * argv[]) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<ROSSensorSimulator>());
    rclcpp::shutdown();
    return 0;
}
```

## Isaac Sim Integration for Sensor Simulation

```mermaid
graph TB
    A[Real World Sensors] --> B{Isaac Sim Sensor Simulation}
    B --> C[LiDAR Sensor]
    B --> D[IMU Sensor]
    B --> E[Camera Sensor]
    B --> F[GPS Sensor]

    C --> G[RosBridge - LiDAR]
    D --> H[RosBridge - IMU]
    E --> I[RosBridge - Camera]
    F --> J[RosBridge - GPS]

    G --> K[ROS 2 Network]
    H --> K
    I --> K
    J --> K

    K --> L[Robot Control System]
    L --> M[Perception Algorithms]
    L --> N[Localization System]
    L --> O[Planning System]

    style A fill:#e1f5fe
    style B fill:#f3e5f5
    style L fill:#e8f5e8
    style M fill:#fff3e0
</graph>

## Sensor Calibration in Simulation

Calibration is critical for accurate sensor simulation. The process involves determining both intrinsic and extrinsic parameters:

### Intrinsic Calibration

```python
import numpy as np
import cv2
from scipy.optimize import least_squares
from typing import List, Tuple

class SensorCalibrator:
    """Class for sensor calibration in simulation environments."""

    def __init__(self):
        self.calibration_data = []

    def calibrate_camera_intrinsic(self, images: List[np.ndarray],
                                 pattern_size: Tuple[int, int] = (9, 6)) -> Tuple[np.ndarray, np.ndarray]:
        """Calibrate camera intrinsic parameters using a chessboard pattern."""
        # Prepare object points (3D points in real world space)
        objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)

        # Arrays to store object points and image points from all images
        objpoints = []  # 3d points in real world space
        imgpoints = []  # 2d points in image plane

        for img in images:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Find the chess board corners
            ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)

            # If found, add object points, image points (after refining them)
            if ret:
                objpoints.append(objp)

                # Refine corner locations
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                imgpoints.append(corners2)

        # Perform camera calibration
        if len(objpoints) > 0 and len(imgpoints) > 0:
            ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
                objpoints, imgpoints, gray.shape[::-1], None, None
            )

            return mtx, dist  # Camera matrix and distortion coefficients
        else:
            raise ValueError("Could not find chessboard corners in any images")

    def calibrate_lidar_camera_extrinsics(self, lidar_points: np.ndarray,
                                        camera_points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Calibrate LiDAR to camera extrinsic parameters."""
        # This is a simplified version - in practice, you'd need corresponding points
        # from both sensors to perform the calibration

        # For simulation purposes, we can synthesize the transformation
        # using known sensor placements in the robot model

        # Assuming we have corresponding points from both sensors
        # Use RANSAC or other robust estimation methods
        if len(lidar_points) != len(camera_points):
            raise ValueError("Mismatched number of points between sensors")

        # Compute transformation using Open3D or similar
        # For this example, we'll return a plausible transformation matrix

        # Create a plausible transformation matrix
        transform = np.eye(4, dtype=np.float32)
        # Example: LiDAR is 0.2m forward, 0.1m up, and 0.05m to the right of camera
        transform[0, 3] = 0.2  # x translation
        transform[1, 3] = 0.05 # y translation
        transform[2, 3] = 0.1  # z translation (height)

        # Example: Small rotation between sensors
        rx, ry, rz = 0.01, 0.02, 0.005  # Small rotations in radians
        transform[0:3, 0:3] = self.euler_to_rotation_matrix(rx, ry, rz)

        return transform[0:3, 0:3], transform[0:3, 3]  # Rotation matrix and translation vector

    def calibrate_imu_to_body(self, measured_data: List[dict]) -> np.ndarray:
        """Calibrate IMU to body frame transformation."""
        # This would use IMU measurements in known static positions
        # to determine the IMU's bias and mounting orientation

        # Accumulate measurements over time
        accel_measurements = []
        gyro_measurements = []
        expected_gravity = np.array([0, 0, 9.81])  # Expected gravity in body frame

        for measurement in measured_data:
            accel = np.array(measurement['accel'])
            gyro = np.array(measurement['gyro'])
            accel_measurements.append(accel)
            gyro_measurements.append(gyro)

        # Calculate average bias
        avg_accel = np.mean(accel_measurements, axis=0)
        avg_gyro = np.mean(gyro_measurements, axis=0)

        # Determine transformation from measured gravity direction
        # This is a simplified approach - full calibration is more complex
        gravity_direction = avg_accel - np.array([0, 0, 9.81])  # Remove expected gravity

        # Create calibration result structure
        calibration_result = {
            'accel_bias': avg_accel,
            'gyro_bias': avg_gyro,
            'gravity_alignment': gravity_direction
        }

        return calibration_result

    def euler_to_rotation_matrix(self, rx: float, ry: float, rz: float) -> np.ndarray:
        """Convert Euler angles to rotation matrix."""
        # Rotation around x axis
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(rx), -np.sin(rx)],
            [0, np.sin(rx), np.cos(rx)]
        ])

        # Rotation around y axis
        Ry = np.array([
            [np.cos(ry), 0, np.sin(ry)],
            [0, 1, 0],
            [-np.sin(ry), 0, np.cos(ry)]
        ])

        # Rotation around z axis
        Rz = np.array([
            [np.cos(rz), -np.sin(rz), 0],
            [np.sin(rz), np.cos(rz), 0],
            [0, 0, 1]
        ])

        # Combined rotation
        R = Rz @ Ry @ Rx
        return R

# Calibration pipeline example
def calibration_pipeline_example():
    """Example of a complete sensor calibration pipeline."""
    calibrator = SensorCalibrator()

    # Step 1: Camera intrinsic calibration
    print("Performing camera intrinsic calibration...")

    # In a real scenario, you would capture images of a calibration pattern
    # For this example, we'll create synthetic calibration images
    calib_images = []
    for i in range(10):
        # Create synthetic calibration image
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        # Add a synthetic chessboard pattern with distortions
        for row in range(6):
            for col in range(9):
                x = col * 60 + 50
                y = row * 60 + 50
                if (row + col) % 2 == 0:
                    cv2.rectangle(img, (x, y), (x+50, y+50), (255, 255, 255), -1)
                else:
                    cv2.rectangle(img, (x, y), (x+50, y+50), (0, 0, 0), -1)

        # Add some synthetic distortion
        k1, k2, p1, p2 = -0.1, 0.05, 0.001, -0.001
        # Apply distortion would happen here in a real implementation
        calib_images.append(img)

    try:
        camera_matrix, distortion_coeffs = calibrator.calibrate_camera_intrinsic(calib_images)
        print(f"Camera calibration completed!")
        print(f"Camera matrix:\n{camera_matrix}")
        print(f"Distortion coefficients: {distortion_coeffs}")
    except ValueError as e:
        print(f"Camera calibration failed: {e}")

    # Step 2: LiDAR-Camera extrinsic calibration
    print("\nPerforming LiDAR-camera extrinsic calibration...")

    # Create synthetic corresponding points
    lidar_pts = np.random.rand(100, 3).astype(np.float32) * 10  # Points within 10m
    camera_pts = lidar_pts.copy()  # Simplified - in reality these would be different

    try:
        rotation, translation = calibrator.calibrate_lidar_camera_extrinsics(lidar_pts, camera_pts)
        print(f"LiDAR-Camera extrinsic calibration completed!")
        print(f"Rotation matrix:\n{rotation}")
        print(f"Translation vector: {translation}")
    except ValueError as e:
        print(f"LiDAR-camera calibration failed: {e}")

    # Step 3: IMU calibration
    print("\nPerforming IMU calibration...")

    # Create synthetic IMU data (stationary)
    imu_data = []
    for i in range(100):
        # Simulate IMU readings with some bias
        accel = np.array([0.1, -0.05, 9.85]) + np.random.normal(0, 0.01, 3)  # Slightly offset from true gravity
        gyro = np.array([0.001, -0.002, 0.003]) + np.random.normal(0, 0.001, 3)  # Small drift

        imu_data.append({
            'accel': accel,
            'gyro': gyro,
            'timestamp': i * 0.01  # 100Hz data
        })

    imu_calibration = calibrator.calibrate_imu_to_body(imu_data)
    print(f"IMU calibration completed!")
    print(f"Accel bias: {imu_calibration['accel_bias']}")
    print(f"Gyro bias: {imu_calibration['gyro_bias']}")

if __name__ == "__main__":
    calibration_pipeline_example()
```

### Isaac Sim Configuration for Sensor Simulation

```json
{
  "isaac_sim_sensor_config": {
    "lidar": {
      "enabled": true,
      "lidar_params": {
        "rotation_frequency": 10,
        "channels": 64,
        "points_per_channel": 1080,
        "upper_fov": 2.0,
        "lower_fov": -24.8,
        "range": 100.0,
        "return_empty_spaces": false,
        "return_strongest_echo": true,
        "return_all_echoes": false,
        "sensor_x": 0.3,
        "sensor_y": 0.0,
        "sensor_z": 1.0,
        "sensor_rotx": 0.0,
        "sensor_roty": 0.0,
        "sensor_rotz": 0.0,
        "enable_composite_sweep": true,
        "noise_mean": 0.0,
        "noise_std": 0.01,
        "motion_blur_enabled": true,
        "motion_blur_samples": 4
      }
    },
    "camera": {
      "enabled": true,
      "camera_params": {
        "resolution": [640, 480],
        "focal_length": [320.0, 320.0],
        "principal_point": [320.0, 240.0],
        "distortion_model": "plumb_bob",
        "distortion_coefficients": [0.0, 0.0, 0.0, 0.0, 0.0],
        "sensor_x": 0.2,
        "sensor_y": 0.0,
        "sensor_z": 0.8,
        "sensor_rotx": 0.0,
        "sensor_roty": 0.0,
        "sensor_rotz": 0.0,
        "fps": 30,
        "exposure": 0.033,
        "iso": 100,
        "gain": 1.0,
        "noise_params": {
          "gaussian_noise_std": 10.0,
          "poisson_noise_factor": 0.01,
          "dark_current": 0.1,
          "read_noise": 2.0
        }
      }
    },
    "imu": {
      "enabled": true,
      "imu_params": {
        "sensor_x": 0.0,
        "sensor_y": 0.0,
        "sensor_z": 0.7,
        "sensor_rotx": 0.0,
        "sensor_roty": 0.0,
        "sensor_rotz": 0.0,
        "update_frequency": 100,
        "linear_acceleration_noise": 0.002,
        "angular_velocity_noise": 0.00024,
        "linear_acceleration_bias": [0.0, 0.0, 0.0],
        "angular_velocity_bias": [0.0, 0.0, 0.0],
        "linear_acceleration_random_walk": 0.0004,
        "angular_velocity_random_walk": 2.6e-06
      }
    },
    "ros_bridge": {
      "enabled": true,
      "topics": {
        "lidar_topic": "/scan",
        "camera_topic": "/camera/image_raw",
        "camera_info_topic": "/camera/camera_info",
        "imu_topic": "/imu/data",
        "tf_topic": "/tf"
      },
      "frame_ids": {
        "lidar_frame": "laser_frame",
        "camera_frame": "camera_frame",
        "imu_frame": "imu_frame",
        "base_frame": "base_link"
      }
    },
    "sensor_fusion": {
      "enabled": true,
      "fusion_params": {
        "time_sync_tolerance": 0.01,
        "calibration_file": "/path/to/calibration.json",
        "extrinsic_transforms": {
          "lidar_to_camera": [0.2, 0.05, 0.1, 0.01, 0.02, 0.005],
          "imu_to_body": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        }
      }
    }
  }
}
```

## Performance Optimization for Real-time Sensor Simulation

For real-time digital twin applications, sensor simulation must be optimized for performance:

```cpp
// Optimized sensor simulation using multi-threading
#include <thread>
#include <mutex>
#include <queue>
#include <atomic>
#include <memory>
#include <functional>

class OptimizedSensorSimulator {
private:
    std::atomic<bool> running_{false};
    std::vector<std::thread> worker_threads_;
    std::queue<std::function<void()>> task_queue_;
    std::mutex queue_mutex_;
    std::condition_variable condition_;
    int num_threads_;

    // Sensor simulation models
    std::unique_ptr<LiDARSensorModel> lidar_model_;
    std::unique_ptr<IMUSensorModel> imu_model_;
    std::unique_ptr<CameraSensorModel> camera_model_;

public:
    OptimizedSensorSimulator(int num_threads = 4) : num_threads_(num_threads) {
        // Initialize sensor models
        LiDARSensorModel::SensorConfig lidar_config;
        lidar_config.range_min = 0.1f;
        lidar_config.range_max = 100.0f;
        lidar_config.beams_horizontal = 1080;
        lidar_config.beams_vertical = 64;
        lidar_model_ = std::make_unique<LiDARSensorModel>(lidar_config);

        IMUSensorModel::SensorSpecs imu_specs;
        imu_model_ = std::make_unique<IMUSensorModel>(imu_specs, 100.0f);

        CameraSensorModel::CameraConfig camera_config;
        camera_config.width = 640;
        camera_config.height = 480;
        camera_model_ = std::make_unique<CameraSensorModel>(camera_config);

        start();
    }

    ~OptimizedSensorSimulator() {
        stop();
    }

    void start() {
        running_ = true;

        // Launch worker threads
        for (int i = 0; i < num_threads_; ++i) {
            worker_threads_.emplace_back([this] {
                worker_thread();
            });
        }
    }

    void stop() {
        running_ = false;
        condition_.notify_all();

        for (auto& thread : worker_threads_) {
            if (thread.joinable()) {
                thread.join();
            }
        }
    }

    void schedule_sensor_task(std::function<void()> task) {
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            task_queue_.push(task);
        }
        condition_.notify_one();
    }

    void simulate_lidar_async(const std::vector<std::vector<float>>& env_data,
                            std::function<void(const std::vector<std::vector<LiDARSensorModel::RayData>>&)> callback) {
        auto task = [this, env_data, callback]() {
            std::vector<std::vector<LiDARSensorModel::RayData>> scan_data;
            lidar_model_->simulateScan(env_data, std::vector<std::vector<float>>(), scan_data);
            callback(scan_data);
        };

        schedule_sensor_task(task);
    }

    void simulate_imu_async(const std::array<float, 3>& true_accel,
                          const std::array<float, 3>& true_gyro,
                          const std::array<float, 3>& true_mag,
                          std::function<void(const IMUSensorModel::IMUReading&)> callback) {
        auto task = [this, true_accel, true_gyro, true_mag, callback]() {
            auto reading = imu_model_->simulateReading(true_accel, true_gyro, true_mag);
            callback(reading);
        };

        schedule_sensor_task(task);
    }

private:
    void worker_thread() {
        while (running_) {
            std::function<void()> task;

            {
                std::unique_lock<std::mutex> lock(queue_mutex_);
                condition_.wait(lock, [this] { return !task_queue_.empty() || !running_; });

                if (!running_ && task_queue_.empty()) {
                    return;
                }

                if (!task_queue_.empty()) {
                    task = task_queue_.front();
                    task_queue_.pop();
                }
            }

            if (task) {
                task();
            }
        }
    }
};
```

## Conclusion

Sensor simulation in digital twin environments for humanoid robots requires sophisticated modeling that accounts for the full range of sensor characteristics, environmental interactions, and system integration requirements. The implementation of realistic sensor models with proper noise, bias, and calibration considerations enables effective development and testing of perception and control algorithms before deployment on physical systems.

The integration with ROS 2 and Isaac Sim provides the necessary infrastructure for real-time simulation that can be used in conjunction with robot control systems, making the transition from simulation to reality more reliable and predictable.