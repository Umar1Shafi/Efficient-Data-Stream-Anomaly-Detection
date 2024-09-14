import numpy as np  # numpy library for numerical operations
import random # random library for generating random numbers and choices
import matplotlib.pyplot as plt # pyplot module from matplotlib for creating visualizations
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score # metrics functions from scikit-learn for evaluating model performance
import mplcursors  # mplcursors for interactive annotations in matplotlib plots



# Step 1: Data Stream Simulation
def generate_data_stream(num_points=1000, noise_level=0.1, anomaly_chance=0.01, trend_slope=0.001, abrupt_anomalies=True):
    """
    Generate a simulated data stream with optional anomalies.

    Parameters:
    - num_points: Number of data points to generate.
    - noise_level: Maximum noise to add to each data point.
    - anomaly_chance: Probability of a data point being an anomaly.
    - trend_slope: Slope of the linear trend in the data.
    - abrupt_anomalies: Whether anomalies are abrupt or gradual.

    Yields:
    - Tuple of (data_point, true_label): Data point and its true label (0 for normal, 1 for anomaly).
    """
    # Validate parameters
    if not isinstance(num_points, int) or num_points <= 0:
        raise ValueError("num_points should be a positive integer.")
    if not (0 <= noise_level <= 1):
        raise ValueError("noise_level should be between 0 and 1.")
    if not (0 <= anomaly_chance <= 1):
        raise ValueError("anomaly_chance should be between 0 and 1.")
    if not isinstance(trend_slope, (int, float)):
        raise ValueError("trend_slope should be a number.")
    if not isinstance(abrupt_anomalies, bool):
        raise ValueError("abrupt_anomalies should be a boolean.")

    stream = []  # List to store the generated data points
    true_labels = []  # List to store the true labels (0 for normal, 1 for anomaly)
    seasonal_period = 50  # Number of points per season (sinusoidal period)
    gradual_anomalies = False  # Flag for gradual anomalies

    for i in range(num_points):
        # Create sinusoidal seasonality component to simulate seasonal variations
        seasonality = np.sin(2 * np.pi * i / seasonal_period)
        # Add random noise to the data point
        noise = random.uniform(-noise_level, noise_level)
        # Compute the trend component based on a linear slope
        trend = trend_slope * i
        # Calculate the value of the data point
        value = seasonality + trend + noise
        is_anomaly = False
        if random.random() < anomaly_chance:
            is_anomaly = True
            if abrupt_anomalies:
                # Add abrupt anomaly effect
                value += random.uniform(5, 10) if random.random() > 0.5 else random.uniform(-10, -5)
            else:
                gradual_anomalies = True
        
        if gradual_anomalies:
            # Add gradual anomaly drift effect
            value += trend_slope * 10
            if random.random() < 0.1:
                gradual_anomalies = False
        
        true_labels.append(1 if is_anomaly else 0)
        stream.append(value)
        yield value, true_labels[-1]



# Step 2: Enhanced Anomaly Detection
from collections import deque

class AnomalyDetector:
    """
    Anomaly Detector using Adaptive Z-Score method with moving average and incremental mean and variance updates.

    Attributes:
    - alpha: Smoothing factor for updating mean and standard deviation.
    - initial_z_threshold: Initial Z-score threshold during the warm-up period.
    - final_z_threshold: Final Z-score threshold after warm-up.
    - warmup_period: Number of points to adjust the threshold during warm-up.
    - moving_avg_window: Window size for calculating moving average of the data.
    """
    def __init__(self, alpha=0.05, initial_z_threshold=4, final_z_threshold=2, warmup_period=50, moving_avg_window=30):
        """
        Initialize the AnomalyDetector with specific parameters.

        Parameters:
        - alpha: Smoothing factor for mean and standard deviation updates.
        - initial_z_threshold: Initial threshold for Z-score during the warm-up phase.
        - final_z_threshold: Final threshold for Z-score after the warm-up phase.
        - warmup_period: Number of data points to use for adjusting the threshold.
        - moving_avg_window: Size of the window for calculating the moving average.
        """
        # Validate parameters
        if not (0 < alpha <= 1):
            raise ValueError("alpha should be between 0 and 1.")
        if not (0 < initial_z_threshold <= 10):
            raise ValueError("initial_z_threshold should be between 0 and 10.")
        if not (0 < final_z_threshold <= 10):
            raise ValueError("final_z_threshold should be between 0 and 10.")
        if not isinstance(warmup_period, int) or warmup_period <= 0:
            raise ValueError("warmup_period should be a positive integer.")
        if not isinstance(moving_avg_window, int) or moving_avg_window <= 0:
            raise ValueError("moving_avg_window should be a positive integer.")
        
        self.alpha = alpha
        self.initial_z_threshold = initial_z_threshold
        self.final_z_threshold = final_z_threshold
        self.z_threshold = initial_z_threshold  # Start with initial threshold
        self.mean = None  # Mean of the data points
        self.var = None  # Variance
        self.std = None  # Standard deviation of the data points
        self.counter = 0  # Counter for the number of data points processed
        self.warmup_period = warmup_period  # Warm-up period for threshold adjustment
        self.moving_avg_window = moving_avg_window  # Size of the moving average window
        
        # Use deque for efficient windowed calculations
        self.moving_avg = deque(maxlen=self.moving_avg_window)  # Initialize a deque with a fixed size

    def update(self, x):
        """
        Update the anomaly detector with a new data point.

        Parameters:
        - x: New data point to be analyzed.

        Returns:
        - bool: True if the data point is detected as an anomaly, False otherwise.
        """
        # Validate data point
        if not isinstance(x, (int, float)):
            raise ValueError("Data point x should be a number.")

        self.counter += 1  # Increment the counter for the number of data points

        if self.mean is None:
            # Initialize mean and standard deviation on the first data point
            self.mean = x
            self.std = 0
        else:
            # Update mean and standard deviation using exponential moving average
            self.mean = self.alpha * x + (1 - self.alpha) * self.mean
            self.std = self.alpha * abs(x - self.mean) + (1 - self.alpha) * self.std

        # Update the moving average deque with the new data point
        self.moving_avg.append(x)

        # Calculate smoothed mean and standard deviation using the moving average window
        smoothed_mean = np.mean(self.moving_avg)
        smoothed_std = np.std(self.moving_avg)

        # Adjust the Z-score threshold gradually during the warm-up period
        if self.counter <= self.warmup_period:
            self.z_threshold = (self.initial_z_threshold - self.final_z_threshold) * (1 - self.counter / self.warmup_period) + self.final_z_threshold
        else:
            self.z_threshold = self.final_z_threshold

        # Calculate the Z-score of the new data point
        z_score = (x - smoothed_mean) / (smoothed_std if smoothed_std != 0 else 1)
        # Determine if the data point is an anomaly based on the Z-score
        return abs(z_score) > self.z_threshold




# Step 3: Rolling Mean and Standard Deviation Plot
def plot_rolling_statistics(data, window_size=50):
    """
    Plot rolling mean and standard deviation of the data.

    Parameters:
    - data: List of data points.
    - window_size: Window size for calculating rolling statistics.
    """
    # Validate parameters
    if not isinstance(window_size, int) or window_size <= 0:
        raise ValueError("window_size should be a positive integer.")
    if len(data) < window_size:
        raise ValueError("Data length should be greater than or equal to window_size.")

    # Calculate rolling mean using convolution to smooth out the data
    rolling_mean = np.convolve(data, np.ones(window_size)/window_size, mode='valid')
    # Calculate rolling standard deviation over the specified window size
    rolling_std = [np.std(data[i:i+window_size]) for i in range(len(data) - window_size + 1)]

    # Create a plot to visualize rolling statistics
    plt.figure(figsize=(12, 6))
    plt.plot(data, label='Data Stream', alpha=0.5)  # Plot the raw data stream
    plt.plot(range(window_size-1, len(data)), rolling_mean, label='Rolling Mean', color='orange')  # Plot rolling mean
    plt.plot(range(window_size-1, len(data)), rolling_std, label='Rolling Std Dev', color='red')  # Plot rolling std deviation
    plt.legend()
    plt.title('Rolling Mean and Standard Deviation')
    plt.show()




# Step 4: Enhanced Real-Time Visualization with Missed Anomaly Logging
def plot_real_time_with_missed_anomalies(data_stream, detector, update_freq=10):
    """
    Plot the data stream in real-time with detected anomalies and log missed anomalies.

    Parameters:
    - data_stream: Generator function yielding data points and true labels.
    - detector: Anomaly detector instance.
    - update_freq: Frequency of plot updates.
    """
    plt.ion()  # Turn on interactive mode
    fig, ax = plt.subplots(figsize=(14, 7))  # Create a larger figure for better clarity
    data, anomalies, predictions, true_labels = [], [], [], []
    z_scores = []  # To store z-scores for enhanced visualization
    missed_anomalies = []  # List to track missed anomalies

    for i, (point, true_label) in enumerate(data_stream):
        try:
            data.append(point)  # Append new data point
            true_labels.append(true_label)  # Append true label to the true_labels list

            # Perform anomaly detection on the new data point
            is_anomaly = detector.update(point)
            predictions.append(int(is_anomaly))  # Append prediction

            # Calculate Z-score for visualization
            smoothed_mean = np.mean(data[-detector.moving_avg_window:])
            smoothed_std = np.std(data[-detector.moving_avg_window:])
            z_score = (point - smoothed_mean) / (smoothed_std if smoothed_std != 0 else 1)
            z_scores.append(z_score)

            if is_anomaly:
                anomalies.append((i, point, z_score))  # Record anomaly with Z-score

            # Track missed anomalies
            if true_label == 1 and is_anomaly == False:
                missed_anomalies.append({
                    "index": i,
                    "value": point,
                    "z_score": z_score,
                    "mean": np.mean(data[-detector.moving_avg_window:]),
                    "std_dev": np.std(data[-detector.moving_avg_window:]),
                    "threshold": detector.z_threshold
                })

            # Update plot every update_freq points
            if i % update_freq == 0:
                ax.clear()  # Clear previous plot
                ax.plot(data, label='Data Stream', alpha=0.6)  # Plot data stream

                # Plot true anomalies as green dots
                true_anomalies = [(index, value) for index, value in enumerate(data) if true_labels[index] == 1]
                if true_anomalies:
                    x_true, y_true = zip(*true_anomalies)
                    ax.scatter(x_true, y_true, color='green', label='True Anomalies', s=50, marker='x')

                # Plot detected anomalies with distinct colors based on severity
                severe_anomalies = [anomaly for anomaly in anomalies if abs(anomaly[2]) > 3]
                moderate_anomalies = [anomaly for anomaly in anomalies if abs(anomaly[2]) <= 3]

                if severe_anomalies:
                    x_anoms, y_anoms, _ = zip(*severe_anomalies)
                    ax.scatter(x_anoms, y_anoms, color='red', label='Severe Detected Anomalies', s=60, edgecolor='black')
                
                if moderate_anomalies:
                    x_anoms, y_anoms, _ = zip(*moderate_anomalies)
                    ax.scatter(x_anoms, y_anoms, color='orange', label='Moderate Detected Anomalies', s=60, edgecolor='black')

                # Dynamic Z-score threshold line
                threshold_line = [detector.z_threshold] * len(data)
                ax.plot(threshold_line, label=f'Threshold Z-Score: {detector.z_threshold:.2f}', linestyle='--', color='blue')

                ax.legend()
                ax.set_title(f"Real-Time Data Stream (Point {i})")
                plt.pause(0.001)  # Pause to update the plot

        except Exception as e:
            print(f"Error processing data point {i}: {e}")

    plt.ioff()  # Turn off interactive mode
    plt.show()

    # Calculate and print performance metrics
    accuracy = accuracy_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions)
    recall = recall_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions)
    anomaly_count = sum(predictions)

    print("Effectiveness of Algorithm:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Anomalies Detected: {anomaly_count}")
    
    # Log missed anomalies
    if missed_anomalies:
        print("\nMissed Anomalies:")
        for anomaly in missed_anomalies:
            print(f"Index: {anomaly['index']}, Value: {anomaly['value']:.4f}, Z-Score: {anomaly['z_score']:.4f}, "
                  f"Mean: {anomaly['mean']:.4f}, Std Dev: {anomaly['std_dev']:.4f}, Threshold: {anomaly['threshold']:.4f}")
    else:
        print("No missed anomalies.")

    # Plot rolling statistics
    plot_rolling_statistics(data)




# Step 5: Running the Enhanced Detection System with Missed Anomaly Logging
if __name__ == "__main__":
    try:
        # Initialize the anomaly detector with specific parameters
        detector = AnomalyDetector(alpha=0.1, initial_z_threshold=5, final_z_threshold=2.5, warmup_period=20, moving_avg_window=50)
        
        # Generate a data stream for testing
        data_stream = generate_data_stream(num_points=5000, noise_level=0.2, anomaly_chance=0.02, trend_slope=0.002, abrupt_anomalies=True)

        # Run the real-time anomaly detection and enhanced visualization
        plot_real_time_with_missed_anomalies(data_stream, detector, update_freq=40)
    
    except Exception as e:
        print(f"An error occurred: {e}")

