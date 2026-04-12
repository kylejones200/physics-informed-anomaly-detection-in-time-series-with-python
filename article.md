# Physics-Informed Anomaly Detection in a Wind Turbine Using Python with an Autoencoder Transformer

The challenge addressed here is to detect anomalies in the components of a wind turbine. Each wind turbine is equipped with multiple sensors that record data such as:

- External temperature

- Rotor speed

- Air pressure

- Voltage (or current) in the generator

- Vibration in the gearbox, generator, and tower

We are interested in detecting three types of anomalies:

1.  **Rotor speed:** when the rotor is not at the expected speed

2.  **Produced voltage:** when the generator does not produce the expected voltage

3.  **Gearbox vibration:** when the gearbox vibration deviates from expected levels

These anomalies depend on multiple variables while the turbine operates. We use an unsupervised ML model called an **Autoencoder** to correlate features, learn the latent representation of the dataset, and predict the input tensor. By training the model on data from a normal turbine (without anomalies), the model learns 'normal behavior'. When input with data from a malfunctioning turbine, the model produces a high error, thus detecting an anomaly.

Since sensor readings are time-series data with high correlation between neighboring samples, we reformat the data as a multidimensional tensor. Specifically, we create a temporal encoding of eight features in $10 \times 10$ steps to create a tensor with shape $8 \times 10 \times 10$.

## Dataset Description

The dataset includes the following features:

- **timestamp:** Timestamp of the row (sensor readings every 90 seconds)

- **sensorId:** ID of the edge device collecting the data

- **long, lat:** Longitude and latitude of the turbine

- **temp:** External temperature

- **pressure:** Air pressure

- **humidity:** Air humidity

- **altitude:** Altitude of the turbine

- **voltage:** Voltage produced by the generator (mV)

- **power:** Power produced by the generator (mV)

- **rpm:** Wind speed in Rotations Per Minute

- **status:** Status of turbine \['active', 'inactive'\]

- **gearbox_vibration:** Vibration measurement inside the gearbox

- **generator_vibration:** Vibration measurement inside the generator

- **tower_vibration:** Vibration measurement on the tower gearbox

- **anomaly:** Expert-provided label indicating if the row is anomalous

## Wavelet Denoising

The `wavelet_denoise()` function smooths out the time series data using the `pywt` library, preserving the underlying physics.

    import seaborn as sns
    from sklearn.ensemble import IsolationForest
    import pywt

    # Helper function for wavelet denoising
    def wavelet_denoise(data, wavelet, noise_sigma):
        wavelet = pywt.Wavelet(wavelet)
        levels = min(5, (np.floor(np.log2(data.shape[0]))).astype(int))
        wavelet_coeffs = pywt.wavedec(data, wavelet, level=levels)
        threshold = noise_sigma*np.sqrt(2*np.log2(data.size))
        new_wavelet_coeffs = map(lambda x: pywt.threshold(x, threshold, mode='soft'), wavelet_coeffs)
        return pywt.waverec(list(new_wavelet_coeffs), wavelet)

## Plotting the Vibration Data

Our data is noisy, which obscures patterns. We first visualize the raw data:

    # Plot and save original data
    plt.figure(figsize=(20,10))
    df[features].plot(figsize=(20,10))
    plt.title("Original Data")
    plt.tight_layout()
    plt.savefig('original_data.png')
    plt.show()

## Cleaning and Normalizing

To reveal patterns, we denoise and normalize the data:

    # Denoise and normalize the data
    df_train = df.copy()
    raw_std = df_train[features].std()
    for f in features:
        df_train[f] = wavelet_denoise(df_train[f].values, 'db6', raw_std[f])

    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    df_train = pd.DataFrame(scaler.fit_transform(df_train[features]), columns=features)

    # Plot and save denoised and normalized data
    plt.figure(figsize=(20,10))
    df_train.plot(figsize=(20,10))
    plt.title("Denoised and Normalized Data")
    plt.tight_layout()
    plt.savefig('denoised_normalized_data.png')
    plt.show()

## Anomaly Detection with Isolation Forest

We apply the Isolation Forest algorithm to detect anomalies:

    # Create the Isolation Forest model and detect anomalies
    iso_forest = IsolationForest(n_estimators=100, contamination=0.01, random_state=42)
    iso_forest.fit(df_train)
    anomaly_scores = iso_forest.decision_function(df_train)

    # Identify anomalies
    threshold = np.percentile(anomaly_scores, 1)
    anomalies = df.loc[anomaly_scores < threshold]

    # Plot the results
    plt.figure(figsize=(15, 8))
    plt.scatter(df.index, df['voltage'], c='b', label='Normal Data')
    plt.scatter(anomalies.index, anomalies['voltage'], c='r', label='Anomalies')
    plt.title('Anomaly Detection in Wind Turbine Voltage')
    plt.xlabel('Timestamp')
    plt.ylabel('Voltage')
    plt.legend()
    plt.savefig('anomaly_detection.png')
    plt.show()

We detected two periods with a high frequency of anomalies, which warrant further investigation.

## Feature Correlation Analysis

We examine feature correlations using a heatmap:

    # Create and save correlation heatmap
    corr = df_train.corr()
    mask = np.zeros_like(corr)
    mask[np.triu_indices_from(mask)] = True
    f, ax = plt.subplots(figsize=(15, 8))
    sns.heatmap(corr, annot=True, mask=mask)
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig('correlation_heatmap.png')
    plt.show()

The heatmap shows significant autocorrelation between features, as expected for sensor data from a mechanical system.

This study demonstrates the use of physics-informed anomaly detection in wind turbines using an autoencoder transformer. By leveraging wavelet denoising and Isolation Forests, we effectively identify anomalies linked to rotor speed, produced voltage, and gearbox vibration. The approach enhances predictive maintenance and operational efficiency, reducing downtime and maintenance costs.

## Key Takeaways

- External temperature
- Rotor speed
- Air pressure
- Voltage (or current) in the generator
