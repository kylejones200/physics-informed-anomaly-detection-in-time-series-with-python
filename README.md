# Repository

Companion code for a Medium article.

## Business context

The challenge addressed here is to detect anomalies in the components of a wind turbine. Each wind turbine is equipped with multiple sensors that record data such as:

These anomalies depend on multiple variables while the turbine operates. We use an unsupervised ML model called an Autoencoder to correlate features, learn the latent representation of the dataset, and predict the input tensor. By training the model on data from a normal turbine (without anomalies), the model learns 'normal behavior'. When input with data from a malfunctioning turbine, the model produces a high error, thus detecting an anomaly.

Since sensor readings are time-series data with high correlation between neighboring samples, we reformat the data as a multidimensional tensor. Specifically, we create a temporal encoding of eight features in $10 times 10$ steps to create a tensor with shape $8 times 10 times 10$.

## Disclaimer

Educational/demo code only. Not financial, safety, or engineering advice. Use at your own risk. Verify results independently before any production or operational use.

## License

MIT — see [LICENSE](LICENSE).